use std::{
    collections::HashSet,
    f32::consts::FRAC_PI_4,
    sync::{Arc, Mutex as StdMutex},
    time::Duration,
};

use anyhow::{Context, anyhow};
use async_trait::async_trait;
use nalgebra::{Vector2, Vector3};

use jge_core::{
    Game,
    config::GameConfig,
    event::{
        DeviceEvent, ElementState, Event, Key, KeyCode, KeyEvent, NamedKey, PhysicalKey,
        WindowEvent, split_event_mapper,
    },
    game::{
        component::{
            background::Background,
            camera::Camera,
            layer::{Layer, LayerShader, LayerViewport, ShaderLanguage},
            light::{Light, ParallelLight, PointLight},
            material::Material,
            node::Node,
            renderable::Renderable,
            scene2d::Scene2D,
            scene3d::Scene3D,
            shape::Shape,
            transform::Transform,
        },
        entity::Entity,
        system::logic::GameLogic,
    },
    logger, scene,
    text::{Font, TextRenderOptions, TextShadow, render_text_to_material_texture},
};
use tracing::{error, info, warn};
use winit::dpi::PhysicalPosition;
use winit::event::MouseScrollDelta;
use winit::window::CursorGrabMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MouseInputMode {
    LockedRelative,
    Recenter,
}

#[derive(Debug)]
struct MouseState {
    window: Option<Arc<winit::window::Window>>,
    mode: MouseInputMode,
    ignore_next_cursor_moved: bool,
}

fn main() -> anyhow::Result<()> {
    logger::init()?;

    jge_core::resource!("resources.yaml")?;

    let root = Entity::new()?;

    let mouse_state = Arc::new(StdMutex::new(MouseState {
        window: None,
        mode: MouseInputMode::LockedRelative,
        ignore_next_cursor_moved: false,
    }));

    let mouse_state_for_init = Arc::clone(&mouse_state);
    let mouse_state_for_device = Arc::clone(&mouse_state);
    let mouse_state_for_window = Arc::clone(&mouse_state);

    let game = Game::new(GameConfig::default(), root)?
        .with_window_init(move |game| {
            let Some(window) = game.winit_window_arc() else {
                return;
            };

            let mut state = mouse_state_for_init
                .lock()
                .expect("mouse state mutex poisoned");
            state.window = Some(Arc::clone(&window));

            // Wayland 下 set_cursor_position 仅在 Locked 时可用；同时“允许移动+回中法”也经常受限。
            // 优先采用：抓取光标 + 隐藏光标 + 使用 DeviceEvent::MouseMotion 获取相对 delta。
            // 如果 Locked 失败，则退化为“CursorMoved + 每次移动后回中”方案，避免光标到边缘后没法继续转向。
            match window.set_cursor_grab(CursorGrabMode::Locked) {
                Ok(()) => {
                    state.mode = MouseInputMode::LockedRelative;
                }
                Err(err) => {
                    warn!(target = "jge-demo", error = %err, "cursor grab (locked) failed, fallback to recenter mode");
                    state.mode = MouseInputMode::Recenter;
                    // 先确保不处于抓取状态，保持“允许鼠标移动”的语义。
                    let _ = window.set_cursor_grab(CursorGrabMode::None);

                    // 尝试将鼠标放到窗口中心，后续每次移动事件也会继续回中。
                    let size = window.inner_size();
                    if size.width > 0 && size.height > 0 {
                        let center_x = (size.width as f64) * 0.5;
                        let center_y = (size.height as f64) * 0.5;
                        if window
                            .set_cursor_position(PhysicalPosition::new(center_x, center_y))
                            .is_ok()
                        {
                            state.ignore_next_cursor_moved = true;
                        }
                    }
                }
            }
            window.set_cursor_visible(false);
        })
        .with_event_mapper(split_event_mapper(
            move |event: &WindowEvent| match event {
                WindowEvent::MouseWheel { delta, .. } => {
                    let steps = match delta {
                        MouseScrollDelta::LineDelta(_, y) => *y,
                        MouseScrollDelta::PixelDelta(position) => (position.y as f32) / 120.0,
                    };
                    if steps.abs() <= f32::EPSILON {
                        return None;
                    }
                    Some(Event::custom(InputEvent::MouseWheel { steps }))
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let mut state = mouse_state_for_window
                        .lock()
                        .expect("mouse state mutex poisoned");
                    if state.mode != MouseInputMode::Recenter {
                        return None;
                    }

                    let Some(window) = state.window.clone() else {
                        return None;
                    };

                    let size = window.inner_size();
                    if size.width == 0 || size.height == 0 {
                        return None;
                    }

                    let center_x = (size.width as f64) * 0.5;
                    let center_y = (size.height as f64) * 0.5;

                    if state.ignore_next_cursor_moved {
                        // 我们主动 set_cursor_position 会触发一次 CursorMoved（到中心）。
                        // 这次事件不应产生视角旋转。
                        state.ignore_next_cursor_moved = false;
                        return None;
                    }

                    let dx = (position.x - center_x) as f32;
                    let dy = (position.y - center_y) as f32;

                    let _ = window
                        .set_cursor_grab(CursorGrabMode::Locked)
                        .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
                    if window
                        .set_cursor_position(PhysicalPosition::new(center_x, center_y))
                        .is_ok()
                    {
                        state.ignore_next_cursor_moved = true;
                    }
                    let _ = window.set_cursor_grab(CursorGrabMode::None);

                    Some(Event::custom(InputEvent::MouseDelta { dx, dy }))
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    let action = map_key_event_to_camera_action(event)?;
                    let pressed = matches!(event.state, ElementState::Pressed);
                    Some(Event::custom(InputEvent::Action { action, pressed }))
                }
                WindowEvent::Focused(false) => Some(Event::custom(InputEvent::Clear)),
                _ => None,
            },
            move |event: &DeviceEvent| {
                let mode = mouse_state_for_device
                    .lock()
                    .expect("mouse state mutex poisoned")
                    .mode;
                if mode != MouseInputMode::LockedRelative {
                    return None;
                }

                match event {
                    DeviceEvent::MouseMotion { delta } => Some(Event::custom(InputEvent::MouseDelta {
                        dx: delta.0 as f32,
                        dy: delta.1 as f32,
                    })),
                    _ => None,
                }
            },
        ));

    game.spawn(async move {
        tokio::time::sleep(Duration::from_secs(1)).await;
        let scene = build_demo_scene().await.context("构建测试场景失败")?;
        if root.get_component::<Node>().is_some() {
            let attach_future = {
                let mut node = root
                    .get_component_mut::<Node>()
                    .expect("root missing Node component");
                node.attach(scene)
            };

            if let Err(e) = attach_future.await {
                error!(target: "jge-demo", error = %e, "附加场景失败");
            }
        }
        anyhow::Result::<()>::Ok(())
    });

    game.run()
}

#[derive(Debug)]
enum InputEvent {
    Action { action: CameraAction, pressed: bool },
    MouseDelta { dx: f32, dy: f32 },
    MouseWheel { steps: f32 },
    Clear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CameraAction {
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
}

fn map_key_event_to_camera_action(event: &KeyEvent) -> Option<CameraAction> {
    match &event.logical_key {
        Key::Character(value) => match value.as_str() {
            "w" | "W" => Some(CameraAction::MoveForward),
            "s" | "S" => Some(CameraAction::MoveBackward),
            "a" | "A" => Some(CameraAction::MoveLeft),
            "d" | "D" => Some(CameraAction::MoveRight),
            _ => None,
        },
        Key::Named(NamedKey::Space) => Some(CameraAction::MoveUp),
        _ => match event.physical_key {
            PhysicalKey::Code(KeyCode::AltLeft) => Some(CameraAction::MoveDown),
            _ => None,
        },
    }
}

async fn build_demo_scene() -> anyhow::Result<Entity> {
    let text = render_text_to_material_texture(
        "Hello Jungle Engine\n你好， Jungle 引擎",
        &Font::System("Sarasa UI SC".to_string()),
        TextRenderOptions {
            font_size_px: 42.0,
            color: [255, 255, 255, 255],
            shadow: Some(TextShadow {
                offset_px: (2, 2),
                color: [0, 0, 0, 220],
            }),
            ..Default::default()
        },
    )
    .context("生成 HelloWorld 文本贴图失败")?;

    // Scene2D 使用 `MaterialPatch` 逐三角形提供 UV，这里用两三角覆盖整个纹理。
    let text_triangles = quad_triangles(text.width as f32, text.height as f32, 0.0);
    let text_patches = quad_uv_patches();

    let bindings = scene! {
        node {
            node "scene" as scene3d_layer {
                + Layer::new() => |_, mut layer| -> anyhow::Result<()> {
                    if std::env::var("JGE_DEMO_LAYER_VIEWPORT").ok().as_deref() == Some("1") {
                        layer.set_viewport(LayerViewport::normalized(0.05, 0.05, 0.45, 0.45));
                    }
                    Ok(())
                };

                + Scene3D::new();

                + Background::new() => resource(horizon_fs = "shaders/demo/background_horizon.fs") |_, mut bg| -> anyhow::Result<()> {
                    // 天蓝色（SkyBlue）
                    bg.set_color([0.53, 0.81, 0.92, 1.0]);
                    bg.set_fragment_shader(Some(LayerShader::new(ShaderLanguage::Wgsl, horizon_fs)));
                    Ok(())
                };

                node "camera" as camera {
                    + Camera::new();

                    * CameraControllerLogic::new();

                    with(mut transform: Transform) {
                        transform.set_position(Vector3::new(0.0, 6.0, 6.0));
                        // +Y 向上：默认 forward 为 -Z，因此 pitch 需要为负才能“向下看”到地面。
                        transform.set_rotation(Vector3::new(-FRAC_PI_4, 0.0, 0.0));
                        Ok(())
                    }
                }

                // 绑定摄像机并同步变换（依赖执行顺序与作用域规则：camera 在此已可见）。
                with(mut scene: Scene3D) {
                    scene
                        .bind_camera(camera)
                        .context("为 Scene3D 图层绑定摄像机失败")?;
                    Ok(())
                }

                with(scene: Scene3D) {
                    scene.sync_camera_transform().context("同步 Scene3D 摄像机变换失败")?;
                    Ok(())
                }

                node "ground" {
                    + Shape::from_triangles(ground_triangles());
                    with(mut transform: Transform) {
                        transform.set_position(Vector3::new(0.0, 0.0, 0.0));
                        transform.set_rotation(Vector3::new(0.0, 0.0, 0.0));
                        transform.set_scale(Vector3::new(6.0, 1.0, 6.0));
                        Ok(())
                    }
                }

                node {
                    + PointLight::new(15.0);
                    with(mut transform: Transform, mut light: Light) {
                        if let Some(mut renderable) = e.get_component_mut::<Renderable>() {
                            renderable.set_enabled(false);
                        }
                        transform.set_position(Vector3::new(6.0, 1.0, 6.0));
                        light.set_lightness(1.0);
                        Ok(())
                    }
                }

                node {
                    + ParallelLight::new();
                    with(mut transform: Transform, mut light: Light) {
                        if let Some(mut renderable) = e.get_component_mut::<Renderable>() {
                            renderable.set_enabled(false);
                        }
                        transform.set_rotation(Vector3::new(-0.3, -std::f32::consts::PI, 0.0));
                        light.set_lightness(0.7);
                        Ok(())
                    }
                }

                node "tri1" {
                    + Shape::from_triangles(triangle_triangles());
                    with(mut transform: Transform) {
                        transform.set_position(Vector3::new(0.0, 0.5, -4.0));
                        transform.set_rotation(Vector3::new(0.0, 0.3, 0.0));
                        transform.set_scale(Vector3::new(1.6, 1.6, 1.6));
                        Ok(())
                    }
                }

                node "tri2" {
                    + Shape::from_triangles(triangle_triangles());
                    with(mut transform: Transform) {
                        transform.set_position(Vector3::new(-2.5, 0.9, -6.0));
                        transform.set_rotation(Vector3::new(0.2, -0.3, 0.1));
                        transform.set_scale(Vector3::new(1.2, 1.2, 1.2));
                        Ok(())
                    }
                }

                node "cube" {
                    + Shape::from_triangles(cube_triangles());

                    + Material::new(bamboo, Vec::new()) => resource(bamboo = "textures/bamboo.png") |_, mut mat| -> anyhow::Result<()> {
                        let triangles = cube_triangles();
                        let patches = compute_cube_uv_patches(&triangles).context("计算立方体 UV patches 失败")?;
                        mat.set_regions(patches);
                        Ok(())
                    };

                    * CubeLogic;

                    with(mut transform: Transform) {
                        transform.set_position(Vector3::new(-3.0, 0.5, 0.0));
                        Ok(())
                    }
                }
            }

            // UI Layer：在屏幕左上角显示带阴影的文字。
            node "HUD" as ui_layer {
                + Layer::new();
                + Scene2D::new() => |e, mut scene| -> anyhow::Result<()> {
                    // 让 1 世界单位 = 1 像素，方便用像素尺寸摆放 UI。
                    scene.set_pixels_per_unit(1.0);

                    // Scene2D 的可见性查询依赖 Layer 的 LOD（chunk_positions）。
                    // UI Layer 做一次 warmup，避免出现 "no visible draws"。
                    let Some(mut layer) = e.get_component_mut::<Layer>() else {
                        anyhow::bail!("UI Layer 缺少 Layer 组件")
                    };
                    let _ = scene.warmup_lod(&mut layer);
                    Ok(())
                };

                // UI 平行光：用于让 Scene2D 渲染也有方向光照（不渲染任何实体几何）。
                node {
                    + ParallelLight::new();
                    with(mut transform: Transform, mut light: Light) {
                        if let Some(mut renderable) = e.get_component_mut::<Renderable>() {
                            renderable.set_enabled(false);
                        }
                        // 方向：绕 Y 轴转 180°，并略微向下俯（与 3D 示例一致的方向习惯）。
                        transform.set_rotation(Vector3::new(0., -std::f32::consts::PI, 0.0));
                        // UI 通常希望更“平”，把强度控制在较小范围。
                        light.set_lightness(1.);
                        Ok(())
                    }
                }

                * UiHelloLogic::new(ui_text);

                node "text" as ui_text {
                    + Shape::from_triangles(text_triangles);
                    + Material::new(text.resource.clone(), text_patches);

                    with(mut transform: Transform) {
                        // 初始值无所谓：真正的左上角定位会在 UiHelloLogic 中等 framebuffer 尺寸初始化后写入。
                        transform.set_position(Vector3::new(0.0, 0.0, 0.9));
                        Ok(())
                    }
                }
            }
        }
    }
    .await?;

    Ok(bindings.root)
}

fn quad_triangles(width: f32, height: f32, z: f32) -> Vec<[Vector3<f32>; 3]> {
    // 局部坐标：原点在左上角，y 向下为负（Scene2D 世界坐标 y 向上）。
    let p0 = Vector3::new(0.0, 0.0, z);
    let p1 = Vector3::new(width, 0.0, z);
    let p2 = Vector3::new(0.0, -height, z);
    let p3 = Vector3::new(width, -height, z);

    vec![[p0, p1, p2], [p1, p3, p2]]
}

fn quad_uv_patches() -> Vec<[Vector2<f32>; 3]> {
    vec![
        [
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(0.0, 1.0),
        ],
        [
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
            Vector2::new(0.0, 1.0),
        ],
    ]
}

struct UiHelloLogic {
    text_entity: Entity,
}

impl UiHelloLogic {
    fn new(text_entity: Entity) -> Self {
        Self { text_entity }
    }
}

#[async_trait]
impl GameLogic for UiHelloLogic {
    async fn update(&mut self, layer_entity: Entity, _delta: Duration) -> anyhow::Result<()> {
        // 在 framebuffer 尺寸由渲染阶段初始化后，把文字贴到视口左上角。
        let Some(scene) = layer_entity.get_component::<Scene2D>() else {
            return Ok(());
        };

        let Some(world_top_left) = scene.pixel_to_world(Vector2::new(16.0, 16.0)) else {
            return Ok(());
        };

        if let Some(mut transform) = self.text_entity.get_component_mut::<Transform>() {
            transform.set_position(Vector3::new(world_top_left.x, world_top_left.y, 0.9));
        }
        Ok(())
    }
}

struct CameraControllerLogic {
    pressed: HashSet<CameraAction>,
    mouse_delta: Vector2<f32>,
    move_speed: f32,
    mouse_sensitivity: f32,
    wheel_steps: f32,
    zoom_sensitivity_degrees: f32,
}

impl CameraControllerLogic {
    fn new() -> Self {
        Self {
            pressed: HashSet::new(),
            mouse_delta: Vector2::new(0.0, 0.0),
            move_speed: 6.0,
            mouse_sensitivity: 0.003,
            wheel_steps: 0.0,
            zoom_sensitivity_degrees: 2.0,
        }
    }

    fn is_pressed(&self, action: CameraAction) -> bool {
        self.pressed.contains(&action)
    }
}

#[async_trait]
impl GameLogic for CameraControllerLogic {
    async fn on_event(&mut self, _entity: Entity, event: &Event) -> anyhow::Result<()> {
        let Some(input) = event.downcast_ref::<InputEvent>() else {
            if let Event::CloseRequested = event {
                info!("关闭请求收到，退出游戏");
            }
            return Ok(());
        };

        match *input {
            InputEvent::Clear => {
                self.pressed.clear();
                self.mouse_delta = Vector2::new(0.0, 0.0);
                self.wheel_steps = 0.0;
            }
            InputEvent::Action { action, pressed } => {
                if pressed {
                    self.pressed.insert(action);
                } else {
                    self.pressed.remove(&action);
                }
            }
            InputEvent::MouseDelta { dx, dy } => {
                self.mouse_delta.x += dx;
                self.mouse_delta.y += dy;
            }
            InputEvent::MouseWheel { steps } => {
                self.wheel_steps += steps;
            }
        }

        Ok(())
    }

    async fn update(&mut self, entity: Entity, delta: Duration) -> anyhow::Result<()> {
        let dt = delta.as_secs_f32();
        if dt <= 0.0 {
            return Ok(());
        }

        let Some(mut transform) = entity.get_component_mut::<Transform>() else {
            return Ok(());
        };

        // 平移：沿摄像机朝向的水平分量移动（+Y 作为全局“上”）。
        let basis = Camera::orientation_basis(&transform).normalize();
        let mut forward = basis.forward;
        forward.y = 0.0;
        if forward.norm_squared() > 1e-6 {
            forward = forward.normalize();
        }

        let mut right = basis.right;
        right.y = 0.0;
        if right.norm_squared() > 1e-6 {
            right = right.normalize();
        }

        let up = Vector3::new(0.0, 1.0, 0.0);

        let mut move_dir = Vector3::new(0.0, 0.0, 0.0);
        if self.is_pressed(CameraAction::MoveForward) {
            move_dir += forward;
        }
        if self.is_pressed(CameraAction::MoveBackward) {
            move_dir -= forward;
        }
        if self.is_pressed(CameraAction::MoveRight) {
            move_dir += right;
        }
        if self.is_pressed(CameraAction::MoveLeft) {
            move_dir -= right;
        }
        if self.is_pressed(CameraAction::MoveUp) {
            move_dir += up;
        }
        if self.is_pressed(CameraAction::MoveDown) {
            move_dir -= up;
        }

        if move_dir.norm_squared() > 1e-6 {
            let step = move_dir.normalize() * (self.move_speed * dt);
            transform.translate(step);
        }

        // 旋转：鼠标控制（yaw 绕 Y 轴，pitch 绕 X 轴）。
        // 注意 CursorMoved 的 y 轴向下G::Cl为正，因此这里对 pitch 取负号以保持“鼠标上移=抬头”。
        let mut rotation = transform.rotation();

        let mouse_delta = self.mouse_delta;
        self.mouse_delta = Vector2::new(0.0, 0.0);

        if mouse_delta.x.abs() > 0.0 {
            // 右移鼠标应向右转（yaw 变小）。
            rotation.y -= mouse_delta.x * self.mouse_sensitivity;
        }
        if mouse_delta.y.abs() > 0.0 {
            rotation.x = (rotation.x - mouse_delta.y * self.mouse_sensitivity).clamp(-1.55, 1.55);
        }

        transform.set_rotation(rotation);

        // 视角（垂直半视场角）调节：滚轮上推 = 缩小视角（更“放大”）。
        let wheel_steps = self.wheel_steps;
        self.wheel_steps = 0.0;
        if wheel_steps.abs() > f32::EPSILON {
            if let Some(mut camera) = entity.get_component_mut::<Camera>() {
                let current = camera.vertical_half_fov_degrees();
                let candidate = current - wheel_steps * self.zoom_sensitivity_degrees;
                let _ = camera.set_vertical_half_fov_degrees(candidate);
            }
        }

        Ok(())
    }
}

fn ground_triangles() -> Vec<[Vector3<f32>; 3]> {
    vec![
        [
            Vector3::new(-1.0, 0.0, -1.0),
            Vector3::new(-1.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, -1.0),
        ],
        [
            Vector3::new(1.0, 0.0, -1.0),
            Vector3::new(-1.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 1.0),
        ],
    ]
}

fn triangle_triangles() -> Vec<[Vector3<f32>; 3]> {
    vec![[
        Vector3::new(0.0, 0.6, 0.0),
        Vector3::new(-0.5, -0.4, 0.0),
        Vector3::new(0.5, -0.4, 0.0),
    ]]
}

fn cube_triangles() -> Vec<[Vector3<f32>; 3]> {
    let half = 0.5;
    vec![
        [
            Vector3::new(-half, -half, half),
            Vector3::new(half, -half, half),
            Vector3::new(half, half, half),
        ],
        [
            Vector3::new(-half, -half, half),
            Vector3::new(half, half, half),
            Vector3::new(-half, half, half),
        ],
        [
            Vector3::new(-half, -half, -half),
            Vector3::new(half, half, -half),
            Vector3::new(half, -half, -half),
        ],
        [
            Vector3::new(-half, -half, -half),
            Vector3::new(-half, half, -half),
            Vector3::new(half, half, -half),
        ],
        [
            Vector3::new(-half, -half, -half),
            Vector3::new(-half, -half, half),
            Vector3::new(-half, half, half),
        ],
        [
            Vector3::new(-half, -half, -half),
            Vector3::new(-half, half, half),
            Vector3::new(-half, half, -half),
        ],
        [
            Vector3::new(half, -half, -half),
            Vector3::new(half, half, half),
            Vector3::new(half, -half, half),
        ],
        [
            Vector3::new(half, -half, -half),
            Vector3::new(half, half, -half),
            Vector3::new(half, half, half),
        ],
        [
            Vector3::new(-half, half, -half),
            Vector3::new(-half, half, half),
            Vector3::new(half, half, half),
        ],
        [
            Vector3::new(-half, half, -half),
            Vector3::new(half, half, half),
            Vector3::new(half, half, -half),
        ],
        [
            Vector3::new(-half, -half, -half),
            Vector3::new(half, -half, half),
            Vector3::new(-half, -half, half),
        ],
        [
            Vector3::new(-half, -half, -half),
            Vector3::new(half, -half, -half),
            Vector3::new(half, -half, half),
        ],
    ]
}

fn compute_cube_uv_patches(
    triangles: &[[Vector3<f32>; 3]],
) -> anyhow::Result<Vec<[Vector2<f32>; 3]>> {
    #[derive(Copy, Clone)]
    enum CubeFace {
        Front,
        Back,
        Left,
        Right,
        Top,
        Bottom,
    }

    // 图集布局（你更新后的结构）：
    // - 每个面占据一个 20x20 的格子（cell）
    // - 格子中心 16x16 为有效图像
    // - 四周 2px 为透明边（用于避免线性过滤时串到相邻面）
    // 该布局仍沿用旧的“方块展开图”排布，只是把 tile 从 16x16 扩展为 20x20。
    let width = 80.0;
    let height = 60.0;
    let cell = 20.0;
    let inner = 16.0;
    let inset = 2.0;
    let inner_u = inner / width;
    let inner_v = inner / height;
    let inset_u = inset / width;
    let inset_v = inset / height;
    let half = 0.5;

    fn identify_face(triangle: &[Vector3<f32>; 3], half: f32) -> anyhow::Result<CubeFace> {
        let eps = 1.0e-4;
        let is_z_pos = triangle.iter().all(|v| (v.z - half).abs() <= eps);
        if is_z_pos {
            return Ok(CubeFace::Front);
        }
        let is_z_neg = triangle.iter().all(|v| (v.z + half).abs() <= eps);
        if is_z_neg {
            return Ok(CubeFace::Back);
        }
        let is_x_pos = triangle.iter().all(|v| (v.x - half).abs() <= eps);
        if is_x_pos {
            return Ok(CubeFace::Right);
        }
        let is_x_neg = triangle.iter().all(|v| (v.x + half).abs() <= eps);
        if is_x_neg {
            return Ok(CubeFace::Left);
        }
        let is_y_pos = triangle.iter().all(|v| (v.y - half).abs() <= eps);
        if is_y_pos {
            return Ok(CubeFace::Top);
        }
        let is_y_neg = triangle.iter().all(|v| (v.y + half).abs() <= eps);
        if is_y_neg {
            return Ok(CubeFace::Bottom);
        }

        Err(anyhow!("无法识别的面：三角形不在轴对齐立方体表面"))
    }

    let origin = |x: f32, y: f32| Vector2::new(x / width, y / height);
    let top_origin = origin(cell * 1.0, cell * 0.0);
    let bottom_origin = origin(cell * 1.0, cell * 2.0);
    let front_origin = origin(cell * 1.0, cell * 1.0);
    let back_origin = origin(cell * 3.0, cell * 1.0);
    let left_origin = origin(cell * 0.0, cell * 1.0);
    let right_origin = origin(cell * 2.0, cell * 1.0);

    let mut patches = Vec::with_capacity(triangles.len());
    for triangle in triangles {
        let face = identify_face(triangle, half)?;
        let mut uv_patch = [Vector2::zeros(); 3];

        for (index, vertex) in triangle.iter().enumerate() {
            let (origin, u_offset, v_offset) = match face {
                CubeFace::Front => (front_origin, vertex.x + half, half - vertex.y),
                CubeFace::Back => (back_origin, half - vertex.x, half - vertex.y),
                CubeFace::Left => (left_origin, vertex.z + half, half - vertex.y),
                CubeFace::Right => (right_origin, half - vertex.z, half - vertex.y),
                CubeFace::Top => (top_origin, vertex.x + half, vertex.z + half),
                CubeFace::Bottom => (bottom_origin, vertex.x + half, half - vertex.z),
            };

            let u = u_offset.clamp(0.0, 1.0);
            let v = v_offset.clamp(0.0, 1.0);
            // 将 UV 映射到 cell 中央的 16x16（跳过四周 2px 透明边）。
            let u = origin.x + inset_u + u * inner_u;
            let v = origin.y + inset_v + v * inner_v;
            uv_patch[index] = Vector2::new(u, v);
        }

        patches.push(uv_patch);
    }

    Ok(patches)
}

struct CubeLogic;

#[async_trait]
impl GameLogic for CubeLogic {
    async fn on_attach(&mut self, _entity: Entity) -> anyhow::Result<()> {
        Ok(())
    }

    async fn on_render(&mut self, entity: Entity, delta: Duration) -> anyhow::Result<()> {
        if let Some(mut trans) = entity.get_component_mut::<Transform>() {
            let new_pos = trans.position() + Vector3::new(1., 0., 0.) * 0.8 * delta.as_secs_f32();
            trans.set_position(new_pos);
        }
        Ok(())
    }

    async fn update(&mut self, _entity: Entity, _delta: Duration) -> anyhow::Result<()> {
        Ok(())
    }

    async fn on_detach(&mut self, _entity: Entity) -> anyhow::Result<()> {
        info!("Cube 节点 on_detach 被调用");
        Ok(())
    }
}
