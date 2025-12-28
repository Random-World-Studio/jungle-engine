use std::{
    collections::HashSet,
    f32::consts::FRAC_PI_4,
    sync::{Arc, Mutex as StdMutex},
    time::Duration,
};

use anyhow::{Context, anyhow, ensure};
use async_trait::async_trait;
use nalgebra::{Vector2, Vector3};

use jge_core::{
    Game,
    config::GameConfig,
    event::{
        DeviceEvent, ElementState, Event, Key, KeyCode, KeyEvent, NamedKey, PhysicalKey,
        WindowEvent,
    },
    game::{
        component::{
            camera::Camera,
            layer::Layer,
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
        logic::{GameLogic, GameLogicHandle},
    },
    logger,
    resource::{Resource, ResourceHandle, ResourcePath},
};
use tokio::sync::Mutex;
use tracing::{info, warn};
use winit::dpi::PhysicalPosition;
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
    let root = build_demo_scene().context("构建测试场景失败")?;

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
            // 这里采用更稳的方案：抓取光标 + 隐藏光标 + 使用 DeviceEvent::MouseMotion 获取相对 delta。
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
        .with_device_event_mapper(move |event: &DeviceEvent| {
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
        })
        .with_window_event_mapper(move |event: &WindowEvent| match event {
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
        });

    game.run()
}

#[derive(Debug)]
enum InputEvent {
    Action { action: CameraAction, pressed: bool },
    MouseDelta { dx: f32, dy: f32 },
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

fn build_demo_scene() -> anyhow::Result<Entity> {
    let root = Entity::new().context("创建根实体失败")?;

    spawn_scene3d_layer(root).context("创建 Scene3D 图层失败")?;

    // spawn_scene2d_layer(root).context("创建 Scene2D 图层失败")?;

    Ok(root)
}

fn attach_to_parent(entity: Entity, parent: Entity, message: &str) -> anyhow::Result<()> {
    let mut parent_node = parent
        .get_component_mut::<Node>()
        .with_context(|| format!("{message}: 父节点缺少 Node 组件"))?;
    parent_node
        .attach(entity)
        .with_context(|| message.to_owned())?;
    Ok(())
}

fn spawn_camera(parent: Entity) -> anyhow::Result<Entity> {
    let entity = Entity::new().context("创建摄像机实体失败")?;
    let _ = entity
        .register_component(Camera::new())
        .context("为实体注册 Camera 组件失败")?;
    attach_to_parent(entity, parent, "将摄像机挂载到父节点失败")?;

    if let Some(mut transform) = entity.get_component_mut::<Transform>() {
        transform.set_position(Vector3::new(0.0, 6.0, 6.0));
        // +Y 向上：默认 forward 为 -Z，因此 pitch 需要为负才能“向下看”到地面。
        transform.set_rotation(Vector3::new(-FRAC_PI_4, 0.0, 0.0));
    }

    // 为摄像头挂载控制逻辑：WASD/QE 移动，方向键旋转。
    let logic_handle: GameLogicHandle =
        Arc::new(Mutex::new(Box::new(CameraControllerLogic::new())));
    let mut node = entity
        .get_component_mut::<Node>()
        .context("为摄像机挂载逻辑失败: 缺少 Node 组件")?;
    node.set_logic(logic_handle);

    Ok(entity)
}

struct CameraControllerLogic {
    pressed: HashSet<CameraAction>,
    mouse_delta: Vector2<f32>,
    move_speed: f32,
    mouse_sensitivity: f32,
}

impl CameraControllerLogic {
    fn new() -> Self {
        Self {
            pressed: HashSet::new(),
            mouse_delta: Vector2::new(0.0, 0.0),
            move_speed: 6.0,
            mouse_sensitivity: 0.003,
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
            return Ok(());
        };

        match *input {
            InputEvent::Clear => {
                self.pressed.clear();
                self.mouse_delta = Vector2::new(0.0, 0.0);
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
        // 注意 CursorMoved 的 y 轴向下为正，因此这里对 pitch 取负号以保持“鼠标上移=抬头”。
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

        Ok(())
    }
}

fn spawn_ground(parent: Entity) -> anyhow::Result<()> {
    let triangles = vec![
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
    ];

    let _ = spawn_shape(
        parent,
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(6.0, 1.0, 6.0),
        triangles,
        None,
    )?;

    Ok(())
}

fn spawn_point_light(parent: Entity) -> anyhow::Result<()> {
    let entity = Entity::new().context("创建点光源实体失败")?;
    let _ = entity
        .register_component(PointLight::new(15.0))
        .context("为点光源注册 PointLight 组件失败")?;
    attach_to_parent(entity, parent, "将点光源挂载到父节点失败")?;

    if let Some(mut renderable) = entity.get_component_mut::<Renderable>() {
        renderable.set_enabled(false);
    }
    if let Some(mut transform) = entity.get_component_mut::<Transform>() {
        transform.set_position(Vector3::new(6.0, 1.0, 6.0));
    }
    if let Some(mut light) = entity.get_component_mut::<Light>() {
        light.set_lightness(1.0);
    }

    Ok(())
}

fn spawn_triangle(
    parent: Entity,
    position: Vector3<f32>,
    rotation: Vector3<f32>,
    scale: Vector3<f32>,
) -> anyhow::Result<()> {
    let triangles = vec![[
        Vector3::new(0.0, 0.6, 0.0),
        Vector3::new(-0.5, -0.4, 0.0),
        Vector3::new(0.5, -0.4, 0.0),
    ]];

    spawn_shape(parent, position, rotation, scale, triangles, None).map(|_| ())
}

fn spawn_cube(parent: Entity, position: Vector3<f32>) -> anyhow::Result<()> {
    let half = 0.5;
    let triangles = vec![
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
    ];

    let (material_handle, patches) =
        prepare_bamboo_material(&triangles).context("准备竹子材质失败")?;

    let cube = spawn_shape(
        parent,
        position,
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
        triangles,
        Some((material_handle, patches)),
    )?;

    // Attach demo logic so each frame reports render timing for the cube entity.
    let logic_handle: GameLogicHandle = Arc::new(Mutex::new(Box::new(CubeLogic)));
    let mut node = cube
        .get_component_mut::<Node>()
        .context("为立方体注册逻辑失败: 缺少 Node 组件")?;
    node.set_logic(logic_handle);

    Ok(())
}

fn prepare_bamboo_material(
    triangles: &[[Vector3<f32>; 3]],
) -> anyhow::Result<(ResourceHandle, Vec<[Vector2<f32>; 3]>)> {
    let resource_path = ResourcePath::from("textures/bamboo.png");
    Resource::register(
        resource_path.clone(),
        Resource::from_memory(Vec::from(include_bytes!("resource/bamboo.png"))),
    )
    .context("注册竹子材质资源失败")?;
    let handle = Resource::from(resource_path.clone()).context("获取竹子材质资源句柄失败")?;

    let patches = compute_cube_uv_patches(triangles)?;
    Ok((handle, patches))
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

fn spawn_shape(
    parent: Entity,
    position: Vector3<f32>,
    rotation: Vector3<f32>,
    scale: Vector3<f32>,
    triangles: Vec<[Vector3<f32>; 3]>,
    material: Option<(ResourceHandle, Vec<[Vector2<f32>; 3]>)>,
) -> anyhow::Result<Entity> {
    let entity = Entity::new().context("创建子实体失败")?;
    let shape = Shape::from_triangles(triangles);
    let triangle_count = shape.triangle_count();
    let _ = entity
        .register_component(shape)
        .context("为子实体注册 Shape 组件失败")?;
    attach_to_parent(entity, parent, "将子实体挂载到父节点失败")?;

    if let Some((resource, regions)) = material {
        ensure!(
            regions.len() == triangle_count,
            "材质贴图区域数量与三角形数量不匹配"
        );
        let _ = entity
            .register_component(Material::new(resource, regions))
            .context("为子实体注册 Material 组件失败")?;
    }

    if let Some(mut transform) = entity.get_component_mut::<Transform>() {
        transform.set_position(position);
        transform.set_rotation(rotation);
        transform.set_scale(scale);
    }

    Ok(entity)
}

fn spawn_parallel_light(parent: Entity) -> anyhow::Result<()> {
    let entity = Entity::new().context("创建平行光实体失败")?;
    let _ = entity
        .register_component(ParallelLight::new())
        .context("为平行光注册 ParallelLight 组件失败")?;
    attach_to_parent(entity, parent, "将平行光挂载到父节点失败")?;

    if let Some(mut renderable) = entity.get_component_mut::<Renderable>() {
        renderable.set_enabled(false);
    }

    if let Some(mut transform) = entity.get_component_mut::<Transform>() {
        transform.set_rotation(Vector3::new(-0.3, -std::f32::consts::PI, 0.0));
    }

    if let Some(mut light) = entity.get_component_mut::<Light>() {
        light.set_lightness(0.7);
    }

    Ok(())
}

fn spawn_scene3d_layer(parent: Entity) -> anyhow::Result<Entity> {
    let entity = Entity::new().context("创建 Scene3D 图层实体失败")?;
    let _ = entity
        .register_component(Layer::new())
        .context("为 Scene3D 图层注册 Layer 组件失败")?;
    let _ = entity
        .register_component(Scene3D::new())
        .context("注册 Scene3D 组件失败")?;
    attach_to_parent(entity, parent, "将 Scene3D 图层挂载到父节点失败")?;

    let camera = spawn_camera(entity).context("创建 Scene3D 摄像机失败")?;
    {
        let mut scene = entity
            .get_component_mut::<Scene3D>()
            .context("Scene3D 图层缺少 Scene3D 组件")?;
        scene
            .bind_camera(camera)
            .context("为 Scene3D 图层绑定摄像机失败")?;
    }

    {
        let scene = entity
            .get_component::<Scene3D>()
            .context("Scene3D 图层缺少 Scene3D 组件")?;
        scene
            .sync_camera_transform()
            .context("同步 Scene3D 摄像机变换失败")?;
    }

    spawn_ground(entity).context("Scene3D 图层创建地面失败")?;
    spawn_point_light(entity).context("Scene3D 图层创建点光源失败")?;
    spawn_parallel_light(entity).context("Scene3D 图层创建平行光失败")?;

    spawn_triangle(
        entity,
        Vector3::new(0.0, 0.5, -4.0),
        Vector3::new(0.0, 0.3, 0.0),
        Vector3::new(1.6, 1.6, 1.6),
    )
    .context("Scene3D 图层创建中央三角形失败")?;

    spawn_triangle(
        entity,
        Vector3::new(-2.5, 0.9, -6.0),
        Vector3::new(0.2, -0.3, 0.1),
        Vector3::new(1.2, 1.2, 1.2),
    )
    .context("Scene3D 图层创建侧边三角形失败")?;

    spawn_cube(entity, Vector3::new(-3.0, 0.5, 0.0)).context("Scene3D 图层创建立方体失败")?;

    Ok(entity)
}

fn spawn_scene2d_layer(parent: Entity) -> anyhow::Result<Entity> {
    let entity = Entity::new().context("创建 Scene2D 图层实体失败")?;
    let _ = entity
        .register_component(Layer::new())
        .context("为 Scene2D 图层注册 Layer 组件失败")?;
    let _ = entity
        .register_component(Scene2D::new())
        .context("注册 Scene2D 组件失败")?;
    attach_to_parent(entity, parent, "将 Scene2D 图层挂载到父节点失败")?;

    {
        let scene = entity
            .get_component::<Scene2D>()
            .context("Scene2D 图层缺少 Scene2D 组件")?;
        let mut layer = entity
            .get_component_mut::<Layer>()
            .context("Scene2D 图层缺少 Layer 组件")?;
        if !scene.warmup_lod(&mut layer) {
            info!(target = "jge-demo", "Scene2D LOD warmup 未激活任何节点");
        }
    }

    spawn_scene2d_quad(entity).context("创建 Scene2D 正方形失败")?;
    spawn_parallel_light(entity).context("为 Scene2D 图层创建平行光失败")?;

    Ok(entity)
}

fn spawn_scene2d_quad(parent: Entity) -> anyhow::Result<()> {
    let triangles = vec![
        [
            Vector3::new(-0.5, -0.5, 0.0),
            Vector3::new(0.5, -0.5, 0.0),
            Vector3::new(0.5, 0.5, 0.0),
        ],
        [
            Vector3::new(-0.5, -0.5, 0.0),
            Vector3::new(0.5, 0.5, 0.0),
            Vector3::new(-0.5, 0.5, 0.0),
        ],
    ];

    let _ = spawn_shape(
        parent,
        Vector3::new(0.0, 0.0, 0.1),
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(2.0, 2.0, 1.0),
        triangles,
        None,
    )?;

    Ok(())
}

struct CubeLogic;

#[async_trait]
impl GameLogic for CubeLogic {
    fn on_attach(&mut self, _entity: Entity) -> anyhow::Result<()> {
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

    fn on_detach(&mut self, _entity: Entity) -> anyhow::Result<()> {
        Ok(())
    }
}
