use anyhow::{Context, anyhow, ensure};
use jge_core::{
    Game,
    config::GameConfig,
    game::{
        component::{
            camera::Camera,
            light::{Light, ParallelLight, PointLight},
            material::Material,
            node::Node,
            renderable::Renderable,
            scene3d::Scene3D,
            shape::Shape,
            transform::Transform,
        },
        entity::Entity,
    },
    logger,
    resource::{Resource, ResourceHandle, ResourcePath},
};
use nalgebra::{Vector2, Vector3};
use std::f32::consts::FRAC_PI_4;

fn main() -> anyhow::Result<()> {
    logger::init()?;
    let root = build_demo_scene().context("构建测试场景失败")?;
    let game = Game::new(GameConfig::default(), root)?;
    game.run()
}

fn build_demo_scene() -> anyhow::Result<Entity> {
    let root = Entity::new().context("创建根实体失败")?;

    let _ = Scene3D::insert(root, Scene3D::new("tpl_scene3d"))
        .context("为根实体注册 Scene3D 组件失败")?;

    let camera = spawn_camera(root).context("创建摄像机失败")?;
    Scene3D::attach_camera(root, camera).context("绑定摄像机失败")?;

    spawn_ground(root).context("创建地面失败")?;

    spawn_point_light(root).context("创建点光源失败")?;
    spawn_parallel_light(root).context("创建平行光失败")?;

    spawn_triangle(
        root,
        Vector3::new(0.0, 0.5, -4.0),
        Vector3::new(0.0, 0.3, 0.0),
        Vector3::new(1.6, 1.6, 1.6),
    )
    .context("创建中央三角形失败")?;

    spawn_triangle(
        root,
        Vector3::new(-2.5, 0.9, -6.0),
        Vector3::new(0.2, -0.3, 0.1),
        Vector3::new(1.2, 1.2, 1.2),
    )
    .context("创建侧边三角形失败")?;

    spawn_cube(root, Vector3::new(-3.0, 0.5, 0.0)).context("创建立方体失败")?;

    Scene3D::sync_attached_transform(root).context("同步 Scene3D 变换失败")?;

    Ok(root)
}

fn spawn_camera(parent: Entity) -> anyhow::Result<Entity> {
    let entity = Entity::new().context("创建摄像机实体失败")?;
    let _ = entity
        .register_component(Camera::new("main_camera"))
        .context("为实体注册 Camera 组件失败")?;
    Node::attach(entity, parent).context("将摄像机挂载到父节点失败")?;

    if let Some(mut transform) = entity.get_component_mut::<Transform>() {
        transform.set_position(Vector3::new(0.0, 6.0, 6.0));
        transform.set_rotation(Vector3::new(FRAC_PI_4, 0.0, 0.0));
    }

    Ok(entity)
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

    spawn_shape(
        parent,
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(6.0, 1.0, 6.0),
        triangles,
        None,
    )
}

fn spawn_point_light(parent: Entity) -> anyhow::Result<()> {
    let entity = Entity::new().context("创建点光源实体失败")?;
    let _ = entity
        .register_component(PointLight::new(15.0))
        .context("为点光源注册 PointLight 组件失败")?;
    Node::attach(entity, parent).context("将点光源挂载到父节点失败")?;

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

    spawn_shape(parent, position, rotation, scale, triangles, None)
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

    spawn_shape(
        parent,
        position,
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
        triangles,
        Some((material_handle, patches)),
    )
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

    fn identify_face(normal: Vector3<f32>) -> anyhow::Result<CubeFace> {
        let normalized = normal
            .try_normalize(1.0e-6)
            .ok_or_else(|| anyhow!("无法识别的面：法线长度为零"))?;
        if normalized.z > 0.5 {
            Ok(CubeFace::Front)
        } else if normalized.z < -0.5 {
            Ok(CubeFace::Back)
        } else if normalized.x > 0.5 {
            Ok(CubeFace::Right)
        } else if normalized.x < -0.5 {
            Ok(CubeFace::Left)
        } else if normalized.y > 0.5 {
            Ok(CubeFace::Top)
        } else if normalized.y < -0.5 {
            Ok(CubeFace::Bottom)
        } else {
            Err(anyhow!("无法识别的面：法线方向异常"))
        }
    }

    let width = 64.0;
    let height = 48.0;
    let tile = 16.0;
    let tile_u = tile / width;
    let tile_v = tile / height;
    let half = 0.5;

    let origin = |x: f32, y: f32| Vector2::new(x / width, y / height);
    let top_origin = origin(16.0, 0.0);
    let bottom_origin = origin(16.0, 32.0);
    let front_origin = origin(16.0, 16.0);
    let back_origin = origin(48.0, 16.0);
    let left_origin = origin(0.0, 16.0);
    let right_origin = origin(32.0, 16.0);

    let mut patches = Vec::with_capacity(triangles.len());
    for triangle in triangles {
        let normal = (triangle[1] - triangle[0]).cross(&(triangle[2] - triangle[0]));
        let face = identify_face(normal)?;
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

            uv_patch[index] =
                Vector2::new(origin.x + u_offset * tile_u, origin.y + v_offset * tile_v);
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
) -> anyhow::Result<()> {
    let entity = Entity::new().context("创建子实体失败")?;
    let shape = Shape::from_triangles(triangles);
    let triangle_count = shape.triangle_count();
    let _ = entity
        .register_component(shape)
        .context("为子实体注册 Shape 组件失败")?;
    Node::attach(entity, parent).context("将子实体挂载到父节点失败")?;

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

    Ok(())
}

fn spawn_parallel_light(parent: Entity) -> anyhow::Result<()> {
    let entity = Entity::new().context("创建平行光实体失败")?;
    let _ = entity
        .register_component(ParallelLight::new())
        .context("为平行光注册 ParallelLight 组件失败")?;
    Node::attach(entity, parent).context("将平行光挂载到父节点失败")?;

    if let Some(mut renderable) = entity.get_component_mut::<Renderable>() {
        renderable.set_enabled(false);
    }

    if let Some(mut transform) = entity.get_component_mut::<Transform>() {
        transform.set_rotation(Vector3::new(-0.3, -3.14, 0.0));
    }

    if let Some(mut light) = entity.get_component_mut::<Light>() {
        light.set_lightness(0.7);
    }

    Ok(())
}
