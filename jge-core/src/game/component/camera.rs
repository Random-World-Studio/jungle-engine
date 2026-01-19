use super::transform::Transform;
use super::{component, component_impl};
use crate::game::entity::Entity;
use nalgebra::{Rotation3, Vector3};
use std::f32::consts::PI;

/// 摄像机的默认垂直视场（以弧度表示）。
const DEFAULT_VERTICAL_FOV: f32 = 60.0_f32.to_radians();
/// 摄像机默认的近裁剪面距离。
const DEFAULT_NEAR_PLANE: f32 = 0.1;
/// 摄像机默认的远裁剪面距离（视距）。
const DEFAULT_FAR_PLANE: f32 = 1024.0;
/// 允许的最小垂直视场角（接近 1°）。
const MIN_VERTICAL_FOV: f32 = PI / 180.0;
/// 允许的最大垂直视场角（略小于 180°，避免 tan 函数发散）。
const MAX_VERTICAL_FOV: f32 = PI - MIN_VERTICAL_FOV;
const DEFAULT_REFERENCE_FRAMEBUFFER_HEIGHT: u32 = 1080;

#[component(Transform)]
#[derive(Debug, Clone)]
/// 3D 摄像机组件。
///
/// 摄像机定义了视锥参数（FOV、近/远裁剪面等），并依赖实体的 [`Transform`](jge_core::game::component::transform::Transform)
/// 来确定相机在世界中的位置与朝向。
///
/// 常见用法：把 `Camera` 挂在一个实体上，然后在 `Scene3D` 中绑定该实体作为当前摄像机。
///
/// # 示例
///
/// ```no_run
/// use jge_core::game::{
///     component::{camera::Camera, transform::Transform},
///     entity::Entity,
/// };
///
/// # fn main() -> anyhow::Result<()> {
/// let camera = Entity::new()?;
/// camera.register_component(Transform::new())?;
/// camera.register_component(Camera::new())?;
/// Ok(())
/// # }
/// ```
pub struct Camera {
    entity_id: Option<Entity>,
    vertical_fov: f32,
    near_plane: f32,
    far_plane: f32,
    reference_framebuffer_height: u32,
}

#[component_impl]
impl Camera {
    /// 使用默认参数创建摄像机。
    #[default()]
    pub fn new() -> Self {
        Self {
            entity_id: None,
            vertical_fov: DEFAULT_VERTICAL_FOV,
            near_plane: DEFAULT_NEAR_PLANE,
            far_plane: DEFAULT_FAR_PLANE,
            reference_framebuffer_height: DEFAULT_REFERENCE_FRAMEBUFFER_HEIGHT,
        }
    }

    pub fn vertical_fov(&self) -> f32 {
        self.vertical_fov
    }

    /// 获取摄像机的垂直半视场角（弧度制）。
    ///
    /// 该角度定义为 3D 视锥的“棱”（边界射线）与视锥对称轴（forward 方向）的夹角，
    /// 也就是 `vertical_fov / 2`。
    pub fn vertical_half_fov(&self) -> f32 {
        0.5 * self.vertical_fov
    }

    /// 获取摄像机的垂直半视场角（角度制）。
    pub fn vertical_half_fov_degrees(&self) -> f32 {
        self.vertical_half_fov().to_degrees()
    }

    /// 设置摄像机的垂直视场角（弧度制）。
    pub fn set_vertical_fov(&mut self, fov: f32) -> Result<(), CameraPropertyError> {
        if !(MIN_VERTICAL_FOV..MAX_VERTICAL_FOV).contains(&fov) {
            return Err(CameraPropertyError::VerticalFovOutOfRange(fov));
        }
        self.vertical_fov = fov;
        Ok(())
    }

    /// 设置摄像机的垂直半视场角（弧度制）。
    ///
    /// 该值会被转换为垂直全视场角：`vertical_fov = vertical_half_fov * 2`。
    pub fn set_vertical_half_fov(&mut self, half_fov: f32) -> Result<(), CameraPropertyError> {
        self.set_vertical_fov(half_fov * 2.0)
    }

    /// 设置摄像机的垂直半视场角（角度制）。
    pub fn set_vertical_half_fov_degrees(
        &mut self,
        half_fov_degrees: f32,
    ) -> Result<(), CameraPropertyError> {
        self.set_vertical_half_fov(half_fov_degrees.to_radians())
    }

    pub fn reference_framebuffer_height(&self) -> u32 {
        self.reference_framebuffer_height
    }

    pub fn set_reference_framebuffer_height(
        &mut self,
        height: u32,
    ) -> Result<(), CameraPropertyError> {
        if height == 0 {
            return Err(CameraPropertyError::InvalidReferenceHeight(height));
        }
        self.reference_framebuffer_height = height;
        Ok(())
    }

    pub fn set_clip_planes(
        &mut self,
        near_plane: f32,
        far_plane: f32,
    ) -> Result<(), CameraPropertyError> {
        if near_plane <= 0.0 {
            return Err(CameraPropertyError::InvalidClipPlaneDistance(near_plane));
        }
        if far_plane <= near_plane {
            return Err(CameraPropertyError::InvalidClipPlaneRange {
                near: near_plane,
                far: far_plane,
            });
        }
        self.near_plane = near_plane;
        self.far_plane = far_plane;
        Ok(())
    }

    pub fn near_plane(&self) -> f32 {
        self.near_plane
    }

    pub fn far_plane(&self) -> f32 {
        self.far_plane
    }

    /// 根据窗口尺寸计算垂直视场角。
    pub fn horizontal_fov(&self, framebuffer_size: (u32, u32)) -> Result<f32, CameraViewportError> {
        let (width, height) = framebuffer_size;
        if width == 0 || height == 0 {
            return Err(CameraViewportError::InvalidViewport { width, height });
        }
        let vertical = self.vertical_fov_for_height(height)?;
        let aspect_ratio = width as f32 / height as f32;
        Ok(horizontal_fov_from_vertical(vertical, aspect_ratio))
    }

    pub fn vertical_fov_for_height(
        &self,
        framebuffer_height: u32,
    ) -> Result<f32, CameraViewportError> {
        if framebuffer_height == 0 {
            return Err(CameraViewportError::InvalidViewport {
                width: 0,
                height: framebuffer_height,
            });
        }
        let reference = self.reference_framebuffer_height as f32;
        let half_base = 0.5 * self.vertical_fov;
        let scale = framebuffer_height as f32 / reference;
        let scaled_tan = half_base.tan() * scale;
        Ok((scaled_tan).atan() * 2.0)
    }

    /// 基于当前姿态计算摄像机的方向基底。
    pub fn orientation_basis(transform: &Transform) -> CameraBasis {
        let rotation = transform.rotation();
        let rotation_matrix = Rotation3::from_euler_angles(rotation.x, rotation.y, rotation.z);

        // 坐标系约定：+Y 向上；默认朝向为 -Z（场景中 z 越小越“前”）。
        let forward = rotation_matrix * Vector3::new(0.0, 0.0, -1.0);
        let up = rotation_matrix * Vector3::new(0.0, 1.0, 0.0);
        let right = rotation_matrix * Vector3::new(1.0, 0.0, 0.0);

        CameraBasis { forward, up, right }
    }
}

/// 摄像机观察空间的三个正交基向量。
#[derive(Debug, Clone, Copy)]
pub struct CameraBasis {
    pub forward: Vector3<f32>,
    pub up: Vector3<f32>,
    pub right: Vector3<f32>,
}

impl CameraBasis {
    pub fn normalize(self) -> Self {
        let forward = nalgebra::Unit::new_normalize(self.forward);
        let up = nalgebra::Unit::new_normalize(self.up);
        let right = nalgebra::Unit::new_normalize(self.right);
        Self {
            forward: *forward,
            up: *up,
            right: *right,
        }
    }
}

fn horizontal_fov_from_vertical(vertical_fov: f32, aspect_ratio: f32) -> f32 {
    let half_vertical = 0.5 * vertical_fov;
    let horizontal_half = (half_vertical.tan() * aspect_ratio).atan();
    horizontal_half * 2.0
}

#[derive(Debug, PartialEq)]
pub enum CameraPropertyError {
    VerticalFovOutOfRange(f32),
    InvalidClipPlaneDistance(f32),
    InvalidClipPlaneRange { near: f32, far: f32 },
    InvalidReferenceHeight(u32),
}

impl std::fmt::Display for CameraPropertyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CameraPropertyError::VerticalFovOutOfRange(fov) => {
                write!(f, "垂直视场角超出范围: {}", fov)
            }
            CameraPropertyError::InvalidClipPlaneDistance(distance) => {
                write!(f, "裁剪面距离必须为正值，收到 {}", distance)
            }
            CameraPropertyError::InvalidClipPlaneRange { near, far } => {
                write!(f, "远裁剪面 ({}) 必须大于近裁剪面 ({})", far, near)
            }
            CameraPropertyError::InvalidReferenceHeight(height) => {
                write!(f, "参考帧高度必须大于零，收到 {}", height)
            }
        }
    }
}

impl std::error::Error for CameraPropertyError {}

#[derive(Debug, PartialEq)]
pub enum CameraViewportError {
    InvalidViewport { width: u32, height: u32 },
}

impl std::fmt::Display for CameraViewportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CameraViewportError::InvalidViewport { width, height } => write!(
                f,
                "无效的视口尺寸: 宽度 {}、高度 {} 必须大于零",
                width, height
            ),
        }
    }
}

impl std::error::Error for CameraViewportError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::component::{node::Node, renderable::Renderable};
    use crate::game::entity::Entity;

    fn detach_node(entity: Entity) {
        if let Some(mut node) = entity.get_component_mut::<Node>() {
            let _ = node.detach();
        }
    }

    fn prepare_entity(entity: &Entity, node_name: &str) {
        let _ = entity.unregister_component::<Camera>();
        let _ = entity.unregister_component::<Transform>();
        let _ = entity.unregister_component::<Renderable>();
        detach_node(*entity);
        let _ = entity.unregister_component::<Node>();

        let _ = entity
            .register_component(Node::new(node_name).expect("应能创建节点"))
            .expect("应能插入 Node");
        let _ = entity
            .register_component(Renderable::new())
            .expect("应能插入 Renderable");
        let _ = entity
            .register_component(Transform::new())
            .expect("应能插入 Transform");
    }

    #[test]
    fn camera_requires_transform_dependency() {
        let entity = Entity::new().expect("应能创建实体");
        let _ = entity.unregister_component::<Camera>();
        let _ = entity.unregister_component::<Transform>();
        let _ = entity.unregister_component::<Renderable>();
        detach_node(entity);
        let _ = entity.unregister_component::<Node>();

        let inserted = entity
            .register_component(Camera::new())
            .expect("缺少 Transform 时应自动注册依赖");
        assert!(inserted.is_none());

        assert!(
            entity.get_component::<Transform>().is_some(),
            "Transform 应被自动注册"
        );
        assert!(
            entity.get_component::<Renderable>().is_some(),
            "Renderable 应被自动注册"
        );
        assert!(
            entity.get_component::<Node>().is_some(),
            "Node 应被自动注册"
        );

        let previous = entity
            .register_component(Camera::new())
            .expect("重复插入应返回旧的 Camera");
        assert!(previous.is_some());

        prepare_entity(&entity, "camera_node");
        let reinserted = entity
            .register_component(Camera::new())
            .expect("满足依赖后应能插入 Camera");
        assert!(reinserted.is_none());
    }

    #[test]
    fn camera_property_validations() {
        let entity = Entity::new().expect("应能创建实体");
        prepare_entity(&entity, "camera_props");

        let mut camera = Camera::new();
        assert!(camera.set_vertical_fov(MIN_VERTICAL_FOV / 2.0).is_err());
        assert!(camera.set_vertical_fov(DEFAULT_VERTICAL_FOV * 1.2).is_ok());
        assert!(camera.set_reference_framebuffer_height(0).is_err());
        assert!(camera.set_reference_framebuffer_height(1440).is_ok());
        assert!(camera.set_clip_planes(0.0, camera.far_plane()).is_err());
        assert!(camera.set_clip_planes(0.5, 0.4).is_err());
        assert!(camera.set_clip_planes(0.5, 200.0).is_ok());
    }

    #[test]
    fn camera_supports_vertical_half_fov_api() {
        let mut camera = Camera::new();

        let base_full = camera.vertical_fov();
        let base_half = camera.vertical_half_fov();
        assert!((base_half * 2.0 - base_full).abs() < 1e-6);

        camera
            .set_vertical_half_fov_degrees(20.0)
            .expect("半视场角应能设置");
        assert!((camera.vertical_half_fov_degrees() - 20.0).abs() < 1e-6);
        assert!((camera.vertical_fov().to_degrees() - 40.0).abs() < 1e-6);

        // 太小（小于 MIN_VERTICAL_FOV/2）应被拒绝。
        assert!(
            camera
                .set_vertical_half_fov(MIN_VERTICAL_FOV / 4.0)
                .is_err()
        );
    }

    #[test]
    fn camera_vertical_fov_respects_aspect_ratio() {
        let entity = Entity::new().expect("应能创建实体");
        prepare_entity(&entity, "camera_fov");

        let camera = Camera::new();
        let vertical_base = camera
            .vertical_fov_for_height(1080)
            .expect("有效视口不应报错");
        let vertical_double = camera
            .vertical_fov_for_height(2160)
            .expect("有效视口不应报错");
        let base_ratio = (0.5 * vertical_base).tan() / 1080.0;
        let double_ratio = (0.5 * vertical_double).tan() / 2160.0;
        assert!((base_ratio - double_ratio).abs() < 1e-6);

        let horizontal_wide = camera
            .horizontal_fov((1920, 1080))
            .expect("有效视口不应报错");
        let horizontal_tall = camera
            .horizontal_fov((1080, 1920))
            .expect("有效视口不应报错");
        assert!(horizontal_wide > horizontal_tall);
        assert!(camera.horizontal_fov((0, 1080)).is_err());
    }
}
