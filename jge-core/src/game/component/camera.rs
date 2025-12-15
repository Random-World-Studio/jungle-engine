use super::transform::Transform;
use super::{component, component_impl};
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
pub struct Camera {
    name: String,
    vertical_fov: f32,
    near_plane: f32,
    far_plane: f32,
    reference_framebuffer_height: u32,
}

#[component_impl]
impl Camera {
    #[allow(dead_code)]
    #[default(Transform::new())]
    fn ensure_defaults(_transform: Transform) -> Self {
        Self::new("auto_camera")
    }
    /// 使用默认参数创建摄像机。
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            vertical_fov: DEFAULT_VERTICAL_FOV,
            near_plane: DEFAULT_NEAR_PLANE,
            far_plane: DEFAULT_FAR_PLANE,
            reference_framebuffer_height: DEFAULT_REFERENCE_FRAMEBUFFER_HEIGHT,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn vertical_fov(&self) -> f32 {
        self.vertical_fov
    }

    /// 设置摄像机的垂直视场角（弧度制）。
    pub fn set_vertical_fov(&mut self, fov: f32) -> Result<(), CameraPropertyError> {
        if !(MIN_VERTICAL_FOV..MAX_VERTICAL_FOV).contains(&fov) {
            return Err(CameraPropertyError::VerticalFovOutOfRange(fov));
        }
        self.vertical_fov = fov;
        Ok(())
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

    pub fn near_plane(&self) -> f32 {
        self.near_plane
    }

    pub fn far_plane(&self) -> f32 {
        self.far_plane
    }

    /// 同时设置近、远裁剪面。
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

        let forward = rotation_matrix * Vector3::new(0.0, -1.0, 0.0);
        let up = rotation_matrix * Vector3::new(0.0, 0.0, 1.0);
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

    fn prepare_entity(entity: &Entity, node_name: &str) {
        let _ = entity.unregister_component::<Camera>();
        let _ = entity.unregister_component::<Transform>();
        let _ = entity.unregister_component::<Renderable>();
        let _ = Node::detach(*entity);
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
        let _ = Node::detach(entity);
        let _ = entity.unregister_component::<Node>();

        let missing = entity.register_component(Camera::new("primary"));
        assert!(matches!(
            missing,
            Err(crate::game::component::ComponentDependencyError { .. })
        ));

        prepare_entity(&entity, "camera_node");
        let inserted = entity
            .register_component(Camera::new("primary"))
            .expect("满足依赖后应能插入 Camera");
        assert!(inserted.is_none());
    }

    #[test]
    fn camera_property_validations() {
        let entity = Entity::new().expect("应能创建实体");
        prepare_entity(&entity, "camera_props");

        let mut camera = Camera::new("test");
        assert!(camera.set_vertical_fov(MIN_VERTICAL_FOV / 2.0).is_err());
        assert!(camera.set_vertical_fov(DEFAULT_VERTICAL_FOV * 1.2).is_ok());
        assert!(camera.set_reference_framebuffer_height(0).is_err());
        assert!(camera.set_reference_framebuffer_height(1440).is_ok());
        assert!(camera.set_clip_planes(0.0, camera.far_plane()).is_err());
        assert!(camera.set_clip_planes(0.5, 0.4).is_err());
        assert!(camera.set_clip_planes(0.5, 200.0).is_ok());
    }

    #[test]
    fn camera_vertical_fov_respects_aspect_ratio() {
        let entity = Entity::new().expect("应能创建实体");
        prepare_entity(&entity, "camera_fov");

        let camera = Camera::new("fov");
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
