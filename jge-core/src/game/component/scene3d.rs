//! 三维场景组件（`Scene3D`）。
//!
//! 把一个挂载了 [`Layer`](jge_core::game::component::layer::Layer) 的实体声明为“3D 渲染层”。
//! 典型用法：
//! - 在 Layer 根实体上注册 `Scene3D`（会自动补齐默认 3D 着色器）。
//! - 选择一个摄像机实体（挂载 [`Camera`](jge_core::game::component::camera::Camera) + `Transform`），并绑定到该场景。
//! - 每帧通过 `Scene3D::visible_renderables` 获取当前视锥内的可见几何集合，用于渲染。

use super::{
    camera::{Camera, CameraBasis, CameraViewportError},
    component_impl,
    layer::{
        Layer, LayerLodUpdate, LayerRenderableBundle, LayerRenderableCollection,
        LayerTraversalError, RenderPipelineStage, ShaderLanguage,
    },
};
use crate::resource::ResourcePath;
use lodtree::coords::OctVec;
use nalgebra::Vector3;
use std::{f32::consts::PI, sync::OnceLock};

use crate::game::{
    component::{
        Component, ComponentDependencyError, ComponentRead, ComponentStorage, ComponentWrite,
        renderable::Renderable, transform::Transform,
    },
    entity::Entity,
};

const MIN_VERTICAL_FOV: f32 = PI / 180.0;
const MAX_VERTICAL_FOV: f32 = PI - MIN_VERTICAL_FOV;
const DEFAULT_VERTICAL_FOV: f32 = 60.0_f32.to_radians();
const DEFAULT_NEAR_PLANE: f32 = 0.1;
const DEFAULT_VIEW_DISTANCE: f32 = 1024.0;
const FRUSTUM_MARGIN: f32 = 1.0_f32.to_radians();
const DEFAULT_REFERENCE_FRAMEBUFFER_HEIGHT: u32 = 1080;

static SCENE3D_STORAGE: OnceLock<ComponentStorage<Scene3D>> = OnceLock::new();

/// 三维场景组件。
///
/// 该组件管理：
/// - 场景自身的视锥参数（FOV、近裁剪面、视距等）
/// - （可选）绑定的摄像机实体，用于把摄像机的 `Transform` 同步到场景实体
///
/// # 依赖
/// - 该组件依赖 [`Layer`](jge_core::game::component::layer::Layer)。
/// - 3D 场景实体通常也会有 [`Transform`](jge_core::game::component::transform::Transform)，用于渲染时的坐标基准。
///
/// # 示例
///
/// ```no_run
/// # use jge_core::game::{entity::Entity, component::{scene3d::Scene3D, layer::Layer, camera::Camera, transform::Transform}};
/// # fn main() -> anyhow::Result<()> {
/// let scene = Entity::new()?;
/// scene.register_component(Layer::new())?;
/// scene.register_component(Transform::new())?;
/// scene.register_component(Scene3D::new())?;
///
/// let camera = Entity::new()?;
/// camera.register_component(Transform::new())?;
/// camera.register_component(Camera::new())?;
///
/// if let Some(mut scene3d) = scene.get_component_mut::<Scene3D>() {
///     scene3d.bind_camera(camera)?;
/// }
/// Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Scene3D {
    entity_id: Option<Entity>,
    vertical_fov: f32,
    reference_framebuffer_height: u32,
    near_plane: f32,
    view_distance: f32,
    attached_camera: Option<Entity>,
}

#[component_impl]
impl Scene3D {
    /// 创建一个 3D 场景组件。
    #[default()]
    pub fn new() -> Self {
        Self {
            entity_id: None,
            vertical_fov: DEFAULT_VERTICAL_FOV,
            reference_framebuffer_height: DEFAULT_REFERENCE_FRAMEBUFFER_HEIGHT,
            near_plane: DEFAULT_NEAR_PLANE,
            view_distance: DEFAULT_VIEW_DISTANCE,
            attached_camera: None,
        }
    }

    /// 获取该组件绑定到的实体。
    ///
    /// 仅在组件已通过 `Entity::register_component` 挂载后可用。
    pub fn entity(&self) -> Entity {
        self.entity_id.expect("Scene3D 组件尚未绑定实体")
    }

    fn ensure_default_layer_shaders(entity: Entity) {
        if let Some(mut layer) = entity.get_component_mut::<Layer>() {
            if layer.shader(RenderPipelineStage::Vertex).is_none() {
                let _ = layer.attach_shader_from_path(
                    RenderPipelineStage::Vertex,
                    ShaderLanguage::Wgsl,
                    ResourcePath::from("shaders/3d.vs"),
                );
            }
            if layer.shader(RenderPipelineStage::Fragment).is_none() {
                let _ = layer.attach_shader_from_path(
                    RenderPipelineStage::Fragment,
                    ShaderLanguage::Wgsl,
                    ResourcePath::from("shaders/3d.fs"),
                );
            }
        }
    }

    /// 当前垂直视场角（弧度）。
    pub fn vertical_fov(&self) -> f32 {
        self.vertical_fov
    }

    /// 设置垂直视场角（弧度）。
    pub fn set_vertical_fov(&mut self, fov: f32) -> Result<(), Scene3DPropertyError> {
        if !(MIN_VERTICAL_FOV..MAX_VERTICAL_FOV).contains(&fov) {
            return Err(Scene3DPropertyError::VerticalFovOutOfRange(fov));
        }
        self.vertical_fov = fov;
        Ok(())
    }

    /// 设置参考帧缓冲高度（用于让 FOV 随窗口高度缩放）。
    ///
    /// 该值必须大于 0。
    pub fn set_reference_framebuffer_height(
        &mut self,
        height: u32,
    ) -> Result<(), Scene3DPropertyError> {
        if height == 0 {
            return Err(Scene3DPropertyError::InvalidReferenceHeight(height));
        }
        self.reference_framebuffer_height = height;
        Ok(())
    }

    /// 当前近裁剪面距离。
    pub fn near_plane(&self) -> f32 {
        self.near_plane
    }

    /// 设置近裁剪面距离。
    ///
    /// 必须满足 `near_plane > 0` 且 `near_plane < view_distance`。
    pub fn set_near_plane(&mut self, near_plane: f32) -> Result<(), Scene3DPropertyError> {
        if near_plane <= 0.0 {
            return Err(Scene3DPropertyError::InvalidNearPlane(near_plane));
        }
        if near_plane >= self.view_distance {
            return Err(Scene3DPropertyError::InvalidViewDistanceRange {
                near: near_plane,
                distance: self.view_distance,
            });
        }
        self.near_plane = near_plane;
        Ok(())
    }

    /// 当前绑定的摄像机实体（若存在）。
    pub fn attached_camera(&self) -> Option<Entity> {
        self.attached_camera
    }

    /// 设置场景视距（远裁剪面）。
    ///
    /// 必须满足 `distance > near_plane`。
    pub fn set_view_distance(&mut self, distance: f32) -> Result<(), Scene3DPropertyError> {
        if distance <= self.near_plane {
            return Err(Scene3DPropertyError::InvalidViewDistanceRange {
                near: self.near_plane,
                distance,
            });
        }
        self.view_distance = distance;
        Ok(())
    }

    /// 当前场景视距（远裁剪面）。
    pub fn view_distance(&self) -> f32 {
        self.view_distance
    }

    /// 根据帧缓冲高度计算“随高度缩放”的垂直视场角（弧度）。
    pub fn vertical_fov_for_height(
        &self,
        framebuffer_height: u32,
    ) -> Result<f32, Scene3DViewportError> {
        if framebuffer_height == 0 {
            return Err(Scene3DViewportError::InvalidViewport {
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

    pub(crate) fn bind_camera_internal(
        &mut self,
        camera_entity: Entity,
    ) -> Result<(), Scene3DAttachError> {
        let scene_entity = self.entity();

        if camera_entity.get_component::<Camera>().is_none() {
            return Err(Scene3DAttachError::MissingCamera(camera_entity));
        }

        let camera_transform_values = {
            let transform = try_get_transform(camera_entity)
                .ok_or(Scene3DAttachError::MissingCameraTransform(camera_entity))?;
            (
                transform.position(),
                transform.rotation(),
                transform.scale(),
            )
        };

        {
            let mut scene_transform = scene_entity
                .get_component_mut::<Transform>()
                .ok_or(Scene3DAttachError::MissingSceneTransform(scene_entity))?;
            scene_transform.set_position(camera_transform_values.0);
            scene_transform.set_rotation(camera_transform_values.1);
            scene_transform.set_scale(camera_transform_values.2);
        }

        self.attached_camera = Some(camera_entity);

        Ok(())
    }

    /// 解除绑定的摄像机。
    pub fn detach_camera(&mut self) -> Option<Entity> {
        self.attached_camera.take()
    }

    pub(crate) fn sync_camera_transform_internal(&self) -> Result<(), Scene3DAttachError> {
        let Some(camera_entity) = self.attached_camera else {
            return Ok(());
        };

        let scene_entity = self.entity();

        if camera_entity.get_component::<Camera>().is_none() {
            return Err(Scene3DAttachError::MissingCamera(camera_entity));
        }

        let (position, rotation, scale) = {
            let transform = try_get_transform(camera_entity)
                .ok_or(Scene3DAttachError::MissingCameraTransform(camera_entity))?;
            (
                transform.position(),
                transform.rotation(),
                transform.scale(),
            )
        };

        let mut scene_transform = scene_entity
            .get_component_mut::<Transform>()
            .ok_or(Scene3DAttachError::MissingSceneTransform(scene_entity))?;
        scene_transform.set_position(position);
        scene_transform.set_rotation(rotation);
        scene_transform.set_scale(scale);

        Ok(())
    }

    fn sync_camera_transform_for_entity(
        &self,
        camera_entity: Entity,
    ) -> Result<(), Scene3DAttachError> {
        let scene_entity = self.entity();

        if camera_entity.get_component::<Camera>().is_none() {
            return Err(Scene3DAttachError::MissingCamera(camera_entity));
        }

        let (position, rotation, scale) = {
            let transform = camera_entity
                .get_component::<Transform>()
                .ok_or(Scene3DAttachError::MissingCameraTransform(camera_entity))?;
            (
                transform.position(),
                transform.rotation(),
                transform.scale(),
            )
        };

        let mut scene_transform = scene_entity
            .get_component_mut::<Transform>()
            .ok_or(Scene3DAttachError::MissingSceneTransform(scene_entity))?;
        scene_transform.set_position(position);
        scene_transform.set_rotation(rotation);
        scene_transform.set_scale(scale);

        Ok(())
    }

    /// 根据帧缓冲宽高计算水平视场角（弧度）。
    pub fn horizontal_fov(
        &self,
        framebuffer_size: (u32, u32),
    ) -> Result<f32, Scene3DViewportError> {
        let (width, height) = framebuffer_size;
        if width == 0 || height == 0 {
            return Err(Scene3DViewportError::InvalidViewport { width, height });
        }
        let vertical = self.vertical_fov_for_height(height)?;
        let aspect_ratio = width as f32 / height as f32;
        Ok(horizontal_from_vertical(vertical, aspect_ratio))
    }

    pub(crate) fn visible_renderables_internal(
        &self,
        camera_entity: Entity,
        framebuffer_size: (u32, u32),
    ) -> Result<LayerRenderableCollection, Scene3DVisibilityError> {
        self.sync_camera_transform_for_entity(camera_entity)
            .map_err(Scene3DVisibilityError::from)?;

        let scene_entity = self.entity();

        let (width, height) = framebuffer_size;
        if width == 0 || height == 0 {
            return Err(Scene3DVisibilityError::InvalidViewport { width, height });
        }

        let scene_vertical = self
            .vertical_fov_for_height(height)
            .map_err(|error| match error {
                Scene3DViewportError::InvalidViewport { width, height } => {
                    Scene3DVisibilityError::InvalidViewport { width, height }
                }
            })?;
        let scene_near = self.near_plane;
        let scene_distance = self.view_distance;

        let camera_guard = camera_entity
            .get_component::<Camera>()
            .ok_or(Scene3DVisibilityError::MissingCamera(camera_entity))?;
        let camera_vertical = camera_guard
            .vertical_fov_for_height(height)
            .map_err(|error| match error {
                CameraViewportError::InvalidViewport { width, height } => {
                    Scene3DVisibilityError::InvalidViewport { width, height }
                }
            })?;
        let camera_near = camera_guard.near_plane();
        let camera_far = camera_guard.far_plane();
        drop(camera_guard);

        let vertical_fov = scene_vertical.min(camera_vertical);
        if !(MIN_VERTICAL_FOV..MAX_VERTICAL_FOV).contains(&vertical_fov) {
            return Err(Scene3DVisibilityError::VerticalFovOutOfRange(vertical_fov));
        }

        let aspect_ratio = width as f32 / height as f32;
        let horizontal_fov = horizontal_from_vertical(vertical_fov, aspect_ratio);

        let near_plane = camera_near.max(scene_near);
        let far_plane = camera_far.min(scene_distance);
        if near_plane >= far_plane {
            return Err(Scene3DVisibilityError::InvalidClipRange {
                near: near_plane,
                far: far_plane,
            });
        }

        ensure_scene_transform(scene_entity)?;

        let camera_transform = try_get_transform(camera_entity).ok_or(
            Scene3DVisibilityError::MissingCameraTransform(camera_entity),
        )?;
        let camera_position = camera_transform.position();
        let basis = Camera::orientation_basis(&camera_transform).normalize();

        let layer = scene_entity.get_component::<Layer>().ok_or_else(|| {
            Scene3DVisibilityError::LayerTraversal(LayerTraversalError::MissingLayer(scene_entity))
        })?;
        let collection = layer
            .collect_renderables()
            .map_err(Scene3DVisibilityError::LayerTraversal)?;

        let horizontal_limit = 0.5 * horizontal_fov;
        let vertical_limit = 0.5 * vertical_fov;
        let horizontal_tan = (horizontal_limit + FRUSTUM_MARGIN).tan();
        let vertical_tan = (vertical_limit + FRUSTUM_MARGIN).tan();
        let mut visible = Vec::new();

        for bundle in collection.iter() {
            let mut triangles = Vec::new();
            for triangle in bundle.triangles() {
                if triangle_visible(
                    triangle,
                    camera_position,
                    &basis,
                    horizontal_tan,
                    vertical_tan,
                    near_plane,
                    far_plane,
                ) {
                    triangles.push(*triangle);
                }
            }
            if !triangles.is_empty() {
                visible.push(LayerRenderableBundle::new(
                    bundle.entity(),
                    triangles,
                    bundle.material().cloned(),
                ));
            }
        }

        Ok(LayerRenderableCollection::from_bundles(visible))
    }

    /// 推进该 Layer 的 LOD（通常用于体素/分块场景）。
    pub fn step_lod(&self, layer: &mut Layer, targets: &[OctVec], detail: u64) -> LayerLodUpdate {
        let _ = self;
        layer.step_lod(targets, detail)
    }

    pub fn chunk_count(&self, layer: &Layer) -> usize {
        let _ = self;
        layer.chunk_count()
    }

    pub fn chunk_positions(&self, layer: &Layer) -> Vec<OctVec> {
        let _ = self;
        layer.chunk_positions()
    }

    pub fn chunk_neighbors(&self, layer: &Layer, center: OctVec, radius: u64) -> Vec<OctVec> {
        let _ = self;
        layer.chunk_neighbors(center, radius)
    }

    /// 将摄像机绑定到该 3D 场景。
    ///
    /// 绑定后：
    /// - 场景实体的 `Transform` 会被同步为摄像机的 `Transform`（用于渲染时以摄像机为视点）。
    /// - 你可以在每帧渲染前调用 [`sync_camera_transform`](Self::sync_camera_transform) 刷新。
    pub fn bind_camera(&mut self, camera_entity: Entity) -> Result<(), Scene3DAttachError> {
        self.bind_camera_internal(camera_entity)
    }

    /// 将当前绑定摄像机的 `Transform` 同步到场景实体。
    ///
    /// 当你在游戏逻辑里更新了摄像机实体的位置/旋转/缩放后，可在渲染前调用此方法。
    pub fn sync_camera_transform(&self) -> Result<(), Scene3DAttachError> {
        self.sync_camera_transform_internal()
    }
}

impl ComponentWrite<Scene3D> {
    pub fn bind_camera(&mut self, camera_entity: Entity) -> Result<(), Scene3DAttachError> {
        self.bind_camera_internal(camera_entity)
    }
}

impl ComponentRead<Scene3D> {
    pub fn sync_camera_transform(&self) -> Result<(), Scene3DAttachError> {
        self.sync_camera_transform_internal()
    }

    /// 计算当前摄像机视锥内的可见几何集合。
    ///
    /// 返回值是一个便于渲染的集合（按实体聚合三角面，并保留材质信息）。
    /// 常见调用点：每帧渲染时。
    pub fn visible_renderables(
        &self,
        camera_entity: Entity,
        framebuffer_size: (u32, u32),
    ) -> Result<LayerRenderableCollection, Scene3DVisibilityError> {
        self.visible_renderables_internal(camera_entity, framebuffer_size)
    }
}

fn try_get_transform(entity: Entity) -> Option<ComponentRead<Transform>> {
    for _ in 0..3 {
        if let Some(transform) = entity.get_component::<Transform>() {
            return Some(transform);
        }
        std::thread::yield_now();
    }
    None
}

fn ensure_scene_transform(entity: Entity) -> Result<(), Scene3DVisibilityError> {
    if try_get_transform(entity).is_some() {
        Ok(())
    } else {
        Err(Scene3DVisibilityError::MissingSceneTransform(entity))
    }
}

fn horizontal_from_vertical(vertical_fov: f32, aspect_ratio: f32) -> f32 {
    let half_vertical = 0.5 * vertical_fov;
    let horizontal_half = (half_vertical.tan() * aspect_ratio).atan();
    horizontal_half * 2.0
}

fn triangle_visible(
    triangle: &[Vector3<f32>; 3],
    camera_position: Vector3<f32>,
    basis: &CameraBasis,
    horizontal_tan: f32,
    vertical_tan: f32,
    near_plane: f32,
    far_plane: f32,
) -> bool {
    let v0 = world_to_camera_space(triangle[0] - camera_position, basis);
    let v1 = world_to_camera_space(triangle[1] - camera_position, basis);
    let v2 = world_to_camera_space(triangle[2] - camera_position, basis);
    triangle_intersects_frustum(
        [v0, v1, v2],
        horizontal_tan,
        vertical_tan,
        near_plane,
        far_plane,
    )
}

fn world_to_camera_space(offset: Vector3<f32>, basis: &CameraBasis) -> Vector3<f32> {
    Vector3::new(
        offset.dot(&basis.right),
        offset.dot(&basis.up),
        offset.dot(&basis.forward),
    )
}

fn triangle_intersects_frustum(
    vertices: [Vector3<f32>; 3],
    horizontal_tan: f32,
    vertical_tan: f32,
    near_plane: f32,
    far_plane: f32,
) -> bool {
    // 只要三角形没有被任何一个视锥平面“完全排除”，就认为它可能可见。
    // 这样可以避免“所有顶点都在窗口外，但边/面仍穿过窗口”的误剔除。

    // Near/Far：使用摄像机前向轴上的距离（Z），而不是欧氏距离。
    if vertices.iter().all(|v| v.z < near_plane) {
        return false;
    }
    if vertices.iter().all(|v| v.z > far_plane) {
        return false;
    }

    // Left/Right：|x| <= z * tan(fov/2)
    if vertices.iter().all(|v| v.x > v.z * horizontal_tan) {
        return false;
    }
    if vertices.iter().all(|v| v.x < -v.z * horizontal_tan) {
        return false;
    }

    // Top/Bottom：|y| <= z * tan(fov/2)
    if vertices.iter().all(|v| v.y > v.z * vertical_tan) {
        return false;
    }
    if vertices.iter().all(|v| v.y < -v.z * vertical_tan) {
        return false;
    }

    true
}

#[derive(Debug, PartialEq)]
pub enum Scene3DPropertyError {
    VerticalFovOutOfRange(f32),
    InvalidNearPlane(f32),
    InvalidViewDistanceRange { near: f32, distance: f32 },
    InvalidReferenceHeight(u32),
}

impl std::fmt::Display for Scene3DPropertyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scene3DPropertyError::VerticalFovOutOfRange(fov) => {
                write!(f, "垂直视场角超出范围: {}", fov)
            }
            Scene3DPropertyError::InvalidNearPlane(near) => {
                write!(f, "近裁剪面必须为正值，收到 {}", near)
            }
            Scene3DPropertyError::InvalidViewDistanceRange { near, distance } => {
                write!(f, "视距 ({}) 必须大于近裁剪面 ({})", distance, near)
            }
            Scene3DPropertyError::InvalidReferenceHeight(height) => {
                write!(f, "参考帧高度必须大于零，收到 {}", height)
            }
        }
    }
}

impl std::error::Error for Scene3DPropertyError {}

#[derive(Debug, PartialEq)]
pub enum Scene3DViewportError {
    InvalidViewport { width: u32, height: u32 },
}

impl std::fmt::Display for Scene3DViewportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scene3DViewportError::InvalidViewport { width, height } => {
                write!(
                    f,
                    "无效的视口尺寸: 宽度 {}、高度 {} 必须大于零",
                    width, height
                )
            }
        }
    }
}

impl std::error::Error for Scene3DViewportError {}

#[derive(Debug)]
pub enum Scene3DVisibilityError {
    MissingScene(Entity),
    MissingCamera(Entity),
    MissingSceneTransform(Entity),
    MissingCameraTransform(Entity),
    VerticalFovOutOfRange(f32),
    InvalidViewport { width: u32, height: u32 },
    InvalidClipRange { near: f32, far: f32 },
    LayerTraversal(LayerTraversalError),
}

impl std::fmt::Display for Scene3DVisibilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scene3DVisibilityError::MissingScene(entity) => {
                write!(f, "实体 {} 缺少 Scene3D 组件", entity.id())
            }
            Scene3DVisibilityError::MissingCamera(entity) => {
                write!(f, "实体 {} 缺少 Camera 组件", entity.id())
            }
            Scene3DVisibilityError::MissingSceneTransform(entity) => {
                write!(f, "实体 {} 缺少 Transform 组件", entity.id())
            }
            Scene3DVisibilityError::MissingCameraTransform(entity) => {
                write!(f, "实体 {} 缺少摄像机 Transform 组件", entity.id())
            }
            Scene3DVisibilityError::VerticalFovOutOfRange(fov) => {
                write!(f, "有效垂直视场角超出范围: {}", fov)
            }
            Scene3DVisibilityError::InvalidViewport { width, height } => {
                write!(
                    f,
                    "无效的视口尺寸: 宽度 {}、高度 {} 必须大于零",
                    width, height
                )
            }
            Scene3DVisibilityError::InvalidClipRange { near, far } => {
                write!(f, "远裁剪面 ({}) 必须大于近裁剪面 ({})", far, near)
            }
            Scene3DVisibilityError::LayerTraversal(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for Scene3DVisibilityError {}

impl From<LayerTraversalError> for Scene3DVisibilityError {
    fn from(value: LayerTraversalError) -> Self {
        Scene3DVisibilityError::LayerTraversal(value)
    }
}

impl From<Scene3DAttachError> for Scene3DVisibilityError {
    fn from(value: Scene3DAttachError) -> Self {
        match value {
            Scene3DAttachError::MissingScene(entity) => {
                Scene3DVisibilityError::MissingScene(entity)
            }
            Scene3DAttachError::MissingCamera(entity) => {
                Scene3DVisibilityError::MissingCamera(entity)
            }
            Scene3DAttachError::MissingCameraTransform(entity) => {
                Scene3DVisibilityError::MissingCameraTransform(entity)
            }
            Scene3DAttachError::MissingSceneTransform(entity) => {
                Scene3DVisibilityError::MissingSceneTransform(entity)
            }
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Scene3DAttachError {
    MissingScene(Entity),
    MissingCamera(Entity),
    MissingCameraTransform(Entity),
    MissingSceneTransform(Entity),
}

impl std::fmt::Display for Scene3DAttachError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scene3DAttachError::MissingScene(entity) => {
                write!(f, "实体 {} 缺少 Scene3D 组件", entity.id())
            }
            Scene3DAttachError::MissingCamera(entity) => {
                write!(f, "实体 {} 缺少 Camera 组件", entity.id())
            }
            Scene3DAttachError::MissingCameraTransform(entity) => {
                write!(f, "实体 {} 缺少摄像机 Transform 组件", entity.id())
            }
            Scene3DAttachError::MissingSceneTransform(entity) => {
                write!(f, "实体 {} 缺少 Scene3D Transform 组件", entity.id())
            }
        }
    }
}

impl std::error::Error for Scene3DAttachError {}

impl Component for Scene3D {
    fn storage() -> &'static ComponentStorage<Self> {
        SCENE3D_STORAGE.get_or_init(ComponentStorage::new)
    }

    fn register_dependencies(entity: Entity) -> Result<(), ComponentDependencyError> {
        if entity.get_component::<Layer>().is_none() {
            let component = Layer::__jge_component_default(entity)?;
            let _ = entity.register_component(component)?;
        }
        if entity.get_component::<Renderable>().is_none() {
            let component = Renderable::__jge_component_default(entity)?;
            let _ = entity.register_component(component)?;
        }
        if entity.get_component::<Transform>().is_none() {
            let component = Transform::__jge_component_default(entity)?;
            let _ = entity.register_component(component)?;
        }
        Ok(())
    }

    fn attach_entity(&mut self, entity: Entity) {
        self.entity_id = Some(entity);
        Self::ensure_default_layer_shaders(entity);
    }

    fn detach_entity(&mut self) {
        self.entity_id = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::component::{
        camera::Camera,
        layer::{Layer, RenderPipelineStage, ShaderLanguage},
        node::Node,
        renderable::Renderable,
        shape::Shape,
        transform::Transform,
    };
    use crate::game::entity::Entity;
    use lodtree::{coords::OctVec, traits::LodVec};

    fn detach_node(entity: Entity) {
        if let Some(mut node) = entity.get_component_mut::<Node>() {
            let _ = node.detach();
        }
    }

    fn attach_node(entity: Entity, parent: Entity, message: &str) {
        let mut parent_node = parent
            .get_component_mut::<Node>()
            .expect("父实体应持有 Node 组件");
        parent_node.attach(entity).expect(message);
    }

    fn prepare_scene_entity(entity: &Entity, name: &str) {
        let _ = entity.unregister_component::<Scene3D>();
        let _ = entity.unregister_component::<Layer>();
        let _ = entity.unregister_component::<Renderable>();
        let _ = entity.unregister_component::<Transform>();
        detach_node(*entity);
        let _ = entity.unregister_component::<Node>();

        let _ = entity
            .register_component(Node::new(name).expect("应能创建节点"))
            .expect("应能插入 Node");
        let _ = entity
            .register_component(Renderable::new())
            .expect("应能插入 Renderable");
        let _ = entity
            .register_component(Transform::new())
            .expect("应能插入 Transform");
        let _ = entity
            .register_component(Layer::new())
            .expect("应能插入 Layer");
    }

    fn prepare_child(entity: &Entity, name: &str, parent: Entity) {
        prepare_scene_entity(entity, name);
        let _ = entity.unregister_component::<Layer>();
        attach_node(*entity, parent, "应能挂载到父节点");
    }

    fn prepare_camera_entity(entity: &Entity, name: &str) {
        let _ = entity.unregister_component::<Camera>();
        let _ = entity.unregister_component::<Transform>();
        let _ = entity.unregister_component::<Renderable>();
        detach_node(*entity);
        let _ = entity.unregister_component::<Node>();

        let _ = entity
            .register_component(Node::new(name).expect("应能创建节点"))
            .expect("应能插入 Node");
        let _ = entity
            .register_component(Renderable::new())
            .expect("应能插入 Renderable");
        let _ = entity
            .register_component(Transform::new())
            .expect("应能插入 Transform");
    }

    #[test]
    fn scene3d_requires_layer_dependency() {
        let entity = Entity::new().expect("应能创建实体");
        let _ = entity.unregister_component::<Scene3D>();
        let _ = entity.unregister_component::<Layer>();
        let _ = entity.unregister_component::<Renderable>();
        let _ = entity.unregister_component::<Transform>();
        detach_node(entity);
        let _ = entity.unregister_component::<Node>();

        let inserted = entity
            .register_component(Scene3D::new())
            .expect("缺少 Layer 时应自动注册依赖");
        assert!(inserted.is_none());

        assert!(
            entity.get_component::<Layer>().is_some(),
            "Layer 应被自动注册"
        );
        assert!(
            entity.get_component::<Renderable>().is_some(),
            "Renderable 应被注册"
        );
        assert!(entity.get_component::<Node>().is_some(), "Node 应被注册");
        let layer = entity.get_component::<Layer>().expect("应能读取 Layer");
        assert!(
            layer.shader(RenderPipelineStage::Vertex).is_some(),
            "默认应挂载顶点着色器"
        );
        assert!(
            layer.shader(RenderPipelineStage::Fragment).is_some(),
            "默认应挂载片段着色器"
        );
        assert_eq!(
            layer
                .shader(RenderPipelineStage::Vertex)
                .unwrap()
                .language(),
            ShaderLanguage::Wgsl
        );
        assert_eq!(
            layer
                .shader(RenderPipelineStage::Fragment)
                .unwrap()
                .language(),
            ShaderLanguage::Wgsl
        );
        drop(layer);

        let _ = entity.unregister_component::<Scene3D>();
        let _ = entity.unregister_component::<Layer>();
        let _ = entity.unregister_component::<Renderable>();
        let _ = entity.unregister_component::<Transform>();
        detach_node(entity);
        let _ = entity.unregister_component::<Node>();
    }

    #[test]
    fn scene3d_view_property_constraints() {
        let entity = Entity::new().expect("应能创建实体");
        prepare_scene_entity(&entity, "scene3d_props");
        entity
            .register_component(Scene3D::new())
            .expect("应能插入 Scene3D");

        {
            let mut scene = entity
                .get_component_mut::<Scene3D>()
                .expect("应能写入 Scene3D");
            assert!(scene.set_vertical_fov(MIN_VERTICAL_FOV / 2.0).is_err());
            assert!(scene.set_vertical_fov(DEFAULT_VERTICAL_FOV * 1.1).is_ok());
            assert!(scene.set_reference_framebuffer_height(0).is_err());
            assert!(scene.set_reference_framebuffer_height(1440).is_ok());
            assert!(scene.set_near_plane(0.0).is_err());
            let near = scene.near_plane();
            assert!(scene.set_view_distance(near * 0.5).is_err());
            assert!(scene.set_view_distance(900.0).is_ok());
            assert!(scene.set_near_plane(0.5).is_ok());
        }

        let _ = entity.unregister_component::<Scene3D>();
        let _ = entity.unregister_component::<Layer>();
        let _ = entity.unregister_component::<Renderable>();
        let _ = entity.unregister_component::<Transform>();
        detach_node(entity);
        let _ = entity.unregister_component::<Node>();
    }

    #[test]
    fn scene3d_vertical_fov_depends_on_aspect() {
        let entity = Entity::new().expect("应能创建实体");
        prepare_scene_entity(&entity, "scene3d_fov");
        entity
            .register_component(Scene3D::new())
            .expect("应能插入 Scene3D");

        let scene = entity.get_component::<Scene3D>().expect("应能读取 Scene3D");
        let tall = scene
            .horizontal_fov((1080, 1920))
            .expect("有效视口不应报错");
        let wide = scene
            .horizontal_fov((1920, 1080))
            .expect("有效视口不应报错");
        drop(scene);
        assert!(wide > tall);
        {
            let scene_guard = entity.get_component::<Scene3D>().expect("应能读取 Scene3D");
            assert!(scene_guard.visible_renderables(entity, (0, 1080)).is_err());
        }

        let _ = entity.unregister_component::<Scene3D>();
        let _ = entity.unregister_component::<Layer>();
        let _ = entity.unregister_component::<Renderable>();
        let _ = entity.unregister_component::<Transform>();
        detach_node(entity);
        let _ = entity.unregister_component::<Node>();
    }

    #[test]
    fn scene3d_visible_renderables_culls_geometry() {
        let scene = Entity::new().expect("应能创建实体");
        prepare_scene_entity(&scene, "scene3d_cull_root");
        scene
            .register_component(Scene3D::new())
            .expect("应能插入 Scene3D");

        let inside = Entity::new().expect("应能创建实体");
        prepare_child(&inside, "inside", scene);
        inside
            .register_component(Shape::from_triangles(vec![[
                // Camera 默认朝向为 -Z，因此 z<0 在视野前方。
                Vector3::new(-1.0, -1.0, -5.0),
                Vector3::new(1.0, -1.0, -5.0),
                Vector3::new(0.0, 1.0, -5.0),
            ]]))
            .expect("应能插入 Shape");

        let outside = Entity::new().expect("应能创建实体");
        prepare_child(&outside, "outside", scene);
        outside
            .register_component(Shape::from_triangles(vec![[
                // z>0 在相机后方，应被剔除。
                Vector3::new(-1.0, -1.0, 5.0),
                Vector3::new(1.0, -1.0, 5.0),
                Vector3::new(0.0, 1.0, 5.0),
            ]]))
            .expect("应能插入 Shape");

        let scene_component = scene.get_component::<Scene3D>().expect("应能读取 Scene3D");
        let vertical_base = scene_component
            .vertical_fov_for_height(1080)
            .expect("有效视口不应报错");
        let vertical_double = scene_component
            .vertical_fov_for_height(2160)
            .expect("有效视口不应报错");
        let base_ratio = (0.5 * vertical_base).tan() / 1080.0;
        let double_ratio = (0.5 * vertical_double).tan() / 2160.0;
        assert!((base_ratio - double_ratio).abs() < 1e-6);

        let horizontal_wide = scene_component
            .horizontal_fov((1920, 1080))
            .expect("有效视口不应报错");
        let horizontal_tall = scene_component
            .horizontal_fov((1080, 1920))
            .expect("有效视口不应报错");
        assert!(horizontal_wide > horizontal_tall);
        drop(scene_component);

        let camera = Entity::new().expect("应能创建实体");
        prepare_scene_entity(&camera, "camera_node");
        camera
            .register_component(Camera::new())
            .expect("应能插入 Camera");

        let visible = {
            let scene_guard = scene.get_component::<Scene3D>().expect("应能读取 Scene3D");
            scene_guard
                .visible_renderables(camera, (1280, 720))
                .expect("可见性计算不应失败")
        };
        let bundles = visible.into_bundles();
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].entity(), inside);

        let _ = camera.unregister_component::<Camera>();
        let _ = camera.unregister_component::<Scene3D>();
        let _ = camera.unregister_component::<Layer>();
        let _ = camera.unregister_component::<Renderable>();
        let _ = camera.unregister_component::<Transform>();
        detach_node(camera);
        let _ = camera.unregister_component::<Node>();

        let _ = inside.unregister_component::<Shape>();
        let _ = inside.unregister_component::<Scene3D>();
        let _ = inside.unregister_component::<Layer>();
        let _ = inside.unregister_component::<Renderable>();
        let _ = inside.unregister_component::<Transform>();
        detach_node(inside);
        let _ = inside.unregister_component::<Node>();

        let _ = outside.unregister_component::<Shape>();
        let _ = outside.unregister_component::<Scene3D>();
        let _ = outside.unregister_component::<Layer>();
        let _ = outside.unregister_component::<Renderable>();
        let _ = outside.unregister_component::<Transform>();
        detach_node(outside);
        let _ = outside.unregister_component::<Node>();

        let _ = scene.unregister_component::<Scene3D>();
        let _ = scene.unregister_component::<Layer>();
        let _ = scene.unregister_component::<Renderable>();
        let _ = scene.unregister_component::<Transform>();
        detach_node(scene);
        let _ = scene.unregister_component::<Node>();
    }

    #[test]
    fn scene3d_accesses_layer_spatial_index() {
        let entity = Entity::new().expect("应能创建实体");
        prepare_scene_entity(&entity, "scene3d_spatial");
        entity
            .register_component(Scene3D::new())
            .expect("依赖满足后应能插入 Scene3D");
        let scene = entity
            .get_component::<Scene3D>()
            .expect("Scene3D 组件应已注册");

        {
            let layer = entity.get_component::<Layer>().expect("Layer 组件应已注册");
            assert_eq!(scene.chunk_count(&layer), 0);
        }

        let target = OctVec::root();
        let detail = 1;
        {
            let mut layer = entity
                .get_component_mut::<Layer>()
                .expect("Layer 组件应已注册");
            let update = scene.step_lod(&mut layer, &[target], detail);
            assert_eq!(update.added.len(), 1);
            assert_eq!(update.removed.len(), 0);
        }

        {
            let layer = entity.get_component::<Layer>().expect("Layer 组件应已注册");
            let positions = scene.chunk_positions(&layer);
            assert_eq!(positions.len(), 1);
            assert!(positions.contains(&target));

            let center = target;
            let neighbors = scene.chunk_neighbors(&layer, center, 1);
            assert!(neighbors.contains(&center));
        }

        drop(scene);

        let _ = entity.unregister_component::<Scene3D>();
        let _ = entity.unregister_component::<Layer>();
        let _ = entity.unregister_component::<Renderable>();
        let _ = entity.unregister_component::<Transform>();
        detach_node(entity);
        let _ = entity.unregister_component::<Node>();
    }

    #[test]
    fn scene3d_attaches_and_syncs_camera_transform() {
        let scene = Entity::new().expect("应能创建实体");
        prepare_scene_entity(&scene, "scene3d_attach_root");
        scene
            .register_component(Scene3D::new())
            .expect("应能插入 Scene3D");

        let camera = Entity::new().expect("应能创建实体");
        prepare_camera_entity(&camera, "camera_attach");
        attach_node(camera, scene, "应能将摄像机挂载到场景");
        let _ = camera
            .register_component(Camera::new())
            .expect("应能插入 Camera");

        if let Some(mut transform) = camera.get_component_mut::<Transform>() {
            transform.set_position(Vector3::new(3.0, 2.0, -5.0));
            transform.set_rotation(Vector3::new(0.1, 0.2, 0.0));
            transform.set_scale(Vector3::new(1.0, 1.0, 1.0));
        }

        {
            let mut scene_component = scene
                .get_component_mut::<Scene3D>()
                .expect("应能读取 Scene3D");
            scene_component.bind_camera(camera).expect("应能绑定摄像机");
        }
        let stored = scene.get_component::<Scene3D>().expect("应能读取 Scene3D");
        assert_eq!(stored.attached_camera(), Some(camera));
        drop(stored);

        {
            let transform = scene
                .get_component::<Transform>()
                .expect("场景应持有 Transform");
            assert_eq!(transform.position(), Vector3::new(3.0, 2.0, -5.0));
            assert_eq!(transform.rotation(), Vector3::new(0.1, 0.2, 0.0));
        }

        if let Some(mut transform) = camera.get_component_mut::<Transform>() {
            transform.set_position(Vector3::new(-2.0, 1.5, -3.0));
            transform.set_rotation(Vector3::new(-0.1, 0.3, 0.2));
        }

        {
            let scene_component = scene.get_component::<Scene3D>().expect("应能读取 Scene3D");
            scene_component.sync_camera_transform().expect("同步应成功");
        }

        {
            let transform = scene
                .get_component::<Transform>()
                .expect("场景应持有 Transform");
            assert_eq!(transform.position(), Vector3::new(-2.0, 1.5, -3.0));
            assert_eq!(transform.rotation(), Vector3::new(-0.1, 0.3, 0.2));
        }

        {
            let mut scene_component = scene
                .get_component_mut::<Scene3D>()
                .expect("应能读取 Scene3D");
            let _ = scene_component.detach_camera();
        }
        let _ = camera.unregister_component::<Camera>();
        let _ = camera.unregister_component::<Transform>();
        let _ = camera.unregister_component::<Renderable>();
        detach_node(camera);
        let _ = camera.unregister_component::<Node>();

        let _ = scene.unregister_component::<Scene3D>();
        let _ = scene.unregister_component::<Layer>();
        let _ = scene.unregister_component::<Renderable>();
        let _ = scene.unregister_component::<Transform>();
        detach_node(scene);
        let _ = scene.unregister_component::<Node>();
    }

    #[test]
    fn scene3d_detach_camera_clears_attachment() {
        let scene = Entity::new().expect("应能创建实体");
        prepare_scene_entity(&scene, "scene3d_detach_root");
        scene
            .register_component(Scene3D::new())
            .expect("应能插入 Scene3D");

        let camera = Entity::new().expect("应能创建实体");
        prepare_camera_entity(&camera, "camera_detach");
        attach_node(camera, scene, "应能将摄像机挂载到场景");
        let _ = camera
            .register_component(Camera::new())
            .expect("应能插入 Camera");

        {
            let mut scene_component = scene
                .get_component_mut::<Scene3D>()
                .expect("应能读取 Scene3D");
            scene_component.bind_camera(camera).expect("应能绑定摄像机");
        }
        let detached = {
            let mut scene_component = scene
                .get_component_mut::<Scene3D>()
                .expect("应能读取 Scene3D");
            scene_component.detach_camera()
        };
        assert_eq!(detached, Some(camera));
        let stored = scene.get_component::<Scene3D>().expect("应能读取 Scene3D");
        assert_eq!(stored.attached_camera(), None);
        drop(stored);

        let _ = camera.unregister_component::<Camera>();
        let _ = camera.unregister_component::<Transform>();
        let _ = camera.unregister_component::<Renderable>();
        detach_node(camera);
        let _ = camera.unregister_component::<Node>();

        let _ = scene.unregister_component::<Scene3D>();
        let _ = scene.unregister_component::<Layer>();
        let _ = scene.unregister_component::<Renderable>();
        let _ = scene.unregister_component::<Transform>();
        detach_node(scene);
        let _ = scene.unregister_component::<Node>();
    }
}
