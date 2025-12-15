use super::{
    camera::{Camera, CameraBasis, CameraViewportError},
    component, component_impl,
    layer::{
        Layer, LayerLodUpdate, LayerRenderableBundle, LayerRenderableCollection, LayerSpatialError,
        LayerTraversalError, RenderPipelineStage, ShaderLanguage,
    },
    renderable::Renderable,
    transform::Transform,
};
use crate::game::{
    component::{Component, ComponentDependencyError},
    entity::Entity,
};
use crate::resource::ResourcePath;
use lodtree::coords::OctVec;
use nalgebra::Vector3;
use std::f32::consts::PI;

const DEFAULT_VERTICAL_FOV: f32 = 60.0_f32.to_radians();
const DEFAULT_NEAR_PLANE: f32 = 0.1;
const DEFAULT_VIEW_DISTANCE: f32 = 1024.0;
const MIN_VERTICAL_FOV: f32 = PI / 180.0;
const MAX_VERTICAL_FOV: f32 = PI - MIN_VERTICAL_FOV;
const FRUSTUM_MARGIN: f32 = 1.0_f32.to_radians();
const DEFAULT_REFERENCE_FRAMEBUFFER_HEIGHT: u32 = 1080;

#[component(Layer, Renderable, Transform)]
#[derive(Debug)]
pub struct Scene3D {
    name: String,
    vertical_fov: f32,
    near_plane: f32,
    view_distance: f32,
    attached_camera: Option<Entity>,
    reference_framebuffer_height: u32,
}

impl Clone for Scene3D {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            vertical_fov: self.vertical_fov,
            near_plane: self.near_plane,
            view_distance: self.view_distance,
            attached_camera: self.attached_camera,
            reference_framebuffer_height: self.reference_framebuffer_height,
        }
    }
}

#[component_impl]
impl Scene3D {
    #[allow(dead_code)]
    #[default(Layer::new(), Renderable::new(), Transform::new())]
    fn ensure_defaults(_layer: Layer, _renderable: Renderable, _transform: Transform) -> Self {
        Self::new("auto_scene3d")
    }
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            vertical_fov: DEFAULT_VERTICAL_FOV,
            near_plane: DEFAULT_NEAR_PLANE,
            view_distance: DEFAULT_VIEW_DISTANCE,
            attached_camera: None,
            reference_framebuffer_height: DEFAULT_REFERENCE_FRAMEBUFFER_HEIGHT,
        }
    }

    pub fn insert(
        entity: Entity,
        scene: Scene3D,
    ) -> Result<Option<Scene3D>, ComponentDependencyError> {
        let previous = entity.register_component(scene)?;
        Self::ensure_default_layer_shaders(entity);
        Ok(previous)
    }

    fn ensure_default_layer_shaders(entity: Entity) {
        if let Some(mut layer) = Layer::write(entity) {
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

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn vertical_fov(&self) -> f32 {
        self.vertical_fov
    }

    pub fn set_vertical_fov(&mut self, fov: f32) -> Result<(), Scene3DPropertyError> {
        if !(MIN_VERTICAL_FOV..MAX_VERTICAL_FOV).contains(&fov) {
            return Err(Scene3DPropertyError::VerticalFovOutOfRange(fov));
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
    ) -> Result<(), Scene3DPropertyError> {
        if height == 0 {
            return Err(Scene3DPropertyError::InvalidReferenceHeight(height));
        }
        self.reference_framebuffer_height = height;
        Ok(())
    }

    pub fn near_plane(&self) -> f32 {
        self.near_plane
    }

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

    pub fn attached_camera(&self) -> Option<Entity> {
        self.attached_camera
    }

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

    pub fn view_distance(&self) -> f32 {
        self.view_distance
    }

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

    pub fn attach_camera(
        scene_entity: Entity,
        camera_entity: Entity,
    ) -> Result<(), Scene3DAttachError> {
        if Scene3D::read(scene_entity).is_none() {
            return Err(Scene3DAttachError::MissingScene(scene_entity));
        }

        if Camera::read(camera_entity).is_none() {
            return Err(Scene3DAttachError::MissingCamera(camera_entity));
        }

        let camera_transform_values = {
            let transform = Transform::read(camera_entity)
                .ok_or(Scene3DAttachError::MissingCameraTransform(camera_entity))?;
            (
                transform.position(),
                transform.rotation(),
                transform.scale(),
            )
        };

        {
            let mut scene_transform = Transform::write(scene_entity)
                .ok_or(Scene3DAttachError::MissingSceneTransform(scene_entity))?;
            scene_transform.set_position(camera_transform_values.0);
            scene_transform.set_rotation(camera_transform_values.1);
            scene_transform.set_scale(camera_transform_values.2);
        }

        if let Some(mut scene_guard) = Scene3D::write(scene_entity) {
            scene_guard.attached_camera = Some(camera_entity);
        } else {
            return Err(Scene3DAttachError::MissingScene(scene_entity));
        }

        Ok(())
    }

    pub fn detach_camera(scene_entity: Entity) -> Result<Option<Entity>, Scene3DAttachError> {
        let mut scene_guard =
            Scene3D::write(scene_entity).ok_or(Scene3DAttachError::MissingScene(scene_entity))?;
        Ok(scene_guard.attached_camera.take())
    }

    pub fn sync_attached_transform(scene_entity: Entity) -> Result<(), Scene3DAttachError> {
        let camera_entity = {
            let scene_guard = Scene3D::read(scene_entity)
                .ok_or(Scene3DAttachError::MissingScene(scene_entity))?;
            scene_guard.attached_camera
        };

        let Some(camera_entity) = camera_entity else {
            return Ok(());
        };

        if Camera::read(camera_entity).is_none() {
            return Err(Scene3DAttachError::MissingCamera(camera_entity));
        }

        let (position, rotation, scale) = {
            let transform = Transform::read(camera_entity)
                .ok_or(Scene3DAttachError::MissingCameraTransform(camera_entity))?;
            (
                transform.position(),
                transform.rotation(),
                transform.scale(),
            )
        };

        let mut scene_transform = Transform::write(scene_entity)
            .ok_or(Scene3DAttachError::MissingSceneTransform(scene_entity))?;
        scene_transform.set_position(position);
        scene_transform.set_rotation(rotation);
        scene_transform.set_scale(scale);

        Ok(())
    }

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

    pub fn visible_renderables(
        scene_entity: Entity,
        camera_entity: Entity,
        framebuffer_size: (u32, u32),
    ) -> Result<LayerRenderableCollection, Scene3DVisibilityError> {
        Scene3D::sync_attached_transform(scene_entity).map_err(Scene3DVisibilityError::from)?;

        let (width, height) = framebuffer_size;
        if width == 0 || height == 0 {
            return Err(Scene3DVisibilityError::InvalidViewport { width, height });
        }

        let scene_guard = scene_entity
            .get_component::<Scene3D>()
            .ok_or(Scene3DVisibilityError::MissingScene(scene_entity))?;
        let scene_vertical = scene_guard
            .vertical_fov_for_height(height)
            .map_err(|error| match error {
                Scene3DViewportError::InvalidViewport { width, height } => {
                    Scene3DVisibilityError::InvalidViewport { width, height }
                }
            })?;
        let scene_near = scene_guard.near_plane;
        let scene_distance = scene_guard.view_distance;
        drop(scene_guard);

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

        let camera_transform = camera_entity.get_component::<Transform>().ok_or(
            Scene3DVisibilityError::MissingCameraTransform(camera_entity),
        )?;
        let camera_position = camera_transform.position();
        let basis = Camera::orientation_basis(&camera_transform).normalize();
        drop(camera_transform);

        let collection = Layer::collect_renderables(scene_entity)
            .map_err(Scene3DVisibilityError::LayerTraversal)?;

        let horizontal_limit = 0.5 * horizontal_fov;
        let vertical_limit = 0.5 * vertical_fov;
        let mut visible = Vec::new();

        for bundle in collection.iter() {
            let mut triangles = Vec::new();
            for triangle in bundle.triangles() {
                if triangle_visible(
                    triangle,
                    camera_position,
                    &basis,
                    horizontal_limit,
                    vertical_limit,
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

    pub fn step_lod(
        entity: Entity,
        targets: &[OctVec],
        detail: u64,
    ) -> Result<LayerLodUpdate, LayerSpatialError> {
        let mut layer = Layer::write(entity).ok_or(LayerSpatialError::MissingLayer(entity))?;
        Ok(layer.step_lod(targets, detail))
    }

    pub fn chunk_count(entity: Entity) -> Result<usize, LayerSpatialError> {
        let layer = Layer::read(entity).ok_or(LayerSpatialError::MissingLayer(entity))?;
        Ok(layer.chunk_count())
    }

    pub fn chunk_positions(entity: Entity) -> Result<Vec<OctVec>, LayerSpatialError> {
        let layer = Layer::read(entity).ok_or(LayerSpatialError::MissingLayer(entity))?;
        Ok(layer.chunk_positions())
    }

    pub fn chunk_neighbors(
        entity: Entity,
        center: OctVec,
        radius: u64,
    ) -> Result<Vec<OctVec>, LayerSpatialError> {
        let layer = Layer::read(entity).ok_or(LayerSpatialError::MissingLayer(entity))?;
        Ok(layer.chunk_neighbors(center, radius))
    }
}

fn ensure_scene_transform(entity: Entity) -> Result<(), Scene3DVisibilityError> {
    if entity.get_component::<Transform>().is_some() {
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
    horizontal_limit: f32,
    vertical_limit: f32,
    near_plane: f32,
    far_plane: f32,
) -> bool {
    for vertex in triangle.iter() {
        if vertex_visible(
            *vertex,
            camera_position,
            basis,
            horizontal_limit,
            vertical_limit,
            near_plane,
            far_plane,
        ) {
            return true;
        }
    }

    let centroid = (triangle[0] + triangle[1] + triangle[2]) / 3.0;
    vertex_visible(
        centroid,
        camera_position,
        basis,
        horizontal_limit,
        vertical_limit,
        near_plane,
        far_plane,
    )
}

fn vertex_visible(
    vertex: Vector3<f32>,
    camera_position: Vector3<f32>,
    basis: &CameraBasis,
    horizontal_limit: f32,
    vertical_limit: f32,
    near_plane: f32,
    far_plane: f32,
) -> bool {
    let offset = vertex - camera_position;
    let distance = offset.norm();
    if distance < near_plane || distance > far_plane {
        return false;
    }

    let direction = offset / distance;
    let forward_component = direction.dot(&basis.forward);
    if forward_component <= 0.0 {
        return false;
    }

    let horizontal_component = direction.dot(&basis.right).abs();
    let vertical_component = direction.dot(&basis.up).abs();

    let horizontal_angle = horizontal_component.atan2(forward_component);
    if horizontal_angle > horizontal_limit + FRUSTUM_MARGIN {
        return false;
    }

    let vertical_angle = vertical_component.atan2(forward_component);
    vertical_angle <= vertical_limit + FRUSTUM_MARGIN
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::component::{
        camera::Camera,
        layer::{Layer, LayerSpatialError, RenderPipelineStage, ShaderLanguage},
        node::Node,
        renderable::Renderable,
        shape::Shape,
        transform::Transform,
    };
    use crate::game::entity::Entity;
    use lodtree::{coords::OctVec, traits::LodVec};

    fn prepare_scene_entity(entity: &Entity, name: &str) {
        let _ = entity.unregister_component::<Scene3D>();
        let _ = entity.unregister_component::<Layer>();
        let _ = entity.unregister_component::<Renderable>();
        let _ = entity.unregister_component::<Transform>();
        let _ = Node::detach(*entity);
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
        Node::attach(*entity, parent).expect("应能挂载到父节点");
    }

    fn prepare_camera_entity(entity: &Entity, name: &str) {
        let _ = entity.unregister_component::<Camera>();
        let _ = entity.unregister_component::<Transform>();
        let _ = entity.unregister_component::<Renderable>();
        let _ = Node::detach(*entity);
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
        let _ = Node::detach(entity);
        let _ = entity.unregister_component::<Node>();

        let missing = Scene3D::insert(entity, Scene3D::new("scene3d"));
        assert!(matches!(
            missing,
            Err(crate::game::component::ComponentDependencyError { .. })
        ));

        prepare_scene_entity(&entity, "scene3d_root");
        let inserted =
            Scene3D::insert(entity, Scene3D::new("scene3d")).expect("依赖满足后应能插入");
        assert!(inserted.is_none());

        let stored = entity.get_component::<Scene3D>().expect("应能读取 Scene3D");
        assert_eq!(stored.name(), "scene3d");

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
        drop(layer);
        drop(stored);

        let _ = entity.unregister_component::<Scene3D>();
        let _ = entity.unregister_component::<Layer>();
        let _ = entity.unregister_component::<Renderable>();
        let _ = entity.unregister_component::<Transform>();
        let _ = Node::detach(entity);
        let _ = entity.unregister_component::<Node>();
    }

    #[test]
    fn scene3d_view_property_constraints() {
        let entity = Entity::new().expect("应能创建实体");
        prepare_scene_entity(&entity, "scene3d_props");
        Scene3D::insert(entity, Scene3D::new("scene_props")).expect("应能插入 Scene3D");

        {
            let mut scene = Scene3D::write(entity).expect("应能写入 Scene3D");
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
        let _ = Node::detach(entity);
        let _ = entity.unregister_component::<Node>();
    }

    #[test]
    fn scene3d_vertical_fov_depends_on_aspect() {
        let entity = Entity::new().expect("应能创建实体");
        prepare_scene_entity(&entity, "scene3d_fov");
        Scene3D::insert(entity, Scene3D::new("scene_fov")).expect("应能插入 Scene3D");

        let scene = entity.get_component::<Scene3D>().expect("应能读取 Scene3D");
        let tall = scene
            .horizontal_fov((1080, 1920))
            .expect("有效视口不应报错");
        let wide = scene
            .horizontal_fov((1920, 1080))
            .expect("有效视口不应报错");
        drop(scene);
        assert!(wide > tall);
        assert!(Scene3D::visible_renderables(entity, entity, (0, 1080)).is_err());

        let _ = entity.unregister_component::<Scene3D>();
        let _ = entity.unregister_component::<Layer>();
        let _ = entity.unregister_component::<Renderable>();
        let _ = entity.unregister_component::<Transform>();
        let _ = Node::detach(entity);
        let _ = entity.unregister_component::<Node>();
    }

    #[test]
    fn scene3d_visible_renderables_culls_geometry() {
        let scene = Entity::new().expect("应能创建实体");
        prepare_scene_entity(&scene, "scene3d_cull_root");
        Scene3D::insert(scene, Scene3D::new("scene3d_cull")).expect("应能插入 Scene3D");

        let inside = Entity::new().expect("应能创建实体");
        prepare_child(&inside, "inside", scene);
        inside
            .register_component(Shape::from_triangles(vec![[
                Vector3::new(-1.0, -5.0, 0.0),
                Vector3::new(1.0, -5.0, 0.0),
                Vector3::new(0.0, -5.0, 2.0),
            ]]))
            .expect("应能插入 Shape");

        let outside = Entity::new().expect("应能创建实体");
        prepare_child(&outside, "outside", scene);
        outside
            .register_component(Shape::from_triangles(vec![[
                Vector3::new(-1.0, 5.0, 0.0),
                Vector3::new(1.0, 5.0, 0.0),
                Vector3::new(0.0, 5.0, 2.0),
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
            .register_component(Camera::new("player"))
            .expect("应能插入 Camera");

        let visible =
            Scene3D::visible_renderables(scene, camera, (1280, 720)).expect("可见性计算不应失败");
        let bundles = visible.into_bundles();
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].entity(), inside);

        let _ = camera.unregister_component::<Camera>();
        let _ = camera.unregister_component::<Scene3D>();
        let _ = camera.unregister_component::<Layer>();
        let _ = camera.unregister_component::<Renderable>();
        let _ = camera.unregister_component::<Transform>();
        let _ = Node::detach(camera);
        let _ = camera.unregister_component::<Node>();

        let _ = inside.unregister_component::<Shape>();
        let _ = inside.unregister_component::<Scene3D>();
        let _ = inside.unregister_component::<Layer>();
        let _ = inside.unregister_component::<Renderable>();
        let _ = inside.unregister_component::<Transform>();
        let _ = Node::detach(inside);
        let _ = inside.unregister_component::<Node>();

        let _ = outside.unregister_component::<Shape>();
        let _ = outside.unregister_component::<Scene3D>();
        let _ = outside.unregister_component::<Layer>();
        let _ = outside.unregister_component::<Renderable>();
        let _ = outside.unregister_component::<Transform>();
        let _ = Node::detach(outside);
        let _ = outside.unregister_component::<Node>();

        let _ = scene.unregister_component::<Scene3D>();
        let _ = scene.unregister_component::<Layer>();
        let _ = scene.unregister_component::<Renderable>();
        let _ = scene.unregister_component::<Transform>();
        let _ = Node::detach(scene);
        let _ = scene.unregister_component::<Node>();
    }

    #[test]
    fn scene3d_accesses_layer_spatial_index() {
        let entity = Entity::new().expect("应能创建实体");
        let _ = entity.unregister_component::<Scene3D>();
        let _ = entity.unregister_component::<Layer>();
        let _ = entity.unregister_component::<Renderable>();
        let _ = entity.unregister_component::<Transform>();
        let _ = Node::detach(entity);
        let _ = entity.unregister_component::<Node>();

        let missing_layer = Scene3D::chunk_count(entity);
        assert!(matches!(
            missing_layer,
            Err(LayerSpatialError::MissingLayer(_))
        ));

        prepare_scene_entity(&entity, "scene3d_spatial");
        Scene3D::insert(entity, Scene3D::new("scene3d_spatial"))
            .expect("依赖满足后应能插入 Scene3D");
        assert_eq!(Scene3D::chunk_count(entity).unwrap(), 0);

        let target = OctVec::root();
        let detail = 1;
        {
            let update = Scene3D::step_lod(entity, &[target], detail).expect("应能推进八叉树 LOD");
            assert_eq!(update.added.len(), 1);
            assert_eq!(update.removed.len(), 0);
        }

        let positions = Scene3D::chunk_positions(entity).expect("应能获取节点集合");
        assert_eq!(positions.len(), 1);
        assert!(positions.contains(&target));

        let center = target;
        let neighbors = Scene3D::chunk_neighbors(entity, center, 1).expect("应能查询邻居");
        assert!(neighbors.contains(&center));

        let stray = Entity::new().expect("应能创建实体");
        let missing_layer = Scene3D::chunk_neighbors(stray, center, 1);
        assert!(matches!(
            missing_layer,
            Err(LayerSpatialError::MissingLayer(_))
        ));

        let _ = entity.unregister_component::<Scene3D>();
        let _ = entity.unregister_component::<Layer>();
        let _ = entity.unregister_component::<Renderable>();
        let _ = entity.unregister_component::<Transform>();
        let _ = Node::detach(entity);
        let _ = entity.unregister_component::<Node>();
    }

    #[test]
    fn scene3d_attaches_and_syncs_camera_transform() {
        let scene = Entity::new().expect("应能创建实体");
        prepare_scene_entity(&scene, "scene3d_attach_root");
        Scene3D::insert(scene, Scene3D::new("scene3d_attach")).expect("应能插入 Scene3D");

        let camera = Entity::new().expect("应能创建实体");
        prepare_camera_entity(&camera, "camera_attach");
        Node::attach(camera, scene).expect("应能将摄像机挂载到场景");
        let _ = camera
            .register_component(Camera::new("main_camera"))
            .expect("应能插入 Camera");

        if let Some(mut transform) = camera.get_component_mut::<Transform>() {
            transform.set_position(Vector3::new(3.0, 2.0, -5.0));
            transform.set_rotation(Vector3::new(0.1, 0.2, 0.0));
            transform.set_scale(Vector3::new(1.0, 1.0, 1.0));
        }

        Scene3D::attach_camera(scene, camera).expect("应能绑定摄像机");
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

        Scene3D::sync_attached_transform(scene).expect("同步应成功");

        {
            let transform = scene
                .get_component::<Transform>()
                .expect("场景应持有 Transform");
            assert_eq!(transform.position(), Vector3::new(-2.0, 1.5, -3.0));
            assert_eq!(transform.rotation(), Vector3::new(-0.1, 0.3, 0.2));
        }

        let _ = Scene3D::detach_camera(scene);
        let _ = camera.unregister_component::<Camera>();
        let _ = camera.unregister_component::<Transform>();
        let _ = camera.unregister_component::<Renderable>();
        let _ = Node::detach(camera);
        let _ = camera.unregister_component::<Node>();

        let _ = scene.unregister_component::<Scene3D>();
        let _ = scene.unregister_component::<Layer>();
        let _ = scene.unregister_component::<Renderable>();
        let _ = scene.unregister_component::<Transform>();
        let _ = Node::detach(scene);
        let _ = scene.unregister_component::<Node>();
    }

    #[test]
    fn scene3d_detach_camera_clears_attachment() {
        let scene = Entity::new().expect("应能创建实体");
        prepare_scene_entity(&scene, "scene3d_detach_root");
        Scene3D::insert(scene, Scene3D::new("scene3d_detach")).expect("应能插入 Scene3D");

        let camera = Entity::new().expect("应能创建实体");
        prepare_camera_entity(&camera, "camera_detach");
        Node::attach(camera, scene).expect("应能将摄像机挂载到场景");
        let _ = camera
            .register_component(Camera::new("detachable"))
            .expect("应能插入 Camera");

        Scene3D::attach_camera(scene, camera).expect("应能绑定摄像机");
        let detached = Scene3D::detach_camera(scene).expect("应能解绑摄像机");
        assert_eq!(detached, Some(camera));
        let stored = scene.get_component::<Scene3D>().expect("应能读取 Scene3D");
        assert_eq!(stored.attached_camera(), None);
        drop(stored);

        let _ = camera.unregister_component::<Camera>();
        let _ = camera.unregister_component::<Transform>();
        let _ = camera.unregister_component::<Renderable>();
        let _ = Node::detach(camera);
        let _ = camera.unregister_component::<Node>();

        let _ = scene.unregister_component::<Scene3D>();
        let _ = scene.unregister_component::<Layer>();
        let _ = scene.unregister_component::<Renderable>();
        let _ = scene.unregister_component::<Transform>();
        let _ = Node::detach(scene);
        let _ = scene.unregister_component::<Node>();
    }
}
