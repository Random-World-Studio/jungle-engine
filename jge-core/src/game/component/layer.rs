//! 渲染层组件（`Layer`）。
//!
//! `Layer` 用来把一棵节点子树声明为一个独立的“渲染层”（一次渲染 pass 的逻辑单位）。
//! 引擎渲染时会：
//! - 从根实体开始遍历节点树
//! - 遇到挂载了 `Layer` 的实体时，把它当作一个 Layer 根来渲染
//! - **不会继续深入遍历该实体的子树来寻找其他 Layer**（避免嵌套 Layer 被重复渲染）
//!
//! 你通常会把 `Layer` 与下列组件搭配使用：
//! - [`Scene2D`](jge_core::game::component::scene2d::Scene2D) / [`Scene3D`](jge_core::game::component::scene3d::Scene3D)：选择具体渲染路径
//! - [`Background`](jge_core::game::component::background::Background)：渲染层背景
//! - `Renderable + Shape + Transform (+ Material)`：可渲染实体
//!
//! # 示例
//!
//! ```no_run
//! use jge_core::game::{
//!     component::{layer::Layer, node::Node, renderable::Renderable, transform::Transform},
//!     entity::Entity,
//! };
//!
//! # fn main() -> anyhow::Result<()> {
//! # let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
//! # rt.block_on(async move {
//! // 创建一个 Layer 根实体
//! let root = Entity::new()?;
//! root.register_component(Layer::new())?;
//!
//! // 在 Layer 子树下创建一个可渲染实体
//! let child = Entity::new()?;
//! child.register_component(Renderable::new())?;
//! child.register_component(Transform::new())?;
//! // child.register_component(Shape::from_triangles(...))?;
//!
//! // 维护节点关系
//! let attach_future = {
//!     let mut root_node = root.get_component_mut::<Node>().unwrap();
//!     root_node.attach(child)
//! };
//! attach_future.await?;
//! Ok(())
//! # })
//! # }
//! ```

use std::{collections::HashMap, fmt, sync::Arc};

use lodtree::{coords::OctVec, tree::Tree};

use super::light::{Light, ParallelLight, PointLight};
use super::material::{Material, MaterialPatch};
use super::node::Node;
use super::renderable::Renderable;
use super::shape::Shape;
use super::transform::Transform;
use super::{component, component_impl};
use crate::game::{component::ComponentRead, entity::Entity};
use crate::resource::{Resource, ResourceHandle, ResourcePath};
use nalgebra::{Matrix4, Vector3, Vector4};

/// 渲染层视口（归一化矩形）。
///
/// 坐标为 0..=1 的归一化空间，原点在左上角。
///
/// 典型用法：把 UI Layer 放在屏幕一角，或做分屏。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerViewport {
    /// 归一化坐标（0..=1），原点在左上角。
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl LayerViewport {
    /// 创建一个归一化视口矩形（0..=1）。
    pub fn normalized(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

#[component(Renderable)]
#[derive(Debug)]
/// 渲染层组件。
///
/// 把某个实体标记为一个独立的渲染层根（一次渲染 pass 的逻辑单位）。
///
/// 常见工作流：
/// - 在“层根实体”上注册 `Layer`，并在同一实体上注册 `Scene2D` 或 `Scene3D`。
/// - 在该层的节点子树下挂载 `Renderable + Shape + Transform (+ Material)` 的实体。
/// - 如需自定义着色器，通过 [`attach_shader_from_path`](Self::attach_shader_from_path) 绑定。
pub struct Layer {
    entity_id: Option<Entity>,
    shaders: HashMap<RenderPipelineStage, LayerShader>,
    spatial: LayerSpatialIndex,
    viewport: Option<LayerViewport>,
}

#[component_impl]
impl Layer {
    /// 创建一个 Layer。
    #[default()]
    pub fn new() -> Self {
        Self {
            entity_id: None,
            shaders: HashMap::new(),
            spatial: LayerSpatialIndex::new(),
            viewport: None,
        }
    }

    /// 获取该 Layer 的视口裁剪区域（归一化坐标），未设置则表示全屏。
    pub fn viewport(&self) -> Option<LayerViewport> {
        self.viewport
    }

    /// 设置该 Layer 的视口裁剪区域（归一化坐标）。
    pub fn set_viewport(&mut self, viewport: LayerViewport) {
        self.viewport = Some(viewport);
    }

    /// 清除视口裁剪区域，恢复为全屏渲染。
    pub fn clear_viewport(&mut self) {
        self.viewport = None;
    }

    /// 获取指定渲染阶段关联的着色器描述。
    pub fn shader(&self, stage: RenderPipelineStage) -> Option<&LayerShader> {
        self.shaders.get(&stage)
    }

    /// 为指定渲染阶段挂载着色器，返回之前的挂载（若存在）。
    pub fn attach_shader(
        &mut self,
        stage: RenderPipelineStage,
        shader: LayerShader,
    ) -> Option<LayerShader> {
        self.shaders.insert(stage, shader)
    }

    /// 通过资源路径与语言快速挂载着色器。
    ///
    /// - `resource_path` 必须已注册到资源系统（例如通过 `resource!` 宏）。
    /// - 成功时返回旧的着色器（如果该 stage 之前已绑定）。
    pub fn attach_shader_from_path(
        &mut self,
        stage: RenderPipelineStage,
        language: ShaderLanguage,
        resource_path: ResourcePath,
    ) -> Result<Option<LayerShader>, LayerShaderAttachError> {
        let resource = Resource::from(resource_path.clone())
            .ok_or(LayerShaderAttachError::ResourceMissing(resource_path))?;
        Ok(self.attach_shader(stage, LayerShader::new(language, resource)))
    }

    /// 移除指定阶段的着色器。
    pub fn detach_shader(&mut self, stage: RenderPipelineStage) -> Option<LayerShader> {
        self.shaders.remove(&stage)
    }

    /// 基于给定目标更新八叉树 LOD，并返回本次迭代的变化。
    ///
    /// `targets` 通常来自玩家/摄像机在世界中的位置映射到 `OctVec` 的结果。
    pub fn step_lod(&mut self, targets: &[OctVec], detail: u64) -> LayerLodUpdate {
        self.spatial.step_lod(targets, detail)
    }

    /// 当前处于激活状态的节点数量。
    pub fn chunk_count(&self) -> usize {
        self.spatial.chunk_count()
    }

    /// 返回所有激活节点的位置。
    pub fn chunk_positions(&self) -> Vec<OctVec> {
        self.spatial.chunk_positions()
    }

    /// 查询指定节点在给定半径内的激活邻居。
    pub fn chunk_neighbors(&self, center: OctVec, radius: u64) -> Vec<OctVec> {
        self.spatial.chunk_neighbors(center, radius)
    }

    /// 重置空间索引，清空树结构。
    pub fn clear_spatial_index(&mut self) {
        self.spatial.clear();
    }

    /// 以该 Layer 为根，收集同一 Layer 树下的所有可渲染实体。
    ///
    /// - 会按照节点树的先序遍历顺序返回实体 ID。
    /// - 子树中若遇到其他 Layer，将整体跳过，交由对应 Layer 单独处理。
    /// - 若当前 Layer 自身处于其他 Layer 管理的子树内，则直接返回空列表。
    pub fn renderable_entities(root: Entity) -> Result<Vec<Entity>, LayerTraversalError> {
        let ordered = Self::traverse_layer_entities(root)?;
        let mut renderables = Vec::new();

        for entity in ordered {
            if let Some(renderable) = entity.get_component::<Renderable>() {
                if renderable.is_enabled() {
                    renderables.push(entity);
                }
            }
        }

        Ok(renderables)
    }

    /// 收集当前 Layer 子树下的点光源实体。
    ///
    /// 只会返回同时拥有 [`Light`] 与 [`PointLight`] 的实体。
    /// 遍历规则与 [`renderable_entities`](Self::renderable_entities) 一致（会跳过嵌套 Layer 子树）。
    pub fn point_light_entities(root: Entity) -> Result<Vec<Entity>, LayerTraversalError> {
        let ordered = Self::traverse_layer_entities(root)?;
        let mut lights = Vec::new();

        for entity in ordered {
            if entity.get_component::<Light>().is_some()
                && entity.get_component::<PointLight>().is_some()
            {
                lights.push(entity);
            }
        }

        Ok(lights)
    }

    /// 收集当前 Layer 子树下的平行光实体。
    ///
    /// 只会返回同时拥有 [`Light`] 与 [`ParallelLight`] 的实体。
    /// 遍历规则与 [`renderable_entities`](Self::renderable_entities) 一致（会跳过嵌套 Layer 子树）。
    pub fn parallel_light_entities(root: Entity) -> Result<Vec<Entity>, LayerTraversalError> {
        let ordered = Self::traverse_layer_entities(root)?;
        let mut lights = Vec::new();

        for entity in ordered {
            if entity.get_component::<Light>().is_some()
                && entity.get_component::<ParallelLight>().is_some()
            {
                lights.push(entity);
            }
        }

        Ok(lights)
    }

    /// 收集当前 Layer 下所有可渲染实体的世界空间三角面集合。
    ///
    /// 返回的顺序与 [`Self::renderable_entities`] 保持一致，便于复用渲染排序。
    pub fn world_triangles(root: Entity) -> Result<Vec<LayerTriangle>, LayerTraversalError> {
        let bundles = Self::gather_renderables(root)?;
        let mut triangles = Vec::new();
        for bundle in bundles {
            triangles.extend(bundle.into_layer_triangles());
        }
        Ok(triangles)
    }

    /// 收集当前 Layer 下所有可渲染实体的世界空间渲染数据。
    ///
    /// 每个条目包含实体标识、转换后的三角面列表以及（若存在）绑定的材质描述，
    /// 便于在 2D 与 3D 渲染路径之间共享基础数据。
    pub fn world_renderables(
        root: Entity,
    ) -> Result<Vec<LayerRenderableBundle>, LayerTraversalError> {
        Self::gather_renderables(root)
    }

    /// 构造渲染前数据集合，便于多处渲染路径共享。
    pub fn collect_renderables(
        root: Entity,
    ) -> Result<LayerRenderableCollection, LayerTraversalError> {
        Self::gather_renderables(root).map(LayerRenderableCollection::from_bundles)
    }

    fn gather_renderables(root: Entity) -> Result<Vec<LayerRenderableBundle>, LayerTraversalError> {
        let renderables = Self::renderable_entities(root)?;
        let mut bundles = Vec::new();

        for entity in renderables {
            let shape_guard = match entity.get_component::<Shape>() {
                Some(shape) => shape,
                None => continue,
            };
            let triangle_count = shape_guard.triangle_count();
            if triangle_count == 0 {
                continue;
            }

            let transform_guard = match entity.get_component::<Transform>() {
                Some(transform) => transform,
                None => continue,
            };
            let matrix = transform_guard.matrix();
            drop(transform_guard);

            let mut world_triangles = Vec::with_capacity(triangle_count);
            for (a, b, c) in shape_guard.triangles() {
                world_triangles.push([
                    transform_vertex_to_world(&matrix, a),
                    transform_vertex_to_world(&matrix, b),
                    transform_vertex_to_world(&matrix, c),
                ]);
            }
            drop(shape_guard);

            let material = entity.get_component::<Material>().map(|material_guard| {
                let regions: Vec<MaterialPatch> =
                    material_guard.regions().iter().copied().collect();
                let regions_arc = Arc::<[MaterialPatch]>::from(regions);
                LayerMaterialDescriptor::new(material_guard.resource(), regions_arc)
            });

            bundles.push(LayerRenderableBundle::new(
                entity,
                world_triangles,
                material,
            ));
        }

        Ok(bundles)
    }

    fn traverse_layer_entities(root: Entity) -> Result<Vec<Entity>, LayerTraversalError> {
        if root.get_component::<Layer>().is_none() {
            return Err(LayerTraversalError::MissingLayer(root));
        }

        if root.get_component::<Node>().is_none() {
            return Err(LayerTraversalError::MissingNode(root));
        }

        if Self::has_layer_ancestor(root)? {
            return Ok(Vec::new());
        }

        let mut stack = vec![root];
        let mut ordered = Vec::new();

        while let Some(current) = stack.pop() {
            if current != root && current.get_component::<Layer>().is_some() {
                continue;
            }

            ordered.push(current);

            let child_ids = {
                let node_guard = current
                    .get_component::<Node>()
                    .ok_or(LayerTraversalError::MissingNode(current))?;
                let mut ids: Vec<Entity> = node_guard.children().iter().copied().collect();
                ids.reverse();
                ids
            };

            for child in child_ids {
                stack.push(child);
            }
        }

        Ok(ordered)
    }

    fn has_layer_ancestor(entity: Entity) -> Result<bool, LayerTraversalError> {
        let mut current_parent = {
            let node_guard = entity
                .get_component::<Node>()
                .ok_or(LayerTraversalError::MissingNode(entity))?;
            node_guard.parent()
        };

        while let Some(id) = current_parent {
            if id.get_component::<Layer>().is_some() {
                return Ok(true);
            }

            let parent = {
                let node_guard = id
                    .get_component::<Node>()
                    .ok_or(LayerTraversalError::MissingNode(id))?;
                node_guard.parent()
            };
            current_parent = parent;
        }

        Ok(false)
    }
}

#[derive(Debug, Clone)]
pub struct LayerMaterialDescriptor {
    resource: ResourceHandle,
    regions: Arc<[MaterialPatch]>,
}

impl LayerMaterialDescriptor {
    fn new(resource: ResourceHandle, regions: Arc<[MaterialPatch]>) -> Self {
        Self { resource, regions }
    }

    pub fn resource(&self) -> &ResourceHandle {
        &self.resource
    }

    pub fn regions(&self) -> &[MaterialPatch] {
        &self.regions
    }
}

#[derive(Debug, Clone)]
pub struct LayerRenderableBundle {
    entity: Entity,
    triangles: Vec<[Vector3<f32>; 3]>,
    material: Option<LayerMaterialDescriptor>,
}

impl LayerRenderableBundle {
    pub(crate) fn new(
        entity: Entity,
        triangles: Vec<[Vector3<f32>; 3]>,
        material: Option<LayerMaterialDescriptor>,
    ) -> Self {
        Self {
            entity,
            triangles,
            material,
        }
    }

    pub fn entity(&self) -> Entity {
        self.entity
    }

    pub fn triangles(&self) -> &[[Vector3<f32>; 3]] {
        &self.triangles
    }

    pub fn material(&self) -> Option<&LayerMaterialDescriptor> {
        self.material.as_ref()
    }

    pub fn into_material(self) -> Option<LayerMaterialDescriptor> {
        self.material
    }

    pub fn into_components(
        self,
    ) -> (
        Entity,
        Vec<[Vector3<f32>; 3]>,
        Option<LayerMaterialDescriptor>,
    ) {
        (self.entity, self.triangles, self.material)
    }

    fn into_layer_triangles(self) -> Vec<LayerTriangle> {
        let entity = self.entity;
        self.triangles
            .into_iter()
            .map(|triangle| LayerTriangle::new(entity, triangle))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct LayerRenderableCollection {
    bundles: Vec<LayerRenderableBundle>,
    lookup: HashMap<Entity, usize>,
}

impl LayerRenderableCollection {
    pub(crate) fn from_bundles(bundles: Vec<LayerRenderableBundle>) -> Self {
        let lookup = bundles
            .iter()
            .enumerate()
            .map(|(index, bundle)| (bundle.entity(), index))
            .collect();
        Self { bundles, lookup }
    }

    pub fn bundles(&self) -> &[LayerRenderableBundle] {
        &self.bundles
    }

    pub fn iter(&self) -> impl Iterator<Item = &LayerRenderableBundle> {
        self.bundles.iter()
    }

    pub fn get(&self, entity: Entity) -> Option<&LayerRenderableBundle> {
        self.lookup
            .get(&entity)
            .and_then(|index| self.bundles.get(*index))
    }

    pub fn material(&self, entity: Entity) -> Option<&LayerMaterialDescriptor> {
        self.get(entity).and_then(|bundle| bundle.material())
    }

    pub fn into_bundles(self) -> Vec<LayerRenderableBundle> {
        self.bundles
    }
}

impl ComponentRead<Layer> {
    pub fn renderable_entities(&self) -> Result<Vec<Entity>, LayerTraversalError> {
        Layer::renderable_entities(self.entity())
    }

    pub fn point_light_entities(&self) -> Result<Vec<Entity>, LayerTraversalError> {
        Layer::point_light_entities(self.entity())
    }

    pub fn parallel_light_entities(&self) -> Result<Vec<Entity>, LayerTraversalError> {
        Layer::parallel_light_entities(self.entity())
    }

    pub fn world_renderables(&self) -> Result<Vec<LayerRenderableBundle>, LayerTraversalError> {
        Layer::world_renderables(self.entity())
    }

    pub fn collect_renderables(&self) -> Result<LayerRenderableCollection, LayerTraversalError> {
        Layer::collect_renderables(self.entity())
    }

    pub fn world_triangles(&self) -> Result<Vec<LayerTriangle>, LayerTraversalError> {
        Layer::world_triangles(self.entity())
    }
}

#[derive(Debug, Clone)]
pub struct LayerTriangle {
    entity: Entity,
    vertices: [Vector3<f32>; 3],
}

impl LayerTriangle {
    fn new(entity: Entity, vertices: [Vector3<f32>; 3]) -> Self {
        Self { entity, vertices }
    }

    pub fn entity(&self) -> Entity {
        self.entity
    }

    pub fn vertices(&self) -> &[Vector3<f32>; 3] {
        &self.vertices
    }

    pub fn into_vertices(self) -> [Vector3<f32>; 3] {
        self.vertices
    }
}

fn transform_vertex_to_world(matrix: &Matrix4<f32>, local: &Vector3<f32>) -> Vector3<f32> {
    let vector = matrix * Vector4::new(local.x, local.y, local.z, 1.0);
    Vector3::new(vector.x, vector.y, vector.z)
}

/// 层（Layer）遍历过程中可能出现的错误。
#[derive(Debug, PartialEq, Eq)]
pub enum LayerTraversalError {
    MissingLayer(Entity),
    MissingNode(Entity),
}

impl fmt::Display for LayerTraversalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerTraversalError::MissingLayer(entity) => {
                write!(f, "实体 {} 未注册 Layer 组件", entity.id())
            }
            LayerTraversalError::MissingNode(entity) => {
                write!(f, "实体 {} 未注册 Node 组件", entity.id())
            }
        }
    }
}

impl std::error::Error for LayerTraversalError {}

/// 渲染管线可用的着色器阶段。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderPipelineStage {
    Vertex,
    Fragment,
}

/// 支持的着色器语言。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderLanguage {
    Glsl,
    Wgsl,
}

/// 层（Layer）上挂载的着色器描述。
#[derive(Debug, Clone)]
pub struct LayerShader {
    language: ShaderLanguage,
    resource: ResourceHandle,
}

impl LayerShader {
    /// 使用资源句柄创建一个着色器描述。
    ///
    /// 资源内容通常是源码（GLSL/WGSL），具体解析由渲染路径决定。
    pub fn new(language: ShaderLanguage, resource: ResourceHandle) -> Self {
        Self { language, resource }
    }

    pub fn language(&self) -> ShaderLanguage {
        self.language
    }

    pub fn resource(&self) -> &ResourceHandle {
        &self.resource
    }

    /// 获取挂载的资源句柄副本。
    pub fn resource_handle(&self) -> ResourceHandle {
        Arc::clone(&self.resource)
    }
}

impl PartialEq for LayerShader {
    fn eq(&self, other: &Self) -> bool {
        self.language == other.language && Arc::ptr_eq(&self.resource, &other.resource)
    }
}

impl Eq for LayerShader {}

#[derive(Debug, PartialEq, Eq)]
pub enum LayerShaderAttachError {
    /// 指定资源路径未注册。
    ResourceMissing(ResourcePath),
}

impl fmt::Display for LayerShaderAttachError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerShaderAttachError::ResourceMissing(path) => {
                write!(f, "资源路径 {} 未注册", path.join("/"))
            }
        }
    }
}

impl std::error::Error for LayerShaderAttachError {}

/// 层（Layer）空间索引操作过程中可能出现的错误。
#[derive(Debug, PartialEq, Eq)]
pub enum LayerSpatialError {
    MissingLayer(Entity),
}

impl fmt::Display for LayerSpatialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerSpatialError::MissingLayer(entity) => {
                write!(f, "实体 {} 未注册 Layer 组件", entity.id())
            }
        }
    }
}

impl std::error::Error for LayerSpatialError {}

/// 细节层次（LOD）更新过程中的节点变更集合。
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LayerLodUpdate {
    /// 新增的节点（创建）。
    pub added: Vec<OctVec>,
    /// 移除的节点（销毁）。
    pub removed: Vec<OctVec>,
    /// 从非激活变为激活的节点。
    pub activated: Vec<OctVec>,
    /// 从激活变为非激活的节点。
    pub deactivated: Vec<OctVec>,
}

impl LayerLodUpdate {
    /// 是否没有任何变更。
    pub fn is_empty(&self) -> bool {
        self.added.is_empty()
            && self.removed.is_empty()
            && self.activated.is_empty()
            && self.deactivated.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LayerChunk {
    position: OctVec,
    active: bool,
}

impl LayerChunk {
    fn new(position: OctVec) -> Self {
        Self {
            position,
            active: true,
        }
    }

    fn position(&self) -> OctVec {
        self.position
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn set_active(&mut self, value: bool) {
        self.active = value;
    }

    fn reset(&mut self, position: OctVec) {
        self.position = position;
        self.active = true;
    }
}

#[derive(Debug)]
struct LayerSpatialIndex {
    tree: Tree<LayerChunk, OctVec>,
}

impl LayerSpatialIndex {
    fn new() -> Self {
        Self { tree: Tree::new() }
    }

    fn clear(&mut self) {
        self.tree.clear();
    }

    fn chunk_count(&self) -> usize {
        (0..self.tree.get_num_chunks())
            .filter(|&idx| self.tree.get_chunk(idx).is_active())
            .count()
    }

    fn chunk_positions(&self) -> Vec<OctVec> {
        (0..self.tree.get_num_chunks())
            .filter_map(|idx| {
                let chunk = self.tree.get_chunk(idx);
                if chunk.is_active() {
                    Some(chunk.position())
                } else {
                    None
                }
            })
            .collect()
    }

    fn chunk_neighbors(&self, center: OctVec, radius: u64) -> Vec<OctVec> {
        let radius = radius as i64;
        self.chunk_positions()
            .into_iter()
            .filter(|pos| pos.depth == center.depth)
            .filter(|pos| {
                let dx = pos.x as i64 - center.x as i64;
                let dy = pos.y as i64 - center.y as i64;
                let dz = pos.z as i64 - center.z as i64;
                dx.abs() <= radius && dy.abs() <= radius && dz.abs() <= radius
            })
            .collect()
    }

    fn step_lod(&mut self, targets: &[OctVec], detail: u64) -> LayerLodUpdate {
        let mut update = LayerLodUpdate::default();

        if targets.is_empty() {
            return update;
        }

        let needs_update = self
            .tree
            .prepare_update(targets, detail, |position| LayerChunk::new(position));

        if !needs_update {
            return update;
        }

        {
            let pending = self.tree.get_chunks_to_add_slice_mut();
            update.added.reserve(pending.len());
            for (position, chunk) in pending.iter_mut() {
                chunk.reset(*position);
                update.added.push(*position);
            }
        }

        for i in 0..self.tree.get_num_chunks_to_activate() {
            let chunk = self.tree.get_chunk_to_activate_mut(i);
            chunk.set_active(true);
            update.activated.push(chunk.position());
        }

        for i in 0..self.tree.get_num_chunks_to_deactivate() {
            let chunk = self.tree.get_chunk_to_deactivate_mut(i);
            chunk.set_active(false);
            update.deactivated.push(chunk.position());
        }

        for i in 0..self.tree.get_num_chunks_to_remove() {
            let chunk = self.tree.get_chunk_to_remove(i);
            update.removed.push(chunk.position());
        }

        self.tree.do_update();

        update
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::entity::Entity;
    use crate::resource::{Resource, ResourceHandle};
    use lodtree::{coords::OctVec, traits::LodVec};
    use nalgebra::Vector2;

    async fn detach_node(entity: Entity) {
        if entity.get_component::<Node>().is_some() {
            let detach_future = {
                let mut node = entity
                    .get_component_mut::<Node>()
                    .expect("node component disappeared");
                node.detach()
            };
            let _ = detach_future.await;
        }
    }

    async fn attach_node(entity: Entity, parent: Entity, message: &str) {
        let attach_future = {
            let mut parent_node = parent
                .get_component_mut::<Node>()
                .expect("父实体应持有 Node 组件");
            parent_node.attach(entity)
        };
        attach_future.await.expect(message);
    }

    async fn prepare_node(entity: &Entity, name: &str) {
        let _ = entity.unregister_component::<Layer>();
        let _ = entity.unregister_component::<Renderable>();
        detach_node(*entity).await;
        let _ = entity.unregister_component::<Node>();
        let _ = entity
            .register_component(Node::new(name).expect("应能创建节点"))
            .expect("应能注册 Node");
    }

    async fn cleanup(entity: &Entity) {
        let _ = entity.unregister_component::<Layer>();
        let _ = entity.unregister_component::<Renderable>();
        detach_node(*entity).await;
        let _ = entity.unregister_component::<Node>();
    }

    #[tokio::test]
    async fn layer_requires_renderable_dependency() {
        let entity = Entity::new().expect("应能创建实体");
        let inserted = entity
            .register_component(Layer::new())
            .expect("缺少 Renderable 时应自动注册依赖");
        assert!(inserted.is_none());

        assert!(
            entity.get_component::<Renderable>().is_some(),
            "依赖的 Renderable 应已注册"
        );
        assert!(
            entity.get_component::<Node>().is_some(),
            "Renderable 的依赖 Node 应已注册"
        );

        let previous_auto = entity
            .register_component(Layer::new())
            .expect("重复插入应返回旧的 Layer");
        assert!(previous_auto.is_some());

        prepare_node(&entity, "layer_root").await;
        let _ = entity
            .register_component(Renderable::new())
            .expect("应能插入 Renderable");

        let reinserted = entity
            .register_component(Layer::new())
            .expect("满足依赖后应能插入 Layer");
        assert!(reinserted.is_none());

        cleanup(&entity).await;
    }

    #[tokio::test]
    async fn renderable_entities_skip_nested_layers() {
        let root = Entity::new().expect("应能创建 root 实体");
        let child_a = Entity::new().expect("应能创建 child_a 实体");
        let child_b = Entity::new().expect("应能创建 child_b 实体");
        let nested_root = Entity::new().expect("应能创建 nested_root 实体");
        let nested_child = Entity::new().expect("应能创建 nested_child 实体");

        prepare_node(&root, "root").await;
        prepare_node(&child_a, "child_a").await;
        prepare_node(&child_b, "child_b").await;
        prepare_node(&nested_root, "nested_root").await;
        prepare_node(&nested_child, "nested_child").await;

        // 构建节点树
        attach_node(child_a, root, "应能挂载 child_a").await;
        attach_node(child_b, root, "应能挂载 child_b").await;
        attach_node(nested_root, child_b, "应能挂载 nested_root").await;
        attach_node(nested_child, nested_root, "应能挂载 nested_child").await;

        // 注册可渲染组件
        let _ = root
            .register_component(Renderable::new())
            .expect("应能插入 root rend");
        let _ = child_a
            .register_component(Renderable::new())
            .expect("应能插入 child_a rend");
        let _ = child_b
            .register_component(Renderable::new())
            .expect("应能插入 child_b rend");
        let _ = nested_root
            .register_component(Renderable::new())
            .expect("应能插入 nested_root rend");
        let _ = nested_child
            .register_component(Renderable::new())
            .expect("应能插入 nested_child rend");

        // 注册 Layer（根以及子树）
        let _ = root
            .register_component(Layer::new())
            .expect("应能插入 root layer");
        let _ = nested_root
            .register_component(Layer::new())
            .expect("应能插入 nested layer");

        let root_renderables =
            Layer::renderable_entities(root).expect("应能收集 root layer 的渲染实体");
        assert_eq!(root_renderables, vec![root, child_a, child_b]);

        let nested_renderables =
            Layer::renderable_entities(nested_root).expect("嵌套 Layer 应被忽略并返回空列表");
        assert!(nested_renderables.is_empty());

        // 清理（按照叶 -> 根顺序）
        cleanup(&nested_child).await;
        cleanup(&nested_root).await;
        cleanup(&child_b).await;
        cleanup(&child_a).await;
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn point_light_entities_skip_nested_layers() {
        let root = Entity::new().expect("应能创建 root 实体");
        let nested_root = Entity::new().expect("应能创建 nested_root 实体");
        let light_a = Entity::new().expect("应能创建 light_a 实体");
        let light_b = Entity::new().expect("应能创建 light_b 实体");

        prepare_node(&root, "root").await;
        prepare_node(&nested_root, "nested_root").await;
        prepare_node(&light_a, "light_a").await;
        prepare_node(&light_b, "light_b").await;

        attach_node(light_a, root, "应能挂载光源 A 到根").await;
        attach_node(nested_root, root, "应能挂载 nested_root").await;
        attach_node(light_b, nested_root, "应能挂载光源 B 到嵌套 Layer").await;

        let _ = root
            .register_component(Renderable::new())
            .expect("应能插入 root Renderable");
        let _ = nested_root
            .register_component(Renderable::new())
            .expect("应能插入 nested Renderable");

        let _ = root
            .register_component(Layer::new())
            .expect("应能插入 root Layer");
        let _ = nested_root
            .register_component(Layer::new())
            .expect("应能插入 nested Layer");

        let _ = light_a
            .register_component(Renderable::new())
            .expect("应能为光源 A 注册 Renderable");
        if let Some(mut renderable) = light_a.get_component_mut::<Renderable>() {
            renderable.set_enabled(false);
        }
        let _ = light_a
            .register_component(Transform::new())
            .expect("应能为光源 A 注册 Transform");
        if let Some(mut transform) = light_a.get_component_mut::<Transform>() {
            transform.set_position(Vector3::new(0.5, 0.0, 0.0));
        }
        let _ = light_a
            .register_component(Light::new(1.0))
            .expect("应能插入 Light 组件");
        let _ = light_a
            .register_component(PointLight::new(2.0))
            .expect("应能插入 PointLight 组件");

        let _ = light_b
            .register_component(Renderable::new())
            .expect("应能为光源 B 注册 Renderable");
        if let Some(mut renderable) = light_b.get_component_mut::<Renderable>() {
            renderable.set_enabled(false);
        }
        let _ = light_b
            .register_component(Transform::new())
            .expect("应能为光源 B 注册 Transform");
        if let Some(mut transform) = light_b.get_component_mut::<Transform>() {
            transform.set_position(Vector3::new(-0.25, 0.25, 0.0));
        }
        let _ = light_b
            .register_component(Light::new(1.0))
            .expect("应能插入嵌套 Light 组件");
        let _ = light_b
            .register_component(PointLight::new(2.0))
            .expect("应能插入嵌套 PointLight 组件");

        assert!(
            light_a.get_component::<Light>().is_some(),
            "光源 A 应具备 Light 组件"
        );
        assert!(
            light_a.get_component::<PointLight>().is_some(),
            "光源 A 应具备 PointLight 组件"
        );

        let root_lights = Layer::point_light_entities(root).expect("根 Layer 应能收集光源");
        assert_eq!(root_lights, vec![light_a], "根 Layer 应仅包含直接子光源");

        let nested_lights =
            Layer::point_light_entities(nested_root).expect("嵌套 Layer 应仅返回自身与子光源");
        assert!(
            nested_lights.is_empty(),
            "嵌套 Layer 的光源应由自身渲染时处理"
        );

        cleanup(&light_b).await;
        cleanup(&nested_root).await;
        cleanup(&light_a).await;
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn parallel_light_entities_skip_nested_layers() {
        let root = Entity::new().expect("应能创建 root 实体");
        let nested_root = Entity::new().expect("应能创建 nested_root 实体");
        let light_a = Entity::new().expect("应能创建平行光实体 A");
        let light_b = Entity::new().expect("应能创建平行光实体 B");

        prepare_node(&root, "root_parallel").await;
        prepare_node(&nested_root, "nested_parallel").await;
        prepare_node(&light_a, "parallel_a").await;
        prepare_node(&light_b, "parallel_b").await;

        attach_node(light_a, root, "应能挂载平行光 A").await;
        attach_node(nested_root, root, "应能挂载 nested_root").await;
        attach_node(light_b, nested_root, "应能挂载平行光 B 到嵌套 Layer").await;

        let _ = root
            .register_component(Renderable::new())
            .expect("应能插入 root Renderable");
        let _ = nested_root
            .register_component(Renderable::new())
            .expect("应能插入 nested Renderable");

        let _ = root
            .register_component(Layer::new())
            .expect("应能插入 root Layer");
        let _ = nested_root
            .register_component(Layer::new())
            .expect("应能插入 nested Layer");

        let _ = light_a
            .register_component(Renderable::new())
            .expect("应能为平行光 A 注册 Renderable");
        if let Some(mut renderable) = light_a.get_component_mut::<Renderable>() {
            renderable.set_enabled(false);
        }
        let _ = light_a
            .register_component(Transform::new())
            .expect("应能为平行光 A 注册 Transform");
        let _ = light_a
            .register_component(Light::new(0.8))
            .expect("应能插入光源亮度");
        let _ = light_a
            .register_component(ParallelLight::new())
            .expect("应能插入平行光组件");

        let _ = light_b
            .register_component(Renderable::new())
            .expect("应能为平行光 B 注册 Renderable");
        if let Some(mut renderable) = light_b.get_component_mut::<Renderable>() {
            renderable.set_enabled(false);
        }
        let _ = light_b
            .register_component(Transform::new())
            .expect("应能为平行光 B 注册 Transform");
        let _ = light_b
            .register_component(Light::new(0.5))
            .expect("应能插入嵌套光源亮度");
        let _ = light_b
            .register_component(ParallelLight::new())
            .expect("应能插入嵌套平行光组件");

        let root_lights = Layer::parallel_light_entities(root).expect("根 Layer 应能收集平行光");
        assert_eq!(root_lights, vec![light_a], "根 Layer 应仅包含直接子平行光");

        let nested_lights =
            Layer::parallel_light_entities(nested_root).expect("嵌套 Layer 平行光查询应成功");
        assert!(
            nested_lights.is_empty(),
            "嵌套 Layer 的平行光应由其自身处理"
        );

        cleanup(&light_b).await;
        cleanup(&nested_root).await;
        cleanup(&light_a).await;
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn layer_shader_attachment_cycle() {
        let mut layer = Layer::new();

        let glsl_resource: ResourceHandle = Resource::from_memory(b"glsl".to_vec());
        let glsl_shader = LayerShader::new(ShaderLanguage::Glsl, Arc::clone(&glsl_resource));

        assert!(
            layer
                .attach_shader(RenderPipelineStage::Vertex, glsl_shader.clone())
                .is_none()
        );
        let stored = layer
            .shader(RenderPipelineStage::Vertex)
            .expect("应能读取挂载的着色器");
        assert_eq!(stored.language(), ShaderLanguage::Glsl);
        assert!(Arc::ptr_eq(stored.resource(), &glsl_resource));

        let wgsl_resource: ResourceHandle = Resource::from_memory(b"wgsl".to_vec());
        let wgsl_shader = LayerShader::new(ShaderLanguage::Wgsl, Arc::clone(&wgsl_resource));

        assert_eq!(
            layer
                .attach_shader(RenderPipelineStage::Vertex, wgsl_shader.clone())
                .expect("后一次挂载应返回先前的着色器"),
            glsl_shader
        );

        let current = layer
            .shader(RenderPipelineStage::Vertex)
            .expect("应能读取替换后的着色器");
        assert_eq!(current.language(), ShaderLanguage::Wgsl);
        assert!(Arc::ptr_eq(current.resource(), &wgsl_resource));

        assert_eq!(
            layer
                .detach_shader(RenderPipelineStage::Vertex)
                .expect("移除操作应返回着色器"),
            wgsl_shader
        );
        assert!(layer.shader(RenderPipelineStage::Vertex).is_none());
    }

    #[tokio::test]
    async fn layer_spatial_index_supports_octree_updates() {
        let entity = Entity::new().expect("应能创建实体");
        prepare_node(&entity, "layer_spatial_root").await;
        let _ = entity
            .register_component(Renderable::new())
            .expect("应能插入 Renderable");
        let _ = entity
            .register_component(Layer::new())
            .expect("应能插入 Layer");

        let mut layer = entity.get_component_mut::<Layer>().expect("应能写入 Layer");
        assert_eq!(layer.chunk_count(), 0);

        let target = OctVec::new(128, 128, 128, 5);
        let detail = 3;
        let mut iterations = 0usize;

        loop {
            let update = layer.step_lod(&[target], detail);
            if iterations == 0 {
                assert!(update.added.contains(&OctVec::root()));
            }
            if update.is_empty() {
                break;
            }
            iterations += 1;
            assert!(iterations < 32, "LOD 更新未在合理步数内收敛");
        }

        let positions = layer.chunk_positions();
        assert!(!positions.is_empty(), "八叉树应生成至少一个节点");

        let center = positions
            .iter()
            .copied()
            .max_by_key(|coord| coord.depth)
            .expect("至少应存在一个活跃节点");

        let neighbors = layer.chunk_neighbors(center, 1);
        assert!(!neighbors.is_empty(), "应能找到邻居节点");
        assert!(neighbors.contains(&center), "邻居结果应包含自身");

        drop(layer);
        cleanup(&entity).await;
    }

    #[tokio::test]
    async fn world_renderables_collects_triangles_and_materials() {
        let root = Entity::new().expect("应能创建渲染根实体");
        let child = Entity::new().expect("应能创建渲染子实体");

        prepare_node(&root, "renderables_root").await;
        prepare_node(&child, "renderables_child").await;
        attach_node(child, root, "应能挂载子实体到根").await;

        let _ = root
            .register_component(Renderable::new())
            .expect("应能插入根 Renderable");
        let _ = root
            .register_component(Layer::new())
            .expect("应能插入根 Layer");

        let _ = child
            .register_component(Renderable::new())
            .expect("应能插入子 Renderable");
        let _ = child
            .register_component(Transform::new())
            .expect("应能插入子 Transform");
        if let Some(mut transform) = child.get_component_mut::<Transform>() {
            transform.set_position(Vector3::new(0.5, -0.25, 1.0));
        }

        let shape = Shape::from_triangles(vec![[
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ]]);
        let _ = child.register_component(shape).expect("应能插入 Shape");

        let resource: ResourceHandle = Resource::from_memory(vec![1, 2, 3, 4]);
        let material = Material::new(
            resource.clone(),
            vec![[
                Vector2::new(0.0, 0.0),
                Vector2::new(1.0, 0.0),
                Vector2::new(0.0, 1.0),
            ]],
        );
        let _ = child
            .register_component(material)
            .expect("应能插入 Material");

        let bundles = Layer::world_renderables(root).expect("应能收集世界渲染数据");
        assert_eq!(bundles.len(), 1);
        let bundle = &bundles[0];
        assert_eq!(bundle.entity(), child);
        assert_eq!(bundle.triangles().len(), 1);
        let transformed = bundle.triangles()[0];
        assert!(
            (transformed[0].x - 0.5).abs() < 1e-6
                && (transformed[0].y + 0.25).abs() < 1e-6
                && (transformed[0].z - 1.0).abs() < 1e-6,
            "世界空间顶点应包含 Transform 偏移"
        );
        let descriptor = bundle.material().expect("应收集材质信息");
        assert_eq!(descriptor.regions().len(), 1);
        assert!(
            Arc::ptr_eq(descriptor.resource(), &resource),
            "材质资源句柄应保持引用一致"
        );

        let triangles = Layer::world_triangles(root).expect("应能扁平化三角面");
        assert_eq!(triangles.len(), 1);
        assert_eq!(triangles[0].entity(), child);
        let vertex = triangles[0].vertices()[0];
        assert!((vertex.x - 0.5).abs() < 1e-6 && (vertex.z - 1.0).abs() < 1e-6);

        let collection = Layer::collect_renderables(root).expect("应能构建渲染集合");
        assert_eq!(collection.bundles().len(), 1);
        assert!(collection.get(child).is_some());
        assert!(collection.material(child).is_some());

        let _ = child.unregister_component::<Material>();
        let _ = child.unregister_component::<Shape>();
        let _ = child.unregister_component::<Transform>();
        let _ = child.unregister_component::<Renderable>();
        cleanup(&child).await;
        cleanup(&root).await;
    }
}
