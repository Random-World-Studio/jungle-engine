//! 二维场景组件（`Scene2D`）。
//!
//! 把一个挂载了 [`Layer`] 的实体声明为“2D 渲染层”。
//! 常见用法：
//! - 在 Layer 根实体上注册 `Scene2D`（会自动补齐默认 2D 着色器）。
//! - 在该 Layer 子树下放置 `Renderable + Shape + Transform (+ Material)` 的实体参与渲染。
//! - 使用 [`Scene2D::visible_faces`] 获取可见三角面（用于 2D 渲染/拾取/遮挡分析等）。

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::OnceLock,
};

use nalgebra::{Vector2, Vector3};

use lodtree::{LodVec, coords::OctVec};

use async_trait::async_trait;

use super::{
    component_impl,
    layer::{
        Layer, LayerLodUpdate, LayerRenderableBundle, LayerTraversalError, LayerViewport,
        RenderPipelineStage, ShaderLanguage,
    },
};
use crate::game::{
    component::{Component, ComponentDependencyError, ComponentStorage},
    entity::Entity,
};
use crate::resource::ResourcePath;

/// 二维场景组件。
///
/// 二维场景默认按照世界空间的 `z` 进行遮挡：更大的 `z` 代表更靠前。
///
/// ## 深度（z）约定
///
/// `Scene2D` 渲染启用了深度测试，并使用顶点的 `z` 作为深度值：
/// - `z` **必须**在 `[0,1]` 范围内，否则该三角形会被直接丢弃。
/// - 遮挡关系：`z` 越大越靠前（越小越靠后）。
///
/// ## 坐标系与“屏幕原点”
///
/// `Scene2D` 的渲染坐标采用 NDC（Normalized Device Coordinates）映射：
/// - 视口中心对应 NDC 的 `(0,0)`。
/// - 默认 `offset = (0,0)` 时，世界坐标 `(0,0)` 会落在视口中心。
/// - `offset` 表示“视口中心对应的世界坐标”。
/// - `pixels_per_unit` 表示世界单位到像素的缩放（值越大，同样大小的物体在屏幕上越大）。
/// - x 轴向右为正，y 轴向上为正。
///
/// 依赖与约定：
/// - 该组件依赖 [`Layer`]。注册 `Scene2D` 时会按需自动注册 `Layer`。
/// - 参与渲染的实体通常需要：
///   - [`Renderable`](jge_core::game::component::renderable::Renderable)
///   - [`Shape`](jge_core::game::component::shape::Shape)
///   - [`Transform`](jge_core::game::component::transform::Transform)
///   - （可选）[`Material`](jge_core::game::component::material::Material)
///
/// # 示例
///
/// ```no_run
/// # use jge_core::game::{entity::Entity, component::{scene2d::Scene2D, layer::Layer}};
/// # fn main() -> anyhow::Result<()> {
/// let rt = tokio::runtime::Runtime::new()?;
/// rt.block_on(async {
///     let layer_root = Entity::new().await?;
///     layer_root.register_component(Layer::new()).await?;
///     layer_root.register_component(Scene2D::new()).await?;
///     Ok::<(), anyhow::Error>(())
/// })?;
/// Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Scene2D {
    entity_id: Option<Entity>,
    offset: Vector2<f32>,
    pixels_per_unit: f32,
    framebuffer_size: Option<(u32, u32)>,
}

/// Scene2D 在当前视口内可见的世界坐标范围（单位：世界单位）。
///
/// `min/max` 对应的轴向边界为：
/// - `min.x`：左边界
/// - `max.x`：右边界
/// - `min.y`：下边界
/// - `max.y`：上边界
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scene2DVisibleWorldBounds {
    pub min: Vector2<f32>,
    pub max: Vector2<f32>,
}

const DEFAULT_PIXELS_PER_UNIT: f32 = 100.0;

static SCENE2D_STORAGE: OnceLock<ComponentStorage<Scene2D>> = OnceLock::new();

#[component_impl]
impl Scene2D {
    /// 创建一个 2D 场景组件。
    ///
    /// 默认值：
    /// - `offset = (0,0)`
    /// - `pixels_per_unit = 100`
    #[default()]
    pub fn new() -> Self {
        Self {
            entity_id: None,
            offset: Vector2::zeros(),
            pixels_per_unit: DEFAULT_PIXELS_PER_UNIT,
            framebuffer_size: None,
        }
    }

    /// 设置坐标原点的偏移量（世界单位）。
    ///
    /// 该偏移量表示“视口中心点（NDC=(0,0)）对应的世界坐标”。
    pub fn set_offset(&mut self, offset: Vector2<f32>) {
        self.offset = offset;
    }

    /// 当前坐标偏移量（世界单位）。
    pub fn offset(&self) -> Vector2<f32> {
        self.offset
    }

    /// 设置世界坐标中的单位长度对应的像素数量。
    ///
    /// 该值必须为有限正数；否则会忽略本次设置。
    pub fn set_pixels_per_unit(&mut self, value: f32) {
        if value.is_sign_positive() && value.is_finite() {
            self.pixels_per_unit = value.max(f32::EPSILON);
        }
    }

    /// 当前单位长度对应的像素数量。
    pub fn pixels_per_unit(&self) -> f32 {
        self.pixels_per_unit
    }

    pub(crate) fn set_framebuffer_size(&mut self, framebuffer_size: (u32, u32)) {
        self.framebuffer_size = Some((framebuffer_size.0.max(1), framebuffer_size.1.max(1)));
    }

    fn viewport_framebuffer_size(
        framebuffer_size: (u32, u32),
        viewport: Option<LayerViewport>,
    ) -> (u32, u32) {
        let fb_width = framebuffer_size.0.max(1) as f32;
        let fb_height = framebuffer_size.1.max(1) as f32;

        if let Some(viewport) = viewport {
            let values = [viewport.x, viewport.y, viewport.width, viewport.height];
            if values.iter().any(|v| !v.is_finite()) {
                return (framebuffer_size.0.max(1), framebuffer_size.1.max(1));
            }

            let x = viewport.x.clamp(0.0, 1.0);
            let y = viewport.y.clamp(0.0, 1.0);
            let w = viewport.width.clamp(0.0, 1.0);
            let h = viewport.height.clamp(0.0, 1.0);

            let x2 = (x + w).clamp(0.0, 1.0);
            let y2 = (y + h).clamp(0.0, 1.0);

            let x_px = (x * fb_width).floor() as i64;
            let y_px = (y * fb_height).floor() as i64;
            let x2_px = (x2 * fb_width).ceil() as i64;
            let y2_px = (y2 * fb_height).ceil() as i64;

            let fb_w_i64 = fb_width as i64;
            let fb_h_i64 = fb_height as i64;
            let x_px = x_px.clamp(0, fb_w_i64);
            let y_px = y_px.clamp(0, fb_h_i64);
            let x2_px = x2_px.clamp(0, fb_w_i64);
            let y2_px = y2_px.clamp(0, fb_h_i64);

            let width_px = (x2_px - x_px).max(1) as u32;
            let height_px = (y2_px - y_px).max(1) as u32;
            return (width_px, height_px);
        }

        (framebuffer_size.0.max(1), framebuffer_size.1.max(1))
    }

    /// 获取当前视口内“屏幕可见”的世界坐标范围。
    ///
    /// - 返回值是世界单位下的轴对齐边界框。
    ///
    /// 当 `Scene2D` 已绑定到一个实体上时，该函数会从该实体的 [`Layer`] 中读取
    /// `LayerViewport`，从而在存在视口裁剪（分屏/角落 UI Layer）时返回“裁剪后的可见范围”。
    ///
    /// 注意：该函数不依赖 wgpu/winit；但它依赖“窗口/渲染目标像素尺寸”。
    /// 引擎会在渲染阶段（以及窗口 resize 后的下一帧渲染）自动更新该尺寸。
    /// 在尺寸尚未初始化时会返回 `None`。
    pub fn visible_world_bounds(&self) -> Option<Scene2DVisibleWorldBounds> {
        let framebuffer_size = self.framebuffer_size?;

        let viewport = match (self.entity_id, tokio::runtime::Handle::try_current()) {
            (Some(entity), Ok(handle)) => tokio::task::block_in_place(|| {
                handle.block_on(async {
                    entity
                        .get_component::<Layer>()
                        .await
                        .and_then(|layer| layer.viewport())
                })
            }),
            _ => None,
        };

        let (vp_width_px, vp_height_px) =
            Self::viewport_framebuffer_size(framebuffer_size, viewport);

        let half_width = vp_width_px as f32 * 0.5;
        let half_height = vp_height_px as f32 * 0.5;
        let ppu = self.pixels_per_unit.max(f32::EPSILON);

        let dx = half_width / ppu;
        let dy = half_height / ppu;

        Some(Scene2DVisibleWorldBounds {
            min: Vector2::new(self.offset.x - dx, self.offset.y - dy),
            max: Vector2::new(self.offset.x + dx, self.offset.y + dy),
        })
    }

    /// 把“视口内像素坐标”转换为世界坐标。
    ///
    /// 该函数是 `Scene2D` 的统一屏幕坐标转换入口：用于把鼠标/触控等输入的像素位置映射到世界空间。
    ///
    /// ## 像素坐标系（输入）
    ///
    /// - 该像素坐标系**相对视口的左上角**：
    ///   - 如果当前 Layer 设置了 [`LayerViewport`]，那么这里的“视口”指 **viewport 与 framebuffer 的交集**（即被裁剪后的可见区域）。
    ///   - 如果未设置 viewport，则视口等价于整个 framebuffer。
    /// - 方向约定（相对于监视器屏幕）：
    ///   - x+：向右
    ///   - y+：向下
    ///
    /// ## 世界坐标系（输出）
    ///
    /// - 与 `Scene2D` 其余接口保持一致：x+ 向右，y+ 向上。
    /// - `offset` 表示“视口中心（NDC=(0,0)）对应的世界坐标”。
    /// - `pixels_per_unit` 表示 1 世界单位对应的像素数量。
    ///
    /// ## 返回值
    ///
    /// 返回 `None` 表示 framebuffer 尺寸尚未由引擎渲染阶段初始化（或输入坐标不是有限数）。
    pub fn pixel_to_world(&self, pixel: Vector2<f32>) -> Option<Vector2<f32>> {
        if !pixel.x.is_finite() || !pixel.y.is_finite() {
            return None;
        }

        let bounds = self.visible_world_bounds()?;
        let ppu = self.pixels_per_unit.max(f32::EPSILON);

        // pixel 原点在视口左上，y+ 向下；世界坐标 y+ 向上，因此需要翻转 y。
        Some(Vector2::new(
            bounds.min.x + pixel.x / ppu,
            bounds.max.y - pixel.y / ppu,
        ))
    }

    /// 预热场景的 LOD，使 Layer 至少构建出一批八叉树节点。
    ///
    /// 返回值表示是否成功生成了至少一个激活节点。
    pub fn warmup_lod(&self, layer: &mut Layer) -> bool {
        // 最小可用策略：先激活 root chunk（depth=0）。
        // 这样可见性查询至少能命中一个 chunk，避免出现“没有可见 draw”的空结果。
        let _ = self.step_lod(layer, &[OctVec::root()], 0);
        if !self.chunk_positions(layer).is_empty() {
            return true;
        }

        const DETAIL_LEVEL: u64 = 3;
        const MAX_ITERATIONS: usize = 32;
        let target = OctVec::new(128, 128, 128, 5);

        for _ in 0..MAX_ITERATIONS {
            let update = self.step_lod(layer, &[target], DETAIL_LEVEL);
            if update.is_empty() {
                break;
            }
        }

        let chunks = self.chunk_positions(layer);
        !chunks.is_empty()
    }

    fn ensure_default_layer_shaders(entity: Entity) {
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };

        tokio::task::block_in_place(|| {
            handle.block_on(async {
                if let Some(mut layer) = entity.get_component_mut::<Layer>().await {
                    if layer.shader(RenderPipelineStage::Vertex).is_none() {
                        let _ = layer.attach_shader_from_path(
                            RenderPipelineStage::Vertex,
                            ShaderLanguage::Wgsl,
                            ResourcePath::from("shaders/2d.vs"),
                        );
                    }
                    if layer.shader(RenderPipelineStage::Fragment).is_none() {
                        let _ = layer.attach_shader_from_path(
                            RenderPipelineStage::Fragment,
                            ShaderLanguage::Wgsl,
                            ResourcePath::from("shaders/2d.fs"),
                        );
                    }
                }
            })
        });
    }

    /// 基于 Layer 的八叉树推进 LOD。
    pub fn step_lod(&self, layer: &mut Layer, targets: &[OctVec], detail: u64) -> LayerLodUpdate {
        let _ = self;
        layer.step_lod(targets, detail)
    }

    /// 当前激活节点数量。
    pub fn chunk_count(&self, layer: &Layer) -> usize {
        let _ = self;
        layer.chunk_count()
    }

    /// 返回激活节点的位置集合。
    pub fn chunk_positions(&self, layer: &Layer) -> Vec<OctVec> {
        let _ = self;
        layer.chunk_positions()
    }

    /// 查找指定节点在半径范围内的邻居。
    pub fn chunk_neighbors(&self, layer: &Layer, center: OctVec, radius: u64) -> Vec<OctVec> {
        let _ = self;
        layer.chunk_neighbors(center, radius)
    }

    /// 收集当前 Layer 树下所有启用 Shape 的未被完全遮挡的三角面集合。
    /// 返回的数组按照 Layer 八叉树节点顺序分组，每个子数组包含同一实体的连续面。
    ///
    /// 适合用于：
    /// - 2D 渲染（按 chunk/实体批处理）
    /// - 2D 逻辑拾取/可见性判定
    ///
    /// 注意：需要该组件已绑定到实体，且该实体已注册 [`Layer`]。
    pub fn visible_faces(&self) -> Result<Vec<Scene2DFaceGroup>, Scene2DVisibilityError> {
        let entity = self
            .entity_id
            .expect("Scene2D::visible_faces requires the component to be attached to an entity");

        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return Err(Scene2DVisibilityError::LayerTraversal(
                LayerTraversalError::MissingLayer(entity),
            ));
        };

        let (layer, renderables) = tokio::task::block_in_place(|| {
            let layer = handle
                .block_on(entity.get_component::<Layer>())
                .expect("Scene2D::visible_faces requires Layer component");
            let renderables = handle
                .block_on(layer.world_renderables())
                .map_err(Scene2DVisibilityError::from)?;
            Ok::<_, Scene2DVisibilityError>((layer, renderables))
        })?;
        self.visible_faces_with_renderables(&layer, &renderables)
    }

    /// 与 [`visible_faces`](Self::visible_faces) 相同，但允许调用方复用已收集好的渲染数据。
    ///
    /// 当你在同一帧内需要多次查询（例如渲染 + 物理拾取）时，可先通过
    /// `layer.world_renderables()` 获取一次数据，再多次传入此函数以减少重复遍历。
    pub fn visible_faces_with_renderables(
        &self,
        layer: &Layer,
        renderables: &[LayerRenderableBundle],
    ) -> Result<Vec<Scene2DFaceGroup>, Scene2DVisibilityError> {
        // Scene2D 过去依赖 Layer 的 LOD（chunk_positions）来进行分组/加速。
        // 但如果调用方忘记 warmup/step_lod，就会出现“没有可见 draw”的空结果。
        //
        // 与 Scene3D 的行为保持一致：默认不要求调用方先准备 LOD。
        // 当 Layer 没有任何激活 chunk 时，我们退化为“单 chunk（root）”模式，
        // 让 Scene2D 仍能产出可见面集合（只是少了 chunk 级别的分组优化）。
        let mut chunks = self.chunk_positions(layer);
        if chunks.is_empty() {
            chunks.push(OctVec::root());
        }

        let mut chunk_index = BTreeMap::new();
        let mut depths = BTreeSet::new();
        for (index, chunk) in chunks.iter().copied().enumerate() {
            chunk_index.insert(chunk, index);
            depths.insert(chunk.depth);
        }

        #[derive(Clone)]
        struct FaceRecord {
            entity: Entity,
            vertices: [Vector3<f32>; 3],
            min_z: f32,
            max_z: f32,
            min_xy: Vector2<f32>,
            max_xy: Vector2<f32>,
        }

        fn bounding_box_contains(container: &FaceRecord, target: &FaceRecord) -> bool {
            const XY_EPSILON: f32 = 1e-4;
            container.min_xy.x - XY_EPSILON <= target.min_xy.x
                && container.min_xy.y - XY_EPSILON <= target.min_xy.y
                && container.max_xy.x + XY_EPSILON >= target.max_xy.x
                && container.max_xy.y + XY_EPSILON >= target.max_xy.y
        }

        let mut faces_per_chunk: Vec<Vec<FaceRecord>> = vec![Vec::new(); chunks.len()];

        for bundle in renderables {
            for world_vertices in bundle.triangles() {
                let vertices = world_vertices.clone();

                let mut chunk_indices = BTreeSet::new();
                for depth in &depths {
                    for vertex in &vertices {
                        let normalized = Self::normalize_world_point(vertex);
                        let oct = OctVec::from_float_coords(
                            normalized[0],
                            normalized[1],
                            normalized[2],
                            *depth,
                        );
                        if let Some(index) = chunk_index.get(&oct) {
                            chunk_indices.insert(*index);
                        }
                    }
                }

                if chunk_indices.is_empty() {
                    continue;
                }

                let min_z = vertices
                    .iter()
                    .fold(f32::INFINITY, |acc, vertex| acc.min(vertex.z));
                let max_z = vertices
                    .iter()
                    .fold(f32::NEG_INFINITY, |acc, vertex| acc.max(vertex.z));

                let mut min_xy = Vector2::new(f32::INFINITY, f32::INFINITY);
                let mut max_xy = Vector2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
                for vertex in &vertices {
                    min_xy.x = min_xy.x.min(vertex.x);
                    min_xy.y = min_xy.y.min(vertex.y);
                    max_xy.x = max_xy.x.max(vertex.x);
                    max_xy.y = max_xy.y.max(vertex.y);
                }

                let record = FaceRecord {
                    entity: bundle.entity(),
                    vertices,
                    min_z,
                    max_z,
                    min_xy,
                    max_xy,
                };

                for index in chunk_indices {
                    faces_per_chunk[index].push(record.clone());
                }
            }
        }

        const DEPTH_EPSILON: f32 = 1e-4;
        let mut result: Vec<Scene2DFaceGroup> = Vec::new();

        for records in faces_per_chunk.into_iter() {
            if records.is_empty() {
                continue;
            }

            let mut filtered: Vec<FaceRecord> = Vec::new();
            for (index, face) in records.iter().enumerate() {
                let mut occluded = false;
                for (other_index, other) in records.iter().enumerate() {
                    if index == other_index {
                        continue;
                    }
                    if other.min_z - DEPTH_EPSILON > face.max_z
                        && bounding_box_contains(other, face)
                    {
                        occluded = true;
                        break;
                    }
                }
                if !occluded {
                    filtered.push(face.clone());
                }
            }

            if filtered.is_empty() {
                continue;
            }

            let mut current_entity = None;
            let mut current_faces: Vec<[Vector3<f32>; 3]> = Vec::new();

            for face in filtered {
                if current_entity != Some(face.entity) {
                    if let Some(entity_id) = current_entity {
                        result.push(Scene2DFaceGroup {
                            entity: entity_id,
                            faces: current_faces,
                        });
                        current_faces = Vec::new();
                    }
                    current_entity = Some(face.entity);
                }

                current_faces.push(face.vertices);
            }

            if let Some(entity_id) = current_entity {
                if !current_faces.is_empty() {
                    result.push(Scene2DFaceGroup {
                        entity: entity_id,
                        faces: current_faces,
                    });
                }
            }
        }

        Ok(result)
    }
    fn normalize_world_point(point: &Vector3<f32>) -> [f64; 3] {
        [
            Self::normalize_coordinate(point.x),
            Self::normalize_coordinate(point.y),
            Self::normalize_coordinate(point.z),
        ]
    }

    fn normalize_coordinate(value: f32) -> f64 {
        const NORMALIZED_MAX: f64 = 1.0 - 1e-9;
        let clamped = value.clamp(-1.0, 1.0);
        let normalized = ((clamped + 1.0) * 0.5) as f64;
        normalized.clamp(0.0, NORMALIZED_MAX)
    }
}

#[derive(Debug, Clone)]
pub struct Scene2DFaceGroup {
    entity: Entity,
    faces: Vec<[Vector3<f32>; 3]>,
}

impl Scene2DFaceGroup {
    /// 该组面片所属的实体。
    pub fn entity(&self) -> Entity {
        self.entity
    }

    /// 该实体在当前可见集合中的三角面列表（世界空间）。
    pub fn faces(&self) -> &[[Vector3<f32>; 3]] {
        &self.faces
    }

    /// 消费该对象并返回面片列表。
    pub fn into_faces(self) -> Vec<[Vector3<f32>; 3]> {
        self.faces
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Scene2DVisibilityError {
    /// 传入实体缺少 Scene2D 组件。
    MissingScene(Entity),
    /// Layer 遍历失败（例如缺少 Layer/Node）。
    LayerTraversal(LayerTraversalError),
}

impl From<LayerTraversalError> for Scene2DVisibilityError {
    fn from(value: LayerTraversalError) -> Self {
        Self::LayerTraversal(value)
    }
}

impl fmt::Display for Scene2DVisibilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scene2DVisibilityError::MissingScene(entity) => {
                write!(f, "实体 {} 未注册 Scene2D 组件", entity.id())
            }
            Scene2DVisibilityError::LayerTraversal(error) => write!(f, "{}", error),
        }
    }
}

impl std::error::Error for Scene2DVisibilityError {}

#[async_trait]
impl Component for Scene2D {
    fn storage() -> &'static ComponentStorage<Self> {
        SCENE2D_STORAGE.get_or_init(ComponentStorage::new)
    }

    async fn register_dependencies(entity: Entity) -> Result<(), ComponentDependencyError> {
        if entity.get_component::<Layer>().await.is_none() {
            let component = Layer::__jge_component_default(entity)?;
            let _ = entity.register_component(component).await?;
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
    use super::Scene2D;
    use crate::game::component::{
        Component,
        layer::{Layer, RenderPipelineStage, ShaderLanguage},
        node::Node,
        renderable::Renderable,
        shape::Shape,
        transform::Transform,
    };
    use crate::game::entity::Entity;
    use lodtree::{coords::OctVec, traits::LodVec};
    use nalgebra::{Vector2, Vector3};

    async fn detach_node(entity: Entity) {
        if entity.get_component::<Node>().await.is_some() {
            let detach_future = {
                let mut node = entity
                    .get_component_mut::<Node>()
                    .await
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
                .await
                .expect("父实体应持有 Node 组件");
            parent_node.attach(entity)
        };
        attach_future.await.expect(message);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scene2d_requires_layer_dependency() {
        let entity = Entity::new().await.expect("应能创建实体");
        let _ = entity.unregister_component::<Scene2D>().await;
        let _ = entity.unregister_component::<Layer>().await;
        let _ = entity.unregister_component::<Renderable>().await;
        detach_node(entity).await;
        let _ = entity.unregister_component::<Node>().await;

        let inserted = entity
            .register_component(Scene2D::new())
            .await
            .expect("缺少 Layer 时应自动注册依赖");
        assert!(inserted.is_none());

        assert!(
            entity.get_component::<Layer>().await.is_some(),
            "Layer 应被自动注册"
        );
        assert!(
            entity.get_component::<Renderable>().await.is_some(),
            "Renderable 应被注册"
        );
        assert!(
            entity.get_component::<Node>().await.is_some(),
            "Node 应被注册"
        );
        let layer = entity
            .get_component::<Layer>()
            .await
            .expect("应能读取 Layer");
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
        let _ = entity.unregister_component::<Scene2D>().await;
        let _ = entity.unregister_component::<Layer>().await;
        let _ = entity.unregister_component::<Renderable>().await;
        detach_node(entity).await;
        let _ = entity.unregister_component::<Node>().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scene2d_offset_and_scale_configuration() {
        let mut scene = Scene2D::new();
        scene.set_offset(Vector2::new(3.0, -2.0));
        scene.set_pixels_per_unit(64.0);

        assert_eq!(scene.offset(), Vector2::new(3.0, -2.0));
        assert!((scene.pixels_per_unit() - 64.0).abs() < f32::EPSILON);

        // 非法数值应被忽略。
        scene.set_pixels_per_unit(-10.0);
        assert!((scene.pixels_per_unit() - 64.0).abs() < f32::EPSILON);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scene2d_visible_world_bounds_defaults_to_center_origin() {
        let mut scene = Scene2D::new();
        scene.set_framebuffer_size((200, 100));
        let bounds = scene
            .visible_world_bounds()
            .expect("framebuffer size should be set");

        // pixels_per_unit=100 => half width=100px => 1.0 world units.
        assert!((bounds.min.x - (-1.0)).abs() < 1.0e-6);
        assert!((bounds.max.x - (1.0)).abs() < 1.0e-6);
        // half height=50px => 0.5 world units.
        assert!((bounds.min.y - (-0.5)).abs() < 1.0e-6);
        assert!((bounds.max.y - (0.5)).abs() < 1.0e-6);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scene2d_visible_world_bounds_uses_layer_viewport_when_attached() {
        let entity = Entity::new().await.expect("应能创建实体");

        // Layer + Scene2D 需要 Renderable 依赖（Scene2D 依赖 Layer，Layer 依赖 Renderable）。
        entity
            .register_component(Renderable::new())
            .await
            .expect("应能注册 Renderable");

        let mut layer = Layer::new();
        layer.set_viewport(crate::game::component::layer::LayerViewport::normalized(
            0.0, 0.0, 0.5, 0.5,
        ));
        entity
            .register_component(layer)
            .await
            .expect("应能注册 Layer");

        let mut scene = Scene2D::new();
        scene.set_offset(Vector2::new(0.0, 0.0));
        scene.set_pixels_per_unit(100.0);
        entity
            .register_component(scene)
            .await
            .expect("应能注册 Scene2D");

        // 模拟引擎在渲染阶段更新的 framebuffer size。
        {
            let mut scene = entity
                .get_component_mut::<Scene2D>()
                .await
                .expect("应能读取 Scene2D");
            scene.set_framebuffer_size((200, 100));
        }

        let scene = entity
            .get_component::<Scene2D>()
            .await
            .expect("应能读取 Scene2D");

        // framebuffer 200x100，viewport 0.5x0.5 => 100x50 pixels.
        // ppu=100 => half width=50px => 0.5 world units; half height=25px => 0.25 world units.
        let bounds = scene
            .visible_world_bounds()
            .expect("framebuffer size should be set");
        assert!((bounds.min.x - (-0.5)).abs() < 1.0e-6);
        assert!((bounds.max.x - 0.5).abs() < 1.0e-6);
        assert!((bounds.min.y - (-0.25)).abs() < 1.0e-6);
        assert!((bounds.max.y - 0.25).abs() < 1.0e-6);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scene2d_visible_world_bounds_respects_offset_and_pixels_per_unit_when_attached() {
        let entity = Entity::new().await.expect("应能创建实体");
        entity
            .register_component(Renderable::new())
            .await
            .expect("应能注册 Renderable");
        entity
            .register_component(Layer::new())
            .await
            .expect("应能注册 Layer");

        let mut scene = Scene2D::new();
        scene.set_offset(Vector2::new(10.0, -4.0));
        scene.set_pixels_per_unit(50.0);
        entity
            .register_component(scene)
            .await
            .expect("应能注册 Scene2D");

        {
            let mut scene = entity
                .get_component_mut::<Scene2D>()
                .await
                .expect("应能读取 Scene2D");
            scene.set_framebuffer_size((400, 200));
        }

        let scene = entity
            .get_component::<Scene2D>()
            .await
            .expect("应能读取 Scene2D");
        let bounds = scene
            .visible_world_bounds()
            .expect("framebuffer size should be set");

        // half width=200px / 50ppu => 4.0 world units.
        assert!((bounds.min.x - 6.0).abs() < 1.0e-6);
        assert!((bounds.max.x - 14.0).abs() < 1.0e-6);
        // half height=100px / 50ppu => 2.0 world units.
        assert!((bounds.min.y - (-6.0)).abs() < 1.0e-6);
        assert!((bounds.max.y - (-2.0)).abs() < 1.0e-6);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scene2d_pixel_to_world_respects_framebuffer_size_and_axis_directions() {
        let mut scene = Scene2D::new();
        scene.set_framebuffer_size((200, 100));
        scene.set_offset(Vector2::new(0.0, 0.0));
        scene.set_pixels_per_unit(100.0);

        // 像素坐标原点在左上，y+ 向下；默认可见范围为 x:[-1,1], y:[-0.5,0.5]
        let p0 = scene
            .pixel_to_world(Vector2::new(0.0, 0.0))
            .expect("framebuffer size should be set");
        assert!((p0.x - (-1.0)).abs() < 1.0e-6);
        assert!((p0.y - 0.5).abs() < 1.0e-6);

        let center = scene
            .pixel_to_world(Vector2::new(100.0, 50.0))
            .expect("framebuffer size should be set");
        assert!(center.x.abs() < 1.0e-6);
        assert!(center.y.abs() < 1.0e-6);

        let br = scene
            .pixel_to_world(Vector2::new(200.0, 100.0))
            .expect("framebuffer size should be set");
        assert!((br.x - 1.0).abs() < 1.0e-6);
        assert!((br.y - (-0.5)).abs() < 1.0e-6);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scene2d_pixel_to_world_uses_layer_viewport_when_attached() {
        let entity = Entity::new().await.expect("应能创建实体");
        entity
            .register_component(Renderable::new())
            .await
            .expect("应能注册 Renderable");

        let mut layer = Layer::new();
        layer.set_viewport(crate::game::component::layer::LayerViewport::normalized(
            0.0, 0.0, 0.5, 0.5,
        ));
        entity
            .register_component(layer)
            .await
            .expect("应能注册 Layer");

        let mut scene = Scene2D::new();
        scene.set_offset(Vector2::new(0.0, 0.0));
        scene.set_pixels_per_unit(100.0);
        entity
            .register_component(scene)
            .await
            .expect("应能注册 Scene2D");

        {
            let mut scene = entity
                .get_component_mut::<Scene2D>()
                .await
                .expect("应能读取 Scene2D");
            scene.set_framebuffer_size((200, 100));
        }

        let scene = entity
            .get_component::<Scene2D>()
            .await
            .expect("应能读取 Scene2D");

        // viewport 0.5x0.5 => 100x50 pixels，可见范围为 x:[-0.5,0.5], y:[-0.25,0.25]
        let p0 = scene
            .pixel_to_world(Vector2::new(0.0, 0.0))
            .expect("framebuffer size should be set");
        assert!((p0.x - (-0.5)).abs() < 1.0e-6);
        assert!((p0.y - 0.25).abs() < 1.0e-6);

        let center = scene
            .pixel_to_world(Vector2::new(50.0, 25.0))
            .expect("framebuffer size should be set");
        assert!(center.x.abs() < 1.0e-6);
        assert!(center.y.abs() < 1.0e-6);
    }

    async fn register_layer_scene(entity: Entity) {
        let _ = Scene2D::remove(entity).await;
        let _ = Layer::remove(entity).await;
        let _ = Renderable::remove(entity).await;
        detach_node(entity).await;

        let _ = entity
            .register_component(Renderable::new())
            .await
            .expect("应能插入 Renderable");
        let _ = entity
            .register_component(Layer::new())
            .await
            .expect("应能插入 Layer");

        let inserted = entity
            .register_component(Scene2D::new())
            .await
            .expect("满足依赖后应能插入 Scene2D");
        assert!(inserted.is_none());
    }

    async fn activate_root_chunk(entity: Entity) {
        let scene = entity
            .get_component::<Scene2D>()
            .await
            .expect("Scene2D 组件应已注册");
        let mut layer = entity
            .get_component_mut::<Layer>()
            .await
            .expect("Scene2D 应持有 Layer");
        let update = scene.step_lod(&mut layer, &[OctVec::root()], 0);
        assert!(
            !update.added.is_empty() || !update.activated.is_empty(),
            "推进 LOD 后应至少激活根节点"
        );
        drop(layer);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn visible_faces_no_longer_requires_lod_warmup() {
        let root = Entity::new().await.expect("应能创建根实体");
        register_layer_scene(root).await;

        let child = Entity::new().await.expect("应能创建子实体");
        let triangle = [
            Vector3::new(-0.2, -0.2, 0.2),
            Vector3::new(0.2, -0.2, 0.2),
            Vector3::new(0.0, 0.2, 0.2),
        ];
        register_shape(child, &[triangle], Vector3::zeros()).await;
        attach_node(child, root, "应能挂载子实体").await;

        let visible = {
            let scene = root
                .get_component::<Scene2D>()
                .await
                .expect("根节点应挂载 Scene2D");
            scene.visible_faces().expect("应能收集可见面片")
        };

        assert_eq!(visible.len(), 1, "未 warmup 时也应能产出可见面组");
        assert_eq!(visible[0].faces().len(), 1);
        assert_eq!(visible[0].faces()[0], triangle);

        detach_node(child).await;
        let _ = Scene2D::remove(root).await;
        let _ = Layer::remove(root).await;
        let _ = Renderable::remove(root).await;

        let _ = Shape::remove(child).await;
        let _ = Transform::remove(child).await;
        let _ = Renderable::remove(child).await;
    }

    async fn register_shape(
        entity: Entity,
        triangles: &[[Vector3<f32>; 3]],
        translation: Vector3<f32>,
    ) {
        let _ = Shape::remove(entity).await;
        let _ = Transform::remove(entity).await;
        let _ = Renderable::remove(entity).await;
        detach_node(entity).await;

        let _ = entity
            .register_component(Renderable::new())
            .await
            .expect("应能插入 Renderable");
        let transform =
            Transform::with_components(translation, Vector3::zeros(), Vector3::repeat(1.0));
        let _ = entity
            .register_component(transform)
            .await
            .expect("应能插入 Transform");

        let shape = Shape::from_triangles(triangles.to_vec());
        let _ = entity
            .register_component(shape)
            .await
            .expect("应能插入 Shape");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn visible_faces_excludes_fully_occluded_triangles() {
        let root = Entity::new().await.expect("应能创建根实体");
        register_layer_scene(root).await;
        activate_root_chunk(root).await;

        let back = Entity::new().await.expect("应能创建后景实体");
        let front = Entity::new().await.expect("应能创建前景实体");

        let far_triangle = [
            Vector3::new(-0.25, -0.25, -0.5),
            Vector3::new(0.25, -0.25, -0.5),
            Vector3::new(0.0, 0.25, -0.5),
        ];
        let near_triangle = [
            Vector3::new(-0.25, -0.25, 0.5),
            Vector3::new(0.25, -0.25, 0.5),
            Vector3::new(0.0, 0.25, 0.5),
        ];

        register_shape(back, &[far_triangle], Vector3::zeros()).await;
        register_shape(front, &[near_triangle], Vector3::zeros()).await;

        attach_node(back, root, "应能挂载后景").await;
        attach_node(front, root, "应能挂载前景").await;

        let visible = {
            let scene = root
                .get_component::<Scene2D>()
                .await
                .expect("根节点应挂载 Scene2D");
            scene.visible_faces().expect("应能收集可见面片")
        };
        assert_eq!(visible.len(), 1, "只有前景面的实体应可见");
        assert_eq!(visible[0].faces().len(), 1, "仅有一个三角面应可见");
        assert_eq!(visible[0].faces()[0], near_triangle);

        detach_node(front).await;
        detach_node(back).await;
        let _ = Scene2D::remove(root).await;
        let _ = Layer::remove(root).await;
        let _ = Renderable::remove(root).await;

        let _ = Shape::remove(front).await;
        let _ = Transform::remove(front).await;
        let _ = Renderable::remove(front).await;
        let _ = Shape::remove(back).await;
        let _ = Transform::remove(back).await;
        let _ = Renderable::remove(back).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn visible_faces_keep_partially_occluded_triangles() {
        let root = Entity::new().await.expect("应能创建根实体");
        register_layer_scene(root).await;
        activate_root_chunk(root).await;

        let partial = Entity::new().await.expect("应能创建部分遮挡实体");
        let ground = Entity::new().await.expect("应能创建背景实体");

        let partial_triangle = [
            Vector3::new(-0.25, -0.25, -0.4),
            Vector3::new(0.25, -0.25, 0.6),
            Vector3::new(0.0, 0.25, -0.2),
        ];
        let ground_triangle = [
            Vector3::new(-0.3, -0.3, 0.0),
            Vector3::new(0.3, -0.3, 0.0),
            Vector3::new(0.0, 0.3, 0.0),
        ];

        register_shape(partial, &[partial_triangle], Vector3::zeros()).await;
        register_shape(ground, &[ground_triangle], Vector3::zeros()).await;

        attach_node(partial, root, "应能挂载部分遮挡实体").await;
        attach_node(ground, root, "应能挂载背景实体").await;

        let visible = {
            let scene = root
                .get_component::<Scene2D>()
                .await
                .expect("根节点应挂载 Scene2D");
            scene.visible_faces().expect("应能收集可见面片")
        };
        assert_eq!(visible.len(), 2, "部分遮挡情况下应保留两个面组");
        assert_eq!(visible[0].faces().len(), 1);
        assert_eq!(visible[0].faces()[0], partial_triangle);
        assert_eq!(visible[1].faces().len(), 1);
        assert_eq!(visible[1].faces()[0], ground_triangle);

        detach_node(ground).await;
        detach_node(partial).await;
        let _ = Scene2D::remove(root).await;
        let _ = Layer::remove(root).await;
        let _ = Renderable::remove(root).await;

        let _ = Shape::remove(ground).await;
        let _ = Transform::remove(ground).await;
        let _ = Renderable::remove(ground).await;
        let _ = Shape::remove(partial).await;
        let _ = Transform::remove(partial).await;
        let _ = Renderable::remove(partial).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn visible_faces_ignores_disabled_renderables() {
        let root = Entity::new().await.expect("应能创建根实体");
        register_layer_scene(root).await;
        activate_root_chunk(root).await;

        let visible_entity = Entity::new().await.expect("应能创建可见实体");
        let hidden_entity = Entity::new().await.expect("应能创建隐藏实体");

        let visible_triangle = [
            Vector3::new(-0.2, -0.2, 0.2),
            Vector3::new(0.2, -0.2, 0.2),
            Vector3::new(0.0, 0.2, 0.2),
        ];
        let hidden_triangle = [
            Vector3::new(-0.2, -0.2, 0.6),
            Vector3::new(0.2, -0.2, 0.6),
            Vector3::new(0.0, 0.2, 0.6),
        ];

        register_shape(visible_entity, &[visible_triangle], Vector3::zeros()).await;
        register_shape(hidden_entity, &[hidden_triangle], Vector3::zeros()).await;

        attach_node(visible_entity, root, "应能挂载可见实体").await;
        attach_node(hidden_entity, root, "应能挂载隐藏实体").await;

        {
            let mut hidden_renderable = hidden_entity
                .get_component_mut::<Renderable>()
                .await
                .expect("隐藏实体应有 Renderable");
            hidden_renderable.set_enabled(false);
        }

        let visible = {
            let scene = root
                .get_component::<Scene2D>()
                .await
                .expect("根节点应挂载 Scene2D");
            scene.visible_faces().expect("应能收集可见面片")
        };
        assert_eq!(visible.len(), 1, "禁用 Renderable 后应仅保留可见实体");
        assert_eq!(visible[0].faces().len(), 1);
        assert_eq!(visible[0].faces()[0], visible_triangle);

        detach_node(hidden_entity).await;
        detach_node(visible_entity).await;
        let _ = Scene2D::remove(root).await;
        let _ = Layer::remove(root).await;
        let _ = Renderable::remove(root).await;

        let _ = Shape::remove(hidden_entity).await;
        let _ = Transform::remove(hidden_entity).await;
        let _ = Renderable::remove(hidden_entity).await;
        let _ = Shape::remove(visible_entity).await;
        let _ = Transform::remove(visible_entity).await;
        let _ = Renderable::remove(visible_entity).await;
    }
}
