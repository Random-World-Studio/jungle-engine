use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use nalgebra::{Vector2, Vector3};

use lodtree::coords::OctVec;

use super::{
    component, component_impl,
    layer::{
        Layer, LayerLodUpdate, LayerRenderableBundle, LayerSpatialError, LayerTraversalError,
        RenderPipelineStage, ShaderLanguage,
    },
};
use crate::game::{
    component::{Component, ComponentDependencyError},
    entity::Entity,
};
use crate::resource::ResourcePath;

#[component(Layer)]
#[derive(Debug)]
/// 二维场景默认按照 Renderable 的 z 值进行遮挡，高 z 表示位于前方。
pub struct Scene2D {
    name: String,
    offset: Vector2<f32>,
    pixels_per_unit: f32,
}

impl Clone for Scene2D {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            offset: self.offset,
            pixels_per_unit: self.pixels_per_unit,
        }
    }
}

const DEFAULT_PIXELS_PER_UNIT: f32 = 100.0;

#[component_impl]
impl Scene2D {
    #[allow(dead_code)]
    #[default(Layer::new())]
    fn ensure_defaults(_layer: Layer) -> Self {
        Self::new("auto_scene2d")
    }
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            offset: Vector2::zeros(),
            pixels_per_unit: DEFAULT_PIXELS_PER_UNIT,
        }
    }

    /// 插入组件并自动为依赖的 Layer 挂载默认的 2D 着色器。
    pub fn insert(
        entity: Entity,
        scene: Scene2D,
    ) -> Result<Option<Scene2D>, ComponentDependencyError> {
        let previous = entity.register_component(scene)?;
        Self::ensure_default_layer_shaders(entity);
        Ok(previous)
    }

    /// 设置坐标原点的偏移量（世界单位）。
    pub fn set_offset(&mut self, offset: Vector2<f32>) {
        self.offset = offset;
    }

    /// 当前坐标偏移量（世界单位）。
    pub fn offset(&self) -> Vector2<f32> {
        self.offset
    }

    /// 设置世界坐标中的单位长度对应的像素数量。
    pub fn set_pixels_per_unit(&mut self, value: f32) {
        if value.is_sign_positive() && value.is_finite() {
            self.pixels_per_unit = value.max(f32::EPSILON);
        }
    }

    /// 当前单位长度对应的像素数量。
    pub fn pixels_per_unit(&self) -> f32 {
        self.pixels_per_unit
    }

    /// 预热场景的 LOD，使 Layer 至少构建出一批八叉树节点。
    ///
    /// 返回值表示是否成功生成了至少一个激活节点。
    pub fn warmup_lod(entity: Entity) -> Result<bool, LayerSpatialError> {
        const DETAIL_LEVEL: u64 = 3;
        const MAX_ITERATIONS: usize = 32;
        let target = OctVec::new(128, 128, 128, 5);

        for _ in 0..MAX_ITERATIONS {
            let update = Self::step_lod(entity, &[target], DETAIL_LEVEL)?;
            if update.is_empty() {
                break;
            }
        }

        let chunks = Self::chunk_positions(entity)?;
        Ok(!chunks.is_empty())
    }

    fn ensure_default_layer_shaders(entity: Entity) {
        if let Some(mut layer) = Layer::write(entity) {
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
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// 基于 Layer 的八叉树推进 LOD。
    pub fn step_lod(
        entity: Entity,
        targets: &[OctVec],
        detail: u64,
    ) -> Result<LayerLodUpdate, LayerSpatialError> {
        let mut layer = Layer::write(entity).ok_or(LayerSpatialError::MissingLayer(entity))?;
        Ok(layer.step_lod(targets, detail))
    }

    /// 当前激活节点数量。
    pub fn chunk_count(entity: Entity) -> Result<usize, LayerSpatialError> {
        let layer = Layer::read(entity).ok_or(LayerSpatialError::MissingLayer(entity))?;
        Ok(layer.chunk_count())
    }

    /// 返回激活节点的位置集合。
    pub fn chunk_positions(entity: Entity) -> Result<Vec<OctVec>, LayerSpatialError> {
        let layer = Layer::read(entity).ok_or(LayerSpatialError::MissingLayer(entity))?;
        Ok(layer.chunk_positions())
    }

    /// 查找指定节点在半径范围内的邻居。
    pub fn chunk_neighbors(
        entity: Entity,
        center: OctVec,
        radius: u64,
    ) -> Result<Vec<OctVec>, LayerSpatialError> {
        let layer = Layer::read(entity).ok_or(LayerSpatialError::MissingLayer(entity))?;
        Ok(layer.chunk_neighbors(center, radius))
    }

    /// 收集当前 Layer 树下所有启用 Shape 的未被完全遮挡的三角面集合。
    /// 返回的数组按照 Layer 八叉树节点顺序分组，每个子数组包含同一实体的连续面。
    pub fn visible_faces(entity: Entity) -> Result<Vec<Scene2DFaceGroup>, Scene2DVisibilityError> {
        let renderables = Layer::world_renderables(entity).map_err(Scene2DVisibilityError::from)?;
        Self::visible_faces_with_renderables(entity, &renderables)
    }

    pub fn visible_faces_with_renderables(
        entity: Entity,
        renderables: &[LayerRenderableBundle],
    ) -> Result<Vec<Scene2DFaceGroup>, Scene2DVisibilityError> {
        if Scene2D::read(entity).is_none() {
            return Err(Scene2DVisibilityError::MissingScene(entity));
        }

        let chunks = Self::chunk_positions(entity).map_err(Scene2DVisibilityError::from)?;
        if chunks.is_empty() {
            return Ok(Vec::new());
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
    pub fn entity(&self) -> Entity {
        self.entity
    }

    pub fn faces(&self) -> &[[Vector3<f32>; 3]] {
        &self.faces
    }

    pub fn into_faces(self) -> Vec<[Vector3<f32>; 3]> {
        self.faces
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Scene2DVisibilityError {
    MissingScene(Entity),
    LayerTraversal(LayerTraversalError),
    LayerSpatial(LayerSpatialError),
}

impl From<LayerTraversalError> for Scene2DVisibilityError {
    fn from(value: LayerTraversalError) -> Self {
        Self::LayerTraversal(value)
    }
}

impl From<LayerSpatialError> for Scene2DVisibilityError {
    fn from(value: LayerSpatialError) -> Self {
        Self::LayerSpatial(value)
    }
}

impl fmt::Display for Scene2DVisibilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scene2DVisibilityError::MissingScene(entity) => {
                write!(f, "实体 {} 未注册 Scene2D 组件", entity.id())
            }
            Scene2DVisibilityError::LayerTraversal(error) => write!(f, "{}", error),
            Scene2DVisibilityError::LayerSpatial(error) => write!(f, "{}", error),
        }
    }
}

impl std::error::Error for Scene2DVisibilityError {}

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

    #[test]
    fn scene2d_requires_layer_dependency() {
        let entity = Entity::new().expect("应能创建实体");
        let _ = entity.unregister_component::<Scene2D>();
        let _ = entity.unregister_component::<Layer>();
        let _ = entity.unregister_component::<Renderable>();
        let _ = Node::detach(entity);
        let _ = entity.unregister_component::<Node>();

        let missing = Scene2D::insert(entity, Scene2D::new("scene2d"));
        assert!(matches!(
            missing,
            Err(crate::game::component::ComponentDependencyError { .. })
        ));

        let _ = entity
            .register_component(Node::new("scene2d_root").expect("应能创建节点"))
            .expect("应能插入 Node");
        let _ = entity
            .register_component(Renderable::new())
            .expect("应能插入 Renderable");
        let _ = entity
            .register_component(Layer::new())
            .expect("应能插入 Layer");
        let inserted =
            Scene2D::insert(entity, Scene2D::new("scene2d")).expect("依赖满足后应能插入");
        assert!(inserted.is_none());
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
        let _ = entity.unregister_component::<Scene2D>();
        let _ = entity.unregister_component::<Layer>();
        let _ = entity.unregister_component::<Renderable>();
        let _ = Node::detach(entity);
        let _ = entity.unregister_component::<Node>();
    }

    #[test]
    fn scene2d_offset_and_scale_configuration() {
        let mut scene = Scene2D::new("scene2d_scale");
        scene.set_offset(Vector2::new(3.0, -2.0));
        scene.set_pixels_per_unit(64.0);

        assert_eq!(scene.offset(), Vector2::new(3.0, -2.0));
        assert!((scene.pixels_per_unit() - 64.0).abs() < f32::EPSILON);

        // 非法数值应被忽略。
        scene.set_pixels_per_unit(-10.0);
        assert!((scene.pixels_per_unit() - 64.0).abs() < f32::EPSILON);
    }

    fn register_layer_scene(entity: Entity, name: &str) {
        let _ = Scene2D::remove(entity);
        let _ = Layer::remove(entity);
        let _ = Renderable::remove(entity);
        let _ = Node::detach(entity);

        let _ = entity
            .register_component(Renderable::new())
            .expect("应能插入 Renderable");
        let _ = entity
            .register_component(Layer::new())
            .expect("应能插入 Layer");

        let inserted =
            Scene2D::insert(entity, Scene2D::new(name)).expect("满足依赖后应能插入 Scene2D");
        assert!(inserted.is_none());
    }

    fn activate_root_chunk(entity: Entity) {
        let update =
            Scene2D::step_lod(entity, &[OctVec::root()], 0).expect("应能推进 Layer 八叉树");
        assert!(
            !update.added.is_empty() || !update.activated.is_empty(),
            "推进 LOD 后应至少激活根节点"
        );
    }

    fn register_shape(entity: Entity, triangles: &[[Vector3<f32>; 3]], translation: Vector3<f32>) {
        let _ = Shape::remove(entity);
        let _ = Transform::remove(entity);
        let _ = Renderable::remove(entity);
        let _ = Node::detach(entity);

        let _ = entity
            .register_component(Renderable::new())
            .expect("应能插入 Renderable");
        let transform =
            Transform::with_components(translation, Vector3::zeros(), Vector3::repeat(1.0));
        let _ = entity
            .register_component(transform)
            .expect("应能插入 Transform");

        let shape = Shape::from_triangles(triangles.to_vec());
        let _ = entity.register_component(shape).expect("应能插入 Shape");
    }

    #[test]
    fn visible_faces_excludes_fully_occluded_triangles() {
        let root = Entity::new().expect("应能创建根实体");
        register_layer_scene(root, "scene2d_visible");
        activate_root_chunk(root);

        let back = Entity::new().expect("应能创建后景实体");
        let front = Entity::new().expect("应能创建前景实体");

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

        register_shape(back, &[far_triangle], Vector3::zeros());
        register_shape(front, &[near_triangle], Vector3::zeros());

        Node::attach(back, root).expect("应能挂载后景");
        Node::attach(front, root).expect("应能挂载前景");

        let visible = Scene2D::visible_faces(root).expect("应能收集可见面片");
        assert_eq!(visible.len(), 1, "只有前景面的实体应可见");
        assert_eq!(visible[0].faces().len(), 1, "仅有一个三角面应可见");
        assert_eq!(visible[0].faces()[0], near_triangle);

        let _ = Node::detach(front);
        let _ = Node::detach(back);
        let _ = Scene2D::remove(root);
        let _ = Layer::remove(root);
        let _ = Renderable::remove(root);

        let _ = Shape::remove(front);
        let _ = Transform::remove(front);
        let _ = Renderable::remove(front);
        let _ = Shape::remove(back);
        let _ = Transform::remove(back);
        let _ = Renderable::remove(back);
    }

    #[test]
    fn visible_faces_keep_partially_occluded_triangles() {
        let root = Entity::new().expect("应能创建根实体");
        register_layer_scene(root, "scene2d_partial");
        activate_root_chunk(root);

        let partial = Entity::new().expect("应能创建部分遮挡实体");
        let ground = Entity::new().expect("应能创建背景实体");

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

        register_shape(partial, &[partial_triangle], Vector3::zeros());
        register_shape(ground, &[ground_triangle], Vector3::zeros());

        Node::attach(partial, root).expect("应能挂载部分遮挡实体");
        Node::attach(ground, root).expect("应能挂载背景实体");

        let visible = Scene2D::visible_faces(root).expect("应能收集可见面片");
        assert_eq!(visible.len(), 2, "部分遮挡情况下应保留两个面组");
        assert_eq!(visible[0].faces().len(), 1);
        assert_eq!(visible[0].faces()[0], partial_triangle);
        assert_eq!(visible[1].faces().len(), 1);
        assert_eq!(visible[1].faces()[0], ground_triangle);

        let _ = Node::detach(ground);
        let _ = Node::detach(partial);
        let _ = Scene2D::remove(root);
        let _ = Layer::remove(root);
        let _ = Renderable::remove(root);

        let _ = Shape::remove(ground);
        let _ = Transform::remove(ground);
        let _ = Renderable::remove(ground);
        let _ = Shape::remove(partial);
        let _ = Transform::remove(partial);
        let _ = Renderable::remove(partial);
    }

    #[test]
    fn visible_faces_ignores_disabled_renderables() {
        let root = Entity::new().expect("应能创建根实体");
        register_layer_scene(root, "scene2d_disabled");
        activate_root_chunk(root);

        let visible_entity = Entity::new().expect("应能创建可见实体");
        let hidden_entity = Entity::new().expect("应能创建隐藏实体");

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

        register_shape(visible_entity, &[visible_triangle], Vector3::zeros());
        register_shape(hidden_entity, &[hidden_triangle], Vector3::zeros());

        Node::attach(visible_entity, root).expect("应能挂载可见实体");
        Node::attach(hidden_entity, root).expect("应能挂载隐藏实体");

        {
            let mut hidden_renderable = hidden_entity
                .get_component_mut::<Renderable>()
                .expect("隐藏实体应有 Renderable");
            hidden_renderable.set_enabled(false);
        }

        let visible = Scene2D::visible_faces(root).expect("应能收集可见面片");
        assert_eq!(visible.len(), 1, "禁用 Renderable 后应仅保留可见实体");
        assert_eq!(visible[0].faces().len(), 1);
        assert_eq!(visible[0].faces()[0], visible_triangle);

        let _ = Node::detach(hidden_entity);
        let _ = Node::detach(visible_entity);
        let _ = Scene2D::remove(root);
        let _ = Layer::remove(root);
        let _ = Renderable::remove(root);

        let _ = Shape::remove(hidden_entity);
        let _ = Transform::remove(hidden_entity);
        let _ = Renderable::remove(hidden_entity);
        let _ = Shape::remove(visible_entity);
        let _ = Transform::remove(visible_entity);
        let _ = Renderable::remove(visible_entity);
    }
}
