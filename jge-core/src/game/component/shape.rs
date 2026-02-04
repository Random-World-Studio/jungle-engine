use super::transform::Transform;
use super::{component, component_impl};
use crate::game::entity::Entity;
use nalgebra::Vector3;

/// 网格/形状组件（三角形网格）。
///
/// `Shape` 用“顶点数组 + 三角面索引”描述几何：
/// - `vertices`：局部空间顶点
/// - `faces`：每个三角面是 `[usize; 3]` 索引
///
/// 渲染时通常会结合实体的 [`Transform`](jge_core::game::component::transform::Transform) 把顶点变换到世界空间。
///
/// 依赖：该组件依赖 `Transform`。
///
/// # 示例
///
/// ```no_run
/// use jge_core::game::{component::shape::Shape, entity::Entity};
/// use nalgebra::Vector3;
///
/// # fn main() -> anyhow::Result<()> {
/// let rt = tokio::runtime::Runtime::new()?;
/// rt.block_on(async {
///     let e = Entity::new().await?;
///     e.register_component(Shape::from_triangles(vec![[
///         Vector3::new(0.0, 0.0, 0.0),
///         Vector3::new(1.0, 0.0, 0.0),
///         Vector3::new(0.0, 1.0, 0.0),
///     ]]))
///     .await?;
///     Ok::<(), anyhow::Error>(())
/// })?;
/// Ok(())
/// # }
/// ```
#[component(Transform)]
#[derive(Debug, Clone)]
pub struct Shape {
    entity_id: Option<Entity>,
    vertices: Vec<Vector3<f32>>,
    faces: Vec<[usize; 3]>,
}

#[component_impl]
impl Shape {
    /// 使用顶点及索引构造形状。顶点坐标为相对 Transform 组件的位置偏移，
    /// 三角面通过索引列表明确指定顺序，便于控制背面剔除方向。
    #[default(Vec::<Vector3<f32>>::new(), Vec::<[usize; 3]>::new())]
    pub fn new(vertices: Vec<Vector3<f32>>, faces: Vec<[usize; 3]>) -> Self {
        debug_assert!(faces.iter().all(|[a, b, c]| {
            *a < vertices.len() && *b < vertices.len() && *c < vertices.len()
        }));
        Self {
            entity_id: None,
            vertices,
            faces,
        }
    }

    /// 直接使用三角面顶点构造形状，每个面都会复制三份顶点数据。
    pub fn from_triangles(triangles: Vec<[Vector3<f32>; 3]>) -> Self {
        let mut vertices = Vec::with_capacity(triangles.len() * 3);
        let mut faces = Vec::with_capacity(triangles.len());
        for triangle in triangles {
            let base = vertices.len();
            vertices.push(triangle[0]);
            vertices.push(triangle[1]);
            vertices.push(triangle[2]);
            faces.push([base, base + 1, base + 2]);
        }
        Self::new(vertices, faces)
    }

    /// 返回顶点数组的只读视图。
    pub fn vertices(&self) -> &[Vector3<f32>] {
        &self.vertices
    }

    /// 返回顶点数组的可写视图。
    pub fn vertices_mut(&mut self) -> &mut [Vector3<f32>] {
        &mut self.vertices
    }

    /// 返回三角面索引列表。
    pub fn faces(&self) -> &[[usize; 3]] {
        &self.faces
    }

    /// 返回三角面索引列表的可写视图。
    pub fn faces_mut(&mut self) -> &mut [[usize; 3]] {
        &mut self.faces
    }

    /// 返回形状包含的三角面数量，不足三个顶点时为 0。
    pub fn triangle_count(&self) -> usize {
        self.faces.len()
    }

    /// 按顶点数组顺序迭代三角面。
    pub fn triangles(&self) -> Triangles<'_> {
        Triangles {
            vertices: &self.vertices,
            faces: &self.faces,
            index: 0,
        }
    }
}

/// `Shape::triangles()` 的迭代器。
///
/// 每次迭代返回一个三元组：`(&v0, &v1, &v2)`。
pub struct Triangles<'a> {
    vertices: &'a [Vector3<f32>],
    faces: &'a [[usize; 3]],
    index: usize,
}

impl<'a> Iterator for Triangles<'a> {
    type Item = (&'a Vector3<f32>, &'a Vector3<f32>, &'a Vector3<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.faces.len() {
            return None;
        }
        let [a, b, c] = self.faces[self.index];
        debug_assert!(
            a < self.vertices.len() && b < self.vertices.len() && c < self.vertices.len()
        );
        let tri = (&self.vertices[a], &self.vertices[b], &self.vertices[c]);
        self.index += 1;
        Some(tri)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{
        component::{node::Node, renderable::Renderable},
        entity::Entity,
    };
    use nalgebra::Vector3;

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

    async fn ensure_transform(entity: &Entity, name: &str) {
        let _ = entity.unregister_component::<Shape>().await;
        let _ = entity.unregister_component::<Transform>().await;
        let _ = entity.unregister_component::<Renderable>().await;
        detach_node(*entity).await;
        let _ = entity.unregister_component::<Node>().await;

        let _ = entity
            .register_component(Node::new(name).expect("应能创建节点"))
            .await
            .expect("应能插入 Node");
        let _ = entity
            .register_component(Renderable::new())
            .await
            .expect("应能插入 Renderable");
        let _ = entity
            .register_component(Transform::new())
            .await
            .expect("应能插入 Transform");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn shape_requires_transform_dependency() {
        let entity = Entity::new().await.expect("应能创建实体");
        let _ = entity.unregister_component::<Shape>().await;
        let _ = entity.unregister_component::<Transform>().await;
        let _ = entity.unregister_component::<Renderable>().await;
        detach_node(entity).await;
        let _ = entity.unregister_component::<Node>().await;

        let inserted = entity
            .register_component(Shape::new(Vec::new(), Vec::new()))
            .await
            .expect("缺少 Transform 时应自动注册依赖");
        assert!(inserted.is_none());

        assert!(
            entity.get_component::<Transform>().await.is_some(),
            "Transform 应被自动注册"
        );
        assert!(
            entity.get_component::<Renderable>().await.is_some(),
            "Renderable 应被注册"
        );
        assert!(
            entity.get_component::<Node>().await.is_some(),
            "Node 应被注册"
        );

        let previous = entity
            .register_component(Shape::new(Vec::new(), Vec::new()))
            .await
            .expect("重复插入应返回旧的 Shape");
        assert!(previous.is_some());

        let _ = entity.unregister_component::<Shape>().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn triangles_iterates_triplets() {
        let entity = Entity::new().await.expect("应能创建实体");
        ensure_transform(&entity, "shape_triangles").await;

        let shape = Shape::new(
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            vec![[0, 1, 2], [1, 2, 3]],
        );

        entity
            .register_component(shape.clone())
            .await
            .expect("应能插入 Shape");

        let stored = entity
            .get_component::<Shape>()
            .await
            .expect("应能读取 Shape");

        let count = stored.triangle_count();
        assert_eq!(count, 2);

        let tris: Vec<_> = stored.triangles().collect();
        assert_eq!(tris.len(), 2);
        assert_eq!(*tris[0].0, Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(*tris[0].1, Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(*tris[0].2, Vector3::new(0.0, 1.0, 0.0));
        assert_eq!(*tris[1].0, Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(*tris[1].1, Vector3::new(0.0, 1.0, 0.0));
        assert_eq!(*tris[1].2, Vector3::new(0.0, 0.0, 1.0));

        drop(stored);
        let _ = entity.unregister_component::<Shape>().await;
    }
}
