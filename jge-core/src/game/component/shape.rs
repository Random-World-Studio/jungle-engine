use super::transform::Transform;
use super::{component, component_impl};
use crate::game::entity::Entity;
use nalgebra::Vector3;

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

    fn detach_node(entity: Entity) {
        if let Some(mut node) = entity.get_component_mut::<Node>() {
            let _ = node.detach();
        }
    }

    fn ensure_transform(entity: &Entity, name: &str) {
        let _ = entity.unregister_component::<Shape>();
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
    fn shape_requires_transform_dependency() {
        let entity = Entity::new().expect("应能创建实体");
        let _ = entity.unregister_component::<Shape>();
        let _ = entity.unregister_component::<Transform>();
        let _ = entity.unregister_component::<Renderable>();
        detach_node(entity);
        let _ = entity.unregister_component::<Node>();

        let inserted = entity
            .register_component(Shape::new(Vec::new(), Vec::new()))
            .expect("缺少 Transform 时应自动注册依赖");
        assert!(inserted.is_none());

        assert!(
            entity.get_component::<Transform>().is_some(),
            "Transform 应被自动注册"
        );
        assert!(
            entity.get_component::<Renderable>().is_some(),
            "Renderable 应被注册"
        );
        assert!(entity.get_component::<Node>().is_some(), "Node 应被注册");

        let previous = entity
            .register_component(Shape::new(Vec::new(), Vec::new()))
            .expect("重复插入应返回旧的 Shape");
        assert!(previous.is_some());

        let _ = entity.unregister_component::<Shape>();
    }

    #[test]
    fn triangles_iterates_triplets() {
        let entity = Entity::new().expect("应能创建实体");
        ensure_transform(&entity, "shape_triangles");

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
            .expect("应能插入 Shape");

        let stored = entity.get_component::<Shape>().expect("应能读取 Shape");

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
        let _ = entity.unregister_component::<Shape>();
    }
}
