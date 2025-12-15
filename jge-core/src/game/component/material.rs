use super::{component, component_impl, shape::Shape};
use crate::resource::{Resource, ResourceHandle};
use nalgebra::Vector2;
use parking_lot::RwLock;
use std::sync::Arc;

pub type MaterialPatch = [Vector2<f32>; 3];

#[component(Shape)]
#[derive(Debug, Clone)]
pub struct Material {
    resource: ResourceHandle,
    regions: Vec<MaterialPatch>,
}

#[component_impl]
impl Material {
    #[allow(dead_code)]
    #[default(Shape::from_triangles(Vec::new()))]
    fn ensure_defaults(_shape: Shape) -> Self {
        Self::new(Self::placeholder_resource(), Vec::new())
    }
    pub fn new(resource: ResourceHandle, regions: Vec<MaterialPatch>) -> Self {
        Self { resource, regions }
    }

    pub fn resource(&self) -> ResourceHandle {
        self.resource.clone()
    }

    pub fn set_resource(&mut self, resource: ResourceHandle) {
        self.resource = resource;
    }

    pub fn regions(&self) -> &[MaterialPatch] {
        &self.regions
    }

    pub fn regions_mut(&mut self) -> &mut [MaterialPatch] {
        self.regions.as_mut_slice()
    }

    pub fn set_regions(&mut self, regions: Vec<MaterialPatch>) {
        self.regions = regions;
    }

    fn placeholder_resource() -> ResourceHandle {
        Arc::new(RwLock::new(Resource::from_memory(Vec::new())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::component::{
        node::Node, renderable::Renderable, shape::Shape, transform::Transform,
    };
    use crate::game::entity::Entity;
    use crate::resource::Resource;
    use nalgebra::{Vector2, Vector3};
    use parking_lot::RwLock;
    use std::sync::Arc;

    fn prepare_entity(name: &str) -> Entity {
        let entity = Entity::new().expect("应能创建实体");
        let _ = entity.unregister_component::<Material>();
        let _ = entity.unregister_component::<Shape>();
        let _ = entity.unregister_component::<Transform>();
        let _ = entity.unregister_component::<Renderable>();
        let _ = Node::detach(entity);
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
        let shape = Shape::from_triangles(vec![[
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ]]);
        let _ = entity.register_component(shape).expect("应能插入 Shape");
        entity
    }

    fn mock_resource() -> ResourceHandle {
        Arc::new(RwLock::new(Resource::from_memory(vec![1, 2, 3])))
    }

    #[test]
    fn material_requires_shape_dependency() {
        let entity = Entity::new().expect("应能创建实体");
        let resource = mock_resource();
        let missing = entity.register_component(Material::new(resource.clone(), Vec::new()));
        assert!(missing.is_err(), "缺少 Shape 依赖应当失败");

        let entity = prepare_entity("material_dependency");
        let resource = mock_resource();
        let material = Material::new(
            resource.clone(),
            vec![[
                Vector2::new(0.0, 0.0),
                Vector2::new(1.0, 0.0),
                Vector2::new(0.0, 1.0),
            ]],
        );
        let previous = entity
            .register_component(material)
            .expect("满足依赖应能插入 Material");
        assert!(previous.is_none());

        let stored = entity
            .get_component::<Material>()
            .expect("应能读取 Material");
        assert_eq!(stored.regions().len(), 1);
        drop(stored);
        let mut guard = entity
            .get_component_mut::<Material>()
            .expect("应能写入 Material");
        guard.set_regions(vec![[
            Vector2::new(0.0, 0.0),
            Vector2::new(0.5, 0.0),
            Vector2::new(0.0, 0.5),
        ]]);
        drop(guard);
        assert_eq!(
            entity.get_component::<Material>().unwrap().regions().len(),
            1
        );
    }
}
