use super::{component, component_impl, shape::Shape};
use crate::game::entity::Entity;
use crate::resource::{Resource, ResourceHandle};
use nalgebra::Vector2;

/// 材质贴图区域（三角形 UV）。
///
/// 用三个 `Vector2<f32>` 表示一个三角形的 UV 坐标。
/// 常见用途：同一张贴图里裁剪/拼图（atlas）。
pub type MaterialPatch = [Vector2<f32>; 3];

/// 材质组件。
///
/// - `resource` 通常指向一张纹理或材质数据（由渲染路径解释）。
/// - `regions` 可用于为同一 `Shape` 提供多个三角形区域的 UV 映射。
///
/// 依赖：该组件依赖 [`Shape`](jge_core::game::component::shape::Shape)。
///
/// # 示例
///
/// ```no_run
/// use jge_core::game::{component::material::Material, entity::Entity};
/// use jge_core::resource::Resource;
///
/// # fn main() -> anyhow::Result<()> {
/// let e = Entity::new()?;
/// // resource 的具体内容（纹理/材质数据）由渲染路径解释。
/// let resource = Resource::from_memory(vec![1, 2, 3]);
/// e.register_component(Material::new(resource, Vec::new()))?;
/// Ok(())
/// # }
/// ```
#[component(Shape)]
#[derive(Debug, Clone)]
pub struct Material {
    entity_id: Option<Entity>,
    resource: ResourceHandle,
    regions: Vec<MaterialPatch>,
}

#[component_impl]
impl Material {
    /// 创建材质组件。
    ///
    /// - `resource`：材质资源句柄
    /// - `regions`：UV 区域列表（可为空）
    #[default(Resource::from_memory(Vec::new()), Vec::new())]
    pub fn new(resource: ResourceHandle, regions: Vec<MaterialPatch>) -> Self {
        Self {
            entity_id: None,
            resource,
            regions,
        }
    }

    /// 获取资源句柄（克隆 `Arc`）。
    pub fn resource(&self) -> ResourceHandle {
        self.resource.clone()
    }

    /// 替换材质资源。
    pub fn set_resource(&mut self, resource: ResourceHandle) {
        self.resource = resource;
    }

    /// 获取 UV 区域列表（只读）。
    pub fn regions(&self) -> &[MaterialPatch] {
        &self.regions
    }

    /// 获取 UV 区域列表（可写）。
    pub fn regions_mut(&mut self) -> &mut [MaterialPatch] {
        self.regions.as_mut_slice()
    }

    /// 替换 UV 区域列表。
    pub fn set_regions(&mut self, regions: Vec<MaterialPatch>) {
        self.regions = regions;
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

    async fn prepare_entity(name: &str) -> Entity {
        let entity = Entity::new().expect("应能创建实体");
        let _ = entity.unregister_component::<Material>();
        let _ = entity.unregister_component::<Shape>();
        let _ = entity.unregister_component::<Transform>();
        let _ = entity.unregister_component::<Renderable>();
        detach_node(entity).await;
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
        Resource::from_memory(vec![1, 2, 3])
    }

    #[tokio::test]
    async fn material_requires_shape_dependency() {
        let entity = Entity::new().expect("应能创建实体");
        let resource = mock_resource();
        let inserted = entity
            .register_component(Material::new(resource.clone(), Vec::new()))
            .expect("缺少 Shape 时应自动注册依赖");
        assert!(inserted.is_none());

        assert!(
            entity.get_component::<Shape>().is_some(),
            "Shape 应被自动注册"
        );
        assert!(
            entity.get_component::<Transform>().is_some(),
            "Transform 应作为 Shape 依赖被注册"
        );
        assert!(
            entity.get_component::<Renderable>().is_some(),
            "Renderable 应被注册"
        );
        assert!(entity.get_component::<Node>().is_some(), "Node 应被注册");

        let entity = prepare_entity("material_dependency").await;
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
