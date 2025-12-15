use super::node::Node;
use super::{component, component_impl};

#[component(Node)]
#[derive(Debug, Clone)]
pub struct Renderable {
    enabled: bool,
}

#[component_impl]
impl Renderable {
    #[allow(dead_code)]
    #[default(Node::new(format!("entity_{}", entity.id())).expect("auto node name valid"))]
    fn ensure_defaults(_node: Node) -> Self {
        Self::new()
    }
    /// 创建一个默认启用的可渲染组件。
    pub fn new() -> Self {
        Self { enabled: true }
    }

    /// 返回当前组件是否参与渲染。
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// 设置组件是否参与渲染。
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::entity::Entity;

    fn prepare_node(entity: &Entity, name: &str) {
        let _ = entity
            .register_component(Node::new(name).expect("应能创建节点"))
            .expect("应能注册 Node 组件");
    }

    #[test]
    fn renderable_requires_node_dependency() {
        let entity = Entity::new().expect("应能创建实体");
        entity.unregister_component::<Node>();

        let missing = entity.register_component(Renderable::new());
        assert!(matches!(
            missing,
            Err(crate::game::component::ComponentDependencyError { .. })
        ));

        prepare_node(&entity, "renderable_node");
        let inserted = entity
            .register_component(Renderable::new())
            .expect("满足依赖后应能插入 Renderable");
        assert!(inserted.is_none());

        let _ = entity.unregister_component::<Renderable>();
    }

    #[test]
    fn toggling_enabled_state_updates_renderable() {
        let entity = Entity::new().expect("应能创建实体");
        prepare_node(&entity, "toggle_node");

        entity
            .register_component(Renderable::new())
            .expect("应能插入 Renderable");

        {
            let mut renderable = entity
                .get_component_mut::<Renderable>()
                .expect("应能获得 Renderable 的可写引用");
            renderable.set_enabled(false);
        }

        let renderable = entity
            .get_component::<Renderable>()
            .expect("应能读取 Renderable 组件");
        assert!(!renderable.is_enabled());

        drop(renderable);
        let _ = entity.unregister_component::<Renderable>();
    }
}
