use std::sync::atomic::{AtomicU64, Ordering};

use crate::game::component::{
    Component, ComponentDependencyError, ComponentRead, ComponentWrite, node::Node,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Entity {
    id: u64,
}

impl Entity {
    pub fn new() -> anyhow::Result<Self> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let entity = Self { id };
        let node_component = Node::new(format!("entity_{id}"))?;
        entity.register_component(node_component)?;
        Ok(entity)
    }

    pub fn from(id: u64) -> Self {
        Self { id }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn register_component<C: Component>(
        &self,
        component: C,
    ) -> Result<Option<C>, ComponentDependencyError> {
        C::register_dependencies(*self)?;
        C::insert(*self, component)
    }

    pub fn unregister_component<C: Component>(&self) -> Option<C> {
        let removed = C::remove(*self);
        if removed.is_some() {
            C::unregister_dependencies(*self);
        }
        removed
    }

    pub fn get_component<C: Component>(&self) -> Option<ComponentRead<C>> {
        C::read(*self)
    }

    pub fn get_component_mut<C: Component>(&self) -> Option<ComponentWrite<C>> {
        C::write(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::component::node::Node;

    #[test]
    fn entity_new_assigns_unique_ids() {
        let a = Entity::new().expect("应能创建实体");
        let b = Entity::new().expect("应能创建实体");
        assert_ne!(a.id(), b.id(), "不同实体应有不同 id");
    }

    #[test]
    fn entity_new_auto_registers_node_component() {
        let entity = Entity::new().expect("应能创建实体");
        let node = entity
            .get_component::<Node>()
            .expect("Entity::new 应自动挂载 Node");
        assert_eq!(node.name(), format!("entity_{}", entity.id()));
    }

    #[test]
    fn entity_from_preserves_id() {
        let entity = Entity::from(123);
        assert_eq!(entity.id(), 123);
    }
}
