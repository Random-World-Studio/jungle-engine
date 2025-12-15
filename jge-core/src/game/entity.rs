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

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn register_component<C: Component>(
        &self,
        component: C,
    ) -> Result<Option<C>, ComponentDependencyError> {
        C::insert(*self, component)
    }

    pub fn unregister_component<C: Component>(&self) -> Option<C> {
        C::remove(*self)
    }

    pub fn get_component<C: Component>(&self) -> Option<ComponentRead<C>> {
        C::read(*self)
    }

    pub fn get_component_mut<C: Component>(&self) -> Option<ComponentWrite<C>> {
        C::write(*self)
    }
}
