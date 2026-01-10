use uuid::Uuid;

use crate::game::component::{
    Component, ComponentDependencyError, ComponentRead, ComponentWrite, node::Node,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityId(Uuid);

impl EntityId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// 生成一个更短、更适合作为默认节点名的字符串（前 8 个十六进制字符）。
    pub fn short(self) -> String {
        let s = self.0.simple().to_string();
        s.chars().take(8).collect()
    }
}

impl std::fmt::Display for EntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Entity {
    id: EntityId,
}

impl Entity {
    pub fn new() -> anyhow::Result<Self> {
        let id = EntityId::new();
        Self::new_with_id(id)
    }

    pub fn new_with_id(id: EntityId) -> anyhow::Result<Self> {
        let entity = Self { id };
        let node_component = Node::new(format!("entity_{}", entity.id().short()))?;
        entity.register_component(node_component)?;
        Ok(entity)
    }

    pub fn from(id: EntityId) -> Self {
        Self { id }
    }

    pub fn id(&self) -> EntityId {
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
        assert_eq!(node.name(), format!("entity_{}", entity.id().short()));
    }

    #[test]
    fn entity_from_preserves_id() {
        let id = EntityId::new();
        let entity = Entity::from(id);
        assert_eq!(entity.id(), id);
    }
}
