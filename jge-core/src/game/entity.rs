use uuid::Uuid;

use crate::game::component::{
    Component, ComponentDependencyError, ComponentRead, ComponentWrite, node::Node,
};

/// 实体唯一标识。
///
/// 通常你不需要手动创建 `EntityId`：直接使用 [`Entity::new`](Entity::new) 创建实体即可。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityId(Uuid);

impl EntityId {
    /// 创建一个新的随机实体 ID。
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

/// 游戏世界中的实体句柄。
///
/// `Entity` 是组件的“宿主”。你通过它来注册/卸载组件、以及读取/修改组件数据。
///
/// 约定：
/// - [`Entity::new`](Self::new) 创建实体时，会自动注册一个 [`Node`]。
///   这让实体可以加入节点树（父子层级、Layer 子树遍历等）。
///
/// # 示例
///
/// ```no_run
/// use jge_core::game::{
///     component::{transform::Transform, layer::Layer},
///     entity::Entity,
/// };
///
/// # fn main() -> anyhow::Result<()> {
/// let e = Entity::new()?;
/// e.register_component(Transform::new())?;
/// e.register_component(Layer::new())?;
///
/// if let Some(mut t) = e.get_component_mut::<Transform>() {
///     t.set_position([1.0, 2.0, 3.0].into());
/// }
/// Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Entity {
    id: EntityId,
}

impl Entity {
    /// 创建一个新实体。
    ///
    /// 该函数会自动注册一个默认 [`Node`]。
    pub fn new() -> anyhow::Result<Self> {
        let id = EntityId::new();
        Self::new_with_id(id)
    }

    /// 使用指定的 [`EntityId`] 创建实体。
    ///
    /// 一般用于：从存档/网络同步恢复实体 ID。
    /// 同样会自动注册默认 [`Node`]。
    pub fn new_with_id(id: EntityId) -> anyhow::Result<Self> {
        let entity = Self { id };
        let node_component = Node::new(format!("entity_{}", entity.id().short()))?;
        entity.register_component(node_component)?;
        Ok(entity)
    }

    /// 从 ID 构造一个实体句柄。
    ///
    /// 注意：该函数不会自动注册任何组件。
    /// 如果你需要一个“全新实体”，请使用 [`Entity::new`](Self::new)。
    pub fn from(id: EntityId) -> Self {
        Self { id }
    }

    /// 获取实体 ID。
    pub fn id(&self) -> EntityId {
        self.id
    }

    /// 注册一个组件到实体上。
    ///
    /// - 如果组件已存在，会返回旧值（`Some(old)`）。
    /// - 如果组件声明了依赖关系，会自动注册其依赖组件。
    pub fn register_component<C: Component>(
        &self,
        component: C,
    ) -> Result<Option<C>, ComponentDependencyError> {
        C::register_dependencies(*self)?;
        C::insert(*self, component)
    }

    /// 卸载一个组件。
    ///
    /// 若卸载成功，会同时按组件声明解除其依赖关系。
    pub fn unregister_component<C: Component>(&self) -> Option<C> {
        let removed = C::remove(*self);
        if removed.is_some() {
            C::unregister_dependencies(*self);
        }
        removed
    }

    /// 读取组件（共享借用）。
    ///
    /// 返回的 [`ComponentRead`] 是一个读 guard，生命周期内会保持一致性。
    pub fn get_component<C: Component>(&self) -> Option<ComponentRead<C>> {
        C::read(*self)
    }

    /// 读取组件（可变借用）。
    ///
    /// 返回的 [`ComponentWrite`] 是一个写 guard。
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
