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
/// ## 销毁语义（推荐约定）
///
/// 在 Jungle Engine 中，`Entity` 是一个轻量句柄；“销毁实体”通常指把它上面**显式注册**的组件卸载掉。
///
/// - 你只需要对每个显式组件调用 [`Entity::unregister_component`](Self::unregister_component)。
/// - 依赖关系：`unregister_component` 在卸载成功后会调用组件的 `unregister_dependencies` 钩子；
///   具体是否会卸载依赖组件，取决于组件的实现策略。
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
/// let rt = tokio::runtime::Runtime::new()?;
/// rt.block_on(async {
///     let e = Entity::new().await?;
///     e.register_component(Transform::new()).await?;
///     e.register_component(Layer::new()).await?;
///
///     if let Some(mut t) = e.get_component_mut::<Transform>().await {
///         t.set_position([1.0, 2.0, 3.0].into());
///     }
///
///     Ok::<(), anyhow::Error>(())
/// })?;
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
    pub async fn new() -> anyhow::Result<Self> {
        let id = EntityId::new();
        Self::new_with_id(id).await
    }

    /// 使用指定的 [`EntityId`] 创建实体。
    ///
    /// 一般用于：从存档/网络同步恢复实体 ID。
    /// 同样会自动注册默认 [`Node`]。
    pub async fn new_with_id(id: EntityId) -> anyhow::Result<Self> {
        let entity = Self { id };
        let node_component = Node::new(format!("entity_{}", entity.id().short()))?;
        entity.register_component(node_component).await?;
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
    pub async fn register_component<C: Component>(
        &self,
        component: C,
    ) -> Result<Option<C>, ComponentDependencyError> {
        C::register_dependencies(*self).await?;
        C::insert(*self, component).await
    }

    /// 卸载一个组件。
    ///
    /// 若卸载成功，会同时按组件声明解除其依赖关系。
    pub async fn unregister_component<C: Component>(&self) -> Option<C> {
        let removed = C::remove(*self).await;
        if removed.is_some() {
            C::unregister_dependencies(*self).await;
        }
        removed
    }

    /// 读取组件（共享借用）。
    ///
    /// 返回的 [`ComponentRead`] 是一个读 guard，生命周期内会保持一致性。
    ///
    /// 该 guard 是 `Send`，因此在需要 `Future: Send` 的 async 任务里也可以跨 `.await` 存活。
    /// 但仍建议尽量缩短持锁时间，并避免在持有 guard 时 `.await` 可能回到 ECS/节点树的 Future（避免死锁）。
    pub async fn get_component<C: Component>(&self) -> Option<ComponentRead<C>> {
        C::read(*self).await
    }

    /// 读取组件（可变借用）。
    ///
    /// 返回的 [`ComponentWrite`] 是一个写 guard。
    ///
    /// 该 guard 是 `Send`，因此在需要 `Future: Send` 的 async 任务里也可以跨 `.await` 存活。
    /// 但在持有写 guard 时执行耗时 `.await` 会阻塞其它读写；同时请避免在持有 guard 时 `.await` 可能回到 ECS/节点树的 Future（避免死锁）。
    pub async fn get_component_mut<C: Component>(&self) -> Option<ComponentWrite<C>> {
        C::write(*self).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::component::node::Node;

    #[tokio::test(flavor = "multi_thread")]
    async fn entity_new_assigns_unique_ids() {
        let a = Entity::new().await.expect("应能创建实体");
        let b = Entity::new().await.expect("应能创建实体");
        assert_ne!(a.id(), b.id(), "不同实体应有不同 id");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn entity_new_auto_registers_node_component() {
        let entity = Entity::new().await.expect("应能创建实体");
        let node = entity
            .get_component::<Node>()
            .await
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
