use super::node::Node;
use super::{Component, ComponentDependencyError, ComponentStorage, component_impl};
use crate::game::entity::Entity;
use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::OnceLock;

static RENDERABLE_STORAGE: OnceLock<ComponentStorage<Renderable>> = OnceLock::new();

#[derive(Debug, Clone)]
/// 可渲染标记组件。
///
/// 挂载该组件的实体会被渲染系统视为“可参与渲染的对象”。
///
/// - 你通常会同时挂载 `Shape` 与 `Transform`（以及可选的 `Material`）。
/// - 想临时隐藏一个实体（以及它的整棵子树）时，可用 [`set_enabled`](Self::set_enabled)。
///   该调用会返回一个 future，需要 `.await`。
/// - 当重新显示时，引擎会使用“可见性缓存栈”恢复子树中每个节点原先的设置：
///   - 关闭时对子树内每个 `Renderable` 执行一次 `push`
///   - 恢复时对子树内每个 `Renderable` 执行一次 `pop`
///
/// # 示例
///
/// ```no_run
/// use jge_core::game::{
///     component::{renderable::Renderable, transform::Transform},
///     entity::Entity,
/// };
///
/// # fn main() -> anyhow::Result<()> {
/// let rt = tokio::runtime::Runtime::new()?;
/// rt.block_on(async {
///     let e = Entity::new().await?;
///     e.register_component(Renderable::new()).await?;
///     e.register_component(Transform::new()).await?;
///
///     // set_enabled 返回 future；像 Node::attach 一样，建议先取出 future 再 await。
///     let hide_future = {
///         let mut renderable = e.get_component_mut::<Renderable>().await.unwrap();
///         renderable.set_enabled(false)
///     };
///     hide_future.await;
///
///     Ok::<(), anyhow::Error>(())
/// })?;
/// Ok(())
/// # }
/// ```
pub struct Renderable {
    entity_id: Option<Entity>,
    /// 用户配置的“期望可见性”（初始值）。
    configured_enabled: bool,
    /// 可见性缓存栈：用于“父级 Renderable 隐藏/显示子树”时恢复子树节点之前的设置。
    ///
    /// 规则：
    /// - 当某个祖先 Renderable 被设置为不可见时，会对其子树内每个 Renderable：
    ///   - `push(configured_enabled)`
    ///   - 临时强制 `configured_enabled = false`
    /// - 当祖先 Renderable 被重新设置为可见时：
    ///   - `pop()` 恢复；若栈为空则跟随最近 Renderable 父节点的可见性。
    enabled_cache: Vec<bool>,
    /// 实际可见性（会被节点可达性覆盖）。
    enabled: bool,
    /// 是否从引擎根节点可达（由节点挂载/卸载传播维护）。
    reachable: bool,
}

#[async_trait]
impl Component for Renderable {
    fn storage() -> &'static ComponentStorage<Self> {
        RENDERABLE_STORAGE.get_or_init(ComponentStorage::new)
    }

    async fn register_dependencies(entity: Entity) -> Result<(), ComponentDependencyError> {
        if entity.get_component::<Node>().await.is_none() {
            let component = Node::__jge_component_default(entity)?;
            let _ = entity.register_component(component).await?;
        }
        Ok(())
    }

    async fn attach_entity(&mut self, entity: Entity) {
        self.entity_id = Some(entity);

        // Renderable 的实际可见性由“是否对引擎根可达”驱动。
        // 这一步用于覆盖“先建树、后注册 Renderable”的场景。
        let reachable = crate::game::reachability::is_reachable_from_engine_root(entity).await;
        self.set_reachable(reachable);
    }

    async fn detach_entity(&mut self) {
        self.entity_id = None;
        self.set_reachable(false);
    }
}

#[component_impl]
impl Renderable {
    /// 创建一个默认“期望可见”的可渲染组件。
    ///
    /// 注意：Renderable 在初始化后会先处于不可见状态；只有当节点树挂载使其从引擎根节点可达时，
    /// 才会将实际可见性切换为此处的初始设置。
    #[default()]
    pub fn new() -> Self {
        Self {
            entity_id: None,
            configured_enabled: true,
            enabled_cache: Vec::new(),
            enabled: false,
            reachable: false,
        }
    }

    /// 返回当前组件的**实际**可见性（是否参与渲染）。
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// 返回当前组件的“期望可见性”。
    ///
    /// 该值会在 `reachable == true` 时成为实际可见性的候选；
    /// 也会被“子树隐藏/恢复”逻辑用于缓存与恢复。
    pub fn configured_enabled(&self) -> bool {
        self.configured_enabled
    }

    /// 设置组件的“期望可见性”。
    ///
    /// 语义：
    /// - 该设置会影响当前实体的可见性；
    /// - 当设置为不可见时，会把**整个子树**内所有 `Renderable` 临时强制为不可见（并把每个节点
    ///   的可见性压入缓存栈）；
    /// - 当设置为可见时，会对**整个子树**执行恢复：对每个节点从缓存栈 pop；栈为空则跟随最近
    ///   `Renderable` 父节点的可见性。
    ///
    /// 该函数返回一个 future：调用方应当像 `Node::attach/detach` 一样先取出 future，
    /// 再在 guard 释放后 `.await`，避免把组件写锁跨 `await`。
    pub fn set_enabled(
        &mut self,
        enabled: bool,
    ) -> impl std::future::Future<Output = ()> + Send + 'static + use<> {
        self.set_enabled_local(enabled);

        let root = self.entity_id;
        let root_configured_enabled = enabled;

        async move {
            let Some(root) = root else {
                return;
            };

            if root_configured_enabled {
                restore_descendant_visibility_from_cache_or_inherit(root, root_configured_enabled)
                    .await;
            } else {
                cache_and_force_descendant_visibility_hidden(root).await;
            }
        }
    }

    fn set_enabled_local(&mut self, enabled: bool) {
        self.configured_enabled = enabled;
        self.enabled = if self.reachable { enabled } else { false };
    }

    pub(crate) fn set_reachable(&mut self, reachable: bool) {
        self.reachable = reachable;
        self.enabled = if reachable {
            self.configured_enabled
        } else {
            false
        };
    }
}

pub(crate) async fn nearest_renderable_ancestor_configured_enabled(entity: Entity) -> Option<bool> {
    let mut visited = HashSet::new();
    let mut current = Some(entity);

    while let Some(e) = current {
        if !visited.insert(e) {
            return None;
        }

        if let Some(renderable) = e.get_component::<Renderable>().await {
            return Some(renderable.configured_enabled());
        }

        let parent = e.get_component::<Node>().await.map(|n| n.parent());
        current = parent.flatten();
    }

    None
}

pub(crate) async fn force_subtree_hidden_without_cache(root: Entity) {
    let mut stack = vec![root];
    let mut visited = HashSet::new();

    while let Some(entity) = stack.pop() {
        if !visited.insert(entity) {
            continue;
        }

        if let Some(mut renderable) = entity.get_component_mut::<Renderable>().await {
            renderable.set_enabled_local(false);
        }

        let children = match entity.get_component::<Node>().await {
            Some(node) => node.children().to_vec(),
            None => Vec::new(),
        };
        stack.extend(children);
    }
}

async fn cache_and_force_descendant_visibility_hidden(root: Entity) {
    let children = match root.get_component::<Node>().await {
        Some(node) => node.children().to_vec(),
        None => Vec::new(),
    };

    let mut stack = children;
    let mut visited = HashSet::new();

    while let Some(entity) = stack.pop() {
        if !visited.insert(entity) {
            continue;
        }

        if let Some(mut renderable) = entity.get_component_mut::<Renderable>().await {
            let configured = renderable.configured_enabled;
            renderable.enabled_cache.push(configured);
            renderable.set_enabled_local(false);
        }

        let children = match entity.get_component::<Node>().await {
            Some(node) => node.children().to_vec(),
            None => Vec::new(),
        };
        stack.extend(children);
    }
}

async fn restore_descendant_visibility_from_cache_or_inherit(root: Entity, inherited: bool) {
    let children = match root.get_component::<Node>().await {
        Some(node) => node.children().to_vec(),
        None => Vec::new(),
    };

    let mut stack: Vec<(Entity, bool)> = children.into_iter().map(|c| (c, inherited)).collect();
    let mut visited = HashSet::new();

    while let Some((entity, inherited)) = stack.pop() {
        if !visited.insert(entity) {
            continue;
        }

        let mut next_inherited = inherited;

        if let Some(mut renderable) = entity.get_component_mut::<Renderable>().await {
            let restored = renderable.enabled_cache.pop().unwrap_or(inherited);
            renderable.set_enabled_local(restored);
            next_inherited = renderable.configured_enabled;
        }

        let children = match entity.get_component::<Node>().await {
            Some(node) => node.children().to_vec(),
            None => Vec::new(),
        };

        stack.extend(children.into_iter().map(|c| (c, next_inherited)));
    }
}

impl Default for Renderable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::entity::Entity;

    async fn prepare_node(entity: &Entity, name: &str) {
        let _ = entity
            .register_component(Node::new(name).expect("应能创建节点"))
            .await
            .expect("应能注册 Node 组件");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn renderable_requires_node_dependency() {
        let entity = Entity::new().await.expect("应能创建实体");
        entity.unregister_component::<Node>().await;

        let inserted = entity
            .register_component(Renderable::new())
            .await
            .expect("缺少 Node 时应自动注册依赖");
        assert!(inserted.is_none());

        assert!(
            entity.get_component::<Node>().await.is_some(),
            "Node 应被自动注册"
        );

        prepare_node(&entity, "renderable_node").await;
        let previous = entity
            .register_component(Renderable::new())
            .await
            .expect("重复插入应返回旧的 Renderable");
        assert!(previous.is_some());

        let _ = entity.unregister_component::<Renderable>().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn toggling_enabled_state_updates_renderable() {
        let entity = Entity::new().await.expect("应能创建实体");
        prepare_node(&entity, "toggle_node").await;

        entity
            .register_component(Renderable::new())
            .await
            .expect("应能插入 Renderable");

        let set_future = {
            let mut renderable = entity
                .get_component_mut::<Renderable>()
                .await
                .expect("应能获得 Renderable 的可写引用");
            renderable.set_enabled(false)
        };
        set_future.await;

        let renderable = entity
            .get_component::<Renderable>()
            .await
            .expect("应能读取 Renderable 组件");
        assert!(!renderable.is_enabled());

        drop(renderable);
        let _ = entity.unregister_component::<Renderable>().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn set_enabled_hides_and_restores_descendant_renderables_with_cache_stack() {
        let root = Entity::new().await.expect("应能创建 root 实体");
        crate::game::reachability::register_engine_root(root);
        crate::game::reachability::set_subtree_reachable(root, true).await;

        let child = Entity::new().await.expect("应能创建 child 实体");
        let grandchild = Entity::new().await.expect("应能创建 grandchild 实体");

        prepare_node(&root, "root").await;
        prepare_node(&child, "child").await;
        prepare_node(&grandchild, "grandchild").await;

        root.register_component(Renderable::new())
            .await
            .expect("应能插入 root Renderable");
        child
            .register_component(Renderable::new())
            .await
            .expect("应能插入 child Renderable");
        grandchild
            .register_component(Renderable::new())
            .await
            .expect("应能插入 grandchild Renderable");

        // 建树：root -> child -> grandchild
        let attach_child = {
            let mut root_node = root.get_component_mut::<Node>().await.unwrap();
            root_node.attach(child)
        };
        attach_child.await.unwrap();

        let attach_grandchild = {
            let mut child_node = child.get_component_mut::<Node>().await.unwrap();
            child_node.attach(grandchild)
        };
        attach_grandchild.await.unwrap();

        // grandchild 预设为不可见，child 保持可见。
        let grandchild_set_future = {
            let mut renderable = grandchild.get_component_mut::<Renderable>().await.unwrap();
            renderable.set_enabled(false)
        };
        grandchild_set_future.await;

        assert!(
            child
                .get_component::<Renderable>()
                .await
                .unwrap()
                .is_enabled()
        );
        assert!(
            !grandchild
                .get_component::<Renderable>()
                .await
                .unwrap()
                .is_enabled(),
            "grandchild 应按显式设置保持隐藏"
        );

        // root 隐藏：应强制整个子树不可见。
        let root_hide_future = {
            let mut renderable = root.get_component_mut::<Renderable>().await.unwrap();
            renderable.set_enabled(false)
        };
        root_hide_future.await;

        assert!(
            !root
                .get_component::<Renderable>()
                .await
                .unwrap()
                .is_enabled()
        );
        assert!(
            !child
                .get_component::<Renderable>()
                .await
                .unwrap()
                .is_enabled()
        );
        assert!(
            !grandchild
                .get_component::<Renderable>()
                .await
                .unwrap()
                .is_enabled(),
            "root 隐藏后 grandchild 也必须不可见"
        );

        // root 恢复可见：应按缓存恢复 child/grandchild 之前的设置。
        let root_show_future = {
            let mut renderable = root.get_component_mut::<Renderable>().await.unwrap();
            renderable.set_enabled(true)
        };
        root_show_future.await;

        assert!(
            root.get_component::<Renderable>()
                .await
                .unwrap()
                .is_enabled()
        );
        assert!(
            child
                .get_component::<Renderable>()
                .await
                .unwrap()
                .is_enabled(),
            "child 应恢复为可见"
        );
        assert!(
            !grandchild
                .get_component::<Renderable>()
                .await
                .unwrap()
                .is_enabled(),
            "grandchild 应恢复为隐藏"
        );

        crate::game::reachability::unregister_engine_root(root);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn renderable_visibility_is_driven_by_engine_root_reachability() {
        let engine_root = Entity::new().await.expect("应能创建根实体");
        crate::game::reachability::register_engine_root(engine_root);
        crate::game::reachability::set_subtree_reachable(engine_root, true).await;

        let child = Entity::new().await.expect("应能创建子实体");
        prepare_node(&child, "renderable_child").await;
        child
            .register_component(Renderable::new())
            .await
            .expect("应能插入 Renderable");

        // 初始化后必须不可见。
        assert!(
            !child
                .get_component::<Renderable>()
                .await
                .unwrap()
                .is_enabled()
        );

        // 挂载到引擎根节点后，可达则恢复为初始可见性（默认可见）。
        let attach_future = {
            let mut root_node = engine_root.get_component_mut::<Node>().await.unwrap();
            root_node.attach(child)
        };
        attach_future.await.unwrap();

        assert!(
            child
                .get_component::<Renderable>()
                .await
                .unwrap()
                .is_enabled()
        );

        // 卸载后不可见。
        let detach_future = {
            let mut node = child.get_component_mut::<Node>().await.unwrap();
            node.detach()
        };
        detach_future.await.unwrap();

        assert!(
            !child
                .get_component::<Renderable>()
                .await
                .unwrap()
                .is_enabled()
        );

        crate::game::reachability::unregister_engine_root(engine_root);
    }
}
