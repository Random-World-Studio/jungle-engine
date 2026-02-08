use std::{collections::HashSet, fmt};

use super::layer::Layer;
use super::{Component, ComponentWrite, component, component_impl};
use crate::game::reachability::{is_reachable_from_engine_root, set_subtree_reachable};
use crate::game::system::logic::GameLogic;
use crate::game::{entity::Entity, system::logic::GameLogicHandle};
use tracing::warn;

/// 节点组件：用于构建实体之间的树形层级（父子关系）并携带节点名称。
///
/// 你可以把 `Node` 理解为“场景树”里的一个节点。
///
/// 约定：
/// - [`Entity::new`](crate::game::entity::Entity::new) 会自动注册一个默认 `Node`。
/// - 维护父子关系使用 `attach` / `detach`。
///
/// 名称约束：
/// - 不能为空字符串；
/// - 不得包含空白字符；
/// - 不得包含字符 `/`（因为 `path` 使用 `/` 作为分隔符）。
///
/// # 示例
///
/// ```no_run
/// use jge_core::game::{component::node::Node, entity::Entity};
///
/// # async fn demo() -> anyhow::Result<()> {
/// let parent = Entity::new().await?;
/// let child = Entity::new().await?;
///
/// // 把 child 挂到 parent 下（注意：attach 返回 Future）
/// let attach_future = {
///     let mut parent_node = parent.get_component_mut::<Node>().await.unwrap();
///     parent_node.attach(child)
/// };
/// attach_future.await?;
///
/// // 获取当前 path（根到自身）
/// let path = child
///     .get_component::<Node>()
///     .await
///     .unwrap()
///     .path()
///     .await?;
/// println!("{path}");
/// Ok(())
/// # }
/// ```
#[component]
pub struct Node {
    entity_id: Option<Entity>,
    name: String,
    parent: Option<Entity>,
    children: Vec<Entity>,
    logic: Option<GameLogicHandle>,
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("name", &self.name)
            .field("parent", &self.parent)
            .field("children", &self.children)
            .field("logic", &self.logic.as_ref().map(|_| "GameLogicHandle"))
            .field("entity_id", &self.entity_id)
            .finish()
    }
}

#[component_impl]
impl Node {
    /// 构造节点组件并校验名称规则。
    #[default(format!("{}", entity.id()))]
    pub fn new(name: impl Into<String>) -> Result<Self, NodeNameError> {
        let name = name.into();
        Self::validate_name(&name)?;
        Ok(Self {
            entity_id: None,
            name,
            parent: None,
            children: Vec::new(),
            logic: None,
        })
    }

    fn entity(&self) -> Entity {
        self.entity_id
            .expect("Node component must be attached to an entity before use")
    }

    /// 返回节点名称。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 设置节点名称并校验名称规则。
    pub fn set_name(&mut self, name: impl Into<String>) -> Result<(), NodeNameError> {
        let name = name.into();
        Self::validate_name(&name)?;
        self.name = name;
        Ok(())
    }

    /// 返回父节点实体（若存在）。
    pub fn parent(&self) -> Option<Entity> {
        self.parent
    }

    /// 返回子节点实体列表的只读视图。
    pub fn children(&self) -> &[Entity] {
        &self.children
    }

    /// 返回当前挂载的 `GameLogic`（若存在）。
    pub fn logic(&self) -> Option<&GameLogicHandle> {
        self.logic.as_ref()
    }

    /// 友好的调试展示。
    pub fn fmt_debug(&self) -> String {
        format!(
            "Node {{ name: \"{}\", parent: {:?}, children: {:?} }}",
            self.name, self.parent, self.children
        )
    }

    /// 构建从根节点开始的路径表示。
    ///
    /// 返回值形如 `root/child/grandchild`。
    ///
    /// 注意：若层级关系存在环，或层级中某个实体缺失 `Node` 组件，将返回错误。
    pub async fn path(&self) -> Result<String, NodeHierarchyError> {
        let mut segments = Vec::new();

        let descendant = self.entity();
        let mut visited = HashSet::new();
        let mut current = Some(descendant);
        while let Some(id) = current {
            if !visited.insert(id) {
                return Err(NodeHierarchyError::HierarchyCycle {
                    ancestor: id,
                    descendant,
                });
            }

            let node_guard = id
                .get_component::<Node>()
                .await
                .ok_or(NodeHierarchyError::MissingNode(id))?;
            segments.push(node_guard.name.clone());
            current = node_guard.parent;
        }

        segments.reverse();
        Ok(segments.join("/"))
    }

    async fn has_ancestor_with_hint(
        entity: Entity,
        parent_hint: Option<Entity>,
        candidate: Entity,
    ) -> Result<bool, NodeHierarchyError> {
        let descendant = entity;
        let mut visited = HashSet::new();
        let mut current = parent_hint;
        while let Some(id) = current {
            if !visited.insert(id) {
                return Err(NodeHierarchyError::HierarchyCycle {
                    ancestor: id,
                    descendant,
                });
            }
            if id == candidate {
                return Ok(true);
            }
            let node = id
                .get_component::<Node>()
                .await
                .ok_or(NodeHierarchyError::MissingNode(id))?;
            current = node.parent;
        }

        Ok(false)
    }

    async fn attach_impl(
        parent_entity: Entity,
        parent_hint: Option<Entity>,
        child: Entity,
    ) -> Result<(), NodeHierarchyError> {
        if child == parent_entity {
            return Err(NodeHierarchyError::SelfAttachment(child));
        }

        Self::ensure_exists(child).await?;

        if Self::has_ancestor_with_hint(parent_entity, parent_hint, child).await? {
            return Err(NodeHierarchyError::HierarchyCycle {
                ancestor: parent_entity,
                descendant: child,
            });
        }

        if child.get_component::<Layer>().await.is_some()
            && let Some(ancestor_layer) =
                Self::nearest_layer_ancestor_with_hint(parent_entity, parent_hint).await?
        {
            warn!(
                child_id = %child.id(),
                parent_id = %parent_entity.id(),
                ancestor_layer_id = %ancestor_layer.id(),
                "尝试在已有 Layer 树中挂载子 Layer，子 Layer 将在遍历时被忽略"
            );
        }

        let (child_logic, previous_parent) = {
            let mut child_node = Self::storage()
                .get_mut(child.id())
                .await
                .ok_or(NodeHierarchyError::MissingNode(child))?;
            let child_logic = child_node.logic.clone();
            let previous_parent = child_node.parent;

            if previous_parent == Some(parent_entity) {
                drop(child_node);
                let mut parent_node = Self::storage()
                    .get_mut(parent_entity.id())
                    .await
                    .ok_or(NodeHierarchyError::MissingNode(parent_entity))?;
                if !parent_node.children.contains(&child) {
                    parent_node.children.push(child);
                }
                return Ok(());
            }

            child_node.parent = Some(parent_entity);
            (child_logic, previous_parent)
        };

        if let Some(old_parent) = previous_parent {
            if let Some(mut old_parent_node) = Self::storage().get_mut(old_parent.id()).await {
                old_parent_node
                    .children
                    .retain(|existing| *existing != child);
            } else {
                return Err(NodeHierarchyError::MissingNode(old_parent));
            }
        }

        {
            let mut parent_node = Self::storage()
                .get_mut(parent_entity.id())
                .await
                .ok_or(NodeHierarchyError::MissingNode(parent_entity))?;
            if !parent_node.children.contains(&child) {
                parent_node.children.push(child);
            }
        }

        // 可达性：子树是否“对引擎根可达”由 parent 是否可达决定。
        // 这会直接影响 Renderable 的实际可见性。
        let reachable = is_reachable_from_engine_root(parent_entity).await;
        set_subtree_reachable(child, reachable).await;

        if previous_parent.is_some() {
            Self::notify_logic(child, child_logic.clone(), NodeLogicEvent::Detach).await;
        }
        Self::notify_logic(child, child_logic, NodeLogicEvent::Attach).await;
        Ok(())
    }

    /// 将当前节点与父节点解除关联。
    ///
    /// 调用后该节点会成为一棵子树的根（其 children 保持不变）。
    pub fn detach(
        &mut self,
    ) -> impl std::future::Future<Output = Result<(), NodeHierarchyError>> + Send + 'static {
        let entity = self.entity();
        async move { Self::detach_impl(entity).await }
    }

    async fn detach_impl(entity: Entity) -> Result<(), NodeHierarchyError> {
        let (previous_parent, logic_handle) = {
            let mut node = Self::storage()
                .get_mut(entity.id())
                .await
                .ok_or(NodeHierarchyError::MissingNode(entity))?;
            let previous_parent = node.parent.take();
            let logic_handle = node.logic.clone();
            (previous_parent, logic_handle)
        };

        if let Some(parent) = previous_parent {
            let mut parent_guard = Self::storage()
                .get_mut(parent.id())
                .await
                .ok_or(NodeHierarchyError::MissingNode(parent))?;
            parent_guard.children.retain(|child| *child != entity);

            // 只有确实从树上卸载时才会失去可达性；根节点（无 parent）不受影响。
            set_subtree_reachable(entity, false).await;
        }

        if previous_parent.is_some() {
            Self::notify_logic(entity, logic_handle, NodeLogicEvent::Detach).await;
        }
        Ok(())
    }

    /// 为节点设置或替换 `GameLogic` 实例。
    pub fn set_logic(
        &mut self,
        logic: impl GameLogic + 'static,
    ) -> impl std::future::Future<Output = ()> + Send + 'static {
        self.set_logic_handle(GameLogicHandle::new(logic))
    }

    /// 为节点设置或替换 `GameLogic` 句柄。
    pub fn set_logic_handle(
        &mut self,
        logic: GameLogicHandle,
    ) -> impl std::future::Future<Output = ()> + Send + 'static {
        let entity = self.entity();
        let is_attached = self.parent.is_some();
        let previous_logic = self.logic.replace(logic.clone());

        async move {
            if is_attached {
                if let Some(old_logic) = previous_logic {
                    Self::notify_logic(entity, Some(old_logic), NodeLogicEvent::Detach).await;
                }
                Self::notify_logic(entity, Some(logic), NodeLogicEvent::Attach).await;
            }
        }
    }

    /// 从节点移除并返回 `GameLogic`（若存在）。
    pub fn take_logic(
        &mut self,
    ) -> impl std::future::Future<Output = Option<GameLogicHandle>> + Send + 'static {
        let entity = self.entity();
        let is_attached = self.parent.is_some();
        let logic = self.logic.take();
        async move {
            if is_attached {
                Self::notify_logic(entity, logic.clone(), NodeLogicEvent::Detach).await;
            }
            logic
        }
    }

    /// 判断节点是否有 `GameLogic`。
    pub fn has_logic(&self) -> bool {
        self.logic.is_some()
    }

    async fn ensure_exists(entity: Entity) -> Result<(), NodeHierarchyError> {
        if entity.get_component::<Node>().await.is_some() {
            Ok(())
        } else {
            Err(NodeHierarchyError::MissingNode(entity))
        }
    }

    fn validate_name(name: &str) -> Result<(), NodeNameError> {
        if name.is_empty() {
            return Err(NodeNameError::Empty);
        }
        if name.contains('/') {
            return Err(NodeNameError::ContainsSlash);
        }
        if name.chars().any(char::is_whitespace) {
            return Err(NodeNameError::ContainsWhitespace);
        }
        Ok(())
    }

    async fn nearest_layer_ancestor_with_hint(
        entity: Entity,
        parent_hint: Option<Entity>,
    ) -> Result<Option<Entity>, NodeHierarchyError> {
        if entity.get_component::<Layer>().await.is_some() {
            return Ok(Some(entity));
        }

        let descendant = entity;
        let mut visited = HashSet::new();
        let mut current = parent_hint;
        while let Some(id) = current {
            if !visited.insert(id) {
                return Err(NodeHierarchyError::HierarchyCycle {
                    ancestor: id,
                    descendant,
                });
            }

            if id.get_component::<Layer>().await.is_some() {
                return Ok(Some(id));
            }

            let parent = {
                let node_guard = id
                    .get_component::<Node>()
                    .await
                    .ok_or(NodeHierarchyError::MissingNode(id))?;
                node_guard.parent()
            };
            current = parent;
        }

        Ok(None)
    }

    async fn notify_logic(entity: Entity, logic: Option<GameLogicHandle>, event: NodeLogicEvent) {
        if let Some(handle) = logic {
            let mut guard = handle.lock().await;
            let result = match event {
                NodeLogicEvent::Attach => guard.on_attach(entity).await,
                NodeLogicEvent::Detach => guard.on_detach(entity).await,
            };

            if let Err(err) = result {
                warn!(
                    target: "jge-core",
                    entity_id = %entity.id(),
                    lifecycle_event = event.label(),
                    error = %err,
                    "GameLogic lifecycle callback failed"
                );
            }
        }
    }
}

impl ComponentWrite<Node> {
    /// 将 `child` 挂到当前节点下。
    pub fn attach(
        &mut self,
        child: Entity,
    ) -> impl std::future::Future<Output = Result<(), NodeHierarchyError>> + Send + 'static {
        let parent_entity = self.entity();
        let parent_hint = self.parent;
        async move { Node::attach_impl(parent_entity, parent_hint, child).await }
    }
}

#[derive(Clone, Copy)]
enum NodeLogicEvent {
    Attach,
    Detach,
}

impl NodeLogicEvent {
    fn label(self) -> &'static str {
        match self {
            NodeLogicEvent::Attach => "attach",
            NodeLogicEvent::Detach => "detach",
        }
    }
}

/// 节点名称格式错误。
#[derive(Debug, PartialEq, Eq)]
pub enum NodeNameError {
    Empty,
    ContainsSlash,
    ContainsWhitespace,
}

impl fmt::Display for NodeNameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeNameError::Empty => write!(f, "节点名称不能为空"),
            NodeNameError::ContainsSlash => write!(f, "节点名称不能包含 '/'"),
            NodeNameError::ContainsWhitespace => write!(f, "节点名称不能包含空白字符"),
        }
    }
}

impl std::error::Error for NodeNameError {}

/// 节点关系维护中的错误类型。
#[derive(Debug, PartialEq, Eq)]
pub enum NodeHierarchyError {
    /// 指定实体尚未注册该组件。
    MissingNode(Entity),
    /// 尝试将实体挂载到自身。
    SelfAttachment(Entity),
    /// 操作会导致祖先/后代之间形成环。
    HierarchyCycle {
        ancestor: Entity,
        descendant: Entity,
    },
}

impl fmt::Display for NodeHierarchyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeHierarchyError::MissingNode(entity) => {
                write!(f, "实体 {} 未注册 Node 组件", entity.id())
            }
            NodeHierarchyError::SelfAttachment(entity) => {
                write!(f, "实体 {} 不能挂载到自身", entity.id())
            }
            NodeHierarchyError::HierarchyCycle {
                ancestor,
                descendant,
            } => {
                write!(
                    f,
                    "将实体 {} 挂载到 {} 会导致层级循环",
                    descendant.id(),
                    ancestor.id()
                )
            }
        }
    }
}

impl std::error::Error for NodeHierarchyError {}

#[cfg(test)]
mod tests {
    use super::*;

    async fn prepare_node(entity: &Entity, name: &str) {
        let _ = entity.unregister_component::<Node>().await;
        let _ = entity
            .register_component(Node::new(name).expect("应能创建节点"))
            .await
            .expect("应能注册节点");
    }

    async fn attach(parent: Entity, child: Entity, message: &str) {
        let attach_future = {
            let mut parent_node = parent
                .get_component_mut::<Node>()
                .await
                .expect("父节点应存在");
            parent_node.attach(child)
        };
        attach_future.await.expect(message);
    }

    async fn detach(entity: Entity, message: &str) {
        let detach_future = {
            let mut node = entity
                .get_component_mut::<Node>()
                .await
                .expect("节点应存在");
            node.detach()
        };
        detach_future.await.expect(message);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn attach_sets_parent_and_child() {
        let parent = Entity::new().await.expect("应能创建父实体");
        let child = Entity::new().await.expect("应能创建子实体");

        prepare_node(&parent, "node_parent").await;
        prepare_node(&child, "node_child").await;

        attach(parent, child, "应当可以挂载子节点").await;

        let parent_node = parent.get_component::<Node>().await.expect("父节点应存在");
        assert_eq!(parent_node.children(), &[child]);
        assert_eq!(parent_node.name(), "node_parent");
        drop(parent_node);

        let child_node = child.get_component::<Node>().await.expect("子节点应存在");
        assert_eq!(child_node.parent(), Some(parent));
        assert_eq!(child_node.name(), "node_child");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn detach_removes_relationship() {
        let parent = Entity::new().await.expect("应能创建父实体");
        let child = Entity::new().await.expect("应能创建子实体");

        prepare_node(&parent, "node_parent").await;
        prepare_node(&child, "node_child").await;

        attach(parent, child, "应当可以挂载子节点").await;
        detach(child, "应能将子节点脱离父节点").await;

        let parent_node = parent.get_component::<Node>().await.unwrap();
        assert!(parent_node.children().is_empty());
        drop(parent_node);

        let child_node = child.get_component::<Node>().await.unwrap();
        assert_eq!(child_node.parent(), None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reattach_moves_child_between_parents() {
        let parent_a = Entity::new().await.expect("应能创建第一个父实体");
        let parent_b = Entity::new().await.expect("应能创建第二个父实体");
        let child = Entity::new().await.expect("应能创建子实体");

        prepare_node(&parent_a, "node_a").await;
        prepare_node(&parent_b, "node_b").await;
        prepare_node(&child, "node_child").await;

        attach(parent_a, child, "挂载到 parent_a 应成功").await;
        attach(parent_b, child, "重新挂载到 parent_b 应成功").await;

        let old_parent = parent_a.get_component::<Node>().await.unwrap();
        assert!(old_parent.children().is_empty());
        drop(old_parent);

        let new_parent = parent_b.get_component::<Node>().await.unwrap();
        assert_eq!(new_parent.children(), &[child]);
        drop(new_parent);

        let child_component = child.get_component::<Node>().await.unwrap();
        assert_eq!(child_component.parent(), Some(parent_b));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn attach_detects_cycle() {
        let parent = Entity::new().await.expect("应能创建父实体");
        let child = Entity::new().await.expect("应能创建子实体");

        prepare_node(&parent, "node_parent").await;
        prepare_node(&child, "node_child").await;

        attach(parent, child, "首次挂载应成功").await;

        let attach_cycle_future = {
            let mut child_node = child.get_component_mut::<Node>().await.unwrap();
            child_node.attach(parent)
        };
        let result: NodeHierarchyError = attach_cycle_future
            .await
            .expect_err("挂载应当检测到层级循环");
        assert!(matches!(result, NodeHierarchyError::HierarchyCycle { .. }));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn node_name_validation_rules() {
        assert!(matches!(Node::new(""), Err(NodeNameError::Empty)));
        assert!(matches!(
            Node::new("has space"),
            Err(NodeNameError::ContainsWhitespace)
        ));
        assert!(matches!(
            Node::new("slash/inside"),
            Err(NodeNameError::ContainsSlash)
        ));
        assert!(Node::new("valid_name").is_ok());
    }
}
