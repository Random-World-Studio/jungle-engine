use std::{collections::HashSet, fmt, ops::ControlFlow};

use super::layer::Layer;
use super::{Component, ComponentWrite, component, component_impl};
use crate::game::system::logic::GameLogic;
use crate::game::{entity::Entity, system::logic::GameLogicHandle};
use tracing::warn;

/// [`Node`] 组件用于构建实体之间的树形关系并携带节点名称。
///
/// 名称约束如下：
/// - 不能为空字符串；
/// - 不得包含空白字符；
/// - 不得包含字符 `/`。
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
    #[default(format!("entity_{}", entity.id()))]
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
    pub fn path(&self) -> Result<String, NodeHierarchyError> {
        let mut segments = Vec::new();
        Self::walk_ancestor_chain::<(), _>(Some(self.entity()), self.entity(), |id| {
            let node_guard = id
                .get_component::<Node>()
                .ok_or(NodeHierarchyError::MissingNode(id))?;
            segments.push(node_guard.name.clone());
            Ok(ControlFlow::Continue(node_guard.parent))
        })?;

        segments.reverse();
        Ok(segments.join("/"))
    }

    fn walk_ancestor_chain<R, F>(
        start: Option<Entity>,
        descendant: Entity,
        mut step: F,
    ) -> Result<Option<R>, NodeHierarchyError>
    where
        F: FnMut(Entity) -> Result<ControlFlow<R, Option<Entity>>, NodeHierarchyError>,
    {
        let mut visited = HashSet::new();
        let mut current = start;
        while let Some(id) = current {
            if !visited.insert(id) {
                return Err(NodeHierarchyError::HierarchyCycle {
                    ancestor: id,
                    descendant,
                });
            }
            match step(id)? {
                ControlFlow::Break(result) => return Ok(Some(result)),
                ControlFlow::Continue(next) => current = next,
            }
        }
        Ok(None)
    }

    pub(crate) fn attach_internal(&mut self, child: Entity) -> Result<(), NodeHierarchyError> {
        let parent_entity = self.entity();
        if child == parent_entity {
            return Err(NodeHierarchyError::SelfAttachment(child));
        }

        Self::ensure_exists(child)?;

        if self.has_ancestor(child)? {
            return Err(NodeHierarchyError::HierarchyCycle {
                ancestor: parent_entity,
                descendant: child,
            });
        }

        if child.get_component::<Layer>().is_some() {
            if let Some(ancestor_layer) =
                Self::nearest_layer_ancestor_with_hint(parent_entity, self.parent)?
            {
                warn!(
                    child_id = child.id(),
                    parent_id = parent_entity.id(),
                    ancestor_layer_id = ancestor_layer.id(),
                    "尝试在已有 Layer 树中挂载子 Layer，子 Layer 将在遍历时被忽略"
                );
            }
        }

        let mut child_node = Self::storage()
            .get_mut(child.id())
            .ok_or(NodeHierarchyError::MissingNode(child))?;
        let child_logic = child_node.logic.clone();

        if child_node.parent == Some(parent_entity) {
            drop(child_node);
            if !self.children.contains(&child) {
                self.children.push(child);
            }
            return Ok(());
        }

        let previous_parent = child_node.parent;

        if let Some(old_parent) = previous_parent {
            if let Some(mut old_parent_node) = Self::storage().get_mut(old_parent.id()) {
                old_parent_node
                    .children
                    .retain(|existing| *existing != child);
            } else {
                return Err(NodeHierarchyError::MissingNode(old_parent));
            }
        }

        child_node.parent = Some(parent_entity);
        drop(child_node);

        if !self.children.contains(&child) {
            self.children.push(child);
        }

        if previous_parent.is_some() {
            Self::notify_logic(child, child_logic.clone(), NodeLogicEvent::Detach);
        }
        Self::notify_logic(child, child_logic, NodeLogicEvent::Attach);

        Ok(())
    }

    fn has_ancestor(&self, candidate: Entity) -> Result<bool, NodeHierarchyError> {
        let found = Self::walk_ancestor_chain(self.parent, self.entity(), |id| {
            if id == candidate {
                return Ok(ControlFlow::Break(()));
            }
            let node = id
                .get_component::<Node>()
                .ok_or(NodeHierarchyError::MissingNode(id))?;
            Ok(ControlFlow::Continue(node.parent))
        })?;

        Ok(found.is_some())
    }

    /// 将当前节点与父节点解除关联。
    pub fn detach(&mut self) -> Result<(), NodeHierarchyError> {
        self.detach_from_parent()
    }

    /// 为节点设置或替换 `GameLogic` 实例。
    pub fn set_logic(&mut self, logic: impl GameLogic + 'static) {
        self.set_logic_handle(GameLogicHandle::new(logic));
    }

    /// 为节点设置或替换 `GameLogic` 句柄。
    ///
    /// 仅在你需要复用/转移同一个逻辑实例（例如在节点间移动）时使用；
    /// 一般情况下建议直接调用 [`Node::set_logic`] 传入具体逻辑类型。
    pub fn set_logic_handle(&mut self, logic: GameLogicHandle) {
        let entity = self.entity();
        let is_attached = self.parent.is_some();
        let previous_logic = self.logic.replace(logic.clone());

        if is_attached {
            if let Some(old_logic) = previous_logic {
                Self::notify_logic(entity, Some(old_logic), NodeLogicEvent::Detach);
            }
            Self::notify_logic(entity, Some(logic), NodeLogicEvent::Attach);
        }
    }

    /// 从节点移除并返回 `GameLogic`（若存在）。
    pub fn take_logic(&mut self) -> Option<GameLogicHandle> {
        let entity = self.entity();
        let is_attached = self.parent.is_some();
        let logic = self.logic.take();

        if is_attached {
            Self::notify_logic(entity, logic.clone(), NodeLogicEvent::Detach);
        }

        logic
    }

    /// 判断节点是否有 `GameLogic`。
    pub fn has_logic(&self) -> bool {
        self.logic.is_some()
    }

    fn detach_from_parent(&mut self) -> Result<(), NodeHierarchyError> {
        let entity = self.entity();
        let logic_handle = self.logic.clone();
        let previous_parent = self.parent.take();
        if let Some(parent) = previous_parent {
            let mut parent_guard = Self::storage()
                .get_mut(parent.id())
                .ok_or(NodeHierarchyError::MissingNode(parent))?;
            parent_guard.children.retain(|child| *child != entity);
        }

        if previous_parent.is_some() {
            Self::notify_logic(entity, logic_handle, NodeLogicEvent::Detach);
        }

        Ok(())
    }

    fn ensure_exists(entity: Entity) -> Result<(), NodeHierarchyError> {
        if entity.get_component::<Node>().is_some() {
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

    fn nearest_layer_ancestor_with_hint(
        entity: Entity,
        parent_hint: Option<Entity>,
    ) -> Result<Option<Entity>, NodeHierarchyError> {
        if entity.get_component::<Layer>().is_some() {
            return Ok(Some(entity));
        }

        let start_parent = parent_hint;
        Self::walk_ancestor_chain(start_parent, entity, |id| {
            if id == entity {
                return Ok(ControlFlow::Continue(start_parent));
            }
            if id.get_component::<Layer>().is_some() {
                return Ok(ControlFlow::Break(id));
            }
            let parent = {
                let node_guard = id
                    .get_component::<Node>()
                    .ok_or(NodeHierarchyError::MissingNode(id))?;
                node_guard.parent()
            };
            Ok(ControlFlow::Continue(parent))
        })
    }
}

impl ComponentWrite<Node> {
    pub fn attach(&mut self, child: Entity) -> Result<(), NodeHierarchyError> {
        Node::attach_internal(&mut *self, child)
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

impl Node {
    fn notify_logic(entity: Entity, logic: Option<GameLogicHandle>, event: NodeLogicEvent) {
        if let Some(handle) = logic {
            let mut guard = handle.blocking_lock();
            let result = match event {
                NodeLogicEvent::Attach => guard.on_attach(entity),
                NodeLogicEvent::Detach => guard.on_detach(entity),
            };

            if let Err(err) = result {
                warn!(
                    target: "jge-core",
                    entity_id = entity.id(),
                    lifecycle_event = event.label(),
                    error = %err,
                    "GameLogic lifecycle callback failed"
                );
            }
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

impl std::fmt::Display for NodeHierarchyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
    use crate::game::component::layer::Layer;

    fn prepare_node(entity: &Entity, name: &str) {
        let _ = entity
            .register_component(Node::new(name).expect("应能创建节点"))
            .expect("应能注册节点");
    }

    #[test]
    fn attach_sets_parent_and_child() {
        let parent = Entity::new().expect("应能创建父实体");
        let child = Entity::new().expect("应能创建子实体");

        prepare_node(&parent, "node_parent");
        prepare_node(&child, "node_child");

        {
            let mut parent_node = parent.get_component_mut::<Node>().expect("父节点应存在");
            parent_node.attach(child).expect("应当可以挂载子节点");
        }

        let parent_node = parent.get_component::<Node>().expect("父节点应存在");
        assert_eq!(parent_node.children(), &[child]);
        assert_eq!(parent_node.name(), "node_parent");
        drop(parent_node);

        let child_node = child.get_component::<Node>().expect("子节点应存在");
        assert_eq!(child_node.parent(), Some(parent));
        assert_eq!(child_node.name(), "node_child");
        drop(child_node);
    }

    #[test]
    fn detach_removes_relationship() {
        let parent = Entity::new().expect("应能创建父实体");
        let child = Entity::new().expect("应能创建子实体");

        prepare_node(&parent, "node_parent");
        prepare_node(&child, "node_child");

        {
            let mut parent_node = parent.get_component_mut::<Node>().unwrap();
            parent_node.attach(child).unwrap();
        }

        {
            let mut child_node = child.get_component_mut::<Node>().unwrap();
            child_node.detach().expect("应能将子节点脱离父节点");
        }

        let parent_node = parent.get_component::<Node>().unwrap();
        assert!(parent_node.children().is_empty());
        drop(parent_node);

        let child_node = child.get_component::<Node>().unwrap();
        assert_eq!(child_node.parent(), None);
        drop(child_node);
    }

    #[test]
    fn reattach_moves_child_between_parents() {
        let parent_a = Entity::new().expect("应能创建第一个父实体");
        let parent_b = Entity::new().expect("应能创建第二个父实体");
        let child = Entity::new().expect("应能创建子实体");

        prepare_node(&parent_a, "node_a");
        prepare_node(&parent_b, "node_b");
        prepare_node(&child, "node_child");

        {
            let mut parent_node = parent_a.get_component_mut::<Node>().unwrap();
            parent_node.attach(child).unwrap();
        }

        {
            let mut parent_node = parent_b.get_component_mut::<Node>().unwrap();
            parent_node.attach(child).expect("重新挂载应成功");
        }

        let old_parent = parent_a.get_component::<Node>().unwrap();
        assert!(old_parent.children().is_empty());
        drop(old_parent);

        let new_parent = parent_b.get_component::<Node>().unwrap();
        assert_eq!(new_parent.children(), &[child]);
        drop(new_parent);

        let child_component = child.get_component::<Node>().unwrap();
        assert_eq!(child_component.parent(), Some(parent_b));
        drop(child_component);
    }

    #[test]
    fn attach_detects_cycle() {
        let parent = Entity::new().expect("应能创建父实体");
        let child = Entity::new().expect("应能创建子实体");

        prepare_node(&parent, "node_parent");
        prepare_node(&child, "node_child");

        {
            let mut parent_node = parent.get_component_mut::<Node>().unwrap();
            parent_node.attach(child).unwrap();
        }

        let result = {
            let mut child_node = child.get_component_mut::<Node>().unwrap();
            child_node.attach(parent)
        };
        assert!(matches!(
            result,
            Err(NodeHierarchyError::HierarchyCycle { ancestor, descendant })
                if ancestor == child && descendant == parent
        ));
    }

    #[test]
    fn nearest_layer_ancestor_detects_cycle() {
        let entity = Entity::new().expect("应能创建实体");
        prepare_node(&entity, "cycle_node");

        {
            let mut node = entity.get_component_mut::<Node>().expect("节点应存在");
            node.parent = Some(entity);
        }

        let error =
            Node::nearest_layer_ancestor_with_hint(entity, Some(entity)).expect_err("应检测到循环");
        assert!(matches!(
            error,
            NodeHierarchyError::HierarchyCycle { ancestor, descendant }
                if ancestor == entity && descendant == entity
        ));
    }

    #[test]
    fn nearest_layer_ancestor_with_hint_handles_cycle_while_locked() {
        let entity = Entity::new().expect("应能创建实体");
        prepare_node(&entity, "cycle_hint_node");

        let mut node = entity.get_component_mut::<Node>().expect("节点应存在");
        node.parent = Some(entity);
        let parent_hint = node.parent;

        let error =
            Node::nearest_layer_ancestor_with_hint(entity, parent_hint).expect_err("应检测到循环");
        assert!(matches!(
            error,
            NodeHierarchyError::HierarchyCycle { ancestor, descendant }
                if ancestor == entity && descendant == entity
        ));
    }

    #[test]
    fn path_reports_cycle_error() {
        let entity = Entity::new().expect("应能创建实体");
        prepare_node(&entity, "loop_node");

        {
            let mut node = entity.get_component_mut::<Node>().expect("节点应存在");
            node.parent = Some(entity);
        }

        let guard = entity.get_component::<Node>().expect("节点应存在");
        let error = guard.path().expect_err("路径计算应发现循环");
        assert!(matches!(
            error,
            NodeHierarchyError::HierarchyCycle { ancestor, descendant }
                if ancestor == entity && descendant == entity
        ));
    }

    #[test]
    fn has_ancestor_detects_cycle() {
        let root = Entity::new().expect("应能创建根节点");
        let loop_a = Entity::new().expect("应能创建循环节点 A");
        let loop_b = Entity::new().expect("应能创建循环节点 B");
        let candidate = Entity::new().expect("应能创建候选节点");
        prepare_node(&root, "root_node");
        prepare_node(&loop_a, "loop_a");
        prepare_node(&loop_b, "loop_b");
        prepare_node(&candidate, "candidate");

        {
            let mut root_node = root.get_component_mut::<Node>().expect("根节点应存在");
            root_node.parent = Some(loop_a);
        }
        {
            let mut loop_a_node = loop_a
                .get_component_mut::<Node>()
                .expect("循环节点 A 应存在");
            loop_a_node.parent = Some(loop_b);
        }
        {
            let mut loop_b_node = loop_b
                .get_component_mut::<Node>()
                .expect("循环节点 B 应存在");
            loop_b_node.parent = Some(loop_a);
        }

        let guard = root.get_component::<Node>().expect("根节点应存在");
        let error = guard
            .has_ancestor(candidate)
            .expect_err("祖先检测应在循环中失败");
        assert!(matches!(
            error,
            NodeHierarchyError::HierarchyCycle { ancestor, descendant }
                if ancestor == loop_a && descendant == root
        ));
    }

    #[test]
    fn node_name_validation_rules() {
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

    #[test]
    fn path_builds_hierarchy_identifier() {
        let root = Entity::new().expect("应能创建根实体");
        let branch = Entity::new().expect("应能创建中间实体");
        let leaf = Entity::new().expect("应能创建叶子实体");

        prepare_node(&root, "root");
        prepare_node(&branch, "branch");
        prepare_node(&leaf, "leaf");

        {
            let mut root_node = root.get_component_mut::<Node>().unwrap();
            root_node.attach(branch).unwrap();
        }
        {
            let mut branch_node = branch.get_component_mut::<Node>().unwrap();
            branch_node.attach(leaf).unwrap();
        }

        let root_node = root.get_component::<Node>().unwrap();
        assert_eq!(root_node.path().unwrap(), "root");
        drop(root_node);
        let branch_node = branch.get_component::<Node>().unwrap();
        assert_eq!(branch_node.path().unwrap(), "root/branch");
        drop(branch_node);
        let leaf_node = leaf.get_component::<Node>().unwrap();
        assert_eq!(leaf_node.path().unwrap(), "root/branch/leaf");
    }

    #[test]
    fn logic_callbacks_fire_on_attach_and_detach() {
        use std::sync::{Arc, Mutex as StdMutex};

        use crate::game::system::logic::GameLogic;

        struct TrackingLogic {
            events: Arc<StdMutex<Vec<&'static str>>>,
        }

        #[async_trait::async_trait]
        impl GameLogic for TrackingLogic {
            fn on_attach(&mut self, _e: Entity) -> anyhow::Result<()> {
                self.events.lock().unwrap().push("attach");
                Ok(())
            }

            fn on_detach(&mut self, _e: Entity) -> anyhow::Result<()> {
                self.events.lock().unwrap().push("detach");
                Ok(())
            }
        }

        let parent = Entity::new().expect("应能创建父实体");
        let child = Entity::new().expect("应能创建子实体");

        prepare_node(&parent, "logic_parent");
        prepare_node(&child, "logic_child");

        let events = Arc::new(StdMutex::new(Vec::new()));
        let logic = TrackingLogic {
            events: events.clone(),
        };

        {
            let mut child_node = child.get_component_mut::<Node>().unwrap();
            child_node.set_logic(logic);
        }

        {
            let mut parent_node = parent.get_component_mut::<Node>().unwrap();
            parent_node.attach(child).unwrap();
        }

        {
            let log = events.lock().unwrap();
            assert_eq!(log.as_slice(), &["attach"]);
        }

        {
            let mut child_node = child.get_component_mut::<Node>().unwrap();
            child_node.detach().unwrap();
        }

        let log = events.lock().unwrap();
        assert_eq!(log.as_slice(), &["attach", "detach"]);
    }

    #[test]
    fn logic_callbacks_fire_when_logic_set_after_attachment() {
        use std::sync::{Arc, Mutex as StdMutex};

        use crate::game::system::logic::GameLogic;

        struct TrackingLogic {
            events: Arc<StdMutex<Vec<&'static str>>>,
        }

        #[async_trait::async_trait]
        impl GameLogic for TrackingLogic {
            fn on_attach(&mut self, _e: Entity) -> anyhow::Result<()> {
                self.events.lock().unwrap().push("attach");
                Ok(())
            }

            fn on_detach(&mut self, _e: Entity) -> anyhow::Result<()> {
                self.events.lock().unwrap().push("detach");
                Ok(())
            }
        }

        let parent = Entity::new().expect("应能创建父实体");
        let child = Entity::new().expect("应能创建子实体");

        prepare_node(&parent, "logic_parent_after");
        prepare_node(&child, "logic_child_after");

        {
            let mut parent_node = parent.get_component_mut::<Node>().unwrap();
            parent_node.attach(child).unwrap();
        }

        let events = Arc::new(StdMutex::new(Vec::new()));
        let logic = TrackingLogic {
            events: events.clone(),
        };

        {
            let mut child_node = child.get_component_mut::<Node>().unwrap();
            child_node.set_logic(logic);
        }

        {
            let log = events.lock().unwrap();
            assert_eq!(log.as_slice(), &["attach"]);
        }

        {
            let mut child_node = child.get_component_mut::<Node>().unwrap();
            let _ = child_node.take_logic();
        }

        let log = events.lock().unwrap();
        assert_eq!(log.as_slice(), &["attach", "detach"]);
    }

    #[test]
    fn set_logic_handle_attaches_when_node_is_attached() {
        use std::sync::{Arc, Mutex as StdMutex};

        use crate::game::system::logic::GameLogic;

        struct TrackingLogic {
            events: Arc<StdMutex<Vec<&'static str>>>,
        }

        #[async_trait::async_trait]
        impl GameLogic for TrackingLogic {
            fn on_attach(&mut self, _e: Entity) -> anyhow::Result<()> {
                self.events.lock().unwrap().push("attach");
                Ok(())
            }
        }

        let parent = Entity::new().expect("应能创建父实体");
        let child = Entity::new().expect("应能创建子实体");

        prepare_node(&parent, "logic_parent_handle");
        prepare_node(&child, "logic_child_handle");

        {
            let mut parent_node = parent.get_component_mut::<Node>().unwrap();
            parent_node.attach(child).unwrap();
        }

        let events = Arc::new(StdMutex::new(Vec::new()));
        let handle = GameLogicHandle::new(TrackingLogic {
            events: events.clone(),
        });

        {
            let mut child_node = child.get_component_mut::<Node>().unwrap();
            child_node.set_logic_handle(handle);
        }

        let log = events.lock().unwrap();
        assert_eq!(log.as_slice(), &["attach"]);
    }

    #[test]
    fn replacing_logic_on_attached_node_detaches_old_then_attaches_new() {
        use std::sync::{Arc, Mutex as StdMutex};

        use crate::game::system::logic::GameLogic;

        struct RecordingLogic {
            label: &'static str,
            events: Arc<StdMutex<Vec<&'static str>>>,
        }

        #[async_trait::async_trait]
        impl GameLogic for RecordingLogic {
            fn on_attach(&mut self, _e: Entity) -> anyhow::Result<()> {
                self.events.lock().unwrap().push(if self.label == "a" {
                    "attach_a"
                } else {
                    "attach_b"
                });
                Ok(())
            }

            fn on_detach(&mut self, _e: Entity) -> anyhow::Result<()> {
                self.events.lock().unwrap().push(if self.label == "a" {
                    "detach_a"
                } else {
                    "detach_b"
                });
                Ok(())
            }
        }

        let parent = Entity::new().expect("应能创建父实体");
        let child = Entity::new().expect("应能创建子实体");

        prepare_node(&parent, "logic_parent_replace");
        prepare_node(&child, "logic_child_replace");

        {
            let mut parent_node = parent.get_component_mut::<Node>().unwrap();
            parent_node.attach(child).unwrap();
        }

        let events = Arc::new(StdMutex::new(Vec::new()));

        {
            let mut child_node = child.get_component_mut::<Node>().unwrap();
            child_node.set_logic(RecordingLogic {
                label: "a",
                events: events.clone(),
            });
            child_node.set_logic(RecordingLogic {
                label: "b",
                events: events.clone(),
            });
        }

        let log = events.lock().unwrap();
        assert_eq!(log.as_slice(), &["attach_a", "detach_a", "attach_b"]);
    }

    #[test]
    fn attach_layer_child_under_existing_layer_tree() {
        let root = Entity::new().expect("应能创建根节点");
        let parent = Entity::new().expect("应能创建父节点");
        let child = Entity::new().expect("应能创建子节点");

        prepare_node(&root, "layer_root");
        prepare_node(&parent, "layer_parent");
        prepare_node(&child, "layer_child");

        let _ = root
            .register_component(Layer::new())
            .expect("应能为根节点注册 Layer");
        let _ = child
            .register_component(Layer::new())
            .expect("应能为子节点注册 Layer");

        {
            let mut root_node = root.get_component_mut::<Node>().expect("根节点应存在");
            root_node.attach(parent).expect("应能挂载父节点");
        }

        {
            let mut parent_node = parent.get_component_mut::<Node>().expect("父节点应存在");
            parent_node.attach(child).expect("应能挂载子 Layer");
        }

        let child_node = child.get_component::<Node>().expect("子节点应存在");
        assert_eq!(child_node.parent(), Some(parent));
    }
}
