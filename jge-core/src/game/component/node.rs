use std::fmt;

use super::layer::Layer;
use super::{Component, component, component_impl};
use crate::game::{entity::Entity, logic::GameLogicHandle};
use tracing::warn;

/// [`Node`] 组件用于构建实体之间的树形关系并携带节点名称。
///
/// 名称约束如下：
/// - 不能为空字符串；
/// - 不得包含空白字符；
/// - 不得包含字符 `/`。
#[component]
pub struct Node {
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
            .finish()
    }
}

#[component_impl]
impl Node {
    /// 构造节点组件并校验名称规则。
    pub fn new(name: impl Into<String>) -> Result<Self, NodeNameError> {
        let name = name.into();
        Self::validate_name(&name)?;
        Ok(Self {
            name,
            parent: None,
            children: Vec::new(),
            logic: None,
        })
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
    pub fn path(entity: Entity) -> Result<String, NodeHierarchyError> {
        let mut segments = Vec::new();
        let mut current = Some(entity);

        while let Some(id) = current {
            let node_guard =
                <Node as Component>::read(id).ok_or(NodeHierarchyError::MissingNode(id))?;
            let name = node_guard.name.clone();
            let parent = node_guard.parent;
            drop(node_guard);

            segments.push(name);
            current = parent;
        }

        segments.reverse();
        Ok(segments.join("/"))
    }

    /// 将 `child` 挂载到 `parent` 之下。
    pub fn attach(child: Entity, parent: Entity) -> Result<(), NodeHierarchyError> {
        if child == parent {
            return Err(NodeHierarchyError::SelfAttachment(child));
        }

        Self::ensure_exists(child)?;
        Self::ensure_exists(parent)?;

        if Self::would_create_cycle(child, parent)? {
            return Err(NodeHierarchyError::HierarchyCycle {
                ancestor: parent,
                descendant: child,
            });
        }

        if Layer::read(child).is_some() {
            if let Some(ancestor_layer) = Self::nearest_layer_ancestor(parent)? {
                warn!(
                    child_id = child.id(),
                    parent_id = parent.id(),
                    ancestor_layer_id = ancestor_layer.id(),
                    "尝试在已有 Layer 树中挂载子 Layer，子 Layer 将在遍历时被忽略"
                );
            }
        }

        // 确保先从旧父节点移除，避免重复记录。
        Self::detach(child)?;

        let storage = Self::storage();

        {
            let mut child_guard = storage
                .get_mut(child.id())
                .ok_or(NodeHierarchyError::MissingNode(child))?;
            child_guard.parent = Some(parent);
        }

        let mut parent_guard = storage
            .get_mut(parent.id())
            .ok_or(NodeHierarchyError::MissingNode(parent))?;
        if !parent_guard.children.contains(&child) {
            parent_guard.children.push(child);
        }

        Ok(())
    }

    /// 将节点从当前父节点中移除。
    pub fn detach(entity: Entity) -> Result<(), NodeHierarchyError> {
        let mut node_guard = Self::storage()
            .get_mut(entity.id())
            .ok_or(NodeHierarchyError::MissingNode(entity))?;
        let parent = node_guard.parent.take();
        drop(node_guard);

        if let Some(parent) = parent {
            let mut parent_guard = Self::storage()
                .get_mut(parent.id())
                .ok_or(NodeHierarchyError::MissingNode(parent))?;
            parent_guard.children.retain(|child| *child != entity);
        }

        Ok(())
    }

    /// 为节点设置或替换 `GameLogic` 实例。
    pub fn set_logic(entity: Entity, logic: GameLogicHandle) -> Result<(), NodeHierarchyError> {
        let mut guard = Self::storage()
            .get_mut(entity.id())
            .ok_or(NodeHierarchyError::MissingNode(entity))?;
        guard.logic = Some(logic);
        Ok(())
    }

    /// 从节点移除并返回 `GameLogic`（若存在）。
    pub fn take_logic(entity: Entity) -> Result<Option<GameLogicHandle>, NodeHierarchyError> {
        let mut guard = Self::storage()
            .get_mut(entity.id())
            .ok_or(NodeHierarchyError::MissingNode(entity))?;
        Ok(guard.logic.take())
    }

    /// 判断节点是否有 `GameLogic`。
    pub fn has_logic(entity: Entity) -> Result<bool, NodeHierarchyError> {
        let guard = Self::storage()
            .get(entity.id())
            .ok_or(NodeHierarchyError::MissingNode(entity))?;
        Ok(guard.logic.is_some())
    }

    fn ensure_exists(entity: Entity) -> Result<(), NodeHierarchyError> {
        if let Some(node) = <Node as Component>::read(entity) {
            drop(node);
            Ok(())
        } else {
            Err(NodeHierarchyError::MissingNode(entity))
        }
    }

    fn would_create_cycle(child: Entity, ancestor: Entity) -> Result<bool, NodeHierarchyError> {
        let mut current = Some(ancestor);
        while let Some(id) = current {
            if id == child {
                return Ok(true);
            }
            let next = {
                let node =
                    <Node as Component>::read(id).ok_or(NodeHierarchyError::MissingNode(id))?;
                node.parent
            };
            current = next;
        }
        Ok(false)
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

    fn nearest_layer_ancestor(entity: Entity) -> Result<Option<Entity>, NodeHierarchyError> {
        let mut current = Some(entity);
        while let Some(id) = current {
            if Layer::read(id).is_some() {
                return Ok(Some(id));
            }
            let parent = {
                let node_guard =
                    <Node as Component>::read(id).ok_or(NodeHierarchyError::MissingNode(id))?;
                node_guard.parent()
            };
            current = parent;
        }

        Ok(None)
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

        Node::attach(child, parent).expect("应当可以挂载子节点");

        let parent_node = Node::read(parent).expect("父节点应存在");
        assert_eq!(parent_node.children(), &[child]);
        assert_eq!(parent_node.name(), "node_parent");
        drop(parent_node);

        let child_node = Node::read(child).expect("子节点应存在");
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

        Node::attach(child, parent).unwrap();

        Node::detach(child).expect("应能将子节点脱离父节点");

        let parent_node = Node::read(parent).unwrap();
        assert!(parent_node.children().is_empty());
        drop(parent_node);

        let child_node = Node::read(child).unwrap();
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

        Node::attach(child, parent_a).unwrap();

        Node::attach(child, parent_b).expect("重新挂载应成功");

        let old_parent = Node::read(parent_a).unwrap();
        assert!(old_parent.children().is_empty());
        drop(old_parent);

        let new_parent = Node::read(parent_b).unwrap();
        assert_eq!(new_parent.children(), &[child]);
        drop(new_parent);

        let child_component = Node::read(child).unwrap();
        assert_eq!(child_component.parent(), Some(parent_b));
        drop(child_component);
    }

    #[test]
    fn attach_detects_cycle() {
        let parent = Entity::new().expect("应能创建父实体");
        let child = Entity::new().expect("应能创建子实体");

        prepare_node(&parent, "node_parent");
        prepare_node(&child, "node_child");

        Node::attach(child, parent).unwrap();

        let result = Node::attach(parent, child);
        assert!(matches!(
            result,
            Err(NodeHierarchyError::HierarchyCycle { ancestor, descendant })
                if ancestor == child && descendant == parent
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

        Node::attach(branch, root).unwrap();
        Node::attach(leaf, branch).unwrap();

        assert_eq!(Node::path(root).unwrap(), "root");
        assert_eq!(Node::path(branch).unwrap(), "root/branch");
        assert_eq!(Node::path(leaf).unwrap(), "root/branch/leaf");
    }
}
