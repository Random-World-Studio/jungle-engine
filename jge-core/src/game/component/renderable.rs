use super::node::Node;
use super::{Component, ComponentDependencyError, ComponentStorage, component_impl};
use crate::game::entity::Entity;
use std::sync::OnceLock;

static RENDERABLE_STORAGE: OnceLock<ComponentStorage<Renderable>> = OnceLock::new();

#[derive(Debug, Clone)]
/// 可渲染标记组件。
///
/// 挂载该组件的实体会被渲染系统视为“可参与渲染的对象”。
///
/// - 你通常会同时挂载 `Shape` 与 `Transform`（以及可选的 `Material`）。
/// - 想临时隐藏一个实体时，可用 [`set_enabled`](Self::set_enabled) 关闭渲染。
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
/// let e = Entity::new()?;
/// e.register_component(Renderable::new())?;
/// e.register_component(Transform::new())?;
/// Ok(())
/// # }
/// ```
pub struct Renderable {
    entity_id: Option<Entity>,
    /// 用户配置的“期望可见性”（初始值）。
    configured_enabled: bool,
    /// 实际可见性（会被节点可达性覆盖）。
    enabled: bool,
    /// 是否从引擎根节点可达（由节点挂载/卸载传播维护）。
    reachable: bool,
}

impl Component for Renderable {
    fn storage() -> &'static ComponentStorage<Self> {
        RENDERABLE_STORAGE.get_or_init(ComponentStorage::new)
    }

    fn register_dependencies(entity: Entity) -> Result<(), ComponentDependencyError> {
        if entity.get_component::<Node>().is_none() {
            let component = Node::__jge_component_default(entity)?;
            let _ = entity.register_component(component)?;
        }
        Ok(())
    }

    fn attach_entity(&mut self, entity: Entity) {
        self.entity_id = Some(entity);

        // Renderable 的实际可见性由“是否对引擎根可达”驱动。
        // 这一步用于覆盖“先建树、后注册 Renderable”的场景。
        let reachable = crate::game::is_reachable_from_engine_root(entity);
        self.set_reachable(reachable);
    }

    fn detach_entity(&mut self) {
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
            enabled: false,
            reachable: false,
        }
    }

    /// 返回当前组件的**实际**可见性（是否参与渲染）。
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// 设置组件的“期望可见性”。
    ///
    /// 若节点当前从引擎根节点可达，则会立即影响实际可见性；否则会在下次变为可达时生效。
    pub fn set_enabled(&mut self, enabled: bool) {
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

        let inserted = entity
            .register_component(Renderable::new())
            .expect("缺少 Node 时应自动注册依赖");
        assert!(inserted.is_none());

        assert!(
            entity.get_component::<Node>().is_some(),
            "Node 应被自动注册"
        );

        prepare_node(&entity, "renderable_node");
        let previous = entity
            .register_component(Renderable::new())
            .expect("重复插入应返回旧的 Renderable");
        assert!(previous.is_some());

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

    #[tokio::test]
    async fn renderable_visibility_is_driven_by_engine_root_reachability() {
        let engine_root = Entity::new().expect("应能创建根实体");
        crate::game::register_engine_root(engine_root);
        crate::game::set_subtree_reachable(engine_root, true);

        let child = Entity::new().expect("应能创建子实体");
        prepare_node(&child, "renderable_child");
        child
            .register_component(Renderable::new())
            .expect("应能插入 Renderable");

        // 初始化后必须不可见。
        assert!(!child.get_component::<Renderable>().unwrap().is_enabled());

        // 挂载到引擎根节点后，可达则恢复为初始可见性（默认可见）。
        let attach_future = {
            let mut root_node = engine_root.get_component_mut::<Node>().unwrap();
            root_node.attach(child)
        };
        attach_future.await.unwrap();

        assert!(child.get_component::<Renderable>().unwrap().is_enabled());

        // 卸载后不可见。
        let detach_future = {
            let mut node = child.get_component_mut::<Node>().unwrap();
            node.detach()
        };
        detach_future.await.unwrap();

        assert!(!child.get_component::<Renderable>().unwrap().is_enabled());

        crate::game::unregister_engine_root(engine_root);
    }
}
