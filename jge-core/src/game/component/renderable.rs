use super::node::Node;
use super::{component, component_impl};
use crate::game::entity::Entity;

#[component(Node)]
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
    enabled: bool,
}

#[component_impl]
impl Renderable {
    /// 创建一个默认启用的可渲染组件。
    #[default()]
    pub fn new() -> Self {
        Self {
            entity_id: None,
            enabled: true,
        }
    }

    /// 返回当前组件是否参与渲染。
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// 设置组件是否参与渲染。
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
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
}
