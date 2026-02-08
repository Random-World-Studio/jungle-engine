//! 引擎根节点与可达性。
//!
//! 渲染可见性需要一个“从引擎根可达”的概念：
//!
//! - 当 `Game` 存在时：只有挂在某个引擎根子树下的实体才被认为“可达”。
//! - 当未创建 `Game`（例如单测/离线工具）时：不启用全局根约束，视为全部可达。
//!
//! 当前实现用一个全局集合记录 `ENGINE_ROOTS`，并在 `Game` 构造/析构时注册/注销。

use std::{collections::HashSet, sync::OnceLock};

use parking_lot::RwLock;

use super::{
    component::{node::Node, renderable::Renderable},
    entity::{Entity, EntityId},
};

static ENGINE_ROOTS: OnceLock<RwLock<HashSet<EntityId>>> = OnceLock::new();

fn engine_roots() -> &'static RwLock<HashSet<EntityId>> {
    ENGINE_ROOTS.get_or_init(|| RwLock::new(HashSet::new()))
}

pub(crate) fn register_engine_root(root: Entity) {
    engine_roots().write().insert(root.id());
}

/// 注销一个引擎根实体。
pub(crate) fn unregister_engine_root(root: Entity) {
    engine_roots().write().remove(&root.id());
}

/// 判断 `entity` 是否从任意引擎根可达。
///
/// 当未注册任何引擎根（即还没有 `Game`）时，返回 `true`。
pub(crate) async fn is_reachable_from_engine_root(entity: Entity) -> bool {
    let roots = {
        let roots_guard = engine_roots().read();
        if roots_guard.is_empty() {
            // 在未创建 Game 的单测/工具场景下，不引入“全局根节点”约束。
            return true;
        }
        roots_guard.clone()
    };

    let mut visited = HashSet::new();
    let mut current = Some(entity);
    while let Some(e) = current {
        if !visited.insert(e) {
            return false;
        }
        if roots.contains(&e.id()) {
            return true;
        }
        let parent = e.get_component::<Node>().await.map(|n| n.parent());
        current = match parent {
            Some(p) => p,
            None => return false,
        };
    }
    false
}

/// 把整棵子树（从 `root` 开始）内所有 `Renderable` 的“可达性”批量更新。
///
/// 该函数只影响 `Renderable::reachable` 标记，不改变节点树结构。
pub(crate) async fn set_subtree_reachable(root: Entity, reachable: bool) {
    let mut stack = vec![root];
    let mut visited = HashSet::new();
    while let Some(entity) = stack.pop() {
        if !visited.insert(entity) {
            continue;
        }

        if let Some(mut renderable) = entity.get_component_mut::<Renderable>().await {
            renderable.set_reachable(reachable);
        }

        let children = match entity.get_component::<Node>().await {
            Some(node) => node.children().to_vec(),
            None => Vec::new(),
        };
        stack.extend(children);
    }
}
