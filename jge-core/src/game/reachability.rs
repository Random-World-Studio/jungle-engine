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

pub(crate) fn unregister_engine_root(root: Entity) {
    engine_roots().write().remove(&root.id());
}

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
