use std::{collections::HashSet, sync::Arc};

use parking_lot::RwLock;

use super::{Component, Entity, GameLogicHandle, Node, Scene2D};

pub(super) async fn rebuild_render_snapshot(
    root: Entity,
    framebuffer_size: (u32, u32),
    render_snapshot: &RwLock<Arc<super::RenderSnapshot>>,
) {
    let framebuffer_size = (framebuffer_size.0.max(1), framebuffer_size.1.max(1));
    let snapshot = super::RenderSnapshot::build(root, framebuffer_size).await;
    *render_snapshot.write() = Arc::new(snapshot);
}

pub(super) fn pack_framebuffer_size(width: u32, height: u32) -> u64 {
    ((width as u64) << 32) | (height as u64)
}

pub(super) fn unpack_framebuffer_size(value: u64) -> (u32, u32) {
    ((value >> 32) as u32, value as u32)
}

pub(super) async fn update_scene2d_framebuffer_sizes(root: Entity, framebuffer_size: (u32, u32)) {
    let framebuffer_size = (framebuffer_size.0.max(1), framebuffer_size.1.max(1));

    let mut stack = vec![root];
    let mut visited = HashSet::new();

    while let Some(entity) = stack.pop() {
        if !visited.insert(entity) {
            continue;
        }

        if let Some(mut scene) = entity.get_component_mut::<Scene2D>().await {
            scene.set_framebuffer_size(framebuffer_size);
        }

        if let Some(node_guard) = entity.get_component::<Node>().await {
            let children: Vec<Entity> = node_guard.children().to_vec();
            drop(node_guard);

            for child in children.into_iter().rev() {
                stack.push(child);
            }
        }
    }
}

pub(super) async fn collect_logic_handle_chunks() -> Vec<Vec<(super::EntityId, GameLogicHandle)>> {
    Node::storage()
        .collect_chunks_with(|entity_id, node| {
            node.logic().cloned().map(|logic| (entity_id, logic))
        })
        .await
}
