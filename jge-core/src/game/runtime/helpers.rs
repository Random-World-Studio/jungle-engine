use std::{collections::HashSet, sync::Arc};

use parking_lot::RwLock;

use super::{Entity, Node, Scene2D};
use crate::game::system::logic::GameLogicHandle;
use crate::game::system::logic_registry;

/// 重新构建并提交一帧 [`RenderSnapshot`](super::RenderSnapshot)。
///
/// - `framebuffer_size` 会被 clamp 到最小 `(1,1)`。
/// - 内部会对 `render_snapshot` 加写锁并替换为新快照。
///
/// 该 helper 用于把“build + write-back”逻辑集中到一处，避免窗口初始化与 update loop 各写一份。
pub(super) async fn rebuild_render_snapshot(
    root: Entity,
    framebuffer_size: (u32, u32),
    render_snapshot: &RwLock<Arc<super::RenderSnapshot>>,
) {
    let framebuffer_size = (framebuffer_size.0.max(1), framebuffer_size.1.max(1));
    let snapshot = super::RenderSnapshot::build(root, framebuffer_size).await;
    *render_snapshot.write() = Arc::new(snapshot);
}

/// 遍历 `root` 子树，更新所有 `Scene2D` 的 framebuffer size。
///
/// 用于在窗口 resize 时尽快更新 2D 坐标转换逻辑。
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

/// 收集当前所有 `Node` 上绑定的 `GameLogic`，并按“批处理分组（batch chunks）”返回。
///
/// 注意这里的“chunk”仅指为了降低每 tick 的调度开销而做的批处理分组，
/// 与旧的组件存储 chunk、LOD/八叉树分块等概念无关。
///
/// 当前实现使用一个全局 registry（由 `Node::set_logic_*` 维护），避免每 tick 遍历 Node 子树。
pub(super) fn collect_logic_handle_chunks() -> Vec<Vec<(super::EntityId, GameLogicHandle)>> {
    logic_registry::collect_chunks(32)
}
