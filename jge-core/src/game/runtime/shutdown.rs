use anyhow::Context;
use std::collections::HashSet;

use tracing::{error, warn};

use super::{Entity, Game, Node};

impl Game {
    async fn collect_subtree_postorder(root: Entity) -> Vec<Entity> {
        let mut visited = HashSet::new();
        let mut out = Vec::new();

        // (entity, expanded)
        let mut stack = vec![(root, false)];
        while let Some((entity, expanded)) = stack.pop() {
            if expanded {
                out.push(entity);
                continue;
            }

            if !visited.insert(entity.id()) {
                continue;
            }

            stack.push((entity, true));

            if let Some(node) = entity.get_component::<Node>().await {
                let children = node.children().to_vec();
                drop(node);

                // 保持与递归实现一致的遍历顺序。
                for child in children.into_iter().rev() {
                    stack.push((child, false));
                }
            }
        }

        out
    }

    pub(super) fn detach_node_tree(&self, root: Entity) {
        // 没有 Node 的根实体不参与拆树。
        if self
            .runtime
            .block_on(root.get_component::<Node>())
            .is_none()
        {
            return;
        }

        // 这里会调用内部 runtime 的 `block_on(...)` 来执行 async 的拆树与回调逻辑；
        // 因此必须在“非 tokio runtime 上下文”执行，否则会触发 tokio 的 "cannot block_on within a runtime" 类 panic。
        // 常规用法（game.run() 返回后 drop Game）满足该条件。
        if tokio::runtime::Handle::try_current().is_ok() {
            warn!(
                target = "jge-core",
                "Game 被在 tokio runtime 内 drop：跳过节点树 on_detach 回调以避免 panic"
            );
            return;
        }

        let runtime = &self.runtime;
        runtime.block_on(async move {
            let order = Self::collect_subtree_postorder(root).await;

            // 拆散节点树：对除根以外的所有节点执行 detach。
            // detach 会触发 GameLogic::on_detach（节点此前确实有 parent 时）。
            for entity in order.iter().take(order.len().saturating_sub(1)) {
                if entity.get_component::<Node>().await.is_some() {
                    let detach_future = {
                        let mut node = entity
                            .get_component_mut::<Node>()
                            .await
                            .expect("node component disappeared");
                        node.detach()
                    };
                    let _ = detach_future.await;
                }
            }

            // 退出时额外对根节点触发一次 on_detach（根节点没有 parent，detach 不会触发生命周期回调）。
            if let Some(root_node) = root.get_component::<Node>().await
                && let Some(handle) = root_node.logic().cloned()
            {
                let result = {
                    let mut logic = handle.lock().await;
                    logic.on_detach(root).await
                }
                .with_context(|| "root on_detach failed");

                if let Err(err) = result {
                    error!(
                        target: "jge-core",
                        error = %err,
                        "根实体调用 GameLogic::on_detach 失败"
                    );
                }
            }
        });
    }
}
