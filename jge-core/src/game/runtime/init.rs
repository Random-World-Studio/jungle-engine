use anyhow::Context;
use tokio::runtime::Runtime;
use tracing::error;

use super::{Entity, GameLogicHandle};

pub(super) fn schedule_root_on_attach(runtime: &Runtime, root: Entity, handle: GameLogicHandle) {
    // 注意：tokio::sync::Mutex::blocking_lock 在 runtime 内会 panic。
    // 这里优先在“非 runtime 上下文”同步等待；若当前已在某个 runtime 内，则退化为异步 task。
    if tokio::runtime::Handle::try_current().is_ok() {
        runtime.spawn(async move {
            let result = {
                let mut logic = handle.lock().await;
                logic.on_attach(root).await
            }
            .with_context(|| "root on_attach failed");

            if let Err(err) = result {
                error!(
                    target: "jge-core",
                    error = %err,
                    "根实体调用 GameLogic::on_attach 失败"
                );
            }
        });
    } else {
        let result = runtime
            .block_on(async move {
                let mut logic = handle.lock().await;
                logic.on_attach(root).await
            })
            .with_context(|| "root on_attach failed");

        if let Err(err) = result {
            error!(
                target: "jge-core",
                error = %err,
                "根实体调用 GameLogic::on_attach 失败"
            );
        }
    }
}
