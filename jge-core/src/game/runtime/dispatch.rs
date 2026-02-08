use std::{
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use tokio::{task::JoinSet, time::interval};
use tracing::{trace, warn};

use super::helpers::{
    collect_logic_handle_chunks, rebuild_render_snapshot, unpack_framebuffer_size,
};
use super::{Entity, Game, GameEvent};

impl Game {
    async fn run_joinset<C, MakeTask, Fut>(
        chunks: Vec<C>,
        mut make_task: MakeTask,
        panic_message: &'static str,
    ) where
        C: Send + 'static,
        MakeTask: FnMut(C) -> Fut,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let mut join_set = JoinSet::new();

        for chunk in chunks {
            join_set.spawn(make_task(chunk));
        }

        while let Some(task) = join_set.join_next().await {
            if let Err(err) = task {
                warn!(target: "jge-core", error = %err, "{panic_message}");
            }
        }
    }

    pub(super) fn spawn_update_loop(&self) {
        let game_tick_ms = self.config.game_tick_ms;
        let stopped = Arc::clone(&self.stopped);
        let root = self.root;
        let render_snapshot = Arc::clone(&self.render_snapshot);
        let framebuffer_size: Arc<AtomicU64> = Arc::clone(&self.framebuffer_size);

        self.runtime.spawn(async move {
            let mut itv = interval(Duration::from_millis(game_tick_ms));
            let mut last_tick = Instant::now();

            while !stopped.load(Ordering::Acquire) {
                itv.tick().await;

                let delta = last_tick.elapsed();
                last_tick = Instant::now();

                Self::dispatch_update(delta).await;

                let (width, height) =
                    unpack_framebuffer_size(framebuffer_size.load(Ordering::Acquire));
                rebuild_render_snapshot(root, (width, height), render_snapshot.as_ref()).await;
            }
        });
    }

    async fn dispatch_update(delta: Duration) {
        let node_targets = collect_logic_handle_chunks().await;
        Self::run_joinset(
            node_targets,
            move |chunk| async move {
                for (entity_id, handle) in chunk {
                    let mut logic = handle.lock().await;
                    if let Err(err) = logic.update(Entity::from(entity_id), delta).await {
                        warn!(target: "jge-core", error = %err, "GameLogic update failed");
                    }
                }
            },
            "GameLogic update task panicked",
        )
        .await;
    }

    pub(super) fn dispatch_event(&self, event: GameEvent) {
        trace!(target: "jge-core", "dispatch event: {:?}", event);
        self.runtime.spawn(async move {
            let logic_targets = collect_logic_handle_chunks().await;
            Self::run_joinset(
                logic_targets,
                move |chunk| {
                    let event = event.clone();
                    async move {
                        for (entity_id, handle) in chunk {
                            let mut logic = handle.lock().await;
                            if let Err(err) = logic.on_event(Entity::from(entity_id), &event).await
                            {
                                warn!(
                                    target: "jge-core",
                                    error = %err,
                                    "GameLogic on_event failed"
                                );
                            }
                        }
                    }
                },
                "GameLogic on_event task panicked",
            )
            .await;
        });
    }

    pub(super) fn dispatch_on_render(&self, delta: Duration) {
        self.runtime.spawn(async move {
            let logic_targets = collect_logic_handle_chunks().await;
            Self::run_joinset(
                logic_targets,
                move |chunk| async move {
                    use crate::game::component::renderable::Renderable;

                    let logic_delta = delta;

                    for (entity_id, handle) in chunk {
                        let entity = Entity::from(entity_id);

                        // Renderable 的“实际可见性”需要直接影响 on_render 调度。
                        // - 有 Renderable：不可见则跳过。
                        // - 无 Renderable：保持原语义，仍调度。
                        if let Some(renderable) = entity.get_component::<Renderable>().await
                            && !renderable.is_enabled()
                        {
                            continue;
                        }

                        let mut logic = handle.lock().await;
                        if let Err(err) = logic.on_render(entity, logic_delta).await {
                            warn!(
                                target: "jge-core",
                                error = %err,
                                "GameLogic on_render failed"
                            );
                        }
                    }
                },
                "GameLogic on_render task panicked",
            )
            .await;
        });
    }
}
