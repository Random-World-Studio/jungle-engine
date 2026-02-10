use std::{
    future::Future,
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant},
};

use tokio::{task::JoinSet, time::interval};
use tracing::{trace, warn};

use super::helpers::collect_logic_handle_chunks;
use super::{Entity, Game, GameEvent};

impl Game {
    /// 用 `JoinSet` 并发执行每个“批处理分组（batch chunks）”的任务，并在最后统一收割。
    ///
    /// 注意这里的“chunk”仅指调度层的批处理分组（来自 `collect_logic_handle_chunks` / registry），
    /// 与旧的组件存储 chunk、LOD/分块等概念无关。
    ///
    /// 语义约定：
    ///
    /// - chunk 之间并发执行
    /// - chunk 内按顺序 `await`（同一 `GameLogicHandle` 的互斥锁保证同一时刻只会有一个逻辑在跑）
    /// - 任意任务 panic 时只记录日志，不中断其它 chunk
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

    /// 启动固定 tick 的 update loop。
    ///
    /// 每个 tick：
    ///
    /// 1) 调度 `GameLogic::update`
    ///
    /// 注意：`RenderSnapshot` 的重建不应绑定在 tick 上，否则会把渲染相关的可见更新
    /// 限制到 `game_tick_ms` 的频率（例如默认 50ms 会导致明显“跳帧/卡顿”）。
    /// 当前策略：在 `RedrawRequested` 渲染前按需（dirty）重建快照。
    pub(super) fn spawn_update_loop(&self) {
        let game_tick_ms = self.config.game_tick_ms;
        let stopped = Arc::clone(&self.stopped);
        let framebuffer_size = Arc::clone(&self.framebuffer_size);

        self.runtime.spawn(async move {
            let mut itv = interval(Duration::from_millis(game_tick_ms));
            let mut last_tick = Instant::now();

            while !stopped.load(Ordering::Acquire) {
                itv.tick().await;

                let delta = last_tick.elapsed();
                last_tick = Instant::now();

                Self::dispatch_update(delta).await;

                // 保留 framebuffer_size 读取，避免该值在没有窗口事件时长期为 (1,1)。
                // 实际 snapshot 重建在渲染线程（RedrawRequested）按需触发。
                let _ = *framebuffer_size.read();
            }
        });
    }

    /// 分发一次更新 tick（同步等待本轮所有批处理分组完成）。
    async fn dispatch_update(delta: Duration) {
        let node_targets = collect_logic_handle_chunks();
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

    /// 异步广播一个事件到所有逻辑。
    ///
    /// 注意：该方法会 `spawn`，不会等待所有逻辑处理完成。
    pub(super) fn dispatch_event(&self, event: GameEvent) {
        trace!(target: "jge-core", "dispatch event: {:?}", event);
        self.runtime.spawn(async move {
            let logic_targets = collect_logic_handle_chunks();
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

    /// 异步广播一帧渲染回调到所有逻辑。
    ///
    /// Renderable 的 enabled 状态会影响调度：存在且 disabled 的实体会被跳过。
    pub(super) fn dispatch_on_render(&self, delta: Duration) {
        // RedrawRequested 可能以非常高的频率触发（例如关闭 vsync、窗口空闲时持续 request_redraw）。
        // 若 on_render 本身比较重，持续 spawn 会导致任务堆积、CPU 争用，从而表现为“渲染卡顿”。
        // 这里保证同一时刻最多只有一个 on_render 广播在执行：
        // - 追帧没有意义（过期的 on_render 只会挤占 CPU）
        // - 允许在负载高时自动降频，优先保证渲染线程能拿到时间片
        if self
            .on_render_in_flight
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        let in_flight = Arc::clone(&self.on_render_in_flight);

        self.runtime.spawn(async move {
            struct ResetOnDrop(Arc<std::sync::atomic::AtomicBool>);
            impl Drop for ResetOnDrop {
                fn drop(&mut self) {
                    self.0.store(false, Ordering::Release);
                }
            }

            let _reset = ResetOnDrop(in_flight);

            let logic_targets = collect_logic_handle_chunks();
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
