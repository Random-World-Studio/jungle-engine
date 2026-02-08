use std::{
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Instant,
};

use tokio::{runtime::Runtime, sync::Mutex, task::JoinHandle};
use tracing::{error, info};
use winit::{event_loop::EventLoop, window::Window};

use super::{
    component::{Component, node::Node, scene2d::Scene2D},
    entity::{Entity, EntityId},
    reachability::{register_engine_root, set_subtree_reachable, unregister_engine_root},
    system::logic::GameLogicHandle,
};
use crate::{
    config::{GameConfig, WindowConfig, WindowMode},
    event::{Event as GameEvent, EventMapper, NoopEventMapper},
    window::GameWindow,
};

use crate::game::system::render::RenderSnapshot;

use parking_lot::RwLock;

mod dispatch;
mod helpers;
mod init;
mod shutdown;
mod windowing;

#[cfg(test)]
mod tests;

type WindowInitFn = dyn FnMut(&mut Game) + Send + Sync;

pub struct Game {
    config: GameConfig,
    window: Option<GameWindow>,

    render_snapshot: Arc<RwLock<Arc<RenderSnapshot>>>,
    framebuffer_size: Arc<AtomicU64>,

    window_init: Option<Box<WindowInitFn>>,

    event_mapper: Box<dyn EventMapper>,

    root: Entity,

    last_redraw: Instant,
    stopped: Arc<AtomicBool>,

    runtime: Runtime,
}

impl Drop for Game {
    fn drop(&mut self) {
        // 停止调度循环，避免退出阶段仍然并发执行 update/on_event。
        self.stopped.store(true, Ordering::Release);
        unregister_engine_root(self.root);
        self.runtime
            .block_on(set_subtree_reachable(self.root, false));
        self.detach_node_tree(self.root);
    }
}

impl Game {
    pub fn new(config: GameConfig, root: Entity) -> anyhow::Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        info!(target: "jge-core", "Jungle Engine v{}", env!("CARGO_PKG_VERSION"));

        match runtime.block_on(root.get_component::<Node>()) {
            Some(root_node) => {
                let handle = root_node.logic().cloned();
                drop(root_node);

                if let Some(handle) = handle {
                    init::schedule_root_on_attach(&runtime, root, handle);
                }
            }
            None => {
                error!(target: "jge-core", "根实体缺少 Node 组件");
            }
        }

        // 标记该 root 为引擎根节点，并把其子树内的 Renderable 设为“可达”。
        // 这样当用户在 Game::new 之前就构建并挂载了一棵场景树时，可见性也能正确恢复。
        register_engine_root(root);
        runtime.block_on(set_subtree_reachable(root, true));

        let framebuffer_size = Arc::new(AtomicU64::new(helpers::pack_framebuffer_size(1, 1)));
        let render_snapshot = Arc::new(RwLock::new(Arc::new(RenderSnapshot::empty((1, 1)))));

        Ok(Self {
            config,
            window: None,

            render_snapshot,
            framebuffer_size,

            window_init: None,
            event_mapper: Box::new(NoopEventMapper),
            root,
            last_redraw: Instant::now(),
            stopped: Arc::new(AtomicBool::new(false)),
            runtime,
        })
    }

    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.runtime.spawn(future)
    }

    pub fn spawn_blocking<F, R>(&self, func: F) -> JoinHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.runtime.spawn_blocking(func)
    }

    pub fn block_on<F, R>(&self, future: F) -> R
    where
        F: Future<Output = R>,
    {
        self.runtime.block_on(future)
    }

    pub async fn winit_window(&self) -> Option<tokio::sync::MutexGuard<'_, Window>> {
        Some(self.window.as_ref()?.window.lock().await)
    }

    pub fn winit_window_arc(&self) -> Option<Arc<Mutex<Window>>> {
        self.window
            .as_ref()
            .map(|window| Arc::clone(&window.window))
    }

    pub fn try_winit_window(&self) -> Option<tokio::sync::MutexGuard<'_, Window>> {
        self.window.as_ref()?.window.try_lock().ok()
    }

    pub fn set_window_init<F>(&mut self, init: F)
    where
        F: FnMut(&mut Game) + Send + Sync + 'static,
    {
        self.window_init = Some(Box::new(init));
    }

    pub fn with_window_init<F>(mut self, init: F) -> Self
    where
        F: FnMut(&mut Game) + Send + Sync + 'static,
    {
        self.set_window_init(init);
        self
    }

    pub fn set_event_mapper<M>(&mut self, mapper: M)
    where
        M: EventMapper + 'static,
    {
        self.event_mapper = Box::new(mapper);
    }

    pub fn with_event_mapper<M>(mut self, mapper: M) -> Self
    where
        M: EventMapper + 'static,
    {
        self.set_event_mapper(mapper);
        self
    }

    pub fn run(mut self) -> anyhow::Result<()> {
        self.spawn_update_loop();

        let event_loop = EventLoop::new()?;
        Ok(event_loop.run_app(&mut self)?)
    }
}
