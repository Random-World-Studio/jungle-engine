use std::{
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use crate::sync::{Mutex, MutexGuard};
use tokio::{runtime::Runtime, task::JoinHandle};
use tracing::{error, info, warn};
use winit::{event_loop::EventLoop, window::Window};

use super::{
    component::{node::Node, scene2d::Scene2D},
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

/// 引擎运行时入口。
///
/// `Game` 把两件事绑在一起：
///
/// - `winit` 的事件循环（在 [`Game::run`] 中启动）
/// - 内部的 Tokio runtime（用于 ECS/逻辑/资源等异步任务）
///
/// # 窗口创建时机
///
/// 窗口是懒创建的：在 `winit` 回调（`resumed`）里才会创建。
/// 因此在调用 [`Game::run`] 之前，或在窗口尚未创建时，窗口访问方法会返回 `None`。
///
/// # Drop 注意事项
///
/// `Game` 在 `Drop` 中会执行一部分需要 `Runtime::block_on(...)` 的清理逻辑（例如更新可达性、触发 on_detach）。
/// 由于 tokio 不允许在一个 runtime 上下文里再调用另一个 runtime 的 `block_on`，
/// 若你在**外部 tokio runtime 上下文**里 drop 了 `Game`，引擎会跳过这些清理逻辑并输出警告日志，以避免 panic。
pub struct Game {
    config: GameConfig,
    window: Option<GameWindow>,

    render_snapshot: Arc<RwLock<Arc<RenderSnapshot>>>,
    framebuffer_size: Arc<RwLock<(u32, u32)>>,

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

        // `tokio::runtime::Runtime::block_on` 不能在另一个 tokio runtime 上下文中被调用。
        // 若用户在 tokio runtime 内 drop 了 `Game`，这里直接跳过需要 block_on 的清理逻辑，避免 panic。
        if tokio::runtime::Handle::try_current().is_ok() {
            warn!(
                target = "jge-core",
                "Game 被在 tokio runtime 内 drop：跳过可达性更新与节点树 on_detach 清理以避免 panic"
            );
            return;
        }

        self.runtime
            .block_on(set_subtree_reachable(self.root, false));
        self.detach_node_tree(self.root);
    }
}

impl Game {
    /// 创建一个新的 `Game`。
    ///
    /// - `root` 会被标记为“引擎根”，并把其子树的可达性设置为可达。
    /// - 如果 `root` 上存在 `Node` 且持有 `GameLogic`，会调度一次 `on_attach`。
    ///
    /// 注意：窗口不会在这里创建；窗口创建发生在 `run` 之后的 winit 生命周期回调中。
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

        let framebuffer_size = Arc::new(RwLock::new((1, 1)));
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

    /// 在引擎的 Tokio runtime 上 spawn 一个异步任务。
    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.runtime.spawn(future)
    }

    /// 在引擎 runtime 的 blocking 线程池上执行一个阻塞任务。
    pub fn spawn_blocking<F, R>(&self, func: F) -> JoinHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.runtime.spawn_blocking(func)
    }

    /// 在引擎 runtime 上阻塞等待一个 future 完成。
    ///
    /// 典型用法：在 `main` 里构建/挂载场景树（见教程）。
    pub fn block_on<F, R>(&self, future: F) -> R
    where
        F: Future<Output = R>,
    {
        self.runtime.block_on(future)
    }

    /// 同步获取 winit 的 `Window` 锁。
    ///
    /// 若窗口尚未创建（例如 `run` 之前），返回 `None`。
    pub fn winit_window(&self) -> Option<MutexGuard<'_, Window>> {
        Some(self.window.as_ref()?.window.lock())
    }

    /// 获取 `Arc<Mutex<Window>>`，便于在引擎外部线程里请求重绘等操作。
    ///
    /// 若窗口尚未创建，返回 `None`。
    pub fn winit_window_arc(&self) -> Option<Arc<Mutex<Window>>> {
        self.window
            .as_ref()
            .map(|window| Arc::clone(&window.window))
    }

    /// 尝试立即获取 `Window` 的锁（不等待）。
    ///
    /// - 窗口不存在：返回 `None`
    /// - 锁已被占用：返回 `None`
    pub fn try_winit_window(&self) -> Option<MutexGuard<'_, Window>> {
        self.window.as_ref()?.window.try_lock()
    }

    /// 设置窗口创建后的初始化回调。
    ///
    /// 回调会在窗口第一次创建完成后被调用一次，然后被丢弃（不会重复调用）。
    /// 适合在此处创建渲染资源、注册 UI/场景等需要窗口句柄的初始化逻辑。
    pub fn set_window_init<F>(&mut self, init: F)
    where
        F: FnMut(&mut Game) + Send + Sync + 'static,
    {
        self.window_init = Some(Box::new(init));
    }

    /// builder 风格的 [`Game::set_window_init`]。
    pub fn with_window_init<F>(mut self, init: F) -> Self
    where
        F: FnMut(&mut Game) + Send + Sync + 'static,
    {
        self.set_window_init(init);
        self
    }

    /// 设置事件映射器。
    ///
    /// 映射器会在 winit 回调线程被调用：将 `WindowEvent/DeviceEvent` 转换为引擎 [`GameEvent`]。
    /// 返回 `None` 表示忽略该事件。
    pub fn set_event_mapper<M>(&mut self, mapper: M)
    where
        M: EventMapper + 'static,
    {
        self.event_mapper = Box::new(mapper);
    }

    /// builder 风格的 [`Game::set_event_mapper`]。
    pub fn with_event_mapper<M>(mut self, mapper: M) -> Self
    where
        M: EventMapper + 'static,
    {
        self.set_event_mapper(mapper);
        self
    }

    /// 启动主循环。
    ///
    /// 该方法会：
    ///
    /// - 启动固定 tick 的 update loop
    /// - 启动 winit 事件循环
    pub fn run(mut self) -> anyhow::Result<()> {
        self.spawn_update_loop();

        let event_loop = EventLoop::new()?;
        Ok(event_loop.run_app(&mut self)?)
    }
}
