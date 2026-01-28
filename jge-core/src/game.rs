//! ECS / 场景树 / 系统模块。
//!
//! 你会在这里找到：
//! - [`entity::Entity`]：实体句柄与组件读写入口
//! - [`component`]：内置组件（渲染、摄像机、灯光、节点树等）与组件访问 guard
//! - [`system`]：逻辑与渲染系统（通常由 [`crate::Game`] 自动驱动）
//!
//! 大多数游戏项目并不需要直接操作系统层；更推荐通过：
//! - [`crate::scene!`] 构建场景树（得到根实体）
//! - 在组件上配置渲染/逻辑能力
//! - 在 [`system::logic::GameLogic`] 中编写每帧更新与事件响应

pub mod component;
pub mod entity;
pub mod system;

use anyhow::Context;
use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::{runtime::Runtime, task::JoinSet, time::interval};
use tracing::{error, info, trace, warn};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Fullscreen, Window, WindowAttributes, WindowId},
};

use crate::{
    config::{GameConfig, WindowConfig, WindowMode},
    event::{Event as GameEvent, EventMapper, NoopEventMapper},
    game::{
        component::{Component, node::Node},
        entity::Entity,
        system::logic::GameLogicHandle,
    },
    window::GameWindow,
};

/// 引擎运行时入口。
///
/// `Game` 负责：
/// - 创建并持有窗口（基于 `winit`）
/// - 驱动 tick/update、事件分发与渲染
/// - 管理内部 Tokio runtime，用于并发执行游戏逻辑
///
/// # 最小示例
///
/// ```no_run
/// fn main() -> ::anyhow::Result<()> {
///     ::jge_core::logger::init()?;
///
///     let root = ::jge_core::game::entity::Entity::new()?;
///     let game = ::jge_core::Game::new(::jge_core::config::GameConfig::default(), root)?;
///
///     game.run()?;
///     Ok(())
/// }
/// ```
pub struct Game {
    config: GameConfig,
    window: Option<GameWindow>,

    window_init: Option<Box<dyn FnMut(&mut Game) + Send + Sync>>,

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
        Self::detach_node_tree(self.root);
    }
}

impl Game {
    /// 创建一个新的引擎实例。
    ///
    /// `root` 是场景树根实体，通常应已挂载 [`Node`] 组件；若缺失会记录错误日志（但仍会返回 `Ok`）。
    ///
    /// 通常你会用 [`crate::scene!`] 来构建 `root`：
    ///
    /// ```no_run
    /// fn build_root() -> ::anyhow::Result<::jge_core::game::entity::Entity> {
    ///     let bindings = ::jge_core::scene! {
    ///         node "root" as root {
    ///             // ... 在这里挂 Layer/Scene/Renderable 等组件
    ///         }
    ///     }?;
    ///     Ok(bindings.root)
    /// }
    /// ```
    pub fn new(config: GameConfig, root: Entity) -> anyhow::Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        info!(target: "jge-core", "Jungle Engine v{}", env!("CARGO_PKG_VERSION"));

        if let Some(rootn) = root.get_component::<Node>() {
            if let Some(logic) = rootn.logic() {
                if let Err(err) = logic
                    .blocking_lock()
                    .on_attach(root)
                    .with_context(|| "root on_attach failed")
                {
                    error!(target: "jge-core", error = %err, "根实体调用 GameLogic::on_attach 失败");
                }
            }
        } else {
            error!(target: "jge-core", "根实体缺少 Node 组件");
        }

        Ok(Self {
            config,
            window: None,
            window_init: None,
            event_mapper: Box::new(NoopEventMapper),
            root,
            last_redraw: Instant::now(),
            stopped: Arc::new(AtomicBool::new(false)),
            runtime,
        })
    }

    fn collect_subtree_postorder(root: Entity) -> Vec<Entity> {
        fn walk(
            entity: Entity,
            visited: &mut HashSet<crate::game::entity::EntityId>,
            out: &mut Vec<Entity>,
        ) {
            if !visited.insert(entity.id()) {
                return;
            }

            if let Some(node) = entity.get_component::<Node>() {
                let children = node.children().to_vec();
                drop(node);
                for child in children {
                    walk(child, visited, out);
                }
            }

            out.push(entity);
        }

        let mut visited = HashSet::new();
        let mut out = Vec::new();
        walk(root, &mut visited, &mut out);
        out
    }

    fn detach_node_tree(root: Entity) {
        // 没有 Node 的根实体不参与拆树。
        if root.get_component::<Node>().is_none() {
            return;
        }

        let order = Self::collect_subtree_postorder(root);

        // 拆散节点树：对除根以外的所有节点执行 detach。
        // detach 会触发 GameLogic::on_detach（节点此前确实有 parent 时）。
        for entity in order.iter().take(order.len().saturating_sub(1)) {
            if let Some(mut node) = entity.get_component_mut::<Node>() {
                let _ = node.detach();
            }
        }

        // 退出时额外对根节点触发一次 on_detach（根节点没有 parent，detach 不会触发生命周期回调）。
        if let Some(root_node) = root.get_component::<Node>() {
            if let Some(handle) = root_node.logic().cloned() {
                let result = handle
                    .blocking_lock()
                    .on_detach(root)
                    .with_context(|| "root on_detach failed");
                if let Err(err) = result {
                    error!(target: "jge-core", error = %err, "根实体调用 GameLogic::on_detach 失败");
                }
            }
        }
    }

    /// 获取内部 `winit::window::Window` 的共享引用。
    ///
    /// 注意：窗口在 `ApplicationHandler::resumed` 之后才会创建；因此在 `run()` 之前调用通常会返回 `None`。
    pub fn winit_window(&self) -> Option<&Window> {
        self.window.as_ref().map(|window| window.window.as_ref())
    }

    /// 获取内部 `winit::window::Window` 的共享引用（通过 `&mut self` 访问）。
    ///
    /// 由于窗口在引擎内通过 `Arc<Window>` 持有，这里返回的是 `&Window`（`winit::Window` 的大多数操作只需要 `&self`）。
    pub fn winit_window_mut(&mut self) -> Option<&Window> {
        self.window.as_ref().map(|window| window.window.as_ref())
    }

    /// 获取内部 `winit::window::Window` 的 `Arc`（可跨闭包/线程保存）。
    pub fn winit_window_arc(&self) -> Option<Arc<Window>> {
        self.window
            .as_ref()
            .map(|window| Arc::clone(&window.window))
    }

    /// 设置窗口创建后的初始化回调。
    ///
    /// 回调会在 `resumed` 创建窗口后立即执行一次，常用于设置光标抓取、隐藏光标等与窗口相关的初始化逻辑。
    pub fn set_window_init<F>(&mut self, init: F)
    where
        F: FnMut(&mut Game) + Send + Sync + 'static,
    {
        self.window_init = Some(Box::new(init));
    }

    /// 构建器风格：返回带窗口初始化回调的新 `Game`。
    pub fn with_window_init<F>(mut self, init: F) -> Self
    where
        F: FnMut(&mut Game) + Send + Sync + 'static,
    {
        self.set_window_init(init);
        self
    }

    /// 设置事件映射器：把 `winit::WindowEvent` 转换为引擎事件并分发给 `GameLogic::on_event`。
    pub fn set_event_mapper<M>(&mut self, mapper: M)
    where
        M: EventMapper + 'static,
    {
        self.event_mapper = Box::new(mapper);
    }

    /// 构建器风格：返回带映射器的新 `Game`。
    pub fn with_event_mapper<M>(mut self, mapper: M) -> Self
    where
        M: EventMapper + 'static,
    {
        self.set_event_mapper(mapper);
        self
    }

    /// 运行引擎主循环（阻塞当前线程）。
    ///
    /// 该方法会：
    /// - 在内部 Tokio runtime 中启动固定 tick 的 `GameLogic::update` 调度任务
    /// - 创建 `winit` 事件循环并进入 `run_app`，处理窗口/输入事件与渲染
    ///
    /// 返回值：
    /// - 当事件循环正常退出时返回 `Ok(())`
    /// - 当 `winit` 创建事件循环或运行时出现错误时返回 `Err`
    pub fn run(mut self) -> anyhow::Result<()> {
        self.spawn_update_loop();

        let event_loop = EventLoop::new()?;
        Ok(event_loop.run_app(&mut self)?)
    }

    /// 在内部 runtime 中启动固定 tick 的 `GameLogic::update` 调度循环。
    ///
    /// 调度语义：
    /// - tick 周期由 `config.game_tick_ms` 决定。
    /// - 每个 tick 会等待本轮 `update` 调度全部完成后再进入下一轮（避免积压）。
    /// - 以 chunk 为并发粒度：每个 chunk 一个任务并行执行；chunk 内按存储顺序顺序 `await`。
    ///
    /// 停止条件：当 `stopped` 被置为 `true`（例如窗口关闭）时退出循环。
    fn spawn_update_loop(&self) {
        let game_tick_ms = self.config.game_tick_ms;
        let stopped = Arc::clone(&self.stopped);
        self.runtime.spawn(async move {
            let mut itv = interval(Duration::from_millis(game_tick_ms));
            let mut last_tick = Instant::now();

            while !stopped.load(Ordering::Acquire) {
                itv.tick().await;

                let delta = last_tick.elapsed();
                last_tick = Instant::now();

                Game::dispatch_update(delta).await;
            }
        });
    }

    /// 执行一次 `GameLogic::update` 调度（会等待所有 chunk 完成）。
    ///
    /// 错误处理：
    /// - 单个逻辑返回 `Err`：记录 `warn` 并继续处理同 chunk 的后续逻辑。
    /// - chunk 任务 panic：记录 `warn`；其他 chunk 不受影响。
    async fn dispatch_update(delta: Duration) {
        let mut join_set = JoinSet::new();

        let node_targets = Game::collect_logic_handle_chunks();
        for chunk in node_targets {
            let delta = delta;
            join_set.spawn(async move {
                for (entity_id, handle) in chunk {
                    let mut logic = handle.lock().await;
                    if let Err(err) = logic.update(Entity::from(entity_id), delta).await {
                        warn!(
                            target: "jge-core",
                            error = %err,
                            "GameLogic update failed"
                        );
                    }
                }
            });
        }

        while let Some(task) = join_set.join_next().await {
            if let Err(err) = task {
                warn!(
                    target: "jge-core",
                    error = %err,
                    "GameLogic update task panicked"
                );
            }
        }
    }

    /// 分发一条引擎事件到所有已注册逻辑（`GameLogic::on_event`）。
    ///
    /// 调度语义：
    /// - 以 chunk 为并发粒度：每个 chunk 启动一个异步任务并行执行。
    /// - chunk 内按存储顺序顺序执行：同一 chunk 中多个逻辑依次 `await`，避免为每个逻辑单独建任务。
    /// - 本方法为“发射后不等待”（fire-and-forget）：不会阻塞 `winit` 事件回调线程。
    /// - 事件对象会为每个 chunk `clone` 一份：用于把同一事件同时发送给多个并发任务。
    ///   这要求事件类型可克隆，并带来一定的内存/拷贝开销；如果事件 payload 很大，建议把 payload 设计为
    ///   共享所有权（例如 `Arc<T>`），以降低 clone 成本。
    ///
    /// 错误处理：
    /// - 单个逻辑返回 `Err`：记录 `warn` 并继续处理同 chunk 的后续逻辑。
    /// - chunk 任务 panic：记录 `warn`；其他 chunk 不受影响。
    fn dispatch_event(&self, event: GameEvent) {
        trace!(target: "jge-core", "dispatch event: {:?}", event);
        let logic_targets = Game::collect_logic_handle_chunks();
        self.runtime.spawn(async move {
            let mut join_set = JoinSet::new();

            for chunk in logic_targets {
                let event = event.clone();
                join_set.spawn(async move {
                    for (entity_id, handle) in chunk {
                        let mut logic = handle.lock().await;
                        if let Err(err) = logic.on_event(Entity::from(entity_id), &event).await {
                            warn!(
                                target: "jge-core",
                                error = %err,
                                "GameLogic on_event failed"
                            );
                        }
                    }
                });
            }

            while let Some(task) = join_set.join_next().await {
                if let Err(err) = task {
                    warn!(target: "jge-core", error = %err, "GameLogic on_event task panicked");
                }
            }
        });
    }
}

impl ApplicationHandler for Game {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.ensure_window_created(event_loop);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // 在事件循环即将等待时请求重绘，确保高刷新率下能及时准备下一帧
        if let Some(window) = &self.window {
            window.window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(mapped) = self.event_mapper.map_window_event(&event) {
            self.dispatch_event(mapped);
        }

        match event {
            WindowEvent::CloseRequested => {
                self.dispatch_event(GameEvent::CloseRequested);
                self.stopped.store(true, Ordering::Release);
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } if self.config.escape_closes => {
                self.dispatch_event(GameEvent::CloseRequested);
                self.stopped.store(true, Ordering::Release);
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                if physical_size.width == 0 || physical_size.height == 0 {
                    // 忽略最小化
                } else {
                    self.window
                        .as_mut()
                        .unwrap()
                        .set_window_resized(physical_size);
                }
            }
            WindowEvent::RedrawRequested => {
                self.handle_redraw_requested();
            }
            _ => (),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(mapped) = self.event_mapper.map_device_event(&event) {
            self.dispatch_event(mapped);
        }
    }
}

impl Game {
    /// 确保窗口与渲染上下文已创建。
    ///
    /// 该方法只会创建一次：如果窗口已存在会直接返回。
    ///
    /// 创建完成后会立即执行 `window_init` 回调（若存在），以便进行与窗口绑定的初始化逻辑。
    fn ensure_window_created(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attributes = Self::window_attributes(&self.config.window);
        let window = event_loop.create_window(window_attributes).unwrap();

        self.window = Some(
            self.runtime
                .block_on(GameWindow::new(Arc::new(window), &self.config.window))
                .unwrap(),
        );

        if let Some(mut init) = self.window_init.take() {
            init(self);
        }
    }

    /// 处理 `winit` 的重绘请求：执行一帧渲染，并异步调度 `GameLogic::on_render`。
    ///
    /// 注意：`on_render` 的调度与渲染本身解耦，不会阻塞当前 `winit` 回调。
    /// 重绘请求通过 `AboutToWait` 事件触发，以确保在高刷新率VSync下能及时准备下一帧。
    fn handle_redraw_requested(&mut self) {
        let delta = self.last_redraw.elapsed();
        self.last_redraw = Instant::now();

        {
            let gwin = self.window.as_mut().unwrap();

            gwin.resize_surface_if_needed();

            match gwin.render(&self.root, delta) {
                Ok(_) => {}
                // 展示平面的上下文丢失
                Err(wgpu::SurfaceError::Lost) => warn!(target: "jge-core", "Surface is lost"),
                // 所有其他错误
                Err(e) => warn!(target: "jge-core", "{:?}", e),
            }
        }

        self.dispatch_on_render(delta);
    }

    /// 调度一帧渲染回调到所有已注册逻辑（`GameLogic::on_render`）。
    ///
    /// 调度语义：
    /// - 以 chunk 为并发粒度：每个 chunk 启动一个异步任务并行执行。
    /// - chunk 内按存储顺序顺序执行：同一 chunk 中多个逻辑依次 `await`。
    /// - 本方法为“发射后不等待”（fire-and-forget）。
    ///
    /// 错误处理：
    /// - 单个逻辑返回 `Err`：记录 `warn` 并继续处理同 chunk 的后续逻辑。
    /// - chunk 任务 panic：记录 `warn`；其他 chunk 不受影响。
    fn dispatch_on_render(&self, delta: Duration) {
        let logic_targets = Game::collect_logic_handle_chunks();
        self.runtime.spawn(async move {
            let mut join_set = JoinSet::new();

            for chunk in logic_targets {
                let logic_delta = delta;
                join_set.spawn(async move {
                    for (entity_id, handle) in chunk {
                        let mut logic = handle.lock().await;
                        if let Err(err) =
                            logic.on_render(Entity::from(entity_id), logic_delta).await
                        {
                            warn!(
                                target: "jge-core",
                                error = %err,
                                "GameLogic on_render failed"
                            );
                        }
                    }
                });
            }

            while let Some(task) = join_set.join_next().await {
                if let Err(err) = task {
                    warn!(
                        target: "jge-core",
                        error = %err,
                        "GameLogic on_render task panicked"
                    );
                }
            }
        });
    }

    /// 根据窗口配置构造 `winit` 的窗口属性。
    fn window_attributes(window_config: &WindowConfig) -> WindowAttributes {
        let width = window_config.width.max(1);
        let height = window_config.height.max(1);

        let mut attributes = Window::default_attributes()
            .with_title(window_config.title.clone())
            .with_inner_size(PhysicalSize::new(width, height));

        if matches!(window_config.mode, WindowMode::Fullscreen) {
            attributes = attributes.with_fullscreen(Some(Fullscreen::Borderless(None)));
        }

        attributes
    }

    /// 收集当前世界里所有挂载了 `GameLogic` 的实体，并按存储的 chunk 组织。
    ///
    /// 返回值的 chunk 划分与顺序由 `Node` 的内部存储决定；调用方通常会对每个 chunk 并发执行，
    /// 同时保持 chunk 内顺序遍历，以获得更好的缓存局部性并降低任务开销。
    fn collect_logic_handle_chunks() -> Vec<Vec<(crate::game::entity::EntityId, GameLogicHandle)>> {
        Node::storage().collect_chunks_with(|entity_id, node| {
            node.logic().cloned().map(|logic| (entity_id, logic))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::component::node::Node;
    use crate::game::system::logic::GameLogic;
    use std::sync::{Arc, Mutex as StdMutex};

    struct TrackingLogic {
        label: &'static str,
        events: Arc<StdMutex<Vec<&'static str>>>,
    }

    #[async_trait::async_trait]
    impl GameLogic for TrackingLogic {
        fn on_attach(&mut self, _e: Entity) -> anyhow::Result<()> {
            self.events.lock().unwrap().push(if self.label == "root" {
                "root_attach"
            } else {
                "child_attach"
            });
            Ok(())
        }

        fn on_detach(&mut self, _e: Entity) -> anyhow::Result<()> {
            self.events.lock().unwrap().push(if self.label == "root" {
                "root_detach"
            } else {
                "child_detach"
            });
            Ok(())
        }
    }

    #[test]
    fn game_drop_detaches_tree_and_calls_root_on_detach() {
        let events = Arc::new(StdMutex::new(Vec::new()));

        let root = Entity::new().expect("should create root entity");
        root.register_component(Node::new("root").unwrap())
            .expect("should register root Node");
        {
            let mut node = root.get_component_mut::<Node>().unwrap();
            node.set_logic(TrackingLogic {
                label: "root",
                events: events.clone(),
            });
        }

        let child = Entity::new().expect("should create child entity");
        child
            .register_component(Node::new("child").unwrap())
            .expect("should register child Node");
        {
            let mut node = child.get_component_mut::<Node>().unwrap();
            node.set_logic(TrackingLogic {
                label: "child",
                events: events.clone(),
            });
        }

        let game = Game::new(GameConfig::default(), root).expect("should create game");

        {
            let mut root_node = root.get_component_mut::<Node>().unwrap();
            root_node.attach(child).unwrap();
        }

        drop(game);

        let log = events.lock().unwrap().clone();
        assert_eq!(
            log.as_slice(),
            &["root_attach", "child_attach", "child_detach", "root_detach"]
        );
    }
}
