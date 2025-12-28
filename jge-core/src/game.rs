pub mod component;
pub mod entity;
pub mod logic;

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::{runtime::Runtime, task::JoinSet, time::interval};
use tracing::{error, info, warn};
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
    event::{
        DeviceEventMapper, Event as GameEvent, NoopDeviceEventMapper, NoopWindowEventMapper,
        WindowEventMapper,
    },
    game::{
        component::{Component, node::Node},
        entity::Entity,
        logic::GameLogicHandle,
    },
    window::GameWindow,
};

pub struct Game {
    config: GameConfig,
    window: Option<GameWindow>,

    window_init: Option<Box<dyn FnMut(&mut Game) + Send + Sync>>,

    window_event_mapper: Box<dyn WindowEventMapper>,
    device_event_mapper: Box<dyn DeviceEventMapper>,

    root: Entity,

    last_redraw: Instant,
    stopped: Arc<AtomicBool>,

    runtime: Runtime,
}

impl Game {
    pub fn new(config: GameConfig, root: Entity) -> anyhow::Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        info!(target: "jge-core", "Jungle Engine v{}", env!("CARGO_PKG_VERSION"));

        if root.get_component::<Node>().is_none() {
            error!(target: "jge-core", "根实体缺少 Node 组件");
        }

        Ok(Self {
            config,
            window: None,
            window_init: None,
            window_event_mapper: Box::new(NoopWindowEventMapper),
            device_event_mapper: Box::new(NoopDeviceEventMapper),
            root,
            last_redraw: Instant::now(),
            stopped: Arc::new(AtomicBool::new(false)),
            runtime,
        })
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

    /// Builder 风格：返回带窗口初始化回调的新 `Game`。
    pub fn with_window_init<F>(mut self, init: F) -> Self
    where
        F: FnMut(&mut Game) + Send + Sync + 'static,
    {
        self.set_window_init(init);
        self
    }

    /// 设置窗口事件映射器：把 `winit::WindowEvent` 转换为引擎事件并分发给 `GameLogic::on_event`。
    pub fn set_window_event_mapper<M>(&mut self, mapper: M)
    where
        M: WindowEventMapper + 'static,
    {
        self.window_event_mapper = Box::new(mapper);
    }

    /// Builder 风格：返回带映射器的新 `Game`。
    pub fn with_window_event_mapper<M>(mut self, mapper: M) -> Self
    where
        M: WindowEventMapper + 'static,
    {
        self.set_window_event_mapper(mapper);
        self
    }

    /// 设置设备事件映射器：把 `winit::DeviceEvent` 转换为引擎事件并分发给 `GameLogic::on_event`。
    pub fn set_device_event_mapper<M>(&mut self, mapper: M)
    where
        M: DeviceEventMapper + 'static,
    {
        self.device_event_mapper = Box::new(mapper);
    }

    /// Builder 风格：返回带设备事件映射器的新 `Game`。
    pub fn with_device_event_mapper<M>(mut self, mapper: M) -> Self
    where
        M: DeviceEventMapper + 'static,
    {
        self.set_device_event_mapper(mapper);
        self
    }

    pub fn run(mut self) -> anyhow::Result<()> {
        let game_tick_ms = self.config.game_tick_ms;
        let stopped = Arc::clone(&self.stopped);
        self.runtime.spawn(async move {
            let mut itv = interval(Duration::from_millis(game_tick_ms));
            let mut last_tick = Instant::now();

            while !stopped.load(Ordering::Acquire) {
                itv.tick().await;

                let delta = last_tick.elapsed();
                last_tick = Instant::now();

                let mut join_set = JoinSet::new();

                let node_targets = Game::collect_logic_handles();
                for (entity_id, handle) in node_targets {
                    let delta = delta;
                    join_set.spawn(async move {
                        let mut logic = handle.lock().await;
                        logic.update(Entity::from(entity_id), delta).await?;
                        Ok::<u64, anyhow::Error>(entity_id)
                    });
                }

                while let Some(task) = join_set.join_next().await {
                    match task {
                        Ok(Ok(_entity_id)) => {}
                        Ok(Err(err)) => {
                            warn!(target: "jge-core", error = %err, "GameLogic update task failed");
                        }
                        Err(err) => {
                            warn!(target: "jge-core", error = %err, "GameLogic update task panicked");
                        }
                    }
                }
            }
        });

        let event_loop = EventLoop::new()?;
        Ok(event_loop.run_app(&mut self)?)
    }

    fn dispatch_event(&self, event: GameEvent) {
        let logic_targets = Game::collect_logic_handles();
        self.runtime.spawn(async move {
            let mut join_set = JoinSet::new();

            for (entity_id, handle) in logic_targets {
                let event = event.clone();
                join_set.spawn(async move {
                    let mut logic = handle.lock().await;
                    logic.on_event(Entity::from(entity_id), &event).await?;
                    Ok::<u64, anyhow::Error>(entity_id)
                });
            }

            while let Some(task) = join_set.join_next().await {
                match task {
                    Ok(Ok(_entity_id)) => {}
                    Ok(Err(err)) => {
                        warn!(target: "jge-core", error = %err, "GameLogic on_event task failed");
                    }
                    Err(err) => {
                        warn!(target: "jge-core", error = %err, "GameLogic on_event task panicked");
                    }
                }
            }
        });
    }
}

impl ApplicationHandler for Game {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
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

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(mapped) = self.window_event_mapper.map_window_event(&event) {
            self.dispatch_event(mapped);
        }

        match event {
            WindowEvent::CloseRequested => {
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
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } if self.config.escape_closes => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let delta = self.last_redraw.elapsed();
                self.last_redraw = Instant::now();

                {
                    let gwin = self.window.as_mut().unwrap();

                    gwin.resize_surface_if_needed();

                    gwin.window.pre_present_notify();

                    match gwin.render(&self.root, delta) {
                        Ok(_) => {}
                        // 展示平面的上下文丢失
                        Err(wgpu::SurfaceError::Lost) => {
                            warn!(target: "jge-core", "Surface is lost")
                        }
                        // 所有其他错误
                        Err(e) => warn!(target: "jge-core", "{:?}", e),
                    }
                    gwin.window.request_redraw();
                }

                let logic_targets = Game::collect_logic_handles();
                let frame_delta = delta;
                self.runtime.spawn(async move {
                    let mut join_set = JoinSet::new();

                    for (entity_id, handle) in logic_targets {
                        let logic_delta = frame_delta;
                        join_set.spawn(async move {
                            let mut logic = handle.lock().await;
                            logic.on_render(Entity::from(entity_id), logic_delta).await?;
                            Ok::<u64, anyhow::Error>(entity_id)
                        });
                    }

                    while let Some(task) = join_set.join_next().await {
                        match task {
                            Ok(_) => {}
                            Err(err) => {
                            warn!(target: "jge-core", error = %err, "GameLogic on_render task panicked");
                            }
                        }
                    }
                });
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
        if let Some(mapped) = self.device_event_mapper.map_device_event(&event) {
            self.dispatch_event(mapped);
        }
    }
}

impl Game {
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

    fn collect_logic_handles() -> Vec<(u64, GameLogicHandle)> {
        Node::storage()
            .collect_with(|entity_id, node| node.logic().cloned().map(|logic| (entity_id, logic)))
    }
}
