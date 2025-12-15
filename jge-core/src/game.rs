pub mod component;
pub mod entity;
pub mod logic;

use std::{sync::Arc, time::Instant};
use tokio::runtime::Runtime;
use tracing::{error, info, warn};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Fullscreen, Window, WindowAttributes, WindowId},
};

use crate::{
    config::{GameConfig, WindowConfig, WindowMode},
    game::{component::node::Node, entity::Entity},
    window::GameWindow,
};

pub struct Game {
    config: GameConfig,
    window: Option<GameWindow>,

    root: Entity,

    last_redraw: Instant,

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
            root,
            last_redraw: Instant::now(),
            runtime,
        })
    }

    pub fn run(mut self) -> anyhow::Result<()> {
        let event_loop = EventLoop::new()?;
        Ok(event_loop.run_app(&mut self)?)
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
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
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

                let gwin = self.window.as_mut().unwrap();

                gwin.resize_surface_if_needed();

                gwin.window.pre_present_notify();

                match gwin.render(&self.root, delta) {
                    Ok(_) => {}
                    // 当展示平面的上下文丢失，就需重新配置
                    Err(wgpu::SurfaceError::Lost) => warn!(target: "jge-core", "Surface is lost"),
                    // 所有其他错误（过期、超时等）应在下一帧解决
                    Err(e) => warn!(target: "jge-core", "{:?}", e),
                }
                // 除非我们手动请求，RedrawRequested 将只会触发一次。
                gwin.window.request_redraw();
            }
            _ => (),
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
}
