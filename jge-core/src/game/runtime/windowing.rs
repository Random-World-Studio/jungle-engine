use std::{
    sync::{Arc, atomic::Ordering},
    time::Instant,
};

use crate::sync::Mutex;
use tracing::{error, warn};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{Key, NamedKey},
    window::{Fullscreen, Window, WindowAttributes, WindowId},
};

use super::helpers::{rebuild_render_snapshot, update_scene2d_framebuffer_sizes};
use super::{Game, GameEvent, WindowConfig, WindowMode};
use crate::window::GameWindow;

impl Game {
    /// 在窗口尺寸已知的前提下，刷新 framebuffer_size 并重建渲染快照。
    ///
    /// 该方法会阻塞等待 snapshot 构建完成（在 winit 线程上执行）。
    fn refresh_framebuffer_and_snapshot(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);

        *self.framebuffer_size.write() = (width, height);
        self.runtime.block_on(rebuild_render_snapshot(
            self.root,
            (width, height),
            self.render_snapshot.as_ref(),
        ));
    }

    /// 异步更新场景树里所有 `Scene2D` 的 framebuffer_size。
    ///
    /// 用于 resize 后尽快修正 2D 坐标转换；不在 winit 回调中同步遍历，避免卡顿。
    fn spawn_update_scene2d_framebuffer_sizes(&self) {
        // 让 Scene2D 的坐标转换/可见范围在下一次 tick 前也能尽快感知到新尺寸。
        // 不在 winit 回调线程同步阻塞；改为异步任务遍历节点树并更新。
        let root = self.root;
        let framebuffer_size = Arc::clone(&self.framebuffer_size);
        self.runtime.spawn(async move {
            let (width, height) = *framebuffer_size.read();
            update_scene2d_framebuffer_sizes(root, (width, height)).await;
        });
    }

    /// 触发引擎退出：广播 CloseRequested，并退出 winit 事件循环。
    fn request_exit(&mut self, event_loop: &ActiveEventLoop) {
        self.dispatch_event(GameEvent::CloseRequested);
        self.stopped.store(true, Ordering::Release);
        event_loop.exit();
    }

    /// 处理窗口尺寸变化：更新 framebuffer_size、通知 GameWindow、并异步刷新所有 Scene2D 的 framebuffer。
    fn handle_window_resized(&mut self, physical_size: PhysicalSize<u32>) {
        if physical_size.width == 0 || physical_size.height == 0 {
            // 忽略最小化
            return;
        }

        *self.framebuffer_size.write() = (physical_size.width.max(1), physical_size.height.max(1));

        let Some(window) = self.window.as_mut() else {
            return;
        };
        window.set_window_resized(physical_size);

        self.spawn_update_scene2d_framebuffer_sizes();
    }

    /// 确保窗口与渲染后端已初始化。
    ///
    /// 窗口在 `resumed` 时懒创建；创建失败会记录错误并退出事件循环。
    pub(super) fn ensure_window_created(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attributes = Self::window_attributes(&self.config.window);
        let window = match event_loop.create_window(window_attributes) {
            Ok(window) => window,
            Err(err) => {
                error!(target: "jge-core", error = %err, "创建窗口失败");
                self.stopped.store(true, Ordering::Release);
                event_loop.exit();
                return;
            }
        };

        let window = Arc::new(Mutex::new(window));

        let game_window = match self
            .runtime
            .block_on(GameWindow::new(Arc::clone(&window), &self.config.window))
        {
            Ok(window) => window,
            Err(err) => {
                error!(target: "jge-core", error = %err, "初始化 GameWindow 失败");
                self.stopped.store(true, Ordering::Release);
                event_loop.exit();
                return;
            }
        };

        let (width, height) = game_window.framebuffer_size();
        self.window = Some(game_window);
        self.refresh_framebuffer_and_snapshot(width, height);

        if let Some(mut init) = self.window_init.take() {
            init(self);
        }
    }

    /// 处理 `RedrawRequested`：渲染当前快照并分发 `on_render`。
    pub(super) fn handle_redraw_requested(&mut self) {
        let delta = self.last_redraw.elapsed();
        self.last_redraw = Instant::now();

        {
            let Some(gwin) = self.window.as_mut() else {
                return;
            };

            gwin.resize_surface_if_needed();

            let snapshot = { self.render_snapshot.read().clone() };

            match gwin.render_snapshot(&self.runtime, snapshot.as_ref(), delta) {
                Ok(_) => {}
                // 展示平面的上下文丢失
                Err(wgpu::SurfaceError::Lost) => warn!(target: "jge-core", "Surface is lost"),
                // 所有其他错误
                Err(e) => warn!(target: "jge-core", "{:?}", e),
            }
        }

        self.dispatch_on_render(delta);
    }

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

impl ApplicationHandler for Game {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.ensure_window_created(event_loop);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // 在事件循环即将等待时请求重绘，确保高刷新率下能及时准备下一帧
        if let Some(window) = &self.window
            && let Some(guard) = window.window.try_lock()
        {
            guard.request_redraw();
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
                self.request_exit(event_loop);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } if self.config.escape_closes => {
                self.request_exit(event_loop);
            }
            WindowEvent::Resized(physical_size) => {
                self.handle_window_resized(physical_size);
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::GameConfig;
    use crate::game::component::node::Node;
    use crate::game::entity::Entity;

    #[test]
    fn handle_window_resized_updates_framebuffer_size_without_window() {
        let setup_rt = tokio::runtime::Runtime::new().expect("should create setup runtime");
        let root = setup_rt.block_on(async {
            let root = Entity::new().await.expect("should create root entity");
            root.register_component(Node::new("root").unwrap())
                .await
                .expect("should register root Node");
            root
        });
        drop(setup_rt);

        let mut game = Game::new(GameConfig::default(), root).expect("should create game");
        assert_eq!(*game.framebuffer_size.read(), (1, 1));

        game.handle_window_resized(PhysicalSize::new(800, 600));
        assert_eq!(*game.framebuffer_size.read(), (800, 600));
    }

    #[test]
    fn refresh_framebuffer_and_snapshot_clamps_size_to_at_least_1() {
        let setup_rt = tokio::runtime::Runtime::new().expect("should create setup runtime");
        let root = setup_rt.block_on(async {
            let root = Entity::new().await.expect("should create root entity");
            root.register_component(Node::new("root").unwrap())
                .await
                .expect("should register root Node");
            root
        });
        drop(setup_rt);

        let mut game = Game::new(GameConfig::default(), root).expect("should create game");
        game.refresh_framebuffer_and_snapshot(0, 10);
        assert_eq!(*game.framebuffer_size.read(), (1, 10));
    }
}
