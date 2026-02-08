use std::{sync::Arc, time::Duration};

use wgpu::Backends;
use wgpu::SurfaceTargetUnsafe;
use winit::{dpi::PhysicalSize, window::Window};

use tokio::sync::Mutex;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::{
    config::WindowConfig,
    game::system::render::{RenderSnapshot, RenderSystem},
};
use tokio::runtime::Runtime;
use tracing::trace;

/// 窗口与渲染上下文。
///
/// `GameWindow` 封装了：
/// - `winit::Window`（以 `Arc` 形式共享）
/// - wgpu 的 `Surface/Device/Queue` 以及表面配置
/// - 引擎渲染系统 [`RenderSystem`]
///
/// 通常由 [`crate::Game`] 在窗口创建后内部构造并持有。
pub struct GameWindow {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surf_config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    size_changed: bool,
    renderer: RenderSystem,

    pub(crate) window: Arc<Mutex<Window>>,
}

impl GameWindow {
    pub(crate) fn framebuffer_size(&self) -> (u32, u32) {
        (self.size.width, self.size.height)
    }

    /// 创建一个 `GameWindow`。
    ///
    /// 会初始化 wgpu 实例/适配器/设备，并根据 [`WindowConfig`] 配置表面（例如 vsync）。
    pub async fn new(
        window: Arc<Mutex<Window>>,
        window_config: &WindowConfig,
    ) -> anyhow::Result<Self> {
        let mut backends = Backends::all();
        backends.remove(Backends::GL);
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        // 注意：我们把 `winit::Window` 放进了 `Arc<tokio::sync::Mutex<_>>`，因此无法直接作为
        // `SurfaceTarget` 传入（它需要实现 HasWindowHandle/HasDisplayHandle）。
        // 这里改用 unsafe API：从 Window 取出 raw handle 创建 surface，并保证 Window 的生命周期
        // 由 `GameWindow.window` 持有，从而满足“handle 在 Surface 生命周期内保持有效”的安全约束。
        let (raw_display_handle, raw_window_handle, size) = {
            let guard = window.lock().await;
            let raw_display_handle = guard
                .display_handle()
                .map(|h| h.as_raw())
                .map_err(|e| anyhow::anyhow!("wgpu surface: display_handle unavailable: {e}"))?;
            let raw_window_handle = guard
                .window_handle()
                .map(|h| h.as_raw())
                .map_err(|e| anyhow::anyhow!("wgpu surface: window_handle unavailable: {e}"))?;

            let mut size = guard.inner_size();
            size.width = size.width.max(1);
            size.height = size.height.max(1);
            (raw_display_handle, raw_window_handle, size)
        };

        let surface = unsafe {
            instance.create_surface_unsafe(SurfaceTargetUnsafe::RawHandle {
                raw_display_handle,
                raw_window_handle,
            })?
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                label: None,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await?;

        let caps = surface.get_capabilities(&adapter);
        let desired_present_mode = if window_config.vsync {
            wgpu::PresentMode::Fifo
        } else {
            wgpu::PresentMode::Immediate
        };
        let present_mode = if caps.present_modes.contains(&desired_present_mode) {
            desired_present_mode
        } else {
            caps.present_modes[0]
        };
        let surf_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: caps.formats[0],
            width: size.width,
            height: size.height,
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &surf_config);

        Ok(Self {
            surface,
            device,
            queue,
            surf_config,
            size,
            size_changed: false,
            renderer: RenderSystem::new(),
            window,
        })
    }

    /// 标记窗口尺寸发生变化。
    ///
    /// 实际的 surface 重配发生在 [`Self::resize_surface_if_needed`]。
    pub fn set_window_resized(&mut self, new_size: PhysicalSize<u32>) {
        if new_size == self.size {
            return;
        }
        self.size = new_size;
        self.size_changed = true;
    }

    /// 如有需要，重新配置 wgpu surface。
    ///
    /// 典型调用点：收到 `WindowEvent::Resized` 或 `ScaleFactorChanged` 后。
    pub fn resize_surface_if_needed(&mut self) {
        if self.size_changed {
            self.surf_config.width = self.size.width;
            self.surf_config.height = self.size.height;
            self.surface.configure(&self.device, &self.surf_config);
            self.size_changed = false;
        }
    }

    pub fn render_snapshot(
        &mut self,
        runtime: &Runtime,
        snapshot: &RenderSnapshot,
        delta: Duration,
    ) -> Result<(), wgpu::SurfaceError> {
        trace!(
            target: "jge-core",
            frame_time_ms = delta.as_secs_f64() * 1000.0,
            "begin frame"
        );

        self.renderer.begin_frame();

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let layers = snapshot.layers();
        if layers.is_empty() {
            let ops = wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            };
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear Frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
        } else {
            for (index, layer_snapshot) in layers.iter().enumerate() {
                let load_op = if index == 0 {
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                } else {
                    wgpu::LoadOp::Load
                };
                self.renderer.render_layer_snapshot(
                    layer_snapshot,
                    crate::game::system::render::RenderLayerParams {
                        runtime,
                        device: &self.device,
                        queue: &self.queue,
                        encoder: &mut encoder,
                        target_view: &view,
                        surface_format: self.surf_config.format,
                        framebuffer_size: (self.size.width, self.size.height),
                        load_op,
                    },
                );
            }
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();

        self.renderer.end_frame();

        Ok(())
    }
}
