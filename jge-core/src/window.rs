use std::{sync::Arc, time::Duration};

use wgpu::Backends;
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    config::WindowConfig,
    game::{
        component::{layer::Layer, node::Node},
        entity::Entity,
        system::render::RenderSystem,
    },
};
use tracing::{debug, warn};

/// 窗口与渲染上下文。
///
/// `GameWindow` 封装了：
/// - `winit::Window`（以 `Arc` 形式共享）
/// - wgpu 的 `Surface/Device/Queue` 以及表面配置
/// - 引擎渲染系统 [`RenderSystem`]
///
/// 通常由 [`crate::Game`] 在窗口创建后内部构造并持有。
pub struct GameWindow {
    pub(crate) window: Arc<Window>,

    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surf_config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    size_changed: bool,
    renderer: RenderSystem,
}

impl GameWindow {
    /// 创建一个 `GameWindow`。
    ///
    /// 会初始化 wgpu 实例/适配器/设备，并根据 [`WindowConfig`] 配置表面（例如 vsync）。
    pub async fn new(window: Arc<Window>, window_config: &WindowConfig) -> anyhow::Result<Self> {
        let mut backends = Backends::all();
        backends.remove(Backends::GL);
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone())?;

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
        let mut size = window.inner_size();
        size.width = size.width.max(1);
        size.height = size.height.max(1);
        let surf_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: caps.formats[0],
            width: size.width,
            height: size.height,
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surf_config);

        Ok(Self {
            window,
            surface,
            device,
            queue,
            surf_config,
            size,
            size_changed: false,
            renderer: RenderSystem::new(),
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

    /// 渲染一帧。
    ///
    /// - `root`：场景树根实体（用于收集 Layer 根）
    /// - `delta`：上一帧到当前帧的时间间隔（用于统计/日志/可能的动画驱动）
    ///
    /// Layer 根的收集规则：
    /// - 从 `root` 开始深度优先遍历节点树。
    /// - 遇到挂载了 [`Layer`] 的实体时，将其视为一个可渲染 Layer 根：加入结果，并**不再深入遍历其子树**。
    ///   这样可以避免“嵌套 Layer”被重复渲染；子 Layer 会作为独立根在各自子树内单独处理。
    ///
    /// 返回 `wgpu::SurfaceError` 表示 surface 获取当前帧纹理失败，通常可通过重试/重建 surface 恢复。
    pub fn render(&mut self, root: &Entity, delta: Duration) -> Result<(), wgpu::SurfaceError> {
        debug!(
            target: "jge-core",
            frame_time_ms = delta.as_secs_f64() * 1000.0,
            "begin frame"
        );

        let layer_roots = Self::collect_layer_roots(*root);
        debug!(
            target: "jge-core",
            layer_count = layer_roots.len(),
            "collected layers for rendering"
        );

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        if layer_roots.is_empty() {
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
            for (index, layer_entity) in layer_roots.iter().enumerate() {
                let load_op = if index == 0 {
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                } else {
                    wgpu::LoadOp::Load
                };
                self.render_layer(&mut encoder, &view, *layer_entity, load_op);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }

    fn render_layer(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        layer_entity: Entity,
        load_op: wgpu::LoadOp<wgpu::Color>,
    ) {
        self.renderer.render_layer(
            &self.device,
            &self.queue,
            encoder,
            view,
            self.surf_config.format,
            (self.size.width, self.size.height),
            layer_entity,
            load_op,
        );
    }

    /// 从节点树中收集需要渲染的 Layer 根实体。
    ///
    /// 该方法与 [`Layer::renderable_entities`] 的“跳过嵌套 Layer 子树”策略保持一致。
    fn collect_layer_roots(root: Entity) -> Vec<Entity> {
        let mut result = Vec::new();
        let mut stack = Vec::new();
        stack.push(root);

        while let Some(entity) = stack.pop() {
            let node_guard = match entity.get_component::<Node>() {
                Some(node) => node,
                None => {
                    warn!(
                        target: "jge-core",
                        entity_id = entity.id(),
                        "entity missing Node component, skip layer traversal"
                    );
                    continue;
                }
            };

            let children: Vec<Entity> = node_guard.children().iter().copied().collect();
            drop(node_guard);

            if entity.get_component::<Layer>().is_some() {
                result.push(entity);
                continue;
            }

            for child in children.into_iter().rev() {
                stack.push(child);
            }
        }

        result
    }
}
