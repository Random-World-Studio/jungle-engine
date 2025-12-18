use std::{sync::Arc, time::Duration};

use wgpu::Backends;
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    config::WindowConfig,
    game::{
        component::{
            Component,
            layer::{Layer, LayerRenderContext, LayerRendererCache},
            node::Node,
        },
        entity::Entity,
    },
};
use tracing::{debug, warn};

pub struct GameWindow {
    pub(crate) window: Arc<Window>,

    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surf_config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    size_changed: bool,
    layer_caches: LayerRendererCache,
}

impl GameWindow {
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
            layer_caches: LayerRendererCache::new(),
        })
    }

    pub fn set_window_resized(&mut self, new_size: PhysicalSize<u32>) {
        if new_size == self.size {
            return;
        }
        self.size = new_size;
        self.size_changed = true;
    }

    pub fn resize_surface_if_needed(&mut self) {
        if self.size_changed {
            self.surf_config.width = self.size.width;
            self.surf_config.height = self.size.height;
            self.surface.configure(&self.device, &self.surf_config);
            self.size_changed = false;
        }
    }

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
        let mut context = LayerRenderContext {
            device: &self.device,
            queue: &self.queue,
            encoder,
            target_view: view,
            surface_format: self.surf_config.format,
            framebuffer_size: (self.size.width, self.size.height),
            load_op,
            caches: &mut self.layer_caches,
        };

        if let Some(layer) = Layer::read(layer_entity) {
            layer.render(&mut context);
        } else {
            warn!(
                target: "jge-core",
                layer_id = layer_entity.id(),
                "skip rendering for entity missing Layer component"
            );
        }
    }

    fn collect_layer_roots(root: Entity) -> Vec<Entity> {
        let mut result = Vec::new();
        let mut stack = Vec::new();
        stack.push(root);

        while let Some(entity) = stack.pop() {
            let node_guard = match Node::read(entity) {
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

            if Layer::read(entity).is_some() {
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
