use std::{borrow::Cow, collections::HashMap, sync::Arc};

use anyhow::Context;
use image::GenericImageView;
use tracing::{trace, warn};
use wgpu::{self, util::DeviceExt};

use crate::game::{
    component::{background::Background, node::Node},
    entity::Entity,
};
use crate::resource::{Resource, ResourceHandle};

use super::RenderSystem;
use super::cache::LayerRenderContext;
use super::util;
use crate::game::component::{camera::Camera, scene3d::Scene3D};

const BACKGROUND_VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: (4 * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &[
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x2,
        },
        wgpu::VertexAttribute {
            offset: (2 * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32x2,
        },
    ],
};

pub(in crate::game::system::render) struct BackgroundPipeline {
    pub(in crate::game::system::render) pipeline: wgpu::RenderPipeline,
    pub(in crate::game::system::render) bind_group_layout: wgpu::BindGroupLayout,
    vertex_shader: ResourceHandle,
    fragment_shader: ResourceHandle,
    pub(in crate::game::system::render) vertex_buffer: wgpu::Buffer,
}

impl BackgroundPipeline {
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        vertex_shader: ResourceHandle,
        fragment_shader: ResourceHandle,
    ) -> anyhow::Result<Self> {
        let vertex_source = load_shader_source(&vertex_shader)?;
        let fragment_source = load_shader_source(&fragment_shader)?;

        let vertex_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Background Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(vertex_source)),
        });

        let fragment_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Background Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(fragment_source)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Background Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Background Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Background Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_module,
                entry_point: Some("vs_main"),
                buffers: &[BACKGROUND_VERTEX_LAYOUT],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        let vertex_data: [f32; 24] = [
            -1.0, -1.0, 0.0, 1.0, //
            1.0, -1.0, 1.0, 1.0, //
            -1.0, 1.0, 0.0, 0.0, //
            -1.0, 1.0, 0.0, 0.0, //
            1.0, -1.0, 1.0, 1.0, //
            1.0, 1.0, 1.0, 0.0, //
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Background Vertex Buffer"),
            contents: cast_slice_f32(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            vertex_shader,
            fragment_shader,
            vertex_buffer,
        })
    }

    fn matches(&self, vertex: &ResourceHandle, fragment: &ResourceHandle) -> bool {
        Arc::ptr_eq(&self.vertex_shader, vertex) && Arc::ptr_eq(&self.fragment_shader, fragment)
    }
}

pub(in crate::game::system::render) struct BackgroundPipelineCache {
    pipeline: Option<BackgroundPipeline>,
    generation: u64,
}

impl Default for BackgroundPipelineCache {
    fn default() -> Self {
        Self {
            pipeline: None,
            generation: 0,
        }
    }
}

impl BackgroundPipelineCache {
    pub(in crate::game::system::render) fn ensure(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        vertex: ResourceHandle,
        fragment: ResourceHandle,
    ) -> anyhow::Result<&BackgroundPipeline> {
        let needs_rebuild = self
            .pipeline
            .as_ref()
            .map(|pipeline| !pipeline.matches(&vertex, &fragment))
            .unwrap_or(true);

        if needs_rebuild {
            self.pipeline = Some(BackgroundPipeline::new(device, format, vertex, fragment)?);
            self.generation = self.generation.wrapping_add(1);
            if self.generation == 0 {
                self.generation = 1;
            }
        } else if self.generation == 0 {
            self.generation = 1;
        }

        Ok(self.pipeline.as_ref().expect("pipeline initialized"))
    }
}

#[derive(Clone)]
pub(in crate::game::system::render) struct BackgroundTextureInstance {
    _texture: wgpu::Texture,
    pub(in crate::game::system::render) view: wgpu::TextureView,
}

pub(in crate::game::system::render) struct BackgroundTextureCache {
    sampler: Option<wgpu::Sampler>,
    fallback: Option<BackgroundTextureInstance>,
    textures: HashMap<usize, BackgroundTextureInstance>,
}

impl Default for BackgroundTextureCache {
    fn default() -> Self {
        Self {
            sampler: None,
            fallback: None,
            textures: HashMap::new(),
        }
    }
}

impl BackgroundTextureCache {
    fn ensure_sampler(&mut self, device: &wgpu::Device) -> &wgpu::Sampler {
        if self.sampler.is_none() {
            self.sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Background Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            }));
        }
        self.sampler.as_ref().expect("sampler initialized")
    }

    fn ensure_fallback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> &BackgroundTextureInstance {
        if self.fallback.is_none() {
            let (data, bytes_per_row) = util::pad_rgba_data(vec![255u8, 255u8, 255u8, 255u8], 1, 1)
                .expect("pad_rgba_data for 1x1 should not fail");
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Background Fallback Texture"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                texture.as_image_copy(),
                &data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.fallback = Some(BackgroundTextureInstance {
                _texture: texture,
                view,
            });
        }
        self.fallback.as_ref().expect("fallback initialized")
    }

    pub(in crate::game::system::render) fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        handle: Option<&ResourceHandle>,
    ) -> &BackgroundTextureInstance {
        if let Some(handle) = handle {
            let key = Arc::as_ptr(handle) as usize;
            if !self.textures.contains_key(&key) {
                match load_texture_from_resource(device, queue, handle) {
                    Ok(instance) => {
                        self.textures.insert(key, instance);
                    }
                    Err(err) => {
                        warn!(target: "jge-core", error = %err, "failed to load background texture, fallback to default");
                        return self.ensure_fallback(device, queue);
                    }
                }
            }
            return self.textures.get(&key).expect("inserted");
        }

        self.ensure_fallback(device, queue)
    }

    pub(in crate::game::system::render) fn sampler(
        &mut self,
        device: &wgpu::Device,
    ) -> &wgpu::Sampler {
        self.ensure_sampler(device)
    }
}

pub(in crate::game::system::render) fn render_background_if_present(
    layer_entity: Entity,
    context: &mut LayerRenderContext<'_, '_>,
) -> bool {
    let bg_entity = match find_first_background(layer_entity) {
        Some(entity) => entity,
        None => return false,
    };

    let bg_guard = match bg_entity.get_component::<Background>() {
        Some(bg) => bg,
        None => return false,
    };

    let color = bg_guard.color();
    let image = bg_guard.image();
    let fragment_override = bg_guard
        .fragment_shader()
        .map(|shader| shader.resource_handle());
    drop(bg_guard);

    let vertex_shader = Resource::from(ResourceHandlePath::BACKGROUND_VERTEX.into())
        .expect("built-in background vertex shader should exist");
    let fragment_shader = fragment_override.unwrap_or_else(|| {
        Resource::from(ResourceHandlePath::BACKGROUND_FRAGMENT.into())
            .expect("built-in background fragment shader should exist")
    });

    let pipeline = match context.caches.background.ensure(
        context.device,
        context.surface_format,
        vertex_shader,
        fragment_shader,
    ) {
        Ok(pipeline) => pipeline,
        Err(err) => {
            warn!(target: "jge-core", error = %err, "failed to prepare background pipeline");
            return false;
        }
    };

    let (texture_view, sampler) = {
        let cache = &mut context.caches.background_textures;
        let view = cache
            .get_or_create(context.device, context.queue, image.as_ref())
            .view
            .clone();
        let sampler = cache.sampler(context.device).clone();
        (view, sampler)
    };

    let use_texture = if image.is_some() { 1u32 } else { 0u32 };

    let (camera_pos, camera_forward) = resolve_layer_camera(layer_entity);
    let uniform_bytes = pack_background_uniform(color, use_texture, camera_pos, camera_forward);
    let uniform_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Background Uniform Buffer"),
            contents: &uniform_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Background Bind Group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

    let label = format!("Layer {} Background", layer_entity.id());
    let ops = wgpu::Operations {
        load: if context.viewport.is_some() {
            wgpu::LoadOp::Load
        } else {
            context.load_op
        },
        store: wgpu::StoreOp::Store,
    };

    let mut pass = context
        .encoder
        .begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: context.target_view,
                resolve_target: None,
                depth_slice: None,
                ops,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

    if let Some(viewport) = context.viewport {
        pass.set_viewport(
            viewport.x as f32,
            viewport.y as f32,
            viewport.width as f32,
            viewport.height as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(viewport.x, viewport.y, viewport.width, viewport.height);
    }

    pass.set_pipeline(&pipeline.pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.set_vertex_buffer(0, pipeline.vertex_buffer.slice(..));
    pass.draw(0..6, 0..1);

    trace!(
        target: "jge-core",
        layer_id = layer_entity.id(),
        background_id = bg_entity.id(),
        "rendered layer background"
    );

    true
}

fn resolve_layer_camera(layer_entity: Entity) -> ([f32; 3], [f32; 3]) {
    let scene_guard = match layer_entity.get_component::<Scene3D>() {
        Some(scene) => scene,
        None => return ([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
    };
    let preferred = scene_guard.attached_camera();
    drop(scene_guard);

    let camera_entity = match RenderSystem::select_scene3d_camera(layer_entity, preferred) {
        Ok(Some(camera)) => camera,
        Ok(None) => return ([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        Err(_) => return ([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
    };

    let transform_guard = match RenderSystem::try_get_transform(camera_entity) {
        Some(transform) => transform,
        None => return ([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
    };

    let position = transform_guard.position();
    let basis = Camera::orientation_basis(&transform_guard).normalize();
    drop(transform_guard);

    (
        [position.x, position.y, position.z],
        [basis.forward.x, basis.forward.y, basis.forward.z],
    )
}

fn find_first_background(root: Entity) -> Option<Entity> {
    let mut stack = Vec::new();
    stack.push(root);

    while let Some(entity) = stack.pop() {
        if entity.get_component::<Background>().is_some() {
            return Some(entity);
        }

        let node_guard = match entity.get_component::<Node>() {
            Some(node) => node,
            None => continue,
        };

        let children: Vec<Entity> = node_guard.children().iter().copied().collect();
        drop(node_guard);

        for child in children.into_iter().rev() {
            stack.push(child);
        }
    }

    None
}

fn load_shader_source(handle: &ResourceHandle) -> anyhow::Result<String> {
    let mut resource = handle.write();
    let bytes = if resource.data_loaded() {
        resource
            .try_get_data()
            .context("shader resource missing cached data")?
    } else {
        resource.get_data()
    };
    let source = std::str::from_utf8(bytes).context("shader source is not valid UTF-8")?;
    Ok(source.to_owned())
}

fn load_texture_from_resource(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    handle: &ResourceHandle,
) -> anyhow::Result<BackgroundTextureInstance> {
    let mut resource = handle.write();
    let bytes = if resource.data_loaded() {
        resource
            .try_get_data()
            .context("texture resource missing cached data")?
    } else {
        resource.get_data()
    };

    let img = image::load_from_memory(bytes).context("failed to decode image")?;
    let rgba = img.to_rgba8();
    let (width, height) = img.dimensions();

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Background Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let (data, bytes_per_row) = util::pad_rgba_data(rgba.to_vec(), width, height)
        .context("failed to pad RGBA texture data")?;

    queue.write_texture(
        texture.as_image_copy(),
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok(BackgroundTextureInstance {
        _texture: texture,
        view,
    })
}

fn pack_background_uniform(
    color: [f32; 4],
    use_texture: u32,
    camera_pos: [f32; 3],
    camera_forward: [f32; 3],
) -> [u8; 64] {
    let mut out = [0u8; 64];
    let mut offset = 0usize;

    for value in color {
        out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        offset += 4;
    }

    // params: vec4<u32>，其中 x = use_texture，其余为 0
    out[offset..offset + 4].copy_from_slice(&use_texture.to_le_bytes());
    // 其余 12 字节保持为 0
    offset = 32;

    for value in camera_pos {
        out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        offset += 4;
    }
    out[offset..offset + 4].copy_from_slice(&1.0f32.to_le_bytes());
    offset += 4;

    for value in camera_forward {
        out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        offset += 4;
    }
    out[offset..offset + 4].copy_from_slice(&0.0f32.to_le_bytes());

    out
}

fn cast_slice_f32(data: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    }
}

struct ResourceHandlePath;

impl ResourceHandlePath {
    const BACKGROUND_VERTEX: &'static str = "shaders/background.vs";
    const BACKGROUND_FRAGMENT: &'static str = "shaders/background.fs";
}
