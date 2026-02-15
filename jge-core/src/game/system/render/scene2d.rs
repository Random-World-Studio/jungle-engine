use std::{borrow::Cow, collections::HashMap, sync::Arc};

use anyhow::{Context, anyhow, ensure};
use image::GenericImageView;
use nalgebra::{Vector2, Vector3};
use tracing::{debug, warn};
use wgpu::{self, util::DeviceExt};

use crate::game::{
    component::{
        layer::{Layer, RenderPipelineStage},
        light::{Light, PointLight},
        scene2d::Scene2D,
        transform::Transform,
    },
    entity::Entity,
};
use crate::resource::ResourceHandle;

use super::{RenderSystem, cache::LayerRenderContext, resource_io, util};

use super::snapshot::{Scene2DPointLightSnapshot, Scene2DSnapshot};

const SCENE2D_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub(in crate::game::system::render) struct Scene2DDepthAttachment {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
}

#[derive(Default)]
pub(in crate::game::system::render) struct Scene2DDepthCache {
    attachment: Option<Scene2DDepthAttachment>,
}

impl Scene2DDepthCache {
    pub(in crate::game::system::render) fn ensure(
        &mut self,
        device: &wgpu::Device,
        size: (u32, u32),
    ) -> &wgpu::TextureView {
        let (width, height) = size;
        let needs_rebuild = self
            .attachment
            .as_ref()
            .map(|attachment| attachment.size != size)
            .unwrap_or(true);

        if needs_rebuild {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Scene2D Depth Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: SCENE2D_DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.attachment = Some(Scene2DDepthAttachment {
                _texture: texture,
                view,
                size,
            });
        }

        &self
            .attachment
            .as_ref()
            .expect("Scene2D depth attachment initialized")
            .view
    }
}

impl RenderSystem {
    pub(in crate::game::system::render) fn render_scene2d_from_snapshot(
        entity: Entity,
        snapshot: &Scene2DSnapshot,
        context: &mut LayerRenderContext<'_, '_>,
    ) {
        let viewport_framebuffer_size = context
            .viewport
            .map(|viewport| (viewport.width, viewport.height))
            .unwrap_or(context.framebuffer_size);

        let (render_pipeline, bind_group_layout, pipeline_generation) =
            match context.caches.scene2d.ensure(
                context.device,
                context.surface_format,
                snapshot.vertex_shader.clone(),
                snapshot.fragment_shader.clone(),
            ) {
                Ok((pipeline, generation)) => (
                    pipeline.pipeline.clone(),
                    pipeline.bind_group_layout().clone(),
                    generation,
                ),
                Err(error) => {
                    warn!(
                        target: "jge-core",
                        layer_id = %entity.id(),
                        error = %error,
                        "failed to prepare Scene2D pipeline"
                    );
                    return;
                }
            };

        let scene_offset = Vector2::new(snapshot.offset[0], snapshot.offset[1]);
        let pixels_per_unit = snapshot.pixels_per_unit;

        const MAX_POINT_LIGHT_BRIGHTNESS: f32 = 1.0;
        const MAX_DIRECTIONAL_BRIGHTNESS: f32 = 4.0;
        const MAX_TOTAL_BRIGHTNESS: f32 = 6.0;

        let directional_brightness = snapshot
            .parallel_light_brightness
            .clamp(0.0, MAX_DIRECTIONAL_BRIGHTNESS);

        fn evaluate_point_lights(
            vertex: &Vector3<f32>,
            lights: &[Scene2DPointLightSnapshot],
        ) -> f32 {
            if lights.is_empty() {
                return 0.0;
            }

            let mut total = 0.0;
            for light in lights {
                let dx = vertex.x - light.center[0];
                let dy = vertex.y - light.center[1];
                let distance = (dx * dx + dy * dy).sqrt();
                if distance >= light.radius {
                    continue;
                }
                let normalized = 1.0 - (distance / light.radius);
                let smooth = normalized * normalized * (3.0 - 2.0 * normalized);
                total += light.lightness * smooth;
            }
            total
        }

        let renderables = &snapshot.renderables;
        let face_groups = &snapshot.face_groups;

        let mut draws = Vec::new();
        let mut total_vertices = 0usize;

        for group in face_groups {
            let faces = group.faces();
            if faces.is_empty() {
                continue;
            }

            let _profile_scope = context
                .caches
                .profiler
                .entity_scope(context.runtime, group.entity());

            let (material_regions, bind_group) = match renderables.material(group.entity()) {
                Some(descriptor) => {
                    let handle = descriptor.resource().clone();
                    match context.caches.scene2d_materials.ensure_material(
                        context.device,
                        context.queue,
                        &bind_group_layout,
                        pipeline_generation,
                        &handle,
                    ) {
                        Ok(instance) => (Some(descriptor.regions()), instance.bind_group.clone()),
                        Err(error) => {
                            warn!(
                                target: "jge-core",
                                layer_id = %entity.id(),
                                error = %error,
                                "failed to prepare material texture, fallback to default"
                            );
                            let fallback = context
                                .caches
                                .scene2d_materials
                                .ensure_default(
                                    context.device,
                                    context.queue,
                                    &bind_group_layout,
                                    pipeline_generation,
                                )
                                .expect("default material should be available")
                                .bind_group
                                .clone();
                            (None, fallback)
                        }
                    }
                }
                None => {
                    let instance = context
                        .caches
                        .scene2d_materials
                        .ensure_default(
                            context.device,
                            context.queue,
                            &bind_group_layout,
                            pipeline_generation,
                        )
                        .expect("default material should be available");
                    (None, instance.bind_group.clone())
                }
            };

            let mut vertex_data = Vec::with_capacity(faces.len() * 18);

            for (triangle_index, triangle) in faces.iter().enumerate() {
                if triangle
                    .iter()
                    .any(|vertex| !vertex.z.is_finite() || vertex.z < 0.0 || vertex.z > 1.0)
                {
                    continue;
                }

                for (vertex_index, vertex) in triangle.iter().enumerate() {
                    let point_brightness = evaluate_point_lights(vertex, &snapshot.point_lights)
                        .clamp(0.0, MAX_POINT_LIGHT_BRIGHTNESS);
                    let brightness = (directional_brightness + point_brightness)
                        .clamp(0.0, MAX_TOTAL_BRIGHTNESS);
                    let ndc = util::scene2d_vertex_to_ndc(
                        viewport_framebuffer_size,
                        vertex,
                        &scene_offset,
                        pixels_per_unit,
                    );
                    let uv = material_regions
                        .and_then(|regions| regions.get(triangle_index))
                        .map(|patch| patch[vertex_index])
                        .unwrap_or_else(|| Vector2::new(0.0, 0.0));
                    vertex_data.push(ndc.x);
                    vertex_data.push(ndc.y);
                    vertex_data.push(ndc.z);
                    vertex_data.push(uv.x);
                    vertex_data.push(uv.y);
                    vertex_data.push(brightness);
                }
            }

            if vertex_data.is_empty() {
                continue;
            }

            let vertex_buffer =
                context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Scene2D Vertex Buffer"),
                        contents: util::cast_slice_f32(&vertex_data),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
            let vertex_count = (vertex_data.len() / 6) as u32;
            total_vertices += vertex_count as usize;
            draws.push(Scene2DDraw {
                vertex_buffer,
                vertex_count,
                bind_group,
            });
        }

        if draws.is_empty() {
            debug!(
                target: "jge-core",
                layer_id = %entity.id(),
                "Scene2D layer has no visible draws"
            );
            return;
        }

        let label = format!("Layer {}", entity.id());
        let ops = wgpu::Operations {
            load: if context.viewport.is_some() {
                wgpu::LoadOp::Load
            } else {
                context.load_op
            },
            store: wgpu::StoreOp::Store,
        };

        let depth_view = context
            .caches
            .scene2d_depth
            .ensure(context.device, context.framebuffer_size);

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
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
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

        pass.set_pipeline(&render_pipeline);
        for draw in &draws {
            pass.set_bind_group(0, &draw.bind_group, &[]);
            pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
            pass.draw(0..draw.vertex_count, 0..1);
        }

        debug!(
            target: "jge-core",
            layer_id = %entity.id(),
            draw_calls = draws.len(),
            vertex_total = total_vertices,
            "Scene2D layer rendered"
        );
    }

    pub(in crate::game::system::render) fn render_scene2d(
        entity: Entity,
        context: &mut LayerRenderContext<'_, '_>,
    ) {
        let mut scene_guard = match context
            .runtime
            .block_on(entity.get_component_mut::<Scene2D>())
        {
            Some(scene) => scene,
            None => return,
        };
        scene_guard.set_framebuffer_size(context.framebuffer_size);
        let scene_offset = scene_guard.offset();
        let pixels_per_unit = scene_guard.pixels_per_unit();
        drop(scene_guard);

        let viewport_framebuffer_size = context
            .viewport
            .map(|viewport| (viewport.width, viewport.height))
            .unwrap_or(context.framebuffer_size);

        let layer_guard = match context.runtime.block_on(entity.get_component::<Layer>()) {
            Some(layer) => layer,
            None => {
                warn!(
                    target: "jge-core",
                    layer_id = %entity.id(),
                    "missing Layer component, skip rendering"
                );
                return;
            }
        };

        let vertex_shader = match layer_guard
            .shader(RenderPipelineStage::Vertex)
            .map(|shader| shader.resource_handle())
        {
            Some(handle) => handle,
            None => {
                warn!(
                    target: "jge-core",
                    layer_id = %entity.id(),
                    "layer has no vertex shader attached"
                );
                return;
            }
        };

        let fragment_shader = match layer_guard
            .shader(RenderPipelineStage::Fragment)
            .map(|shader| shader.resource_handle())
        {
            Some(handle) => handle,
            None => {
                warn!(
                    target: "jge-core",
                    layer_id = %entity.id(),
                    "layer has no fragment shader attached"
                );
                return;
            }
        };
        drop(layer_guard);

        let (render_pipeline, bind_group_layout, pipeline_generation) =
            match context.caches.scene2d.ensure(
                context.device,
                context.surface_format,
                vertex_shader,
                fragment_shader,
            ) {
                Ok((pipeline, generation)) => (
                    pipeline.pipeline.clone(),
                    pipeline.bind_group_layout().clone(),
                    generation,
                ),
                Err(error) => {
                    warn!(
                        target: "jge-core",
                        layer_id = %entity.id(),
                        error = %error,
                        "failed to prepare Scene2D pipeline"
                    );
                    return;
                }
            };

        let runtime = context.runtime;

        let point_light_entities = match runtime.block_on(Layer::point_light_entities(entity)) {
            Ok(lights) => lights,
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = %entity.id(),
                    error = %error,
                    "Scene2D lighting query failed"
                );
                Vec::new()
            }
        };

        let parallel_light_brightness =
            match runtime.block_on(Layer::parallel_light_entities(entity)) {
                Ok(lights) => lights
                    .into_iter()
                    .filter_map(|light_entity| {
                        let light = runtime.block_on(light_entity.get_component::<Light>())?;
                        let value = light.lightness();
                        drop(light);
                        if value <= 0.0 { None } else { Some(value) }
                    })
                    .sum::<f32>(),
                Err(error) => {
                    warn!(
                        target: "jge-core",
                        layer_id = %entity.id(),
                        error = %error,
                        "Scene2D directional lighting query failed"
                    );
                    0.0
                }
            };

        struct ScenePointLight {
            center: Vector2<f32>,
            radius: f32,
            lightness: f32,
        }

        let point_lights: Vec<ScenePointLight> = point_light_entities
            .into_iter()
            .filter_map(|light_entity| {
                let light = runtime.block_on(light_entity.get_component::<Light>())?;
                let point = runtime.block_on(light_entity.get_component::<PointLight>())?;
                let world = runtime.block_on(Transform::world_matrix(light_entity))?;
                let radius = point.distance();
                let lightness = light.lightness();
                let position = Transform::translation_from_matrix(&world);
                drop(point);
                drop(light);
                if radius <= f32::EPSILON || lightness <= 0.0 {
                    return None;
                }
                Some(ScenePointLight {
                    center: Vector2::new(position.x, position.y),
                    radius,
                    lightness,
                })
            })
            .collect();

        fn evaluate_point_lights(vertex: &Vector3<f32>, lights: &[ScenePointLight]) -> f32 {
            if lights.is_empty() {
                return 0.0;
            }

            let mut total = 0.0;
            for light in lights {
                let dx = vertex.x - light.center.x;
                let dy = vertex.y - light.center.y;
                let distance = (dx * dx + dy * dy).sqrt();
                if distance >= light.radius {
                    continue;
                }
                let normalized = 1.0 - (distance / light.radius);
                // 使用平滑的三次插值，实现自然的递减曲线。
                let smooth = normalized * normalized * (3.0 - 2.0 * normalized);
                total += light.lightness * smooth;
            }
            total
        }

        let renderables = match runtime.block_on(Layer::collect_renderables(entity)) {
            Ok(collection) => collection,
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = %entity.id(),
                    error = %error,
                    "Scene2D renderable collection failed"
                );
                return;
            }
        };

        let scene_guard = match context.runtime.block_on(entity.get_component::<Scene2D>()) {
            Some(scene) => scene,
            None => {
                warn!(
                    target: "jge-core",
                    layer_id = %entity.id(),
                    "Scene2D component disappeared before visibility query"
                );
                return;
            }
        };
        let layer_guard = match context.runtime.block_on(entity.get_component::<Layer>()) {
            Some(layer) => layer,
            None => {
                warn!(
                    target: "jge-core",
                    layer_id = %entity.id(),
                    "Layer component missing during visibility query"
                );
                return;
            }
        };

        let face_groups =
            match scene_guard.visible_faces_with_renderables(&layer_guard, renderables.bundles()) {
                Ok(faces) => faces,
                Err(error) => {
                    warn!(
                        target: "jge-core",
                        layer_id = %entity.id(),
                        error = %error,
                        "Scene2D visibility query failed"
                    );
                    Vec::new()
                }
            };
        drop(layer_guard);
        drop(scene_guard);

        let mut draws = Vec::new();
        let mut total_vertices = 0usize;

        const MAX_POINT_LIGHT_BRIGHTNESS: f32 = 1.0;
        const MAX_DIRECTIONAL_BRIGHTNESS: f32 = 4.0;
        const MAX_TOTAL_BRIGHTNESS: f32 = 6.0;

        let directional_brightness =
            parallel_light_brightness.clamp(0.0, MAX_DIRECTIONAL_BRIGHTNESS);

        for group in face_groups {
            let faces = group.faces();
            if faces.is_empty() {
                continue;
            }

            let _profile_scope = context
                .caches
                .profiler
                .entity_scope(context.runtime, group.entity());

            let (material_regions, bind_group) = match renderables.material(group.entity()) {
                Some(descriptor) => {
                    let handle = descriptor.resource().clone();
                    match context.caches.scene2d_materials.ensure_material(
                        context.device,
                        context.queue,
                        &bind_group_layout,
                        pipeline_generation,
                        &handle,
                    ) {
                        Ok(instance) => (Some(descriptor.regions()), instance.bind_group.clone()),
                        Err(error) => {
                            warn!(
                                target: "jge-core",
                                layer_id = %entity.id(),
                                error = %error,
                                "failed to prepare material texture, fallback to default"
                            );
                            let fallback = context
                                .caches
                                .scene2d_materials
                                .ensure_default(
                                    context.device,
                                    context.queue,
                                    &bind_group_layout,
                                    pipeline_generation,
                                )
                                .expect("default material should be available")
                                .bind_group
                                .clone();
                            (None, fallback)
                        }
                    }
                }
                None => {
                    let instance = context
                        .caches
                        .scene2d_materials
                        .ensure_default(
                            context.device,
                            context.queue,
                            &bind_group_layout,
                            pipeline_generation,
                        )
                        .expect("default material should be available");
                    (None, instance.bind_group.clone())
                }
            };

            let mut vertex_data = Vec::with_capacity(faces.len() * 18);

            for (triangle_index, triangle) in faces.iter().enumerate() {
                // Scene2D 约定：z 必须在 [0,1]，否则该三角形直接丢弃。
                if triangle
                    .iter()
                    .any(|vertex| !vertex.z.is_finite() || vertex.z < 0.0 || vertex.z > 1.0)
                {
                    continue;
                }
                for (vertex_index, vertex) in triangle.iter().enumerate() {
                    let point_brightness = evaluate_point_lights(vertex, &point_lights)
                        .clamp(0.0, MAX_POINT_LIGHT_BRIGHTNESS);
                    let brightness = (directional_brightness + point_brightness)
                        .clamp(0.0, MAX_TOTAL_BRIGHTNESS);
                    let ndc = util::scene2d_vertex_to_ndc(
                        viewport_framebuffer_size,
                        vertex,
                        &scene_offset,
                        pixels_per_unit,
                    );
                    let uv = material_regions
                        .and_then(|regions| regions.get(triangle_index))
                        .map(|patch| patch[vertex_index])
                        .unwrap_or_else(|| Vector2::new(0.0, 0.0));
                    vertex_data.push(ndc.x);
                    vertex_data.push(ndc.y);
                    vertex_data.push(ndc.z);
                    vertex_data.push(uv.x);
                    vertex_data.push(uv.y);
                    vertex_data.push(brightness);
                }
            }

            if vertex_data.is_empty() {
                continue;
            }

            let vertex_buffer =
                context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Scene2D Vertex Buffer"),
                        contents: util::cast_slice_f32(&vertex_data),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
            let vertex_count = (vertex_data.len() / 6) as u32;
            total_vertices += vertex_count as usize;
            draws.push(Scene2DDraw {
                vertex_buffer,
                vertex_count,
                bind_group,
            });
        }

        if draws.is_empty() {
            debug!(
                target: "jge-core",
                layer_id = %entity.id(),
                "Scene2D layer has no visible draws"
            );
            return;
        }

        let label = format!("Layer {}", entity.id());
        let ops = wgpu::Operations {
            load: if context.viewport.is_some() {
                wgpu::LoadOp::Load
            } else {
                context.load_op
            },
            store: wgpu::StoreOp::Store,
        };

        let depth_view = context
            .caches
            .scene2d_depth
            .ensure(context.device, context.framebuffer_size);

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
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        // 约定：z 越大越靠前，因此清 0.0 作为“最远”。
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
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

        pass.set_pipeline(&render_pipeline);
        for draw in &draws {
            pass.set_bind_group(0, &draw.bind_group, &[]);
            pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
            pass.draw(0..draw.vertex_count, 0..1);
        }

        debug!(
            target: "jge-core",
            layer_id = %entity.id(),
            draw_calls = draws.len(),
            vertex_total = total_vertices,
            "Scene2D layer rendered"
        );
    }
}

pub(in crate::game::system::render) struct Scene2DDraw {
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    bind_group: wgpu::BindGroup,
}

pub(in crate::game::system::render) struct Scene2DPipeline {
    pub(in crate::game::system::render) pipeline: wgpu::RenderPipeline,
    vertex_shader: ResourceHandle,
    fragment_shader: ResourceHandle,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl Scene2DPipeline {
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        vertex_shader: ResourceHandle,
        fragment_shader: ResourceHandle,
    ) -> anyhow::Result<Self> {
        let vertex_source = Self::load_shader_source(&vertex_shader)?;
        let fragment_source = Self::load_shader_source(&fragment_shader)?;

        let vertex_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Scene2D Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(vertex_source)),
        });

        let fragment_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Scene2D Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(fragment_source)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Scene2D Material Bind Group Layout"),
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Scene2D Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Scene2D Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_module,
                entry_point: Some("vs_main"),
                buffers: &[SCENE2D_VERTEX_LAYOUT],
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: SCENE2D_DEPTH_FORMAT,
                depth_write_enabled: true,
                // 约定：z 越大越靠前 => 更大的深度值通过测试。
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
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

        Ok(Self {
            pipeline,
            vertex_shader,
            fragment_shader,
            bind_group_layout,
        })
    }

    fn matches(&self, vertex: &ResourceHandle, fragment: &ResourceHandle) -> bool {
        Arc::ptr_eq(&self.vertex_shader, vertex) && Arc::ptr_eq(&self.fragment_shader, fragment)
    }

    pub(in crate::game::system::render) fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    fn load_shader_source(handle: &ResourceHandle) -> anyhow::Result<String> {
        resource_io::load_utf8_string(handle, "scene2d shader")
    }
}

#[derive(Default)]
pub(in crate::game::system::render) struct Scene2DPipelineCache {
    pipeline: Option<Scene2DPipeline>,
    generation: u64,
}

impl Scene2DPipelineCache {
    pub(in crate::game::system::render) fn ensure(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        vertex: ResourceHandle,
        fragment: ResourceHandle,
    ) -> anyhow::Result<(&Scene2DPipeline, u64)> {
        let needs_rebuild = self
            .pipeline
            .as_ref()
            .map(|pipeline| !pipeline.matches(&vertex, &fragment))
            .unwrap_or(true);

        if needs_rebuild {
            let pipeline = Scene2DPipeline::new(device, format, vertex, fragment)?;
            self.pipeline = Some(pipeline);
            self.generation = self.generation.wrapping_add(1);
            if self.generation == 0 {
                self.generation = 1;
            }
        } else if self.generation == 0 {
            // Ensure generation is never zero once a pipeline exists.
            self.generation = 1;
        }

        Ok((
            self.pipeline.as_ref().expect("pipeline just initialized"),
            self.generation,
        ))
    }
}

#[derive(Clone)]
pub(in crate::game::system::render) struct Scene2DMaterialInstance {
    _texture: wgpu::Texture,
    _view: wgpu::TextureView,
    pub(in crate::game::system::render) bind_group: wgpu::BindGroup,
}

#[derive(Default)]
pub(in crate::game::system::render) struct Scene2DMaterialCache {
    sampler: Option<wgpu::Sampler>,
    default: Option<Scene2DMaterialInstance>,
    materials: HashMap<usize, Scene2DMaterialInstance>,
    pipeline_generation: u64,
}

impl Scene2DMaterialCache {
    fn ensure_sampler(&mut self, device: &wgpu::Device) -> &wgpu::Sampler {
        if self.sampler.is_none() {
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Scene2D Material Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            self.sampler = Some(sampler);
        }
        self.sampler.as_ref().expect("sampler initialized")
    }

    fn sync_for_pipeline(&mut self, generation: u64) {
        if self.pipeline_generation != generation {
            self.pipeline_generation = generation;
            self.default = None;
            self.materials.clear();
        }
    }

    pub(in crate::game::system::render) fn ensure_default(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
        pipeline_generation: u64,
    ) -> anyhow::Result<&Scene2DMaterialInstance> {
        ensure!(
            pipeline_generation != 0,
            "Scene2D pipeline must be initialized before requesting default material",
        );
        self.sync_for_pipeline(pipeline_generation);

        if self.default.is_none() {
            let sampler = self.ensure_sampler(device).clone();
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Scene2D Default Material"),
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

            let (data, bytes_per_row) = util::pad_rgba_data(vec![255, 255, 255, 255], 1, 1)?;
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
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Scene2D Default Material Bind Group"),
                layout: bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

            self.default = Some(Scene2DMaterialInstance {
                _texture: texture,
                _view: view,
                bind_group,
            });
        }

        self.default
            .as_ref()
            .ok_or_else(|| anyhow!("failed to prepare default material instance"))
    }

    pub(in crate::game::system::render) fn ensure_material(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
        pipeline_generation: u64,
        handle: &ResourceHandle,
    ) -> anyhow::Result<&Scene2DMaterialInstance> {
        ensure!(
            pipeline_generation != 0,
            "Scene2D pipeline must be initialized before requesting materials",
        );
        self.sync_for_pipeline(pipeline_generation);

        let key = Arc::as_ptr(handle) as usize;
        if !self.materials.contains_key(&key) {
            let instance = self.load_material_instance(device, queue, bind_group_layout, handle)?;
            self.materials.insert(key, instance);
        }

        self.materials
            .get(&key)
            .ok_or_else(|| anyhow!("material instance missing after insertion"))
    }

    fn load_material_instance(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
        handle: &ResourceHandle,
    ) -> anyhow::Result<Scene2DMaterialInstance> {
        let bytes = resource_io::load_bytes_arc(handle, "scene2d material texture")?;
        let image = image::load_from_memory(bytes.as_ref())
            .context("failed to decode material texture resource")?;
        let (width, height) = image.dimensions();
        ensure!(
            width > 0 && height > 0,
            "material texture has invalid dimensions"
        );

        let rgba = image.to_rgba8().into_raw();
        let (data, bytes_per_row) = util::pad_rgba_data(rgba, width, height)?;

        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene2D Material Texture"),
            size: extent,
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
                rows_per_image: Some(height),
            },
            extent,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = self.ensure_sampler(device).clone();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scene2D Material Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Ok(Scene2DMaterialInstance {
            _texture: texture,
            _view: view,
            bind_group,
        })
    }
}

const SCENE2D_VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] = [
    wgpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: wgpu::VertexFormat::Float32x3,
    },
    wgpu::VertexAttribute {
        offset: (std::mem::size_of::<f32>() * 3) as u64,
        shader_location: 1,
        format: wgpu::VertexFormat::Float32x2,
    },
    wgpu::VertexAttribute {
        offset: (std::mem::size_of::<f32>() * 5) as u64,
        shader_location: 2,
        format: wgpu::VertexFormat::Float32,
    },
];

const SCENE2D_VERTEX_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: (std::mem::size_of::<f32>() * 6) as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &SCENE2D_VERTEX_ATTRIBUTES,
};
