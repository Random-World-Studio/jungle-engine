use std::{borrow::Cow, collections::HashMap, sync::Arc};

use anyhow::{Context, anyhow, ensure};
use image::GenericImageView;
use nalgebra::{Matrix4, Perspective3, Point3, Rotation3, Vector3};
use tracing::{debug, warn};
use wgpu::{self, util::DeviceExt};

use crate::game::{
    component::{
        ComponentRead,
        camera::Camera,
        layer::{Layer, LayerShader, LayerTraversalError, RenderPipelineStage, ShaderLanguage},
        light::{Light, ParallelLight, PointLight},
        scene3d::Scene3D,
        transform::Transform,
    },
    entity::Entity,
};
use crate::resource::ResourceHandle;

use super::{RenderSystem, cache::LayerRenderContext, util};

const MAX_SCENE3D_POINT_LIGHTS: usize = 8;
const MAX_SCENE3D_PARALLEL_LIGHTS: usize = 4;
const SCENE3D_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

impl RenderSystem {
    pub(in crate::game::system::render) fn try_get_transform(
        entity: Entity,
    ) -> Option<ComponentRead<Transform>> {
        for _ in 0..3 {
            if let Some(transform) = entity.get_component::<Transform>() {
                return Some(transform);
            }
            std::thread::yield_now();
        }
        None
    }

    pub(in crate::game::system::render) fn select_scene3d_camera(
        root: Entity,
        preferred: Option<Entity>,
    ) -> Result<Option<Entity>, LayerTraversalError> {
        if let Some(candidate) = preferred {
            if candidate.get_component::<Camera>().is_some() && Self::try_get_transform(candidate).is_some() {
                return Ok(Some(candidate));
            }
        }

        let ordered = Layer::renderable_entities(root)?;
        for entity in ordered {
            if entity.get_component::<Camera>().is_some() && Self::try_get_transform(entity).is_some() {
                return Ok(Some(entity));
            }
        }
        Ok(None)
    }

    pub(in crate::game::system::render) fn render_scene3d(
        entity: Entity,
        context: &mut LayerRenderContext<'_, '_>,
    ) {
        let scene_guard = match entity.get_component::<Scene3D>() {
            Some(scene) => scene,
            None => return,
        };
        let scene_near = scene_guard.near_plane();
        let scene_distance = scene_guard.view_distance();
        let scene_vertical = scene_guard.vertical_fov();
        let attached_camera = scene_guard.attached_camera();

        let (vertex_shader, fragment_shader) = {
            let layer_guard = match entity.get_component::<Layer>() {
                Some(layer) => layer,
                None => {
                    warn!(
                        target: "jge-core",
                        layer_id = entity.id(),
                        "Scene3D layer missing Layer component"
                    );
                    return;
                }
            };

            let vertex = match layer_guard.shader(RenderPipelineStage::Vertex) {
                Some(shader) => shader.clone(),
                None => {
                    warn!(
                        target: "jge-core",
                        layer_id = entity.id(),
                        "Scene3D layer has no vertex shader attached"
                    );
                    return;
                }
            };

            let fragment = match layer_guard.shader(RenderPipelineStage::Fragment) {
                Some(shader) => shader.clone(),
                None => {
                    warn!(
                        target: "jge-core",
                        layer_id = entity.id(),
                        "Scene3D layer has no fragment shader attached"
                    );
                    return;
                }
            };

            (vertex, fragment)
        };

        let (width, height) = context.framebuffer_size;
        if width == 0 || height == 0 {
            warn!(
                target: "jge-core",
                layer_id = entity.id(),
                framebuffer_width = width,
                framebuffer_height = height,
                "Scene3D framebuffer has invalid dimensions"
            );
            return;
        }

        let pipeline = match context.caches.scene3d.ensure(
            context.device,
            context.surface_format,
            &vertex_shader,
            &fragment_shader,
        ) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = entity.id(),
                    error = %error,
                    "failed to prepare Scene3D pipeline"
                );
                return;
            }
        };

        let camera_entity = match Self::select_scene3d_camera(entity, attached_camera) {
            Ok(camera) => match camera {
                Some(camera) => camera,
                None => {
                    warn!(
                        target: "jge-core",
                        layer_id = entity.id(),
                        "Scene3D layer has no camera available"
                    );
                    return;
                }
            },
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = entity.id(),
                    error = %error,
                    "Scene3D camera lookup failed"
                );
                return;
            }
        };

        let camera_guard = match camera_entity.get_component::<Camera>() {
            Some(camera) => camera,
            None => {
                warn!(
                    target: "jge-core",
                    layer_id = entity.id(),
                    camera_id = camera_entity.id(),
                    "Scene3D camera missing Camera component"
                );
                return;
            }
        };
        let camera_vertical = camera_guard.vertical_fov();
        let camera_near = camera_guard.near_plane();
        let camera_far = camera_guard.far_plane();
        drop(camera_guard);

        let near_plane = camera_near.max(scene_near);
        let far_plane = camera_far.min(scene_distance);
        if !(near_plane < far_plane) {
            warn!(
                target: "jge-core",
                layer_id = entity.id(),
                near_plane,
                far_plane,
                "Scene3D clip range invalid"
            );
            return;
        }

        let vertical_fov = scene_vertical.min(camera_vertical);
        let aspect_ratio = width as f32 / height as f32;

        let visible = match scene_guard.visible_renderables(camera_entity, context.framebuffer_size)
        {
            Ok(collection) => collection,
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = entity.id(),
                    camera_id = camera_entity.id(),
                    error = %error,
                    "Scene3D visibility query failed"
                );
                return;
            }
        };

        let transform_guard = match Self::try_get_transform(camera_entity) {
            Some(transform) => transform,
            None => {
                warn!(
                    target: "jge-core",
                    layer_id = entity.id(),
                    camera_id = camera_entity.id(),
                    "Scene3D camera missing Transform component"
                );
                return;
            }
        };
        let camera_position = transform_guard.position();
        let basis = Camera::orientation_basis(&transform_guard).normalize();
        drop(transform_guard);

        let point_light_entities = match Layer::point_light_entities(entity) {
            Ok(lights) => lights,
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = entity.id(),
                    error = %error,
                    "Scene3D lighting query failed"
                );
                Vec::new()
            }
        };

        let parallel_light_entities = match Layer::parallel_light_entities(entity) {
            Ok(lights) => lights,
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = entity.id(),
                    error = %error,
                    "Scene3D parallel lighting query failed"
                );
                Vec::new()
            }
        };

        struct ScenePointLight {
            position: Vector3<f32>,
            radius: f32,
            intensity: f32,
        }

        struct SceneParallelLight {
            direction: Vector3<f32>,
            intensity: f32,
        }

        let mut point_lights: Vec<ScenePointLight> = point_light_entities
            .into_iter()
            .filter_map(|light_entity| {
                let light = light_entity.get_component::<Light>()?;
                let point = light_entity.get_component::<PointLight>()?;
                let transform = light_entity.get_component::<Transform>()?;
                let radius = point.distance();
                let intensity = light.lightness();
                let position = transform.position();
                drop(transform);
                drop(point);
                drop(light);
                if radius <= f32::EPSILON || intensity <= 0.0 {
                    return None;
                }
                Some(ScenePointLight {
                    position,
                    radius,
                    intensity,
                })
            })
            .collect();

        let mut parallel_lights: Vec<SceneParallelLight> = parallel_light_entities
            .into_iter()
            .filter_map(|light_entity| {
                let light = light_entity.get_component::<Light>()?;
                let parallel = light_entity.get_component::<ParallelLight>()?;
                let transform = light_entity.get_component::<Transform>()?;
                let rotation = transform.rotation();
                let intensity = light.lightness();
                drop(transform);
                drop(parallel);
                drop(light);
                if intensity <= 0.0 {
                    return None;
                }
                let rotation_matrix =
                    Rotation3::from_euler_angles(rotation.x, rotation.y, rotation.z);
                let forward = rotation_matrix * Vector3::new(0.0, -1.0, 0.0);
                let incoming = -forward;
                if let Some(direction) = incoming.try_normalize(1.0e-6) {
                    Some(SceneParallelLight {
                        direction,
                        intensity,
                    })
                } else {
                    None
                }
            })
            .collect();

        if point_lights.len() > MAX_SCENE3D_POINT_LIGHTS {
            warn!(
                target: "jge-core",
                layer_id = entity.id(),
                light_count = point_lights.len(),
                max_supported = MAX_SCENE3D_POINT_LIGHTS,
                "Scene3D truncating point lights to fit uniform capacity"
            );
            point_lights.truncate(MAX_SCENE3D_POINT_LIGHTS);
        }

        if parallel_lights.len() > MAX_SCENE3D_PARALLEL_LIGHTS {
            warn!(
                target: "jge-core",
                layer_id = entity.id(),
                light_count = parallel_lights.len(),
                max_supported = MAX_SCENE3D_PARALLEL_LIGHTS,
                "Scene3D truncating parallel lights to fit uniform capacity"
            );
            parallel_lights.truncate(MAX_SCENE3D_PARALLEL_LIGHTS);
        }

        let material_layout = pipeline.material_layout();
        let mut draws = Vec::new();
        let mut total_vertices = 0u32;

        for bundle in visible.iter() {
            let triangles = bundle.triangles();
            if triangles.is_empty() {
                continue;
            }

            let material_descriptor = bundle.material();
            let regions = material_descriptor.map(|descriptor| descriptor.regions());
            let mut vertex_data = Vec::with_capacity(triangles.len() * 27);

            for (triangle_index, triangle) in triangles.iter().enumerate() {
                let ab = triangle[1] - triangle[0];
                let ac = triangle[2] - triangle[0];
                let cross = ab.cross(&ac);
                let normal = if let Some(normalized) = cross.try_normalize(1.0e-6) {
                    normalized
                } else {
                    Vector3::new(0.0, 1.0, 0.0)
                };

                let patch = regions.and_then(|collection| collection.get(triangle_index));

                for (vertex_index, vertex) in triangle.iter().enumerate() {
                    vertex_data.push(vertex.x);
                    vertex_data.push(vertex.y);
                    vertex_data.push(vertex.z);
                    vertex_data.push(normal.x);
                    vertex_data.push(normal.y);
                    vertex_data.push(normal.z);
                    if let Some(uv_patch) = patch {
                        let uv = uv_patch[vertex_index];
                        vertex_data.push(uv.x);
                        vertex_data.push(uv.y);
                        vertex_data.push(1.0);
                    } else {
                        vertex_data.push(0.0);
                        vertex_data.push(0.0);
                        vertex_data.push(0.0);
                    }
                }
            }

            if vertex_data.is_empty() {
                continue;
            }

            let vertex_buffer =
                context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Scene3D Vertex Buffer"),
                        contents: util::cast_slice_f32(&vertex_data),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
            let vertex_count = (vertex_data.len() / 9) as u32;
            total_vertices = total_vertices.saturating_add(vertex_count);

            let material_instance = if let Some(descriptor) = material_descriptor {
                match context.caches.scene3d_materials.ensure_material(
                    context.device,
                    context.queue,
                    material_layout,
                    descriptor.resource(),
                ) {
                    Ok(instance) => instance,
                    Err(error) => {
                        warn!(
                            target: "jge-core",
                            layer_id = entity.id(),
                            bundle_entity = bundle.entity().id(),
                            error = %error,
                            "failed to prepare 3D material texture, fallback to default"
                        );
                        context
                            .caches
                            .scene3d_materials
                            .ensure_default(context.device, context.queue, material_layout)
                            .expect("default material should be available")
                    }
                }
            } else {
                context
                    .caches
                    .scene3d_materials
                    .ensure_default(context.device, context.queue, material_layout)
                    .expect("default material should be available")
            };

            draws.push(Scene3DDraw {
                vertex_buffer,
                vertex_count,
                bind_group: material_instance.bind_group.clone(),
            });
        }

        if draws.is_empty() {
            debug!(
                target: "jge-core",
                layer_id = entity.id(),
                "Scene3D layer has no visible geometry"
            );
            return;
        }

        // 坐标系约定（世界/物理意义）：右手系，+X 向右，+Y 向上，-Z 为“前”。
        // nalgebra 的 look_at_rh + Perspective3 组合会生成 OpenGL 风格的裁剪空间：NDC Z 范围为 [-1, 1]。
        // wgpu/WebGPU 的裁剪空间 Z 范围为 [0, 1]，因此下面会额外乘一个转换矩阵做 Z 区间映射。
        let view = Matrix4::look_at_rh(
            &Point3::new(camera_position.x, camera_position.y, camera_position.z),
            &Point3::new(
                camera_position.x + basis.forward.x,
                camera_position.y + basis.forward.y,
                camera_position.z + basis.forward.z,
            ),
            &basis.up,
        );
        let projection = Perspective3::new(aspect_ratio, vertical_fov, near_plane, far_plane);
        let view_proj = util::opengl_to_wgpu_matrix() * projection.to_homogeneous() * view;

        let point_light_count = point_lights.len();
        let parallel_light_count = parallel_lights.len();
        let mut uniform_data = Vec::with_capacity(
            16 + 4 + MAX_SCENE3D_POINT_LIGHTS * 8 + MAX_SCENE3D_PARALLEL_LIGHTS * 8,
        );
        uniform_data.extend_from_slice(view_proj.as_slice());
        uniform_data.extend_from_slice(&[
            point_light_count as f32,
            parallel_light_count as f32,
            0.0,
            0.0,
        ]);

        for index in 0..MAX_SCENE3D_POINT_LIGHTS {
            if let Some(light) = point_lights.get(index) {
                uniform_data.extend_from_slice(&[
                    light.position.x,
                    light.position.y,
                    light.position.z,
                    light.radius,
                    light.intensity,
                    light.intensity,
                    light.intensity,
                    1.0,
                ]);
            } else {
                uniform_data.extend_from_slice(&[0.0; 8]);
            }
        }

        for index in 0..MAX_SCENE3D_PARALLEL_LIGHTS {
            if let Some(light) = parallel_lights.get(index) {
                uniform_data.extend_from_slice(&[
                    light.direction.x,
                    light.direction.y,
                    light.direction.z,
                    0.0,
                    light.intensity,
                    light.intensity,
                    light.intensity,
                    1.0,
                ]);
            } else {
                uniform_data.extend_from_slice(&[0.0; 8]);
            }
        }

        let uniform_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Scene3D Lighting Buffer"),
                contents: util::cast_slice_f32(&uniform_data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let scene_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Scene3D Uniform Bind Group"),
                layout: pipeline.uniform_layout(),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                }],
            });

        let label = format!("Layer {} Scene3D", entity.id());
        let ops = wgpu::Operations {
            load: context.load_op,
            store: wgpu::StoreOp::Store,
        };

        let depth_view =
            context
                .caches
                .scene3d_depth
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
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

        pass.set_pipeline(pipeline.pipeline());
        pass.set_bind_group(0, &scene_bind_group, &[]);
        for draw in &draws {
            pass.set_bind_group(1, &draw.bind_group, &[]);
            pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
            pass.draw(0..draw.vertex_count, 0..1);
        }

        debug!(
            target: "jge-core",
            layer_id = entity.id(),
            camera_id = camera_entity.id(),
            draw_calls = draws.len(),
            vertex_total = total_vertices,
            "Scene3D layer rendered"
        );
    }
}

pub(in crate::game::system::render) struct Scene3DDepthAttachment {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
}

#[derive(Default)]
pub(in crate::game::system::render) struct Scene3DDepthCache {
    attachment: Option<Scene3DDepthAttachment>,
}

impl Scene3DDepthCache {
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
                label: Some("Scene3D Depth Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: SCENE3D_DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.attachment = Some(Scene3DDepthAttachment {
                _texture: texture,
                view,
                size,
            });
        }

        &self
            .attachment
            .as_ref()
            .expect("Scene3D depth attachment initialized")
            .view
    }
}

pub(in crate::game::system::render) struct Scene3DPipelineCache {
    pipeline: Option<Scene3DPipeline>,
}

impl Default for Scene3DPipelineCache {
    fn default() -> Self {
        Self { pipeline: None }
    }
}

impl Scene3DPipelineCache {
    pub(in crate::game::system::render) fn ensure(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        vertex_shader: &LayerShader,
        fragment_shader: &LayerShader,
    ) -> anyhow::Result<&Scene3DPipeline> {
        let needs_rebuild = self
            .pipeline
            .as_ref()
            .map(|pipeline| !pipeline.matches(vertex_shader, fragment_shader))
            .unwrap_or(true);

        if needs_rebuild {
            let pipeline = Scene3DPipeline::new(device, format, vertex_shader, fragment_shader)?;
            self.pipeline = Some(pipeline);
        }

        Ok(self
            .pipeline
            .as_ref()
            .expect("Scene3D pipeline initialized"))
    }
}

pub(in crate::game::system::render) struct Scene3DPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_layout: wgpu::BindGroupLayout,
    material_layout: wgpu::BindGroupLayout,
    vertex_shader: Scene3DShaderKey,
    fragment_shader: Scene3DShaderKey,
}

impl Scene3DPipeline {
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        vertex_shader: &LayerShader,
        fragment_shader: &LayerShader,
    ) -> anyhow::Result<Self> {
        let vertex_key = Scene3DShaderKey::from_shader(vertex_shader);
        let fragment_key = Scene3DShaderKey::from_shader(fragment_shader);

        let (vertex_module, vertex_entry) = Self::compile_shader(
            device,
            vertex_shader,
            wgpu::ShaderStages::VERTEX,
            "Scene3D Vertex Shader",
        )?;
        let (fragment_module, fragment_entry) = Self::compile_shader(
            device,
            fragment_shader,
            wgpu::ShaderStages::FRAGMENT,
            "Scene3D Fragment Shader",
        )?;

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Scene3D Uniform Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Scene3D Material Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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
            label: Some("Scene3D Pipeline Layout"),
            bind_group_layouts: &[&uniform_layout, &material_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Scene3D Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_module,
                entry_point: Some(vertex_entry),
                buffers: &[SCENE3D_VERTEX_LAYOUT],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: SCENE3D_DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
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
                entry_point: Some(fragment_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        Ok(Self {
            pipeline,
            uniform_layout,
            material_layout,
            vertex_shader: vertex_key,
            fragment_shader: fragment_key,
        })
    }

    fn matches(&self, vertex_shader: &LayerShader, fragment_shader: &LayerShader) -> bool {
        self.vertex_shader.matches(vertex_shader) && self.fragment_shader.matches(fragment_shader)
    }

    pub(in crate::game::system::render) fn uniform_layout(&self) -> &wgpu::BindGroupLayout {
        &self.uniform_layout
    }

    pub(in crate::game::system::render) fn material_layout(&self) -> &wgpu::BindGroupLayout {
        &self.material_layout
    }

    pub(in crate::game::system::render) fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }

    fn compile_shader(
        device: &wgpu::Device,
        shader: &LayerShader,
        stage: wgpu::ShaderStages,
        label: &str,
    ) -> anyhow::Result<(wgpu::ShaderModule, &'static str)> {
        let source = Self::load_shader_source(&shader.resource_handle())?;
        let entry_point = match shader.language() {
            ShaderLanguage::Glsl => {
                let naga_stage = match stage {
                    wgpu::ShaderStages::VERTEX => wgpu::naga::ShaderStage::Vertex,
                    wgpu::ShaderStages::FRAGMENT => wgpu::naga::ShaderStage::Fragment,
                    wgpu::ShaderStages::COMPUTE => wgpu::naga::ShaderStage::Compute,
                    _ => {
                        return Err(anyhow!("unsupported GLSL shader stage: {:?}", stage));
                    }
                };
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Glsl {
                        shader: Cow::Owned(source),
                        stage: naga_stage,
                        defines: Default::default(),
                    },
                });
                return Ok((module, "main"));
            }
            ShaderLanguage::Wgsl => match stage {
                wgpu::ShaderStages::VERTEX => "vs_main",
                wgpu::ShaderStages::FRAGMENT => "fs_main",
                _ => "main",
            },
        };

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
        });
        Ok((module, entry_point))
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
}

#[derive(Clone)]
pub(in crate::game::system::render) struct Scene3DShaderKey {
    language: ShaderLanguage,
    resource: ResourceHandle,
}

impl Scene3DShaderKey {
    fn from_shader(shader: &LayerShader) -> Self {
        Self {
            language: shader.language(),
            resource: shader.resource_handle(),
        }
    }

    fn matches(&self, shader: &LayerShader) -> bool {
        if self.language != shader.language() {
            return false;
        }
        let handle = shader.resource_handle();
        Arc::ptr_eq(&self.resource, &handle)
    }
}

const SCENE3D_VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
    wgpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: wgpu::VertexFormat::Float32x3,
    },
    wgpu::VertexAttribute {
        offset: (std::mem::size_of::<f32>() * 3) as u64,
        shader_location: 1,
        format: wgpu::VertexFormat::Float32x3,
    },
    wgpu::VertexAttribute {
        offset: (std::mem::size_of::<f32>() * 6) as u64,
        shader_location: 2,
        format: wgpu::VertexFormat::Float32x2,
    },
    wgpu::VertexAttribute {
        offset: (std::mem::size_of::<f32>() * 8) as u64,
        shader_location: 3,
        format: wgpu::VertexFormat::Float32,
    },
];

const SCENE3D_VERTEX_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: (std::mem::size_of::<f32>() * 9) as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &SCENE3D_VERTEX_ATTRIBUTES,
};

pub(in crate::game::system::render) struct Scene3DDraw {
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    bind_group: wgpu::BindGroup,
}

#[derive(Clone)]
pub(in crate::game::system::render) struct Scene3DMaterialInstance {
    _texture: wgpu::Texture,
    _view: wgpu::TextureView,
    pub(in crate::game::system::render) bind_group: wgpu::BindGroup,
}

pub(in crate::game::system::render) struct Scene3DMaterialCache {
    sampler: Option<wgpu::Sampler>,
    default: Option<Scene3DMaterialInstance>,
    materials: HashMap<usize, Scene3DMaterialInstance>,
}

impl Default for Scene3DMaterialCache {
    fn default() -> Self {
        Self {
            sampler: None,
            default: None,
            materials: HashMap::new(),
        }
    }
}

impl Scene3DMaterialCache {
    fn ensure_sampler(&mut self, device: &wgpu::Device) -> &wgpu::Sampler {
        if self.sampler.is_none() {
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Scene3D Material Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            self.sampler = Some(sampler);
        }
        self.sampler.as_ref().expect("sampler initialized")
    }

    pub(in crate::game::system::render) fn ensure_default(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> anyhow::Result<&Scene3DMaterialInstance> {
        if self.default.is_none() {
            let sampler = self.ensure_sampler(device).clone();
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Scene3D Default Material"),
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
                label: Some("Scene3D Default Material Bind Group"),
                layout,
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

            self.default = Some(Scene3DMaterialInstance {
                _texture: texture,
                _view: view,
                bind_group,
            });
        }

        self.default
            .as_ref()
            .ok_or_else(|| anyhow!("failed to prepare default 3D material instance"))
    }

    pub(in crate::game::system::render) fn ensure_material(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        handle: &ResourceHandle,
    ) -> anyhow::Result<&Scene3DMaterialInstance> {
        let key = Arc::as_ptr(handle) as usize;
        if !self.materials.contains_key(&key) {
            let instance = self.load_material_instance(device, queue, layout, handle)?;
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
        layout: &wgpu::BindGroupLayout,
        handle: &ResourceHandle,
    ) -> anyhow::Result<Scene3DMaterialInstance> {
        let data_slice: &'static [u8] = {
            let mut resource = handle.write();
            let bytes = if resource.data_loaded() {
                resource
                    .try_get_data()
                    .context("material resource missing cached data")?
            } else {
                resource.get_data()
            };
            bytes.as_slice()
        };

        let image = image::load_from_memory(data_slice)
            .context("failed to decode 3D material texture resource")?;
        let (width, height) = image.dimensions();
        ensure!(
            width > 0 && height > 0,
            "3D material texture has invalid dimensions"
        );

        let rgba = image.to_rgba8().into_raw();
        let (data, bytes_per_row) = util::pad_rgba_data(rgba, width, height)?;

        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene3D Material Texture"),
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
            label: Some("Scene3D Material Bind Group"),
            layout,
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

        Ok(Scene3DMaterialInstance {
            _texture: texture,
            _view: view,
            bind_group,
        })
    }
}
