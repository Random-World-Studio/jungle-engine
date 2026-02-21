mod camera;
mod depth;
mod material;
mod pipeline;

use nalgebra::{Matrix4, Perspective3, Point3, Vector3};
use tokio::runtime::Runtime;
use tracing::{trace, warn};
use wgpu::{self, util::DeviceExt};

use crate::game::{
    component::{
        camera::Camera,
        layer::{Layer, LayerTraversalError, RenderPipelineStage},
        light::{Light, ParallelLight, PointLight},
        scene3d::Scene3D,
        transform::Transform,
    },
    entity::Entity,
};

use super::{RenderSystem, cache::LayerRenderContext, util};

use super::snapshot::Scene3DSnapshot;

pub(in crate::game::system::render) use depth::Scene3DDepthCache;
pub(in crate::game::system::render) use material::Scene3DMaterialCache;
pub(in crate::game::system::render) use pipeline::Scene3DPipelineCache;

const MAX_SCENE3D_POINT_LIGHTS: usize = 8;
const MAX_SCENE3D_PARALLEL_LIGHTS: usize = 4;

pub(super) const SCENE3D_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub(in crate::game::system::render) struct Scene3DDraw {
    pub(in crate::game::system::render) vertex_buffer: wgpu::Buffer,
    pub(in crate::game::system::render) vertex_count: u32,
    pub(in crate::game::system::render) bind_group: wgpu::BindGroup,
    pub(in crate::game::system::render) clip_offset: u32,
}

impl RenderSystem {
    pub(in crate::game::system::render) fn select_scene3d_camera(
        runtime: &Runtime,
        root: Entity,
        preferred: Option<Entity>,
    ) -> Result<Option<Entity>, LayerTraversalError> {
        camera::select_scene3d_camera(runtime, root, preferred)
    }

    pub(in crate::game::system::render) fn render_scene3d(
        entity: Entity,
        context: &mut LayerRenderContext<'_, '_>,
    ) {
        let runtime = context.runtime;

        let scene_guard = match runtime.block_on(entity.get_component::<Scene3D>()) {
            Some(scene) => scene,
            None => return,
        };
        let scene_near = scene_guard.near_plane();
        let scene_distance = scene_guard.view_distance();
        let attached_camera = scene_guard.attached_camera();

        let (vertex_shader, fragment_shader) = {
            let layer_guard = match runtime.block_on(entity.get_component::<Layer>()) {
                Some(layer) => layer,
                None => {
                    warn!(
                        target: "jge-core",
                        layer_id = %entity.id(),
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
                        layer_id = %entity.id(),
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
                        layer_id = %entity.id(),
                        "Scene3D layer has no fragment shader attached"
                    );
                    return;
                }
            };

            (vertex, fragment)
        };

        let viewport_framebuffer_size = context
            .viewport
            .map(|viewport| (viewport.width, viewport.height))
            .unwrap_or(context.framebuffer_size);

        let (width, height) = viewport_framebuffer_size;
        if width == 0 || height == 0 {
            warn!(
                target: "jge-core",
                layer_id = %entity.id(),
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
                    layer_id = %entity.id(),
                    error = %error,
                    "failed to prepare Scene3D pipeline"
                );
                return;
            }
        };

        let camera_entity = match Self::select_scene3d_camera(runtime, entity, attached_camera) {
            Ok(camera) => match camera {
                Some(camera) => camera,
                None => {
                    warn!(
                        target: "jge-core",
                        layer_id = %entity.id(),
                        "Scene3D layer has no camera available"
                    );
                    return;
                }
            },
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = %entity.id(),
                    error = %error,
                    "Scene3D camera lookup failed"
                );
                return;
            }
        };

        let camera_guard = match runtime.block_on(camera_entity.get_component::<Camera>()) {
            Some(camera) => camera,
            None => {
                warn!(
                    target: "jge-core",
                    layer_id = %entity.id(),
                    camera_id = %camera_entity.id(),
                    "Scene3D camera missing Camera component"
                );
                return;
            }
        };
        let scene_vertical = match scene_guard.vertical_fov_for_height(height) {
            Ok(value) => value,
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = %entity.id(),
                    framebuffer_height = height,
                    error = %error,
                    "Scene3D viewport config invalid"
                );
                return;
            }
        };

        let camera_vertical = match camera_guard.vertical_fov_for_height(height) {
            Ok(value) => value,
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = %entity.id(),
                    camera_id = %camera_entity.id(),
                    framebuffer_height = height,
                    error = %error,
                    "Scene3D camera viewport config invalid"
                );
                return;
            }
        };
        let camera_near = camera_guard.near_plane();
        let camera_far = camera_guard.far_plane();
        drop(camera_guard);

        let near_plane = camera_near.max(scene_near);
        let far_plane = camera_far.min(scene_distance);
        match near_plane.partial_cmp(&far_plane) {
            Some(std::cmp::Ordering::Less) => {}
            _ => {
                warn!(
                    target: "jge-core",
                    layer_id = %entity.id(),
                    near_plane,
                    far_plane,
                    "Scene3D clip range invalid"
                );
                return;
            }
        }

        let vertical_fov = scene_vertical.min(camera_vertical);
        let aspect_ratio = width as f32 / height as f32;

        let visible = match runtime
            .block_on(scene_guard.visible_renderables(camera_entity, viewport_framebuffer_size))
        {
            Ok(collection) => collection,
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = %entity.id(),
                    camera_id = %camera_entity.id(),
                    error = %error,
                    "Scene3D visibility query failed"
                );
                return;
            }
        };

        let camera_world = match runtime.block_on(Transform::world_matrix(camera_entity)) {
            Some(matrix) => matrix,
            None => {
                warn!(
                    target: "jge-core",
                    layer_id = %entity.id(),
                    camera_id = %camera_entity.id(),
                    "Scene3D camera missing (hierarchical) Transform"
                );
                return;
            }
        };
        let camera_position = Transform::translation_from_matrix(&camera_world);
        let basis = Transform::basis_from_matrix(&camera_world).normalize();

        let point_light_entities = match runtime.block_on(Layer::point_light_entities(entity)) {
            Ok(lights) => lights,
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = %entity.id(),
                    error = %error,
                    "Scene3D lighting query failed"
                );
                Vec::new()
            }
        };

        let parallel_light_entities = match runtime.block_on(Layer::parallel_light_entities(entity))
        {
            Ok(lights) => lights,
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = %entity.id(),
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
                let light = runtime.block_on(light_entity.get_component::<Light>())?;
                let point = runtime.block_on(light_entity.get_component::<PointLight>())?;
                let world = runtime.block_on(Transform::world_matrix(light_entity))?;
                let radius = point.distance();
                let intensity = light.lightness();
                let position = Transform::translation_from_matrix(&world);
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
                let light = runtime.block_on(light_entity.get_component::<Light>())?;
                let parallel = runtime.block_on(light_entity.get_component::<ParallelLight>())?;
                let world = runtime.block_on(Transform::world_matrix(light_entity))?;
                let intensity = light.lightness();
                drop(parallel);
                drop(light);
                if intensity <= 0.0 {
                    return None;
                }
                let rotation_matrix = world.fixed_view::<3, 3>(0, 0).into_owned();
                let forward = rotation_matrix * Vector3::new(0.0, -1.0, 0.0);
                let incoming = -forward;
                incoming
                    .try_normalize(1.0e-6)
                    .map(|direction| SceneParallelLight {
                        direction,
                        intensity,
                    })
            })
            .collect();

        if point_lights.len() > MAX_SCENE3D_POINT_LIGHTS {
            warn!(
                target: "jge-core",
                layer_id = %entity.id(),
                light_count = point_lights.len(),
                max_supported = MAX_SCENE3D_POINT_LIGHTS,
                "Scene3D truncating point lights to fit uniform capacity"
            );
            point_lights.truncate(MAX_SCENE3D_POINT_LIGHTS);
        }

        if parallel_lights.len() > MAX_SCENE3D_PARALLEL_LIGHTS {
            warn!(
                target: "jge-core",
                layer_id = %entity.id(),
                light_count = parallel_lights.len(),
                max_supported = MAX_SCENE3D_PARALLEL_LIGHTS,
                "Scene3D truncating parallel lights to fit uniform capacity"
            );
            parallel_lights.truncate(MAX_SCENE3D_PARALLEL_LIGHTS);
        }

        let material_layout = pipeline.material_layout();
        let clip_layout = pipeline.clip_layout();
        let mut draws = Vec::new();
        let mut total_vertices = 0u32;

        let clip_uniform_size = std::mem::size_of::<[f32; 8]>();
        let min_alignment = context
            .device
            .limits()
            .min_uniform_buffer_offset_alignment as usize;
        let clip_stride = clip_uniform_size.div_ceil(min_alignment) * min_alignment;
        let mut clip_data: Vec<u8> = Vec::new();

        for bundle in visible.iter() {
            let triangles = bundle.triangles();
            if triangles.is_empty() {
                continue;
            }

            let clip_uniform: [f32; 8] = match bundle.clip_aabb() {
                Some(clip) => [
                    clip.min.x,
                    clip.min.y,
                    clip.min.z,
                    1.0,
                    clip.max.x,
                    clip.max.y,
                    clip.max.z,
                    0.0,
                ],
                None => [0.0; 8],
            };

            let _profile_scope = context
                .caches
                .profiler
                .entity_scope(context.runtime, bundle.entity());

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

            let clip_offset = clip_data.len() as u32;
            clip_data.extend_from_slice(util::cast_slice_f32(&clip_uniform));
            clip_data.resize(clip_offset as usize + clip_stride, 0);

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
                            layer_id = %entity.id(),
                            bundle_entity = %bundle.entity().id(),
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
                clip_offset,
            });
        }

        if draws.is_empty() {
            trace!(
                target: "jge-core",
                layer_id = %entity.id(),
                "Scene3D layer has no visible geometry"
            );
            return;
        }

        let clip_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scene3D Clip Buffer"),
            size: clip_data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        context.queue.write_buffer(&clip_buffer, 0, &clip_data);

        let clip_bind_group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scene3D Clip Bind Group"),
            layout: clip_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &clip_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(clip_uniform_size as u64),
                }),
            }],
        });

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
            load: if context.viewport.is_some() {
                wgpu::LoadOp::Load
            } else {
                context.load_op
            },
            store: wgpu::StoreOp::Store,
        };

        let depth_view = context
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

        pass.set_pipeline(pipeline.pipeline());
        pass.set_bind_group(0, &scene_bind_group, &[]);
        for draw in &draws {
            pass.set_bind_group(1, &draw.bind_group, &[]);
            pass.set_bind_group(2, &clip_bind_group, &[draw.clip_offset]);
            pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
            pass.draw(0..draw.vertex_count, 0..1);
        }

        trace!(
            target: "jge-core",
            layer_id = %entity.id(),
            camera_id = %camera_entity.id(),
            draw_calls = draws.len(),
            vertex_total = total_vertices,
            "Scene3D layer rendered"
        );
    }

    pub(in crate::game::system::render) fn render_scene3d_from_snapshot(
        entity: Entity,
        snapshot: &Scene3DSnapshot,
        context: &mut LayerRenderContext<'_, '_>,
    ) {
        let viewport_framebuffer_size = context
            .viewport
            .map(|viewport| (viewport.width, viewport.height))
            .unwrap_or(context.framebuffer_size);

        let (width, height) = viewport_framebuffer_size;
        if width == 0 || height == 0 {
            warn!(
                target: "jge-core",
                layer_id = %entity.id(),
                framebuffer_width = width,
                framebuffer_height = height,
                "Scene3D framebuffer has invalid dimensions"
            );
            return;
        }

        let pipeline = match context.caches.scene3d.ensure(
            context.device,
            context.surface_format,
            &snapshot.vertex_shader,
            &snapshot.fragment_shader,
        ) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = %entity.id(),
                    error = %error,
                    "failed to prepare Scene3D pipeline"
                );
                return;
            }
        };

        let visible = &snapshot.visible;

        let point_light_count = snapshot.point_lights.len().min(MAX_SCENE3D_POINT_LIGHTS);
        if snapshot.point_lights.len() > MAX_SCENE3D_POINT_LIGHTS {
            warn!(
                target: "jge-core",
                layer_id = %entity.id(),
                light_count = snapshot.point_lights.len(),
                max_supported = MAX_SCENE3D_POINT_LIGHTS,
                "Scene3D truncating point lights to fit uniform capacity"
            );
        }

        let parallel_light_count = snapshot
            .parallel_lights
            .len()
            .min(MAX_SCENE3D_PARALLEL_LIGHTS);
        if snapshot.parallel_lights.len() > MAX_SCENE3D_PARALLEL_LIGHTS {
            warn!(
                target: "jge-core",
                layer_id = %entity.id(),
                light_count = snapshot.parallel_lights.len(),
                max_supported = MAX_SCENE3D_PARALLEL_LIGHTS,
                "Scene3D truncating parallel lights to fit uniform capacity"
            );
        }

        let material_layout = pipeline.material_layout();
        let clip_layout = pipeline.clip_layout();
        let mut draws = Vec::new();
        let mut total_vertices = 0u32;

        let clip_uniform_size = std::mem::size_of::<[f32; 8]>();
        let min_alignment = context
            .device
            .limits()
            .min_uniform_buffer_offset_alignment as usize;
        let clip_stride = clip_uniform_size.div_ceil(min_alignment) * min_alignment;
        let mut clip_data: Vec<u8> = Vec::new();

        for bundle in visible.iter() {
            let triangles = bundle.triangles();
            if triangles.is_empty() {
                continue;
            }

            let clip_uniform: [f32; 8] = match bundle.clip_aabb() {
                Some(clip) => [
                    clip.min.x,
                    clip.min.y,
                    clip.min.z,
                    1.0,
                    clip.max.x,
                    clip.max.y,
                    clip.max.z,
                    0.0,
                ],
                None => [0.0; 8],
            };

            let _profile_scope = context
                .caches
                .profiler
                .entity_scope(context.runtime, bundle.entity());

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

            let clip_offset = clip_data.len() as u32;
            clip_data.extend_from_slice(util::cast_slice_f32(&clip_uniform));
            clip_data.resize(clip_offset as usize + clip_stride, 0);

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
                            layer_id = %entity.id(),
                            bundle_entity = %bundle.entity().id(),
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
                clip_offset,
            });
        }

        if draws.is_empty() {
            trace!(
                target: "jge-core",
                layer_id = %entity.id(),
                "Scene3D layer has no visible geometry"
            );
            return;
        }

        let clip_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scene3D Clip Buffer"),
            size: clip_data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        context.queue.write_buffer(&clip_buffer, 0, &clip_data);

        let clip_bind_group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scene3D Clip Bind Group"),
            layout: clip_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &clip_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(clip_uniform_size as u64),
                }),
            }],
        });

        let mut uniform_data = Vec::with_capacity(
            16 + 4 + MAX_SCENE3D_POINT_LIGHTS * 8 + MAX_SCENE3D_PARALLEL_LIGHTS * 8,
        );
        uniform_data.extend_from_slice(&snapshot.view_proj);
        uniform_data.extend_from_slice(&[
            point_light_count as f32,
            parallel_light_count as f32,
            0.0,
            0.0,
        ]);

        for index in 0..MAX_SCENE3D_POINT_LIGHTS {
            if index < point_light_count {
                let light = &snapshot.point_lights[index];
                uniform_data.extend_from_slice(&[
                    light.position[0],
                    light.position[1],
                    light.position[2],
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
            if index < parallel_light_count {
                let light = &snapshot.parallel_lights[index];
                uniform_data.extend_from_slice(&[
                    light.direction[0],
                    light.direction[1],
                    light.direction[2],
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
            load: if context.viewport.is_some() {
                wgpu::LoadOp::Load
            } else {
                context.load_op
            },
            store: wgpu::StoreOp::Store,
        };

        let depth_view = context
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

        pass.set_pipeline(pipeline.pipeline());
        pass.set_bind_group(0, &scene_bind_group, &[]);
        for draw in &draws {
            pass.set_bind_group(1, &draw.bind_group, &[]);
            pass.set_bind_group(2, &clip_bind_group, &[draw.clip_offset]);
            pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
            pass.draw(0..draw.vertex_count, 0..1);
        }

        trace!(
            target: "jge-core",
            layer_id = %entity.id(),
            camera_id = %snapshot.camera_entity.id(),
            draw_calls = draws.len(),
            vertex_total = total_vertices,
            "Scene3D layer rendered"
        );
    }
}
