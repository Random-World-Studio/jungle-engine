use std::{borrow::Cow, collections::HashMap, fmt, sync::Arc};

use anyhow::{Context, anyhow, ensure};
use image::GenericImageView;
use lodtree::{coords::OctVec, tree::Tree};
use tracing::{debug, warn};
use wgpu::{self, util::DeviceExt};

use super::camera::Camera;
use super::light::{Light, ParallelLight, PointLight};
use super::material::{Material, MaterialPatch};
use super::node::Node;
use super::renderable::Renderable;
use super::scene2d::Scene2D;
use super::scene3d::Scene3D;
use super::shape::Shape;
use super::transform::Transform;
use super::{component, component_impl};
use crate::game::{component::ComponentRead, entity::Entity};
use crate::resource::{Resource, ResourceHandle, ResourcePath};
use nalgebra::{Matrix4, Perspective3, Point3, Rotation3, Vector2, Vector3, Vector4};

const MAX_SCENE3D_POINT_LIGHTS: usize = 8;
const MAX_SCENE3D_PARALLEL_LIGHTS: usize = 4;

#[component(Renderable)]
#[derive(Debug)]
pub struct Layer {
    entity_id: Option<Entity>,
    shaders: HashMap<RenderPipelineStage, LayerShader>,
    spatial: LayerSpatialIndex,
}

#[component_impl]
impl Layer {
    /// 创建一个 Layer。
    #[default()]
    pub fn new() -> Self {
        Self {
            entity_id: None,
            shaders: HashMap::new(),
            spatial: LayerSpatialIndex::new(),
        }
    }

    /// 获取指定渲染阶段关联的着色器描述。
    pub fn shader(&self, stage: RenderPipelineStage) -> Option<&LayerShader> {
        self.shaders.get(&stage)
    }

    /// 为指定渲染阶段挂载着色器，返回之前的挂载（若存在）。
    pub fn attach_shader(
        &mut self,
        stage: RenderPipelineStage,
        shader: LayerShader,
    ) -> Option<LayerShader> {
        self.shaders.insert(stage, shader)
    }

    /// 通过资源路径与语言快速挂载着色器。
    pub fn attach_shader_from_path(
        &mut self,
        stage: RenderPipelineStage,
        language: ShaderLanguage,
        resource_path: ResourcePath,
    ) -> Result<Option<LayerShader>, LayerShaderAttachError> {
        let resource = Resource::from(resource_path.clone())
            .ok_or(LayerShaderAttachError::ResourceMissing(resource_path))?;
        Ok(self.attach_shader(stage, LayerShader::new(language, resource)))
    }

    /// 移除指定阶段的着色器。
    pub fn detach_shader(&mut self, stage: RenderPipelineStage) -> Option<LayerShader> {
        self.shaders.remove(&stage)
    }

    /// 基于给定目标更新八叉树 LOD，并返回本次迭代的变化。
    pub fn step_lod(&mut self, targets: &[OctVec], detail: u64) -> LayerLodUpdate {
        self.spatial.step_lod(targets, detail)
    }

    /// 当前处于激活状态的节点数量。
    pub fn chunk_count(&self) -> usize {
        self.spatial.chunk_count()
    }

    /// 返回所有激活节点的位置。
    pub fn chunk_positions(&self) -> Vec<OctVec> {
        self.spatial.chunk_positions()
    }

    /// 查询指定节点在给定半径内的激活邻居。
    pub fn chunk_neighbors(&self, center: OctVec, radius: u64) -> Vec<OctVec> {
        self.spatial.chunk_neighbors(center, radius)
    }

    /// 根据当前场景组件类型执行渲染。
    pub fn render(entity: Entity, context: &mut LayerRenderContext<'_, '_>) {
        if entity.get_component::<Scene2D>().is_some() {
            Self::render_scene2d(entity, context);
        } else if entity.get_component::<Scene3D>().is_some() {
            Self::render_scene3d(entity, context);
        } else {
            debug!(
                target: "jge-core",
                layer_id = entity.id(),
                "skip layer without registered renderer"
            );
        }
    }

    /// 重置空间索引，清空树结构。
    pub fn clear_spatial_index(&mut self) {
        self.spatial.clear();
    }

    /// 以该 Layer 为根，收集同一 Layer 树下的所有可渲染实体。
    ///
    /// - 会按照节点树的先序遍历顺序返回实体 ID。
    /// - 子树中若遇到其他 Layer，将整体跳过，交由对应 Layer 单独处理。
    /// - 若当前 Layer 自身处于其他 Layer 管理的子树内，则直接返回空列表。
    pub fn renderable_entities(root: Entity) -> Result<Vec<Entity>, LayerTraversalError> {
        let ordered = Self::traverse_layer_entities(root)?;
        let mut renderables = Vec::new();

        for entity in ordered {
            if let Some(renderable) = entity.get_component::<Renderable>() {
                if renderable.is_enabled() {
                    renderables.push(entity);
                }
            }
        }

        Ok(renderables)
    }

    pub fn point_light_entities(root: Entity) -> Result<Vec<Entity>, LayerTraversalError> {
        let ordered = Self::traverse_layer_entities(root)?;
        let mut lights = Vec::new();

        for entity in ordered {
            if entity.get_component::<Light>().is_some()
                && entity.get_component::<PointLight>().is_some()
            {
                lights.push(entity);
            }
        }

        Ok(lights)
    }

    pub fn parallel_light_entities(root: Entity) -> Result<Vec<Entity>, LayerTraversalError> {
        let ordered = Self::traverse_layer_entities(root)?;
        let mut lights = Vec::new();

        for entity in ordered {
            if entity.get_component::<Light>().is_some()
                && entity.get_component::<ParallelLight>().is_some()
            {
                lights.push(entity);
            }
        }

        Ok(lights)
    }

    /// 收集当前 Layer 下所有可渲染实体的世界空间三角面集合。
    ///
    /// 返回的顺序与 [`Self::renderable_entities`] 保持一致，便于复用渲染排序。
    pub fn world_triangles(root: Entity) -> Result<Vec<LayerTriangle>, LayerTraversalError> {
        let bundles = Self::gather_renderables(root)?;
        let mut triangles = Vec::new();
        for bundle in bundles {
            triangles.extend(bundle.into_layer_triangles());
        }
        Ok(triangles)
    }

    /// 收集当前 Layer 下所有可渲染实体的世界空间渲染数据。
    ///
    /// 每个条目包含实体标识、转换后的三角面列表以及（若存在）绑定的材质描述，
    /// 便于在 2D 与 3D 渲染路径之间共享基础数据。
    pub fn world_renderables(
        root: Entity,
    ) -> Result<Vec<LayerRenderableBundle>, LayerTraversalError> {
        Self::gather_renderables(root)
    }

    /// 构造渲染前数据集合，便于多处渲染路径共享。
    pub fn collect_renderables(
        root: Entity,
    ) -> Result<LayerRenderableCollection, LayerTraversalError> {
        Self::gather_renderables(root).map(LayerRenderableCollection::from_bundles)
    }

    fn gather_renderables(root: Entity) -> Result<Vec<LayerRenderableBundle>, LayerTraversalError> {
        let renderables = Self::renderable_entities(root)?;
        let mut bundles = Vec::new();

        for entity in renderables {
            let shape_guard = match entity.get_component::<Shape>() {
                Some(shape) => shape,
                None => continue,
            };
            let triangle_count = shape_guard.triangle_count();
            if triangle_count == 0 {
                continue;
            }

            let transform_guard = match entity.get_component::<Transform>() {
                Some(transform) => transform,
                None => continue,
            };
            let matrix = transform_guard.matrix();
            drop(transform_guard);

            let mut world_triangles = Vec::with_capacity(triangle_count);
            for (a, b, c) in shape_guard.triangles() {
                world_triangles.push([
                    transform_vertex_to_world(&matrix, a),
                    transform_vertex_to_world(&matrix, b),
                    transform_vertex_to_world(&matrix, c),
                ]);
            }
            drop(shape_guard);

            let material = entity.get_component::<Material>().map(|material_guard| {
                let regions: Vec<MaterialPatch> =
                    material_guard.regions().iter().copied().collect();
                let regions_arc = Arc::<[MaterialPatch]>::from(regions);
                LayerMaterialDescriptor::new(material_guard.resource(), regions_arc)
            });

            bundles.push(LayerRenderableBundle::new(
                entity,
                world_triangles,
                material,
            ));
        }

        Ok(bundles)
    }

    fn traverse_layer_entities(root: Entity) -> Result<Vec<Entity>, LayerTraversalError> {
        if root.get_component::<Layer>().is_none() {
            return Err(LayerTraversalError::MissingLayer(root));
        }

        if root.get_component::<Node>().is_none() {
            return Err(LayerTraversalError::MissingNode(root));
        }

        if Self::has_layer_ancestor(root)? {
            return Ok(Vec::new());
        }

        let mut stack = vec![root];
        let mut ordered = Vec::new();

        while let Some(current) = stack.pop() {
            if current != root && current.get_component::<Layer>().is_some() {
                continue;
            }

            ordered.push(current);

            let child_ids = {
                let node_guard = current
                    .get_component::<Node>()
                    .ok_or(LayerTraversalError::MissingNode(current))?;
                let mut ids: Vec<Entity> = node_guard.children().iter().copied().collect();
                ids.reverse();
                ids
            };

            for child in child_ids {
                stack.push(child);
            }
        }

        Ok(ordered)
    }

    fn has_layer_ancestor(entity: Entity) -> Result<bool, LayerTraversalError> {
        let mut current_parent = {
            let node_guard = entity
                .get_component::<Node>()
                .ok_or(LayerTraversalError::MissingNode(entity))?;
            node_guard.parent()
        };

        while let Some(id) = current_parent {
            if id.get_component::<Layer>().is_some() {
                return Ok(true);
            }

            let parent = {
                let node_guard = id
                    .get_component::<Node>()
                    .ok_or(LayerTraversalError::MissingNode(id))?;
                node_guard.parent()
            };
            current_parent = parent;
        }

        Ok(false)
    }

    fn select_scene3d_camera(
        root: Entity,
        preferred: Option<Entity>,
    ) -> Result<Option<Entity>, LayerTraversalError> {
        if let Some(candidate) = preferred {
            if candidate.get_component::<Camera>().is_some()
                && candidate.get_component::<Transform>().is_some()
            {
                return Ok(Some(candidate));
            }
        }

        let ordered = Self::traverse_layer_entities(root)?;
        for entity in ordered {
            if entity.get_component::<Camera>().is_some()
                && entity.get_component::<Transform>().is_some()
            {
                return Ok(Some(entity));
            }
        }
        Ok(None)
    }

    fn render_scene3d(entity: Entity, context: &mut LayerRenderContext<'_, '_>) {
        let scene_guard = match entity.get_component::<Scene3D>() {
            Some(scene) => scene,
            None => return,
        };
        let scene_near = scene_guard.near_plane();
        let scene_distance = scene_guard.view_distance();
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
                    "Scene3D pipeline initialization failed"
                );
                return;
            }
        };

        let camera_entity = match Self::select_scene3d_camera(entity, attached_camera) {
            Ok(Some(camera)) => camera,
            Ok(None) => {
                warn!(
                    target: "jge-core",
                    layer_id = entity.id(),
                    "Scene3D layer missing active camera"
                );
                return;
            }
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = entity.id(),
                    error = %error,
                    "Scene3D camera traversal failed"
                );
                return;
            }
        };

        let (camera_near, camera_far) = {
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
            (camera_guard.near_plane(), camera_guard.far_plane())
        };

        let (width, height) = context.framebuffer_size;
        if width == 0 || height == 0 {
            warn!(
                target: "jge-core",
                layer_id = entity.id(),
                framebuffer_width = width,
                framebuffer_height = height,
                "Scene3D viewport dimensions invalid"
            );
            return;
        }

        let scene_vertical = match scene_guard.vertical_fov_for_height(height) {
            Ok(value) => value,
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = entity.id(),
                    error = %error,
                    framebuffer_height = height,
                    "Scene3D vertical FOV computation failed"
                );
                return;
            }
        };

        let camera_vertical = match camera_entity.get_component::<Camera>() {
            Some(camera) => match camera.vertical_fov_for_height(height) {
                Ok(value) => value,
                Err(error) => {
                    warn!(
                        target: "jge-core",
                        layer_id = entity.id(),
                        camera_id = camera_entity.id(),
                        error = %error,
                        framebuffer_height = height,
                        "Scene3D camera vertical FOV computation failed"
                    );
                    return;
                }
            },
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

        let transform_guard = match camera_entity.get_component::<Transform>() {
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
                        contents: cast_slice_f32(&vertex_data),
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
        let view_proj = opengl_to_wgpu_matrix() * projection.to_homogeneous() * view;

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
                contents: cast_slice_f32(&uniform_data),
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

    fn render_scene2d(entity: Entity, context: &mut LayerRenderContext<'_, '_>) {
        let scene_guard = match entity.get_component::<Scene2D>() {
            Some(scene) => scene,
            None => return,
        };
        let scene_offset = scene_guard.offset();
        let pixels_per_unit = scene_guard.pixels_per_unit();
        drop(scene_guard);

        let layer_guard = match entity.get_component::<Layer>() {
            Some(layer) => layer,
            None => {
                warn!(
                    target: "jge-core",
                    layer_id = entity.id(),
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
                    layer_id = entity.id(),
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
                    layer_id = entity.id(),
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
                        layer_id = entity.id(),
                        error = %error,
                        "failed to prepare Scene2D pipeline"
                    );
                    return;
                }
            };

        let point_light_entities = match Layer::point_light_entities(entity) {
            Ok(lights) => lights,
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = entity.id(),
                    error = %error,
                    "Scene2D lighting query failed"
                );
                Vec::new()
            }
        };

        let parallel_light_brightness = match Layer::parallel_light_entities(entity) {
            Ok(lights) => lights
                .into_iter()
                .filter_map(|light_entity| {
                    let light = light_entity.get_component::<Light>()?;
                    let value = light.lightness();
                    drop(light);
                    if value <= 0.0 { None } else { Some(value) }
                })
                .sum::<f32>(),
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = entity.id(),
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
                let light = light_entity.get_component::<Light>()?;
                let point = light_entity.get_component::<PointLight>()?;
                let transform = light_entity.get_component::<Transform>()?;
                let radius = point.distance();
                let lightness = light.lightness();
                let position = transform.position();
                drop(transform);
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

        let renderables = match Layer::collect_renderables(entity) {
            Ok(collection) => collection,
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = entity.id(),
                    error = %error,
                    "Scene2D renderable collection failed"
                );
                return;
            }
        };

        let scene_guard = match entity.get_component::<Scene2D>() {
            Some(scene) => scene,
            None => {
                warn!(
                    target: "jge-core",
                    layer_id = entity.id(),
                    "Scene2D component disappeared before visibility query"
                );
                return;
            }
        };
        let layer_guard = match entity.get_component::<Layer>() {
            Some(layer) => layer,
            None => {
                warn!(
                    target: "jge-core",
                    layer_id = entity.id(),
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
                        layer_id = entity.id(),
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
                                layer_id = entity.id(),
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
                for (vertex_index, vertex) in triangle.iter().enumerate() {
                    let point_brightness = evaluate_point_lights(vertex, &point_lights)
                        .clamp(0.0, MAX_POINT_LIGHT_BRIGHTNESS);
                    let brightness = (directional_brightness + point_brightness)
                        .clamp(0.0, MAX_TOTAL_BRIGHTNESS);
                    let ndc = scene2d_vertex_to_ndc(
                        context.framebuffer_size,
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
                        contents: cast_slice_f32(&vertex_data),
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
                layer_id = entity.id(),
                "Scene2D layer has no visible draws"
            );
            return;
        }

        let label = format!("Layer {}", entity.id());
        let ops = wgpu::Operations {
            load: context.load_op,
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

        pass.set_pipeline(&render_pipeline);
        for draw in &draws {
            pass.set_bind_group(0, &draw.bind_group, &[]);
            pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
            pass.draw(0..draw.vertex_count, 0..1);
        }

        debug!(
            target: "jge-core",
            layer_id = entity.id(),
            draw_calls = draws.len(),
            vertex_total = total_vertices,
            "Scene2D layer rendered"
        );
    }
}

#[derive(Clone)]
struct Scene2DMaterialInstance {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

struct Scene2DMaterialCache {
    sampler: Option<wgpu::Sampler>,
    default: Option<Scene2DMaterialInstance>,
    materials: HashMap<usize, Scene2DMaterialInstance>,
    pipeline_generation: u64,
}

impl Default for Scene2DMaterialCache {
    fn default() -> Self {
        Self {
            sampler: None,
            default: None,
            materials: HashMap::new(),
            pipeline_generation: 0,
        }
    }
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

    fn ensure_default(
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

            let (data, bytes_per_row) = pad_rgba_data(vec![255, 255, 255, 255], 1, 1)?;
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
                texture,
                view,
                bind_group,
            });
        }

        self.default
            .as_ref()
            .ok_or_else(|| anyhow!("failed to prepare default material instance"))
    }

    fn ensure_material(
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
            .context("failed to decode material texture resource")?;
        let (width, height) = image.dimensions();
        ensure!(
            width > 0 && height > 0,
            "material texture has invalid dimensions"
        );

        let rgba = image.to_rgba8().into_raw();
        let (data, bytes_per_row) = pad_rgba_data(rgba, width, height)?;

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
            texture,
            view,
            bind_group,
        })
    }
}

pub struct LayerRendererCache {
    scene2d: Scene2DPipelineCache,
    scene2d_materials: Scene2DMaterialCache,
    scene3d: Scene3DPipelineCache,
    scene3d_materials: Scene3DMaterialCache,
}

impl Default for LayerRendererCache {
    fn default() -> Self {
        Self {
            scene2d: Scene2DPipelineCache::default(),
            scene2d_materials: Scene2DMaterialCache::default(),
            scene3d: Scene3DPipelineCache::default(),
            scene3d_materials: Scene3DMaterialCache::default(),
        }
    }
}

impl LayerRendererCache {
    pub fn new() -> Self {
        Self::default()
    }
}

pub struct LayerRenderContext<'a, 'cache> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub target_view: &'a wgpu::TextureView,
    pub surface_format: wgpu::TextureFormat,
    pub framebuffer_size: (u32, u32),
    pub load_op: wgpu::LoadOp<wgpu::Color>,
    pub caches: &'cache mut LayerRendererCache,
}

#[derive(Debug, Clone)]
pub struct LayerMaterialDescriptor {
    resource: ResourceHandle,
    regions: Arc<[MaterialPatch]>,
}

impl LayerMaterialDescriptor {
    fn new(resource: ResourceHandle, regions: Arc<[MaterialPatch]>) -> Self {
        Self { resource, regions }
    }

    pub fn resource(&self) -> &ResourceHandle {
        &self.resource
    }

    pub fn regions(&self) -> &[MaterialPatch] {
        &self.regions
    }
}

#[derive(Debug, Clone)]
pub struct LayerRenderableBundle {
    entity: Entity,
    triangles: Vec<[Vector3<f32>; 3]>,
    material: Option<LayerMaterialDescriptor>,
}

impl LayerRenderableBundle {
    pub(crate) fn new(
        entity: Entity,
        triangles: Vec<[Vector3<f32>; 3]>,
        material: Option<LayerMaterialDescriptor>,
    ) -> Self {
        Self {
            entity,
            triangles,
            material,
        }
    }

    pub fn entity(&self) -> Entity {
        self.entity
    }

    pub fn triangles(&self) -> &[[Vector3<f32>; 3]] {
        &self.triangles
    }

    pub fn material(&self) -> Option<&LayerMaterialDescriptor> {
        self.material.as_ref()
    }

    pub fn into_material(self) -> Option<LayerMaterialDescriptor> {
        self.material
    }

    pub fn into_components(
        self,
    ) -> (
        Entity,
        Vec<[Vector3<f32>; 3]>,
        Option<LayerMaterialDescriptor>,
    ) {
        (self.entity, self.triangles, self.material)
    }

    fn into_layer_triangles(self) -> Vec<LayerTriangle> {
        let entity = self.entity;
        self.triangles
            .into_iter()
            .map(|triangle| LayerTriangle::new(entity, triangle))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct LayerRenderableCollection {
    bundles: Vec<LayerRenderableBundle>,
    lookup: HashMap<Entity, usize>,
}

impl LayerRenderableCollection {
    pub(crate) fn from_bundles(bundles: Vec<LayerRenderableBundle>) -> Self {
        let lookup = bundles
            .iter()
            .enumerate()
            .map(|(index, bundle)| (bundle.entity(), index))
            .collect();
        Self { bundles, lookup }
    }

    pub fn bundles(&self) -> &[LayerRenderableBundle] {
        &self.bundles
    }

    pub fn iter(&self) -> impl Iterator<Item = &LayerRenderableBundle> {
        self.bundles.iter()
    }

    pub fn get(&self, entity: Entity) -> Option<&LayerRenderableBundle> {
        self.lookup
            .get(&entity)
            .and_then(|index| self.bundles.get(*index))
    }

    pub fn material(&self, entity: Entity) -> Option<&LayerMaterialDescriptor> {
        self.get(entity).and_then(|bundle| bundle.material())
    }

    pub fn into_bundles(self) -> Vec<LayerRenderableBundle> {
        self.bundles
    }
}

impl ComponentRead<Layer> {
    pub fn render(&self, context: &mut LayerRenderContext<'_, '_>) {
        Layer::render(self.entity(), context);
    }

    pub fn renderable_entities(&self) -> Result<Vec<Entity>, LayerTraversalError> {
        Layer::renderable_entities(self.entity())
    }

    pub fn point_light_entities(&self) -> Result<Vec<Entity>, LayerTraversalError> {
        Layer::point_light_entities(self.entity())
    }

    pub fn parallel_light_entities(&self) -> Result<Vec<Entity>, LayerTraversalError> {
        Layer::parallel_light_entities(self.entity())
    }

    pub fn world_renderables(&self) -> Result<Vec<LayerRenderableBundle>, LayerTraversalError> {
        Layer::world_renderables(self.entity())
    }

    pub fn collect_renderables(&self) -> Result<LayerRenderableCollection, LayerTraversalError> {
        Layer::collect_renderables(self.entity())
    }

    pub fn world_triangles(&self) -> Result<Vec<LayerTriangle>, LayerTraversalError> {
        Layer::world_triangles(self.entity())
    }
}

#[derive(Debug, Clone)]
pub struct LayerTriangle {
    entity: Entity,
    vertices: [Vector3<f32>; 3],
}

impl LayerTriangle {
    fn new(entity: Entity, vertices: [Vector3<f32>; 3]) -> Self {
        Self { entity, vertices }
    }

    pub fn entity(&self) -> Entity {
        self.entity
    }

    pub fn vertices(&self) -> &[Vector3<f32>; 3] {
        &self.vertices
    }

    pub fn into_vertices(self) -> [Vector3<f32>; 3] {
        self.vertices
    }
}

struct Scene2DPipelineCache {
    pipeline: Option<Scene2DPipeline>,
    generation: u64,
}

impl Default for Scene2DPipelineCache {
    fn default() -> Self {
        Self {
            pipeline: None,
            generation: 0,
        }
    }
}

impl Scene2DPipelineCache {
    fn ensure(
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

struct Scene3DPipelineCache {
    pipeline: Option<Scene3DPipeline>,
}

impl Default for Scene3DPipelineCache {
    fn default() -> Self {
        Self { pipeline: None }
    }
}

impl Scene3DPipelineCache {
    fn ensure(
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

struct Scene3DPipeline {
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
            depth_stencil: None,
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

    fn uniform_layout(&self) -> &wgpu::BindGroupLayout {
        &self.uniform_layout
    }

    fn material_layout(&self) -> &wgpu::BindGroupLayout {
        &self.material_layout
    }

    fn pipeline(&self) -> &wgpu::RenderPipeline {
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
struct Scene3DShaderKey {
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

struct Scene3DDraw {
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    bind_group: wgpu::BindGroup,
}

#[derive(Clone)]
struct Scene3DMaterialInstance {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

struct Scene3DMaterialCache {
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

    fn ensure_default(
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

            let (data, bytes_per_row) = pad_rgba_data(vec![255, 255, 255, 255], 1, 1)?;
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
                texture,
                view,
                bind_group,
            });
        }

        self.default
            .as_ref()
            .ok_or_else(|| anyhow!("failed to prepare default 3D material instance"))
    }

    fn ensure_material(
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
        let (data, bytes_per_row) = pad_rgba_data(rgba, width, height)?;

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
            texture,
            view,
            bind_group,
        })
    }
}

struct Scene2DDraw {
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    bind_group: wgpu::BindGroup,
}

struct Scene2DPipeline {
    pipeline: wgpu::RenderPipeline,
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

    fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
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

fn transform_vertex_to_world(matrix: &Matrix4<f32>, local: &Vector3<f32>) -> Vector3<f32> {
    let vector = matrix * Vector4::new(local.x, local.y, local.z, 1.0);
    Vector3::new(vector.x, vector.y, vector.z)
}

fn scene2d_vertex_to_ndc(
    framebuffer_size: (u32, u32),
    vertex: &Vector3<f32>,
    offset: &Vector2<f32>,
    pixels_per_unit: f32,
) -> Vector3<f32> {
    let width = framebuffer_size.0.max(1) as f32;
    let height = framebuffer_size.1.max(1) as f32;
    let half_width = width * 0.5;
    let half_height = height * 0.5;

    let x_pixels = (vertex.x - offset.x) * pixels_per_unit;
    let y_pixels = (vertex.y - offset.y) * pixels_per_unit;

    let x_ndc = if half_width > 0.0 {
        x_pixels / half_width
    } else {
        0.0
    };
    let y_ndc = if half_height > 0.0 {
        y_pixels / half_height
    } else {
        0.0
    };

    Vector3::new(x_ndc, y_ndc, vertex.z)
}

/// Align raw RGBA rows to `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` for texture uploads.
fn pad_rgba_data(data: Vec<u8>, width: u32, height: u32) -> anyhow::Result<(Vec<u8>, u32)> {
    ensure!(
        width > 0 && height > 0,
        "texture dimensions must be positive"
    );

    let bytes_per_pixel = 4usize;
    let width_usize = width as usize;
    let height_usize = height as usize;
    let unpadded_bytes_per_row = width_usize * bytes_per_pixel;
    let expected_len = unpadded_bytes_per_row * height_usize;
    ensure!(
        data.len() == expected_len,
        "RGBA data length {} does not match expected {} for {}x{} texture",
        data.len(),
        expected_len,
        width,
        height,
    );

    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let padded_bytes_per_row = ((unpadded_bytes_per_row + alignment - 1) / alignment) * alignment;

    if padded_bytes_per_row == unpadded_bytes_per_row {
        return Ok((data, unpadded_bytes_per_row as u32));
    }

    let mut padded = vec![0u8; padded_bytes_per_row * height_usize];
    for row in 0..height_usize {
        let src_start = row * unpadded_bytes_per_row;
        let dst_start = row * padded_bytes_per_row;
        let src_end = src_start + unpadded_bytes_per_row;
        padded[dst_start..dst_start + unpadded_bytes_per_row]
            .copy_from_slice(&data[src_start..src_end]);
    }

    Ok((padded, padded_bytes_per_row as u32))
}

/// 将 OpenGL 风格裁剪空间转换为 wgpu/WebGPU 风格裁剪空间。
///
/// - 将 NDC Z 从 [-1, 1] 映射到 [0, 1]
///
/// 说明：这一步让我们可以继续使用 nalgebra 的 `Perspective3`（OpenGL 语义）生成投影矩阵，
/// 只做 Z 区间差异的修正。
fn opengl_to_wgpu_matrix() -> Matrix4<f32> {
    Matrix4::new(
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0,
    )
}

fn cast_slice_f32(data: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    }
}

/// Layer 遍历过程中可能出现的错误。
#[derive(Debug, PartialEq, Eq)]
pub enum LayerTraversalError {
    MissingLayer(Entity),
    MissingNode(Entity),
}

impl fmt::Display for LayerTraversalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerTraversalError::MissingLayer(entity) => {
                write!(f, "实体 {} 未注册 Layer 组件", entity.id())
            }
            LayerTraversalError::MissingNode(entity) => {
                write!(f, "实体 {} 未注册 Node 组件", entity.id())
            }
        }
    }
}

impl std::error::Error for LayerTraversalError {}

/// 渲染管线可用的着色器阶段。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderPipelineStage {
    Vertex,
    Fragment,
}

/// 支持的着色器语言。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderLanguage {
    Glsl,
    Wgsl,
}

/// Layer 上挂载的着色器描述。
#[derive(Debug, Clone)]
pub struct LayerShader {
    language: ShaderLanguage,
    resource: ResourceHandle,
}

impl LayerShader {
    pub fn new(language: ShaderLanguage, resource: ResourceHandle) -> Self {
        Self { language, resource }
    }

    pub fn language(&self) -> ShaderLanguage {
        self.language
    }

    pub fn resource(&self) -> &ResourceHandle {
        &self.resource
    }

    /// 获取挂载的资源句柄副本。
    pub fn resource_handle(&self) -> ResourceHandle {
        Arc::clone(&self.resource)
    }
}

impl PartialEq for LayerShader {
    fn eq(&self, other: &Self) -> bool {
        self.language == other.language && Arc::ptr_eq(&self.resource, &other.resource)
    }
}

impl Eq for LayerShader {}

#[derive(Debug, PartialEq, Eq)]
pub enum LayerShaderAttachError {
    ResourceMissing(ResourcePath),
}

impl fmt::Display for LayerShaderAttachError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerShaderAttachError::ResourceMissing(path) => {
                write!(f, "资源路径 {} 未注册", path.join("/"))
            }
        }
    }
}

impl std::error::Error for LayerShaderAttachError {}

/// Layer 空间索引操作过程中可能出现的错误。
#[derive(Debug, PartialEq, Eq)]
pub enum LayerSpatialError {
    MissingLayer(Entity),
}

impl fmt::Display for LayerSpatialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerSpatialError::MissingLayer(entity) => {
                write!(f, "实体 {} 未注册 Layer 组件", entity.id())
            }
        }
    }
}

impl std::error::Error for LayerSpatialError {}

/// LOD 更新过程中的节点变更集合。
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LayerLodUpdate {
    pub added: Vec<OctVec>,
    pub removed: Vec<OctVec>,
    pub activated: Vec<OctVec>,
    pub deactivated: Vec<OctVec>,
}

impl LayerLodUpdate {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty()
            && self.removed.is_empty()
            && self.activated.is_empty()
            && self.deactivated.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LayerChunk {
    position: OctVec,
    active: bool,
}

impl LayerChunk {
    fn new(position: OctVec) -> Self {
        Self {
            position,
            active: true,
        }
    }

    fn position(&self) -> OctVec {
        self.position
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn set_active(&mut self, value: bool) {
        self.active = value;
    }

    fn reset(&mut self, position: OctVec) {
        self.position = position;
        self.active = true;
    }
}

#[derive(Debug)]
struct LayerSpatialIndex {
    tree: Tree<LayerChunk, OctVec>,
}

impl LayerSpatialIndex {
    fn new() -> Self {
        Self { tree: Tree::new() }
    }

    fn clear(&mut self) {
        self.tree.clear();
    }

    fn chunk_count(&self) -> usize {
        (0..self.tree.get_num_chunks())
            .filter(|&idx| self.tree.get_chunk(idx).is_active())
            .count()
    }

    fn chunk_positions(&self) -> Vec<OctVec> {
        (0..self.tree.get_num_chunks())
            .filter_map(|idx| {
                let chunk = self.tree.get_chunk(idx);
                if chunk.is_active() {
                    Some(chunk.position())
                } else {
                    None
                }
            })
            .collect()
    }

    fn chunk_neighbors(&self, center: OctVec, radius: u64) -> Vec<OctVec> {
        let radius = radius as i64;
        self.chunk_positions()
            .into_iter()
            .filter(|pos| pos.depth == center.depth)
            .filter(|pos| {
                let dx = pos.x as i64 - center.x as i64;
                let dy = pos.y as i64 - center.y as i64;
                let dz = pos.z as i64 - center.z as i64;
                dx.abs() <= radius && dy.abs() <= radius && dz.abs() <= radius
            })
            .collect()
    }

    fn step_lod(&mut self, targets: &[OctVec], detail: u64) -> LayerLodUpdate {
        let mut update = LayerLodUpdate::default();

        if targets.is_empty() {
            return update;
        }

        let needs_update = self
            .tree
            .prepare_update(targets, detail, |position| LayerChunk::new(position));

        if !needs_update {
            return update;
        }

        {
            let pending = self.tree.get_chunks_to_add_slice_mut();
            update.added.reserve(pending.len());
            for (position, chunk) in pending.iter_mut() {
                chunk.reset(*position);
                update.added.push(*position);
            }
        }

        for i in 0..self.tree.get_num_chunks_to_activate() {
            let chunk = self.tree.get_chunk_to_activate_mut(i);
            chunk.set_active(true);
            update.activated.push(chunk.position());
        }

        for i in 0..self.tree.get_num_chunks_to_deactivate() {
            let chunk = self.tree.get_chunk_to_deactivate_mut(i);
            chunk.set_active(false);
            update.deactivated.push(chunk.position());
        }

        for i in 0..self.tree.get_num_chunks_to_remove() {
            let chunk = self.tree.get_chunk_to_remove(i);
            update.removed.push(chunk.position());
        }

        self.tree.do_update();

        update
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::entity::Entity;
    use crate::resource::{Resource, ResourceHandle};
    use lodtree::{coords::OctVec, traits::LodVec};
    use parking_lot::RwLock;

    fn detach_node(entity: Entity) {
        if let Some(mut node) = entity.get_component_mut::<Node>() {
            let _ = node.detach();
        }
    }

    fn attach_node(entity: Entity, parent: Entity, message: &str) {
        let mut parent_node = parent
            .get_component_mut::<Node>()
            .expect("父实体应持有 Node 组件");
        parent_node.attach(entity).expect(message);
    }

    fn prepare_node(entity: &Entity, name: &str) {
        let _ = entity.unregister_component::<Layer>();
        let _ = entity.unregister_component::<Renderable>();
        detach_node(*entity);
        let _ = entity.unregister_component::<Node>();
        let _ = entity
            .register_component(Node::new(name).expect("应能创建节点"))
            .expect("应能注册 Node");
    }

    fn cleanup(entity: &Entity) {
        let _ = entity.unregister_component::<Layer>();
        let _ = entity.unregister_component::<Renderable>();
        detach_node(*entity);
        let _ = entity.unregister_component::<Node>();
    }

    #[test]
    fn layer_requires_renderable_dependency() {
        let entity = Entity::new().expect("应能创建实体");
        let inserted = entity
            .register_component(Layer::new())
            .expect("缺少 Renderable 时应自动注册依赖");
        assert!(inserted.is_none());

        assert!(
            entity.get_component::<Renderable>().is_some(),
            "依赖的 Renderable 应已注册"
        );
        assert!(
            entity.get_component::<Node>().is_some(),
            "Renderable 的依赖 Node 应已注册"
        );

        let previous_auto = entity
            .register_component(Layer::new())
            .expect("重复插入应返回旧的 Layer");
        assert!(previous_auto.is_some());

        prepare_node(&entity, "layer_root");
        let _ = entity
            .register_component(Renderable::new())
            .expect("应能插入 Renderable");

        let reinserted = entity
            .register_component(Layer::new())
            .expect("满足依赖后应能插入 Layer");
        assert!(reinserted.is_none());

        cleanup(&entity);
    }

    #[test]
    fn renderable_entities_skip_nested_layers() {
        let root = Entity::new().expect("应能创建 root 实体");
        let child_a = Entity::new().expect("应能创建 child_a 实体");
        let child_b = Entity::new().expect("应能创建 child_b 实体");
        let nested_root = Entity::new().expect("应能创建 nested_root 实体");
        let nested_child = Entity::new().expect("应能创建 nested_child 实体");

        prepare_node(&root, "root");
        prepare_node(&child_a, "child_a");
        prepare_node(&child_b, "child_b");
        prepare_node(&nested_root, "nested_root");
        prepare_node(&nested_child, "nested_child");

        // 构建节点树
        attach_node(child_a, root, "应能挂载 child_a");
        attach_node(child_b, root, "应能挂载 child_b");
        attach_node(nested_root, child_b, "应能挂载 nested_root");
        attach_node(nested_child, nested_root, "应能挂载 nested_child");

        // 注册可渲染组件
        let _ = root
            .register_component(Renderable::new())
            .expect("应能插入 root rend");
        let _ = child_a
            .register_component(Renderable::new())
            .expect("应能插入 child_a rend");
        let _ = child_b
            .register_component(Renderable::new())
            .expect("应能插入 child_b rend");
        let _ = nested_root
            .register_component(Renderable::new())
            .expect("应能插入 nested_root rend");
        let _ = nested_child
            .register_component(Renderable::new())
            .expect("应能插入 nested_child rend");

        // 注册 Layer（根以及子树）
        let _ = root
            .register_component(Layer::new())
            .expect("应能插入 root layer");
        let _ = nested_root
            .register_component(Layer::new())
            .expect("应能插入 nested layer");

        let root_renderables =
            Layer::renderable_entities(root).expect("应能收集 root layer 的渲染实体");
        assert_eq!(root_renderables, vec![root, child_a, child_b]);

        let nested_renderables =
            Layer::renderable_entities(nested_root).expect("嵌套 Layer 应被忽略并返回空列表");
        assert!(nested_renderables.is_empty());

        // 清理（按照叶 -> 根顺序）
        cleanup(&nested_child);
        cleanup(&nested_root);
        cleanup(&child_b);
        cleanup(&child_a);
        cleanup(&root);
    }

    #[test]
    fn point_light_entities_skip_nested_layers() {
        let root = Entity::new().expect("应能创建 root 实体");
        let nested_root = Entity::new().expect("应能创建 nested_root 实体");
        let light_a = Entity::new().expect("应能创建 light_a 实体");
        let light_b = Entity::new().expect("应能创建 light_b 实体");

        prepare_node(&root, "root");
        prepare_node(&nested_root, "nested_root");
        prepare_node(&light_a, "light_a");
        prepare_node(&light_b, "light_b");

        attach_node(light_a, root, "应能挂载光源 A 到根");
        attach_node(nested_root, root, "应能挂载 nested_root");
        attach_node(light_b, nested_root, "应能挂载光源 B 到嵌套 Layer");

        let _ = root
            .register_component(Renderable::new())
            .expect("应能插入 root Renderable");
        let _ = nested_root
            .register_component(Renderable::new())
            .expect("应能插入 nested Renderable");

        let _ = root
            .register_component(Layer::new())
            .expect("应能插入 root Layer");
        let _ = nested_root
            .register_component(Layer::new())
            .expect("应能插入 nested Layer");

        let _ = light_a
            .register_component(Renderable::new())
            .expect("应能为光源 A 注册 Renderable");
        if let Some(mut renderable) = light_a.get_component_mut::<Renderable>() {
            renderable.set_enabled(false);
        }
        let _ = light_a
            .register_component(Transform::new())
            .expect("应能为光源 A 注册 Transform");
        if let Some(mut transform) = light_a.get_component_mut::<Transform>() {
            transform.set_position(Vector3::new(0.5, 0.0, 0.0));
        }
        let _ = light_a
            .register_component(Light::new(1.0))
            .expect("应能插入 Light 组件");
        let _ = light_a
            .register_component(PointLight::new(2.0))
            .expect("应能插入 PointLight 组件");

        let _ = light_b
            .register_component(Renderable::new())
            .expect("应能为光源 B 注册 Renderable");
        if let Some(mut renderable) = light_b.get_component_mut::<Renderable>() {
            renderable.set_enabled(false);
        }
        let _ = light_b
            .register_component(Transform::new())
            .expect("应能为光源 B 注册 Transform");
        if let Some(mut transform) = light_b.get_component_mut::<Transform>() {
            transform.set_position(Vector3::new(-0.25, 0.25, 0.0));
        }
        let _ = light_b
            .register_component(Light::new(1.0))
            .expect("应能插入嵌套 Light 组件");
        let _ = light_b
            .register_component(PointLight::new(2.0))
            .expect("应能插入嵌套 PointLight 组件");

        assert!(
            light_a.get_component::<Light>().is_some(),
            "光源 A 应具备 Light 组件"
        );
        assert!(
            light_a.get_component::<PointLight>().is_some(),
            "光源 A 应具备 PointLight 组件"
        );

        let root_lights = Layer::point_light_entities(root).expect("根 Layer 应能收集光源");
        assert_eq!(root_lights, vec![light_a], "根 Layer 应仅包含直接子光源");

        let nested_lights =
            Layer::point_light_entities(nested_root).expect("嵌套 Layer 应仅返回自身与子光源");
        assert!(
            nested_lights.is_empty(),
            "嵌套 Layer 的光源应由自身渲染时处理"
        );

        cleanup(&light_b);
        cleanup(&nested_root);
        cleanup(&light_a);
        cleanup(&root);
    }

    #[test]
    fn parallel_light_entities_skip_nested_layers() {
        let root = Entity::new().expect("应能创建 root 实体");
        let nested_root = Entity::new().expect("应能创建 nested_root 实体");
        let light_a = Entity::new().expect("应能创建平行光实体 A");
        let light_b = Entity::new().expect("应能创建平行光实体 B");

        prepare_node(&root, "root_parallel");
        prepare_node(&nested_root, "nested_parallel");
        prepare_node(&light_a, "parallel_a");
        prepare_node(&light_b, "parallel_b");

        attach_node(light_a, root, "应能挂载平行光 A");
        attach_node(nested_root, root, "应能挂载 nested_root");
        attach_node(light_b, nested_root, "应能挂载平行光 B 到嵌套 Layer");

        let _ = root
            .register_component(Renderable::new())
            .expect("应能插入 root Renderable");
        let _ = nested_root
            .register_component(Renderable::new())
            .expect("应能插入 nested Renderable");

        let _ = root
            .register_component(Layer::new())
            .expect("应能插入 root Layer");
        let _ = nested_root
            .register_component(Layer::new())
            .expect("应能插入 nested Layer");

        let _ = light_a
            .register_component(Renderable::new())
            .expect("应能为平行光 A 注册 Renderable");
        if let Some(mut renderable) = light_a.get_component_mut::<Renderable>() {
            renderable.set_enabled(false);
        }
        let _ = light_a
            .register_component(Transform::new())
            .expect("应能为平行光 A 注册 Transform");
        let _ = light_a
            .register_component(Light::new(0.8))
            .expect("应能插入光源亮度");
        let _ = light_a
            .register_component(ParallelLight::new())
            .expect("应能插入平行光组件");

        let _ = light_b
            .register_component(Renderable::new())
            .expect("应能为平行光 B 注册 Renderable");
        if let Some(mut renderable) = light_b.get_component_mut::<Renderable>() {
            renderable.set_enabled(false);
        }
        let _ = light_b
            .register_component(Transform::new())
            .expect("应能为平行光 B 注册 Transform");
        let _ = light_b
            .register_component(Light::new(0.5))
            .expect("应能插入嵌套光源亮度");
        let _ = light_b
            .register_component(ParallelLight::new())
            .expect("应能插入嵌套平行光组件");

        let root_lights = Layer::parallel_light_entities(root).expect("根 Layer 应能收集平行光");
        assert_eq!(root_lights, vec![light_a], "根 Layer 应仅包含直接子平行光");

        let nested_lights =
            Layer::parallel_light_entities(nested_root).expect("嵌套 Layer 平行光查询应成功");
        assert!(
            nested_lights.is_empty(),
            "嵌套 Layer 的平行光应由其自身处理"
        );

        cleanup(&light_b);
        cleanup(&nested_root);
        cleanup(&light_a);
        cleanup(&root);
    }

    #[test]
    fn layer_shader_attachment_cycle() {
        let mut layer = Layer::new();

        let glsl_resource: ResourceHandle =
            Arc::new(RwLock::new(Resource::from_memory(b"glsl".to_vec())));
        let glsl_shader = LayerShader::new(ShaderLanguage::Glsl, Arc::clone(&glsl_resource));

        assert!(
            layer
                .attach_shader(RenderPipelineStage::Vertex, glsl_shader.clone())
                .is_none()
        );
        let stored = layer
            .shader(RenderPipelineStage::Vertex)
            .expect("应能读取挂载的着色器");
        assert_eq!(stored.language(), ShaderLanguage::Glsl);
        assert!(Arc::ptr_eq(stored.resource(), &glsl_resource));

        let wgsl_resource: ResourceHandle =
            Arc::new(RwLock::new(Resource::from_memory(b"wgsl".to_vec())));
        let wgsl_shader = LayerShader::new(ShaderLanguage::Wgsl, Arc::clone(&wgsl_resource));

        assert_eq!(
            layer
                .attach_shader(RenderPipelineStage::Vertex, wgsl_shader.clone())
                .expect("后一次挂载应返回先前的着色器"),
            glsl_shader
        );

        let current = layer
            .shader(RenderPipelineStage::Vertex)
            .expect("应能读取替换后的着色器");
        assert_eq!(current.language(), ShaderLanguage::Wgsl);
        assert!(Arc::ptr_eq(current.resource(), &wgsl_resource));

        assert_eq!(
            layer
                .detach_shader(RenderPipelineStage::Vertex)
                .expect("移除操作应返回着色器"),
            wgsl_shader
        );
        assert!(layer.shader(RenderPipelineStage::Vertex).is_none());
    }

    #[test]
    fn layer_spatial_index_supports_octree_updates() {
        let entity = Entity::new().expect("应能创建实体");
        prepare_node(&entity, "layer_spatial_root");
        let _ = entity
            .register_component(Renderable::new())
            .expect("应能插入 Renderable");
        let _ = entity
            .register_component(Layer::new())
            .expect("应能插入 Layer");

        let mut layer = entity.get_component_mut::<Layer>().expect("应能写入 Layer");
        assert_eq!(layer.chunk_count(), 0);

        let target = OctVec::new(128, 128, 128, 5);
        let detail = 3;
        let mut iterations = 0usize;

        loop {
            let update = layer.step_lod(&[target], detail);
            if iterations == 0 {
                assert!(update.added.contains(&OctVec::root()));
            }
            if update.is_empty() {
                break;
            }
            iterations += 1;
            assert!(iterations < 32, "LOD 更新未在合理步数内收敛");
        }

        let positions = layer.chunk_positions();
        assert!(!positions.is_empty(), "八叉树应生成至少一个节点");

        let center = positions
            .iter()
            .copied()
            .max_by_key(|coord| coord.depth)
            .expect("至少应存在一个活跃节点");

        let neighbors = layer.chunk_neighbors(center, 1);
        assert!(!neighbors.is_empty(), "应能找到邻居节点");
        assert!(neighbors.contains(&center), "邻居结果应包含自身");

        drop(layer);
        cleanup(&entity);
    }

    #[test]
    fn world_renderables_collects_triangles_and_materials() {
        let root = Entity::new().expect("应能创建渲染根实体");
        let child = Entity::new().expect("应能创建渲染子实体");

        prepare_node(&root, "renderables_root");
        prepare_node(&child, "renderables_child");
        attach_node(child, root, "应能挂载子实体到根");

        let _ = root
            .register_component(Renderable::new())
            .expect("应能插入根 Renderable");
        let _ = root
            .register_component(Layer::new())
            .expect("应能插入根 Layer");

        let _ = child
            .register_component(Renderable::new())
            .expect("应能插入子 Renderable");
        let _ = child
            .register_component(Transform::new())
            .expect("应能插入子 Transform");
        if let Some(mut transform) = child.get_component_mut::<Transform>() {
            transform.set_position(Vector3::new(0.5, -0.25, 1.0));
        }

        let shape = Shape::from_triangles(vec![[
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ]]);
        let _ = child.register_component(shape).expect("应能插入 Shape");

        let resource: ResourceHandle =
            Arc::new(RwLock::new(Resource::from_memory(vec![1, 2, 3, 4])));
        let material = Material::new(
            resource.clone(),
            vec![[
                Vector2::new(0.0, 0.0),
                Vector2::new(1.0, 0.0),
                Vector2::new(0.0, 1.0),
            ]],
        );
        let _ = child
            .register_component(material)
            .expect("应能插入 Material");

        let bundles = Layer::world_renderables(root).expect("应能收集世界渲染数据");
        assert_eq!(bundles.len(), 1);
        let bundle = &bundles[0];
        assert_eq!(bundle.entity(), child);
        assert_eq!(bundle.triangles().len(), 1);
        let transformed = bundle.triangles()[0];
        assert!(
            (transformed[0].x - 0.5).abs() < 1e-6
                && (transformed[0].y + 0.25).abs() < 1e-6
                && (transformed[0].z - 1.0).abs() < 1e-6,
            "世界空间顶点应包含 Transform 偏移"
        );
        let descriptor = bundle.material().expect("应收集材质信息");
        assert_eq!(descriptor.regions().len(), 1);
        assert!(
            Arc::ptr_eq(descriptor.resource(), &resource),
            "材质资源句柄应保持引用一致"
        );

        let triangles = Layer::world_triangles(root).expect("应能扁平化三角面");
        assert_eq!(triangles.len(), 1);
        assert_eq!(triangles[0].entity(), child);
        let vertex = triangles[0].vertices()[0];
        assert!((vertex.x - 0.5).abs() < 1e-6 && (vertex.z - 1.0).abs() < 1e-6);

        let collection = Layer::collect_renderables(root).expect("应能构建渲染集合");
        assert_eq!(collection.bundles().len(), 1);
        assert!(collection.get(child).is_some());
        assert!(collection.material(child).is_some());

        let _ = child.unregister_component::<Material>();
        let _ = child.unregister_component::<Shape>();
        let _ = child.unregister_component::<Transform>();
        let _ = child.unregister_component::<Renderable>();
        cleanup(&child);
        cleanup(&root);
    }
}
