use std::collections::HashSet;

use tracing::warn;

use crate::game::{
    component::{
        background::Background,
        camera::Camera,
        layer::{Layer, LayerShader, LayerTraversalError, RenderPipelineStage},
        node::Node,
        scene2d::Scene2D,
        scene2d::Scene2DFaceGroup,
        scene3d::Scene3D,
        transform::Transform,
    },
    entity::Entity,
};
use crate::resource::ResourceHandle;

use super::{cache::LayerViewportPixels, util};

use nalgebra::{Matrix4, Perspective3, Point3, Rotation3, Vector3};

use crate::game::component::{
    layer::LayerRenderableCollection,
    light::{Light, ParallelLight, PointLight},
};

#[derive(Debug, Clone)]
pub(crate) struct RenderSnapshot {
    layers: Vec<LayerSnapshot>,
}

impl RenderSnapshot {
    pub(crate) fn empty(framebuffer_size: (u32, u32)) -> Self {
        let _ = framebuffer_size;
        Self { layers: Vec::new() }
    }

    pub(crate) fn layers(&self) -> &[LayerSnapshot] {
        &self.layers
    }

    pub(crate) async fn build(root: Entity, framebuffer_size: (u32, u32)) -> Self {
        let layer_roots = collect_layer_roots(root).await;
        let mut layers = Vec::with_capacity(layer_roots.len());
        for layer_entity in layer_roots {
            if let Some(snapshot) = build_layer_snapshot(layer_entity, framebuffer_size).await {
                layers.push(snapshot);
            }
        }

        Self { layers }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LayerSnapshot {
    entity: Entity,
    viewport: Option<LayerViewportPixels>,
    background: Option<BackgroundSnapshot>,
    scene_kind: Option<LayerSceneKind>,
    scene2d: Option<Scene2DSnapshot>,
    scene3d: Option<Scene3DSnapshot>,
}

impl LayerSnapshot {
    pub(crate) fn entity(&self) -> Entity {
        self.entity
    }

    pub(in crate::game::system::render) fn viewport(&self) -> Option<LayerViewportPixels> {
        self.viewport
    }

    pub(in crate::game::system::render) fn background(&self) -> Option<&BackgroundSnapshot> {
        self.background.as_ref()
    }

    pub(in crate::game::system::render) fn scene_kind(&self) -> Option<LayerSceneKind> {
        self.scene_kind
    }

    pub(in crate::game::system::render) fn scene3d(&self) -> Option<&Scene3DSnapshot> {
        self.scene3d.as_ref()
    }

    pub(in crate::game::system::render) fn scene2d(&self) -> Option<&Scene2DSnapshot> {
        self.scene2d.as_ref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::game::system::render) enum LayerSceneKind {
    Scene2D,
    Scene3D,
}

#[derive(Debug, Clone)]
pub(in crate::game::system::render) struct Scene2DPointLightSnapshot {
    pub(in crate::game::system::render) center: [f32; 2],
    pub(in crate::game::system::render) radius: f32,
    pub(in crate::game::system::render) lightness: f32,
}

#[derive(Debug, Clone)]
pub(in crate::game::system::render) struct Scene2DSnapshot {
    pub(in crate::game::system::render) offset: [f32; 2],
    pub(in crate::game::system::render) pixels_per_unit: f32,
    pub(in crate::game::system::render) vertex_shader: ResourceHandle,
    pub(in crate::game::system::render) fragment_shader: ResourceHandle,
    pub(in crate::game::system::render) renderables: LayerRenderableCollection,
    pub(in crate::game::system::render) face_groups: Vec<Scene2DFaceGroup>,
    pub(in crate::game::system::render) parallel_light_brightness: f32,
    pub(in crate::game::system::render) point_lights: Vec<Scene2DPointLightSnapshot>,
}

#[derive(Debug, Clone)]
pub(in crate::game::system::render) struct Scene3DPointLightSnapshot {
    pub(in crate::game::system::render) position: [f32; 3],
    pub(in crate::game::system::render) radius: f32,
    pub(in crate::game::system::render) intensity: f32,
}

#[derive(Debug, Clone)]
pub(in crate::game::system::render) struct Scene3DParallelLightSnapshot {
    pub(in crate::game::system::render) direction: [f32; 3],
    pub(in crate::game::system::render) intensity: f32,
}

#[derive(Debug, Clone)]
pub(in crate::game::system::render) struct Scene3DSnapshot {
    pub(in crate::game::system::render) camera_entity: Entity,
    pub(in crate::game::system::render) vertex_shader: LayerShader,
    pub(in crate::game::system::render) fragment_shader: LayerShader,
    pub(in crate::game::system::render) view_proj: [f32; 16],
    pub(in crate::game::system::render) visible: LayerRenderableCollection,
    pub(in crate::game::system::render) point_lights: Vec<Scene3DPointLightSnapshot>,
    pub(in crate::game::system::render) parallel_lights: Vec<Scene3DParallelLightSnapshot>,
}

#[derive(Debug, Clone)]
pub(in crate::game::system::render) struct BackgroundSnapshot {
    pub(in crate::game::system::render) color: [f32; 4],
    pub(in crate::game::system::render) image: Option<ResourceHandle>,
    pub(in crate::game::system::render) fragment_shader_override: Option<ResourceHandle>,
    pub(in crate::game::system::render) camera_pos: [f32; 3],
    pub(in crate::game::system::render) camera_forward: [f32; 3],
}

async fn collect_layer_roots(root: Entity) -> Vec<Entity> {
    let mut result = Vec::new();
    let mut stack = vec![root];
    let mut visited = HashSet::new();

    while let Some(entity) = stack.pop() {
        if !visited.insert(entity) {
            continue;
        }

        let node_guard = match entity.get_component::<Node>().await {
            Some(node) => node,
            None => {
                warn!(
                    target: "jge-core",
                    entity_id = %entity.id(),
                    "entity missing Node component, skip layer traversal"
                );
                continue;
            }
        };

        let children: Vec<Entity> = node_guard.children().to_vec();
        drop(node_guard);

        if entity.get_component::<Layer>().await.is_some() {
            result.push(entity);
            continue;
        }

        for child in children.into_iter().rev() {
            stack.push(child);
        }
    }

    result
}

async fn build_layer_snapshot(
    entity: Entity,
    framebuffer_size: (u32, u32),
) -> Option<LayerSnapshot> {
    let layer_guard = entity.get_component::<Layer>().await?;
    let viewport = layer_guard
        .viewport()
        .and_then(|v| util::viewport_pixels_from_normalized(framebuffer_size, v));
    drop(layer_guard);

    let viewport_framebuffer_size = viewport
        .map(|v| (v.width, v.height))
        .unwrap_or(framebuffer_size);

    let background = build_background_snapshot(entity, viewport_framebuffer_size).await;

    let (scene_kind, scene2d, scene3d) = if entity.get_component::<Scene2D>().await.is_some() {
        let scene2d = build_scene2d_snapshot(entity, framebuffer_size).await;
        (Some(LayerSceneKind::Scene2D), scene2d, None)
    } else if entity.get_component::<Scene3D>().await.is_some() {
        let scene3d = build_scene3d_snapshot(entity, viewport_framebuffer_size).await;
        (Some(LayerSceneKind::Scene3D), None, scene3d)
    } else {
        (None, None, None)
    };

    Some(LayerSnapshot {
        entity,
        viewport,
        background,
        scene_kind,
        scene2d,
        scene3d,
    })
}

async fn build_scene2d_snapshot(
    layer_entity: Entity,
    framebuffer_size: (u32, u32),
) -> Option<Scene2DSnapshot> {
    let mut scene_guard = layer_entity.get_component_mut::<Scene2D>().await?;
    scene_guard.set_framebuffer_size(framebuffer_size);
    let scene_offset = scene_guard.offset();
    let pixels_per_unit = scene_guard.pixels_per_unit();

    let layer_guard = layer_entity.get_component::<Layer>().await?;

    let vertex_shader = layer_guard
        .shader(RenderPipelineStage::Vertex)
        .map(|shader| shader.resource_handle())?;
    let fragment_shader = layer_guard
        .shader(RenderPipelineStage::Fragment)
        .map(|shader| shader.resource_handle())?;

    let renderables = match Layer::collect_renderables(layer_entity).await {
        Ok(collection) => collection,
        Err(error) => {
            warn!(
                target: "jge-core",
                layer_id = %layer_entity.id(),
                error = %error,
                "Scene2D renderable collection failed"
            );
            return None;
        }
    };

    let face_groups =
        match scene_guard.visible_faces_with_renderables(&layer_guard, renderables.bundles()) {
            Ok(faces) => faces,
            Err(error) => {
                warn!(
                    target: "jge-core",
                    layer_id = %layer_entity.id(),
                    error = %error,
                    "Scene2D visibility query failed"
                );
                Vec::new()
            }
        };

    let point_light_entities = match Layer::point_light_entities(layer_entity).await {
        Ok(lights) => lights,
        Err(error) => {
            warn!(
                target: "jge-core",
                layer_id = %layer_entity.id(),
                error = %error,
                "Scene2D lighting query failed"
            );
            Vec::new()
        }
    };

    let parallel_light_brightness = match Layer::parallel_light_entities(layer_entity).await {
        Ok(lights) => {
            let mut total = 0.0f32;
            for light_entity in lights {
                let light = light_entity.get_component::<Light>().await;
                let Some(light) = light else {
                    continue;
                };
                let value = light.lightness();
                if value > 0.0 {
                    total += value;
                }
            }
            total
        }
        Err(error) => {
            warn!(
                target: "jge-core",
                layer_id = %layer_entity.id(),
                error = %error,
                "Scene2D directional lighting query failed"
            );
            0.0
        }
    };

    let mut point_lights: Vec<Scene2DPointLightSnapshot> = Vec::new();
    for light_entity in point_light_entities {
        let light = light_entity.get_component::<Light>().await;
        let point = light_entity.get_component::<PointLight>().await;
        let transform = light_entity.get_component::<Transform>().await;

        let (Some(light), Some(point), Some(transform)) = (light, point, transform) else {
            continue;
        };

        let radius = point.distance();
        let lightness = light.lightness();
        let position = transform.position();

        if radius <= f32::EPSILON || lightness <= 0.0 {
            continue;
        }

        point_lights.push(Scene2DPointLightSnapshot {
            center: [position.x, position.y],
            radius,
            lightness,
        });
    }

    Some(Scene2DSnapshot {
        offset: [scene_offset.x, scene_offset.y],
        pixels_per_unit,
        vertex_shader,
        fragment_shader,
        renderables,
        face_groups,
        parallel_light_brightness,
        point_lights,
    })
}

async fn build_scene3d_snapshot(
    layer_entity: Entity,
    viewport_framebuffer_size: (u32, u32),
) -> Option<Scene3DSnapshot> {
    let scene_guard = layer_entity.get_component::<Scene3D>().await?;
    let scene_vertical = scene_guard
        .vertical_fov_for_height(viewport_framebuffer_size.1)
        .ok()?;
    let scene_near = scene_guard.near_plane();
    let scene_distance = scene_guard.view_distance();
    let preferred_camera = scene_guard.attached_camera();

    let layer_guard = layer_entity.get_component::<Layer>().await?;
    let vertex_shader = layer_guard.shader(RenderPipelineStage::Vertex)?.clone();
    let fragment_shader = layer_guard.shader(RenderPipelineStage::Fragment)?.clone();
    drop(layer_guard);

    let camera_entity = select_scene3d_camera_async(layer_entity, preferred_camera)
        .await
        .ok()??;

    let camera_guard = camera_entity.get_component::<Camera>().await?;
    let camera_vertical = camera_guard
        .vertical_fov_for_height(viewport_framebuffer_size.1)
        .ok()?;
    let camera_near = camera_guard.near_plane();
    let camera_far = camera_guard.far_plane();
    drop(camera_guard);

    let near_plane = camera_near.max(scene_near);
    let far_plane = camera_far.min(scene_distance);
    match near_plane.partial_cmp(&far_plane) {
        Some(std::cmp::Ordering::Less) => {}
        _ => return None,
    }

    let (width, height) = viewport_framebuffer_size;
    if width == 0 || height == 0 {
        return None;
    }

    let vertical_fov = scene_vertical.min(camera_vertical);
    let aspect_ratio = width as f32 / height as f32;

    let visible = scene_guard
        .visible_renderables(camera_entity, viewport_framebuffer_size)
        .await
        .ok()?;
    drop(scene_guard);

    let transform_guard = camera_entity.get_component::<Transform>().await?;
    let camera_position = transform_guard.position();
    let basis = Camera::orientation_basis(&transform_guard).normalize();
    drop(transform_guard);

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
    let view_proj_matrix = util::opengl_to_wgpu_matrix() * projection.to_homogeneous() * view;
    let view_proj = {
        let mut out = [0.0f32; 16];
        out.copy_from_slice(view_proj_matrix.as_slice());
        out
    };

    let point_light_entities = Layer::point_light_entities(layer_entity).await.ok()?;
    let parallel_light_entities = Layer::parallel_light_entities(layer_entity).await.ok()?;

    let mut point_lights: Vec<Scene3DPointLightSnapshot> = Vec::new();
    for light_entity in point_light_entities {
        let light = light_entity.get_component::<Light>().await;
        let point = light_entity.get_component::<PointLight>().await;
        let transform = light_entity.get_component::<Transform>().await;
        let (Some(light), Some(point), Some(transform)) = (light, point, transform) else {
            continue;
        };

        let radius = point.distance();
        let intensity = light.lightness();
        let position = transform.position();

        if radius <= f32::EPSILON || intensity <= 0.0 {
            continue;
        }

        point_lights.push(Scene3DPointLightSnapshot {
            position: [position.x, position.y, position.z],
            radius,
            intensity,
        });
    }

    let mut parallel_lights: Vec<Scene3DParallelLightSnapshot> = Vec::new();
    for light_entity in parallel_light_entities {
        let light = light_entity.get_component::<Light>().await;
        let parallel = light_entity.get_component::<ParallelLight>().await;
        let transform = light_entity.get_component::<Transform>().await;
        let (Some(light), Some(_parallel), Some(transform)) = (light, parallel, transform) else {
            continue;
        };

        let rotation = transform.rotation();
        let intensity = light.lightness();
        if intensity <= 0.0 {
            continue;
        }

        let rotation_matrix = Rotation3::from_euler_angles(rotation.x, rotation.y, rotation.z);
        let forward = rotation_matrix * Vector3::new(0.0, -1.0, 0.0);
        let incoming = -forward;
        let Some(direction) = incoming.try_normalize(1.0e-6) else {
            continue;
        };

        parallel_lights.push(Scene3DParallelLightSnapshot {
            direction: [direction.x, direction.y, direction.z],
            intensity,
        });
    }

    Some(Scene3DSnapshot {
        camera_entity,
        vertex_shader,
        fragment_shader,
        view_proj,
        visible,
        point_lights,
        parallel_lights,
    })
}

async fn build_background_snapshot(
    layer_entity: Entity,
    viewport_framebuffer_size: (u32, u32),
) -> Option<BackgroundSnapshot> {
    let bg_entity = find_first_background(layer_entity).await?;
    let bg_guard = bg_entity.get_component::<Background>().await?;

    let color = bg_guard.color();
    let image = bg_guard.image();
    let fragment_shader_override = bg_guard
        .fragment_shader()
        .map(|shader| shader.resource_handle());
    drop(bg_guard);

    let (camera_pos, camera_forward) =
        resolve_layer_camera_pose(layer_entity, viewport_framebuffer_size)
            .await
            .unwrap_or(([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]));

    Some(BackgroundSnapshot {
        color,
        image,
        fragment_shader_override,
        camera_pos,
        camera_forward,
    })
}

async fn resolve_layer_camera_pose(
    layer_entity: Entity,
    viewport_framebuffer_size: (u32, u32),
) -> Option<([f32; 3], [f32; 3])> {
    let scene_guard = layer_entity.get_component::<Scene3D>().await?;
    let preferred = scene_guard.attached_camera();
    drop(scene_guard);

    let camera_entity = select_scene3d_camera_async(layer_entity, preferred)
        .await
        .ok()??;
    let camera_guard = camera_entity.get_component::<Camera>().await?;
    let _ = camera_guard
        .vertical_fov_for_height(viewport_framebuffer_size.1)
        .ok()?;
    drop(camera_guard);

    let transform_guard = camera_entity.get_component::<Transform>().await?;
    let pos = transform_guard.position();
    let basis = Camera::orientation_basis(&transform_guard).normalize();

    Some((
        [pos.x, pos.y, pos.z],
        [basis.forward.x, basis.forward.y, basis.forward.z],
    ))
}

async fn select_scene3d_camera_async(
    root: Entity,
    preferred: Option<Entity>,
) -> Result<Option<Entity>, LayerTraversalError> {
    if let Some(candidate) = preferred {
        if candidate.get_component::<Camera>().await.is_some()
            && candidate.get_component::<Transform>().await.is_some()
        {
            return Ok(Some(candidate));
        }
    }

    let ordered = Layer::renderable_entities(root).await?;
    for entity in ordered {
        if entity.get_component::<Camera>().await.is_some()
            && entity.get_component::<Transform>().await.is_some()
        {
            return Ok(Some(entity));
        }
    }

    Ok(None)
}

async fn find_first_background(root: Entity) -> Option<Entity> {
    let mut stack = vec![root];
    let mut visited = HashSet::new();

    while let Some(entity) = stack.pop() {
        if !visited.insert(entity) {
            continue;
        }

        if entity.get_component::<Background>().await.is_some() {
            return Some(entity);
        }

        let node_guard = entity.get_component::<Node>().await?;
        let children: Vec<Entity> = node_guard.children().to_vec();
        drop(node_guard);

        for child in children.into_iter().rev() {
            stack.push(child);
        }
    }

    None
}
