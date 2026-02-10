mod background;
mod cache;
mod dirty;
mod profile;
mod resource_io;
mod scene2d;
mod scene3d;
mod snapshot;
mod util;

pub(crate) use snapshot::RenderSnapshot;

pub(crate) use dirty::{
    mark_render_snapshot_dirty_for_component, record_snapshot_rebuild, snapshot_rebuild_stats,
    take_render_snapshot_dirty,
};

use tracing::{debug, warn};
use wgpu;

use crate::game::{
    component::{layer::Layer, scene2d::Scene2D, scene3d::Scene3D},
    entity::Entity,
};

use cache::{LayerRenderContext, LayerRendererCache};
use snapshot::{LayerSceneKind, LayerSnapshot};
use tokio::runtime::Runtime;

pub struct RenderLayerParams<'a> {
    pub runtime: &'a Runtime,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub target_view: &'a wgpu::TextureView,
    pub surface_format: wgpu::TextureFormat,
    pub framebuffer_size: (u32, u32),
    pub load_op: wgpu::LoadOp<wgpu::Color>,
}

/// 渲染系统。
///
/// `RenderSystem` 负责在给定的渲染目标（surface texture view）上渲染某个 Layer 及其子树。
/// 它会根据 Layer 上挂载的场景类型（例如 `Scene2D` / `Scene3D`）选择对应渲染路径，并通过内部缓存复用管线与绑定资源。
#[derive(Default)]
pub struct RenderSystem {
    caches: LayerRendererCache,
}

impl RenderSystem {
    /// 创建渲染系统。
    pub fn new() -> Self {
        Self::default()
    }

    /// 渲染一个 Layer 实体。
    ///
    /// - 若 `layer_entity` 未挂载 [`Layer`]，将跳过并输出警告日志。
    /// - `load_op` 控制该 Layer 的渲染是清屏还是在已有内容上叠加。
    /// - 渲染路径选择：优先使用 `Scene2D`；若不存在则尝试 `Scene3D`；两者都不存在则跳过。
    pub fn render_layer(&mut self, layer_entity: Entity, params: RenderLayerParams<'_>) {
        let RenderLayerParams {
            runtime,
            device,
            queue,
            encoder,
            target_view,
            surface_format,
            framebuffer_size,
            load_op,
        } = params;

        let layer = runtime.block_on(layer_entity.get_component::<Layer>());
        let viewport = layer
            .as_ref()
            .and_then(|layer| layer.viewport())
            .and_then(|viewport| util::viewport_pixels_from_normalized(framebuffer_size, viewport));

        let mut context = LayerRenderContext {
            runtime,
            device,
            queue,
            encoder,
            target_view,
            surface_format,
            framebuffer_size,
            viewport,
            load_op,
            caches: &mut self.caches,
        };

        if layer.is_some() {
            Self::render_layer_entity(layer_entity, &mut context);
        } else {
            warn!(
                target: "jge-core",
                layer_id = %layer_entity.id(),
                "skip rendering for entity missing Layer component"
            );
        }
    }

    pub(crate) fn render_layer_snapshot(
        &mut self,
        layer_snapshot: &LayerSnapshot,
        params: RenderLayerParams<'_>,
    ) {
        let RenderLayerParams {
            runtime,
            device,
            queue,
            encoder,
            target_view,
            surface_format,
            framebuffer_size,
            load_op,
        } = params;

        let viewport = layer_snapshot.viewport();

        let mut context = LayerRenderContext {
            runtime,
            device,
            queue,
            encoder,
            target_view,
            surface_format,
            framebuffer_size,
            viewport,
            load_op,
            caches: &mut self.caches,
        };

        Self::render_layer_snapshot_entity(layer_snapshot, &mut context);
    }

    /// 标记一帧渲染开始。
    pub fn begin_frame(&mut self) {
        self.caches.profiler.begin_frame();
    }

    pub(crate) fn record_frame_total_cpu(&mut self, duration: std::time::Duration) {
        self.caches.profiler.record_frame_total(duration);
    }

    pub(crate) fn record_frame_phases_cpu(
        &mut self,
        acquire: std::time::Duration,
        encode: std::time::Duration,
        submit: std::time::Duration,
        present: std::time::Duration,
    ) {
        self.caches
            .profiler
            .record_frame_phases(acquire, encode, submit, present);
    }

    /// 标记一帧渲染结束。
    pub fn end_frame(&mut self) {
        self.caches.profiler.end_frame();
    }

    fn render_layer_entity(entity: Entity, context: &mut LayerRenderContext<'_, '_>) {
        // 规则：每次渲染某个 Layer 时，在该 Layer 其它内容渲染之前，
        // 先在“以 Layer 所在节点为根的节点树”中按先序遍历找到第一个 Background 并渲染。
        if background::render_background_if_present(entity, context) {
            // 背景 pass 已使用当前 load_op（通常为 Clear），后续场景渲染应叠加在其上。
            context.load_op = wgpu::LoadOp::Load;
        }

        if context
            .runtime
            .block_on(entity.get_component::<Scene2D>())
            .is_some()
        {
            Self::render_scene2d(entity, context);
        } else if context
            .runtime
            .block_on(entity.get_component::<Scene3D>())
            .is_some()
        {
            Self::render_scene3d(entity, context);
        } else {
            debug!(
                target: "jge-core",
                layer_id = %entity.id(),
                "skip layer without registered renderer"
            );
        }
    }

    fn render_layer_snapshot_entity(
        layer_snapshot: &LayerSnapshot,
        context: &mut LayerRenderContext<'_, '_>,
    ) {
        let entity = layer_snapshot.entity();

        if let Some(background_snapshot) = layer_snapshot.background()
            && background::render_background_from_snapshot(entity, background_snapshot, context)
        {
            context.load_op = wgpu::LoadOp::Load;
        }

        match layer_snapshot.scene_kind() {
            Some(LayerSceneKind::Scene2D) => {
                if let Some(scene2d_snapshot) = layer_snapshot.scene2d() {
                    Self::render_scene2d_from_snapshot(entity, scene2d_snapshot, context);
                } else {
                    warn!(
                        target: "jge-core",
                        layer_id = %entity.id(),
                        "skip Scene2D rendering: missing Scene2D snapshot"
                    );
                }
            }
            Some(LayerSceneKind::Scene3D) => {
                if let Some(scene3d_snapshot) = layer_snapshot.scene3d() {
                    Self::render_scene3d_from_snapshot(entity, scene3d_snapshot, context);
                } else {
                    warn!(
                        target: "jge-core",
                        layer_id = %entity.id(),
                        "skip Scene3D rendering: missing Scene3D snapshot"
                    );
                }
            }
            None => {
                debug!(
                    target: "jge-core",
                    layer_id = %entity.id(),
                    "skip layer without scene kind in snapshot"
                );
            }
        }
    }
}
