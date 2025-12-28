mod cache;
mod scene2d;
mod scene3d;
mod util;

use tracing::{debug, warn};
use wgpu;

use crate::game::{
    component::{layer::Layer, scene2d::Scene2D, scene3d::Scene3D},
    entity::Entity,
};

use cache::{LayerRenderContext, LayerRendererCache};

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
    pub fn render_layer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        surface_format: wgpu::TextureFormat,
        framebuffer_size: (u32, u32),
        layer_entity: Entity,
        load_op: wgpu::LoadOp<wgpu::Color>,
    ) {
        let mut context = LayerRenderContext {
            device,
            queue,
            encoder,
            target_view,
            surface_format,
            framebuffer_size,
            load_op,
            caches: &mut self.caches,
        };

        if layer_entity.get_component::<Layer>().is_some() {
            Self::render_layer_entity(layer_entity, &mut context);
        } else {
            warn!(
                target: "jge-core",
                layer_id = layer_entity.id(),
                "skip rendering for entity missing Layer component"
            );
        }
    }

    fn render_layer_entity(entity: Entity, context: &mut LayerRenderContext<'_, '_>) {
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
}
