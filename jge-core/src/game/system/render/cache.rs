use super::scene2d::{Scene2DDepthCache, Scene2DMaterialCache, Scene2DPipelineCache};
use super::scene3d::{Scene3DDepthCache, Scene3DMaterialCache, Scene3DPipelineCache};
use super::{background::BackgroundPipelineCache, background::BackgroundTextureCache};

pub(in crate::game::system::render) struct LayerRendererCache {
    pub(in crate::game::system::render) scene2d: Scene2DPipelineCache,
    pub(in crate::game::system::render) scene2d_materials: Scene2DMaterialCache,
    pub(in crate::game::system::render) scene2d_depth: Scene2DDepthCache,
    pub(in crate::game::system::render) scene3d: Scene3DPipelineCache,
    pub(in crate::game::system::render) scene3d_materials: Scene3DMaterialCache,
    pub(in crate::game::system::render) scene3d_depth: Scene3DDepthCache,
    pub(in crate::game::system::render) background: BackgroundPipelineCache,
    pub(in crate::game::system::render) background_textures: BackgroundTextureCache,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::game::system::render) struct LayerViewportPixels {
    pub(in crate::game::system::render) x: u32,
    pub(in crate::game::system::render) y: u32,
    pub(in crate::game::system::render) width: u32,
    pub(in crate::game::system::render) height: u32,
}

impl Default for LayerRendererCache {
    fn default() -> Self {
        Self {
            scene2d: Scene2DPipelineCache::default(),
            scene2d_materials: Scene2DMaterialCache::default(),
            scene2d_depth: Scene2DDepthCache::default(),
            scene3d: Scene3DPipelineCache::default(),
            scene3d_materials: Scene3DMaterialCache::default(),
            scene3d_depth: Scene3DDepthCache::default(),
            background: BackgroundPipelineCache::default(),
            background_textures: BackgroundTextureCache::default(),
        }
    }
}

pub(in crate::game::system::render) struct LayerRenderContext<'a, 'cache> {
    pub(in crate::game::system::render) device: &'a wgpu::Device,
    pub(in crate::game::system::render) queue: &'a wgpu::Queue,
    pub(in crate::game::system::render) encoder: &'a mut wgpu::CommandEncoder,
    pub(in crate::game::system::render) target_view: &'a wgpu::TextureView,
    pub(in crate::game::system::render) surface_format: wgpu::TextureFormat,
    pub(in crate::game::system::render) framebuffer_size: (u32, u32),
    pub(in crate::game::system::render) viewport: Option<LayerViewportPixels>,
    pub(in crate::game::system::render) load_op: wgpu::LoadOp<wgpu::Color>,
    pub(in crate::game::system::render) caches: &'cache mut LayerRendererCache,
}
