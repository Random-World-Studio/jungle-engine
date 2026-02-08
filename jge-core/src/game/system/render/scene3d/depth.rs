use wgpu;

use super::SCENE3D_DEPTH_FORMAT;

pub(in crate::game::system::render) struct Scene3DDepthAttachment {
    pub(in crate::game::system::render) _texture: wgpu::Texture,
    pub(in crate::game::system::render) view: wgpu::TextureView,
    pub(in crate::game::system::render) size: (u32, u32),
}

#[derive(Default)]
pub(in crate::game::system::render) struct Scene3DDepthCache {
    pub(in crate::game::system::render) attachment: Option<Scene3DDepthAttachment>,
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
