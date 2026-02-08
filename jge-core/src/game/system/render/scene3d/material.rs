use std::{collections::HashMap, sync::Arc};

use anyhow::{Context, anyhow, ensure};
use image::GenericImageView;
use wgpu;

use crate::resource::ResourceHandle;

use super::super::{resource_io, util};

#[derive(Clone)]
pub(in crate::game::system::render) struct Scene3DMaterialInstance {
    pub(in crate::game::system::render) _texture: wgpu::Texture,
    pub(in crate::game::system::render) _view: wgpu::TextureView,
    pub(in crate::game::system::render) bind_group: wgpu::BindGroup,
}

#[derive(Default)]
pub(in crate::game::system::render) struct Scene3DMaterialCache {
    sampler: Option<wgpu::Sampler>,
    default: Option<Scene3DMaterialInstance>,
    materials: HashMap<usize, Scene3DMaterialInstance>,
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
        let bytes = resource_io::load_bytes_arc(handle, "scene3d material texture")?;
        let image = image::load_from_memory(bytes.as_ref())
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
