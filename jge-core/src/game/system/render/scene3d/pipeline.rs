use std::{borrow::Cow, sync::Arc};

use anyhow::anyhow;
use wgpu;

use crate::{
    game::component::layer::{LayerShader, ShaderLanguage},
    resource::ResourceHandle,
};

use super::super::resource_io;
use super::SCENE3D_DEPTH_FORMAT;

#[derive(Default)]
pub(in crate::game::system::render) struct Scene3DPipelineCache {
    pub(in crate::game::system::render) pipeline: Option<Scene3DPipeline>,
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
    pub(in crate::game::system::render) pipeline: wgpu::RenderPipeline,
    pub(in crate::game::system::render) uniform_layout: wgpu::BindGroupLayout,
    pub(in crate::game::system::render) material_layout: wgpu::BindGroupLayout,
    pub(in crate::game::system::render) clip_layout: wgpu::BindGroupLayout,
    pub(in crate::game::system::render) vertex_shader: Scene3DShaderKey,
    pub(in crate::game::system::render) fragment_shader: Scene3DShaderKey,
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

        let clip_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Scene3D Clip Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<[f32; 8]>() as u64),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Scene3D Pipeline Layout"),
            bind_group_layouts: &[&uniform_layout, &material_layout, &clip_layout],
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
            clip_layout,
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

    pub(in crate::game::system::render) fn clip_layout(&self) -> &wgpu::BindGroupLayout {
        &self.clip_layout
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
        resource_io::load_utf8_string(handle, "scene3d shader")
    }
}

#[derive(Clone)]
pub(in crate::game::system::render) struct Scene3DShaderKey {
    pub(in crate::game::system::render) language: ShaderLanguage,
    pub(in crate::game::system::render) resource: ResourceHandle,
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
