@group(0) @binding(0)
var bg_texture : texture_2d<f32>;

@group(0) @binding(1)
var bg_sampler : sampler;

struct BackgroundUniform {
    color : vec4<f32>,
    params : vec4<u32>,
    camera_pos : vec4<f32>,
    camera_forward : vec4<f32>,
};

@group(0) @binding(2)
var<uniform> bg : BackgroundUniform;

struct FragmentInput {
    @location(0) uv : vec2<f32>,
};

@fragment
fn fs_main(input : FragmentInput) -> @location(0) vec4<f32> {
    if bg.params.x != 0u {
        let tex = textureSample(bg_texture, bg_sampler, input.uv);
        return tex * bg.color;
    }
    return bg.color;
}
