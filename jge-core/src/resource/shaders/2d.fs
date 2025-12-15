@group(0) @binding(0)
var scene_texture : texture_2d<f32>;

@group(0) @binding(1)
var scene_sampler : sampler;

struct FragmentInput {
    @location(0) uv : vec2<f32>,
    @location(1) brightness : f32,
};

const MAX_TOTAL_BRIGHTNESS : f32 = 6.0;

@fragment
fn fs_main(input : FragmentInput) -> @location(0) vec4<f32> {
    let base_color = textureSample(scene_texture, scene_sampler, input.uv);
    let light = clamp(input.brightness, 0.0, MAX_TOTAL_BRIGHTNESS);
    let lit_rgb = base_color.rgb * light;
    return vec4<f32>(lit_rgb, base_color.a);
}
