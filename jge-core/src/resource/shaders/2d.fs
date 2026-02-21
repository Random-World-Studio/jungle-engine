@group(0) @binding(0)
var scene_texture : texture_2d<f32>;

@group(0) @binding(1)
var scene_sampler : sampler;

struct ClipAabbUniform {
    // xyz = min, w = enabled (1.0 / 0.0)
    min_enabled : vec4<f32>,
    // xyz = max
    max_pad : vec4<f32>,
};

@group(1) @binding(0)
var<uniform> clip : ClipAabbUniform;

struct FragmentInput {
    @location(0) uv : vec2<f32>,
    @location(1) brightness : f32,
    @location(2) world_position : vec3<f32>,
};

const MAX_TOTAL_BRIGHTNESS : f32 = 6.0;

@fragment
fn fs_main(input : FragmentInput) -> @location(0) vec4<f32> {
    if clip.min_enabled.w > 0.5 {
        let p = input.world_position;
        if p.x < clip.min_enabled.x || p.y < clip.min_enabled.y || p.z < clip.min_enabled.z ||
            p.x > clip.max_pad.x || p.y > clip.max_pad.y || p.z > clip.max_pad.z {
            discard;
        }
    }

    let base_color = textureSample(scene_texture, scene_sampler, input.uv);
    let light = clamp(input.brightness, 0.0, MAX_TOTAL_BRIGHTNESS);
    let lit_rgb = base_color.rgb * light;
    return vec4<f32>(lit_rgb, base_color.a);
}
