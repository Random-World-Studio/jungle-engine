@group(0) @binding(0)
var bg_texture : texture_2d<f32>;

@group(0) @binding(1)
var bg_sampler : sampler;

struct BackgroundUniform {
    // tint / base color
    color : vec4<f32>,

    // 0 = pure color, 1 = sample texture and multiply
    params : vec4<u32>,

    // 来自 Scene3D 摄像机：世界坐标位置与 forward（+Y 向上，默认 forward 为 -Z）
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
    var base = bg.color;
    if bg.params.x != 0u {
        base = textureSample(bg_texture, bg_sampler, input.uv) * bg.color;
    }

    // 地平线位置：
    // - 相机 y 越高 -> 地平线越低（黑色越少）
    // - 相机 forward.y 越大（越抬头）-> 地平线越低（黑色越少）
    // - 相机 forward.y 越小（越低头）-> 地平线越高（黑色越多）
    let height_scale = 0.12;
    let pitch_scale = 0.45;
    let horizon = clamp(0.5 - bg.camera_pos.y * height_scale - bg.camera_forward.y * pitch_scale, 0.0, 1.0);

    // 交界渐变宽度（UV 空间，0..1）
    let fade = 0.08;

    // 注意：不同 pipeline 的 UV 约定可能是“y 向下增大”。这里统一把 y 翻转成“向上增大”。
    let v = 1.0 - input.uv.y;

    // t=0 表示全黑（在地平线以下），t=1 表示全背景（在地平线以上）
    let t = smoothstep(horizon - fade, horizon + fade, v);
    return mix(vec4<f32>(0.0, 0.0, 0.0, 1.0), base, t);
}
