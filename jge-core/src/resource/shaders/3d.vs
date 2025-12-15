const MAX_SCENE3D_POINT_LIGHTS : u32 = 8u;
const MAX_SCENE3D_PARALLEL_LIGHTS : u32 = 4u;

struct PointLightUniform {
    position_radius : vec4<f32>,
    color : vec4<f32>,
};

struct ParallelLightUniform {
    direction_pad : vec4<f32>,
    color : vec4<f32>,
};

struct Scene3DUniforms {
    view_proj : mat4x4<f32>,
    counts : vec4<f32>,
    point_lights : array<PointLightUniform, MAX_SCENE3D_POINT_LIGHTS>,
    parallel_lights : array<ParallelLightUniform, MAX_SCENE3D_PARALLEL_LIGHTS>,
};

@group(0) @binding(0)
var<uniform> scene : Scene3DUniforms;

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
    @location(3) uv_enabled : f32,
};

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0) world_position : vec3<f32>,
    @location(1) world_normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
    @location(3) uv_enabled : f32,
};

fn normalize_or_default(value : vec3<f32>) -> vec3<f32> {
    let len_sq = dot(value, value);
    if len_sq <= 1.0e-10 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return value * inverseSqrt(len_sq);
}

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;
    let world_position = input.position;
    output.clip_position = scene.view_proj * vec4<f32>(world_position, 1.0);
    output.world_position = world_position;
    output.world_normal = normalize_or_default(input.normal);
    output.uv = input.uv;
    output.uv_enabled = input.uv_enabled;
    return output;
}
