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

@group(1) @binding(0)
var scene_texture : texture_2d<f32>;

@group(1) @binding(1)
var scene_sampler : sampler;

struct FragmentInput {
    @location(0) world_position : vec3<f32>,
    @location(1) world_normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
    @location(3) uv_enabled : f32,
};

const EPSILON : f32 = 1.0e-5;
const AMBIENT_LIGHT : f32 = 0.1;

fn normalize_or_default(value : vec3<f32>) -> vec3<f32> {
    let len_sq = dot(value, value);
    if len_sq <= EPSILON {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return value * inverseSqrt(len_sq);
}

fn evaluate_point_light(light : PointLightUniform, position : vec3<f32>, normal : vec3<f32>) -> vec3<f32> {
    let radius = max(light.position_radius.w, EPSILON);
    let delta = light.position_radius.xyz - position;
    let distance = length(delta);
    if distance >= radius {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let direction = delta / max(distance, EPSILON);
    let normalized_distance = clamp(1.0 - distance / radius, 0.0, 1.0);
    let falloff = normalized_distance * normalized_distance * (3.0 - 2.0 * normalized_distance);
    let diffuse = max(dot(normal, direction), 0.0);
    return light.color.xyz * diffuse * falloff;
}

fn evaluate_parallel_light(light : ParallelLightUniform, normal : vec3<f32>) -> vec3<f32> {
    let direction = normalize_or_default(light.direction_pad.xyz);
    let diffuse = max(dot(normal, direction), 0.0);
    return light.color.xyz * diffuse;
}

@fragment
fn fs_main(input : FragmentInput) -> @location(0) vec4<f32> {
    let normalized_normal = normalize_or_default(input.world_normal);
    let use_texture = input.uv_enabled > 0.5;

    var base_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    if use_texture {
        base_color = textureSample(scene_texture, scene_sampler, input.uv);
    }

    var lighting = vec3<f32>(AMBIENT_LIGHT, AMBIENT_LIGHT, AMBIENT_LIGHT);

    let point_count = min(u32(scene.counts.x), MAX_SCENE3D_POINT_LIGHTS);
    for (var index : u32 = 0u; index < point_count; index = index + 1u) {
        lighting = lighting + evaluate_point_light(scene.point_lights[index], input.world_position, normalized_normal);
    }

    let parallel_count = min(u32(scene.counts.y), MAX_SCENE3D_PARALLEL_LIGHTS);
    for (var index : u32 = 0u; index < parallel_count; index = index + 1u) {
        lighting = lighting + evaluate_parallel_light(scene.parallel_lights[index], normalized_normal);
    }

    let final_rgb = clamp(base_color.rgb * lighting, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
    return vec4<f32>(final_rgb, base_color.a);
}
