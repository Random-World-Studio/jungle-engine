struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) uv : vec2<f32>,
    @location(2) brightness : f32,
};

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0) uv : vec2<f32>,
    @location(1) brightness : f32,
};

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;
    output.clip_position = vec4<f32>(input.position, 1.0);
    output.uv = input.uv;
    output.brightness = input.brightness;
    return output;
}
