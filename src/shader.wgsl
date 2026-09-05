struct CameraUniform {
    view_proj: mat4x4<f32>,
    show_borders: f32,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = model.color;
    out.normal = model.normal;
    out.uv = model.uv;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.4, 0.9, 0.3));
    let ndotl = max(dot(in.normal, light_dir), 0.0);

    // Discrete 3-tone lighting quantization (Cel-Shading)
    var lighting: f32 = 0.42;
    if (ndotl > 0.65) {
        lighting = 1.00;
    } else if (ndotl > 0.20) {
        lighting = 0.72;
    }

    var col = in.color * lighting;

    // Edge definition
    let edge = min(min(in.uv.x, 1.0 - in.uv.x), min(in.uv.y, 1.0 - in.uv.y));
    if (camera.show_borders > 0.5) {
        if (edge < 0.045) {
            let lum = dot(col, vec3<f32>(0.299, 0.587, 0.114));
            if (lum < 0.2) {
                col = col + vec3<f32>(0.35, 0.35, 0.40);
            } else {
                col = col * 0.25;
            }
        }
    } else {
        // Subtle pixel-edge creasing
        if (edge < 0.02) {
            col = col * 0.85;
        }
    }

    return vec4<f32>(col, 1.0);
}