struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    show_borders: f32,
    world_min: vec3<f32>,
    world_size: f32,
    is_ortho: f32,
    ortho_size: f32,
    screen_size: vec2<f32>,
};

struct SvoNode {
    child_mask: u32,
    child_pointer: u32,
    material_id: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<storage, read> svo_nodes: array<SvoNode>;
@group(0) @binding(2) var<storage, read> palette: array<vec4<f32>>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(in_vertex_index << 1u) & 2) * 2.0 - 1.0;
    let y = f32(i32(in_vertex_index & 2u)) * 2.0 - 1.0;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

struct StackNode {
    idx: u32,
    b_min: vec3<f32>,
    size: f32,
    t_enter: f32,
};

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);

    var ray_orig: vec3<f32>;
    var ray_dir: vec3<f32>;

    if (camera.is_ortho > 0.5) {
        let p_near = camera.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
        let p_far  = camera.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
        ray_orig = p_near.xyz / p_near.w;
        ray_dir = normalize((p_far.xyz / p_far.w) - ray_orig);
    } else {
        let p_far = camera.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
        let p_world = p_far.xyz / p_far.w;
        ray_orig = camera.camera_pos;
        ray_dir = normalize(p_world - ray_orig);
    }

    let inv_dir = vec3<f32>(
        select(1.0 / ray_dir.x, 1e7 * sign(ray_dir.x), abs(ray_dir.x) < 1e-6),
        select(1.0 / ray_dir.y, 1e7 * sign(ray_dir.y), abs(ray_dir.y) < 1e-6),
        select(1.0 / ray_dir.z, 1e7 * sign(ray_dir.z), abs(ray_dir.z) < 1e-6),
    );

    let root_min = camera.world_min;
    let root_max = camera.world_min + vec3<f32>(camera.world_size);

    let rt0 = (root_min - ray_orig) * inv_dir;
    let rt1 = (root_max - ray_orig) * inv_dir;
    let rtmin3 = min(rt0, rt1);
    let rtmax3 = max(rt0, rt1);
    let r_enter = max(max(rtmin3.x, rtmin3.y), rtmin3.z);
    let r_exit = min(min(rtmax3.x, rtmax3.y), rtmax3.z);

    var hit_mat = 0u;
    var hit_normal = vec3<f32>(0.0);
    var hit_t = 1e9;

    let root_node = svo_nodes[0];
    let has_voxels = (root_node.child_mask != 0u || root_node.material_id != 0u);

    if (has_voxels && r_exit >= max(r_enter, 0.0)) {
        var stack: array<StackNode, 22>;
        var stack_len: i32 = 1;
        stack[0] = StackNode(0u, root_min, camera.world_size, max(r_enter, 0.0));

        var steps = 0;
        let total_nodes = arrayLength(&svo_nodes);

        while (stack_len > 0 && steps < 128) {
            steps = steps + 1;
            stack_len = stack_len - 1;
            let curr = stack[stack_len];

            if (curr.t_enter >= hit_t || curr.idx >= total_nodes) { continue; }

            let node = svo_nodes[curr.idx];

            if (node.child_pointer == 0u) {
                if (node.material_id != 0u) {
                    hit_mat = node.material_id;
                    hit_t = curr.t_enter;
                    let p_hit = ray_orig + ray_dir * curr.t_enter;
                    let c_min = curr.b_min;
                    let c_max = curr.b_min + vec3<f32>(curr.size);
                    let eps = 0.002 * curr.size;
                    if (abs(p_hit.x - c_min.x) < eps) { hit_normal = vec3<f32>(-1.0, 0.0, 0.0); }
                    else if (abs(p_hit.x - c_max.x) < eps) { hit_normal = vec3<f32>(1.0, 0.0, 0.0); }
                    else if (abs(p_hit.y - c_min.y) < eps) { hit_normal = vec3<f32>(0.0, -1.0, 0.0); }
                    else if (abs(p_hit.y - c_max.y) < eps) { hit_normal = vec3<f32>(0.0, 1.0, 0.0); }
                    else if (abs(p_hit.z - c_min.z) < eps) { hit_normal = vec3<f32>(0.0, 0.0, -1.0); }
                    else { hit_normal = vec3<f32>(0.0, 0.0, 1.0); }
                    break;
                }
                continue;
            }

            let half_s = curr.size * 0.5;

            var count = 0u;
            var cand_idx: array<u32, 8>;
            var cand_min: array<vec3<f32>, 8>;
            var cand_t: array<f32, 8>;

            for (var i = 0u; i < 8u; i = i + 1u) {
                if ((node.child_mask & (1u << i)) != 0u) {
                    let offset = vec3<f32>(
                        select(0.0, half_s, (i & 1u) != 0u),
                        select(0.0, half_s, (i & 2u) != 0u),
                        select(0.0, half_s, (i & 4u) != 0u)
                    );
                    let child_min = curr.b_min + offset;
                    let child_max = child_min + vec3<f32>(half_s);

                    let t0 = (child_min - ray_orig) * inv_dir;
                    let t1 = (child_max - ray_orig) * inv_dir;
                    let tmin3 = min(t0, t1);
                    let tmax3 = max(t0, t1);
                    let ct_enter = max(max(tmin3.x, tmin3.y), tmin3.z);
                    let ct_exit  = min(min(tmax3.x, tmax3.y), tmax3.z);

                    if (ct_exit >= max(ct_enter, 0.0) && ct_enter < hit_t) {
                        cand_idx[count] = node.child_pointer + i;
                        cand_min[count] = child_min;
                        cand_t[count] = max(ct_enter, 0.0);
                        count = count + 1u;
                    }
                }
            }

            // Descending insertion sort: keeps closest child on top of LIFO stack without register thrashing
            for (var a = 1u; a < count; a = a + 1u) {
                let cur_t = cand_t[a];
                let cur_idx = cand_idx[a];
                let cur_min = cand_min[a];
                var j = a;
                while (j > 0u && cand_t[j - 1u] < cur_t) {
                    cand_t[j] = cand_t[j - 1u];
                    cand_idx[j] = cand_idx[j - 1u];
                    cand_min[j] = cand_min[j - 1u];
                    j = j - 1u;
                }
                cand_t[j] = cur_t;
                cand_idx[j] = cur_idx;
                cand_min[j] = cur_min;
            }

            for (var k = 0u; k < count; k = k + 1u) {
                if (stack_len < 21) {
                    stack[stack_len] = StackNode(cand_idx[k], cand_min[k], half_s, cand_t[k]);
                    stack_len = stack_len + 1;
                }
            }
        }
    }

    if (hit_mat == 0u) {
        if (abs(ray_dir.y) > 1e-5) {
            let t_ground = (-ray_orig.y) / ray_dir.y;
            if (t_ground > 0.0 && t_ground < 15000.0) {
                let p_world = ray_orig + ray_dir * t_ground;
                let grid_coord = abs(fract(p_world.xz * 0.5) - 0.5);
                let line = smoothstep(0.0, 0.04, min(grid_coord.x, grid_coord.y));
                let grid_col = mix(vec3<f32>(0.26, 0.30, 0.36), vec3<f32>(0.10, 0.12, 0.15), line);
                let fog = clamp(t_ground / 15000.0, 0.0, 1.0);
                return vec4<f32>(mix(grid_col, vec3<f32>(0.12, 0.14, 0.18), fog), 1.0);
            }
        }
        discard;
    }

    let light_dir = normalize(vec3<f32>(0.4, 0.9, 0.3));
    let ndotl = max(dot(hit_normal, light_dir), 0.0);

    var lighting: f32 = 0.45;
    if (ndotl > 0.65) {
        lighting = 1.00;
    } else if (ndotl > 0.20) {
        lighting = 0.72;
    }

    let pal_idx = min(hit_mat - 1u, arrayLength(&palette) - 1u);
    let base_col = palette[pal_idx].xyz;
    let final_col = base_col * lighting;

    return vec4<f32>(final_col, 1.0);
}