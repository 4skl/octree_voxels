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
    bg_color: vec3<f32>,
    _pad: f32,
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

struct DDAStackEntry {
    idx: u32,
    b_min: vec3<f32>,
    size: f32,
    t_exit_x: f32,
    t_exit_y: f32,
    t_exit_z: f32,
    octant: u32,
};

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);

    var ray_orig: vec3<f32>;
    var ray_dir: vec3<f32>;

    if (camera.is_ortho > 0.5) {
        let p_near = camera.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
        let p_far  = camera.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
        let ro = p_near.xyz / p_near.w;
        ray_dir = normalize((p_far.xyz / p_far.w) - ro);
        ray_orig = ro - ray_dir * 2000.0;
    } else {
        let p_far = camera.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
        let p_world = p_far.xyz / p_far.w;
        ray_orig = camera.camera_pos;
        ray_dir = normalize(p_world - ray_orig);
    }

    let eps_dir = 1e-7;
    let rd = vec3<f32>(
        select(ray_dir.x, select(-eps_dir, eps_dir, ray_dir.x >= 0.0), abs(ray_dir.x) < eps_dir),
        select(ray_dir.y, select(-eps_dir, eps_dir, ray_dir.y >= 0.0), abs(ray_dir.y) < eps_dir),
        select(ray_dir.z, select(-eps_dir, eps_dir, ray_dir.z >= 0.0), abs(ray_dir.z) < eps_dir),
    );
    let inv_dir = 1.0 / rd;

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
        var t_curr = max(r_enter, 0.0);
        let second_half = vec3<u32>(
            select(0u, 1u, rd.x > 0.0),
            select(0u, 1u, rd.y > 0.0),
            select(0u, 1u, rd.z > 0.0)
        );

        var stack: array<DDAStackEntry, 24>;
        var level: i32 = 0;

        stack[0].idx = 0u;
        stack[0].b_min = root_min;
        stack[0].size = camera.world_size;
        stack[0].t_exit_x = rtmax3.x;
        stack[0].t_exit_y = rtmax3.y;
        stack[0].t_exit_z = rtmax3.z;

        let root_half = camera.world_size * 0.5;
        let root_mid = root_min + vec3<f32>(root_half);
        let root_t_mid = (root_mid - ray_orig) * inv_dir;
        let r_bx = select(u32(t_curr < root_t_mid.x), u32(t_curr >= root_t_mid.x), rd.x > 0.0);
        let r_by = select(u32(t_curr < root_t_mid.y), u32(t_curr >= root_t_mid.y), rd.y > 0.0);
        let r_bz = select(u32(t_curr < root_t_mid.z), u32(t_curr >= root_t_mid.z), rd.z > 0.0);
        stack[0].octant = r_bx | (r_by << 1u) | (r_bz << 2u);

        var steps = 0;
        let total_nodes = arrayLength(&svo_nodes);

        while (level >= 0 && steps < 384) {
            steps = steps + 1;
            let curr = stack[level];
            let parent_exit = min(curr.t_exit_x, min(curr.t_exit_y, curr.t_exit_z));

            if (t_curr >= parent_exit - 1e-5) {
                level = level - 1;
                continue;
            }

            if (curr.idx >= total_nodes) {
                level = level - 1;
                continue;
            }

            let node = svo_nodes[curr.idx];

            if (node.child_pointer == 0u) {
                if (node.material_id != 0u) {
                    let p_hit = ray_orig + rd * t_curr;

                    if (camera.show_borders > 0.5) {
                        let local_p = saturate((p_hit - curr.b_min) / curr.size);
                        let dist_x = min(local_p.x, 1.0 - local_p.x);
                        let dist_y = min(local_p.y, 1.0 - local_p.y);
                        let dist_z = min(local_p.z, 1.0 - local_p.z);
                        let edge_thresh = 0.035;

                        if (!((dist_x < edge_thresh && dist_y < edge_thresh) ||
                              (dist_y < edge_thresh && dist_z < edge_thresh) ||
                              (dist_z < edge_thresh && dist_x < edge_thresh))) {
                            t_curr = parent_exit;
                            level = level - 1;
                            continue;
                        }
                    }

                    let half_node = curr.size * 0.5;
                    let center = curr.b_min + vec3<f32>(half_node);
                    let p_rel = (p_hit - center) / half_node;
                    let abs_p = abs(p_rel);
                    let max_c = max(max(abs_p.x, abs_p.y), abs_p.z);

                    if (max_c == abs_p.x) {
                        hit_normal = vec3<f32>(sign(p_rel.x), 0.0, 0.0);
                    } else if (max_c == abs_p.y) {
                        hit_normal = vec3<f32>(0.0, sign(p_rel.y), 0.0);
                    } else {
                        hit_normal = vec3<f32>(0.0, 0.0, sign(p_rel.z));
                    }

                    hit_mat = node.material_id;
                    hit_t = t_curr;
                    break;
                }

                t_curr = parent_exit;
                level = level - 1;
                continue;
            }

            let half_s = curr.size * 0.5;
            let mid = curr.b_min + vec3<f32>(half_s);
            let t_mid = (mid - ray_orig) * inv_dir;

            let oct = stack[level].octant;
            let bx = oct & 1u;
            let by = (oct >> 1u) & 1u;
            let bz = (oct >> 2u) & 1u;

            let c_exit_x = select(t_mid.x, curr.t_exit_x, bx == second_half.x);
            let c_exit_y = select(t_mid.y, curr.t_exit_y, by == second_half.y);
            let c_exit_z = select(t_mid.z, curr.t_exit_z, bz == second_half.z);
            let c_exit = min(c_exit_x, min(c_exit_y, c_exit_z));

            let has_child = (node.child_mask & (1u << oct)) != 0u;

            if (c_exit < parent_exit - 1e-5) {
                if (c_exit == c_exit_x) { stack[level].octant ^= 1u; }
                if (c_exit == c_exit_y) { stack[level].octant ^= 2u; }
                if (c_exit == c_exit_z) { stack[level].octant ^= 4u; }
            }

            if (has_child && level < 23) {
                let child_idx = node.child_pointer + oct;
                let child_min = curr.b_min + vec3<f32>(
                    select(0.0, half_s, bx != 0u),
                    select(0.0, half_s, by != 0u),
                    select(0.0, half_s, bz != 0u)
                );

                let ch_half = half_s * 0.5;
                let ch_mid = child_min + vec3<f32>(ch_half);
                let ch_t_mid = (ch_mid - ray_orig) * inv_dir;
                let cbx = select(u32(t_curr < ch_t_mid.x), u32(t_curr >= ch_t_mid.x), rd.x > 0.0);
                let cby = select(u32(t_curr < ch_t_mid.y), u32(t_curr >= ch_t_mid.y), rd.y > 0.0);
                let cbz = select(u32(t_curr < ch_t_mid.z), u32(t_curr >= ch_t_mid.z), rd.z > 0.0);

                level = level + 1;
                stack[level].idx = child_idx;
                stack[level].b_min = child_min;
                stack[level].size = half_s;
                stack[level].t_exit_x = c_exit_x;
                stack[level].t_exit_y = c_exit_y;
                stack[level].t_exit_z = c_exit_z;
                stack[level].octant = cbx | (cby << 1u) | (cbz << 2u);
            } else {
                t_curr = c_exit;
                if (c_exit >= parent_exit - 1e-5) {
                    level = level - 1;
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
                let grid_col = mix(camera.bg_color * 1.5, camera.bg_color * 0.7, line);
                let fog = clamp(t_ground / 15000.0, 0.0, 1.0);
                return vec4<f32>(mix(grid_col, camera.bg_color, fog), 1.0);
            }
        }
        return vec4<f32>(camera.bg_color, 1.0);
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