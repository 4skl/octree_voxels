use crate::types::*;
use glam::{Mat4, Vec3};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{mpsc, Arc, RwLock};

pub enum VoxelizeMsg {
    Progress { percent: f32, stage: String },
    Done(Result<Vec<(Vec3, f32, u16)>, String>),
}

#[inline(always)]
fn pack_coords(gx: i32, gy: i32, gz: i32) -> u64 {
    const OFFSET: i64 = 1 << 20;
    let ux = ((gx as i64) + OFFSET) as u64 & 0x1F_FFFF;
    let uy = ((gy as i64) + OFFSET) as u64 & 0x1F_FFFF;
    let uz = ((gz as i64) + OFFSET) as u64 & 0x1F_FFFF;
    (ux << 42) | (uy << 21) | uz
}

#[inline(always)]
fn unpack_coords(k: u64) -> (i32, i32, i32) {
    const OFFSET: i64 = 1 << 20;
    let gx = (((k >> 42) & 0x1F_FFFF) as i64 - OFFSET) as i32;
    let gy = (((k >> 21) & 0x1F_FFFF) as i64 - OFFSET) as i32;
    let gz = ((k & 0x1F_FFFF) as i64 - OFFSET) as i32;
    (gx, gy, gz)
}

fn sample_triangle(pts: [Vec3; 3], color: [f32; 3], voxel_size: f32, out: &mut Vec<(u64, [f32; 3])>) {
    let min_p = pts[0].min(pts[1]).min(pts[2]);
    let max_p = pts[0].max(pts[1]).max(pts[2]);
    let min_gx = (min_p.x / voxel_size).floor() as i32;
    let max_gx = (max_p.x / voxel_size).floor() as i32;
    let min_gy = (min_p.y / voxel_size).floor() as i32;
    let max_gy = (max_p.y / voxel_size).floor() as i32;
    let min_gz = (min_p.z / voxel_size).floor() as i32;
    let max_gz = (max_p.z / voxel_size).floor() as i32;
    let ext = voxel_size * 0.5;
    let box_half = Vec3::splat(ext);

    for gx in min_gx..=max_gx {
        for gy in min_gy..=max_gy {
            for gz in min_gz..=max_gz {
                let center = Vec3::new(gx as f32 * voxel_size + ext, gy as f32 * voxel_size + ext, gz as f32 * voxel_size + ext);
                let v0 = pts[0] - center;
                let v1 = pts[1] - center;
                let v2 = pts[2] - center;
                if v0.min(v1).min(v2).max_element() > ext || v0.max(v1).max(v2).min_element() < -ext { continue; }
                let normal = (v1 - v0).cross(v2 - v0);
                let r = box_half.x * normal.x.abs() + box_half.y * normal.y.abs() + box_half.z * normal.z.abs();
                if normal.dot(v0).abs() > r { continue; }
                let f0 = v1 - v0; let f1 = v2 - v1; let f2 = v0 - v2;
                let test_axis = |axis: Vec3| -> bool {
                    let p0 = v0.dot(axis); let p1 = v1.dot(axis); let p2 = v2.dot(axis);
                    let rad = box_half.x * axis.x.abs() + box_half.y * axis.y.abs() + box_half.z * axis.z.abs();
                    p0.min(p1).min(p2) > rad || p0.max(p1).max(p2) < -rad
                };
                if test_axis(Vec3::new(0.0, -f0.z, f0.y))
                    || test_axis(Vec3::new(0.0, -f1.z, f1.y))
                    || test_axis(Vec3::new(0.0, -f2.z, f2.y))
                    || test_axis(Vec3::new(f0.z, 0.0, -f0.x))
                    || test_axis(Vec3::new(f1.z, 0.0, -f1.x))
                    || test_axis(Vec3::new(f2.z, 0.0, -f2.x))
                    || test_axis(Vec3::new(-f0.y, f0.x, 0.0))
                    || test_axis(Vec3::new(-f1.y, f1.x, 0.0))
                    || test_axis(Vec3::new(-f2.y, f2.x, 0.0))
                {
                    continue;
                }
                out.push((pack_coords(gx, gy, gz), color));
            }
        }
    }
}

pub fn run_background_voxelization(
    path: PathBuf, target_pos: Vec3, voxel_size: f32, target_height: f32,
    palette_size: usize, palette_arc: Arc<RwLock<Palette>>, max_nodes_allowed: usize, tx: mpsc::Sender<VoxelizeMsg>,
) {
    let _ = tx.send(VoxelizeMsg::Progress { percent: 0.05, stage: "READING GLB CONTAINER...".into() });
    let (document, buffers, _) = match gltf::import(&path) {
        Ok(res) => res,
        Err(e) => { let _ = tx.send(VoxelizeMsg::Done(Err(format!("Import error: {e}")))); return; }
    };

    let mut min_bound = Vec3::splat(f32::MAX);
    let mut max_bound = Vec3::splat(f32::MIN);
    let mut all_triangles = Vec::new();
    let mut root_nodes: Vec<gltf::Node> = Vec::new();

    if let Some(scene) = document.default_scene() { root_nodes.extend(scene.nodes()); }
    else if let Some(scene) = document.scenes().next() { root_nodes.extend(scene.nodes()); }
    else {
        let mut child_indices = HashSet::new();
        for node in document.nodes() { for child in node.children() { child_indices.insert(child.index()); } }
        for node in document.nodes() { if !child_indices.contains(&node.index()) { root_nodes.push(node); } }
    }

    let mut stack: Vec<(gltf::Node, Mat4)> = root_nodes.into_iter().map(|node| (node, Mat4::IDENTITY)).collect();
    while let Some((node, parent_mat)) = stack.pop() {
        let local_mat = Mat4::from_cols_array_2d(&node.transform().matrix());
        let world_mat = parent_mat * local_mat;
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                let reader = primitive.reader(|b| Some(&buffers[b.index()]));
                let positions: Vec<Vec3> = match reader.read_positions() {
                    Some(iter) => iter.map(|p| world_mat.transform_point3(Vec3::from(p))).collect(),
                    None => continue,
                };
                for p in &positions { min_bound = min_bound.min(*p); max_bound = max_bound.max(*p); }
                let colors: Vec<[f32; 3]> = reader.read_colors(0).map(|iter| iter.into_rgb_f32().collect()).unwrap_or_else(|| vec![[1.0, 1.0, 1.0]; positions.len()]);
                let pbr = primitive.material().pbr_metallic_roughness();
                let base_factor = pbr.base_color_factor();
                let indices: Vec<u32> = reader.read_indices().map(|iter| iter.into_u32().collect()).unwrap_or_else(|| (0..positions.len() as u32).collect());
                for chunk in indices.chunks_exact(3) {
                    let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
                    if i0 < positions.len() && i1 < positions.len() && i2 < positions.len() {
                        let avg_color = [
                            (colors[i0][0] + colors[i1][0] + colors[i2][0]) / 3.0 * base_factor[0],
                            (colors[i0][1] + colors[i1][1] + colors[i2][1]) / 3.0 * base_factor[1],
                            (colors[i0][2] + colors[i1][2] + colors[i2][2]) / 3.0 * base_factor[2],
                        ];
                        all_triangles.push(([positions[i0], positions[i1], positions[i2]], avg_color));
                    }
                }
            }
        }
        for child in node.children() { stack.push((child, world_mat)); }
    }

    if all_triangles.is_empty() {
        let _ = tx.send(VoxelizeMsg::Done(Err("No valid mesh geometry found in GLB".into())));
        return;
    }

    let extent = max_bound - min_bound;
    let max_dim = extent.x.max(extent.y).max(extent.z).max(0.001);
    let scale = target_height / max_dim;
    let center_x = (min_bound.x + max_bound.x) * 0.5;
    let center_z = (min_bound.z + max_bound.z) * 0.5;
    let min_y = min_bound.y;

    let normalized_triangles: Vec<_> = all_triangles.into_iter().map(|(pts, col)| {
        let p0 = Vec3::new((pts[0].x - center_x) * scale, (pts[0].y - min_y) * scale, (pts[0].z - center_z) * scale) + target_pos;
        let p1 = Vec3::new((pts[1].x - center_x) * scale, (pts[1].y - min_y) * scale, (pts[1].z - center_z) * scale) + target_pos;
        let p2 = Vec3::new((pts[2].x - center_x) * scale, (pts[2].y - min_y) * scale, (pts[2].z - center_z) * scale) + target_pos;
        ([p0, p1, p2], col)
    }).collect();

    let sampled_batches: Vec<Vec<(u64, [f32; 3])>> = normalized_triangles.par_chunks(2048).map(|chunk| {
        let mut local_out = Vec::with_capacity(chunk.len() * 2);
        for &(pts, color) in chunk { sample_triangle(pts, color, voxel_size, &mut local_out); }
        local_out.sort_unstable_by_key(|&(k, _)| k);
        local_out.dedup_by(|a, b| a.0 == b.0);
        local_out
    }).collect();

    let mut all_samples: Vec<(u64, [f32; 3])> = sampled_batches.into_par_iter().flatten().collect();
    all_samples.par_sort_unstable_by_key(|&(k, _)| k);
    all_samples.dedup_by(|a, b| a.0 == b.0);

    if all_samples.is_empty() {
        let _ = tx.send(VoxelizeMsg::Done(Err("Surface voxelization produced 0 voxels".into())));
        return;
    }

    let _ = tx.send(VoxelizeMsg::Progress { percent: 0.60, stage: format!("PALETTE MAPPING ({} VOXELS)...", all_samples.len()) });

    let mut pal = palette_arc.write().unwrap();
    let mut color_cache: HashMap<[u8; 3], u16> = HashMap::new();
    for (i, &c) in pal.iter().enumerate() {
        color_cache.insert([(c[0] * 255.0).round() as u8, (c[1] * 255.0).round() as u8, (c[2] * 255.0).round() as u8], (i + 1) as u16);
    }

    let mut sample_mats: Vec<(i32, i32, i32, u16)> = Vec::with_capacity(all_samples.len());
    let mut min_gx = i32::MAX; let mut max_gx = i32::MIN;
    let mut min_gy = i32::MAX; let mut max_gy = i32::MIN;
    let mut min_gz = i32::MAX; let mut max_gz = i32::MIN;

    for (k, color) in all_samples {
        let key = [(color[0] * 255.0).round() as u8, (color[1] * 255.0).round() as u8, (color[2] * 255.0).round() as u8];
        let mat_id = if let Some(&id) = color_cache.get(&key) { id } else {
            if pal.len() < palette_size {
                pal.push(color);
                let new_id = pal.len() as u16;
                color_cache.insert(key, new_id);
                new_id
            } else { 1 }
        };
        let (gx, gy, gz) = unpack_coords(k);
        min_gx = min_gx.min(gx); max_gx = max_gx.max(gx);
        min_gy = min_gy.min(gy); max_gy = max_gy.max(gy);
        min_gz = min_gz.min(gz); max_gz = max_gz.max(gz);
        sample_mats.push((gx, gy, gz, mat_id));
    }
    drop(pal);

    let _ = tx.send(VoxelizeMsg::Progress { percent: 0.75, stage: "SOLID FLOOD-FILL INTERIOR...".into() });

    let size_x = (max_gx - min_gx + 1).max(1) as usize;
    let size_y = (max_gy - min_gy + 1).max(1) as usize;
    let size_z = (max_gz - min_gz + 1).max(1) as usize;

    let dim_x = size_x + 2;
    let dim_y = size_y + 2;
    let dim_z = size_z + 2;
    let total_cells = dim_x.saturating_mul(dim_y).saturating_mul(dim_z);

    let mut voxel_nodes: Vec<(Vec3, f32, u16)> = Vec::new();

    if total_cells > 0 && total_cells <= 24_000_000 {
        const AIR: u16 = 65535;
        let mut grid: Vec<u16> = vec![0; total_cells];
        let stride_y = dim_x;
        let stride_z = dim_x * dim_y;

        let mut color_queue: VecDeque<(usize, usize, usize, u16)> = VecDeque::new();
        for &(gx, gy, gz, mat) in &sample_mats {
            let lx = (gx - min_gx + 1) as usize;
            let ly = (gy - min_gy + 1) as usize;
            let lz = (gz - min_gz + 1) as usize;
            let idx = lx + ly * stride_y + lz * stride_z;
            grid[idx] = mat;
            color_queue.push_back((lx, ly, lz, mat));
        }

        let mut ext_queue: VecDeque<(usize, usize, usize)> = VecDeque::new();
        grid[0] = AIR;
        ext_queue.push_back((0usize, 0usize, 0usize));

        while let Some((cx, cy, cz)) = ext_queue.pop_front() {
            let mut neighbors = [(0usize, 0usize, 0usize); 6];
            let mut count = 0;
            if cx > 0 { neighbors[count] = (cx - 1, cy, cz); count += 1; }
            if cx + 1 < dim_x { neighbors[count] = (cx + 1, cy, cz); count += 1; }
            if cy > 0 { neighbors[count] = (cx, cy - 1, cz); count += 1; }
            if cy + 1 < dim_y { neighbors[count] = (cx, cy + 1, cz); count += 1; }
            if cz > 0 { neighbors[count] = (cx, cy, cz - 1); count += 1; }
            if cz + 1 < dim_z { neighbors[count] = (cx, cy, cz + 1); count += 1; }

            for i in 0..count {
                let (nx, ny, nz) = neighbors[i];
                let n_idx = nx + ny * stride_y + nz * stride_z;
                if grid[n_idx] == 0 {
                    grid[n_idx] = AIR;
                    ext_queue.push_back((nx, ny, nz));
                }
            }
        }

        while let Some((cx, cy, cz, mat_id)) = color_queue.pop_front() {
            let mut neighbors = [(0usize, 0usize, 0usize); 6];
            let mut count = 0;
            if cx > 0 { neighbors[count] = (cx - 1, cy, cz); count += 1; }
            if cx + 1 < dim_x { neighbors[count] = (cx + 1, cy, cz); count += 1; }
            if cy > 0 { neighbors[count] = (cx, cy - 1, cz); count += 1; }
            if cy + 1 < dim_y { neighbors[count] = (cx, cy + 1, cz); count += 1; }
            if cz > 0 { neighbors[count] = (cx, cy, cz - 1); count += 1; }
            if cz + 1 < dim_z { neighbors[count] = (cx, cy, cz + 1); count += 1; }

            for i in 0..count {
                let (nx, ny, nz) = neighbors[i];
                let n_idx = nx + ny * stride_y + nz * stride_z;
                if grid[n_idx] == 0 {
                    grid[n_idx] = mat_id;
                    color_queue.push_back((nx, ny, nz, mat_id));
                }
            }
        }

        for lz in 1..=size_z {
            for ly in 1..=size_y {
                for lx in 1..=size_x {
                    let idx = lx + ly * stride_y + lz * stride_z;
                    let mat = grid[idx];
                    if mat != 0 && mat != AIR {
                        let gx = (lx as i32 - 1) + min_gx;
                        let gy = (ly as i32 - 1) + min_gy;
                        let gz = (lz as i32 - 1) + min_gz;
                        let wx = gx as f32 * voxel_size;
                        let wy = gy as f32 * voxel_size;
                        let wz = gz as f32 * voxel_size;
                        voxel_nodes.push((Vec3::new(wx, wy, wz), voxel_size, mat));
                    }
                }
            }
        }
    } else {
        for (gx, gy, gz, mat) in sample_mats {
            let wx = gx as f32 * voxel_size;
            let wy = gy as f32 * voxel_size;
            let wz = gz as f32 * voxel_size;
            voxel_nodes.push((Vec3::new(wx, wy, wz), voxel_size, mat));
        }
    }

    if voxel_nodes.len() > max_nodes_allowed {
        let _ = tx.send(VoxelizeMsg::Done(Err(format!(
            "Model exceeded limits: {} raw voxels (Limit: {}). Increase voxel resolution.",
            voxel_nodes.len(), max_nodes_allowed
        ))));
        return;
    }

    let _ = tx.send(VoxelizeMsg::Progress { percent: 0.95, stage: "SORTING MORTON SPATIAL CURVE...".into() });

    voxel_nodes.par_sort_unstable_by_key(|(pos, s, _)| {
        let gx = (pos.x / s).round() as i32;
        let gy = (pos.y / s).round() as i32;
        let gz = (pos.z / s).round() as i32;
        morton_encode_coords(gx, gy, gz)
    });

    let _ = tx.send(VoxelizeMsg::Progress { percent: 0.99, stage: "STREAMING INTO SVO HIERARCHY...".into() });
    let _ = tx.send(VoxelizeMsg::Done(Ok(voxel_nodes)));
}