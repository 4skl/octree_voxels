mod types;
mod engine;

use types::*;
use engine::*;

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
    dpi::LogicalSize,
};
#[cfg(target_os = "linux")]
use winit::platform::wayland::WindowAttributesExtWayland;
#[cfg(target_os = "linux")]
use winit::platform::x11::WindowAttributesExtX11;

use wgpu::util::DeviceExt;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use std::collections::{HashMap, HashSet};
use std::sync::mpsc;
use std::path::PathBuf;
use glam::{Vec3, Vec4, Mat4};
use rayon::prelude::*;

pub enum VoxelizeMsg {
    Progress { percent: f32, stage: String },
    Done(Result<Vec<(u32, u32, u32, u8, u16)>, String>),
}

#[inline(always)] fn pack_coords(gx: i32, gy: i32, gz: i32) -> u64 {
    const OFFSET: i64 = 1 << 20;
    let ux = ((gx as i64) + OFFSET) as u64 & 0x1F_FFFF;
    let uy = ((gy as i64) + OFFSET) as u64 & 0x1F_FFFF;
    let uz = ((gz as i64) + OFFSET) as u64 & 0x1F_FFFF;
    (ux << 42) | (uy << 21) | uz
}
#[inline(always)] fn unpack_coords(k: u64) -> (i32, i32, i32) {
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
                let v0 = pts[0] - center; let v1 = pts[1] - center; let v2 = pts[2] - center;
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
                if test_axis(Vec3::new(0.0, -f0.z, f0.y)) || test_axis(Vec3::new(0.0, -f1.z, f1.y)) || test_axis(Vec3::new(0.0, -f2.z, f2.y)) || test_axis(Vec3::new(f0.z, 0.0, -f0.x)) || test_axis(Vec3::new(f1.z, 0.0, -f1.x)) || test_axis(Vec3::new(f2.z, 0.0, -f2.x)) || test_axis(Vec3::new(-f0.y, f0.x, 0.0)) || test_axis(Vec3::new(-f1.y, f1.x, 0.0)) || test_axis(Vec3::new(-f2.y, f2.x, 0.0)) { continue; }
                out.push((pack_coords(gx, gy, gz), color));
            }
        }
    }
}

fn run_background_voxelization(
    path: PathBuf, target_pos: Vec3, voxel_size: f32, target_height: f32,
    palette_size: usize, palette_arc: Arc<RwLock<Palette>>, max_nodes_allowed: usize, tx: mpsc::Sender<VoxelizeMsg>
) {
    let _ = tx.send(VoxelizeMsg::Progress { percent: 0.05, stage: "READING GLB CONTAINER...".into() });
    let (document, buffers, _) = match gltf::import(&path) {
        Ok(res) => res,
        Err(e) => { let _ = tx.send(VoxelizeMsg::Done(Err(format!("Import error: {}", e)))); return; }
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

    if all_samples.len() > max_nodes_allowed {
        let _ = tx.send(VoxelizeMsg::Done(Err(format!(
            "Model too dense: {} voxels (Hardware limit: {}). Increase voxel size or decrease height.",
            all_samples.len(), max_nodes_allowed
        ))));
        return;
    }

    let _ = tx.send(VoxelizeMsg::Progress { percent: 0.80, stage: format!("RESOLVING PALETTE ({} VOXELS)...", all_samples.len()) });

    let mut pal = palette_arc.write().unwrap();
    let mut color_cache: HashMap<[u8; 3], u16> = HashMap::new();
    for (i, &c) in pal.iter().enumerate() {
        color_cache.insert([(c[0] * 255.0).round() as u8, (c[1] * 255.0).round() as u8, (c[2] * 255.0).round() as u8], (i + 1) as u16);
    }

    let grid_size = ((voxel_size / WORLD_SIZE) * GRID_RES as f32).round().max(1.0) as u32;
    let depth = MAX_DEPTH - grid_size.trailing_zeros().min(MAX_DEPTH as u32) as u8;

    let mut voxel_nodes = Vec::with_capacity(all_samples.len());
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
        let wx = gx as f32 * voxel_size;
        let wy = gy as f32 * voxel_size;
        let wz = gz as f32 * voxel_size;
        let ux = (((wx - WORLD_MIN.x) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
        let uy = (((wy - WORLD_MIN.y) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
        let uz = (((wz - WORLD_MIN.z) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
        voxel_nodes.push((ux, uy, uz, depth, mat_id));
    }

    let _ = tx.send(VoxelizeMsg::Progress { percent: 0.99, stage: "INSERTING INTO SVO...".into() });
    let _ = tx.send(VoxelizeMsg::Done(Ok(voxel_nodes)));
}

#[derive(Default)]
struct InputState { forward: bool, backward: bool, left: bool, right: bool, up: bool, down: bool, action_add: bool, action_remove: bool, action_pick: bool, ctrl_pressed: bool }

fn add_quad(verts: &mut Vec<UIVertex>, x0: f32, y0: f32, x1: f32, y1: f32, color: [f32; 4]) {
    verts.extend_from_slice(&[UIVertex { position: [x0, y0], color }, UIVertex { position: [x1, y0], color }, UIVertex { position: [x1, y1], color }, UIVertex { position: [x0, y0], color }, UIVertex { position: [x1, y1], color }, UIVertex { position: [x0, y1], color }]);
}
fn add_line(verts: &mut Vec<UIVertex>, x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32, aspect: f32, color: [f32; 4]) {
    let dx = (x1 - x0) * aspect; let dy = y1 - y0; let len = (dx * dx + dy * dy).sqrt(); if len < 1e-5 { return; }
    let half_t = thickness * 0.5; let nx = (-dy / len) * half_t / aspect; let ny = (dx / len) * half_t;
    verts.extend_from_slice(&[UIVertex { position: [x0 - nx, y0 - ny], color }, UIVertex { position: [x1 - nx, y1 - ny], color }, UIVertex { position: [x1 + nx, y1 + ny], color }, UIVertex { position: [x0 - nx, y0 - ny], color }, UIVertex { position: [x1 + nx, y1 + ny], color }, UIVertex { position: [x0 + nx, y0 + ny], color }]);
}

fn get_glyph_5x7(c: char) -> [u8; 7] {
    match c {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001], 'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110], 'C' => [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110], 'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111], 'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01111, 0b10000, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111], 'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001], 'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00001, 0b00001, 0b00001, 0b00001, 0b10001, 0b10001, 0b01110], 'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001], 'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0b10001, 0b10001], 'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001], 'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000], 'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101], 'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110], 'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100], 'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100], 'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001], 'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100], 'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111], '0' => [0b01110, 0b10011, 0b10101, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110], '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111], '3' => [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010], '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110], '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000], '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110], '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        ':' => [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000], '/' => [0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b00000, 0b00000], '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000], '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100], '%' => [0b11001, 0b11010, 0b00100, 0b01000, 0b01011, 0b10011, 0b00000],
        '[' => [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110], ']' => [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110], '(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        ')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000], _ => [0; 7],
    }
}
fn draw_glyph_raw(verts: &mut Vec<UIVertex>, glyph: &[u8; 7], x: f32, y: f32, pw: f32, ph: f32, color: [f32; 4]) {
    for row in 0..7 {
        let line = glyph[row]; if line == 0 { continue; }
        let y1 = y + (6 - row) as f32 * ph; let y0 = y1 - ph;
        for col in 0..5 { if (line & (1 << (4 - col))) != 0 { let x0 = x + col as f32 * pw; let x1 = x0 + pw; add_quad(verts, x0, y0, x1, y1, color); } }
    }
}
pub fn draw_text(verts: &mut Vec<UIVertex>, text: &str, start_x: f32, start_y: f32, scale: f32, aspect: f32, color: [f32; 4]) {
    let ph = 0.0032 * scale; let pw = ph / aspect;
    let shadow_color = [0.02, 0.02, 0.04, color[3] * 0.9]; let shadow_offset_x = pw * 0.75; let shadow_offset_y = -ph * 0.75;
    let mut cursor_x = start_x;
    for c in text.to_ascii_uppercase().chars() {
        let glyph = get_glyph_5x7(c);
        draw_glyph_raw(verts, &glyph, cursor_x + shadow_offset_x, start_y + shadow_offset_y, pw, ph, shadow_color);
        draw_glyph_raw(verts, &glyph, cursor_x, start_y, pw, ph, color); cursor_x += 6.0 * pw;
    }
}
pub fn draw_text_centered(verts: &mut Vec<UIVertex>, text: &str, cx: f32, cy: f32, scale: f32, aspect: f32, color: [f32; 4]) {
    let ph = 0.0032 * scale; let pw = ph / aspect; let total_w = (text.len() as f32 * 6.0 - 1.0) * pw; let total_h = 7.0 * ph;
    draw_text(verts, text, cx - total_w / 2.0, cy - total_h / 2.0, scale, aspect, color);
}

const PRESET_SWATCHES: [[f32; 3]; 10] = [[0.95, 0.95, 0.95], [0.10, 0.10, 0.12], [0.90, 0.20, 0.20], [0.95, 0.50, 0.15], [0.95, 0.85, 0.15], [0.20, 0.75, 0.25], [0.15, 0.80, 0.85], [0.20, 0.45, 0.90], [0.65, 0.25, 0.85], [0.55, 0.35, 0.20]];
const GIZMO_CENTER_X: f32 = 0.86; const GIZMO_CENTER_Y: f32 = 0.76; const GIZMO_RADIUS: f32 = 0.11;

#[derive(Clone, Copy)]
struct GizmoAxis { dir: Vec3, name: &'static str, color: [f32; 4], yaw: f32, pitch: f32, is_positive: bool }
fn get_gizmo_axes() -> [GizmoAxis; 6] {
    [
        GizmoAxis { dir: Vec3::X,  name: "X", color: [0.90, 0.20, 0.25, 1.0], yaw: std::f32::consts::PI, pitch: 0.0, is_positive: true },
        GizmoAxis { dir: -Vec3::X, name: "-X", color: [0.45, 0.20, 0.20, 0.75], yaw: 0.0, pitch: 0.0, is_positive: false },
        GizmoAxis { dir: Vec3::Y,  name: "Y", color: [0.30, 0.80, 0.20, 1.0], yaw: -std::f32::consts::FRAC_PI_2, pitch: -1.56, is_positive: true },
        GizmoAxis { dir: -Vec3::Y, name: "-Y", color: [0.20, 0.45, 0.20, 0.75], yaw: -std::f32::consts::FRAC_PI_2, pitch: 1.56, is_positive: false },
        GizmoAxis { dir: Vec3::Z,  name: "Z", color: [0.18, 0.55, 0.95, 1.0], yaw: -std::f32::consts::FRAC_PI_2, pitch: 0.0, is_positive: true },
        GizmoAxis { dir: -Vec3::Z, name: "-Z", color: [0.20, 0.30, 0.50, 0.75], yaw: std::f32::consts::FRAC_PI_2, pitch: 0.0, is_positive: false },
    ]
}

fn get_focus_button_bounds(aspect: f32) -> (f32, f32, f32, f32) {
    let g_cx = GIZMO_CENTER_X; let g_cy = GIZMO_CENTER_Y; let disc_rx = (GIZMO_RADIUS + 0.02) / aspect;
    let btn_w = 0.075 / aspect; let btn_h = 0.055;
    let x1 = g_cx - disc_rx - 0.015; let x0 = x1 - btn_w; let y0 = g_cy - btn_h / 2.0; let y1 = g_cy + btn_h / 2.0;
    (x0, y0, x1, y1)
}

fn format_voxel_count(count: usize) -> String {
    if count >= 1_000_000 { format!("{:.2}M", count as f64 / 1_000_000.0) }
    else if count >= 1_000 { format!("{:.1}K", count as f64 / 1_000.0) }
    else { count.to_string() }
}

fn build_ui_vertices(
    selected_slot: usize, active_menu: ActiveMenu, hotbar_colors: &[[f32; 3]; 10], play_mode: PlayMode, is_ortho: bool, world_type: WorldType,
    edit_size: f32, target_pos: Option<[f32; 3]>, aspect: f32, camera_forward: Vec3, camera_right: Vec3, camera_up: Vec3,
    cursor_free: bool, glb_settings: &GlbImportSettings, progress_val: f32, progress_stage: &str, view_proj: Mat4, total_voxels: usize,
    error_banner: Option<&str>
) -> Vec<UIVertex> {
    let mut verts = Vec::new();
    let g_cx = GIZMO_CENTER_X; let g_cy = GIZMO_CENTER_Y; let g_rad = GIZMO_RADIUS; let disc_rx = (g_rad + 0.018) / aspect; let disc_ry = g_rad + 0.018;
    add_quad(&mut verts, g_cx - disc_rx - 0.003, g_cy - disc_ry - 0.003, g_cx + disc_rx + 0.003, g_cy + disc_ry + 0.003, [0.25, 0.30, 0.38, 0.6]);
    add_quad(&mut verts, g_cx - disc_rx, g_cy - disc_ry, g_cx + disc_rx, g_cy + disc_ry, [0.08, 0.10, 0.14, 0.70]);

    let (bx0, by0, bx1, by1) = get_focus_button_bounds(aspect);
    add_quad(&mut verts, bx0 - 0.003, by0 - 0.003, bx1 + 0.003, by1 + 0.003, [0.35, 0.40, 0.50, 0.8]);
    add_quad(&mut verts, bx0, by0, bx1, by1, [0.12, 0.15, 0.22, 0.90]); draw_text_centered(&mut verts, "[.]", (bx0 + bx1) / 2.0, (by0 + by1) / 2.0, 1.1, aspect, [0.3, 0.9, 1.0, 1.0]);

    let mut axes_projected: Vec<(GizmoAxis, f32, f32, f32)> = get_gizmo_axes().into_iter().map(|ax| (ax, ax.dir.dot(camera_right), ax.dir.dot(camera_up), ax.dir.dot(camera_forward))).collect();
    axes_projected.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));

    for (ax, sx, sy, _) in &axes_projected {
        let tip_x = g_cx + (sx * g_rad) / aspect; let tip_y = g_cy + (sy * g_rad);
        if ax.is_positive {
            add_line(&mut verts, g_cx, g_cy, tip_x, tip_y, 0.0045, aspect, [ax.color[0], ax.color[1], ax.color[2], 0.85]);
            let node_r = 0.021; add_quad(&mut verts, tip_x - node_r / aspect, tip_y - node_r, tip_x + node_r / aspect, tip_y + node_r, ax.color);
            draw_text_centered(&mut verts, ax.name, tip_x, tip_y, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
        } else {
            add_line(&mut verts, g_cx, g_cy, tip_x, tip_y, 0.0025, aspect, [ax.color[0], ax.color[1], ax.color[2], 0.35]);
            let node_r = 0.010; add_quad(&mut verts, tip_x - node_r / aspect, tip_y - node_r, tip_x + node_r / aspect, tip_y + node_r, ax.color);
        }
    }

    let num_slots = 10; let slot_w = 0.054; let slot_gap = 0.009; let total_w = num_slots as f32 * slot_w + (num_slots - 1) as f32 * slot_gap;
    let start_x = -total_w / 2.0; let y_bottom = -0.96; let y_top = -0.86;
    for (i, &rgb) in hotbar_colors.iter().enumerate() {
        let x0 = start_x + i as f32 * (slot_w + slot_gap); let x1 = x0 + slot_w;
        if selected_slot == i { add_quad(&mut verts, x0 - 0.006, y_bottom - 0.006, x1 + 0.006, y_top + 0.006, [1.0, 0.9, 0.1, 1.0]); } else { add_quad(&mut verts, x0 - 0.003, y_bottom - 0.003, x1 + 0.003, y_top + 0.003, [0.15, 0.16, 0.20, 0.9]); }
        add_quad(&mut verts, x0, y_bottom, x1, y_top, [rgb[0], rgb[1], rgb[2], 1.0]);
        let num_str = if i == 9 { "0".to_string() } else { format!("{}", i + 1) };
        draw_text_centered(&mut verts, &num_str, (x0 + x1) / 2.0, y_top + 0.02, 1.0, aspect, [0.9, 0.9, 0.9, 0.9]);
    }

    let size_str = if edit_size < 0.001 { format!("RES: {:.1e}", edit_size) } else if edit_size < 1.0 { format!("RES: 1/{} ({:.4})", (1.0 / edit_size).round() as u32, edit_size) } else { format!("RES: {:.0}X{:.0}", edit_size, edit_size) };

    if active_menu == ActiveMenu::None {
        if let Some(pos) = target_pos {
            let s = edit_size;
            let edges = [ ([0.0,0.0,0.0], [s,0.0,0.0]), ([0.0,0.0,0.0], [0.0,s,0.0]), ([0.0,0.0,0.0], [0.0,0.0,s]), ([s,s,s], [0.0,s,s]), ([s,s,s], [s,0.0,s]), ([s,s,s], [s,s,0.0]), ([s,0.0,0.0], [s,s,0.0]), ([s,0.0,0.0], [s,0.0,s]), ([0.0,s,0.0], [s,s,0.0]), ([0.0,s,0.0], [0.0,s,s]), ([0.0,0.0,s], [s,0.0,s]), ([0.0,0.0,s], [0.0,s,s]) ];
            for (p0, p1) in edges {
                let v0 = view_proj * Vec4::new(pos[0]+p0[0], pos[1]+p0[1], pos[2]+p0[2], 1.0);
                let v1 = view_proj * Vec4::new(pos[0]+p1[0], pos[1]+p1[1], pos[2]+p1[2], 1.0);
                if v0.w > 0.1 && v1.w > 0.1 {
                    let ndc0 = v0.truncate() / v0.w; let ndc1 = v1.truncate() / v1.w;
                    add_line(&mut verts, ndc0.x, ndc0.y, ndc1.x, ndc1.y, 0.003, aspect, [0.3, 0.9, 1.0, 0.5]);
                }
            }
        }
    }

    if let Some(err) = error_banner {
        add_quad(&mut verts, -0.85, 0.68, 0.85, 0.80, [0.70, 0.12, 0.12, 0.95]);
        draw_text_centered(&mut verts, err, 0.0, 0.74, 0.95, aspect, [1.0, 1.0, 1.0, 1.0]);
    }

    match active_menu {
        ActiveMenu::None => {
            add_quad(&mut verts, -0.012, -0.002, 0.012, 0.002, [1.0, 1.0, 1.0, 0.95]); add_quad(&mut verts, -0.002, -0.020, 0.002, 0.020, [1.0, 1.0, 1.0, 0.95]);
            draw_text(&mut verts, &format!("MODE: {} | PROJ: {} | WORLD: {} | VOXELS: {} | {}", if play_mode == PlayMode::Flying { "FLY" } else { "REAL" }, if is_ortho { "ORTHO" } else { "PERSP" }, world_type.name(), format_voxel_count(total_voxels), size_str), -0.96, 0.92, 1.25, aspect, [1.0, 1.0, 1.0, 0.95]);
            if cursor_free { draw_text(&mut verts, "CURSOR FREE  |  DRAG & DROP .GLB OR OPEN ESC MENU", -0.96, 0.86, 1.0, aspect, [0.95, 0.45, 0.2, 0.95]); } else if let Some(tpos) = target_pos { draw_text(&mut verts, &format!("AIM: [{:.2}, {:.2}, {:.2}]", tpos[0], tpos[1], tpos[2]), -0.96, 0.86, 1.0, aspect, [0.3, 0.9, 0.9, 0.9]); } else { draw_text(&mut verts, "AIM: [UNBOUNDED VOID]", -0.96, 0.86, 1.0, aspect, [0.6, 0.7, 0.8, 0.8]); }
            draw_text(&mut verts, "[E] PALETTE  [TAB] FREE  [P] ORTHO  [M] PLAYMODE  [.] FOCUS  [ESC] MENU  [F1] UI", -0.96, 0.80, 0.95, aspect, [0.9, 0.85, 0.4, 0.85]);
        }
        ActiveMenu::Edit => {
            add_quad(&mut verts, -1.0, -1.0, 1.0, 1.0, [0.03, 0.04, 0.06, 0.75]); add_quad(&mut verts, -0.566, -0.586, 0.566, 0.656, [0.25, 0.35, 0.50, 1.0]); add_quad(&mut verts, -0.56, -0.58, 0.56, 0.65, [0.10, 0.12, 0.16, 0.98]);
            draw_text_centered(&mut verts, "EDIT STUDIO - PALETTE & OCTREE (E)", 0.0, 0.58, 1.3, aspect, [1.0, 0.9, 0.2, 1.0]);
            let sw_w = 0.082; let sw_gap = 0.015; let sw_tot = 10.0 * sw_w + 9.0 * sw_gap; let s_start_x = -sw_tot / 2.0;
            for (i, &rgb) in hotbar_colors.iter().enumerate() {
                let sx0 = s_start_x + i as f32 * (sw_w + sw_gap); let sx1 = sx0 + sw_w;
                if selected_slot == i { add_quad(&mut verts, sx0 - 0.008, 0.422, sx1 + 0.008, 0.528, [1.0, 0.9, 0.1, 1.0]); } else { add_quad(&mut verts, sx0 - 0.004, 0.426, sx1 + 0.004, 0.524, [0.22, 0.24, 0.30, 1.0]); }
                add_quad(&mut verts, sx0, 0.43, sx1, 0.52, [rgb[0], rgb[1], rgb[2], 1.0]);
                let num_str = if i == 9 { "0".to_string() } else { format!("{}", i + 1) }; draw_text_centered(&mut verts, &num_str, (sx0 + sx1) / 2.0, 0.542, 1.0, aspect, [0.8, 0.8, 0.8, 0.9]);
            }
            let [cur_r, cur_g, cur_b] = hotbar_colors[selected_slot]; add_quad(&mut verts, 0.24, 0.18, 0.46, 0.37, [0.25, 0.28, 0.35, 1.0]); add_quad(&mut verts, 0.248, 0.188, 0.452, 0.362, [cur_r, cur_g, cur_b, 1.0]); draw_text_centered(&mut verts, "ACTIVE COLOR", 0.35, 0.39, 1.0, aspect, [0.85, 0.85, 0.85, 0.9]);
            for (lbl, val, y0, y1, bar_col) in [("R", cur_r, 0.32, 0.36, [0.90, 0.25, 0.25, 1.0]), ("G", cur_g, 0.25, 0.29, [0.25, 0.85, 0.30, 1.0]), ("B", cur_b, 0.18, 0.22, [0.25, 0.50, 0.95, 1.0])] {
                draw_text_centered(&mut verts, lbl, -0.37, (y0 + y1) / 2.0, 1.1, aspect, bar_col); add_quad(&mut verts, -0.32, y0, 0.18, y1, [0.18, 0.20, 0.25, 1.0]); let filled_x = -0.32 + val * 0.50; add_quad(&mut verts, -0.32, y0, filled_x, y1, bar_col); add_quad(&mut verts, filled_x - 0.010, y0 - 0.006, filled_x + 0.010, y1 + 0.006, [1.0, 1.0, 1.0, 1.0]);
            }
            draw_text_centered(&mut verts, "QUICK PALETTE CHIPS", 0.0, 0.135, 1.0, aspect, [0.75, 0.75, 0.8, 0.9]);
            let pw_w = 0.076; let pw_gap = 0.012; let pw_tot = 10.0 * pw_w + 9.0 * pw_gap; let pw_start_x = -pw_tot / 2.0;
            for (i, &rgb) in PRESET_SWATCHES.iter().enumerate() { let px0 = pw_start_x + i as f32 * (pw_w + pw_gap); add_quad(&mut verts, px0 - 0.003, 0.047, px0 + pw_w + 0.003, 0.113, [0.3, 0.3, 0.35, 1.0]); add_quad(&mut verts, px0, 0.05, px0 + pw_w, 0.11, [rgb[0], rgb[1], rgb[2], 1.0]); }
            add_quad(&mut verts, -0.40, -0.10, -0.22, -0.02, [0.35, 0.40, 0.55, 1.0]); draw_text_centered(&mut verts, "/ 2 (F)", -0.31, -0.06, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]); draw_text_centered(&mut verts, &format!("CURRENT: {}", size_str), 0.0, -0.06, 1.25, aspect, [1.0, 0.85, 0.2, 1.0]); add_quad(&mut verts, 0.22, -0.10, 0.40, -0.02, [0.35, 0.40, 0.55, 1.0]); draw_text_centered(&mut verts, "* 2 (R)", 0.31, -0.06, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]); add_quad(&mut verts, -0.22, -0.25, 0.22, -0.17, [0.20, 0.50, 0.30, 1.0]); draw_text_centered(&mut verts, "DONE (PRESS E)", 0.0, -0.21, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
        }
        ActiveMenu::Pause => {
            add_quad(&mut verts, -1.0, -1.0, 1.0, 1.0, [0.03, 0.04, 0.06, 0.80]); add_quad(&mut verts, -0.426, -0.726, 0.426, 0.726, [0.45, 0.45, 0.50, 1.0]); add_quad(&mut verts, -0.42, -0.72, 0.42, 0.72, [0.12, 0.13, 0.17, 0.98]);
            draw_text_centered(&mut verts, "PAUSE / SYSTEM MENU", 0.0, 0.60, 1.3, aspect, [0.95, 0.95, 0.95, 1.0]);
            add_quad(&mut verts, -0.30, 0.46, 0.30, 0.54, [0.20, 0.55, 0.75, 1.0]); draw_text_centered(&mut verts, ">> IMPORT 3D MODEL (GLB) <<", 0.0, 0.50, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, 0.36, 0.30, 0.44, [0.25, 0.35, 0.55, 1.0]); draw_text_centered(&mut verts, if play_mode == PlayMode::Flying { "PLAY MODE: FLYING (M)" } else { "PLAY MODE: REAL (M)" }, 0.0, 0.40, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, 0.26, 0.30, 0.34, [0.22, 0.40, 0.55, 1.0]); draw_text_centered(&mut verts, if is_ortho { "VIEW: ORTHOGRAPHIC (P)" } else { "VIEW: PERSPECTIVE (P)" }, 0.0, 0.30, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, 0.16, 0.30, 0.24, [0.35, 0.25, 0.50, 1.0]); draw_text_centered(&mut verts, &format!("WORLD: {} (F2)", world_type.name()), 0.0, 0.20, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, 0.06, 0.30, 0.14, [0.60, 0.30, 0.20, 1.0]); draw_text_centered(&mut verts, "CLEAR SCENE", 0.0, 0.10, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, -0.06, -0.02, 0.02, [0.25, 0.45, 0.35, 1.0]); draw_text_centered(&mut verts, "SAVE (F5)", -0.16, -0.02, 1.0, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, 0.02, -0.06, 0.30, 0.02, [0.35, 0.45, 0.25, 1.0]); draw_text_centered(&mut verts, "LOAD (F9)", 0.16, -0.02, 1.0, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, -0.16, 0.30, -0.08, [0.4, 0.35, 0.45, 1.0]); draw_text_centered(&mut verts, "CYCLE BACKGROUND COLOR", 0.0, -0.12, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, -0.28, 0.30, -0.20, [0.25, 0.40, 0.55, 1.0]); draw_text_centered(&mut verts, "CONTROLS", 0.0, -0.24, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, -0.42, 0.30, -0.34, [0.20, 0.55, 0.30, 1.0]); draw_text_centered(&mut verts, "RESUME (ESC)", 0.0, -0.38, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, -0.54, 0.30, -0.46, [0.55, 0.20, 0.20, 1.0]); draw_text_centered(&mut verts, "QUIT TO DESKTOP", 0.0, -0.50, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
        }
        ActiveMenu::Controls => {
            add_quad(&mut verts, -1.0, -1.0, 1.0, 1.0, [0.03, 0.04, 0.06, 0.95]);
            draw_text_centered(&mut verts, "CONTROLS", 0.0, 0.60, 1.5, aspect, [1.0, 1.0, 1.0, 1.0]);
            draw_text_centered(&mut verts, "LMB = ADD BLOCK", 0.0, 0.30, 1.1, aspect, [0.9, 0.9, 0.9, 1.0]);
            draw_text_centered(&mut verts, "RMB = REMOVE BLOCK", 0.0, 0.20, 1.1, aspect, [0.9, 0.9, 0.9, 1.0]);
            draw_text_centered(&mut verts, "MMB / C = PICK COLOR", 0.0, 0.10, 1.1, aspect, [0.9, 0.9, 0.9, 1.0]);
            draw_text_centered(&mut verts, "CTRL + WHEEL = ZOOM", 0.0, 0.0, 1.1, aspect, [0.9, 0.9, 0.9, 1.0]);
            draw_text_centered(&mut verts, "WHEEL = CHANGE COLOR", 0.0, -0.10, 1.1, aspect, [0.9, 0.9, 0.9, 1.0]);
            draw_text_centered(&mut verts, "TAB = FREE CAMERA", 0.0, -0.20, 1.1, aspect, [0.9, 0.9, 0.9, 1.0]);
            draw_text_centered(&mut verts, "M = TOGGLE REAL / FLYING", 0.0, -0.30, 1.1, aspect, [0.9, 0.9, 0.9, 1.0]);
            draw_text_centered(&mut verts, ". = FOCUS SCENE IN VIEWPORT", 0.0, -0.40, 1.1, aspect, [0.9, 0.9, 0.9, 1.0]);
            add_quad(&mut verts, -0.30, -0.66, 0.30, -0.56, [0.45, 0.22, 0.22, 1.0]); draw_text_centered(&mut verts, "BACK (ESC)", 0.0, -0.61, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
        }
        ActiveMenu::ImportParams => {
            add_quad(&mut verts, -1.0, -1.0, 1.0, 1.0, [0.02, 0.03, 0.05, 0.85]); add_quad(&mut verts, -0.426, -0.686, 0.426, 0.626, [0.30, 0.45, 0.65, 1.0]); add_quad(&mut verts, -0.42, -0.68, 0.42, 0.62, [0.08, 0.10, 0.14, 0.98]);
            draw_text_centered(&mut verts, "GLB VOXEL IMPORT SETTINGS", 0.0, 0.54, 1.25, aspect, [0.3, 0.9, 1.0, 1.0]);
            let file_label = glb_settings.selected_file.as_ref().and_then(|f| f.file_name()).map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "NO FILE SELECTED".into());
            let short_file = if file_label.len() > 18 { format!("{}...", &file_label[..15]) } else { file_label };
            add_quad(&mut verts, -0.38, 0.38, 0.20, 0.46, [0.05, 0.06, 0.09, 1.0]); draw_text(&mut verts, &format!("FILE: {}", short_file), -0.36, 0.42, 0.95, aspect, [0.85, 0.85, 0.4, 1.0]);
            add_quad(&mut verts, 0.22, 0.38, 0.38, 0.46, [0.25, 0.35, 0.50, 1.0]); draw_text_centered(&mut verts, "CHANGE", 0.30, 0.42, 0.95, aspect, [1.0, 1.0, 1.0, 1.0]);

            draw_text(&mut verts, "TARGET HEIGHT (BLOCKS):", -0.38, 0.29, 1.0, aspect, [0.8, 0.8, 0.8, 1.0]);
            add_quad(&mut verts, -0.38, 0.19, -0.28, 0.27, [0.25, 0.30, 0.40, 1.0]); draw_text_centered(&mut verts, "- 4", -0.33, 0.23, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            draw_text_centered(&mut verts, &format!("{:.0} BLOCKS", glb_settings.target_height), 0.0, 0.23, 1.1, aspect, [0.3, 0.9, 1.0, 1.0]);
            add_quad(&mut verts, 0.28, 0.19, 0.38, 0.27, [0.25, 0.30, 0.40, 1.0]); draw_text_centered(&mut verts, "+ 4", 0.33, 0.23, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);

            draw_text(&mut verts, "VOXEL RESOLUTION:", -0.38, 0.09, 1.0, aspect, [0.8, 0.8, 0.8, 1.0]);
            add_quad(&mut verts, -0.38, -0.01, -0.28, 0.07, [0.25, 0.30, 0.40, 1.0]); draw_text_centered(&mut verts, "/ 2", -0.33, 0.03, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            draw_text_centered(&mut verts, &format!("{:.3}", glb_settings.voxel_size), 0.0, 0.03, 1.1, aspect, [0.3, 0.9, 1.0, 1.0]);
            add_quad(&mut verts, 0.28, -0.01, 0.38, 0.07, [0.25, 0.30, 0.40, 1.0]); draw_text_centered(&mut verts, "* 2", 0.33, 0.03, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);

            draw_text(&mut verts, "MAX PALETTE SIZE:", -0.38, -0.11, 1.0, aspect, [0.8, 0.8, 0.8, 1.0]);
            add_quad(&mut verts, -0.38, -0.21, -0.28, -0.13, [0.25, 0.30, 0.40, 1.0]); draw_text_centered(&mut verts, "/ 2", -0.33, -0.17, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            draw_text_centered(&mut verts, &format!("{}", glb_settings.palette_size), 0.0, -0.17, 1.1, aspect, [0.3, 0.9, 1.0, 1.0]);
            add_quad(&mut verts, 0.28, -0.21, 0.38, -0.13, [0.25, 0.30, 0.40, 1.0]); draw_text_centered(&mut verts, "* 2", 0.33, -0.17, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);

            draw_text(&mut verts, "PLACEMENT ANCHOR:", -0.38, -0.29, 1.0, aspect, [0.8, 0.8, 0.8, 1.0]);
            add_quad(&mut verts, -0.38, -0.39, 0.38, -0.31, [0.18, 0.24, 0.34, 1.0]); draw_text_centered(&mut verts, if glb_settings.place_at_aim { "CROSSHAIR / RAYCAST AIM" } else { "AT PLAYER POSITION" }, 0.0, -0.35, 1.0, aspect, [1.0, 1.0, 1.0, 1.0]);

            let (est_count, est_mb) = glb_settings.estimate_cost();
            let is_safe = glb_settings.is_safe();
            let max_nodes = glb_settings.max_gpu_nodes;
            let cost_str = format!("EST: ~{} VOXELS ({:.0} MB SVO) | HW LIMIT: {}", format_voxel_count(est_count as usize), est_mb, format_voxel_count(max_nodes));
            let cost_col = if is_safe { [0.3, 0.9, 0.4, 1.0] } else { [0.95, 0.25, 0.2, 1.0] };
            draw_text_centered(&mut verts, &cost_str, 0.0, -0.42, 0.85, aspect, cost_col);

            let btn_col = if glb_settings.selected_file.is_some() && is_safe { [0.20, 0.60, 0.30, 1.0] } else { [0.35, 0.20, 0.20, 0.8] };
            add_quad(&mut verts, -0.38, -0.54, 0.38, -0.44, btn_col);
            draw_text_centered(&mut verts, if is_safe { "VOXELIZE & INSERT" } else { "TOO DENSE (REDUCE SETTINGS)" }, 0.0, -0.49, 1.0, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.38, -0.66, 0.38, -0.56, [0.45, 0.22, 0.22, 1.0]); draw_text_centered(&mut verts, "CANCEL (ESC)", 0.0, -0.61, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
        }
        ActiveMenu::Voxelizing => {
            add_quad(&mut verts, -1.0, -1.0, 1.0, 1.0, [0.02, 0.03, 0.05, 0.88]); add_quad(&mut verts, -0.446, -0.246, 0.446, 0.266, [0.25, 0.45, 0.70, 1.0]); add_quad(&mut verts, -0.44, -0.24, 0.44, 0.26, [0.08, 0.10, 0.15, 0.98]);
            draw_text_centered(&mut verts, "VOXELIZING 3D MODEL", 0.0, 0.18, 1.25, aspect, [0.3, 0.9, 1.0, 1.0]); draw_text_centered(&mut verts, progress_stage, 0.0, 0.09, 0.95, aspect, [0.85, 0.85, 0.9, 1.0]);
            add_quad(&mut verts, -0.384, -0.054, 0.384, 0.044, [0.20, 0.25, 0.35, 1.0]); add_quad(&mut verts, -0.38, -0.05, 0.38, 0.04, [0.04, 0.05, 0.07, 1.0]);
            let fill_w = 0.76 * progress_val.clamp(0.0, 1.0); if fill_w > 0.001 { add_quad(&mut verts, -0.38, -0.05, -0.38 + fill_w, 0.04, [0.20, 0.75, 0.90, 1.0]); }
            draw_text_centered(&mut verts, &format!("{:.0}%", (progress_val * 100.0).clamp(0.0, 100.0)), 0.0, -0.10, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]); draw_text_centered(&mut verts, "SVO GENERATOR RUNNING", 0.0, -0.18, 0.85, aspect, [0.5, 0.8, 0.6, 0.9]);
        }
    }
    verts
}

struct State {
    surface: wgpu::Surface<'static>, device: wgpu::Device, queue: wgpu::Queue, config: wgpu::SurfaceConfiguration, size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline, ui_pipeline: wgpu::RenderPipeline, ui_vertex_buffer: wgpu::Buffer, ui_vertices_count: u32,
    camera_buffer: wgpu::Buffer, svo_buffer: wgpu::Buffer, svo_capacity: usize, svo_bind_group: wgpu::BindGroup, svo_bind_group_layout: wgpu::BindGroupLayout,
    palette_buffer: wgpu::Buffer, window: Arc<Window>, camera: Camera, input: InputState, octree: Octree,
    play_mode: PlayMode, velocity: Vec3, selected_slot: usize, hotbar_colors: [[f32; 3]; 10], palette: Arc<RwLock<Palette>>, active_menu: ActiveMenu,
    cursor_pos: [f32; 2], active_slider: Option<usize>, edit_size: f32, last_target: Option<[f32; 3]>, cursor_free: bool, gimbal_dragging: bool, gimbal_drag_moved: bool, prev_cursor_pos: [f32; 2],
    glb_settings: GlbImportSettings, voxelize_rx: Option<mpsc::Receiver<VoxelizeMsg>>, voxelize_progress: f32, voxelize_stage: String, collider: PlayerCollider,
    hide_ui: bool, bg_color: [f32; 3], world_type: WorldType, seed: u32, cube_edits: Vec<CubeEdit>, error_banner: Option<(String, Instant)>,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let mut size = window.inner_size();
        if size.width == 0 || size.height == 0 { size = winit::dpi::PhysicalSize::new(1280, 720); }
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::default(), compatible_surface: Some(&surface), force_fallback_adapter: false, apply_limit_buckets: Default::default() }).await.unwrap();
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { required_limits: adapter.limits(), ..Default::default() }).await.unwrap();
        let mut config = surface.get_default_config(&adapter, size.width, size.height).unwrap();
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);

        let camera = Camera { position: Vec3::new(0.0, 16.0, 48.0), yaw: -std::f32::consts::FRAC_PI_2, pitch: -0.25, is_ortho: false, ortho_size: 48.0 };
        let aspect = if config.height > 0 { config.width as f32 / config.height as f32 } else { 1.0 };
        let vp = camera.view_proj(aspect);

        let camera_uniform = CameraUniform {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            camera_pos: camera.position.to_array(),
            show_borders: 0.0,
            world_min: WORLD_MIN.to_array(),
            world_size: WORLD_SIZE,
            is_ortho: if camera.is_ortho { 1.0 } else { 0.0 },
            ortho_size: camera.ortho_size,
            screen_size: [config.width as f32, config.height as f32],
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let hotbar_colors = PRESET_SWATCHES;
        let palette = Arc::new(RwLock::new(hotbar_colors.to_vec()));

        let mut pal_vec4 = vec![[0.0; 4]; 8192];
        for (i, c) in hotbar_colors.iter().enumerate() { pal_vec4[i] = [c[0], c[1], c[2], 1.0]; }
        let palette_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Palette Storage Buffer"),
            contents: bytemuck::cast_slice(&pal_vec4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let world_type = WorldType::Flat;
        let seed = 42;
        let cube_edits = Vec::new();
        let octree = generate_world_terrain(world_type, seed, &cube_edits);

        let svo_capacity = (octree.nodes.len() * 2).max(16384) * std::mem::size_of::<OctreeNode>();
        let svo_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SVO Storage Buffer"),
            size: svo_capacity as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&svo_buffer, 0, bytemuck::cast_slice(&octree.nodes));

        let svo_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                },
            ],
            label: Some("SVO Layout"),
        });

        let svo_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &svo_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: svo_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: palette_buffer.as_entire_binding() },
            ],
            label: Some("SVO Bind Group"),
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SVO Pipeline Layout"),
            bind_group_layouts: &[Some(&svo_bind_group_layout)],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SVO Ray-Marching Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let ui_shader = device.create_shader_module(wgpu::include_wgsl!("ui.wgsl"));
        let ui_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[], immediate_size: 0 });
        let ui_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UI Pipeline"),
            layout: Some(&ui_pipeline_layout),
            vertex: wgpu::VertexState { module: &ui_shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[Some(UIVertex::desc())] },
            fragment: Some(wgpu::FragmentState { module: &ui_shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let mut glb_settings = GlbImportSettings::default();
        glb_settings.max_gpu_nodes = (device.limits().max_storage_buffer_binding_size as usize / 16).min(35_000_000);

        let ui_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UI Buffer"),
            size: (65536 * std::mem::size_of::<UIVertex>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut app_state = Self {
            window, surface, device, queue, config, size, render_pipeline, ui_pipeline, ui_vertex_buffer, ui_vertices_count: 0,
            camera_buffer, svo_buffer, svo_capacity, svo_bind_group, svo_bind_group_layout, palette_buffer, camera,
            input: InputState::default(), octree, play_mode: PlayMode::Flying, velocity: Vec3::ZERO, selected_slot: 0,
            hotbar_colors, palette, active_menu: ActiveMenu::None, cursor_pos: [0.0, 0.0], active_slider: None, edit_size: 1.0,
            last_target: None, cursor_free: false, gimbal_dragging: false, gimbal_drag_moved: false, prev_cursor_pos: [0.0, 0.0],
            glb_settings, voxelize_rx: None, voxelize_progress: 0.0, voxelize_stage: String::new(), collider: PlayerCollider::default(),
            hide_ui: false, bg_color: [0.12, 0.14, 0.18], world_type, seed, cube_edits, error_banner: None,
        };
        app_state.sync_palette_buffer();
        app_state.update_ui();
        app_state
    }

    pub fn prompt_native_file_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new().add_filter("3D Models", &["glb", "gltf"]).pick_file() {
            self.glb_settings.selected_file = Some(path);
            self.set_menu(ActiveMenu::ImportParams);
        }
    }

    pub fn start_nonblocking_voxelization(&mut self) {
        let path = match self.glb_settings.selected_file.clone() { Some(p) => p, None => return };
        let origin = if self.glb_settings.place_at_aim {
            self.last_target.map(Vec3::from).unwrap_or_else(|| self.camera.position + self.camera.forward() * 16.0)
        } else {
            self.camera.position - Vec3::new(0.0, self.collider.eye_offset, 0.0)
        };
        let voxel_size = self.glb_settings.voxel_size.max(MIN_VOXEL_SIZE);
        let target_height = self.glb_settings.target_height;
        let palette_size = self.glb_settings.palette_size;
        let (tx, rx) = mpsc::channel();
        self.voxelize_rx = Some(rx);
        self.voxelize_progress = 0.0;
        self.voxelize_stage = "INITIALIZING THREADS...".into();
        self.set_menu(ActiveMenu::Voxelizing);
        let pal_arc = Arc::clone(&self.palette);
        let max_nodes = self.glb_settings.max_gpu_nodes;
        std::thread::spawn(move || run_background_voxelization(path, origin, voxel_size, target_height, palette_size, pal_arc, max_nodes, tx));
    }

    pub fn sync_palette_buffer(&self) {
        let pal = self.palette.read().unwrap();
        let mut pal_vec4 = vec![[0.0f32; 4]; pal.len().max(10)];
        for (i, c) in pal.iter().enumerate() {
            pal_vec4[i] = [c[0], c[1], c[2], 1.0];
        }
        self.queue.write_buffer(&self.palette_buffer, 0, bytemuck::cast_slice(&pal_vec4));
    }

    pub fn sync_svo_buffer(&mut self) {
        let req_bytes = self.octree.nodes.len() * std::mem::size_of::<OctreeNode>();
        if req_bytes > self.svo_capacity {
            self.svo_capacity = (req_bytes * 2).max(16384);
            self.svo_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SVO Storage Buffer"),
                size: self.svo_capacity as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.svo_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.svo_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.camera_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.svo_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.palette_buffer.as_entire_binding() },
                ],
                label: Some("SVO Reallocated Bind Group"),
            });
        }
        self.queue.write_buffer(&self.svo_buffer, 0, bytemuck::cast_slice(&self.octree.nodes));
    }

    pub fn focus_on_scene(&mut self) {
        let mut min_bound = Vec3::splat(f32::MAX);
        let mut max_bound = Vec3::splat(f32::MIN);
        let has_voxels = self.octree.compute_voxel_bounds(self.octree.root_index as usize, WORLD_MIN, WORLD_SIZE, &mut min_bound, &mut max_bound);

        let (center, radius, half_extents) = if has_voxels {
            let c = (min_bound + max_bound) * 0.5;
            let extents = max_bound - min_bound;
            let half = extents * 0.5;
            (c, half.length().max(1.0), half)
        } else {
            (Vec3::ZERO, 16.0, Vec3::splat(8.0))
        };

        let aspect = if self.config.height > 0 { self.size.width as f32 / self.size.height as f32 } else { 1.0 };
        let fov_y_rad = VERTICAL_FOV_DEGREES.to_radians();
        let half_fov_y = fov_y_rad * 0.5;
        let half_fov_x = (half_fov_y.tan() * aspect).atan();

        let dist_y = half_extents.y / half_fov_y.tan();
        let dist_x = half_extents.x / half_fov_x.tan();
        let required_dist = dist_y.max(dist_x).max(radius) * 1.15;

        self.camera.position = center - self.camera.forward() * required_dist.clamp(2.0, 50000.0);
        self.camera.ortho_size = (half_extents.y.max(half_extents.x / aspect) * 2.2).clamp(2.0, 50000.0);
        self.velocity = Vec3::ZERO;
        self.update_camera_buffer();
        self.update_ui();
    }

    pub fn toggle_projection(&mut self) {
        self.camera.is_ortho = !self.camera.is_ortho;
        self.update_camera_buffer();
        self.update_ui();
    }

    pub fn update_camera_buffer(&self) {
        let aspect = if self.config.height > 0 { self.config.width as f32 / self.config.height as f32 } else { 1.0 };
        let vp = self.camera.view_proj(aspect);
        let camera_uniform = CameraUniform {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            camera_pos: self.camera.position.to_array(),
            show_borders: if self.cursor_free || self.active_menu == ActiveMenu::Edit { 1.0 } else { 0.0 },
            world_min: WORLD_MIN.to_array(),
            world_size: WORLD_SIZE,
            is_ortho: if self.camera.is_ortho { 1.0 } else { 0.0 },
            ortho_size: self.camera.ortho_size,
            screen_size: [self.config.width as f32, self.config.height as f32],
        };
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));
    }

    pub fn set_cursor_mode(&mut self, free: bool) {
        self.cursor_free = free;
        if free || self.active_menu != ActiveMenu::None {
            let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::None);
            self.window.set_cursor_visible(true);
            self.input = InputState::default();
        } else {
            let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::Locked).or_else(|_| self.window.set_cursor_grab(winit::window::CursorGrabMode::Confined));
            self.window.set_cursor_visible(false);
        }
        self.update_camera_buffer();
        self.update_ui();
    }

    pub fn set_menu(&mut self, menu: ActiveMenu) {
        self.active_menu = menu;
        self.active_slider = None;
        self.gimbal_dragging = false;
        self.set_cursor_mode(self.active_menu != ActiveMenu::None);
    }

    pub fn save_game(&self, filename: &str) -> std::io::Result<()> {
        std::fs::write(filename, serde_json::to_string_pretty(&SaveData {
            player_pos: self.camera.position.to_array(), camera_yaw: self.camera.yaw, camera_pitch: self.camera.pitch,
            is_ortho: self.camera.is_ortho, ortho_size: self.camera.ortho_size, play_mode: self.play_mode,
            world_type: self.world_type, seed: self.seed, hotbar_colors: self.hotbar_colors,
            palette: self.palette.read().unwrap().clone(), cube_edits: self.cube_edits.clone(),
        })?)?;
        Ok(())
    }

    pub fn load_game(&mut self, filename: &str) -> std::io::Result<()> {
        let data: SaveData = serde_json::from_str(&std::fs::read_to_string(filename)?)?;
        self.camera.position = Vec3::from_array(data.player_pos);
        self.camera.yaw = data.camera_yaw;
        self.camera.pitch = data.camera_pitch;
        self.camera.is_ortho = data.is_ortho;
        self.camera.ortho_size = if data.ortho_size > 0.1 { data.ortho_size } else { 36.0 };
        self.play_mode = data.play_mode;
        self.velocity = Vec3::ZERO;
        self.hotbar_colors = data.hotbar_colors;
        *self.palette.write().unwrap() = data.palette;
        self.world_type = data.world_type;
        self.seed = data.seed;
        self.cube_edits = data.cube_edits;
        self.octree = generate_world_terrain(self.world_type, self.seed, &self.cube_edits);
        self.sync_svo_buffer();
        self.sync_palette_buffer();
        self.update_camera_buffer();
        self.update_ui();
        Ok(())
    }

    pub fn clear_all_blocks(&mut self) {
        self.cube_edits.clear();
        self.octree = Octree::new();
        self.sync_svo_buffer();
        self.update_ui();
    }

    pub fn cycle_world_generator(&mut self) {
        self.world_type = self.world_type.next();
        self.cube_edits.clear();
        self.octree = generate_world_terrain(self.world_type, self.seed, &self.cube_edits);
        self.sync_svo_buffer();
        self.update_ui();
    }

    pub fn scale_voxel_size(&mut self, multiply: bool) {
        if multiply { self.edit_size = (self.edit_size * 2.0).min(64.0); }
        else { self.edit_size = (self.edit_size * 0.5).max(MIN_VOXEL_SIZE); }
        self.update_ui();
    }

    pub fn toggle_play_mode(&mut self) {
        self.play_mode = if self.play_mode == PlayMode::Real { PlayMode::Flying } else { PlayMode::Real };
        self.velocity = Vec3::ZERO;
        self.update_ui();
    }

    fn get_or_create_material(&self, color: [f32; 3]) -> u16 {
        let mut pal = self.palette.write().unwrap();
        for (i, &c) in pal.iter().enumerate() {
            if (c[0] - color[0]).abs() < 0.005 && (c[1] - color[1]).abs() < 0.005 && (c[2] - color[2]).abs() < 0.005 {
                return (i + 1) as u16;
            }
        }
        pal.push(color);
        let id = pal.len() as u16;
        drop(pal);
        self.sync_palette_buffer();
        id
    }

    fn modify_cube(&mut self, pos: Vec3, material: u16, size: f32) {
        self.cube_edits.push(CubeEdit { pos: pos.to_array(), size, material });
        let gx = (((pos.x - WORLD_MIN.x) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
        let gy = (((pos.y - WORLD_MIN.y) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
        let gz = (((pos.z - WORLD_MIN.z) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
        let grid_size = ((size / WORLD_SIZE) * GRID_RES as f32).round().max(1.0) as u32;
        let depth = MAX_DEPTH - grid_size.trailing_zeros().min(MAX_DEPTH as u32) as u8;
        self.octree.insert_cube(gx, gy, gz, depth, material, true);
        self.sync_svo_buffer();
    }

    fn update_ui(&mut self) {
        let aspect = if self.config.height > 0 { self.config.width as f32 / self.config.height as f32 } else { 1.0 };
        let total_voxels = self.octree.count_occupied_voxels(self.octree.root_index as usize);
        let error_msg = self.error_banner.as_ref().map(|(msg, _)| msg.as_str());

        let verts = build_ui_vertices(
            self.selected_slot, self.active_menu, &self.hotbar_colors, self.play_mode, self.camera.is_ortho, self.world_type,
            self.edit_size, self.last_target, aspect, self.camera.forward(), self.camera.right(), self.camera.up(), self.cursor_free,
            &self.glb_settings, self.voxelize_progress, &self.voxelize_stage, self.camera.view_proj(aspect), total_voxels, error_msg,
        );
        self.ui_vertices_count = verts.len() as u32;
        self.queue.write_buffer(&self.ui_vertex_buffer, 0, bytemuck::cast_slice(&verts));
    }

    fn player_collides_at(&self, foot_pos: Vec3, collider: &PlayerCollider) -> bool {
        let r = collider.radius; let h = collider.height;
        let min = foot_pos - Vec3::new(r, 0.0, r);
        let max = foot_pos + Vec3::new(r, h, r);
        check_octree_aabb(&self.octree, self.octree.root_index as usize, WORLD_MIN, WORLD_SIZE, min, max)
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.update_camera_buffer();
        }
    }

    fn update(&mut self, dt: f32) {
        if let Some((_, time)) = self.error_banner {
            if time.elapsed().as_secs() > 7 {
                self.error_banner = None;
                self.update_ui();
            }
        }

        let mut messages = Vec::new();
        if let Some(ref rx) = self.voxelize_rx { while let Ok(msg) = rx.try_recv() { messages.push(msg); } }
        for msg in messages {
            match msg {
                VoxelizeMsg::Progress { percent, stage } => {
                    self.voxelize_progress = percent;
                    self.voxelize_stage = stage;
                    self.update_ui();
                }
                VoxelizeMsg::Done(Ok(voxels)) => {
                    for (x, y, z, depth, mat) in voxels {
                        self.octree.insert_cube(x, y, z, depth, mat, false);
                    }
                    self.octree.collapse(self.octree.root_index as usize);
                    self.sync_svo_buffer();
                    self.sync_palette_buffer();
                    self.voxelize_rx = None;
                    self.set_menu(ActiveMenu::None);
                    self.focus_on_scene();
                    return;
                }
                VoxelizeMsg::Done(Err(err)) => {
                    eprintln!("GLB voxelization failed: {}", err);
                    self.error_banner = Some((err, Instant::now()));
                    self.voxelize_rx = None;
                    self.set_menu(ActiveMenu::ImportParams);
                    return;
                }
            }
        }

        if self.active_menu != ActiveMenu::None { return; }

        let (sin_y, cos_y) = self.camera.yaw.sin_cos();
        let forward = Vec3::new(cos_y, 0.0, sin_y).normalize();
        let right = Vec3::new(-sin_y, 0.0, cos_y).normalize();
        let mut movement = Vec3::ZERO;
        if self.input.forward { movement += forward; }
        if self.input.backward { movement -= forward; }
        if self.input.right { movement += right; }
        if self.input.left { movement -= right; }

        match self.play_mode {
            PlayMode::Flying => {
                let speed = 28.0;
                if self.input.up { movement.y += 1.0; }
                if self.input.down { movement.y -= 1.0; }
                if movement.length_squared() > 0.0 { movement = movement.normalize(); }
                self.camera.position += movement * speed * dt;
            }
            PlayMode::Real => {
                let foot_pos = self.camera.position - Vec3::new(0.0, self.collider.eye_offset, 0.0);
                if self.player_collides_at(foot_pos, &self.collider) {
                    self.camera.position.y += 0.25;
                    self.velocity.y = 0.0;
                }
                let walk_speed = 7.0;
                if movement.length_squared() > 0.0 { movement = movement.normalize(); }
                let dx = movement.x * walk_speed * dt;
                if !self.player_collides_at(foot_pos + Vec3::new(dx, 0.0, 0.0), &self.collider) { self.camera.position.x += dx; }
                let dz = movement.z * walk_speed * dt;
                if !self.player_collides_at(foot_pos + Vec3::new(0.0, 0.0, dz), &self.collider) { self.camera.position.z += dz; }
                self.velocity.y -= 38.0 * dt;
                let on_ground = self.player_collides_at(foot_pos - Vec3::new(0.0, 0.08, 0.0), &self.collider);
                if self.input.up && on_ground { self.velocity.y = 11.5; }
                let total_dy = self.velocity.y * dt;
                let step_count = ((total_dy.abs() / 0.08).ceil() as i32).max(1);
                let step_dy = total_dy / step_count as f32;
                for _ in 0..step_count {
                    if self.player_collides_at(foot_pos + Vec3::new(0.0, step_dy, 0.0), &self.collider) {
                        self.velocity.y = 0.0;
                        break;
                    } else {
                        self.camera.position.y += step_dy;
                    }
                }
            }
        }

        let dir = self.camera.forward();
        let hit = raycast_octree(&self.octree, self.camera.position, dir, 300.0);
        let s = self.edit_size;
        if let Some(ref h) = hit {
            let p = if h.normal.x.abs() > 0.5 || h.normal.y.abs() > 0.5 || h.normal.z.abs() > 0.5 {
                h.voxel_min + h.normal * s
            } else {
                h.hit_pos
            };
            self.last_target = Some([(p.x / s).floor() * s, (p.y / s).floor() * s, (p.z / s).floor() * s]);
        } else {
            self.last_target = None;
        }

        if !self.cursor_free && (self.input.action_add || self.input.action_remove || self.input.action_pick) {
            if let Some(ref h) = hit {
                if self.input.action_pick {
                    let pal = self.palette.read().unwrap();
                    if let Some(&color) = pal.get((h.material - 1) as usize) {
                        self.hotbar_colors[self.selected_slot] = color;
                        drop(pal);
                        self.update_ui();
                    }
                } else if self.input.action_remove {
                    let p = h.hit_pos - h.normal * (MIN_VOXEL_SIZE * 0.5);
                    self.modify_cube(Vec3::new((p.x / s).floor() * s, (p.y / s).floor() * s, (p.z / s).floor() * s), 0, s);
                } else if self.input.action_add {
                    let p = h.voxel_min + h.normal * s;
                    let mat_id = self.get_or_create_material(self.hotbar_colors[self.selected_slot]);
                    self.modify_cube(Vec3::new((p.x / s).floor() * s, (p.y / s).floor() * s, (p.z / s).floor() * s), mat_id, s);
                }
            } else if self.input.action_add {
                let p = self.camera.position + dir * (s * 2.0).clamp(8.0, 64.0);
                let mat_id = self.get_or_create_material(self.hotbar_colors[self.selected_slot]);
                self.modify_cube(Vec3::new((p.x / s).floor() * s, (p.y / s).floor() * s, (p.z / s).floor() * s), mat_id, s);
            }
            self.input.action_add = false;
            self.input.action_remove = false;
            self.input.action_pick = false;
        }

        self.update_camera_buffer();
        self.update_ui();
    }

    fn render(&mut self) {
        let mut surface_texture = self.surface.get_current_texture();
        if matches!(surface_texture, wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost) {
            self.surface.configure(&self.device, &self.config);
            surface_texture = self.surface.get_current_texture();
        }
        let frame = match surface_texture {
            wgpu::CurrentSurfaceTexture::Success(frame) | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            _ => return,
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut svo_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SVO Ray-Marching Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.bg_color[0] as f64, g: self.bg_color[1] as f64, b: self.bg_color[2] as f64, a: 1.0
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            svo_pass.set_pipeline(&self.render_pipeline);
            svo_pass.set_bind_group(0, &self.svo_bind_group, &[]);
            svo_pass.draw(0..3, 0..1);
        }

        if !self.hide_ui {
            let mut ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            ui_pass.set_pipeline(&self.ui_pipeline);
            ui_pass.set_vertex_buffer(0, self.ui_vertex_buffer.slice(..));
            ui_pass.draw(0..self.ui_vertices_count, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(frame);
    }
}

struct App { state: Option<State>, last_frame: Instant }

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            #[allow(unused_mut)]
            let mut window_attributes = Window::default_attributes()
                .with_title("Voxel Studio - SVO Ray-Marching Engine")
                .with_inner_size(LogicalSize::new(1280.0, 720.0))
                .with_visible(true);
            #[cfg(target_os = "linux")] {
                window_attributes = WindowAttributesExtWayland::with_name(window_attributes, "octree_voxels", "octree_voxels");
                window_attributes = WindowAttributesExtX11::with_name(window_attributes, "octree_voxels", "octree_voxels");
            }
            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
            let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Confined));
            window.set_cursor_visible(false);
            let mut state = pollster::block_on(State::new(Arc::clone(&window)));
            state.render();
            self.state = Some(state);
            self.last_frame = Instant::now();
            window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let Some(state) = self.state.as_mut() {
            let aspect = if state.config.height > 0 { state.size.width as f32 / state.size.height as f32 } else { 1.0 };
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::DroppedFile(path_buf) => {
                    state.glb_settings.selected_file = Some(path_buf);
                    state.set_menu(ActiveMenu::ImportParams);
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let ndc_x = (position.x as f32 / state.size.width as f32) * 2.0 - 1.0;
                    let ndc_y = 1.0 - (position.y as f32 / state.size.height as f32) * 2.0;
                    state.cursor_pos = [ndc_x, ndc_y];
                    if state.gimbal_dragging {
                        let dx = ndc_x - state.prev_cursor_pos[0];
                        let dy = ndc_y - state.prev_cursor_pos[1];
                        if dx.abs() > 0.0005 || dy.abs() > 0.0005 {
                            state.gimbal_drag_moved = true;
                            state.camera.yaw += dx * 3.8;
                            state.camera.pitch = (state.camera.pitch - dy * 3.8).clamp(-1.56, 1.56);
                            state.update_camera_buffer();
                            state.update_ui();
                        }
                    } else if state.active_menu == ActiveMenu::Edit {
                        if let Some(channel) = state.active_slider {
                            state.hotbar_colors[state.selected_slot][channel] = ((ndc_x - -0.32) / (0.18 - -0.32)).clamp(0.0, 1.0);
                            state.update_ui();
                        }
                    }
                    state.prev_cursor_pos = [ndc_x, ndc_y];
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let step = match delta {
                        MouseScrollDelta::LineDelta(_, y) => if y > 0.0 { -1 } else if y < 0.0 { 1 } else { 0 },
                        MouseScrollDelta::PixelDelta(pos) => if pos.y > 0.0 { -1 } else if pos.y < 0.0 { 1 } else { 0 },
                    };
                    if step != 0 {
                        if state.input.ctrl_pressed {
                            if state.camera.is_ortho {
                                state.camera.ortho_size = (state.camera.ortho_size * if step < 0 { 0.88 } else { 1.14 }).max(0.0001);
                                state.update_camera_buffer();
                                state.update_ui();
                            } else {
                                state.camera.position += state.camera.forward() * (if step < 0 { 2.0 } else { -2.0 });
                                state.update_camera_buffer();
                                state.update_ui();
                            }
                        } else {
                            state.selected_slot = (state.selected_slot as i32 + step).rem_euclid(10) as usize;
                            state.update_ui();
                        }
                    }
                }
                WindowEvent::KeyboardInput { event: key_event, .. } => {
                    let is_pressed = key_event.state == ElementState::Pressed;
                    if key_event.physical_key == PhysicalKey::Code(KeyCode::ControlLeft) || key_event.physical_key == PhysicalKey::Code(KeyCode::ControlRight) {
                        state.input.ctrl_pressed = is_pressed;
                    }

                    let is_dot_key = match &key_event.logical_key {
                        winit::keyboard::Key::Character(s) => s == "." || s == ",",
                        _ => key_event.physical_key == PhysicalKey::Code(KeyCode::Period)
                            || key_event.physical_key == PhysicalKey::Code(KeyCode::NumpadDecimal),
                    };

                    let is_m_key = match &key_event.logical_key {
                        winit::keyboard::Key::Character(s) => s.eq_ignore_ascii_case("m"),
                        _ => key_event.physical_key == PhysicalKey::Code(KeyCode::KeyM)
                            || key_event.physical_key == PhysicalKey::Code(KeyCode::Semicolon),
                    };

                    if is_pressed {
                        if is_dot_key { state.focus_on_scene(); return; }
                        if is_m_key { state.toggle_play_mode(); return; }
                        if key_event.physical_key == PhysicalKey::Code(KeyCode::Tab) {
                            let new_free = !state.cursor_free;
                            state.set_cursor_mode(new_free);
                            return;
                        }
                        if key_event.physical_key == PhysicalKey::Code(KeyCode::KeyP) || key_event.physical_key == PhysicalKey::Code(KeyCode::Numpad5) {
                            state.toggle_projection();
                            return;
                        }
                        if state.camera.is_ortho {
                            match key_event.physical_key {
                                PhysicalKey::Code(KeyCode::NumpadAdd) | PhysicalKey::Code(KeyCode::Equal) => {
                                    state.camera.ortho_size = (state.camera.ortho_size * 0.85).clamp(2.0, 50000.0);
                                    state.update_camera_buffer(); state.update_ui(); return;
                                }
                                PhysicalKey::Code(KeyCode::NumpadSubtract) | PhysicalKey::Code(KeyCode::Minus) => {
                                    state.camera.ortho_size = (state.camera.ortho_size * 1.15).clamp(2.0, 50000.0);
                                    state.update_camera_buffer(); state.update_ui(); return;
                                }
                                _ => {}
                            }
                        }
                        if key_event.physical_key == PhysicalKey::Code(KeyCode::KeyE) {
                            if state.active_menu == ActiveMenu::Edit { state.set_menu(ActiveMenu::None); }
                            else if state.active_menu == ActiveMenu::None { state.set_menu(ActiveMenu::Edit); }
                            return;
                        }
                        if key_event.physical_key == PhysicalKey::Code(KeyCode::F1) {
                            state.hide_ui = !state.hide_ui;
                            state.update_ui();
                            return;
                        }
                        if key_event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                            match state.active_menu {
                                ActiveMenu::None | ActiveMenu::ImportParams | ActiveMenu::Controls => state.set_menu(ActiveMenu::Pause),
                                ActiveMenu::Voxelizing => {},
                                _ => state.set_menu(ActiveMenu::None),
                            }
                            return;
                        }
                        match key_event.physical_key {
                            PhysicalKey::Code(KeyCode::F2) => state.cycle_world_generator(),
                            PhysicalKey::Code(KeyCode::F5) => { let _ = state.save_game("world_save.json"); },
                            PhysicalKey::Code(KeyCode::F9) => { let _ = state.load_game("world_save.json"); },
                            PhysicalKey::Code(KeyCode::KeyF) => state.scale_voxel_size(false),
                            PhysicalKey::Code(KeyCode::KeyR) => state.scale_voxel_size(true),
                            PhysicalKey::Code(KeyCode::Digit1) => { state.selected_slot = 0; state.update_ui(); },
                            PhysicalKey::Code(KeyCode::Digit2) => { state.selected_slot = 1; state.update_ui(); },
                            PhysicalKey::Code(KeyCode::Digit3) => { state.selected_slot = 2; state.update_ui(); },
                            PhysicalKey::Code(KeyCode::Digit4) => { state.selected_slot = 3; state.update_ui(); },
                            PhysicalKey::Code(KeyCode::Digit5) => { state.selected_slot = 4; state.update_ui(); },
                            PhysicalKey::Code(KeyCode::Digit6) => { state.selected_slot = 5; state.update_ui(); },
                            PhysicalKey::Code(KeyCode::Digit7) => { state.selected_slot = 6; state.update_ui(); },
                            PhysicalKey::Code(KeyCode::Digit8) => { state.selected_slot = 7; state.update_ui(); },
                            PhysicalKey::Code(KeyCode::Digit9) => { state.selected_slot = 8; state.update_ui(); },
                            PhysicalKey::Code(KeyCode::Digit0) => { state.selected_slot = 9; state.update_ui(); },
                            PhysicalKey::Code(KeyCode::KeyC) if state.active_menu == ActiveMenu::None && !state.cursor_free => state.input.action_pick = true,
                            _ => {}
                        }
                    }
                    if state.active_menu == ActiveMenu::None {
                        match key_event.physical_key {
                            PhysicalKey::Code(KeyCode::KeyW) => state.input.forward = is_pressed,
                            PhysicalKey::Code(KeyCode::KeyS) => state.input.backward = is_pressed,
                            PhysicalKey::Code(KeyCode::KeyA) => state.input.left = is_pressed,
                            PhysicalKey::Code(KeyCode::KeyD) => state.input.right = is_pressed,
                            PhysicalKey::Code(KeyCode::Space) => state.input.up = is_pressed,
                            PhysicalKey::Code(KeyCode::ShiftLeft) => state.input.down = is_pressed,
                            _ => {}
                        }
                    }
                }
                WindowEvent::MouseInput { state: element_state, button, .. } => {
                    let [mx, my] = state.cursor_pos;
                    if button == MouseButton::Left {
                        if element_state == ElementState::Pressed {
                            if state.cursor_free || state.active_menu != ActiveMenu::None {
                                let (bx0, by0, bx1, by1) = get_focus_button_bounds(aspect);
                                if mx >= bx0 && mx <= bx1 && my >= by0 && my <= by1 { state.focus_on_scene(); return; }
                                if ((mx - GIZMO_CENTER_X) * aspect).powi(2) + (my - GIZMO_CENTER_Y).powi(2) <= (GIZMO_RADIUS + 0.02).powi(2) {
                                    state.gimbal_dragging = true;
                                    state.gimbal_drag_moved = false;
                                    return;
                                }
                            }
                        } else if element_state == ElementState::Released && state.gimbal_dragging {
                            state.gimbal_dragging = false;
                            if !state.gimbal_drag_moved {
                                let right = state.camera.right();
                                let up = state.camera.up();
                                let mut sorted_axes = get_gizmo_axes();
                                sorted_axes.sort_by(|a, b| b.dir.dot(state.camera.forward()).partial_cmp(&a.dir.dot(state.camera.forward())).unwrap_or(std::cmp::Ordering::Equal));
                                for ax in sorted_axes {
                                    if ((mx - (GIZMO_CENTER_X + (ax.dir.dot(right) * GIZMO_RADIUS) / aspect)) * aspect).powi(2) + (my - (GIZMO_CENTER_Y + (ax.dir.dot(up) * GIZMO_RADIUS))).powi(2) <= 0.001225 {
                                        state.camera.yaw = ax.yaw;
                                        state.camera.pitch = ax.pitch;
                                        state.camera.is_ortho = true;
                                        state.update_camera_buffer();
                                        state.update_ui();
                                        return;
                                    }
                                }
                            }
                            return;
                        }
                    }
                    match state.active_menu {
                        ActiveMenu::Edit => {
                            if button == MouseButton::Left && element_state == ElementState::Pressed {
                                let sw_w = 0.082; let sw_gap = 0.015; let sw_tot = 10.0 * sw_w + 9.0 * sw_gap; let s_start_x = -sw_tot / 2.0;
                                for i in 0..10 { if mx >= s_start_x + i as f32 * (sw_w + sw_gap) && mx <= s_start_x + i as f32 * (sw_w + sw_gap) + sw_w && my >= 0.43 && my <= 0.52 { state.selected_slot = i; state.update_ui(); return; } }
                                if mx >= -0.34 && mx <= 0.20 {
                                    state.active_slider = if my >= 0.30 && my <= 0.38 { Some(0) } else if my >= 0.23 && my <= 0.31 { Some(1) } else if my >= 0.16 && my <= 0.24 { Some(2) } else { None };
                                    if let Some(channel) = state.active_slider { state.hotbar_colors[state.selected_slot][channel] = ((mx - -0.32) / (0.18 - -0.32)).clamp(0.0, 1.0); state.update_ui(); return; }
                                }
                                let pw_w = 0.076; let pw_gap = 0.012; let pw_tot = 10.0 * pw_w + 9.0 * pw_gap; let pw_start_x = -pw_tot / 2.0;
                                for (i, &preset_col) in PRESET_SWATCHES.iter().enumerate() { if mx >= pw_start_x + i as f32 * (pw_w + pw_gap) && mx <= pw_start_x + i as f32 * (pw_w + pw_gap) + pw_w && my >= 0.05 && my <= 0.11 { state.hotbar_colors[state.selected_slot] = preset_col; state.update_ui(); return; } }
                                if mx >= -0.40 && mx <= -0.22 && my >= -0.10 && my <= -0.02 { state.scale_voxel_size(false); return; }
                                if mx >= 0.22 && mx <= 0.40 && my >= -0.10 && my <= -0.02 { state.scale_voxel_size(true); return; }
                                if mx >= -0.22 && mx <= 0.22 && my >= -0.25 && my <= -0.17 { state.set_menu(ActiveMenu::None); return; }
                            } else if button == MouseButton::Left { state.active_slider = None; }
                        }
                        ActiveMenu::Pause => {
                            if button == MouseButton::Left && element_state == ElementState::Pressed {
                                if mx >= -0.30 && mx <= 0.30 && my >= 0.46 && my <= 0.54 { state.prompt_native_file_dialog(); return; }
                                if mx >= -0.30 && mx <= 0.30 && my >= 0.36 && my <= 0.44 { state.toggle_play_mode(); return; }
                                if mx >= -0.30 && mx <= 0.30 && my >= 0.26 && my <= 0.34 { state.toggle_projection(); return; }
                                if mx >= -0.30 && mx <= 0.30 && my >= 0.16 && my <= 0.24 { state.cycle_world_generator(); return; }
                                if mx >= -0.30 && mx <= 0.30 && my >= 0.06 && my <= 0.14 { state.clear_all_blocks(); return; }
                                if mx >= -0.30 && mx <= -0.02 && my >= -0.06 && my <= 0.02 { let _ = state.save_game("world_save.json"); return; }
                                if mx >= 0.02 && mx <= 0.30 && my >= -0.06 && my <= 0.02 { let _ = state.load_game("world_save.json"); return; }
                                if mx >= -0.30 && mx <= 0.30 && my >= -0.16 && my <= -0.08 { state.bg_color = if state.bg_color[0] < 0.2 { [0.45, 0.5, 0.6] } else if state.bg_color[0] < 0.6 { [0.8, 0.85, 0.9] } else { [0.12, 0.14, 0.18] }; return; }
                                if mx >= -0.30 && mx <= 0.30 && my >= -0.28 && my <= -0.20 { state.set_menu(ActiveMenu::Controls); return; }
                                if mx >= -0.30 && mx <= 0.30 && my >= -0.42 && my <= -0.34 { state.set_menu(ActiveMenu::None); return; }
                                if mx >= -0.30 && mx <= 0.30 && my >= -0.54 && my <= -0.46 { event_loop.exit(); return; }
                            }
                        }
                        ActiveMenu::Controls => {
                            if button == MouseButton::Left && element_state == ElementState::Pressed {
                                if mx >= -0.30 && mx <= 0.30 && my >= -0.66 && my <= -0.56 { state.set_menu(ActiveMenu::Pause); return; }
                            }
                        }
                        ActiveMenu::ImportParams => {
                            if button == MouseButton::Left && element_state == ElementState::Pressed {
                                let wx0 = -0.42; let wx1 = 0.42;
                                if mx >= wx1 - 0.20 && mx <= wx1 - 0.04 && my >= 0.38 && my <= 0.46 { state.prompt_native_file_dialog(); return; }
                                if my >= 0.19 && my <= 0.27 {
                                    if mx >= wx0 + 0.04 && mx <= wx0 + 0.14 { state.glb_settings.target_height = (state.glb_settings.target_height - 4.0).max(4.0); state.update_ui(); return; }
                                    if mx >= wx1 - 0.14 && mx <= wx1 - 0.04 { state.glb_settings.target_height = (state.glb_settings.target_height + 4.0).min(128.0); state.update_ui(); return; }
                                }
                                if my >= -0.01 && my <= 0.07 {
                                    if mx >= wx0 + 0.04 && mx <= wx0 + 0.14 { state.glb_settings.voxel_size = (state.glb_settings.voxel_size * 0.5).max(MIN_VOXEL_SIZE); state.update_ui(); return; }
                                    if mx >= wx1 - 0.14 && mx <= wx1 - 0.04 { state.glb_settings.voxel_size *= 2.0; state.update_ui(); return; }
                                }
                                if my >= -0.21 && my <= -0.13 {
                                    if mx >= wx0 + 0.04 && mx <= wx0 + 0.14 { state.glb_settings.palette_size = (state.glb_settings.palette_size / 2).max(2); state.update_ui(); return; }
                                    if mx >= wx1 - 0.14 && mx <= wx1 - 0.04 { state.glb_settings.palette_size = (state.glb_settings.palette_size * 2).min(8192); state.update_ui(); return; }
                                }
                                if mx >= wx0 + 0.04 && mx <= wx1 - 0.04 && my >= -0.39 && my <= -0.31 {
                                    state.glb_settings.place_at_aim = !state.glb_settings.place_at_aim; state.update_ui(); return;
                                }
                                if mx >= wx0 + 0.04 && mx <= wx1 - 0.04 && my >= -0.54 && my <= -0.44 {
                                    if state.glb_settings.is_safe() { state.start_nonblocking_voxelization(); }
                                    return;
                                }
                                if mx >= wx0 + 0.04 && mx <= wx1 - 0.04 && my >= -0.66 && my <= -0.56 {
                                    state.set_menu(ActiveMenu::Pause); return;
                                }
                            }
                        }
                        ActiveMenu::Voxelizing => {}
                        ActiveMenu::None => {
                            if element_state == ElementState::Pressed && !state.cursor_free {
                                match button {
                                    MouseButton::Left => state.input.action_add = true,
                                    MouseButton::Right => state.input.action_remove = true,
                                    MouseButton::Middle => state.input.action_pick = true,
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                WindowEvent::Resized(size) => { state.resize(size); state.window.request_redraw(); }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let dt = now.duration_since(self.last_frame).as_secs_f32();
                    self.last_frame = now;
                    state.update(dt);
                    state.render();
                    state.window.request_redraw();
                }
                _ => {}
            }
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: winit::event::DeviceId, event: DeviceEvent) {
        if let Some(state) = self.state.as_mut() {
            if state.active_menu == ActiveMenu::None && !state.cursor_free && !state.gimbal_dragging {
                if let DeviceEvent::MouseMotion { delta } = event {
                    state.camera.yaw += (delta.0 as f32) * 0.002;
                    state.camera.pitch -= (delta.1 as f32) * 0.002;
                    state.camera.pitch = state.camera.pitch.clamp(-1.56, 1.56);
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_mut() { state.window.request_redraw(); }
    }
}

pub fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { state: None, last_frame: Instant::now() };
    event_loop.run_app(&mut app).unwrap();
}