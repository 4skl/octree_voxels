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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::path::PathBuf;
use glam::{Vec3, Mat4};
use noise::{NoiseFn, Fbm, Perlin};
use serde::{Serialize, Deserialize};
use rayon::prelude::*;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const CHUNK_SIZE: f32 = 32.0;
const MAX_DEPTH: u8 = 8;
const GRID_RES: u32 = 1 << MAX_DEPTH; // 256
const MIN_VOXEL_SIZE: f32 = CHUNK_SIZE / (GRID_RES as f32); // 0.125

type ChunkPos = (i32, i32, i32);
pub type Palette = Vec<[f32; 3]>;

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum ActiveMenu {
    None,
    Edit,
    Pause,
    ImportParams,
    Voxelizing,
}

#[derive(PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum PlayMode { Flying, Real }

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum WorldType {
    Empty,
    Flat,
    Hills,
    Mountains,
    FloatingIslands,
}

impl WorldType {
    pub fn next(&self) -> Self {
        match self {
            WorldType::Empty => WorldType::Flat,
            WorldType::Flat => WorldType::Hills,
            WorldType::Hills => WorldType::Mountains,
            WorldType::Mountains => WorldType::FloatingIslands,
            WorldType::FloatingIslands => WorldType::Empty,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            WorldType::Empty => "EMPTY CANVAS",
            WorldType::Flat => "FLAT",
            WorldType::Hills => "HILLS",
            WorldType::Mountains => "MOUNTAINS",
            WorldType::FloatingIslands => "ISLANDS",
        }
    }
}

fn default_ortho_size() -> f32 { 36.0 }

#[derive(Serialize, Deserialize, Clone)]
pub struct CubeEdit {
    pub pos: [f32; 3],
    pub size: f32,
    pub material: u16,
}

#[derive(Serialize, Deserialize)]
pub struct SaveData {
    pub player_pos: [f32; 3],
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    #[serde(default)]
    pub is_ortho: bool,
    #[serde(default = "default_ortho_size")]
    pub ortho_size: f32,
    pub play_mode: PlayMode,
    pub world_type: WorldType,
    pub seed: u32,
    pub hotbar_colors: [[f32; 3]; 10],
    pub palette: Vec<[f32; 3]>,
    #[serde(default)]
    pub cube_edits: Vec<CubeEdit>,
}

#[derive(Clone)]
pub struct GlbImportSettings {
    pub selected_file: Option<PathBuf>,
    pub target_height: f32,
    pub voxel_size: f32,
    pub place_at_aim: bool,
}

impl Default for GlbImportSettings {
    fn default() -> Self {
        Self {
            selected_file: None,
            target_height: 16.0,
            voxel_size: MIN_VOXEL_SIZE,
            place_at_aim: true,
        }
    }
}

pub struct ImportedChunkData {
    pub pos: ChunkPos,
    pub octree: Octree,
    pub mesh: MeshPayload,
}

pub enum VoxelizeMsg {
    Progress { percent: f32, stage: String },
    Done(Result<Vec<ImportedChunkData>, String>),
}

// --- OCTREE WITH AUTOMATIC SIBLING MERGING ---

#[derive(Clone, Copy, Default)]
pub struct OctreeNode {
    pub child_mask: u8,
    pub material_id: u16,
    pub child_pointer: u32,
}

#[derive(Clone)]
pub struct Octree {
    pub nodes: Vec<OctreeNode>,
    pub root_index: u32,
}

impl Octree {
    pub fn new() -> Self { 
        Self { nodes: vec![OctreeNode::default()], root_index: 0 } 
    }

    pub fn insert_cube(&mut self, mut x: u32, mut y: u32, mut z: u32, depth: u8, material: u16, auto_collapse: bool) {
        if depth == 0 {
            self.nodes[self.root_index as usize] = OctreeNode {
                child_mask: if material != 0 { 0xFF } else { 0 },
                material_id: material,
                child_pointer: 0,
            };
            return;
        }

        let mut current_idx = self.root_index as usize;
        let mut half_size = GRID_RES / 2;

        for _ in 0..depth {
            let old_mat = self.nodes[current_idx].material_id;
            let child_ptr = self.nodes[current_idx].child_pointer;

            if child_ptr == 0 {
                if old_mat == material {
                    return;
                }
                let new_ptr = self.nodes.len() as u32;
                self.nodes.resize(self.nodes.len() + 8, OctreeNode::default());
                if old_mat != 0 {
                    for c in 0..8 {
                        self.nodes[(new_ptr + c) as usize].material_id = old_mat;
                    }
                    self.nodes[current_idx].child_mask = 0xFF;
                } else {
                    self.nodes[current_idx].child_mask = 0;
                }
                self.nodes[current_idx].child_pointer = new_ptr;
                self.nodes[current_idx].material_id = 0;
            }

            let mut octant = 0;
            if x >= half_size { octant |= 1; x -= half_size; }
            if y >= half_size { octant |= 2; y -= half_size; }
            if z >= half_size { octant |= 4; z -= half_size; }

            current_idx = (self.nodes[current_idx].child_pointer + octant) as usize;
            half_size >>= 1;
        }

        self.nodes[current_idx].material_id = material;
        self.nodes[current_idx].child_pointer = 0;
        self.nodes[current_idx].child_mask = if material != 0 { 0xFF } else { 0 };

        if auto_collapse {
            self.collapse(self.root_index as usize);
        }
    }

    pub fn collapse(&mut self, node_idx: usize) -> bool {
        let child_ptr = self.nodes[node_idx].child_pointer;
        if child_ptr == 0 {
            return true;
        }

        let mut all_same = true;
        let first_mat = self.nodes[child_ptr as usize].material_id;

        for i in 0..8 {
            let c_idx = (child_ptr + i) as usize;
            let is_leaf = self.collapse(c_idx);
            if !is_leaf || self.nodes[c_idx].material_id != first_mat {
                all_same = false;
            }
        }

        if all_same {
            self.nodes[node_idx].material_id = first_mat;
            self.nodes[node_idx].child_pointer = 0;
            self.nodes[node_idx].child_mask = if first_mat != 0 { 0xFF } else { 0 };
            return true;
        }

        let mut mask = 0;
        for i in 0..8 {
            let c_idx = (child_ptr + i) as usize;
            if self.nodes[c_idx].material_id != 0 || self.nodes[c_idx].child_pointer != 0 {
                mask |= 1 << i;
            }
        }
        self.nodes[node_idx].child_mask = mask;
        false
    }

    pub fn query_node(&self, mut x: u32, mut y: u32, mut z: u32) -> (u16, u32) {
        let mut current_idx = self.root_index as usize;
        let mut size = GRID_RES;
        for _ in 0..MAX_DEPTH {
            let node = &self.nodes[current_idx];
            if node.child_pointer == 0 {
                return (node.material_id, size);
            }
            let half = size / 2;
            let mut octant = 0;
            if x >= half { octant |= 1; x -= half; }
            if y >= half { octant |= 2; y -= half; }
            if z >= half { octant |= 4; z -= half; }

            if (node.child_mask & (1 << octant)) == 0 {
                return (0, half);
            }
            current_idx = (node.child_pointer + octant) as usize;
            size = half;
        }
        (self.nodes[current_idx].material_id, size)
    }
}

// --- FAST 64-BIT SPATIAL KEY PACKING ---

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

// --- SAMPLING ---

fn sample_triangle(
    pts: [Vec3; 3],
    uvs: [[f32; 2]; 3],
    colors: [[f32; 3]; 3],
    base_color: [f32; 3],
    tex_idx: Option<usize>,
    images: &[gltf::image::Data],
    voxel_size: f32,
    out: &mut Vec<(u64, [f32; 3])>,
) {
    let min_p = pts[0].min(pts[1]).min(pts[2]);
    let max_p = pts[0].max(pts[1]).max(pts[2]);

    let min_gx = (min_p.x / voxel_size).floor() as i32;
    let max_gx = (max_p.x / voxel_size).floor() as i32;
    let min_gy = (min_p.y / voxel_size).floor() as i32;
    let max_gy = (max_p.y / voxel_size).floor() as i32;
    let min_gz = (min_p.z / voxel_size).floor() as i32;
    let max_gz = (max_p.z / voxel_size).floor() as i32;

    let sample_color = |u: f32, v: f32, w: f32| -> [f32; 3] {
        let mut col = [
            (colors[0][0] * w + colors[1][0] * u + colors[2][0] * v) * base_color[0],
            (colors[0][1] * w + colors[1][1] * u + colors[2][1] * v) * base_color[1],
            (colors[0][2] * w + colors[1][2] * u + colors[2][2] * v) * base_color[2],
        ];

        if let Some(ti) = tex_idx {
            if let Some(img) = images.get(ti) {
                if img.width > 0 && img.height > 0 {
                    let uv_x = uvs[0][0] * w + uvs[1][0] * u + uvs[2][0] * v;
                    let uv_y = uvs[0][1] * w + uvs[1][0] * u + uvs[2][1] * v;
                    let px_u = (uv_x.rem_euclid(1.0) * (img.width as f32 - 1.0)).round() as u32;
                    let px_v = (uv_y.rem_euclid(1.0) * (img.height as f32 - 1.0)).round() as u32;
                    let bpp = match img.format {
                        gltf::image::Format::R8G8B8 => 3,
                        gltf::image::Format::R8G8B8A8 => 4,
                        _ => 4,
                    };
                    let idx = (px_v * img.width + px_u) as usize * bpp;
                    if idx + 2 < img.pixels.len() {
                        col[0] *= img.pixels[idx] as f32 / 255.0;
                        col[1] *= img.pixels[idx + 1] as f32 / 255.0;
                        col[2] *= img.pixels[idx + 2] as f32 / 255.0;
                    }
                }
            }
        }
        col
    };

    if min_gx == max_gx && min_gy == max_gy && min_gz == max_gz {
        let col = sample_color(0.3333, 0.3333, 0.3334);
        out.push((pack_coords(min_gx, min_gy, min_gz), col));
        return;
    }

    let max_edge = (pts[1] - pts[0]).length()
        .max((pts[2] - pts[0]).length())
        .max((pts[2] - pts[1]).length());
    let steps = ((max_edge / (voxel_size * 0.707)).ceil() as usize).clamp(1, 24);

    for u_i in 0..=steps {
        for v_i in 0..=(steps - u_i) {
            let u = u_i as f32 / steps as f32;
            let v = v_i as f32 / steps as f32;
            let w = 1.0 - u - v;

            let p = pts[0] * w + pts[1] * u + pts[2] * v;
            let gx = (p.x / voxel_size).floor() as i32;
            let gy = (p.y / voxel_size).floor() as i32;
            let gz = (p.z / voxel_size).floor() as i32;

            let col = sample_color(u, v, w);
            out.push((pack_coords(gx, gy, gz), col));
        }
    }
}

// --- BACKGROUND PARALLEL VOXELIZATION & MESHING ---

fn run_background_voxelization(
    path: PathBuf,
    target_pos: Vec3,
    voxel_size: f32,
    target_height: f32,
    palette_arc: Arc<RwLock<Palette>>,
    tx: mpsc::Sender<VoxelizeMsg>,
) {
    let _ = tx.send(VoxelizeMsg::Progress {
        percent: 0.05,
        stage: "READING GLB CONTAINER...".into(),
    });

    let (document, buffers, images) = match gltf::import(&path) {
        Ok(res) => res,
        Err(e) => {
            let _ = tx.send(VoxelizeMsg::Done(Err(format!("Import error: {}", e))));
            return;
        }
    };

    let mut min_bound = Vec3::splat(f32::MAX);
    let mut max_bound = Vec3::splat(f32::MIN);
    let mut all_triangles = Vec::new();

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|b| Some(&buffers[b.index()]));
            let positions: Vec<Vec3> = match reader.read_positions() {
                Some(iter) => iter.map(Vec3::from).collect(),
                None => continue,
            };

            for p in &positions {
                min_bound = min_bound.min(*p);
                max_bound = max_bound.max(*p);
            }

            let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

            let colors: Vec<[f32; 3]> = reader.read_colors(0)
                .map(|iter| iter.into_rgb_f32().collect())
                .unwrap_or_else(|| vec![[1.0, 1.0, 1.0]; positions.len()]);

            let pbr = primitive.material().pbr_metallic_roughness();
            let base_factor = pbr.base_color_factor();
            let tex_index = pbr.base_color_texture().map(|t| t.texture().source().index());

            let indices: Vec<u32> = reader.read_indices()
                .map(|iter| iter.into_u32().collect())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());

            for chunk in indices.chunks_exact(3) {
                let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
                all_triangles.push((
                    [positions[i0], positions[i1], positions[i2]],
                    [uvs[i0], uvs[i1], uvs[i2]],
                    [colors[i0], colors[i1], colors[i2]],
                    [base_factor[0], base_factor[1], base_factor[2]],
                    tex_index,
                ));
            }
        }
    }

    let extent = max_bound - min_bound;
    let max_dim = extent.x.max(extent.y).max(extent.z).max(0.001);
    let scale = target_height / max_dim;
    let center_x = (min_bound.x + max_bound.x) * 0.5;
    let center_z = (min_bound.z + max_bound.z) * 0.5;
    let min_y = min_bound.y;

    let normalized_triangles: Vec<_> = all_triangles.into_iter().map(|(pts, uvs, vcols, base_col, tex_idx)| {
        let p0 = Vec3::new((pts[0].x - center_x) * scale, (pts[0].y - min_y) * scale, (pts[0].z - center_z) * scale) + target_pos;
        let p1 = Vec3::new((pts[1].x - center_x) * scale, (pts[1].y - min_y) * scale, (pts[1].z - center_z) * scale) + target_pos;
        let p2 = Vec3::new((pts[2].x - center_x) * scale, (pts[2].y - min_y) * scale, (pts[2].z - center_z) * scale) + target_pos;
        ([p0, p1, p2], uvs, vcols, base_col, tex_idx)
    }).collect();

    let total_triangles = normalized_triangles.len();
    let processed = Arc::new(AtomicUsize::new(0));
    let processed_clone = Arc::clone(&processed);
    let tx_progress = tx.clone();

    let reporter = std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_millis(33));
            let count = processed_clone.load(Ordering::Relaxed);
            let pct = 0.10 + 0.60 * (count as f32 / total_triangles.max(1) as f32);
            let msg = VoxelizeMsg::Progress {
                percent: pct.min(0.70),
                stage: format!("VOXELIZING: {}/{} TRIS ({:.0}%)", count, total_triangles, (count as f32 / total_triangles.max(1) as f32) * 100.0),
            };
            if tx_progress.send(msg).is_err() || count >= total_triangles {
                break;
            }
        }
    });

    let batch_size = 2048;
    let mut sampled_batches: Vec<Vec<(u64, [f32; 3])>> = normalized_triangles
        .par_chunks(batch_size)
        .map(|chunk| {
            let mut local_out = Vec::with_capacity(chunk.len() * 2);
            for &(pts, uvs, vcols, base_col, tex_idx) in chunk {
                sample_triangle(pts, uvs, vcols, base_col, tex_idx, &images, voxel_size, &mut local_out);
            }
            local_out.sort_unstable_by_key(|&(k, _)| k);
            local_out.dedup_by(|a, b| a.0 == b.0);

            processed.fetch_add(chunk.len(), Ordering::Relaxed);
            local_out
        })
        .collect();

    let _ = reporter.join();

    let _ = tx.send(VoxelizeMsg::Progress {
        percent: 0.72,
        stage: "DEDUPLICATING SURFACE VOXELS...".into(),
    });

    let mut all_samples: Vec<(u64, [f32; 3])> = sampled_batches.into_par_iter().flatten().collect();
    all_samples.par_sort_unstable_by_key(|&(k, _)| k);
    all_samples.dedup_by(|a, b| a.0 == b.0);

    let _ = tx.send(VoxelizeMsg::Progress {
        percent: 0.78,
        stage: format!("RESOLVING PALETTE FOR {} VOXELS...", all_samples.len()),
    });

    let mut pal = palette_arc.write().unwrap();
    let mut color_cache: HashMap<[u8; 3], u16> = HashMap::new();
    for (i, &c) in pal.iter().enumerate() {
        let key = [(c[0] * 255.0).round() as u8, (c[1] * 255.0).round() as u8, (c[2] * 255.0).round() as u8];
        color_cache.insert(key, (i + 1) as u16);
    }

    let grid_size = ((voxel_size / 32.0) * GRID_RES as f32).round().max(1.0) as u32;
    let depth = MAX_DEPTH - grid_size.trailing_zeros().min(MAX_DEPTH as u32) as u8;

    let mut chunks_voxels: HashMap<ChunkPos, Vec<(u32, u32, u32, u16)>> = HashMap::new();

    for (k, color) in all_samples {
        let key = [(color[0] * 255.0).round() as u8, (color[1] * 255.0).round() as u8, (color[2] * 255.0).round() as u8];
        let mat_id = *color_cache.entry(key).or_insert_with(|| {
            pal.push(color);
            pal.len() as u16
        });

        let (gx, gy, gz) = unpack_coords(k);
        let wx = gx as f32 * voxel_size;
        let wy = gy as f32 * voxel_size;
        let wz = gz as f32 * voxel_size;

        let cx = (wx / 32.0).floor() as i32;
        let cy = (wy / 32.0).floor() as i32;
        let cz = (wz / 32.0).floor() as i32;

        let lx = wx - (cx * 32) as f32;
        let ly = wy - (cy * 32) as f32;
        let lz = wz - (cz * 32) as f32;

        let local_gx = (((lx / 32.0) * GRID_RES as f32).floor() as u32).min(GRID_RES - 1);
        let local_gy = (((ly / 32.0) * GRID_RES as f32).floor() as u32).min(GRID_RES - 1);
        let local_gz = (((lz / 32.0) * GRID_RES as f32).floor() as u32).min(GRID_RES - 1);

        chunks_voxels.entry((cx, cy, cz)).or_default().push((local_gx, local_gy, local_gz, mat_id));
    }

    let pal_snapshot = pal.clone();
    drop(pal);

    let _ = tx.send(VoxelizeMsg::Progress {
        percent: 0.85,
        stage: format!("PARALLEL MESHING {} CHUNKS...", chunks_voxels.len()),
    });

    let completed_chunks: Vec<ImportedChunkData> = chunks_voxels
        .into_par_iter()
        .map(|(chunk_pos, voxels)| {
            let mut octree = Octree::new();
            for (x, y, z, mat) in voxels {
                octree.insert_cube(x, y, z, depth, mat, false);
            }
            octree.collapse(octree.root_index as usize);
            let mesh = generate_mesh(&octree, chunk_pos, &pal_snapshot);
            ImportedChunkData { pos: chunk_pos, octree, mesh }
        })
        .collect();

    let _ = tx.send(VoxelizeMsg::Progress {
        percent: 0.99,
        stage: "DISPATCHING TO GPU...".into(),
    });

    let _ = tx.send(VoxelizeMsg::Done(Ok(completed_chunks)));
}

// --- DIRECT OCTREE MESHING ---

pub struct MeshPayload { vertices: Vec<Vertex>, indices: Vec<u32> }

fn generate_octree(
    chunk_pos: ChunkPos,
    world_type: WorldType,
    seed: u32,
    edits: &[CubeEdit],
) -> Octree {
    let mut octree = Octree::new();
    let fbm = Fbm::<Perlin>::new(seed);
    let world_x_offset = chunk_pos.0 * 32;
    let world_z_offset = chunk_pos.2 * 32;

    match world_type {
        WorldType::Empty => {}
        WorldType::Flat => {
            if chunk_pos.1 == 0 {
                for x in 0..32 {
                    for z in 0..32 {
                        for y in 0..4 {
                            let mat = if y == 3 { 2 } else { 1 };
                            octree.insert_cube(x * 8, y * 8, z * 8, 5, mat, false);
                        }
                    }
                }
            }
        }
        WorldType::Hills => {
            for x in 0..32 {
                for z in 0..32 {
                    let wx = (x as i32 + world_x_offset) as f64 * 0.04;
                    let wz = (z as i32 + world_z_offset) as f64 * 0.04;
                    let height = ((fbm.get([wx, wz]) + 1.0) * 12.0).clamp(0.0, 31.0) as u32;
                    for y in 0..=height {
                        let mat = if y == height { 2 } else { 1 };
                        octree.insert_cube(x * 8, y * 8, z * 8, 5, mat, false);
                    }
                }
            }
        }
        WorldType::Mountains => {
            for x in 0..32 {
                for z in 0..32 {
                    let wx = (x as i32 + world_x_offset) as f64 * 0.025;
                    let wz = (z as i32 + world_z_offset) as f64 * 0.025;
                    let height = (fbm.get([wx, wz]).abs() * 30.0 + 2.0).clamp(0.0, 31.0) as u32;
                    for y in 0..=height {
                        let mat = if y >= 25 { 3 } else if y == height { 2 } else { 1 };
                        octree.insert_cube(x * 8, y * 8, z * 8, 5, mat, false);
                    }
                }
            }
        }
        WorldType::FloatingIslands => {
            let world_y_offset = chunk_pos.1 * 32;
            for x in 0..32 {
                for y in 0..32 {
                    for z in 0..32 {
                        let wx = (x as i32 + world_x_offset) as f64 * 0.05;
                        let wy = (y as i32 + world_y_offset) as f64 * 0.08;
                        let wz = (z as i32 + world_z_offset) as f64 * 0.05;
                        let density = fbm.get([wx, wy, wz]) - ((y as f64 + world_y_offset as f64) - 16.0) * 0.04;
                        if density > 0.12 {
                            octree.insert_cube(x * 8, y * 8, z * 8, 5, 2, false);
                        }
                    }
                }
            }
        }
    }

    for edit in edits {
        let chunk_min_x = (chunk_pos.0 * 32) as f32;
        let chunk_min_y = (chunk_pos.1 * 32) as f32;
        let chunk_min_z = (chunk_pos.2 * 32) as f32;

        if edit.pos[0] + edit.size > chunk_min_x && edit.pos[0] < chunk_min_x + 32.0
            && edit.pos[1] + edit.size > chunk_min_y && edit.pos[1] < chunk_min_y + 32.0
            && edit.pos[2] + edit.size > chunk_min_z && edit.pos[2] < chunk_min_z + 32.0
        {
            if edit.size >= 32.0 {
                octree.insert_cube(0, 0, 0, 0, edit.material, false);
            } else {
                let lx = edit.pos[0] - chunk_min_x;
                let ly = edit.pos[1] - chunk_min_y;
                let lz = edit.pos[2] - chunk_min_z;

                let gx = ((lx / 32.0) * GRID_RES as f32).round() as u32;
                let gy = ((ly / 32.0) * GRID_RES as f32).round() as u32;
                let gz = ((lz / 32.0) * GRID_RES as f32).round() as u32;

                let grid_size = ((edit.size / 32.0) * GRID_RES as f32).round().max(1.0) as u32;
                let depth = MAX_DEPTH - grid_size.trailing_zeros().min(MAX_DEPTH as u32) as u8;

                octree.insert_cube(gx, gy, gz, depth, edit.material, false);
            }
        }
    }

    octree.collapse(octree.root_index as usize);
    octree
}

const QUAD_UVS: [[f32; 2]; 4] = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
];

fn emit_quad(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    corners: [[f32; 3]; 4],
    normal: [f32; 3],
    color: [f32; 3],
) {
    let start = vertices.len() as u32;
    for (i, &pos) in corners.iter().enumerate() {
        vertices.push(Vertex {
            position: pos,
            normal,
            color,
            uv: QUAD_UVS[i],
        });
    }
    indices.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
}

fn is_face_occluded(
    octree: &Octree,
    gx: u32, gy: u32, gz: u32,
    size: u32,
    axis: usize,
    positive: bool,
) -> bool {
    let (target_coord, limit) = match axis {
        0 => (if positive { gx + size } else { gx.wrapping_sub(1) }, GRID_RES),
        1 => (if positive { gy + size } else { gy.wrapping_sub(1) }, GRID_RES),
        _ => (if positive { gz + size } else { gz.wrapping_sub(1) }, GRID_RES),
    };

    if target_coord >= limit { return false; }

    let (cx, cy, cz) = match axis {
        0 => (target_coord, gy + size / 2, gz + size / 2),
        1 => (gx + size / 2, target_coord, gz + size / 2),
        _ => (gx + size / 2, gy + size / 2, target_coord),
    };

    let (mat, n_size) = octree.query_node(cx, cy, cz);
    if mat == 0 { return false; }
    if n_size >= size { return true; }

    let q1 = size / 4;
    let q3 = (size * 3) / 4;
    let pts = match axis {
        0 => [(target_coord, gy + q1, gz + q1), (target_coord, gy + q3, gz + q1), (target_coord, gy + q1, gz + q3), (target_coord, gy + q3, gz + q3)],
        1 => [(gx + q1, target_coord, gz + q1), (gx + q3, target_coord, gz + q1), (gx + q1, target_coord, gz + q3), (gx + q3, target_coord, gz + q3)],
        _ => [(gx + q1, gy + q1, target_coord), (gx + q3, gy + q1, target_coord), (gx + q1, gy + q3, target_coord), (gx + q3, gy + q3, target_coord)],
    };

    pts.iter().all(|&(px, py, pz)| octree.query_node(px, py, pz).0 != 0)
}

fn mesh_octree_node(
    octree: &Octree,
    node_idx: usize,
    gx: u32, gy: u32, gz: u32,
    size: u32,
    chunk_pos: ChunkPos,
    palette: &[[f32; 3]],
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
) {
    let node = &octree.nodes[node_idx];
    if node.child_pointer != 0 {
        let half = size / 2;
        for i in 0..8 {
            if (node.child_mask & (1 << i)) != 0 {
                let cx = gx + if (i & 1) != 0 { half } else { 0 };
                let cy = gy + if (i & 2) != 0 { half } else { 0 };
                let cz = gz + if (i & 4) != 0 { half } else { 0 };
                mesh_octree_node(octree, (node.child_pointer + i) as usize, cx, cy, cz, half, chunk_pos, palette, vertices, indices);
            }
        }
        return;
    }

    if node.material_id == 0 {
        return;
    }

    let mat_id = node.material_id;
    let color = if mat_id >= 1 && (mat_id as usize) <= palette.len() {
        palette[(mat_id - 1) as usize]
    } else {
        [0.5, 0.5, 0.5]
    };

    let ox = (chunk_pos.0 * 32) as f32;
    let oy = (chunk_pos.1 * 32) as f32;
    let oz = (chunk_pos.2 * 32) as f32;

    let x0 = ox + (gx as f32 / GRID_RES as f32) * 32.0;
    let y0 = oy + (gy as f32 / GRID_RES as f32) * 32.0;
    let z0 = oz + (gz as f32 / GRID_RES as f32) * 32.0;
    let s = (size as f32 / GRID_RES as f32) * 32.0;
    let x1 = x0 + s;
    let y1 = y0 + s;
    let z1 = z0 + s;

    // +X
    if !is_face_occluded(octree, gx, gy, gz, size, 0, true) {
        emit_quad(vertices, indices, [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], [1.0, 0.0, 0.0], color);
    }
    // -X
    if !is_face_occluded(octree, gx, gy, gz, size, 0, false) {
        emit_quad(vertices, indices, [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], [-1.0, 0.0, 0.0], color);
    }
    // +Y
    if !is_face_occluded(octree, gx, gy, gz, size, 1, true) {
        emit_quad(vertices, indices, [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]], [0.0, 1.0, 0.0], color);
    }
    // -Y
    if !is_face_occluded(octree, gx, gy, gz, size, 1, false) {
        emit_quad(vertices, indices, [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]], [0.0, -1.0, 0.0], color);
    }
    // +Z
    if !is_face_occluded(octree, gx, gy, gz, size, 2, true) {
        emit_quad(vertices, indices, [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], [0.0, 0.0, 1.0], color);
    }
    // -Z
    if !is_face_occluded(octree, gx, gy, gz, size, 2, false) {
        emit_quad(vertices, indices, [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], [0.0, 0.0, -1.0], color);
    }
}

pub fn generate_mesh(octree: &Octree, chunk_pos: ChunkPos, palette: &[[f32; 3]]) -> MeshPayload {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    mesh_octree_node(octree, octree.root_index as usize, 0, 0, 0, GRID_RES, chunk_pos, palette, &mut vertices, &mut indices);
    MeshPayload { vertices, indices }
}

pub struct RenderChunk {
    octree: Octree,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    vertex_capacity: usize,
    index_capacity: usize,
}

pub fn update_chunk_buffers(
    chunk: &mut RenderChunk,
    payload: &MeshPayload,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    if payload.vertices.is_empty() || payload.indices.is_empty() {
        chunk.num_indices = 0;
        return;
    }

    let v_bytes = bytemuck::cast_slice(&payload.vertices);
    let i_bytes = bytemuck::cast_slice(&payload.indices);

    if v_bytes.len() > chunk.vertex_capacity {
        chunk.vertex_capacity = (v_bytes.len() * 2).max(1024);
        chunk.vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: chunk.vertex_capacity as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }

    if i_bytes.len() > chunk.index_capacity {
        chunk.index_capacity = (i_bytes.len() * 2).max(1024);
        chunk.index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: chunk.index_capacity as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }

    queue.write_buffer(&chunk.vertex_buffer, 0, v_bytes);
    queue.write_buffer(&chunk.index_buffer, 0, i_bytes);
    chunk.num_indices = payload.indices.len() as u32;
}

pub struct RaycastHit {
    pub hit_pos: Vec3,
    pub voxel_min: Vec3,
    pub voxel_size: f32,
    pub normal: Vec3,
    pub material: u16,
}

pub struct ChunkManager {
    loaded_chunks: HashMap<ChunkPos, RenderChunk>,
    loading_chunks: HashSet<ChunkPos>,
    tx: mpsc::Sender<(ChunkPos, Octree, MeshPayload)>,
    rx: mpsc::Receiver<(ChunkPos, Octree, MeshPayload)>,
    render_distance: i32,
    palette: Arc<RwLock<Palette>>,
    pub world_type: WorldType,
    pub seed: u32,
    pub cube_edits: Vec<CubeEdit>,
}

impl ChunkManager {
    fn new(palette: Arc<RwLock<Palette>>) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            loaded_chunks: HashMap::new(),
            loading_chunks: HashSet::new(),
            tx,
            rx,
            render_distance: 3,
            palette,
            world_type: WorldType::Empty,
            seed: 42,
            cube_edits: Vec::new(),
        }
    }

    pub fn get_voxel_info_at(&self, p: Vec3) -> (u16, f32, Vec3) {
        let cx = (p.x / 32.0).floor() as i32;
        let cy = (p.y / 32.0).floor() as i32;
        let cz = (p.z / 32.0).floor() as i32;

        if let Some(chunk) = self.loaded_chunks.get(&(cx, cy, cz)) {
            let lx = (p.x - cx as f32 * 32.0).clamp(0.0, 31.999);
            let ly = (p.y - cy as f32 * 32.0).clamp(0.0, 31.999);
            let lz = (p.z - cz as f32 * 32.0).clamp(0.0, 31.999);

            let gx = ((lx / 32.0) * GRID_RES as f32) as u32;
            let gy = ((ly / 32.0) * GRID_RES as f32) as u32;
            let gz = ((lz / 32.0) * GRID_RES as f32) as u32;

            let (mat, gsize) = chunk.octree.query_node(gx, gy, gz);
            let size = (gsize as f32 / GRID_RES as f32) * 32.0;
            let min_gx = (gx / gsize) * gsize;
            let min_gy = (gy / gsize) * gsize;
            let min_gz = (gz / gsize) * gsize;
            let vmin = Vec3::new(
                cx as f32 * 32.0 + (min_gx as f32 / GRID_RES as f32) * 32.0,
                cy as f32 * 32.0 + (min_gy as f32 / GRID_RES as f32) * 32.0,
                cz as f32 * 32.0 + (min_gz as f32 / GRID_RES as f32) * 32.0,
            );
            (mat, size, vmin)
        } else {
            (0, 4.0, p)
        }
    }

    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<RaycastHit> {
        if dir.length_squared() < 1e-6 { return None; }
        let dir = dir.normalize();

        let scale = (GRID_RES as f32) / CHUNK_SIZE;
        let cell_size = MIN_VOXEL_SIZE;

        let mut current_cell = glam::ivec3(
            (origin.x * scale).floor() as i32,
            (origin.y * scale).floor() as i32,
            (origin.z * scale).floor() as i32,
        );

        let step_x = if dir.x > 0.0 { 1 } else if dir.x < 0.0 { -1 } else { 0 };
        let step_y = if dir.y > 0.0 { 1 } else if dir.y < 0.0 { -1 } else { 0 };
        let step_z = if dir.z > 0.0 { 1 } else if dir.z < 0.0 { -1 } else { 0 };

        let delta_tx = if step_x != 0 { (cell_size / dir.x).abs() } else { f32::MAX };
        let delta_ty = if step_y != 0 { (cell_size / dir.y).abs() } else { f32::MAX };
        let delta_tz = if step_z != 0 { (cell_size / dir.z).abs() } else { f32::MAX };

        let next_voxel_boundary_x = if step_x > 0 { (current_cell.x + 1) as f32 * cell_size } else { current_cell.x as f32 * cell_size };
        let next_voxel_boundary_y = if step_y > 0 { (current_cell.y + 1) as f32 * cell_size } else { current_cell.y as f32 * cell_size };
        let next_voxel_boundary_z = if step_z > 0 { (current_cell.z + 1) as f32 * cell_size } else { current_cell.z as f32 * cell_size };

        let mut t_max_x = if step_x != 0 { (next_voxel_boundary_x - origin.x) / dir.x } else { f32::MAX };
        let mut t_max_y = if step_y != 0 { (next_voxel_boundary_y - origin.y) / dir.y } else { f32::MAX };
        let mut t_max_z = if step_z != 0 { (next_voxel_boundary_z - origin.z) / dir.z } else { f32::MAX };

        let mut normal = Vec3::ZERO;
        let mut dist = 0.0_f32;
        let max_steps = (max_dist * scale * 1.8) as i32;

        for _ in 0..max_steps {
            if dist > max_dist { break; }

            let cx = current_cell.x.div_euclid(GRID_RES as i32);
            let cy = current_cell.y.div_euclid(GRID_RES as i32);
            let cz = current_cell.z.div_euclid(GRID_RES as i32);

            if let Some(chunk) = self.loaded_chunks.get(&(cx, cy, cz)) {
                let gx = current_cell.x.rem_euclid(GRID_RES as i32) as u32;
                let gy = current_cell.y.rem_euclid(GRID_RES as i32) as u32;
                let gz = current_cell.z.rem_euclid(GRID_RES as i32) as u32;

                let (mat, gsize) = chunk.octree.query_node(gx, gy, gz);
                if mat != 0 {
                    let voxel_size = gsize as f32 * cell_size;
                    let min_gx = (gx / gsize) * gsize;
                    let min_gy = (gy / gsize) * gsize;
                    let min_gz = (gz / gsize) * gsize;
                    let voxel_min = Vec3::new(
                        cx as f32 * 32.0 + min_gx as f32 * cell_size,
                        cy as f32 * 32.0 + min_gy as f32 * cell_size,
                        cz as f32 * 32.0 + min_gz as f32 * cell_size,
                    );
                    let hit_pos = origin + dir * dist;
                    return Some(RaycastHit {
                        hit_pos,
                        voxel_min,
                        voxel_size,
                        normal,
                        material: mat,
                    });
                }
            }

            if t_max_x < t_max_y {
                if t_max_x < t_max_z {
                    current_cell.x += step_x;
                    dist = t_max_x;
                    t_max_x += delta_tx;
                    normal = Vec3::new(-step_x as f32, 0.0, 0.0);
                } else {
                    current_cell.z += step_z;
                    dist = t_max_z;
                    t_max_z += delta_tz;
                    normal = Vec3::new(0.0, 0.0, -step_z as f32);
                }
            } else {
                if t_max_y < t_max_z {
                    current_cell.y += step_y;
                    dist = t_max_y;
                    t_max_y += delta_ty;
                    normal = Vec3::new(0.0, -step_y as f32, 0.0);
                } else {
                    current_cell.z += step_z;
                    dist = t_max_z;
                    t_max_z += delta_tz;
                    normal = Vec3::new(0.0, 0.0, -step_z as f32);
                }
            }
        }
        None
    }

    fn get_or_create_chunk<'a>(&'a mut self, chunk_pos: ChunkPos, device: &wgpu::Device) -> &'a mut RenderChunk {
        let wtype = self.world_type;
        let seed = self.seed;
        let edits = self.cube_edits.clone();
        self.loaded_chunks.entry(chunk_pos).or_insert_with(|| {
            let octree = generate_octree(chunk_pos, wtype, seed, &edits);
            let vb = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: 1024, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
            });
            let ib = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: 1024, usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
            });
            RenderChunk { octree, vertex_buffer: vb, index_buffer: ib, num_indices: 0, vertex_capacity: 1024, index_capacity: 1024 }
        })
    }

    fn modify_cube(&mut self, pos: Vec3, material: u16, size: f32, device: &wgpu::Device, queue: &wgpu::Queue) {
        let edit = CubeEdit {
            pos: [pos.x, pos.y, pos.z],
            size,
            material,
        };
        self.cube_edits.push(edit);

        let pal = self.palette.read().unwrap().clone();

        if size >= 32.0 {
            let num_chunks = (size / 32.0).round() as i32;
            let start_cx = (pos.x / 32.0).floor() as i32;
            let start_cy = (pos.y / 32.0).floor() as i32;
            let start_cz = (pos.z / 32.0).floor() as i32;

            for dx in 0..num_chunks {
                for dy in 0..num_chunks {
                    for dz in 0..num_chunks {
                        let chunk_pos = (start_cx + dx, start_cy + dy, start_cz + dz);
                        let chunk = self.get_or_create_chunk(chunk_pos, device);
                        chunk.octree.insert_cube(0, 0, 0, 0, material, true);
                        let payload = generate_mesh(&chunk.octree, chunk_pos, &pal);
                        update_chunk_buffers(chunk, &payload, device, queue);
                    }
                }
            }
        } else {
            let cx = (pos.x / 32.0).floor() as i32;
            let cy = (pos.y / 32.0).floor() as i32;
            let cz = (pos.z / 32.0).floor() as i32;
            let chunk_pos = (cx, cy, cz);

            let chunk = self.get_or_create_chunk(chunk_pos, device);

            let lx = pos.x - (cx * 32) as f32;
            let ly = pos.y - (cy * 32) as f32;
            let lz = pos.z - (cz * 32) as f32;

            let gx = ((lx / 32.0) * GRID_RES as f32).round() as u32;
            let gy = ((ly / 32.0) * GRID_RES as f32).round() as u32;
            let gz = ((lz / 32.0) * GRID_RES as f32).round() as u32;

            let grid_size = ((size / 32.0) * GRID_RES as f32).round().max(1.0) as u32;
            let depth = MAX_DEPTH - grid_size.trailing_zeros().min(MAX_DEPTH as u32) as u8;

            chunk.octree.insert_cube(gx, gy, gz, depth, material, true);

            let payload = generate_mesh(&chunk.octree, chunk_pos, &pal);
            update_chunk_buffers(chunk, &payload, device, queue);
        }
    }

    fn update(&mut self, player_pos: Vec3, device: &wgpu::Device) {
        let p_x = (player_pos.x / 32.0).floor() as i32;
        let p_z = (player_pos.z / 32.0).floor() as i32;

        for x in -self.render_distance..=self.render_distance {
            for z in -self.render_distance..=self.render_distance {
                let pos = (p_x + x, 0, p_z + z);
                if !self.loaded_chunks.contains_key(&pos) && !self.loading_chunks.contains(&pos) {
                    self.loading_chunks.insert(pos);
                    let tx_clone = self.tx.clone();
                    let pal_arc = Arc::clone(&self.palette);
                    let edits = self.cube_edits.clone();
                    let wtype = self.world_type;
                    let seed = self.seed;

                    rayon::spawn(move || {
                        let octree = generate_octree(pos, wtype, seed, &edits);
                        let pal = pal_arc.read().unwrap().clone();
                        let payload = generate_mesh(&octree, pos, &pal);
                        let _ = tx_clone.send((pos, octree, payload));
                    });
                }
            }
        }

        while let Ok((pos, octree, payload)) = self.rx.try_recv() {
            self.loading_chunks.remove(&pos);
            if payload.vertices.is_empty() || payload.indices.is_empty() { continue; }
            let v_bytes = bytemuck::cast_slice(&payload.vertices);
            let i_bytes = bytemuck::cast_slice(&payload.indices);
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: v_bytes, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: i_bytes, usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            });
            self.loaded_chunks.insert(pos, RenderChunk {
                octree,
                vertex_buffer,
                index_buffer,
                num_indices: payload.indices.len() as u32,
                vertex_capacity: v_bytes.len(),
                index_capacity: i_bytes.len(),
            });
        }

        self.loaded_chunks.retain(|pos, _| {
            (pos.0 - p_x).abs() <= self.render_distance + 1 && (pos.2 - p_z).abs() <= self.render_distance + 1
        });
    }

    fn is_solid(&self, pos: Vec3) -> bool {
        self.get_voxel_info_at(pos).0 != 0
    }

    pub fn get_surface_y(&self, x: f32, z: f32) -> Option<f32> {
        let mut y = 96.0_f32;
        while y >= -32.0 {
            let test_pos = Vec3::new(x, y, z);
            if self.is_solid(test_pos) {
                let (_, size, vmin) = self.get_voxel_info_at(test_pos);
                return Some(vmin.y + size);
            }
            y -= 0.5;
        }
        None
    }
}

// --- VERTEX & CAMERA DATA ---

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex { 
    pub position: [f32; 3], 
    pub normal: [f32; 3], 
    pub color: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3, 
        1 => Float32x3, 
        2 => Float32x3, 
        3 => Float32x2
    ];
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct UIVertex { position: [f32; 2], color: [f32; 4] }

impl UIVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<UIVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform { 
    view_proj: [[f32; 4]; 4],
    show_borders: f32,
    _pad: [f32; 3],
}

struct Camera { 
    position: Vec3, 
    yaw: f32, 
    pitch: f32,
    is_ortho: bool,
    ortho_size: f32,
}

impl Camera {
    fn view_proj(&self, aspect: f32) -> Mat4 {
        let (sin_p, cos_p) = self.pitch.sin_cos();
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let dir = Vec3::new(cos_y * cos_p, sin_p, sin_y * cos_p).normalize();

        let view = glam::camera::rh::view::look_at_mat4(self.position, self.position + dir, Vec3::Y);
        let proj = if self.is_ortho {
            let half_h = self.ortho_size * 0.5;
            let half_w = half_h * aspect;
            glam::camera::rh::proj::directx::orthographic(-half_w, half_w, -half_h, half_h, -2500.0, 5000.0)
        } else {
            glam::camera::rh::proj::directx::perspective((60.0_f32).to_radians(), aspect, 0.05, 5000.0)
        };
        proj * view
    }
    fn forward(&self) -> Vec3 {
        let (sin_p, cos_p) = self.pitch.sin_cos();
        let (sin_y, cos_y) = self.yaw.sin_cos();
        Vec3::new(cos_y * cos_p, sin_p, sin_y * cos_p).normalize()
    }
    fn right(&self) -> Vec3 {
        let (sin_y, cos_y) = self.yaw.sin_cos();
        Vec3::new(-sin_y, 0.0, cos_y).normalize()
    }
    fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize()
    }
}

#[derive(Default)]
struct InputState { 
    forward: bool, 
    backward: bool, 
    left: bool, 
    right: bool, 
    up: bool, 
    down: bool, 
    action_add: bool, 
    action_remove: bool,
    action_pick: bool,
}

fn add_quad(verts: &mut Vec<UIVertex>, x0: f32, y0: f32, x1: f32, y1: f32, color: [f32; 4]) {
    verts.extend_from_slice(&[
        UIVertex { position: [x0, y0], color },
        UIVertex { position: [x1, y0], color },
        UIVertex { position: [x1, y1], color },
        UIVertex { position: [x0, y0], color },
        UIVertex { position: [x1, y1], color },
        UIVertex { position: [x0, y1], color },
    ]);
}

fn add_line(verts: &mut Vec<UIVertex>, x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32, aspect: f32, color: [f32; 4]) {
    let dx = (x1 - x0) * aspect;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-5 { return; }
    let half_t = thickness * 0.5;
    let nx = (-dy / len) * half_t / aspect;
    let ny = (dx / len) * half_t;

    verts.extend_from_slice(&[
        UIVertex { position: [x0 - nx, y0 - ny], color },
        UIVertex { position: [x1 - nx, y1 - ny], color },
        UIVertex { position: [x1 + nx, y1 + ny], color },
        UIVertex { position: [x0 - nx, y0 - ny], color },
        UIVertex { position: [x1 + nx, y1 + ny], color },
        UIVertex { position: [x0 + nx, y0 + ny], color },
    ]);
}

fn get_glyph_5x7(c: char) -> [u8; 7] {
    match c {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01111, 0b10000, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00001, 0b00001, 0b00001, 0b00001, 0b10001, 0b10001, 0b01110],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '0' => [0b01110, 0b10011, 0b10101, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        '3' => [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        ':' => [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000],
        '/' => [0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b00000, 0b00000],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        '%' => [0b11001, 0b11010, 0b00100, 0b01000, 0b01011, 0b10011, 0b00000],
        '[' => [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
        ']' => [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
        '(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        ')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        _ => [0; 7],
    }
}

fn draw_glyph_raw(
    verts: &mut Vec<UIVertex>,
    glyph: &[u8; 7],
    x: f32,
    y: f32,
    pw: f32,
    ph: f32,
    color: [f32; 4],
) {
    for row in 0..7 {
        let line = glyph[row];
        if line == 0 { continue; }
        let y1 = y + (6 - row) as f32 * ph;
        let y0 = y1 - ph;
        for col in 0..5 {
            if (line & (1 << (4 - col))) != 0 {
                let x0 = x + col as f32 * pw;
                let x1 = x0 + pw;
                add_quad(verts, x0, y0, x1, y1, color);
            }
        }
    }
}

pub fn draw_text(
    verts: &mut Vec<UIVertex>,
    text: &str,
    start_x: f32,
    start_y: f32,
    scale: f32,
    aspect: f32,
    color: [f32; 4],
) {
    let ph = 0.0032 * scale;
    let pw = ph / aspect;
    let shadow_color = [0.02, 0.02, 0.04, color[3] * 0.9];
    let shadow_offset_x = pw * 0.75;
    let shadow_offset_y = -ph * 0.75;

    let mut cursor_x = start_x;
    for c in text.to_ascii_uppercase().chars() {
        let glyph = get_glyph_5x7(c);
        draw_glyph_raw(verts, &glyph, cursor_x + shadow_offset_x, start_y + shadow_offset_y, pw, ph, shadow_color);
        draw_glyph_raw(verts, &glyph, cursor_x, start_y, pw, ph, color);
        cursor_x += 6.0 * pw;
    }
}

pub fn draw_text_centered(
    verts: &mut Vec<UIVertex>,
    text: &str,
    cx: f32,
    cy: f32,
    scale: f32,
    aspect: f32,
    color: [f32; 4],
) {
    let ph = 0.0032 * scale;
    let pw = ph / aspect;
    let total_w = (text.len() as f32 * 6.0 - 1.0) * pw;
    let total_h = 7.0 * ph;
    draw_text(verts, text, cx - total_w / 2.0, cy - total_h / 2.0, scale, aspect, color);
}

const PRESET_SWATCHES: [[f32; 3]; 10] = [
    [0.95, 0.95, 0.95],
    [0.10, 0.10, 0.12],
    [0.90, 0.20, 0.20],
    [0.95, 0.50, 0.15],
    [0.95, 0.85, 0.15],
    [0.20, 0.75, 0.25],
    [0.15, 0.80, 0.85],
    [0.20, 0.45, 0.90],
    [0.65, 0.25, 0.85],
    [0.55, 0.35, 0.20],
];

const GIZMO_CENTER_X: f32 = 0.86;
const GIZMO_CENTER_Y: f32 = 0.76;
const GIZMO_RADIUS: f32 = 0.11;

struct GizmoAxis {
    dir: Vec3,
    name: &'static str,
    color: [f32; 4],
    yaw: f32,
    pitch: f32,
    is_positive: bool,
}

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
    let g_cx = GIZMO_CENTER_X;
    let g_cy = GIZMO_CENTER_Y;
    let disc_rx = (GIZMO_RADIUS + 0.02) / aspect;
    let btn_w = 0.075 / aspect;
    let btn_h = 0.055;
    let x1 = g_cx - disc_rx - 0.015;
    let x0 = x1 - btn_w;
    let y0 = g_cy - btn_h / 2.0;
    let y1 = g_cy + btn_h / 2.0;
    (x0, y0, x1, y1)
}

fn compute_placement_pos(hit: &RaycastHit, edit_size: f32) -> Vec3 {
    let s = edit_size;
    let mut pos = Vec3::ZERO;

    if hit.normal.x > 0.5 {
        let face_x = hit.voxel_min.x + hit.voxel_size;
        pos.x = (face_x / s).ceil() * s;
        pos.y = (hit.hit_pos.y / s).floor() * s;
        pos.z = (hit.hit_pos.z / s).floor() * s;
    } else if hit.normal.x < -0.5 {
        let face_x = hit.voxel_min.x;
        pos.x = ((face_x - s) / s).floor() * s;
        pos.y = (hit.hit_pos.y / s).floor() * s;
        pos.z = (hit.hit_pos.z / s).floor() * s;
    } else if hit.normal.y > 0.5 {
        let face_y = hit.voxel_min.y + hit.voxel_size;
        pos.y = (face_y / s).ceil() * s;
        pos.x = (hit.hit_pos.x / s).floor() * s;
        pos.z = (hit.hit_pos.z / s).floor() * s;
    } else if hit.normal.y < -0.5 {
        let face_y = hit.voxel_min.y;
        pos.y = ((face_y - s) / s).floor() * s;
        pos.x = (hit.hit_pos.x / s).floor() * s;
        pos.z = (hit.hit_pos.z / s).floor() * s;
    } else if hit.normal.z > 0.5 {
        let face_z = hit.voxel_min.z + hit.voxel_size;
        pos.z = (face_z / s).ceil() * s;
        pos.x = (hit.hit_pos.x / s).floor() * s;
        pos.y = (hit.hit_pos.y / s).floor() * s;
    } else if hit.normal.z < -0.5 {
        let face_z = hit.voxel_min.z;
        pos.z = ((face_z - s) / s).floor() * s;
        pos.x = (hit.hit_pos.x / s).floor() * s;
        pos.y = (hit.hit_pos.y / s).floor() * s;
    } else {
        pos = (hit.hit_pos / s).floor() * s;
    }

    pos
}

fn build_ui_vertices(
    selected_slot: usize,
    active_menu: ActiveMenu,
    hotbar_colors: &[[f32; 3]; 10],
    play_mode: PlayMode,
    is_ortho: bool,
    world_type: WorldType,
    edit_size: f32,
    target_pos: Option<[f32; 3]>,
    aspect: f32,
    camera_forward: Vec3,
    camera_right: Vec3,
    camera_up: Vec3,
    cursor_free: bool,
    glb_settings: &GlbImportSettings,
    progress_val: f32,
    progress_stage: &str,
) -> Vec<UIVertex> {
    let mut verts = Vec::new();

    let g_cx = GIZMO_CENTER_X;
    let g_cy = GIZMO_CENTER_Y;
    let g_rad = GIZMO_RADIUS;

    let disc_rx = (g_rad + 0.018) / aspect;
    let disc_ry = g_rad + 0.018;
    add_quad(&mut verts, g_cx - disc_rx - 0.003, g_cy - disc_ry - 0.003, g_cx + disc_rx + 0.003, g_cy + disc_ry + 0.003, [0.25, 0.30, 0.38, 0.6]);
    add_quad(&mut verts, g_cx - disc_rx, g_cy - disc_ry, g_cx + disc_rx, g_cy + disc_ry, [0.08, 0.10, 0.14, 0.70]);

    let (bx0, by0, bx1, by1) = get_focus_button_bounds(aspect);
    add_quad(&mut verts, bx0 - 0.003, by0 - 0.003, bx1 + 0.003, by1 + 0.003, [0.35, 0.40, 0.50, 0.8]);
    add_quad(&mut verts, bx0, by0, bx1, by1, [0.12, 0.15, 0.22, 0.90]);
    draw_text_centered(&mut verts, "[.]", (bx0 + bx1) / 2.0, (by0 + by1) / 2.0, 1.1, aspect, [0.3, 0.9, 1.0, 1.0]);

    let mut axes_projected: Vec<(GizmoAxis, f32, f32, f32)> = get_gizmo_axes().into_iter().map(|ax| {
        let sx = ax.dir.dot(camera_right);
        let sy = ax.dir.dot(camera_up);
        let depth = ax.dir.dot(camera_forward);
        (ax, sx, sy, depth)
    }).collect();

    axes_projected.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));

    for (ax, sx, sy, _) in &axes_projected {
        let tip_x = g_cx + (sx * g_rad) / aspect;
        let tip_y = g_cy + (sy * g_rad);

        if ax.is_positive {
            add_line(&mut verts, g_cx, g_cy, tip_x, tip_y, 0.0045, aspect, [ax.color[0], ax.color[1], ax.color[2], 0.85]);
            let node_r = 0.021;
            let n_rx = node_r / aspect;
            let n_ry = node_r;
            add_quad(&mut verts, tip_x - n_rx, tip_y - n_ry, tip_x + n_rx, tip_y + n_ry, ax.color);
            draw_text_centered(&mut verts, ax.name, tip_x, tip_y, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
        } else {
            add_line(&mut verts, g_cx, g_cy, tip_x, tip_y, 0.0025, aspect, [ax.color[0], ax.color[1], ax.color[2], 0.35]);
            let node_r = 0.010;
            let n_rx = node_r / aspect;
            let n_ry = node_r;
            add_quad(&mut verts, tip_x - n_rx, tip_y - n_ry, tip_x + n_rx, tip_y + n_ry, ax.color);
        }
    }

    let num_slots = 10;
    let slot_w = 0.054;
    let slot_gap = 0.009;
    let total_w = num_slots as f32 * slot_w + (num_slots - 1) as f32 * slot_gap;
    let start_x = -total_w / 2.0;
    let y_bottom = -0.96;
    let y_top = -0.86;

    for (i, &rgb) in hotbar_colors.iter().enumerate() {
        let x0 = start_x + i as f32 * (slot_w + slot_gap);
        let x1 = x0 + slot_w;

        if selected_slot == i {
            add_quad(&mut verts, x0 - 0.006, y_bottom - 0.006, x1 + 0.006, y_top + 0.006, [1.0, 0.9, 0.1, 1.0]);
        } else {
            add_quad(&mut verts, x0 - 0.003, y_bottom - 0.003, x1 + 0.003, y_top + 0.003, [0.15, 0.16, 0.20, 0.9]);
        }
        add_quad(&mut verts, x0, y_bottom, x1, y_top, [rgb[0], rgb[1], rgb[2], 1.0]);

        let num_str = match i { 9 => "0", _ => &format!("{}", i + 1) };
        draw_text_centered(&mut verts, num_str, (x0 + x1) / 2.0, y_top + 0.02, 1.0, aspect, [0.9, 0.9, 0.9, 0.9]);
    }

    let size_str = if edit_size < 1.0 {
        format!("RES: 1/{} ({:.3})", (1.0 / edit_size).round() as u32, edit_size)
    } else {
        format!("RES: {:.0}X{:.0}", edit_size, edit_size)
    };

    match active_menu {
        ActiveMenu::None => {
            add_quad(&mut verts, -0.012, -0.002, 0.012, 0.002, [1.0, 1.0, 1.0, 0.95]);
            add_quad(&mut verts, -0.002, -0.020, 0.002, 0.020, [1.0, 1.0, 1.0, 0.95]);

            let mode_hud = if play_mode == PlayMode::Flying { "FLY" } else { "REAL" };
            let proj_hud = if is_ortho { "ORTHO" } else { "PERSP" };
            let hud_title = format!("MODE: {} | PROJ: {} | WORLD: {} | {}", mode_hud, proj_hud, world_type.name(), size_str);
            draw_text(&mut verts, &hud_title, -0.96, 0.92, 1.25, aspect, [1.0, 1.0, 1.0, 0.95]);

            if cursor_free {
                draw_text(&mut verts, "CURSOR FREE  |  DRAG & DROP .GLB OR OPEN ESC MENU", -0.96, 0.86, 1.0, aspect, [0.95, 0.45, 0.2, 0.95]);
            } else if let Some(tpos) = target_pos {
                let target_str = format!("AIM: [{:.3}, {:.3}, {:.3}] (SNAP)", tpos[0], tpos[1], tpos[2]);
                draw_text(&mut verts, &target_str, -0.96, 0.86, 1.0, aspect, [0.3, 0.9, 0.9, 0.9]);
            } else {
                draw_text(&mut verts, "AIM: [UNBOUNDED VOID - PLACES IN FRONT]", -0.96, 0.86, 1.0, aspect, [0.6, 0.7, 0.8, 0.8]);
            }

            draw_text(&mut verts, "[E] PALETTE  [TAB] FREE  [P] ORTHO  [.] FOCUS  [ESC] MENU / IMPORT", -0.96, 0.80, 0.95, aspect, [0.9, 0.85, 0.4, 0.85]);
        }

        ActiveMenu::Edit => {
            add_quad(&mut verts, -1.0, -1.0, 1.0, 1.0, [0.03, 0.04, 0.06, 0.75]);

            let px0 = -0.56; let px1 = 0.56;
            let py0 = -0.58; let py1 = 0.65;
            add_quad(&mut verts, px0 - 0.006, py0 - 0.006, px1 + 0.006, py1 + 0.006, [0.25, 0.35, 0.50, 1.0]);
            add_quad(&mut verts, px0, py0, px1, py1, [0.10, 0.12, 0.16, 0.98]);

            draw_text_centered(&mut verts, "EDIT STUDIO - PALETTE & OCTREE (E)", 0.0, 0.58, 1.3, aspect, [1.0, 0.9, 0.2, 1.0]);

            let sw_w = 0.082;
            let sw_gap = 0.015;
            let sw_tot = 10.0 * sw_w + 9.0 * sw_gap;
            let s_start_x = -sw_tot / 2.0;
            let sy0 = 0.43;
            let sy1 = 0.52;

            for (i, &rgb) in hotbar_colors.iter().enumerate() {
                let sx0 = s_start_x + i as f32 * (sw_w + sw_gap);
                let sx1 = sx0 + sw_w;

                if selected_slot == i {
                    add_quad(&mut verts, sx0 - 0.008, sy0 - 0.008, sx1 + 0.008, sy1 + 0.008, [1.0, 0.9, 0.1, 1.0]);
                } else {
                    add_quad(&mut verts, sx0 - 0.004, sy0 - 0.004, sx1 + 0.004, sy1 + 0.004, [0.22, 0.24, 0.30, 1.0]);
                }
                add_quad(&mut verts, sx0, sy0, sx1, sy1, [rgb[0], rgb[1], rgb[2], 1.0]);

                let num_str = match i { 9 => "0", _ => &format!("{}", i + 1) };
                draw_text_centered(&mut verts, num_str, (sx0 + sx1) / 2.0, sy1 + 0.022, 1.0, aspect, [0.8, 0.8, 0.8, 0.9]);
            }

            let [cur_r, cur_g, cur_b] = hotbar_colors[selected_slot];
            add_quad(&mut verts, 0.24, 0.18, 0.46, 0.37, [0.25, 0.28, 0.35, 1.0]);
            add_quad(&mut verts, 0.248, 0.188, 0.452, 0.362, [cur_r, cur_g, cur_b, 1.0]);
            draw_text_centered(&mut verts, "ACTIVE COLOR", 0.35, 0.39, 1.0, aspect, [0.85, 0.85, 0.85, 0.9]);

            let sl_x0 = -0.32;
            let sl_x1 = 0.18;
            let channels = [
                ("R", cur_r, 0.32, 0.36, [0.90, 0.25, 0.25, 1.0]),
                ("G", cur_g, 0.25, 0.29, [0.25, 0.85, 0.30, 1.0]),
                ("B", cur_b, 0.18, 0.22, [0.25, 0.50, 0.95, 1.0]),
            ];

            for (lbl, val, y0, y1, bar_col) in channels {
                draw_text_centered(&mut verts, lbl, -0.37, (y0 + y1) / 2.0, 1.1, aspect, bar_col);
                add_quad(&mut verts, sl_x0, y0, sl_x1, y1, [0.18, 0.20, 0.25, 1.0]);
                let filled_x = sl_x0 + val * (sl_x1 - sl_x0);
                add_quad(&mut verts, sl_x0, y0, filled_x, y1, bar_col);
                add_quad(&mut verts, filled_x - 0.010, y0 - 0.006, filled_x + 0.010, y1 + 0.006, [1.0, 1.0, 1.0, 1.0]);
            }

            let pw_w = 0.076;
            let pw_gap = 0.012;
            let pw_tot = 10.0 * pw_w + 9.0 * pw_gap;
            let pw_start_x = -pw_tot / 2.0;
            let pwy0 = 0.05;
            let pwy1 = 0.11;

            draw_text_centered(&mut verts, "QUICK PALETTE CHIPS", 0.0, 0.135, 1.0, aspect, [0.75, 0.75, 0.8, 0.9]);
            for (i, &rgb) in PRESET_SWATCHES.iter().enumerate() {
                let px0 = pw_start_x + i as f32 * (pw_w + pw_gap);
                let px1 = px0 + pw_w;
                add_quad(&mut verts, px0 - 0.003, pwy0 - 0.003, px1 + 0.003, pwy1 + 0.003, [0.3, 0.3, 0.35, 1.0]);
                add_quad(&mut verts, px0, pwy0, px1, pwy1, [rgb[0], rgb[1], rgb[2], 1.0]);
            }

            add_quad(&mut verts, -0.40, -0.10, -0.22, -0.02, [0.35, 0.40, 0.55, 1.0]);
            draw_text_centered(&mut verts, "/ 2 (F)", -0.31, -0.06, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);

            let active_sz_txt = format!("CURRENT: {}", size_str);
            draw_text_centered(&mut verts, &active_sz_txt, 0.0, -0.06, 1.25, aspect, [1.0, 0.85, 0.2, 1.0]);

            add_quad(&mut verts, 0.22, -0.10, 0.40, -0.02, [0.35, 0.40, 0.55, 1.0]);
            draw_text_centered(&mut verts, "* 2 (R)", 0.31, -0.06, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);

            add_quad(&mut verts, -0.22, -0.25, 0.22, -0.17, [0.20, 0.50, 0.30, 1.0]);
            draw_text_centered(&mut verts, "DONE (PRESS E)", 0.0, -0.21, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
        }

        ActiveMenu::Pause => {
            add_quad(&mut verts, -1.0, -1.0, 1.0, 1.0, [0.03, 0.04, 0.06, 0.80]);

            let px0 = -0.38; let px1 = 0.38;
            let py0 = -0.72; let py1 = 0.72;
            add_quad(&mut verts, px0 - 0.006, py0 - 0.006, px1 + 0.006, py1 + 0.006, [0.45, 0.45, 0.50, 1.0]);
            add_quad(&mut verts, px0, py0, px1, py1, [0.12, 0.13, 0.17, 0.98]);

            draw_text_centered(&mut verts, "PAUSE / SYSTEM MENU", 0.0, 0.60, 1.3, aspect, [0.95, 0.95, 0.95, 1.0]);

            add_quad(&mut verts, -0.30, 0.46, 0.30, 0.54, [0.20, 0.55, 0.75, 1.0]);
            draw_text_centered(&mut verts, ">> IMPORT 3D MODEL (GLB) <<", 0.0, 0.50, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);

            let mode_text = if play_mode == PlayMode::Flying { "PLAY MODE: FLYING" } else { "PLAY MODE: REAL" };
            add_quad(&mut verts, -0.30, 0.36, 0.30, 0.44, [0.25, 0.35, 0.55, 1.0]);
            draw_text_centered(&mut verts, mode_text, 0.0, 0.40, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);

            let proj_text = if is_ortho { "VIEW: ORTHOGRAPHIC" } else { "VIEW: PERSPECTIVE" };
            add_quad(&mut verts, -0.30, 0.26, 0.30, 0.34, [0.22, 0.40, 0.55, 1.0]);
            draw_text_centered(&mut verts, proj_text, 0.0, 0.30, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);

            let world_label = format!("WORLD: {}", world_type.name());
            add_quad(&mut verts, -0.30, 0.16, 0.30, 0.24, [0.35, 0.25, 0.50, 1.0]);
            draw_text_centered(&mut verts, &world_label, 0.0, 0.20, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);

            add_quad(&mut verts, -0.30, 0.06, 0.30, 0.14, [0.60, 0.30, 0.20, 1.0]);
            draw_text_centered(&mut verts, "CLEAR SCENE", 0.0, 0.10, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);

            add_quad(&mut verts, -0.30, -0.06, -0.02, 0.02, [0.25, 0.45, 0.35, 1.0]);
            draw_text_centered(&mut verts, "SAVE (F5)", -0.16, -0.02, 1.0, aspect, [1.0, 1.0, 1.0, 1.0]);

            add_quad(&mut verts, 0.02, -0.06, 0.30, 0.02, [0.35, 0.45, 0.25, 1.0]);
            draw_text_centered(&mut verts, "LOAD (F9)", 0.16, -0.02, 1.0, aspect, [1.0, 1.0, 1.0, 1.0]);

            add_quad(&mut verts, -0.30, -0.18, 0.30, -0.10, [0.20, 0.55, 0.30, 1.0]);
            draw_text_centered(&mut verts, "RESUME (ESC)", 0.0, -0.14, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);

            add_quad(&mut verts, -0.30, -0.30, 0.30, -0.22, [0.55, 0.20, 0.20, 1.0]);
            draw_text_centered(&mut verts, "QUIT TO DESKTOP", 0.0, -0.26, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
        }

        ActiveMenu::ImportParams => {
            add_quad(&mut verts, -1.0, -1.0, 1.0, 1.0, [0.02, 0.03, 0.05, 0.85]);

            let wx0 = -0.42; let wx1 = 0.42;
            let wy0 = -0.58; let wy1 = 0.62;
            add_quad(&mut verts, wx0 - 0.006, wy0 - 0.006, wx1 + 0.006, wy1 + 0.006, [0.30, 0.45, 0.65, 1.0]);
            add_quad(&mut verts, wx0, wy0, wx1, wy1, [0.08, 0.10, 0.14, 0.98]);

            draw_text_centered(&mut verts, "GLB VOXEL IMPORT SETTINGS", 0.0, 0.54, 1.25, aspect, [0.3, 0.9, 1.0, 1.0]);

            let file_label = glb_settings.selected_file.as_ref()
                .and_then(|f| f.file_name())
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "NO FILE SELECTED".into());
            let short_file = if file_label.len() > 18 { format!("{}...", &file_label[..15]) } else { file_label };

            add_quad(&mut verts, wx0 + 0.04, 0.38, wx1 - 0.22, 0.46, [0.05, 0.06, 0.09, 1.0]);
            draw_text(&mut verts, &format!("FILE: {}", short_file), wx0 + 0.06, 0.42, 0.95, aspect, [0.85, 0.85, 0.4, 1.0]);

            add_quad(&mut verts, wx1 - 0.20, 0.38, wx1 - 0.04, 0.46, [0.25, 0.35, 0.50, 1.0]);
            draw_text_centered(&mut verts, "CHANGE", wx1 - 0.12, 0.42, 0.95, aspect, [1.0, 1.0, 1.0, 1.0]);

            draw_text(&mut verts, "TARGET HEIGHT (BLOCKS):", wx0 + 0.04, 0.29, 1.0, aspect, [0.8, 0.8, 0.8, 1.0]);
            add_quad(&mut verts, wx0 + 0.04, 0.19, wx0 + 0.14, 0.27, [0.25, 0.30, 0.40, 1.0]);
            draw_text_centered(&mut verts, "- 4", wx0 + 0.09, 0.23, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);

            let h_str = format!("{:.0} BLOCKS", glb_settings.target_height);
            draw_text_centered(&mut verts, &h_str, 0.0, 0.23, 1.1, aspect, [0.3, 0.9, 1.0, 1.0]);

            add_quad(&mut verts, wx1 - 0.14, 0.19, wx1 - 0.04, 0.27, [0.25, 0.30, 0.40, 1.0]);
            draw_text_centered(&mut verts, "+ 4", wx1 - 0.09, 0.23, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);

            draw_text(&mut verts, "VOXEL RESOLUTION:", wx0 + 0.04, 0.09, 1.0, aspect, [0.8, 0.8, 0.8, 1.0]);
            add_quad(&mut verts, wx0 + 0.04, -0.01, wx0 + 0.14, 0.07, [0.25, 0.30, 0.40, 1.0]);
            draw_text_centered(&mut verts, "/ 2", wx0 + 0.09, 0.03, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);

            let vs_str = format!("{:.3}", glb_settings.voxel_size);
            draw_text_centered(&mut verts, &vs_str, 0.0, 0.03, 1.1, aspect, [0.3, 0.9, 1.0, 1.0]);

            add_quad(&mut verts, wx1 - 0.14, -0.01, wx1 - 0.04, 0.07, [0.25, 0.30, 0.40, 1.0]);
            draw_text_centered(&mut verts, "* 2", wx1 - 0.09, 0.03, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);

            draw_text(&mut verts, "PLACEMENT ANCHOR:", wx0 + 0.04, -0.09, 1.0, aspect, [0.8, 0.8, 0.8, 1.0]);
            add_quad(&mut verts, wx0 + 0.04, -0.19, wx1 - 0.04, -0.11, [0.18, 0.24, 0.34, 1.0]);
            let anchor_txt = if glb_settings.place_at_aim { "CROSSHAIR / RAYCAST AIM" } else { "AT PLAYER POSITION" };
            draw_text_centered(&mut verts, anchor_txt, 0.0, -0.15, 1.0, aspect, [1.0, 1.0, 1.0, 1.0]);

            let has_file = glb_settings.selected_file.is_some();
            let load_btn_col = if has_file { [0.20, 0.60, 0.30, 1.0] } else { [0.20, 0.25, 0.25, 0.6] };
            add_quad(&mut verts, wx0 + 0.04, -0.34, wx1 - 0.04, -0.24, load_btn_col);
            draw_text_centered(&mut verts, "VOXELIZE & INSERT", 0.0, -0.29, 1.15, aspect, [1.0, 1.0, 1.0, 1.0]);

            add_quad(&mut verts, wx0 + 0.04, -0.48, wx1 - 0.04, -0.38, [0.45, 0.22, 0.22, 1.0]);
            draw_text_centered(&mut verts, "CANCEL (ESC)", 0.0, -0.43, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
        }

        ActiveMenu::Voxelizing => {
            add_quad(&mut verts, -1.0, -1.0, 1.0, 1.0, [0.02, 0.03, 0.05, 0.88]);

            let bx0 = -0.44; let bx1 = 0.44;
            let by0 = -0.24; let by1 = 0.26;
            add_quad(&mut verts, bx0 - 0.006, by0 - 0.006, bx1 + 0.006, by1 + 0.006, [0.25, 0.45, 0.70, 1.0]);
            add_quad(&mut verts, bx0, by0, bx1, by1, [0.08, 0.10, 0.15, 0.98]);

            draw_text_centered(&mut verts, "VOXELIZING 3D MODEL", 0.0, 0.18, 1.25, aspect, [0.3, 0.9, 1.0, 1.0]);
            draw_text_centered(&mut verts, progress_stage, 0.0, 0.09, 0.95, aspect, [0.85, 0.85, 0.9, 1.0]);

            let bar_x0 = -0.38;
            let bar_x1 = 0.38;
            let bar_y0 = -0.05;
            let bar_y1 = 0.04;
            add_quad(&mut verts, bar_x0 - 0.004, bar_y0 - 0.004, bar_x1 + 0.004, bar_y1 + 0.004, [0.20, 0.25, 0.35, 1.0]);
            add_quad(&mut verts, bar_x0, bar_y0, bar_x1, bar_y1, [0.04, 0.05, 0.07, 1.0]);

            let fill_w = (bar_x1 - bar_x0) * progress_val.clamp(0.0, 1.0);
            if fill_w > 0.001 {
                add_quad(&mut verts, bar_x0, bar_y0, bar_x0 + fill_w, bar_y1, [0.20, 0.75, 0.90, 1.0]);
            }

            let pct_text = format!("{:.0}%", (progress_val * 100.0).clamp(0.0, 100.0));
            draw_text_centered(&mut verts, &pct_text, 0.0, -0.10, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            draw_text_centered(&mut verts, "MULTI-THREAD WORKER ACTIVE", 0.0, -0.18, 0.85, aspect, [0.5, 0.8, 0.6, 0.9]);
        }
    }

    verts
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    ui_pipeline: wgpu::RenderPipeline,
    ui_vertex_buffer: wgpu::Buffer,
    ui_vertices_count: u32,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    depth_texture_view: wgpu::TextureView,
    window: Arc<Window>,
    camera: Camera,
    input: InputState,
    chunk_manager: ChunkManager,
    play_mode: PlayMode,
    velocity: Vec3,
    selected_slot: usize,
    hotbar_colors: [[f32; 3]; 10],
    palette: Arc<RwLock<Palette>>,
    active_menu: ActiveMenu,
    cursor_pos: [f32; 2],
    active_slider: Option<usize>,
    edit_size: f32,
    last_target: Option<[f32; 3]>,
    cursor_free: bool,
    gimbal_dragging: bool,
    gimbal_drag_moved: bool,
    prev_cursor_pos: [f32; 2],
    glb_settings: GlbImportSettings,
    voxelize_rx: Option<mpsc::Receiver<VoxelizeMsg>>,
    voxelize_progress: f32,
    voxelize_stage: String,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let mut size = window.inner_size();
        if size.width == 0 || size.height == 0 { size = winit::dpi::PhysicalSize::new(1280, 720); }

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { 
            power_preference: wgpu::PowerPreference::default(), 
            compatible_surface: Some(&surface), 
            force_fallback_adapter: false,
            apply_limit_buckets: Default::default(),
        }).await.unwrap();
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.unwrap();
        let mut config = surface.get_default_config(&adapter, size.width, size.height).unwrap();
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);

        let camera = Camera { 
            position: Vec3::new(16.0, 12.0, 26.0), 
            yaw: -std::f32::consts::FRAC_PI_2, 
            pitch: -0.3,
            is_ortho: false,
            ortho_size: 36.0,
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[CameraUniform { 
                view_proj: camera.view_proj(1.0).to_cols_array_2d(),
                show_borders: 0.0,
                _pad: [0.0; 3],
            }]), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT, count: None, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None } }], label: None,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { layout: &camera_bind_group_layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() }], label: None });

        let depth_texture_view = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: DEPTH_FORMAT, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, label: None, view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor::default());

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&camera_bind_group_layout)], immediate_size: 0 });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None, layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[Some(Vertex::desc())] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: Some(wgpu::Face::Back), ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH_FORMAT, depth_write_enabled: Some(true), depth_compare: Some(wgpu::CompareFunction::Less), stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default() }),
            multisample: wgpu::MultisampleState::default(), multiview_mask: None, cache: None,
        });

        let ui_shader = device.create_shader_module(wgpu::include_wgsl!("ui.wgsl"));
        let ui_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[], immediate_size: 0 });
        let ui_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UI Pipeline"), layout: Some(&ui_pipeline_layout),
            vertex: wgpu::VertexState { module: &ui_shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[Some(UIVertex::desc())] },
            fragment: Some(wgpu::FragmentState { module: &ui_shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview_mask: None, cache: None,
        });

        let hotbar_colors = [
            [0.55, 0.55, 0.58],
            [0.22, 0.75, 0.32],
            [0.85, 0.20, 0.20],
            [0.20, 0.48, 0.90],
            [0.95, 0.78, 0.18],
            [0.15, 0.85, 0.85],
            [0.82, 0.25, 0.85],
            [0.95, 0.50, 0.15],
            [0.95, 0.95, 0.95],
            [0.20, 0.22, 0.25],
        ];

        let palette = Arc::new(RwLock::new(hotbar_colors.to_vec()));
        let chunk_manager = ChunkManager::new(Arc::clone(&palette));
        let glb_settings = GlbImportSettings::default();

        let initial_ui = build_ui_vertices(
            0, ActiveMenu::None, &hotbar_colors, PlayMode::Flying, camera.is_ortho, chunk_manager.world_type, 1.0, None, 1.0,
            camera.forward(), camera.right(), camera.up(), false, &glb_settings, 0.0, "",
        );
        let ui_vertices_count = initial_ui.len() as u32;
        let ui_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UI Buffer"),
            size: (65536 * std::mem::size_of::<UIVertex>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&ui_vertex_buffer, 0, bytemuck::cast_slice(&initial_ui));

        Self {
            window, surface, device, queue, config, size, render_pipeline, ui_pipeline, ui_vertex_buffer, ui_vertices_count,
            camera_buffer, camera_bind_group, depth_texture_view, camera,
            input: InputState::default(), chunk_manager,
            play_mode: PlayMode::Flying, velocity: Vec3::ZERO, selected_slot: 0, hotbar_colors,
            palette, active_menu: ActiveMenu::None, cursor_pos: [0.0, 0.0], active_slider: None, edit_size: 1.0,
            last_target: None, cursor_free: false, gimbal_dragging: false, gimbal_drag_moved: false, prev_cursor_pos: [0.0, 0.0],
            glb_settings,
            voxelize_rx: None,
            voxelize_progress: 0.0,
            voxelize_stage: String::new(),
        }
    }

    pub fn prompt_native_file_dialog(&mut self) {
        let file = rfd::FileDialog::new()
            .add_filter("3D Models (*.glb, *.gltf)", &["glb", "gltf"])
            .pick_file();

        if let Some(path) = file {
            self.glb_settings.selected_file = Some(path);
            self.set_menu(ActiveMenu::ImportParams);
        }
    }

    pub fn start_nonblocking_voxelization(&mut self) {
        let path = match self.glb_settings.selected_file.clone() {
            Some(p) => p,
            None => return,
        };

        let origin = if self.glb_settings.place_at_aim {
            self.last_target
                .map(Vec3::from)
                .unwrap_or_else(|| self.camera.position + self.camera.forward() * 12.0)
        } else {
            self.camera.position - Vec3::new(0.0, 1.6, 0.0)
        };

        let voxel_size = self.glb_settings.voxel_size.max(MIN_VOXEL_SIZE);
        let target_height = self.glb_settings.target_height;

        let (tx, rx) = mpsc::channel();
        self.voxelize_rx = Some(rx);
        self.voxelize_progress = 0.0;
        self.voxelize_stage = "INITIALIZING THREAD POOL...".into();
        self.set_menu(ActiveMenu::Voxelizing);

        let pal_arc = Arc::clone(&self.palette);
        std::thread::spawn(move || {
            run_background_voxelization(path, origin, voxel_size, target_height, pal_arc, tx);
        });
    }

    pub fn focus_on_scene(&mut self) {
        let mut min_bound = Vec3::splat(f32::MAX);
        let mut max_bound = Vec3::splat(f32::MIN);
        let mut has_blocks = false;

        for (pos, chunk) in &self.chunk_manager.loaded_chunks {
            if chunk.num_indices > 0 {
                has_blocks = true;
                let c_min = Vec3::new((pos.0 * 32) as f32, (pos.1 * 32) as f32, (pos.2 * 32) as f32);
                min_bound = min_bound.min(c_min);
                max_bound = max_bound.max(c_min + Vec3::splat(32.0));
            }
        }

        let (center, radius) = if has_blocks {
            let center = (min_bound + max_bound) * 0.5;
            let radius = ((max_bound - min_bound).length() * 0.5).max(4.0);
            (center, radius)
        } else {
            (Vec3::new(16.0, 0.0, 16.0), 16.0)
        };

        let dist = (radius * 2.2).clamp(10.0, 2000.0);
        let fwd = self.camera.forward();
        self.camera.position = center - fwd * dist;
        self.camera.ortho_size = (radius * 2.5).clamp(8.0, 2000.0);
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
        let show_borders = if self.cursor_free || self.active_menu == ActiveMenu::Edit { 1.0 } else { 0.0 };
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[CameraUniform {
            view_proj: self.camera.view_proj(aspect).to_cols_array_2d(),
            show_borders,
            _pad: [0.0; 3],
        }]));
    }

    pub fn set_cursor_mode(&mut self, free: bool) {
        self.cursor_free = free;
        if free || self.active_menu != ActiveMenu::None {
            let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::None);
            self.window.set_cursor_visible(true);
            self.input = InputState::default();
        } else {
            let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::Locked)
                .or_else(|_| self.window.set_cursor_grab(winit::window::CursorGrabMode::Confined));
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
        let data = SaveData {
            player_pos: self.camera.position.to_array(),
            camera_yaw: self.camera.yaw,
            camera_pitch: self.camera.pitch,
            is_ortho: self.camera.is_ortho,
            ortho_size: self.camera.ortho_size,
            play_mode: self.play_mode,
            world_type: self.chunk_manager.world_type,
            seed: self.chunk_manager.seed,
            hotbar_colors: self.hotbar_colors,
            palette: self.palette.read().unwrap().clone(),
            cube_edits: self.chunk_manager.cube_edits.clone(),
        };

        let json = serde_json::to_string_pretty(&data)?;
        std::fs::write(filename, json)?;
        println!("Game saved to {}", filename);
        Ok(())
    }

    pub fn load_game(&mut self, filename: &str) -> std::io::Result<()> {
        let content = std::fs::read_to_string(filename)?;
        let data: SaveData = serde_json::from_str(&content)?;

        self.camera.position = Vec3::from_array(data.player_pos);
        self.camera.yaw = data.camera_yaw;
        self.camera.pitch = data.camera_pitch;
        self.camera.is_ortho = data.is_ortho;
        self.camera.ortho_size = if data.ortho_size > 0.1 { data.ortho_size } else { 36.0 };
        self.play_mode = data.play_mode;
        self.velocity = Vec3::ZERO;
        self.hotbar_colors = data.hotbar_colors;
        *self.palette.write().unwrap() = data.palette;

        self.chunk_manager.world_type = data.world_type;
        self.chunk_manager.seed = data.seed;
        self.chunk_manager.cube_edits = data.cube_edits;

        self.chunk_manager.loaded_chunks.clear();
        self.chunk_manager.loading_chunks.clear();
        self.update_camera_buffer();
        self.update_ui();
        println!("Game loaded from {}", filename);
        Ok(())
    }

    pub fn clear_all_blocks(&mut self) {
        self.chunk_manager.cube_edits.clear();
        self.chunk_manager.loaded_chunks.clear();
        self.chunk_manager.loading_chunks.clear();
        self.update_ui();
    }

    pub fn cycle_world_generator(&mut self) {
        self.chunk_manager.world_type = self.chunk_manager.world_type.next();
        self.clear_all_blocks();
    }

    pub fn scale_voxel_size(&mut self, multiply: bool) {
        if multiply {
            self.edit_size = (self.edit_size * 2.0).min(64.0);
        } else {
            self.edit_size = (self.edit_size * 0.5).max(MIN_VOXEL_SIZE);
        }
        self.update_ui();
    }

    fn toggle_play_mode(&mut self) {
        self.play_mode = if self.play_mode == PlayMode::Real { PlayMode::Flying } else { PlayMode::Real };
        self.velocity = Vec3::ZERO;
        self.update_ui();
    }

    fn get_or_create_material(&self, color: [f32; 3]) -> u16 {
        let mut pal = self.palette.write().unwrap();
        for (i, &c) in pal.iter().enumerate() {
            if (c[0] - color[0]).abs() < 0.005 
               && (c[1] - color[1]).abs() < 0.005 
               && (c[2] - color[2]).abs() < 0.005 {
                return (i + 1) as u16;
            }
        }
        pal.push(color);
        pal.len() as u16
    }

    fn update_ui(&mut self) {
        let aspect = if self.config.height > 0 { self.config.width as f32 / self.config.height as f32 } else { 1.0 };
        let verts = build_ui_vertices(
            self.selected_slot,
            self.active_menu,
            &self.hotbar_colors,
            self.play_mode,
            self.camera.is_ortho,
            self.chunk_manager.world_type,
            self.edit_size,
            self.last_target,
            aspect,
            self.camera.forward(),
            self.camera.right(),
            self.camera.up(),
            self.cursor_free,
            &self.glb_settings,
            self.voxelize_progress,
            &self.voxelize_stage,
        );
        self.ui_vertices_count = verts.len() as u32;
        self.queue.write_buffer(&self.ui_vertex_buffer, 0, bytemuck::cast_slice(&verts));
    }

    fn player_collides_at(&self, pos: Vec3) -> bool {
        let r = 0.35;
        let heights = [-1.55, -0.75, 0.15];
        for &dy in &heights {
            for &dx in &[-r, r] {
                for &dz in &[-r, r] {
                    if self.chunk_manager.is_solid(pos + Vec3::new(dx, dy, dz)) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size; 
            self.config.width = new_size.width; 
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture_view = self.device.create_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d { width: self.config.width, height: self.config.height, depth_or_array_layers: 1 }, 
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, 
                format: DEPTH_FORMAT, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, label: None, view_formats: &[],
            }).create_view(&wgpu::TextureViewDescriptor::default());
            self.update_camera_buffer();
        }
    }

    fn update(&mut self, dt: f32) {
        let mut messages = Vec::new();
        if let Some(ref rx) = self.voxelize_rx {
            while let Ok(msg) = rx.try_recv() {
                messages.push(msg);
            }
        }

        for msg in messages {
            match msg {
                VoxelizeMsg::Progress { percent, stage } => {
                    self.voxelize_progress = percent;
                    self.voxelize_stage = stage;
                    self.update_ui();
                }
                VoxelizeMsg::Done(Ok(chunks)) => {
                    for item in chunks {
                        let chunk = self.chunk_manager.loaded_chunks.entry(item.pos).or_insert_with(|| {
                            let vb = self.device.create_buffer(&wgpu::BufferDescriptor {
                                label: None, size: 1024, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
                            });
                            let ib = self.device.create_buffer(&wgpu::BufferDescriptor {
                                label: None, size: 1024, usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
                            });
                            RenderChunk { octree: item.octree.clone(), vertex_buffer: vb, index_buffer: ib, num_indices: 0, vertex_capacity: 1024, index_capacity: 1024 }
                        });
                        chunk.octree = item.octree;
                        update_chunk_buffers(chunk, &item.mesh, &self.device, &self.queue);
                    }
                    self.voxelize_rx = None;
                    self.set_menu(ActiveMenu::None);
                    self.focus_on_scene();
                    return;
                }
                VoxelizeMsg::Done(Err(err)) => {
                    eprintln!("GLB voxelization failed: {}", err);
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
        if self.input.forward { movement += forward; } if self.input.backward { movement -= forward; }
        if self.input.right { movement += right; } if self.input.left { movement -= right; }

        match self.play_mode {
            PlayMode::Flying => {
                let speed = 28.0;
                if self.input.up { movement.y += 1.0; } if self.input.down { movement.y -= 1.0; }
                if movement.length_squared() > 0.0 { movement = movement.normalize(); }
                self.camera.position += movement * speed * dt;
            },
            PlayMode::Real => {
                let surface_y = self.chunk_manager.get_surface_y(self.camera.position.x, self.camera.position.z);

                let void_threshold = surface_y.map(|sy| sy - 15.0).unwrap_or(-10.0);
                if self.camera.position.y < void_threshold {
                    self.camera.position.y = surface_y.unwrap_or(12.0) + 1.8;
                    self.velocity = Vec3::ZERO;
                }

                if self.player_collides_at(self.camera.position) {
                    let mut freed = false;
                    for _ in 0..80 {
                        self.camera.position.y += 0.25;
                        if !self.player_collides_at(self.camera.position) {
                            freed = true;
                            break;
                        }
                    }
                    if !freed {
                        if let Some(top_y) = surface_y {
                            self.camera.position.y = top_y + 1.8;
                        }
                    }
                    self.velocity.y = 0.0;
                }

                let walk_speed = 6.5;
                if movement.length_squared() > 0.0 { movement = movement.normalize(); }
                
                let dx = movement.x * walk_speed * dt;
                if !self.player_collides_at(self.camera.position + Vec3::new(dx, 0.0, 0.0)) {
                    self.camera.position.x += dx;
                }
                
                let dz = movement.z * walk_speed * dt;
                if !self.player_collides_at(self.camera.position + Vec3::new(0.0, 0.0, dz)) {
                    self.camera.position.z += dz;
                }

                self.velocity.y -= 38.0 * dt;
                let on_ground = self.player_collides_at(self.camera.position - Vec3::new(0.0, 0.08, 0.0));
                if self.input.up && on_ground { 
                    self.velocity.y = 11.5; 
                }

                let total_dy = self.velocity.y * dt;
                let step_count = ((total_dy.abs() / 0.08).ceil() as i32).max(1);
                let step_dy = total_dy / step_count as f32;

                for _ in 0..step_count {
                    if self.player_collides_at(self.camera.position + Vec3::new(0.0, step_dy, 0.0)) {
                        self.velocity.y = 0.0;
                        break;
                    } else {
                        self.camera.position.y += step_dy;
                    }
                }
            }
        }

        let dir = self.camera.forward();
        let hit = self.chunk_manager.raycast(self.camera.position, dir, 300.0);

        let s = self.edit_size;
        if let Some(ref h) = hit {
            let p = compute_placement_pos(h, s);
            self.last_target = Some([p.x, p.y, p.z]);
        } else {
            self.last_target = None;
        }

        if !self.cursor_free && (self.input.action_add || self.input.action_remove || self.input.action_pick) {
            if let Some(ref h) = hit {
                if self.input.action_pick {
                    let mat = h.material;
                    let pal = self.palette.read().unwrap();
                    if let Some(&color) = pal.get((mat - 1) as usize) {
                        self.hotbar_colors[self.selected_slot] = color;
                        drop(pal);
                        self.update_ui();
                    }
                } else if self.input.action_remove { 
                    let p_inside = h.hit_pos - h.normal * (MIN_VOXEL_SIZE * 0.5);
                    let remove_pos = Vec3::new(
                        (p_inside.x / s).floor() * s,
                        (p_inside.y / s).floor() * s,
                        (p_inside.z / s).floor() * s,
                    );
                    self.chunk_manager.modify_cube(remove_pos, 0, s, &self.device, &self.queue); 
                } else if self.input.action_add { 
                    let place_pos = compute_placement_pos(h, s);
                    if self.play_mode == PlayMode::Flying || !self.player_collides_at(self.camera.position) {
                        let mat_id = self.get_or_create_material(self.hotbar_colors[self.selected_slot]);
                        self.chunk_manager.modify_cube(place_pos, mat_id, s, &self.device, &self.queue); 
                    }
                }
            } else if self.input.action_add {
                let spawn_dist = (s * 2.0).clamp(8.0, 64.0);
                let p = self.camera.position + dir * spawn_dist;
                let place_pos = Vec3::new(
                    (p.x / s).floor() * s,
                    (p.y / s).floor() * s,
                    (p.z / s).floor() * s,
                );
                let mat_id = self.get_or_create_material(self.hotbar_colors[self.selected_slot]);
                self.chunk_manager.modify_cube(place_pos, mat_id, s, &self.device, &self.queue);
            }

            self.input.action_add = false; 
            self.input.action_remove = false;
            self.input.action_pick = false;
        }

        self.update_camera_buffer();
        self.chunk_manager.update(self.camera.position, &self.device);
        self.update_ui();
    }

    fn render(&mut self) {
        let mut surface_texture = self.surface.get_current_texture();
        if matches!(surface_texture, wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost) {
            self.surface.configure(&self.device, &self.config);
            surface_texture = self.surface.get_current_texture();
        }
        let frame = match surface_texture { wgpu::CurrentSurfaceTexture::Success(frame) | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame, _ => return };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.12, g: 0.14, b: 0.18, a: 1.0 }), store: wgpu::StoreOp::Store }, depth_slice: None })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &self.depth_texture_view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            for chunk in self.chunk_manager.loaded_chunks.values() {
                if chunk.num_indices > 0 {
                    render_pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..chunk.num_indices, 0, 0..1);
                }
            }
        }

        {
            let mut ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }, depth_slice: None })],
                depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
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
                .with_title("Voxel Studio - 3D PixelArt Engine")
                .with_inner_size(LogicalSize::new(1280.0, 720.0))
                .with_visible(true);
            #[cfg(target_os = "linux")]
            {
                window_attributes = WindowAttributesExtWayland::with_name(window_attributes, "octree_voxels", "octree_voxels");
                window_attributes = WindowAttributesExtX11::with_name(window_attributes, "octree_voxels", "octree_voxels");
            }

            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
            let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Locked).or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Confined));
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
                            let sl_x0 = -0.32;
                            let sl_x1 = 0.18;
                            let val = ((ndc_x - sl_x0) / (sl_x1 - sl_x0)).clamp(0.0, 1.0);
                            state.hotbar_colors[state.selected_slot][channel] = val;
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
                        if state.camera.is_ortho && state.cursor_free {
                            let zoom = if step < 0 { 0.88 } else { 1.14 };
                            state.camera.ortho_size = (state.camera.ortho_size * zoom).clamp(2.0, 1200.0);
                            state.update_camera_buffer();
                            state.update_ui();
                        } else {
                            state.selected_slot = (state.selected_slot as i32 + step).rem_euclid(10) as usize;
                            state.update_ui();
                        }
                    }
                }
                WindowEvent::KeyboardInput { event: key_event, .. } => {
                    let is_pressed = key_event.state == ElementState::Pressed;

                    if key_event.physical_key == PhysicalKey::Code(KeyCode::Tab) && is_pressed {
                        let new_free = !state.cursor_free;
                        state.set_cursor_mode(new_free);
                        return;
                    }

                    if (key_event.physical_key == PhysicalKey::Code(KeyCode::NumpadDecimal)
                        || key_event.physical_key == PhysicalKey::Code(KeyCode::Period))
                        && is_pressed
                    {
                        state.focus_on_scene();
                        return;
                    }

                    if (key_event.physical_key == PhysicalKey::Code(KeyCode::KeyP) 
                        || key_event.physical_key == PhysicalKey::Code(KeyCode::Numpad5))
                        && is_pressed 
                    {
                        state.toggle_projection();
                        return;
                    }

                    if is_pressed && state.camera.is_ortho {
                        match key_event.physical_key {
                            PhysicalKey::Code(KeyCode::NumpadAdd) | PhysicalKey::Code(KeyCode::Equal) => {
                                state.camera.ortho_size = (state.camera.ortho_size * 0.85).clamp(2.0, 1200.0);
                                state.update_camera_buffer();
                                state.update_ui();
                                return;
                            },
                            PhysicalKey::Code(KeyCode::NumpadSubtract) | PhysicalKey::Code(KeyCode::Minus) => {
                                state.camera.ortho_size = (state.camera.ortho_size * 1.15).clamp(2.0, 1200.0);
                                state.update_camera_buffer();
                                state.update_ui();
                                return;
                            },
                            _ => {}
                        }
                    }

                    if key_event.physical_key == PhysicalKey::Code(KeyCode::KeyE) && is_pressed {
                        if state.active_menu == ActiveMenu::Edit {
                            state.set_menu(ActiveMenu::None);
                        } else if state.active_menu == ActiveMenu::None {
                            state.set_menu(ActiveMenu::Edit);
                        }
                        return;
                    }

                    if key_event.physical_key == PhysicalKey::Code(KeyCode::Escape) && is_pressed {
                        match state.active_menu {
                            ActiveMenu::None => state.set_menu(ActiveMenu::Pause),
                            ActiveMenu::ImportParams => state.set_menu(ActiveMenu::Pause),
                            ActiveMenu::Voxelizing => {},
                            _ => state.set_menu(ActiveMenu::None),
                        }
                        return;
                    }

                    if is_pressed {
                        match key_event.physical_key {
                            PhysicalKey::Code(KeyCode::F2) => state.cycle_world_generator(),
                            PhysicalKey::Code(KeyCode::F5) => { let _ = state.save_game("world_save.json"); },
                            PhysicalKey::Code(KeyCode::F9) => { let _ = state.load_game("world_save.json"); },
                            PhysicalKey::Code(KeyCode::KeyM) => state.toggle_play_mode(),
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
                            PhysicalKey::Code(KeyCode::KeyC) if state.active_menu == ActiveMenu::None && !state.cursor_free => {
                                state.input.action_pick = true;
                            },
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
                                if mx >= bx0 && mx <= bx1 && my >= by0 && my <= by1 {
                                    state.focus_on_scene();
                                    return;
                                }
                            }

                            if state.cursor_free || state.active_menu != ActiveMenu::None {
                                let g_cx = GIZMO_CENTER_X;
                                let g_cy = GIZMO_CENTER_Y;
                                let dist_sq = ((mx - g_cx) * aspect).powi(2) + (my - g_cy).powi(2);

                                if dist_sq <= (GIZMO_RADIUS + 0.02).powi(2) {
                                    state.gimbal_dragging = true;
                                    state.gimbal_drag_moved = false;
                                    return;
                                }
                            }
                        } else if element_state == ElementState::Released {
                            if state.gimbal_dragging {
                                state.gimbal_dragging = false;
                                if !state.gimbal_drag_moved {
                                    let g_cx = GIZMO_CENTER_X;
                                    let g_cy = GIZMO_CENTER_Y;
                                    let right = state.camera.right();
                                    let up = state.camera.up();

                                    let mut sorted_axes = get_gizmo_axes();
                                    sorted_axes.sort_by(|a, b| {
                                        let da = a.dir.dot(state.camera.forward());
                                        let db = b.dir.dot(state.camera.forward());
                                        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
                                    });

                                    for ax in sorted_axes {
                                        let sx = ax.dir.dot(right);
                                        let sy = ax.dir.dot(up);
                                        let tip_x = g_cx + (sx * GIZMO_RADIUS) / aspect;
                                        let tip_y = g_cy + (sy * GIZMO_RADIUS);
                                        let dist_sq = ((mx - tip_x) * aspect).powi(2) + (my - tip_y).powi(2);

                                        if dist_sq <= 0.035 * 0.035 {
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
                    }

                    match state.active_menu {
                        ActiveMenu::Edit => {
                            if button == MouseButton::Left {
                                if element_state == ElementState::Pressed {
                                    let sw_w = 0.082;
                                    let sw_gap = 0.015;
                                    let sw_tot = 10.0 * sw_w + 9.0 * sw_gap;
                                    let s_start_x = -sw_tot / 2.0;
                                    for i in 0..10 {
                                        let sx0 = s_start_x + i as f32 * (sw_w + sw_gap);
                                        let sx1 = sx0 + sw_w;
                                        if mx >= sx0 && mx <= sx1 && my >= 0.43 && my <= 0.52 {
                                            state.selected_slot = i;
                                            state.update_ui();
                                            return;
                                        }
                                    }

                                    let sl_x0 = -0.32;
                                    let sl_x1 = 0.18;
                                    if mx >= sl_x0 - 0.02 && mx <= sl_x1 + 0.02 {
                                        let slider_clicked = if my >= 0.30 && my <= 0.38 {
                                            Some(0)
                                        } else if my >= 0.23 && my <= 0.31 {
                                            Some(1)
                                        } else if my >= 0.16 && my <= 0.24 {
                                            Some(2)
                                        } else {
                                            None
                                        };

                                        if let Some(channel) = slider_clicked {
                                            state.active_slider = Some(channel);
                                            let val = ((mx - sl_x0) / (sl_x1 - sl_x0)).clamp(0.0, 1.0);
                                            state.hotbar_colors[state.selected_slot][channel] = val;
                                            state.update_ui();
                                            return;
                                        }
                                    }

                                    let pw_w = 0.076;
                                    let pw_gap = 0.012;
                                    let pw_tot = 10.0 * pw_w + 9.0 * pw_gap;
                                    let pw_start_x = -pw_tot / 2.0;
                                    for (i, &preset_col) in PRESET_SWATCHES.iter().enumerate() {
                                        let px0 = pw_start_x + i as f32 * (pw_w + pw_gap);
                                        let px1 = px0 + pw_w;
                                        if mx >= px0 && mx <= px1 && my >= 0.05 && my <= 0.11 {
                                            state.hotbar_colors[state.selected_slot] = preset_col;
                                            state.update_ui();
                                            return;
                                        }
                                    }

                                    if mx >= -0.40 && mx <= -0.22 && my >= -0.10 && my <= -0.02 {
                                        state.scale_voxel_size(false);
                                        return;
                                    }
                                    if mx >= 0.22 && mx <= 0.40 && my >= -0.10 && my <= -0.02 {
                                        state.scale_voxel_size(true);
                                        return;
                                    }

                                    if mx >= -0.22 && mx <= 0.22 && my >= -0.25 && my <= -0.17 {
                                        state.set_menu(ActiveMenu::None);
                                        return;
                                    }
                                } else {
                                    state.active_slider = None;
                                }
                            }
                        }

                        ActiveMenu::Pause => {
                            if button == MouseButton::Left && element_state == ElementState::Pressed {
                                if mx >= -0.30 && mx <= 0.30 && my >= 0.46 && my <= 0.54 {
                                    state.prompt_native_file_dialog();
                                    return;
                                }
                                if mx >= -0.30 && mx <= 0.30 && my >= 0.36 && my <= 0.44 {
                                    state.toggle_play_mode();
                                    return;
                                }
                                if mx >= -0.30 && mx <= 0.30 && my >= 0.26 && my <= 0.34 {
                                    state.toggle_projection();
                                    return;
                                }
                                if mx >= -0.30 && mx <= 0.30 && my >= 0.16 && my <= 0.24 {
                                    state.cycle_world_generator();
                                    return;
                                }
                                if mx >= -0.30 && mx <= 0.30 && my >= 0.06 && my <= 0.14 {
                                    state.clear_all_blocks();
                                    return;
                                }
                                if mx >= -0.30 && mx <= -0.02 && my >= -0.06 && my <= 0.02 {
                                    let _ = state.save_game("world_save.json");
                                    return;
                                }
                                if mx >= 0.02 && mx <= 0.30 && my >= -0.06 && my <= 0.02 {
                                    let _ = state.load_game("world_save.json");
                                    return;
                                }
                                if mx >= -0.30 && mx <= 0.30 && my >= -0.18 && my <= -0.10 {
                                    state.set_menu(ActiveMenu::None);
                                    return;
                                }
                                if mx >= -0.30 && mx <= 0.30 && my >= -0.30 && my <= -0.22 {
                                    event_loop.exit();
                                    return;
                                }
                            }
                        }

                        ActiveMenu::ImportParams => {
                            if button == MouseButton::Left && element_state == ElementState::Pressed {
                                let wx0 = -0.42; let wx1 = 0.42;

                                if mx >= wx1 - 0.20 && mx <= wx1 - 0.04 && my >= 0.38 && my <= 0.46 {
                                    state.prompt_native_file_dialog();
                                    return;
                                }

                                if my >= 0.19 && my <= 0.27 {
                                    if mx >= wx0 + 0.04 && mx <= wx0 + 0.14 {
                                        state.glb_settings.target_height = (state.glb_settings.target_height - 4.0).max(4.0);
                                        state.update_ui();
                                        return;
                                    }
                                    if mx >= wx1 - 0.14 && mx <= wx1 - 0.04 {
                                        state.glb_settings.target_height = (state.glb_settings.target_height + 4.0).min(64.0);
                                        state.update_ui();
                                        return;
                                    }
                                }

                                if my >= -0.01 && my <= 0.07 {
                                    if mx >= wx0 + 0.04 && mx <= wx0 + 0.14 {
                                        state.glb_settings.voxel_size = (state.glb_settings.voxel_size * 0.5).max(MIN_VOXEL_SIZE);
                                        state.update_ui();
                                        return;
                                    }
                                    if mx >= wx1 - 0.14 && mx <= wx1 - 0.04 {
                                        state.glb_settings.voxel_size = (state.glb_settings.voxel_size * 2.0).min(2.0);
                                        state.update_ui();
                                        return;
                                    }
                                }

                                if mx >= wx0 + 0.04 && mx <= wx1 - 0.04 && my >= -0.19 && my <= -0.11 {
                                    state.glb_settings.place_at_aim = !state.glb_settings.place_at_aim;
                                    state.update_ui();
                                    return;
                                }

                                if mx >= wx0 + 0.04 && mx <= wx1 - 0.04 && my >= -0.34 && my <= -0.24 {
                                    state.start_nonblocking_voxelization();
                                    return;
                                }

                                if mx >= wx0 + 0.04 && mx <= wx1 - 0.04 && my >= -0.48 && my <= -0.38 {
                                    state.set_menu(ActiveMenu::Pause);
                                    return;
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
        if let Some(state) = self.state.as_mut() { 
            state.window.request_redraw(); 
        } 
    }
}

pub fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { state: None, last_frame: Instant::now() };
    event_loop.run_app(&mut app).unwrap();
}