// engine.rs
use crate::types::*;
use glam::Vec3;
use noise::{NoiseFn, Fbm, Perlin};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock, mpsc};

#[derive(Clone, Copy, Default)]
pub struct OctreeNode { pub child_mask: u8, pub material_id: u16, pub child_pointer: u32 }

#[derive(Clone)]
pub struct Octree { pub nodes: Vec<OctreeNode>, pub root_index: u32 }

impl Octree {
    pub fn new() -> Self {
        let mut nodes = Vec::with_capacity(2048);
        nodes.push(OctreeNode::default());
        Self { nodes, root_index: 0 }
    }
    
    pub fn insert_cube(&mut self, mut x: u32, mut y: u32, mut z: u32, depth: u8, material: u16, auto_collapse: bool) {
        if depth == 0 {
            self.nodes[self.root_index as usize] = OctreeNode { child_mask: if material != 0 { 0xFF } else { 0 }, material_id: material, child_pointer: 0 };
            return;
        }
        let mut current_idx = self.root_index as usize; let mut half_size = GRID_RES / 2;
        for _ in 0..depth {
            let old_mat = self.nodes[current_idx].material_id; let child_ptr = self.nodes[current_idx].child_pointer;
            if child_ptr == 0 {
                if old_mat == material { return; }
                let new_ptr = self.nodes.len() as u32;
                if self.nodes.capacity() < self.nodes.len() + 8 { self.nodes.reserve(2048); }
                self.nodes.resize(self.nodes.len() + 8, OctreeNode::default());
                if old_mat != 0 {
                    for c in 0..8 { self.nodes[(new_ptr + c) as usize].material_id = old_mat; }
                    self.nodes[current_idx].child_mask = 0xFF;
                } else { self.nodes[current_idx].child_mask = 0; }
                self.nodes[current_idx].child_pointer = new_ptr; self.nodes[current_idx].material_id = 0;
            }
            let mut octant = 0;
            if x >= half_size { octant |= 1; x -= half_size; }
            if y >= half_size { octant |= 2; y -= half_size; }
            if z >= half_size { octant |= 4; z -= half_size; }
            current_idx = (self.nodes[current_idx].child_pointer + octant) as usize; half_size >>= 1;
        }
        self.nodes[current_idx].material_id = material; self.nodes[current_idx].child_pointer = 0;
        self.nodes[current_idx].child_mask = if material != 0 { 0xFF } else { 0 };
        if auto_collapse { self.collapse(self.root_index as usize); }
    }
    
    pub fn collapse(&mut self, node_idx: usize) -> bool {
        let child_ptr = self.nodes[node_idx].child_pointer; if child_ptr == 0 { return true; }
        let mut all_same = true; let first_mat = self.nodes[child_ptr as usize].material_id;
        for i in 0..8 {
            let c_idx = (child_ptr + i) as usize; let is_leaf = self.collapse(c_idx);
            if !is_leaf || self.nodes[c_idx].material_id != first_mat { all_same = false; }
        }
        if all_same {
            self.nodes[node_idx].material_id = first_mat; self.nodes[node_idx].child_pointer = 0;
            self.nodes[node_idx].child_mask = if first_mat != 0 { 0xFF } else { 0 };
            return true;
        }
        let mut mask = 0;
        for i in 0..8 { let c_idx = (child_ptr + i) as usize; if self.nodes[c_idx].material_id != 0 || self.nodes[c_idx].child_pointer != 0 { mask |= 1 << i; } }
        self.nodes[node_idx].child_mask = mask; false
    }
    
    pub fn query_node(&self, mut x: u32, mut y: u32, mut z: u32) -> (u16, u32) {
        let mut current_idx = self.root_index as usize; let mut size = GRID_RES;
        for _ in 0..MAX_DEPTH {
            let node = &self.nodes[current_idx];
            if node.child_pointer == 0 { return (node.material_id, size); }
            let half = size / 2; let mut octant = 0;
            if x >= half { octant |= 1; x -= half; } if y >= half { octant |= 2; y -= half; } if z >= half { octant |= 4; z -= half; }
            if (node.child_mask & (1 << octant)) == 0 { return (0, half); }
            current_idx = (node.child_pointer + octant) as usize; size = half;
        }
        (self.nodes[current_idx].material_id, size)
    }
}

pub struct MeshPayload { pub vertices: Vec<Vertex>, pub indices: Vec<u32> }
pub struct ImportedChunkData { pub pos: ChunkPos, pub octree: Octree, pub mesh: MeshPayload }

pub enum VoxelizeMsg {
    Progress { percent: f32, stage: String },
    Done(Result<Vec<ImportedChunkData>, String>),
}

pub struct RenderChunk {
    pub octree: Octree, pub vertex_buffer: wgpu::Buffer, pub index_buffer: wgpu::Buffer, 
    pub num_indices: u32, pub vertex_capacity: usize, pub index_capacity: usize,
}

pub fn update_chunk_buffers(chunk: &mut RenderChunk, payload: &MeshPayload, device: &wgpu::Device, queue: &wgpu::Queue) {
    if payload.vertices.is_empty() || payload.indices.is_empty() { chunk.num_indices = 0; return; }
    let v_bytes = bytemuck::cast_slice(&payload.vertices); let i_bytes = bytemuck::cast_slice(&payload.indices);
    if v_bytes.len() > chunk.vertex_capacity {
        chunk.vertex_capacity = (v_bytes.len() * 2).max(1024);
        chunk.vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: chunk.vertex_capacity as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    }
    if i_bytes.len() > chunk.index_capacity {
        chunk.index_capacity = (i_bytes.len() * 2).max(1024);
        chunk.index_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: chunk.index_capacity as u64, usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    }
    queue.write_buffer(&chunk.vertex_buffer, 0, v_bytes); queue.write_buffer(&chunk.index_buffer, 0, i_bytes); chunk.num_indices = payload.indices.len() as u32;
}

pub fn generate_octree(chunk_pos: ChunkPos, world_type: WorldType, seed: u32, edits: &[CubeEdit], lod_level: u8) -> Octree {
    let mut octree = Octree::new();
    let fbm = Fbm::<Perlin>::new(seed);
    let world_x_offset = chunk_pos.0 * 32; let world_z_offset = chunk_pos.2 * 32;
    let depth_limit = MAX_DEPTH.saturating_sub(lod_level);

    match world_type {
        WorldType::Empty => {}
        WorldType::Flat => { if chunk_pos.1 == 0 { for x in 0..32 { for z in 0..32 { for y in 0..4 { let mat = if y == 3 { 2 } else { 1 }; octree.insert_cube(x * (GRID_RES/32), y * (GRID_RES/32), z * (GRID_RES/32), depth_limit, mat, false); } } } } }
        WorldType::Hills => {
            for x in 0..32 { for z in 0..32 {
                let wx = (x as i32 + world_x_offset) as f64 * 0.04; let wz = (z as i32 + world_z_offset) as f64 * 0.04;
                let height = ((fbm.get([wx, wz]) + 1.0) * 12.0).clamp(0.0, 31.0) as u32;
                for y in 0..=height { let mat = if y == height { 2 } else { 1 }; octree.insert_cube(x * (GRID_RES/32), y * (GRID_RES/32), z * (GRID_RES/32), depth_limit, mat, false); }
            } }
        }
        WorldType::Mountains => {
            for x in 0..32 { for z in 0..32 {
                let wx = (x as i32 + world_x_offset) as f64 * 0.025; let wz = (z as i32 + world_z_offset) as f64 * 0.025;
                let height = (fbm.get([wx, wz]).abs() * 30.0 + 2.0).clamp(0.0, 31.0) as u32;
                for y in 0..=height { let mat = if y >= 25 { 3 } else if y == height { 2 } else { 1 }; octree.insert_cube(x * (GRID_RES/32), y * (GRID_RES/32), z * (GRID_RES/32), depth_limit, mat, false); }
            } }
        }
        WorldType::FloatingIslands => {
            let world_y_offset = chunk_pos.1 * 32;
            for x in 0..32 { for y in 0..32 { for z in 0..32 {
                let wx = (x as i32 + world_x_offset) as f64 * 0.05; let wy = (y as i32 + world_y_offset) as f64 * 0.08; let wz = (z as i32 + world_z_offset) as f64 * 0.05;
                let density = fbm.get([wx, wy, wz]) - ((y as f64 + world_y_offset as f64) - 16.0) * 0.04;
                if density > 0.12 { octree.insert_cube(x * (GRID_RES/32), y * (GRID_RES/32), z * (GRID_RES/32), depth_limit, 2, false); }
            } } }
        }
    }

    for edit in edits {
        let chunk_min_x = (chunk_pos.0 * 32) as f32; let chunk_min_y = (chunk_pos.1 * 32) as f32; let chunk_min_z = (chunk_pos.2 * 32) as f32;
        if edit.pos[0] + edit.size > chunk_min_x && edit.pos[0] < chunk_min_x + 32.0 && edit.pos[1] + edit.size > chunk_min_y && edit.pos[1] < chunk_min_y + 32.0 && edit.pos[2] + edit.size > chunk_min_z && edit.pos[2] < chunk_min_z + 32.0 {
            if edit.size >= 32.0 { octree.insert_cube(0, 0, 0, 0, edit.material, false); } else {
                let lx = edit.pos[0] - chunk_min_x; let ly = edit.pos[1] - chunk_min_y; let lz = edit.pos[2] - chunk_min_z;
                let gx = ((lx / 32.0) * GRID_RES as f32).round() as u32; let gy = ((ly / 32.0) * GRID_RES as f32).round() as u32; let gz = ((lz / 32.0) * GRID_RES as f32).round() as u32;
                let grid_size = ((edit.size / 32.0) * GRID_RES as f32).round().max(1.0) as u32;
                let depth = MAX_DEPTH - grid_size.trailing_zeros().min(MAX_DEPTH as u32) as u8;
                octree.insert_cube(gx, gy, gz, depth.min(depth_limit), edit.material, false);
            }
        }
    }
    octree.collapse(octree.root_index as usize); octree
}

const QUAD_UVS: [[f32; 2]; 4] = [ [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0] ];

pub fn emit_quad(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, corners: [[f32; 3]; 4], normal: [f32; 3], color: [f32; 3]) {
    let start = vertices.len() as u32;
    for (i, &pos) in corners.iter().enumerate() { vertices.push(Vertex { position: pos, normal, color, uv: QUAD_UVS[i] }); }
    indices.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
}

pub fn is_face_occluded(octree: &Octree, gx: u32, gy: u32, gz: u32, size: u32, axis: usize, positive: bool) -> bool {
    let (target_coord, limit) = match axis {
        0 => (if positive { gx + size } else { gx.wrapping_sub(1) }, GRID_RES),
        1 => (if positive { gy + size } else { gy.wrapping_sub(1) }, GRID_RES),
        _ => (if positive { gz + size } else { gz.wrapping_sub(1) }, GRID_RES),
    };
    if target_coord >= limit { return false; }
    let (cx, cy, cz) = match axis { 0 => (target_coord, gy + size / 2, gz + size / 2), 1 => (gx + size / 2, target_coord, gz + size / 2), _ => (gx + size / 2, gy + size / 2, target_coord) };
    let (mat, n_size) = octree.query_node(cx, cy, cz);
    if mat == 0 { return false; } if n_size >= size { return true; }
    let q1 = size / 4; let q3 = (size * 3) / 4;
    let pts = match axis {
        0 => [(target_coord, gy + q1, gz + q1), (target_coord, gy + q3, gz + q1), (target_coord, gy + q1, gz + q3), (target_coord, gy + q3, gz + q3)],
        1 => [(gx + q1, target_coord, gz + q1), (gx + q3, target_coord, gz + q1), (gx + q1, target_coord, gz + q3), (gx + q3, target_coord, gz + q3)],
        _ => [(gx + q1, gy + q1, target_coord), (gx + q3, gy + q1, target_coord), (gx + q1, gy + q3, target_coord), (gx + q3, gy + q3, target_coord)],
    };
    pts.iter().all(|&(px, py, pz)| octree.query_node(px, py, pz).0 != 0)
}

pub fn mesh_octree_node(octree: &Octree, node_idx: usize, gx: u32, gy: u32, gz: u32, size: u32, chunk_pos: ChunkPos, palette: &[[f32; 3]], vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>) {
    let node = &octree.nodes[node_idx];
    if node.child_pointer != 0 {
        let half = size / 2;
        for i in 0..8 {
            if (node.child_mask & (1 << i)) != 0 {
                let cx = gx + if (i & 1) != 0 { half } else { 0 }; let cy = gy + if (i & 2) != 0 { half } else { 0 }; let cz = gz + if (i & 4) != 0 { half } else { 0 };
                mesh_octree_node(octree, (node.child_pointer + i) as usize, cx, cy, cz, half, chunk_pos, palette, vertices, indices);
            }
        }
        return;
    }
    if node.material_id == 0 { return; }
    let mat_id = node.material_id; let color = if mat_id >= 1 && (mat_id as usize) <= palette.len() { palette[(mat_id - 1) as usize] } else { [0.5, 0.5, 0.5] };
    let ox = (chunk_pos.0 * 32) as f32; let oy = (chunk_pos.1 * 32) as f32; let oz = (chunk_pos.2 * 32) as f32;
    let x0 = ox + (gx as f32 / GRID_RES as f32) * 32.0; let y0 = oy + (gy as f32 / GRID_RES as f32) * 32.0; let z0 = oz + (gz as f32 / GRID_RES as f32) * 32.0;
    let s = (size as f32 / GRID_RES as f32) * 32.0; let x1 = x0 + s; let y1 = y0 + s; let z1 = z0 + s;
    if !is_face_occluded(octree, gx, gy, gz, size, 0, true) { emit_quad(vertices, indices, [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], [1.0, 0.0, 0.0], color); }
    if !is_face_occluded(octree, gx, gy, gz, size, 0, false) { emit_quad(vertices, indices, [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], [-1.0, 0.0, 0.0], color); }
    if !is_face_occluded(octree, gx, gy, gz, size, 1, true) { emit_quad(vertices, indices, [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]], [0.0, 1.0, 0.0], color); }
    if !is_face_occluded(octree, gx, gy, gz, size, 1, false) { emit_quad(vertices, indices, [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]], [0.0, -1.0, 0.0], color); }
    if !is_face_occluded(octree, gx, gy, gz, size, 2, true) { emit_quad(vertices, indices, [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], [0.0, 0.0, 1.0], color); }
    if !is_face_occluded(octree, gx, gy, gz, size, 2, false) { emit_quad(vertices, indices, [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], [0.0, 0.0, -1.0], color); }
}

pub fn generate_mesh(octree: &Octree, chunk_pos: ChunkPos, palette: &[[f32; 3]]) -> MeshPayload {
    let mut vertices = Vec::new(); let mut indices = Vec::new();
    mesh_octree_node(octree, octree.root_index as usize, 0, 0, 0, GRID_RES, chunk_pos, palette, &mut vertices, &mut indices);
    MeshPayload { vertices, indices }
}

pub struct RaycastHit { pub hit_pos: Vec3, pub voxel_min: Vec3, pub voxel_size: f32, pub normal: Vec3, pub material: u16 }

pub fn ray_aabb_intersect(ray_origin: Vec3, ray_dir_inv: Vec3, min: Vec3, max: Vec3) -> Option<(f32, Vec3)> {
    let mut t_near = f32::NEG_INFINITY; let mut t_far = f32::INFINITY; let mut normal = Vec3::ZERO;
    for i in 0..3 {
        let mut t0 = (min[i] - ray_origin[i]) * ray_dir_inv[i];
        let mut t1 = (max[i] - ray_origin[i]) * ray_dir_inv[i];
        let mut n0 = Vec3::ZERO; n0[i] = -1.0;
        let mut n1 = Vec3::ZERO; n1[i] = 1.0;
        if t0 > t1 { std::mem::swap(&mut t0, &mut t1); std::mem::swap(&mut n0, &mut n1); }
        if t0 > t_near { t_near = t0; normal = n0; }
        t_far = t_far.min(t1);
        if t_near > t_far { return None; }
    }
    if t_far >= 0.0 { Some((t_near.max(0.0), normal)) } else { None }
}

pub fn raycast_octree_node(
    octree: &Octree, node_idx: usize, origin: Vec3, dir: Vec3, inv_dir: Vec3,
    node_min: Vec3, node_size: f32, closest_dist: &mut f32
) -> Option<RaycastHit> {
    let node = &octree.nodes[node_idx];
    if node.child_mask == 0 && node.material_id == 0 { return None; }
    let node_max = node_min + Vec3::splat(node_size);
    let intersection = ray_aabb_intersect(origin, inv_dir, node_min, node_max);
    if intersection.is_none() { return None; }
    let (t_hit, normal) = intersection.unwrap();
    if t_hit >= *closest_dist { return None; }

    if node.child_pointer == 0 {
        if node.material_id != 0 {
            *closest_dist = t_hit;
            return Some(RaycastHit { hit_pos: origin + dir * t_hit, voxel_min: node_min, voxel_size: node_size, normal, material: node.material_id });
        }
        return None;
    }

    let mut best_hit = None; let half_size = node_size * 0.5; let mut order = 0;
    if inv_dir.x < 0.0 { order |= 1; } if inv_dir.y < 0.0 { order |= 2; } if inv_dir.z < 0.0 { order |= 4; }

    for i in 0..8 {
        let child_idx = i ^ order;
        if (node.child_mask & (1 << child_idx)) != 0 {
            let cx = if (child_idx & 1) != 0 { half_size } else { 0.0 }; let cy = if (child_idx & 2) != 0 { half_size } else { 0.0 }; let cz = if (child_idx & 4) != 0 { half_size } else { 0.0 };
            let c_min = node_min + Vec3::new(cx, cy, cz);
            if let Some(hit) = raycast_octree_node(octree, (node.child_pointer + child_idx) as usize, origin, dir, inv_dir, c_min, half_size, closest_dist) {
                best_hit = Some(hit);
            }
        }
    }
    best_hit
}

pub fn check_octree_aabb(octree: &Octree, node_idx: usize, node_min: Vec3, node_size: f32, target_min: Vec3, target_max: Vec3) -> bool {
    let node = &octree.nodes[node_idx];
    if node.child_mask == 0 && node.material_id == 0 { return false; }
    let node_max = node_min + Vec3::splat(node_size);
    if target_min.x > node_max.x || target_max.x < node_min.x || target_min.y > node_max.y || target_max.y < node_min.y || target_min.z > node_max.z || target_max.z < node_min.z { return false; }
    if node.child_pointer == 0 { return node.material_id != 0; }
    let half_size = node_size * 0.5;
    for i in 0..8 {
        if (node.child_mask & (1 << i)) != 0 {
            let cx = if (i & 1) != 0 { half_size } else { 0.0 }; let cy = if (i & 2) != 0 { half_size } else { 0.0 }; let cz = if (i & 4) != 0 { half_size } else { 0.0 };
            if check_octree_aabb(octree, (node.child_pointer + i) as usize, node_min + Vec3::new(cx, cy, cz), half_size, target_min, target_max) { return true; }
        }
    }
    false
}

pub struct ChunkManager {
    pub loaded_chunks: HashMap<ChunkPos, RenderChunk>, pub loading_chunks: HashSet<ChunkPos>,
    tx: mpsc::Sender<(ChunkPos, Octree, MeshPayload)>, rx: mpsc::Receiver<(ChunkPos, Octree, MeshPayload)>,
    pub render_distance: i32, pub palette: Arc<RwLock<Palette>>, pub world_type: WorldType, pub seed: u32, pub cube_edits: Vec<CubeEdit>,
}

impl ChunkManager {
    pub fn new(palette: Arc<RwLock<Palette>>) -> Self {
        let (tx, rx) = mpsc::channel();
        Self { loaded_chunks: HashMap::new(), loading_chunks: HashSet::new(), tx, rx, render_distance: 3, palette, world_type: WorldType::Empty, seed: 42, cube_edits: Vec::new() }
    }
    
    pub fn get_voxel_info_at(&self, p: Vec3) -> (u16, f32, Vec3) {
        let cx = (p.x / 32.0).floor() as i32; let cy = (p.y / 32.0).floor() as i32; let cz = (p.z / 32.0).floor() as i32;
        if let Some(chunk) = self.loaded_chunks.get(&(cx, cy, cz)) {
            let lx = (p.x - cx as f32 * 32.0).clamp(0.0, 31.999); let ly = (p.y - cy as f32 * 32.0).clamp(0.0, 31.999); let lz = (p.z - cz as f32 * 32.0).clamp(0.0, 31.999);
            let gx = ((lx / 32.0) * GRID_RES as f32) as u32; let gy = ((ly / 32.0) * GRID_RES as f32) as u32; let gz = ((lz / 32.0) * GRID_RES as f32) as u32;
            let (mat, gsize) = chunk.octree.query_node(gx, gy, gz);
            let size = (gsize as f32 / GRID_RES as f32) * 32.0; let min_gx = (gx / gsize) * gsize; let min_gy = (gy / gsize) * gsize; let min_gz = (gz / gsize) * gsize;
            let vmin = Vec3::new(cx as f32 * 32.0 + (min_gx as f32 / GRID_RES as f32) * 32.0, cy as f32 * 32.0 + (min_gy as f32 / GRID_RES as f32) * 32.0, cz as f32 * 32.0 + (min_gz as f32 / GRID_RES as f32) * 32.0);
            (mat, size, vmin)
        } else { (0, 4.0, p) }
    }
    
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<RaycastHit> {
        if dir.length_squared() < 1e-6 { return None; }
        let dir = dir.normalize();
        let inv_dir = Vec3::new(
            if dir.x.abs() > 1e-8 { 1.0 / dir.x } else { 1e8 * dir.x.signum() },
            if dir.y.abs() > 1e-8 { 1.0 / dir.y } else { 1e8 * dir.y.signum() },
            if dir.z.abs() > 1e-8 { 1.0 / dir.z } else { 1e8 * dir.z.signum() },
        );
        let mut closest_hit: Option<RaycastHit> = None;
        let mut closest_dist = max_dist;

        for (&chunk_pos, chunk) in &self.loaded_chunks {
            let chunk_min = Vec3::new((chunk_pos.0 * 32) as f32, (chunk_pos.1 * 32) as f32, (chunk_pos.2 * 32) as f32);
            let chunk_max = chunk_min + Vec3::splat(32.0);
            if let Some((t_near, _)) = ray_aabb_intersect(origin, inv_dir, chunk_min, chunk_max) {
                if t_near < closest_dist {
                    if let Some(hit) = raycast_octree_node(&chunk.octree, chunk.octree.root_index as usize, origin, dir, inv_dir, chunk_min, 32.0, &mut closest_dist) {
                        closest_hit = Some(hit);
                    }
                }
            }
        }
        closest_hit
    }
    
    pub fn check_aabb_collision(&self, min: Vec3, max: Vec3) -> bool {
        for (&chunk_pos, chunk) in &self.loaded_chunks {
            let chunk_min = Vec3::new((chunk_pos.0 * 32) as f32, (chunk_pos.1 * 32) as f32, (chunk_pos.2 * 32) as f32);
            let chunk_max = chunk_min + Vec3::splat(32.0);
            if min.x <= chunk_max.x && max.x >= chunk_min.x && min.y <= chunk_max.y && max.y >= chunk_min.y && min.z <= chunk_max.z && max.z >= chunk_min.z {
                if check_octree_aabb(&chunk.octree, chunk.octree.root_index as usize, chunk_min, 32.0, min, max) { return true; }
            }
        }
        false
    }
    
    pub fn get_or_create_chunk<'a>(&'a mut self, chunk_pos: ChunkPos, device: &wgpu::Device) -> &'a mut RenderChunk {
        let wtype = self.world_type; let seed = self.seed; let edits = self.cube_edits.clone();
        self.loaded_chunks.entry(chunk_pos).or_insert_with(|| {
            let octree = generate_octree(chunk_pos, wtype, seed, &edits, 0);
            let vb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 1024, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let ib = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 1024, usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            RenderChunk { octree, vertex_buffer: vb, index_buffer: ib, num_indices: 0, vertex_capacity: 1024, index_capacity: 1024 }
        })
    }
    
    pub fn modify_cube(&mut self, pos: Vec3, material: u16, size: f32, device: &wgpu::Device, queue: &wgpu::Queue) {
        let edit = CubeEdit { pos: [pos.x, pos.y, pos.z], size, material }; self.cube_edits.push(edit);
        let pal = self.palette.read().unwrap().clone();
        if size >= 32.0 {
            let num_chunks = (size / 32.0).round() as i32; let start_cx = (pos.x / 32.0).floor() as i32; let start_cy = (pos.y / 32.0).floor() as i32; let start_cz = (pos.z / 32.0).floor() as i32;
            for dx in 0..num_chunks { for dy in 0..num_chunks { for dz in 0..num_chunks {
                let chunk_pos = (start_cx + dx, start_cy + dy, start_cz + dz); let chunk = self.get_or_create_chunk(chunk_pos, device);
                chunk.octree.insert_cube(0, 0, 0, 0, material, true); let payload = generate_mesh(&chunk.octree, chunk_pos, &pal); update_chunk_buffers(chunk, &payload, device, queue);
            } } }
        } else {
            let cx = (pos.x / 32.0).floor() as i32; let cy = (pos.y / 32.0).floor() as i32; let cz = (pos.z / 32.0).floor() as i32; let chunk_pos = (cx, cy, cz);
            let chunk = self.get_or_create_chunk(chunk_pos, device);
            let lx = pos.x - (cx * 32) as f32; let ly = pos.y - (cy * 32) as f32; let lz = pos.z - (cz * 32) as f32;
            let gx = ((lx / 32.0) * GRID_RES as f32).round() as u32; let gy = ((ly / 32.0) * GRID_RES as f32).round() as u32; let gz = ((lz / 32.0) * GRID_RES as f32).round() as u32;
            let grid_size = ((size / 32.0) * GRID_RES as f32).round().max(1.0) as u32;
            let depth = MAX_DEPTH - grid_size.trailing_zeros().min(MAX_DEPTH as u32) as u8;
            chunk.octree.insert_cube(gx, gy, gz, depth, material, true);
            let payload = generate_mesh(&chunk.octree, chunk_pos, &pal); update_chunk_buffers(chunk, &payload, device, queue);
        }
    }
    
    pub fn update(&mut self, player_pos: Vec3, device: &wgpu::Device) {
        let p_x = (player_pos.x / 32.0).floor() as i32; let p_z = (player_pos.z / 32.0).floor() as i32;
        let fov_factor = (60.0_f32.to_radians() / 2.0).tan();
        let screen_height = 1080.0;
        
        for x in -self.render_distance..=self.render_distance { 
            for z in -self.render_distance..=self.render_distance {
                let pos = (p_x + x, 0, p_z + z);
                
                let chunk_center = Vec3::new(pos.0 as f32 * 32.0 + 16.0, 16.0, pos.2 as f32 * 32.0 + 16.0);
                let distance = player_pos.distance(chunk_center).max(1.0);
                let projected_size = (4.0 / distance) * (screen_height / (2.0 * fov_factor));
                
                let lod_level = if projected_size > 12.0 { 0 } else if projected_size > 4.0 { 1 } else { 2 };
                
                if !self.loaded_chunks.contains_key(&pos) && !self.loading_chunks.contains(&pos) {
                    self.loading_chunks.insert(pos); let tx_clone = self.tx.clone(); let pal_arc = Arc::clone(&self.palette); let edits = self.cube_edits.clone(); let wtype = self.world_type; let seed = self.seed;
                    rayon::spawn(move || { let octree = generate_octree(pos, wtype, seed, &edits, lod_level); let pal = pal_arc.read().unwrap().clone(); let payload = generate_mesh(&octree, pos, &pal); let _ = tx_clone.send((pos, octree, payload)); });
                }
            } 
        }
        
        while let Ok((pos, octree, payload)) = self.rx.try_recv() {
            self.loading_chunks.remove(&pos); if payload.vertices.is_empty() || payload.indices.is_empty() { continue; }
            let v_bytes = bytemuck::cast_slice(&payload.vertices); let i_bytes = bytemuck::cast_slice(&payload.indices);
            use wgpu::util::DeviceExt;
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: v_bytes, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: i_bytes, usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST });
            self.loaded_chunks.insert(pos, RenderChunk { octree, vertex_buffer, index_buffer, num_indices: payload.indices.len() as u32, vertex_capacity: v_bytes.len(), index_capacity: i_bytes.len() });
        }
        self.loaded_chunks.retain(|pos, _| { (pos.0 - p_x).abs() <= self.render_distance + 1 && (pos.2 - p_z).abs() <= self.render_distance + 1 });
    }
    
    pub fn is_solid(&self, pos: Vec3) -> bool { self.get_voxel_info_at(pos).0 != 0 }
    
    pub fn get_surface_y(&self, x: f32, z: f32) -> Option<f32> {
        let mut y = 96.0_f32;
        while y >= -32.0 {
            let test_pos = Vec3::new(x, y, z);
            if self.is_solid(test_pos) { let (_, size, vmin) = self.get_voxel_info_at(test_pos); return Some(vmin.y + size); }
            y -= 0.5;
        }
        None
    }
}