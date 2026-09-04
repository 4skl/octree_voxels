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
use glam::{Vec3, Mat4};
use block_mesh::{visible_block_faces, UnitQuadBuffer, Voxel, VoxelVisibility, MergeVoxel, RIGHT_HANDED_Y_UP_CONFIG};
use ndshape::{ConstShape, ConstShape3u32};
use noise::{NoiseFn, Fbm, Perlin};
use serde::{Serialize, Deserialize};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
type ChunkShape = ConstShape3u32<34, 34, 34>; 
type ChunkPos = (i32, i32, i32);
pub type Palette = Vec<[f32; 3]>;

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

#[derive(Serialize, Deserialize)]
pub struct SaveData {
    pub player_pos: [f32; 3],
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub play_mode: PlayMode,
    pub world_type: WorldType,
    pub seed: u32,
    pub hotbar_colors: [[f32; 3]; 10],
    pub palette: Vec<[f32; 3]>,
    pub modified_blocks: Vec<([i32; 3], Vec<([u32; 3], u16)>)>,
}

// --- OCTREE MULTI-RÉSOLUTION ---

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

    pub fn insert(&mut self, mut x: u32, mut y: u32, mut z: u32, depth: u8, material: u16) {
        let mut current_idx = self.root_index as usize;
        let mut half_size = 16;

        for _ in 0..depth {
            let old_mat = self.nodes[current_idx].material_id;
            let child_ptr = self.nodes[current_idx].child_pointer;

            // Subdivision de nœud parent si nécessaire
            if child_ptr == 0 {
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

            if material != 0 {
                self.nodes[current_idx].child_mask |= 1 << octant;
            }

            current_idx = (self.nodes[current_idx].child_pointer + octant) as usize;
            half_size >>= 1;
        }

        self.nodes[current_idx].material_id = material;
        self.nodes[current_idx].child_pointer = 0;
        self.nodes[current_idx].child_mask = 0;
    }

    pub fn query(&self, mut x: u32, mut y: u32, mut z: u32) -> u16 {
        let mut current_idx = self.root_index as usize;
        let mut half_size = 16;
        for _ in 0..5 {
            let node = &self.nodes[current_idx];
            if node.child_pointer == 0 {
                return node.material_id;
            }
            let mut octant = 0;
            if x >= half_size { octant |= 1; x -= half_size; }
            if y >= half_size { octant |= 2; y -= half_size; }
            if z >= half_size { octant |= 4; z -= half_size; }

            if (node.child_mask & (1 << octant)) == 0 {
                return 0;
            }
            current_idx = (node.child_pointer + octant) as usize;
            half_size >>= 1;
        }
        self.nodes[current_idx].material_id
    }

    pub fn flatten_into(&self, node_idx: usize, x: u32, y: u32, z: u32, size: u32, voxels: &mut [Block]) {
        let node = &self.nodes[node_idx];
        if node.child_pointer == 0 {
            if node.material_id != 0 {
                for dz in 0..size {
                    for dy in 0..size {
                        for dx in 0..size {
                            let idx = ChunkShape::linearize([x + dx, y + dy, z + dz]);
                            voxels[idx as usize] = Block(node.material_id);
                        }
                    }
                }
            }
            return;
        }
        let half_size = size / 2;
        for i in 0..8 {
            if (node.child_mask & (1 << i)) != 0 {
                let cx = x + if (i & 1) != 0 { half_size } else { 0 };
                let cy = y + if (i & 2) != 0 { half_size } else { 0 };
                let cz = z + if (i & 4) != 0 { half_size } else { 0 };
                let child_idx = (node.child_pointer + i) as usize;
                self.flatten_into(child_idx, cx, cy, cz, half_size, voxels);
            }
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Block(pub u16);

impl Voxel for Block {
    fn get_visibility(&self) -> VoxelVisibility { 
        if self.0 == 0 { VoxelVisibility::Empty } else { VoxelVisibility::Opaque } 
    }
}
impl MergeVoxel for Block {
    type MergeValue = u16; 
    fn merge_value(&self) -> Self::MergeValue { self.0 }
}

// --- MAILLAGE ET GÉNÉRATION SANS OMBRES ---

struct MeshPayload { vertices: Vec<Vertex>, indices: Vec<u32> }

fn generate_octree(
    chunk_pos: ChunkPos,
    world_type: WorldType,
    seed: u32,
    deltas: Option<&HashMap<[u32; 3], u16>>,
) -> Octree {
    let mut octree = Octree::new();
    let fbm = Fbm::<Perlin>::new(seed);
    let world_x_offset = chunk_pos.0 * 32;
    let world_z_offset = chunk_pos.2 * 32;

    match world_type {
        WorldType::Empty => {
            // Vide par défaut pour la sculpture 3D libre
        }
        WorldType::Flat => {
            if chunk_pos.1 == 0 {
                for x in 0..32 {
                    for z in 0..32 {
                        for y in 0..4 {
                            let material = if y == 3 { 2 } else { 1 };
                            octree.insert(x, y, z, 5, material);
                        }
                    }
                }
            }
        }
        WorldType::Hills => {
            for x in 0..32 {
                for z in 0..32 {
                    let world_x = (x as i32 + world_x_offset) as f64 * 0.04;
                    let world_z = (z as i32 + world_z_offset) as f64 * 0.04;
                    let noise_val = fbm.get([world_x, world_z]);
                    let height = ((noise_val + 1.0) * 12.0).clamp(0.0, 31.0) as u32;

                    for y in 0..=height {
                        let material = if y == height { 2 } else { 1 };
                        octree.insert(x, y, z, 5, material);
                    }
                }
            }
        }
        WorldType::Mountains => {
            for x in 0..32 {
                for z in 0..32 {
                    let world_x = (x as i32 + world_x_offset) as f64 * 0.025;
                    let world_z = (z as i32 + world_z_offset) as f64 * 0.025;
                    let n = fbm.get([world_x, world_z]).abs();
                    let height = (n * 30.0 + 2.0).clamp(0.0, 31.0) as u32;

                    for y in 0..=height {
                        let material = if y >= 25 { 3 } else if y == height { 2 } else { 1 };
                        octree.insert(x, y, z, 5, material);
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
                            octree.insert(x, y, z, 5, 2);
                        }
                    }
                }
            }
        }
    }

    if let Some(mods) = deltas {
        for (&[x, y, z], &material) in mods {
            octree.insert(x, y, z, 5, material);
        }
    }

    octree
}

fn generate_mesh(octree: &Octree, chunk_pos: ChunkPos, palette: &[[f32; 3]]) -> MeshPayload {
    let mut voxels = vec![Block(0); ChunkShape::SIZE as usize];
    octree.flatten_into(octree.root_index as usize, 1, 1, 1, 32, &mut voxels);

    let mut buffer = UnitQuadBuffer::new();
    visible_block_faces(&voxels, &ChunkShape {}, [0, 0, 0], [33, 33, 33], &RIGHT_HANDED_Y_UP_CONFIG.faces, &mut buffer);

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let offset_x = (chunk_pos.0 * 32) as f32;
    let offset_y = (chunk_pos.1 * 32) as f32;
    let offset_z = (chunk_pos.2 * 32) as f32;

    for (group, face) in buffer.groups.iter().zip(RIGHT_HANDED_Y_UP_CONFIG.faces.into_iter()) {
        let n = face.signed_normal();
        let normal = [n.x as f32, n.y as f32, n.z as f32];

        for quad in group.into_iter() {
            let start_index = vertices.len() as u32;
            let pos1 = quad.minimum;
            let mut pos2 = quad.minimum;
            if n.x != 0 { pos2[0] += 1; }
            if n.y != 0 { pos2[1] += 1; }
            if n.z != 0 { pos2[2] += 1; }

            let id1 = voxels[ChunkShape::linearize(pos1) as usize].0;
            let id2 = voxels[ChunkShape::linearize(pos2) as usize].0;
            let mat_id = if id1 != 0 { id1 } else { id2 };
            
            let color = if mat_id >= 1 && (mat_id as usize) <= palette.len() {
                palette[(mat_id - 1) as usize]
            } else {
                [0.5, 0.5, 0.5]
            };

            let generic_quad = block_mesh::UnorientedQuad { minimum: quad.minimum, width: 1, height: 1 };

            for corner in face.quad_mesh_positions(&generic_quad, 1.0) {
                // Suppression totale de l'occlusion ambiante (ao_multiplier retiré)
                vertices.push(Vertex {
                    position: [corner[0] - 1.0 + offset_x, corner[1] - 1.0 + offset_y, corner[2] - 1.0 + offset_z],
                    normal,
                    color,
                });
            }
            indices.extend_from_slice(&face.quad_mesh_indices(start_index));
        }
    }
    MeshPayload { vertices, indices }
}

// --- GESTIONNAIRE DE CHUNKS ---

struct RenderChunk {
    octree: Octree,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

struct ChunkManager {
    loaded_chunks: HashMap<ChunkPos, RenderChunk>,
    loading_chunks: HashSet<ChunkPos>,
    tx: mpsc::Sender<(ChunkPos, Octree, MeshPayload)>,
    rx: mpsc::Receiver<(ChunkPos, Octree, MeshPayload)>,
    render_distance: i32,
    palette: Arc<RwLock<Palette>>,
    pub world_type: WorldType,
    pub seed: u32,
    pub modified_blocks: HashMap<ChunkPos, HashMap<[u32; 3], u16>>,
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
            modified_blocks: HashMap::new(),
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
                    let deltas = self.modified_blocks.get(&pos).cloned();
                    let wtype = self.world_type;
                    let seed = self.seed;

                    rayon::spawn(move || {
                        let octree = generate_octree(pos, wtype, seed, deltas.as_ref());
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
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::cast_slice(&payload.vertices), usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::cast_slice(&payload.indices), usage: wgpu::BufferUsages::INDEX,
            });
            self.loaded_chunks.insert(pos, RenderChunk { octree, vertex_buffer, index_buffer, num_indices: payload.indices.len() as u32 });
        }

        self.loaded_chunks.retain(|pos, _| {
            (pos.0 - p_x).abs() <= self.render_distance + 1 && (pos.2 - p_z).abs() <= self.render_distance + 1
        });
    }

    fn modify_cube(&mut self, pos: Vec3, material: u16, size: u32, device: &wgpu::Device) {
        let depth = match size {
            1 => 5, // Sub-cube feuille (1x1)
            2 => 4, // Sub-cube moyen (2x2)
            4 => 3, // Cube standard (4x4)
            8 => 2, // Super-cube (8x8)
            16 => 1, // Super-cube large (16x16)
            _ => 5,
        };
        let s = size as i32;

        let bx = (pos.x.floor() as i32).div_euclid(s) * s;
        let by = (pos.y.floor() as i32).div_euclid(s) * s;
        let bz = (pos.z.floor() as i32).div_euclid(s) * s;

        let cx = bx.div_euclid(32);
        let cy = by.div_euclid(32);
        let cz = bz.div_euclid(32);
        let chunk_pos = (cx, cy, cz);

        let lx = bx.rem_euclid(32) as u32;
        let ly = by.rem_euclid(32) as u32;
        let lz = bz.rem_euclid(32) as u32;

        let chunk_map = self.modified_blocks.entry(chunk_pos).or_default();
        for dz in 0..size {
            for dy in 0..size {
                for dx in 0..size {
                    if material == 0 {
                        chunk_map.remove(&[lx + dx, ly + dy, lz + dz]);
                    } else {
                        chunk_map.insert([lx + dx, ly + dy, lz + dz], material);
                    }
                }
            }
        }

        let chunk = self.loaded_chunks.entry(chunk_pos).or_insert_with(|| {
            let octree = generate_octree(chunk_pos, self.world_type, self.seed, self.modified_blocks.get(&chunk_pos));
            let vb = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: 64, usage: wgpu::BufferUsages::VERTEX, mapped_at_creation: false,
            });
            let ib = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: 64, usage: wgpu::BufferUsages::INDEX, mapped_at_creation: false,
            });
            RenderChunk { octree, vertex_buffer: vb, index_buffer: ib, num_indices: 0 }
        });

        chunk.octree.insert(lx, ly, lz, depth, material);

        let pal = self.palette.read().unwrap().clone();
        let payload = generate_mesh(&chunk.octree, chunk_pos, &pal);
        if payload.vertices.is_empty() || payload.indices.is_empty() {
            chunk.num_indices = 0;
        } else {
            chunk.vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::cast_slice(&payload.vertices), usage: wgpu::BufferUsages::VERTEX,
            });
            chunk.index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::cast_slice(&payload.indices), usage: wgpu::BufferUsages::INDEX,
            });
            chunk.num_indices = payload.indices.len() as u32;
        }
    }

    fn get_material(&self, pos: Vec3) -> u16 {
        let cx = (pos.x / 32.0).floor() as i32;
        let cy = (pos.y / 32.0).floor() as i32;
        let cz = (pos.z / 32.0).floor() as i32;
        let lx = (pos.x.floor() as i32).rem_euclid(32) as u32;
        let ly = (pos.y.floor() as i32).rem_euclid(32) as u32;
        let lz = (pos.z.floor() as i32).rem_euclid(32) as u32;

        if let Some(chunk) = self.loaded_chunks.get(&(cx, cy, cz)) {
            chunk.octree.query(lx, ly, lz)
        } else {
            0
        }
    }

    fn is_solid(&self, pos: Vec3) -> bool {
        self.get_material(pos) != 0
    }

    pub fn get_surface_y(&self, x: f32, z: f32) -> Option<f32> {
        let px = x.floor() as i32;
        let pz = z.floor() as i32;
        for y in (-32..96).rev() {
            let test_pos = Vec3::new(px as f32 + 0.5, y as f32 + 0.5, pz as f32 + 0.5);
            if self.is_solid(test_pos) {
                return Some(y as f32 + 1.0);
            }
        }
        None
    }
}

// --- STRUCTURES RENDU ---

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex { position: [f32; 3], normal: [f32; 3], color: [f32; 3] }

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3];
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
struct CameraUniform { view_proj: [[f32; 4]; 4] }

struct Camera { position: Vec3, yaw: f32, pitch: f32 }
impl Camera {
    fn view_proj(&self, aspect: f32) -> Mat4 {
        let (sin_p, cos_p) = self.pitch.sin_cos();
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let dir = Vec3::new(cos_y * cos_p, sin_p, sin_y * cos_p).normalize();
        let view = glam::camera::rh::view::look_at_mat4(self.position, self.position + dir, Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective((60.0_f32).to_radians(), aspect, 0.05, 500.0);
        proj * view
    }
    fn forward(&self) -> Vec3 {
        let (sin_p, cos_p) = self.pitch.sin_cos();
        let (sin_y, cos_y) = self.yaw.sin_cos();
        Vec3::new(cos_y * cos_p, sin_p, sin_y * cos_p).normalize()
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

fn get_glyph(c: char) -> [u8; 5] {
    match c {
        'A' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'B' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'C' => [0b111, 0b100, 0b100, 0b100, 0b111],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b110, 0b100, 0b111],
        'F' => [0b111, 0b100, 0b110, 0b100, 0b100],
        'G' => [0b111, 0b100, 0b101, 0b101, 0b111],
        'H' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'J' => [0b001, 0b001, 0b001, 0b101, 0b111],
        'K' => [0b101, 0b110, 0b100, 0b110, 0b101],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'M' => [0b101, 0b111, 0b101, 0b101, 0b101],
        'N' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'O' => [0b111, 0b101, 0b101, 0b101, 0b111],
        'P' => [0b111, 0b101, 0b111, 0b100, 0b100],
        'Q' => [0b010, 0b101, 0b101, 0b110, 0b011],
        'R' => [0b110, 0b101, 0b110, 0b101, 0b101],
        'S' => [0b111, 0b100, 0b111, 0b001, 0b111],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'U' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'V' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'W' => [0b101, 0b101, 0b101, 0b111, 0b101],
        'X' => [0b101, 0b101, 0b010, 0b101, 0b101],
        'Y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        'Z' => [0b111, 0b001, 0b010, 0b100, 0b111],
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        ':' => [0b000, 0b010, 0b000, 0b010, 0b000],
        '/' => [0b001, 0b001, 0b010, 0b100, 0b100],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        '[' => [0b110, 0b100, 0b100, 0b100, 0b110],
        ']' => [0b011, 0b001, 0b001, 0b001, 0b011],
        '|' => [0b010, 0b010, 0b010, 0b010, 0b010],
        _ => [0; 5],
    }
}

fn draw_text(verts: &mut Vec<UIVertex>, text: &str, start_x: f32, start_y: f32, pixel_w: f32, pixel_h: f32, color: [f32; 4]) {
    let mut cursor_x = start_x;
    for c in text.chars() {
        let glyph = get_glyph(c);
        for row in 0..5 {
            let line = glyph[row];
            let y1 = start_y + (4 - row) as f32 * pixel_h;
            let y0 = y1 - pixel_h;
            for col in 0..3 {
                if (line & (1 << (2 - col))) != 0 {
                    let x0 = cursor_x + col as f32 * pixel_w;
                    let x1 = x0 + pixel_w;
                    add_quad(verts, x0, y0, x1, y1, color);
                }
            }
        }
        cursor_x += 4.0 * pixel_w;
    }
}

fn draw_text_centered(verts: &mut Vec<UIVertex>, text: &str, cx: f32, cy: f32, pw: f32, ph: f32, color: [f32; 4]) {
    let total_w = (text.len() as f32 * 4.0 - 1.0) * pw;
    let total_h = 5.0 * ph;
    draw_text(verts, text, cx - total_w / 2.0, cy - total_h / 2.0, pw, ph, color);
}

const PRESET_SWATCHES: [[f32; 3]; 10] = [
    [0.95, 0.95, 0.95], // Blanc
    [0.10, 0.10, 0.12], // Noir
    [0.90, 0.20, 0.20], // Rouge
    [0.95, 0.50, 0.15], // Orange
    [0.95, 0.85, 0.15], // Jaune
    [0.20, 0.75, 0.25], // Vert
    [0.15, 0.80, 0.85], // Cyan
    [0.20, 0.45, 0.90], // Bleu
    [0.65, 0.25, 0.85], // Violet
    [0.55, 0.35, 0.20], // Marron
];

fn build_ui_vertices(
    selected_slot: usize,
    menu_open: bool,
    hotbar_colors: &[[f32; 3]; 10],
    play_mode: PlayMode,
    world_type: WorldType,
    edit_size: u32,
    target_pos: Option<[i32; 3]>,
) -> Vec<UIVertex> {
    let mut verts = Vec::new();
    let num_slots = 10;
    let font_pw = 0.0050;
    let font_ph = 0.0085;

    // --- BARRE DE SLOTS INFERIEURE EN JEU ---
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
        draw_text_centered(&mut verts, num_str, (x0 + x1) / 2.0, y_top + 0.02, font_pw * 0.8, font_ph * 0.8, [0.9, 0.9, 0.9, 0.9]);
    }

    let size_tag = match edit_size {
        1 => "1X1 SUB",
        2 => "2X2 SUB",
        4 => "4X4 STD",
        8 => "8X8 SUPER",
        16 => "16X16 MEGA",
        _ => "1X1",
    };

    if !menu_open {
        // Réticule central
        add_quad(&mut verts, -0.012, -0.002, 0.012, 0.002, [1.0, 1.0, 1.0, 0.95]);
        add_quad(&mut verts, -0.002, -0.020, 0.002, 0.020, [1.0, 1.0, 1.0, 0.95]);

        // Bandeau HUD supérieur
        let mode_hud = if play_mode == PlayMode::Flying { "FLY" } else { "REAL" };
        let hud_title = format!("MODE: {}  |  WORLD: {}  |  OCTREE SIZE: {}", mode_hud, world_type.name(), size_tag);
        draw_text(&mut verts, &hud_title, -0.96, 0.92, font_pw, font_ph, [1.0, 1.0, 1.0, 0.95]);

        if let Some(tpos) = target_pos {
            let target_str = format!("AIM: [{}, {}, {}]", tpos[0], tpos[1], tpos[2]);
            draw_text(&mut verts, &target_str, -0.96, 0.86, font_pw * 0.9, font_ph * 0.9, [0.3, 0.9, 0.9, 0.9]);
        } else {
            draw_text(&mut verts, "AIM: [GROUND PLANE Y:0]", -0.96, 0.86, font_pw * 0.9, font_ph * 0.9, [0.6, 0.7, 0.8, 0.8]);
        }

        draw_text(&mut verts, "[E] PALETTE / TOOLS  [Q/R] SIZE  [LMB] PLACE  [RMB] REMOVE", -0.96, 0.80, font_pw * 0.8, font_ph * 0.8, [0.9, 0.85, 0.4, 0.85]);
    } else {
        // --- FENETRE MODALE COMPLETE (TOUCHE E) ---
        add_quad(&mut verts, -1.0, -1.0, 1.0, 1.0, [0.03, 0.04, 0.06, 0.78]);

        let px0 = -0.56; let px1 = 0.56;
        let py0 = -0.68; let py1 = 0.68;
        add_quad(&mut verts, px0 - 0.006, py0 - 0.006, px1 + 0.006, py1 + 0.006, [0.25, 0.30, 0.40, 1.0]);
        add_quad(&mut verts, px0, py0, px1, py1, [0.10, 0.12, 0.16, 0.98]);

        draw_text_centered(&mut verts, "VOXEL STUDIO - COLOR & OCTREE", 0.0, 0.61, font_pw * 1.1, font_ph * 1.1, [1.0, 0.9, 0.2, 1.0]);

        // Rangée 1 : 10 Slots
        let sw_w = 0.082;
        let sw_gap = 0.015;
        let sw_tot = 10.0 * sw_w + 9.0 * sw_gap;
        let s_start_x = -sw_tot / 2.0;
        let sy0 = 0.46;
        let sy1 = 0.55;

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
            draw_text_centered(&mut verts, num_str, (sx0 + sx1) / 2.0, sy1 + 0.022, font_pw * 0.8, font_ph * 0.8, [0.8, 0.8, 0.8, 0.9]);
        }

        // Rangée 2 : Ajusteur RVB & Échantillons rapides
        let [cur_r, cur_g, cur_b] = hotbar_colors[selected_slot];
        add_quad(&mut verts, 0.24, 0.20, 0.46, 0.40, [0.25, 0.28, 0.35, 1.0]);
        add_quad(&mut verts, 0.248, 0.208, 0.452, 0.392, [cur_r, cur_g, cur_b, 1.0]);
        draw_text_centered(&mut verts, "ACTIVE COLOR", 0.35, 0.42, font_pw * 0.8, font_ph * 0.8, [0.85, 0.85, 0.85, 0.9]);

        let sl_x0 = -0.32;
        let sl_x1 = 0.18;
        let channels = [
            ("R", cur_r, 0.35, 0.39, [0.90, 0.25, 0.25, 1.0]),
            ("G", cur_g, 0.28, 0.32, [0.25, 0.85, 0.30, 1.0]),
            ("B", cur_b, 0.21, 0.25, [0.25, 0.50, 0.95, 1.0]),
        ];

        for (lbl, val, y0, y1, bar_col) in channels {
            draw_text_centered(&mut verts, lbl, -0.37, (y0 + y1) / 2.0, font_pw, font_ph, bar_col);
            add_quad(&mut verts, sl_x0, y0, sl_x1, y1, [0.18, 0.20, 0.25, 1.0]);
            let filled_x = sl_x0 + val * (sl_x1 - sl_x0);
            add_quad(&mut verts, sl_x0, y0, filled_x, y1, bar_col);
            add_quad(&mut verts, filled_x - 0.010, y0 - 0.006, filled_x + 0.010, y1 + 0.006, [1.0, 1.0, 1.0, 1.0]);
        }

        // Nuancier rapide (1-clic pour changer la couleur active)
        let pw_w = 0.076;
        let pw_gap = 0.012;
        let pw_tot = 10.0 * pw_w + 9.0 * pw_gap;
        let pw_start_x = -pw_tot / 2.0;
        let pwy0 = 0.08;
        let pwy1 = 0.14;

        draw_text_centered(&mut verts, "QUICK PALETTE CHIPS", 0.0, 0.165, font_pw * 0.8, font_ph * 0.8, [0.75, 0.75, 0.8, 0.9]);
        for (i, &rgb) in PRESET_SWATCHES.iter().enumerate() {
            let px0 = pw_start_x + i as f32 * (pw_w + pw_gap);
            let px1 = px0 + pw_w;
            add_quad(&mut verts, px0 - 0.003, pwy0 - 0.003, px1 + 0.003, pwy1 + 0.003, [0.3, 0.3, 0.35, 1.0]);
            add_quad(&mut verts, px0, pwy0, px1, pwy1, [rgb[0], rgb[1], rgb[2], 1.0]);
        }

        // Rangée 3 : Sélecteur de sous/super cubes octree
        let sizes = [(1, "1X1 SUB"), (2, "2X2 SUB"), (4, "4X4 STD"), (8, "8X8 SUPER"), (16, "16X16 MEGA")];
        let btn_w = 0.17;
        let btn_gap = 0.02;
        let btot = 5.0 * btn_w + 4.0 * btn_gap;
        let bstart = -btot / 2.0;
        let by0 = -0.07;
        let by1 = 0.01;

        draw_text_centered(&mut verts, "OCTREE VOXEL RESOLUTION", 0.0, 0.035, font_pw * 0.85, font_ph * 0.85, [0.85, 0.85, 0.85, 0.9]);
        for (i, &(sz, lbl)) in sizes.iter().enumerate() {
            let bx0 = bstart + i as f32 * (btn_w + btn_gap);
            let bx1 = bx0 + btn_w;
            let is_cur = edit_size == sz;
            let border_col = if is_cur { [1.0, 0.85, 0.2, 1.0] } else { [0.25, 0.30, 0.40, 1.0] };
            let bg_col = if is_cur { [0.35, 0.30, 0.10, 1.0] } else { [0.15, 0.18, 0.24, 1.0] };

            add_quad(&mut verts, bx0 - 0.004, by0 - 0.004, bx1 + 0.004, by1 + 0.004, border_col);
            add_quad(&mut verts, bx0, by0, bx1, by1, bg_col);
            draw_text_centered(&mut verts, lbl, (bx0 + bx1) / 2.0, (by0 + by1) / 2.0, font_pw * 0.85, font_ph * 0.85, [1.0, 1.0, 1.0, 1.0]);
        }

        // Rangée 4 : Boutons d'actions système
        let mode_text = if play_mode == PlayMode::Flying { "MODE: FLY" } else { "MODE: REAL" };
        add_quad(&mut verts, -0.45, -0.22, -0.16, -0.13, [0.20, 0.35, 0.55, 1.0]);
        draw_text_centered(&mut verts, mode_text, -0.305, -0.175, font_pw, font_ph, [1.0, 1.0, 1.0, 1.0]);

        let world_label = format!("PRESET: {}", world_type.name());
        add_quad(&mut verts, -0.14, -0.22, 0.15, -0.13, [0.35, 0.25, 0.50, 1.0]);
        draw_text_centered(&mut verts, &world_label, 0.005, -0.175, font_pw * 0.8, font_ph * 0.8, [1.0, 1.0, 1.0, 1.0]);

        add_quad(&mut verts, 0.17, -0.22, 0.46, -0.13, [0.60, 0.30, 0.20, 1.0]);
        draw_text_centered(&mut verts, "CLEAR ALL", 0.315, -0.175, font_pw, font_ph, [1.0, 1.0, 1.0, 1.0]);

        // Rangée 5 : Reprendre / Quitter
        add_quad(&mut verts, -0.32, -0.36, -0.02, -0.27, [0.20, 0.55, 0.30, 1.0]);
        draw_text_centered(&mut verts, "RESUME (E)", -0.17, -0.315, font_pw, font_ph, [1.0, 1.0, 1.0, 1.0]);

        add_quad(&mut verts, 0.02, -0.36, 0.32, -0.27, [0.55, 0.20, 0.20, 1.0]);
        draw_text_centered(&mut verts, "QUIT", 0.17, -0.315, font_pw, font_ph, [1.0, 1.0, 1.0, 1.0]);

        draw_text_centered(&mut verts, "SHORTCUTS: [E] MENU | [Q/R] SIZE | [1-0] SLOTS | [F5] SAVE | [F9] LOAD", 0.0, -0.47, font_pw * 0.8, font_ph * 0.8, [0.75, 0.80, 0.90, 0.85]);
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
    menu_open: bool,
    cursor_pos: [f32; 2],
    active_slider: Option<usize>,
    edit_size: u32,
    last_target: Option<[i32; 3]>,
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

        let camera = Camera { position: Vec3::new(16.0, 12.0, 26.0), yaw: -std::f32::consts::FRAC_PI_2, pitch: -0.3 };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[CameraUniform { view_proj: camera.view_proj(1.0).to_cols_array_2d() }]), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, count: None, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None } }], label: None,
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
            [0.55, 0.55, 0.58], // 1
            [0.22, 0.75, 0.32], // 2
            [0.85, 0.20, 0.20], // 3
            [0.20, 0.48, 0.90], // 4
            [0.95, 0.78, 0.18], // 5
            [0.15, 0.85, 0.85], // 6
            [0.82, 0.25, 0.85], // 7
            [0.95, 0.50, 0.15], // 8
            [0.95, 0.95, 0.95], // 9
            [0.20, 0.22, 0.25], // 0
        ];

        let palette = Arc::new(RwLock::new(hotbar_colors.to_vec()));
        let chunk_manager = ChunkManager::new(Arc::clone(&palette));

        let initial_ui = build_ui_vertices(0, false, &hotbar_colors, PlayMode::Flying, chunk_manager.world_type, 4, None);
        let ui_vertices_count = initial_ui.len() as u32;
        let ui_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UI Buffer"),
            size: (32768 * std::mem::size_of::<UIVertex>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&ui_vertex_buffer, 0, bytemuck::cast_slice(&initial_ui));

        Self {
            window, surface, device, queue, config, size, render_pipeline, ui_pipeline, ui_vertex_buffer, ui_vertices_count,
            camera_buffer, camera_bind_group, depth_texture_view, camera,
            input: InputState::default(), chunk_manager,
            play_mode: PlayMode::Flying, velocity: Vec3::ZERO, selected_slot: 0, hotbar_colors,
            palette, menu_open: false, cursor_pos: [0.0, 0.0], active_slider: None, edit_size: 4,
            last_target: None,
        }
    }

    pub fn save_game(&self, filename: &str) -> std::io::Result<()> {
        let serialized_deltas: Vec<([i32; 3], Vec<([u32; 3], u16)>)> = self.chunk_manager.modified_blocks.iter()
            .map(|(&(cx, cy, cz), block_map)| {
                ([cx, cy, cz], block_map.iter().map(|(&k, &v)| (k, v)).collect())
            })
            .collect();

        let data = SaveData {
            player_pos: self.camera.position.to_array(),
            camera_yaw: self.camera.yaw,
            camera_pitch: self.camera.pitch,
            play_mode: self.play_mode,
            world_type: self.chunk_manager.world_type,
            seed: self.chunk_manager.seed,
            hotbar_colors: self.hotbar_colors,
            palette: self.palette.read().unwrap().clone(),
            modified_blocks: serialized_deltas,
        };

        let json = serde_json::to_string_pretty(&data)?;
        std::fs::write(filename, json)?;
        println!("Saved: {}", filename);
        Ok(())
    }

    pub fn load_game(&mut self, filename: &str) -> std::io::Result<()> {
        let content = std::fs::read_to_string(filename)?;
        let data: SaveData = serde_json::from_str(&content)?;

        self.camera.position = Vec3::from_array(data.player_pos);
        self.camera.yaw = data.camera_yaw;
        self.camera.pitch = data.camera_pitch;
        self.play_mode = data.play_mode;
        self.velocity = Vec3::ZERO;
        self.hotbar_colors = data.hotbar_colors;
        *self.palette.write().unwrap() = data.palette;

        self.chunk_manager.world_type = data.world_type;
        self.chunk_manager.seed = data.seed;
        self.chunk_manager.modified_blocks.clear();
        for (chunk_coords, edits) in data.modified_blocks {
            let pos = (chunk_coords[0], chunk_coords[1], chunk_coords[2]);
            self.chunk_manager.modified_blocks.insert(pos, edits.into_iter().collect());
        }

        self.chunk_manager.loaded_chunks.clear();
        self.chunk_manager.loading_chunks.clear();
        self.update_ui();
        println!("Loaded: {}", filename);
        Ok(())
    }

    pub fn clear_all_blocks(&mut self) {
        self.chunk_manager.modified_blocks.clear();
        self.chunk_manager.loaded_chunks.clear();
        self.chunk_manager.loading_chunks.clear();
        self.update_ui();
    }

    pub fn cycle_world_generator(&mut self) {
        self.chunk_manager.world_type = self.chunk_manager.world_type.next();
        self.clear_all_blocks();
    }

    pub fn cycle_cube_size(&mut self, forward: bool) {
        let sizes = [1, 2, 4, 8, 16];
        let cur_idx = sizes.iter().position(|&s| s == self.edit_size).unwrap_or(2);
        let next_idx = if forward {
            (cur_idx + 1).min(sizes.len() - 1)
        } else {
            cur_idx.saturating_sub(1)
        };
        self.edit_size = sizes[next_idx];
        self.update_ui();
    }

    fn toggle_menu(&mut self) {
        self.menu_open = !self.menu_open;
        self.active_slider = None;
        if self.menu_open {
            let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::None);
            self.window.set_cursor_visible(true);
            self.input = InputState::default();
        } else {
            let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::Locked)
                .or_else(|_| self.window.set_cursor_grab(winit::window::CursorGrabMode::Confined));
            self.window.set_cursor_visible(false);
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
        let verts = build_ui_vertices(
            self.selected_slot,
            self.menu_open,
            &self.hotbar_colors,
            self.play_mode,
            self.chunk_manager.world_type,
            self.edit_size,
            self.last_target,
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
        }
    }

    fn update(&mut self, dt: f32) {
        if self.menu_open { return; }

        let (sin_y, cos_y) = self.camera.yaw.sin_cos();
        let forward = Vec3::new(cos_y, 0.0, sin_y).normalize();
        let right = Vec3::new(-sin_y, 0.0, cos_y).normalize();

        let mut movement = Vec3::ZERO;
        if self.input.forward { movement += forward; } if self.input.backward { movement -= forward; }
        if self.input.right { movement += right; } if self.input.left { movement -= right; }

        match self.play_mode {
            PlayMode::Flying => {
                let speed = 26.0;
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

        // --- RAYCAST & PLACEMENT LIBRE ---
        let dir = self.camera.forward();
        let mut current_pos = self.camera.position;
        let step = 0.08;
        let mut hit = false;
        let mut hit_pos = Vec3::ZERO;
        let mut prev_pos = current_pos;

        for _ in 0..220 { 
            current_pos += dir * step;
            let mat = self.chunk_manager.get_material(current_pos);
            if mat != 0 {
                hit = true;
                hit_pos = current_pos;
                break;
            }
            prev_pos = current_pos;
        }

        let s = self.edit_size as i32;
        if hit {
            self.last_target = Some([
                (hit_pos.x.floor() as i32).div_euclid(s) * s,
                (hit_pos.y.floor() as i32).div_euclid(s) * s,
                (hit_pos.z.floor() as i32).div_euclid(s) * s,
            ]);
        } else if dir.y.abs() > 0.02 {
            let t = -self.camera.position.y / dir.y;
            if t > 0.5 && t < 100.0 {
                let g = self.camera.position + dir * t;
                self.last_target = Some([
                    (g.x.floor() as i32).div_euclid(s) * s,
                    0,
                    (g.z.floor() as i32).div_euclid(s) * s,
                ]);
            } else {
                self.last_target = None;
            }
        } else {
            self.last_target = None;
        }

        if self.input.action_add || self.input.action_remove || self.input.action_pick {
            if hit {
                if self.input.action_pick {
                    let mat = self.chunk_manager.get_material(hit_pos);
                    let pal = self.palette.read().unwrap();
                    if let Some(&color) = pal.get((mat - 1) as usize) {
                        self.hotbar_colors[self.selected_slot] = color;
                        drop(pal);
                        self.update_ui();
                    }
                } else if self.input.action_remove { 
                    self.chunk_manager.modify_cube(hit_pos, 0, self.edit_size, &self.device); 
                } else if self.input.action_add { 
                    let place_pos = prev_pos;
                    if self.play_mode == PlayMode::Flying || !self.player_collides_at(self.camera.position) {
                        let mat_id = self.get_or_create_material(self.hotbar_colors[self.selected_slot]);
                        self.chunk_manager.modify_cube(place_pos, mat_id, self.edit_size, &self.device); 
                    }
                }
            } else if self.input.action_add {
                // Monde vide : placement directement sur le plan de référence y=0 ou devant soi
                let mut place_pos = self.camera.position + dir * 6.0;
                if dir.y.abs() > 0.02 {
                    let t = -self.camera.position.y / dir.y;
                    if t > 0.5 && t < 100.0 {
                        place_pos = self.camera.position + dir * t;
                    }
                }
                let mat_id = self.get_or_create_material(self.hotbar_colors[self.selected_slot]);
                self.chunk_manager.modify_cube(place_pos, mat_id, self.edit_size, &self.device);
            }

            self.input.action_add = false; 
            self.input.action_remove = false;
            self.input.action_pick = false;
        }

        let aspect = if self.config.height > 0 { self.config.width as f32 / self.config.height as f32 } else { 1.0 };
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[CameraUniform { view_proj: self.camera.view_proj(aspect).to_cols_array_2d() }]));
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
                .with_title("Voxel Engine - Octree Studio")
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
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::CursorMoved { position, .. } => {
                    let ndc_x = (position.x as f32 / state.size.width as f32) * 2.0 - 1.0;
                    let ndc_y = 1.0 - (position.y as f32 / state.size.height as f32) * 2.0;
                    state.cursor_pos = [ndc_x, ndc_y];

                    if state.menu_open {
                        if let Some(channel) = state.active_slider {
                            let sl_x0 = -0.32;
                            let sl_x1 = 0.18;
                            let val = ((ndc_x - sl_x0) / (sl_x1 - sl_x0)).clamp(0.0, 1.0);
                            state.hotbar_colors[state.selected_slot][channel] = val;
                            state.update_ui();
                        }
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let step = match delta {
                        MouseScrollDelta::LineDelta(_, y) => if y > 0.0 { -1 } else if y < 0.0 { 1 } else { 0 },
                        MouseScrollDelta::PixelDelta(pos) => if pos.y > 0.0 { -1 } else if pos.y < 0.0 { 1 } else { 0 },
                    };
                    if step != 0 {
                        state.selected_slot = (state.selected_slot as i32 + step).rem_euclid(10) as usize;
                        state.update_ui();
                    }
                }
                WindowEvent::KeyboardInput { event: key_event, .. } => {
                    let is_pressed = key_event.state == ElementState::Pressed;
                    
                    // Touche E et Échap ouvrent et ferment le menu
                    if (key_event.physical_key == PhysicalKey::Code(KeyCode::KeyE)
                        || key_event.physical_key == PhysicalKey::Code(KeyCode::Escape))
                        && is_pressed
                    {
                        state.toggle_menu();
                        return;
                    }

                    if is_pressed {
                        match key_event.physical_key {
                            PhysicalKey::Code(KeyCode::F2) => state.cycle_world_generator(),
                            PhysicalKey::Code(KeyCode::F5) => { let _ = state.save_game("world_save.json"); },
                            PhysicalKey::Code(KeyCode::F9) => { let _ = state.load_game("world_save.json"); },
                            PhysicalKey::Code(KeyCode::KeyM) => state.toggle_play_mode(),
                            PhysicalKey::Code(KeyCode::KeyQ) | PhysicalKey::Code(KeyCode::KeyZ) => state.cycle_cube_size(false),
                            PhysicalKey::Code(KeyCode::KeyR) | PhysicalKey::Code(KeyCode::KeyX) => state.cycle_cube_size(true),
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
                            PhysicalKey::Code(KeyCode::KeyC) if !state.menu_open => state.input.action_pick = true,
                            _ => {}
                        }
                    }

                    if !state.menu_open {
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
                    if state.menu_open {
                        if button == MouseButton::Left {
                            if element_state == ElementState::Pressed {
                                let [mx, my] = state.cursor_pos;

                                // 10 Slots de couleurs
                                let sw_w = 0.082;
                                let sw_gap = 0.015;
                                let sw_tot = 10.0 * sw_w + 9.0 * sw_gap;
                                let s_start_x = -sw_tot / 2.0;
                                for i in 0..10 {
                                    let sx0 = s_start_x + i as f32 * (sw_w + sw_gap);
                                    let sx1 = sx0 + sw_w;
                                    if mx >= sx0 && mx <= sx1 && my >= 0.46 && my <= 0.55 {
                                        state.selected_slot = i;
                                        state.update_ui();
                                        return;
                                    }
                                }

                                // Curseurs RVB
                                let sl_x0 = -0.32;
                                let sl_x1 = 0.18;
                                if mx >= sl_x0 - 0.02 && mx <= sl_x1 + 0.02 {
                                    let slider_clicked = if my >= 0.33 && my <= 0.41 {
                                        Some(0)
                                    } else if my >= 0.26 && my <= 0.34 {
                                        Some(1)
                                    } else if my >= 0.19 && my <= 0.27 {
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

                                // Nuancier rapide
                                let pw_w = 0.076;
                                let pw_gap = 0.012;
                                let pw_tot = 10.0 * pw_w + 9.0 * pw_gap;
                                let pw_start_x = -pw_tot / 2.0;
                                for (i, &preset_col) in PRESET_SWATCHES.iter().enumerate() {
                                    let px0 = pw_start_x + i as f32 * (pw_w + pw_gap);
                                    let px1 = px0 + pw_w;
                                    if mx >= px0 && mx <= px1 && my >= 0.08 && my <= 0.14 {
                                        state.hotbar_colors[state.selected_slot] = preset_col;
                                        state.update_ui();
                                        return;
                                    }
                                }

                                // Sélecteur de résolutions d'octree (Sub & Super Cubes)
                                let sizes = [1, 2, 4, 8, 16];
                                let btn_w = 0.17;
                                let btn_gap = 0.02;
                                let btot = 5.0 * btn_w + 4.0 * btn_gap;
                                let bstart = -btot / 2.0;
                                for (i, &sz) in sizes.iter().enumerate() {
                                    let bx0 = bstart + i as f32 * (btn_w + btn_gap);
                                    let bx1 = bx0 + btn_w;
                                    if mx >= bx0 && mx <= bx1 && my >= -0.07 && my <= 0.01 {
                                        state.edit_size = sz;
                                        state.update_ui();
                                        return;
                                    }
                                }

                                // Bouton Mode
                                if mx >= -0.45 && mx <= -0.16 && my >= -0.22 && my <= -0.13 {
                                    state.toggle_play_mode();
                                    return;
                                }

                                // Bouton Preset Monde
                                if mx >= -0.14 && mx <= 0.15 && my >= -0.22 && my <= -0.13 {
                                    state.cycle_world_generator();
                                    return;
                                }

                                // Bouton Nettoyer la scène
                                if mx >= 0.17 && mx <= 0.46 && my >= -0.22 && my <= -0.13 {
                                    state.clear_all_blocks();
                                    return;
                                }

                                // Bouton Reprendre
                                if mx >= -0.32 && mx <= -0.02 && my >= -0.36 && my <= -0.27 {
                                    state.toggle_menu();
                                    return;
                                }

                                // Bouton Quitter
                                if mx >= 0.02 && mx <= 0.32 && my >= -0.36 && my <= -0.27 {
                                    event_loop.exit();
                                    return;
                                }
                            } else {
                                state.active_slider = None;
                            }
                        }
                    } else if element_state == ElementState::Pressed {
                        match button {
                            MouseButton::Left => state.input.action_add = true,
                            MouseButton::Right => state.input.action_remove = true,
                            MouseButton::Middle => state.input.action_pick = true,
                            _ => {}
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
            if !state.menu_open {
                if let DeviceEvent::MouseMotion { delta } = event {
                    state.camera.yaw += (delta.0 as f32) * 0.002;
                    state.camera.pitch -= (delta.1 as f32) * 0.002;
                    state.camera.pitch = state.camera.pitch.clamp(-1.5, 1.5);
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