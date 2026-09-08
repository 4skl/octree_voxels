use crate::types::*;
use glam::Vec3;
use noise::{NoiseFn, Fbm, Perlin};

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OctreeNode {
    pub child_mask: u32,
    pub child_pointer: u32,
    pub material_id: u32,
    pub _pad: u32,
}

#[derive(Clone)]
pub struct Octree {
    pub nodes: Vec<OctreeNode>,
    pub root_index: u32,
}

impl Octree {
    pub fn new() -> Self {
        let mut nodes = Vec::with_capacity(8192);
        nodes.push(OctreeNode::default());
        Self { nodes, root_index: 0 }
    }

    pub fn insert_cube(&mut self, mut x: u32, mut y: u32, mut z: u32, depth: u8, material: u16, auto_collapse: bool) {
        if depth == 0 {
            self.nodes[self.root_index as usize] = OctreeNode {
                child_mask: if material != 0 { 0xFF } else { 0 },
                material_id: material as u32,
                child_pointer: 0,
                _pad: 0,
            };
            return;
        }
        let mut current_idx = self.root_index as usize;
        let mut half_size = GRID_RES / 2;

        for _ in 0..depth {
            let old_mat = self.nodes[current_idx].material_id;
            let child_ptr = self.nodes[current_idx].child_pointer;

            if child_ptr == 0 {
                if old_mat == material as u32 { return; }
                let new_ptr = self.nodes.len() as u32;
                if self.nodes.capacity() < self.nodes.len() + 8 { self.nodes.reserve(8192); }
                self.nodes.resize(self.nodes.len() + 8, OctreeNode::default());
                if old_mat != 0 {
                    for c in 0..8 { self.nodes[(new_ptr + c) as usize].material_id = old_mat; }
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

        self.nodes[current_idx].material_id = material as u32;
        self.nodes[current_idx].child_pointer = 0;
        self.nodes[current_idx].child_mask = if material != 0 { 0xFF } else { 0 };
        if auto_collapse { self.collapse(self.root_index as usize); }
    }

    pub fn collapse(&mut self, node_idx: usize) -> bool {
        let child_ptr = self.nodes[node_idx].child_pointer;
        if child_ptr == 0 { return true; }
        let mut all_same = true;
        let first_mat = self.nodes[child_ptr as usize].material_id;

        for i in 0..8 {
            let c_idx = (child_ptr + i) as usize;
            let is_leaf = self.collapse(c_idx);
            if !is_leaf || self.nodes[c_idx].material_id != first_mat { all_same = false; }
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
            if self.nodes[c_idx].material_id != 0 || self.nodes[c_idx].child_pointer != 0 { mask |= 1 << i; }
        }
        self.nodes[node_idx].child_mask = mask;
        false
    }

    #[allow(dead_code)]
    pub fn query_node(&self, mut x: u32, mut y: u32, mut z: u32) -> (u16, u32) {
        let mut current_idx = self.root_index as usize;
        let mut size = GRID_RES;
        for _ in 0..MAX_DEPTH {
            let node = &self.nodes[current_idx];
            if node.child_pointer == 0 { return (node.material_id as u16, size); }
            let half = size / 2;
            let mut octant = 0;
            if x >= half { octant |= 1; x -= half; }
            if y >= half { octant |= 2; y -= half; }
            if z >= half { octant |= 4; z -= half; }
            if (node.child_mask & (1 << octant)) == 0 { return (0, half); }
            current_idx = (node.child_pointer + octant) as usize;
            size = half;
        }
        (self.nodes[current_idx].material_id as u16, size)
    }

    pub fn count_occupied_voxels(&self, node_idx: usize) -> usize {
        let node = &self.nodes[node_idx];
        if node.child_pointer == 0 {
            return if node.material_id != 0 { 1 } else { 0 };
        }
        let mut total = 0;
        for i in 0..8 {
            if (node.child_mask & (1 << i)) != 0 {
                total += self.count_occupied_voxels((node.child_pointer + i) as usize);
            }
        }
        total
    }

    pub fn compute_voxel_bounds(&self, node_idx: usize, min_p: Vec3, size: f32, out_min: &mut Vec3, out_max: &mut Vec3) -> bool {
        let node = &self.nodes[node_idx];
        if node.child_pointer == 0 {
            if node.material_id != 0 {
                *out_min = out_min.min(min_p);
                *out_max = out_max.max(min_p + Vec3::splat(size));
                return true;
            }
            return false;
        }
        let half = size * 0.5;
        let mut has_voxels = false;
        for i in 0..8 {
            if (node.child_mask & (1 << i)) != 0 {
                let offset = Vec3::new(
                    if (i & 1) != 0 { half } else { 0.0 },
                    if (i & 2) != 0 { half } else { 0.0 },
                    if (i & 4) != 0 { half } else { 0.0 },
                );
                if self.compute_voxel_bounds((node.child_pointer + i) as usize, min_p + offset, half, out_min, out_max) {
                    has_voxels = true;
                }
            }
        }
        has_voxels
    }
}

pub fn generate_world_terrain(world_type: WorldType, seed: u32, edits: &[CubeEdit]) -> Octree {
    let mut octree = Octree::new();
    let fbm = Fbm::<Perlin>::new(seed);

    match world_type {
        WorldType::Empty => {}
        WorldType::Flat => {
            for x in -64..64 {
                for z in -64..64 {
                    for y in -2..1 {
                        let wx = x as f32; let wy = y as f32; let wz = z as f32;
                        let gx = (((wx - WORLD_MIN.x) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
                        let gy = (((wy - WORLD_MIN.y) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
                        let gz = (((wz - WORLD_MIN.z) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
                        let grid_size = ((1.0 / WORLD_SIZE) * GRID_RES as f32).round().max(1.0) as u32;
                        let depth = MAX_DEPTH - grid_size.trailing_zeros().min(MAX_DEPTH as u32) as u8;
                        octree.insert_cube(gx, gy, gz, depth, if y == 0 { 2 } else { 1 }, false);
                    }
                }
            }
        }
        WorldType::Hills => {
            for x in -64..64 {
                for z in -64..64 {
                    let height = ((fbm.get([x as f64 * 0.04, z as f64 * 0.04]) + 1.0) * 8.0).floor() as i32;
                    for y in -4..=height {
                        let wx = x as f32; let wy = y as f32; let wz = z as f32;
                        let gx = (((wx - WORLD_MIN.x) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
                        let gy = (((wy - WORLD_MIN.y) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
                        let gz = (((wz - WORLD_MIN.z) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
                        let grid_size = ((1.0 / WORLD_SIZE) * GRID_RES as f32).round().max(1.0) as u32;
                        let depth = MAX_DEPTH - grid_size.trailing_zeros().min(MAX_DEPTH as u32) as u8;
                        octree.insert_cube(gx, gy, gz, depth, if y == height { 2 } else { 1 }, false);
                    }
                }
            }
        }
        WorldType::Mountains => {
            for x in -64..64 {
                for z in -64..64 {
                    let height = (fbm.get([x as f64 * 0.02, z as f64 * 0.02]).abs() * 24.0).floor() as i32;
                    for y in -6..=height {
                        let wx = x as f32; let wy = y as f32; let wz = z as f32;
                        let gx = (((wx - WORLD_MIN.x) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
                        let gy = (((wy - WORLD_MIN.y) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
                        let gz = (((wz - WORLD_MIN.z) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
                        let grid_size = ((1.0 / WORLD_SIZE) * GRID_RES as f32).round().max(1.0) as u32;
                        let depth = MAX_DEPTH - grid_size.trailing_zeros().min(MAX_DEPTH as u32) as u8;
                        let mat = if y >= 18 { 3 } else if y == height { 2 } else { 1 };
                        octree.insert_cube(gx, gy, gz, depth, mat, false);
                    }
                }
            }
        }
        WorldType::FloatingIslands => {
            for x in -48..48 {
                for z in -48..48 {
                    for y in 0..32 {
                        let density = fbm.get([x as f64 * 0.06, y as f64 * 0.08, z as f64 * 0.06]) - ((y as f64) - 16.0) * 0.05;
                        if density > 0.12 {
                            let wx = x as f32; let wy = y as f32; let wz = z as f32;
                            let gx = (((wx - WORLD_MIN.x) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
                            let gy = (((wy - WORLD_MIN.y) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
                            let gz = (((wz - WORLD_MIN.z) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
                            let grid_size = ((1.0 / WORLD_SIZE) * GRID_RES as f32).round().max(1.0) as u32;
                            let depth = MAX_DEPTH - grid_size.trailing_zeros().min(MAX_DEPTH as u32) as u8;
                            octree.insert_cube(gx, gy, gz, depth, 2, false);
                        }
                    }
                }
            }
        }
    }

    for edit in edits {
        let gx = (((edit.pos[0] - WORLD_MIN.x) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
        let gy = (((edit.pos[1] - WORLD_MIN.y) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
        let gz = (((edit.pos[2] - WORLD_MIN.z) / WORLD_SIZE) * GRID_RES as f32).floor() as u32;
        let grid_size = ((edit.size / WORLD_SIZE) * GRID_RES as f32).round().max(1.0) as u32;
        let depth = MAX_DEPTH - grid_size.trailing_zeros().min(MAX_DEPTH as u32) as u8;
        octree.insert_cube(gx, gy, gz, depth, edit.material, false);
    }

    octree.collapse(octree.root_index as usize);
    octree
}

pub struct RaycastHit {
    pub hit_pos: Vec3,
    pub voxel_min: Vec3,
    #[allow(dead_code)]
    pub voxel_size: f32,
    pub normal: Vec3,
    pub material: u16,
}

pub fn ray_aabb_intersect(ray_origin: Vec3, ray_dir_inv: Vec3, min: Vec3, max: Vec3) -> Option<(f32, Vec3)> {
    let mut t_near = f32::NEG_INFINITY;
    let mut t_far = f32::INFINITY;
    let mut normal = Vec3::ZERO;
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

pub fn raycast_octree(octree: &Octree, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<RaycastHit> {
    let dir = dir.normalize();
    let inv_dir = Vec3::new(
        if dir.x.abs() > 1e-8 { 1.0 / dir.x } else { 1e8 * dir.x.signum() },
        if dir.y.abs() > 1e-8 { 1.0 / dir.y } else { 1e8 * dir.y.signum() },
        if dir.z.abs() > 1e-8 { 1.0 / dir.z } else { 1e8 * dir.z.signum() },
    );

    let mut closest_dist = max_dist;
    raycast_node(octree, octree.root_index as usize, origin, dir, inv_dir, WORLD_MIN, WORLD_SIZE, &mut closest_dist)
}

fn raycast_node(
    octree: &Octree, node_idx: usize, origin: Vec3, dir: Vec3, inv_dir: Vec3,
    node_min: Vec3, node_size: f32, closest_dist: &mut f32
) -> Option<RaycastHit> {
    let node = &octree.nodes[node_idx];
    if node.child_mask == 0 && node.material_id == 0 { return None; }
    let node_max = node_min + Vec3::splat(node_size);
    let (t_hit, normal) = ray_aabb_intersect(origin, inv_dir, node_min, node_max)?;
    if t_hit >= *closest_dist { return None; }

    if node.child_pointer == 0 {
        if node.material_id != 0 {
            *closest_dist = t_hit;
            return Some(RaycastHit {
                hit_pos: origin + dir * t_hit,
                voxel_min: node_min,
                voxel_size: node_size,
                normal,
                material: node.material_id as u16,
            });
        }
        return None;
    }

    let mut best_hit = None;
    let half_size = node_size * 0.5;
    let mut order = 0;
    if inv_dir.x < 0.0 { order |= 1; }
    if inv_dir.y < 0.0 { order |= 2; }
    if inv_dir.z < 0.0 { order |= 4; }

    for i in 0..8 {
        let child_idx = i ^ order;
        if (node.child_mask & (1 << child_idx)) != 0 {
            let cx = if (child_idx & 1) != 0 { half_size } else { 0.0 };
            let cy = if (child_idx & 2) != 0 { half_size } else { 0.0 };
            let cz = if (child_idx & 4) != 0 { half_size } else { 0.0 };
            let c_min = node_min + Vec3::new(cx, cy, cz);
            if let Some(hit) = raycast_node(octree, (node.child_pointer + child_idx) as usize, origin, dir, inv_dir, c_min, half_size, closest_dist) {
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
            let cx = if (i & 1) != 0 { half_size } else { 0.0 };
            let cy = if (i & 2) != 0 { half_size } else { 0.0 };
            let cz = if (i & 4) != 0 { half_size } else { 0.0 };
            if check_octree_aabb(octree, (node.child_pointer + i) as usize, node_min + Vec3::new(cx, cy, cz), half_size, target_min, target_max) { return true; }
        }
    }
    false
}