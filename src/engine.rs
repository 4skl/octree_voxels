use crate::types::*;
use glam::Vec3;
use noise::{Fbm, NoiseFn, Perlin};
use std::collections::HashSet;

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
    pub free_list: Vec<u32>,
    pub dirty_pages: HashSet<usize>,
    pub total_voxels: usize,
    pub world_min: Vec3,
    pub world_size: f32,
}

impl Octree {
    pub fn new() -> Self {
        let mut nodes = Vec::with_capacity(8192);
        nodes.push(OctreeNode::default());
        let mut dirty_pages = HashSet::new();
        dirty_pages.insert(0);
        Self {
            nodes,
            root_index: 0,
            free_list: Vec::with_capacity(512),
            dirty_pages,
            total_voxels: 0,
            world_min: Vec3::new(-256.0, -256.0, -256.0),
            world_size: 512.0,
        }
    }

    #[inline]
    pub fn mark_dirty(&mut self, start: usize, count: usize) {
        let start_page = start / SVO_PAGE_SIZE;
        let end_page = (start + count).saturating_sub(1) / SVO_PAGE_SIZE;
        for p in start_page..=end_page {
            self.dirty_pages.insert(p);
        }
    }

    pub fn mark_all_dirty(&mut self) {
        let total_pages = (self.nodes.len() + SVO_PAGE_SIZE - 1) / SVO_PAGE_SIZE;
        for p in 0..total_pages {
            self.dirty_pages.insert(p);
        }
    }

    pub fn allocate_block(&mut self) -> u32 {
        if let Some(ptr) = self.free_list.pop() {
            for c in 0..8 {
                self.nodes[(ptr + c) as usize] = OctreeNode::default();
            }
            self.mark_dirty(ptr as usize, 8);
            ptr
        } else {
            let ptr = self.nodes.len() as u32;
            if self.nodes.capacity() < self.nodes.len() + 8 {
                self.nodes.reserve(8192);
            }
            self.nodes.resize(self.nodes.len() + 8, OctreeNode::default());
            self.mark_dirty(ptr as usize, 8);
            ptr
        }
    }

    pub fn free_subtree(&mut self, child_ptr: u32) {
        if child_ptr == 0 { return; }
        for i in 0..8 {
            let sub = self.nodes[(child_ptr + i) as usize].child_pointer;
            if sub != 0 {
                self.free_subtree(sub);
            }
            self.nodes[(child_ptr + i) as usize] = OctreeNode::default();
        }
        self.mark_dirty(child_ptr as usize, 8);
        self.free_list.push(child_ptr);
    }

    pub fn expand_root(&mut self, pt: Vec3) {
        let mut octant = 0;
        let mut new_min = self.world_min;
        let old_size = self.world_size;

        if pt.x < self.world_min.x {
            octant |= 1;
            new_min.x -= old_size;
        }
        if pt.y < self.world_min.y {
            octant |= 2;
            new_min.y -= old_size;
        }
        if pt.z < self.world_min.z {
            octant |= 4;
            new_min.z -= old_size;
        }

        let old_root = self.nodes[self.root_index as usize];

        if old_root.child_mask == 0 && old_root.material_id == 0 {
            self.world_min = new_min;
            self.world_size = old_size * 2.0;
            return;
        }

        let child_ptr = self.allocate_block();
        self.nodes[(child_ptr + octant as u32) as usize] = old_root;

        let new_mask = if old_root.child_mask != 0 || old_root.material_id != 0 { 1 << octant } else { 0 };
        self.nodes[self.root_index as usize] = OctreeNode {
            child_mask: new_mask,
            child_pointer: child_ptr,
            material_id: 0,
            _pad: 0,
        };

        self.world_min = new_min;
        self.world_size = old_size * 2.0;
        self.mark_dirty(self.root_index as usize, 1);
    }

    pub fn ensure_bounds(&mut self, pos: Vec3, size: f32) {
        let max_pos = pos + Vec3::splat(size);
        while pos.x < self.world_min.x || pos.y < self.world_min.y || pos.z < self.world_min.z
            || max_pos.x > self.world_min.x + self.world_size
            || max_pos.y > self.world_min.y + self.world_size
            || max_pos.z > self.world_min.z + self.world_size
        {
            self.expand_root(pos);
        }
    }

    pub fn insert_cube_world(&mut self, pos: Vec3, size: f32, material: u16, auto_collapse: bool) {
        self.ensure_bounds(pos, size);

        let mut current_idx = self.root_index as usize;
        let mut curr_min = self.world_min;
        let mut curr_size = self.world_size;
        let mut path: Vec<(usize, usize)> = Vec::with_capacity(18);

        while curr_size > size * 1.001 {
            let old_mat = self.nodes[current_idx].material_id;
            let child_ptr = self.nodes[current_idx].child_pointer;

            let c_ptr = if child_ptr == 0 {
                if old_mat == material as u32 { return; }
                let new_ptr = self.allocate_block();
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
                self.mark_dirty(current_idx, 1);
                new_ptr
            } else {
                child_ptr
            };

            let half = curr_size * 0.5;
            let mut octant = 0;
            if pos.x >= curr_min.x + half { octant |= 1; curr_min.x += half; }
            if pos.y >= curr_min.y + half { octant |= 2; curr_min.y += half; }
            if pos.z >= curr_min.z + half { octant |= 4; curr_min.z += half; }

            path.push((current_idx, octant));
            current_idx = (c_ptr + octant as u32) as usize;
            curr_size = half;
        }

        let old_ptr = self.nodes[current_idx].child_pointer;
        if old_ptr != 0 {
            self.free_subtree(old_ptr);
        }
        self.nodes[current_idx].material_id = material as u32;
        self.nodes[current_idx].child_pointer = 0;
        self.nodes[current_idx].child_mask = if material != 0 { 0xFF } else { 0 };
        self.mark_dirty(current_idx, 1);

        if material != 0 {
            for (p_idx, oct) in path.iter().rev() {
                self.nodes[*p_idx].child_mask |= 1 << oct;
            }
        }

        if auto_collapse {
            self.collapse_path(&path);
        }
    }

    pub fn collapse_path(&mut self, path: &[(usize, usize)]) {
        for &(node_idx, _) in path.iter().rev() {
            if !self.try_collapse_node(node_idx) {
                break;
            }
        }
    }

    pub fn try_collapse_node(&mut self, node_idx: usize) -> bool {
        let child_ptr = self.nodes[node_idx].child_pointer;
        if child_ptr == 0 { return true; }

        let mut all_leaves = true;
        for i in 0..8 {
            if self.nodes[(child_ptr + i) as usize].child_pointer != 0 {
                all_leaves = false;
                break;
            }
        }

        if all_leaves {
            let first_mat = self.nodes[child_ptr as usize].material_id;
            let mut all_same = true;
            for i in 1..8 {
                if self.nodes[(child_ptr + i) as usize].material_id != first_mat {
                    all_same = false;
                    break;
                }
            }

            if all_same {
                self.free_subtree(child_ptr);
                self.nodes[node_idx].material_id = first_mat;
                self.nodes[node_idx].child_pointer = 0;
                self.nodes[node_idx].child_mask = if first_mat != 0 { 0xFF } else { 0 };
                self.mark_dirty(node_idx, 1);
                return true;
            }
        }

        let mut mask = 0;
        for i in 0..8 {
            let c = &self.nodes[(child_ptr + i) as usize];
            if c.material_id != 0 || c.child_pointer != 0 {
                mask |= 1 << i;
            }
        }
        self.nodes[node_idx].child_mask = mask;
        self.nodes[node_idx].material_id = 0;
        self.mark_dirty(node_idx, 1);
        false
    }

    pub fn collapse(&mut self, node_idx: usize) -> bool {
        let child_ptr = self.nodes[node_idx].child_pointer;
        if child_ptr == 0 { return true; }

        let mut all_leaves = true;
        for i in 0..8 {
            let is_leaf = self.collapse((child_ptr + i) as usize);
            if !is_leaf {
                all_leaves = false;
            }
        }

        if all_leaves {
            let first_mat = self.nodes[child_ptr as usize].material_id;
            let mut all_same = true;
            for i in 1..8 {
                if self.nodes[(child_ptr + i) as usize].material_id != first_mat {
                    all_same = false;
                    break;
                }
            }

            if all_same {
                self.free_subtree(child_ptr);
                self.nodes[node_idx].material_id = first_mat;
                self.nodes[node_idx].child_pointer = 0;
                self.nodes[node_idx].child_mask = if first_mat != 0 { 0xFF } else { 0 };
                self.mark_dirty(node_idx, 1);
                return true;
            }
        }

        let mut mask = 0;
        for i in 0..8 {
            let c = &self.nodes[(child_ptr + i) as usize];
            if c.material_id != 0 || c.child_pointer != 0 {
                mask |= 1 << i;
            }
        }
        self.nodes[node_idx].child_mask = mask;
        self.nodes[node_idx].material_id = 0;
        self.mark_dirty(node_idx, 1);
        false
    }

    pub fn query_point(&self, pos: Vec3) -> u16 {
        if pos.x < self.world_min.x || pos.y < self.world_min.y || pos.z < self.world_min.z
            || pos.x >= self.world_min.x + self.world_size
            || pos.y >= self.world_min.y + self.world_size
            || pos.z >= self.world_min.z + self.world_size
        {
            return 0;
        }

        let mut current_idx = self.root_index as usize;
        let mut curr_min = self.world_min;
        let mut curr_size = self.world_size;

        loop {
            let node = &self.nodes[current_idx];
            if node.child_pointer == 0 {
                return node.material_id as u16;
            }
            let half = curr_size * 0.5;
            let mut octant = 0;
            if pos.x >= curr_min.x + half { octant |= 1; curr_min.x += half; }
            if pos.y >= curr_min.y + half { octant |= 2; curr_min.y += half; }
            if pos.z >= curr_min.z + half { octant |= 4; curr_min.z += half; }

            if (node.child_mask & (1 << octant)) == 0 {
                return 0;
            }
            current_idx = (node.child_pointer + octant) as usize;
            curr_size = half;
        }
    }

    pub fn recalculate_voxel_count(&mut self) {
        self.total_voxels = self.count_occupied_voxels(self.root_index as usize);
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

pub fn bresenham_3d(p0: [i32; 3], p1: [i32; 3]) -> Vec<[i32; 3]> {
    let mut points = Vec::new();
    let (mut x, mut y, mut z) = (p0[0], p0[1], p0[2]);
    let (x2, y2, z2) = (p1[0], p1[1], p1[2]);

    let dx = (x2 - x).abs();
    let dy = (y2 - y).abs();
    let dz = (z2 - z).abs();

    let xs = if x2 > x { 1 } else { -1 };
    let ys = if y2 > y { 1 } else { -1 };
    let zs = if z2 > z { 1 } else { -1 };

    if dx >= dy && dx >= dz {
        let mut p1_err = 2 * dy - dx;
        let mut p2_err = 2 * dz - dx;
        while x != x2 {
            points.push([x, y, z]);
            x += xs;
            if p1_err >= 0 { y += ys; p1_err -= 2 * dx; }
            if p2_err >= 0 { z += zs; p2_err -= 2 * dx; }
            p1_err += 2 * dy;
            p2_err += 2 * dz;
        }
    } else if dy >= dx && dy >= dz {
        let mut p1_err = 2 * dx - dy;
        let mut p2_err = 2 * dz - dy;
        while y != y2 {
            points.push([x, y, z]);
            y += ys;
            if p1_err >= 0 { x += xs; p1_err -= 2 * dy; }
            if p2_err >= 0 { z += zs; p2_err -= 2 * dz; }
            p1_err += 2 * dx;
            p2_err += 2 * dz;
        }
    } else {
        let mut p1_err = 2 * dy - dz;
        let mut p2_err = 2 * dx - dz;
        while z != z2 {
            points.push([x, y, z]);
            z += zs;
            if p1_err >= 0 { y += ys; p1_err -= 2 * dz; }
            if p2_err >= 0 { x += xs; p2_err -= 2 * dz; }
            p1_err += 2 * dy;
            p2_err += 2 * dx;
        }
    }
    points.push([x2, y2, z2]);
    points
}

pub fn rasterize_sphere(octree: &Octree, center: Vec3, radius: f32, voxel_size: f32, material: u16) -> Vec<VoxelDelta> {
    let mut deltas = Vec::new();
    let r_vox = (radius / voxel_size).ceil() as i32;
    let r_sq = radius * radius;

    for dx in -r_vox..=r_vox {
        for dy in -r_vox..=r_vox {
            for dz in -r_vox..=r_vox {
                let offset = Vec3::new(dx as f32, dy as f32, dz as f32) * voxel_size;
                if offset.length_squared() <= r_sq {
                    let p = center + offset;
                    let old_mat = octree.query_point(p);
                    if old_mat != material {
                        deltas.push(VoxelDelta { pos: p.to_array(), size: voxel_size, old_material: old_mat, new_material: material });
                    }
                }
            }
        }
    }
    deltas
}

pub fn rasterize_box(octree: &Octree, corner_a: Vec3, corner_b: Vec3, voxel_size: f32, material: u16) -> Vec<VoxelDelta> {
    let mut deltas = Vec::new();
    let min_p = corner_a.min(corner_b);
    let max_p = corner_a.max(corner_b);

    let sx = ((max_p.x - min_p.x) / voxel_size).round() as i32;
    let sy = ((max_p.y - min_p.y) / voxel_size).round() as i32;
    let sz = ((max_p.z - min_p.z) / voxel_size).round() as i32;

    for ix in 0..=sx {
        for iy in 0..=sy {
            for iz in 0..=sz {
                let p = min_p + Vec3::new(ix as f32, iy as f32, iz as f32) * voxel_size;
                let old_mat = octree.query_point(p);
                if old_mat != material {
                    deltas.push(VoxelDelta { pos: p.to_array(), size: voxel_size, old_material: old_mat, new_material: material });
                }
            }
        }
    }
    deltas
}

pub fn rasterize_line_pipe(octree: &Octree, p0: Vec3, p1: Vec3, radius: f32, voxel_size: f32, material: u16) -> Vec<VoxelDelta> {
    let mut visited: HashSet<[i32; 3]> = HashSet::new();
    let mut deltas = Vec::new();

    let v0 = (p0 / voxel_size).round();
    let v1 = (p1 / voxel_size).round();
    let line_cells = bresenham_3d(
        [v0.x as i32, v0.y as i32, v0.z as i32],
        [v1.x as i32, v1.y as i32, v1.z as i32],
    );

    let r_vox = (radius / voxel_size).ceil() as i32;
    let r_sq = radius * radius;

    for cell in line_cells {
        let center_world = Vec3::new(cell[0] as f32, cell[1] as f32, cell[2] as f32) * voxel_size;
        for dx in -r_vox..=r_vox {
            for dy in -r_vox..=r_vox {
                for dz in -r_vox..=r_vox {
                    let offset = Vec3::new(dx as f32, dy as f32, dz as f32) * voxel_size;
                    if radius > 0.0 && offset.length_squared() > r_sq { continue; }
                    let p = center_world + offset;
                    let cell_coord = [
                        (p.x / voxel_size).round() as i32,
                        (p.y / voxel_size).round() as i32,
                        (p.z / voxel_size).round() as i32,
                    ];
                    if visited.insert(cell_coord) {
                        let old_mat = octree.query_point(p);
                        if old_mat != material {
                            deltas.push(VoxelDelta { pos: p.to_array(), size: voxel_size, old_material: old_mat, new_material: material });
                        }
                    }
                }
            }
        }
    }
    deltas
}

pub fn apply_deltas(octree: &mut Octree, deltas: &[VoxelDelta], use_new: bool) {
    if deltas.is_empty() { return; }

    if deltas.len() == 1 {
        let d = &deltas[0];
        let mat = if use_new { d.new_material } else { d.old_material };
        octree.insert_cube_world(Vec3::from(d.pos), d.size, mat, true);
    } else {
        for d in deltas {
            let mat = if use_new { d.new_material } else { d.old_material };
            octree.insert_cube_world(Vec3::from(d.pos), d.size, mat, false);
        }
        octree.collapse(octree.root_index as usize);
    }
    octree.recalculate_voxel_count();
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
                        let mat = if y == 0 { 2 } else { 1 };
                        octree.insert_cube_world(Vec3::new(x as f32, y as f32, z as f32), 1.0, mat, false);
                    }
                }
            }
        }
        WorldType::Hills => {
            for x in -64..64 {
                for z in -64..64 {
                    let height = ((fbm.get([x as f64 * 0.04, z as f64 * 0.04]) + 1.0) * 8.0).floor() as i32;
                    for y in -4..=height {
                        let mat = if y == height { 2 } else { 1 };
                        octree.insert_cube_world(Vec3::new(x as f32, y as f32, z as f32), 1.0, mat, false);
                    }
                }
            }
        }
        WorldType::Mountains => {
            for x in -64..64 {
                for z in -64..64 {
                    let height = (fbm.get([x as f64 * 0.02, z as f64 * 0.02]).abs() * 24.0).floor() as i32;
                    for y in -6..=height {
                        let mat = if y >= 18 { 3 } else if y == height { 2 } else { 1 };
                        octree.insert_cube_world(Vec3::new(x as f32, y as f32, z as f32), 1.0, mat, false);
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
                            octree.insert_cube_world(Vec3::new(x as f32, y as f32, z as f32), 1.0, 2, false);
                        }
                    }
                }
            }
        }
    }

    for edit in edits {
        octree.insert_cube_world(Vec3::from(edit.pos), edit.size, edit.material, false);
    }

    octree.collapse(octree.root_index as usize);
    octree.recalculate_voxel_count();
    octree
}

pub struct RaycastHit {
    pub hit_pos: Vec3,
    pub voxel_min: Vec3,
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
    raycast_node(octree, octree.root_index as usize, origin, dir, inv_dir, octree.world_min, octree.world_size, &mut closest_dist)
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

    let mut cand = [(0.0f32, 0usize, Vec3::ZERO); 8];
    let mut count = 0;

    for i in 0..8 {
        if (node.child_mask & (1 << i)) != 0 {
            let cx = if (i & 1) != 0 { half_size } else { 0.0 };
            let cy = if (i & 2) != 0 { half_size } else { 0.0 };
            let cz = if (i & 4) != 0 { half_size } else { 0.0 };
            let c_min = node_min + Vec3::new(cx, cy, cz);
            let c_max = c_min + Vec3::splat(half_size);
            if let Some((ct, _)) = ray_aabb_intersect(origin, inv_dir, c_min, c_max) {
                if ct < *closest_dist {
                    cand[count] = (ct, (node.child_pointer + i as u32) as usize, c_min);
                    count += 1;
                }
            }
        }
    }

    cand[..count].sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    for i in 0..count {
        if cand[i].0 >= *closest_dist { break; }
        if let Some(hit) = raycast_node(octree, cand[i].1, origin, dir, inv_dir, cand[i].2, half_size, closest_dist) {
            best_hit = Some(hit);
        }
    }
    best_hit
}