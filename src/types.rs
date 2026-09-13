use glam::{Mat4, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::PathBuf;

pub const MAX_DEPTH: u8 = 16;
pub const VERTICAL_FOV_DEGREES: f32 = 60.0;
pub const MIN_VOXEL_SIZE: f32 = 0.015625;
pub const SVO_PAGE_SIZE: usize = 2048;

pub type Palette = Vec<[f32; 3]>;

#[inline(always)]
pub fn morton_part_1by2(mut x: u32) -> u64 {
    x &= 0x1f_ffff;
    let mut n = x as u64;
    n = (n | (n << 32)) & 0x1f00000000ffff;
    n = (n | (n << 16)) & 0x1f0000ff0000ff;
    n = (n | (n << 8))  & 0x100f00f00f00f00f;
    n = (n | (n << 4))  & 0x10c30c30c30c30c3;
    n = (n | (n << 2))  & 0x1249249249249249;
    n
}

#[inline(always)]
pub fn morton_encode_coords(x: i32, y: i32, z: i32) -> u64 {
    const OFFSET: i32 = 1 << 20;
    let ux = (x + OFFSET).max(0) as u32;
    let uy = (y + OFFSET).max(0) as u32;
    let uz = (z + OFFSET).max(0) as u32;
    (morton_part_1by2(ux) << 2) | (morton_part_1by2(uy) << 1) | morton_part_1by2(uz)
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum ActiveMenu {
    None, Edit, Pause, ImportParams, Voxelizing, Controls, BgColorModal,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum ToolType {
    Pencil, Sphere, Cylinder, Disc, Box, Line, Paint, Replace,
    Cone, Pyramid, Torus, Bucket, Select,
}

impl ToolType {
    pub fn name(&self) -> &'static str {
        match self {
            ToolType::Pencil => "PENCIL [V]",
            ToolType::Sphere => "SPHERE [O]",
            ToolType::Cylinder => "CYLINDER [Y]",
            ToolType::Disc => "DISC [U]",
            ToolType::Box => "BOX [B]",
            ToolType::Line => "LINE/PIPE [L]",
            ToolType::Paint => "PAINT [K]",
            ToolType::Replace => "REPLACE [G]",
            ToolType::Cone => "CONE [J]",
            ToolType::Pyramid => "PYRAMID [N]",
            ToolType::Torus => "TORUS [T]",
            ToolType::Bucket => "BUCKET [I]",
            ToolType::Select => "SELECT [S]",
        }
    }
    pub fn short_name(&self) -> &'static str {
        match self {
            ToolType::Pencil => "PEN", ToolType::Sphere => "SPH",
            ToolType::Cylinder => "CYL", ToolType::Disc => "DSC",
            ToolType::Box => "BOX", ToolType::Line => "LIN",
            ToolType::Paint => "PNT", ToolType::Replace => "REP",
            ToolType::Cone => "CON", ToolType::Pyramid => "PYR",
            ToolType::Torus => "TOR", ToolType::Bucket => "BCK",
            ToolType::Select => "SEL",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SelectionData {
    pub anchor: Option<Vec3>,
    pub bounds: Option<(Vec3, Vec3)>,
    pub captured_voxels: Vec<(Vec3, f32, u16)>,
    pub floating_voxels: Vec<(Vec3, f32, u16)>,
    pub is_floating: bool,
}

// ---------- Transform Gizmo ----------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TransformAxis { X, Y, Z }

impl TransformAxis {
    pub fn dir(self) -> Vec3 {
        match self { TransformAxis::X => Vec3::X, TransformAxis::Y => Vec3::Y, TransformAxis::Z => Vec3::Z }
    }
    pub fn color(self, hovered: bool, active: bool) -> [f32; 4] {
        let (r, g, b) = match self {
            TransformAxis::X => (0.92, 0.25, 0.28),
            TransformAxis::Y => (0.30, 0.85, 0.25),
            TransformAxis::Z => (0.20, 0.55, 0.98),
        };
        if active       { [1.0, 1.0, 1.0, 1.0] }
        // FIX: Added _f32 suffix to resolve ambiguous float type for .min()
        else if hovered { [(r * 1.25_f32).min(1.0), (g * 1.25_f32).min(1.0), (b * 1.25_f32).min(1.0), 1.0] }
        else            { [r, g, b, 0.95] }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GizmoTransformMode { All, Move, Rotate, Scale }

impl GizmoTransformMode {
    pub fn next(self) -> Self {
        match self {
            Self::All    => Self::Move,
            Self::Move   => Self::Rotate,
            Self::Rotate => Self::Scale,
            Self::Scale  => Self::All,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::All    => "ALL (TRANSFORM)",
            Self::Move   => "MOVE",
            Self::Rotate => "ROTATE",
            Self::Scale  => "SCALE",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TransformGizmo {
    pub mode: GizmoTransformMode,
    pub dragging_handle: Option<(GizmoTransformMode, TransformAxis)>,
    pub hover_handle: Option<(GizmoTransformMode, TransformAxis)>,
    pub drag_start_world: Vec3,
    pub drag_start_cursor: [f32; 2],
    pub accumulated_move: Vec3,
    pub accumulated_angle: f32,
    pub accumulated_scale: f32,
    pub snapshot: Vec<(Vec3, f32, u16)>,
}

impl Default for TransformGizmo {
    fn default() -> Self {
        Self {
            mode: GizmoTransformMode::All,
            dragging_handle: None,
            hover_handle: None,
            drag_start_world: Vec3::ZERO,
            drag_start_cursor: [0.0, 0.0],
            accumulated_move: Vec3::ZERO,
            accumulated_angle: 0.0,
            accumulated_scale: 1.0,
            snapshot: Vec::new(),
        }
    }
}

// ---------- ToolState ----------

#[derive(Clone, Debug)]
pub struct ToolState {
    pub active_tool: ToolType,
    pub brush_radius: f32,
    pub cylinder_height: f32,
    pub line_radius: f32,
    pub hollow: bool,
    pub pending_anchor: Option<Vec3>,
    pub selection: SelectionData,
    pub clipboard: Vec<(Vec3, f32, u16)>,
    pub gizmo: TransformGizmo,
}

impl Default for ToolState {
    fn default() -> Self {
        Self {
            active_tool: ToolType::Pencil,
            brush_radius: 3.0,
            cylinder_height: 5.0,
            line_radius: 0.0,
            hollow: false,
            pending_anchor: None,
            selection: SelectionData::default(),
            clipboard: Vec::new(),
            gizmo: TransformGizmo::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct VoxelDelta {
    pub pos: [f32; 3],
    pub size: f32,
    pub old_material: u16,
    pub new_material: u16,
}

#[derive(Clone)]
pub struct HistoryAction {
    pub removed: Vec<(Vec3, f32, u16)>,
    pub added: Vec<(Vec3, f32, u16)>,
}

pub struct HistoryManager {
    pub undo_stack: VecDeque<HistoryAction>,
    pub redo_stack: VecDeque<HistoryAction>,
    pub max_history: usize,
}

impl HistoryManager {
    pub fn new(max_history: usize) -> Self {
        Self { undo_stack: VecDeque::with_capacity(max_history), redo_stack: VecDeque::new(), max_history }
    }
    pub fn record(&mut self, action: HistoryAction) {
        if action.removed.is_empty() && action.added.is_empty() { return; }
        if self.undo_stack.len() >= self.max_history { self.undo_stack.pop_front(); }
        self.undo_stack.push_back(action);
        self.redo_stack.clear();
    }
}

#[derive(PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum PlayMode { Flying, Real }

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum WorldType { Empty, Flat, Hills, Mountains, FloatingIslands }

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

pub fn default_ortho_size() -> f32 { 36.0 }

#[derive(Serialize, Deserialize, Clone)]
pub struct CubeEdit {
    pub pos: [f32; 3],
    pub size: f32,
    pub material: u16,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct PlayerCollider {
    pub radius: f32,
    pub height: f32,
    pub eye_offset: f32,
}

impl Default for PlayerCollider {
    fn default() -> Self { Self { radius: 0.35, height: 1.8, eye_offset: 1.6 } }
}

#[derive(Serialize, Deserialize)]
pub struct SaveData {
    pub player_pos: [f32; 3],
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    #[serde(default)] pub is_ortho: bool,
    #[serde(default = "default_ortho_size")] pub ortho_size: f32,
    pub play_mode: PlayMode,
    pub world_type: WorldType,
    pub seed: u32,
    pub hotbar_colors: [[f32; 3]; 10],
    pub palette: Vec<[f32; 3]>,
    pub octree: crate::engine::Octree,
    pub bg_color: [f32; 3],
}

#[derive(Clone)]
pub struct GlbImportSettings {
    pub selected_file: Option<PathBuf>,
    pub target_height: f32,
    pub voxel_size: f32,
    pub place_at_aim: bool,
    pub palette_size: usize,
    pub max_gpu_nodes: usize,
}

impl Default for GlbImportSettings {
    fn default() -> Self {
        Self {
            selected_file: None, target_height: 24.0, voxel_size: 0.25,
            place_at_aim: true, palette_size: 256, max_gpu_nodes: 30_000_000,
        }
    }
}

impl GlbImportSettings {
    pub fn estimate_cost(&self) -> (u64, f32) {
        let grid_h = (self.target_height / self.voxel_size.max(0.01)).round() as u64;
        let est_surface = grid_h * grid_h * 10;
        let est_interior_collapsed = (grid_h * grid_h * grid_h) / 20;
        let est_voxels = (est_surface + est_interior_collapsed).min(100_000_000);
        let est_mb = (est_voxels as f32 * 16.0) / (1024.0 * 1024.0);
        (est_voxels, est_mb)
    }
    pub fn is_safe(&self) -> bool {
        let (est_voxels, _) = self.estimate_cost();
        (est_voxels as usize) <= self.max_gpu_nodes
    }
}

pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub is_ortho: bool,
    pub ortho_size: f32,
}

impl Camera {
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let (sin_p, cos_p) = self.pitch.sin_cos();
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let dir = Vec3::new(cos_y * cos_p, sin_p, sin_y * cos_p).normalize();
        let view = glam::camera::rh::view::look_at_mat4(self.position, self.position + dir, Vec3::Y);
        let proj = if self.is_ortho {
            let half_h = self.ortho_size * 0.5;
            let half_w = half_h * aspect;
            glam::camera::rh::proj::directx::orthographic(-half_w, half_w, -half_h, half_h, 0.05, 50000.0)
        } else {
            glam::camera::rh::proj::directx::perspective(VERTICAL_FOV_DEGREES.to_radians(), aspect, 0.05, 100000.0)
        };
        proj * view
    }
    pub fn forward(&self) -> Vec3 {
        let (sin_p, cos_p) = self.pitch.sin_cos();
        let (sin_y, cos_y) = self.yaw.sin_cos();
        Vec3::new(cos_y * cos_p, sin_p, sin_y * cos_p).normalize()
    }
    pub fn right(&self) -> Vec3 {
        let (sin_y, cos_y) = self.yaw.sin_cos();
        Vec3::new(-sin_y, 0.0, cos_y).normalize()
    }
    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize()
    }
    pub fn ray_from_ndc(&self, ndc_x: f32, ndc_y: f32, aspect: f32) -> (Vec3, Vec3) {
        let inv_vp = self.view_proj(aspect).inverse();
        if self.is_ortho {
            let p_near = inv_vp * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
            let dir = self.forward();
            let orig = p_near.truncate() / p_near.w - dir * 2000.0;
            (orig, dir)
        } else {
            let p_near = inv_vp * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
            let p_far  = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
            let p_w_near = p_near.truncate() / p_near.w;
            let p_w_far  = p_far.truncate()  / p_far.w;
            let orig = self.position;
            let dir = (p_w_far - p_w_near).normalize();
            (orig, dir)
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub show_borders: f32,
    pub world_min: [f32; 3],
    pub world_size: f32,
    pub tight_min: [f32; 3],
    pub has_voxels: f32,
    pub tight_max: [f32; 3],
    pub _pad0: f32,
    pub is_ortho: f32,
    pub ortho_size: f32,
    pub screen_size: [f32; 2],
    pub bg_color: [f32; 3],
    pub _pad1: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UIVertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

impl UIVertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<UIVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}