// types.rs
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const CHUNK_SIZE: f32 = 32.0;
pub const MAX_DEPTH: u8 = 16;
pub const GRID_RES: u32 = 1 << MAX_DEPTH;
pub const MIN_VOXEL_SIZE: f32 = CHUNK_SIZE / (GRID_RES as f32);

pub type ChunkPos = (i32, i32, i32);
pub type Palette = Vec<[f32; 3]>;

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum ActiveMenu { None, Edit, Pause, ImportParams, Voxelizing, Controls }

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
            WorldType::Empty => "EMPTY CANVAS", WorldType::Flat => "FLAT",
            WorldType::Hills => "HILLS", WorldType::Mountains => "MOUNTAINS",
            WorldType::FloatingIslands => "ISLANDS",
        }
    }
}

pub fn default_ortho_size() -> f32 { 36.0 }

#[derive(Serialize, Deserialize, Clone)]
pub struct CubeEdit { pub pos: [f32; 3], pub size: f32, pub material: u16 }

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct PlayerCollider { pub radius: f32, pub height: f32, pub eye_offset: f32 }

impl Default for PlayerCollider {
    fn default() -> Self { Self { radius: 0.35, height: 1.8, eye_offset: 1.6 } }
}

#[derive(Serialize, Deserialize)]
pub struct SaveData {
    pub player_pos: [f32; 3], pub camera_yaw: f32, pub camera_pitch: f32,
    #[serde(default)] pub is_ortho: bool,
    #[serde(default = "default_ortho_size")] pub ortho_size: f32,
    pub play_mode: PlayMode, pub world_type: WorldType, pub seed: u32,
    pub hotbar_colors: [[f32; 3]; 10], pub palette: Vec<[f32; 3]>,
    #[serde(default)] pub cube_edits: Vec<CubeEdit>,
}

#[derive(Clone)]
pub struct GlbImportSettings {
    pub selected_file: Option<PathBuf>, pub target_height: f32,
    pub voxel_size: f32, pub place_at_aim: bool, pub palette_size: usize,
}

impl Default for GlbImportSettings {
    fn default() -> Self {
        Self { selected_file: None, target_height: 16.0, voxel_size: 0.25, place_at_aim: true, palette_size: 256 }
    }
}

pub struct Camera { pub position: Vec3, pub yaw: f32, pub pitch: f32, pub is_ortho: bool, pub ortho_size: f32 }
impl Camera {
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let (sin_p, cos_p) = self.pitch.sin_cos(); let (sin_y, cos_y) = self.yaw.sin_cos();
        let dir = Vec3::new(cos_y * cos_p, sin_p, sin_y * cos_p).normalize();
        let view = glam::camera::rh::view::look_at_mat4(self.position, self.position + dir, Vec3::Y);
        let proj = if self.is_ortho {
            let half_h = self.ortho_size * 0.5; let half_w = half_h * aspect;
            glam::camera::rh::proj::directx::orthographic(-half_w, half_w, -half_h, half_h, -100000.0, 100000.0)
        } else {
            glam::camera::rh::proj::directx::perspective((60.0_f32).to_radians(), aspect, 0.05, 100000.0)
        };
        proj * view
    }
    pub fn forward(&self) -> Vec3 { let (sin_p, cos_p) = self.pitch.sin_cos(); let (sin_y, cos_y) = self.yaw.sin_cos(); Vec3::new(cos_y * cos_p, sin_p, sin_y * cos_p).normalize() }
    pub fn right(&self) -> Vec3 { let (sin_y, cos_y) = self.yaw.sin_cos(); Vec3::new(-sin_y, 0.0, cos_y).normalize() }
    pub fn up(&self) -> Vec3 { self.right().cross(self.forward()).normalize() }
}

#[repr(C)] #[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform { pub view_proj: [[f32; 4]; 4], pub show_borders: f32, pub _pad: [f32; 3] }

#[repr(C)] #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex { pub position: [f32; 3], pub normal: [f32; 3], pub color: [f32; 3], pub uv: [f32; 2] }
impl Vertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3, 3 => Float32x2];
    pub fn desc() -> wgpu::VertexBufferLayout<'static> { wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress, step_mode: wgpu::VertexStepMode::Vertex, attributes: &Self::ATTRIBS } }
}

#[repr(C)] #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UIVertex { pub position: [f32; 2], pub color: [f32; 4] }
impl UIVertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];
    pub fn desc() -> wgpu::VertexBufferLayout<'static> { wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<UIVertex>() as wgpu::BufferAddress, step_mode: wgpu::VertexStepMode::Vertex, attributes: &Self::ATTRIBS } }
}