mod engine;
mod types;
mod ui;
mod voxelize;
use engine::*;
use types::*;
use ui::*;
use voxelize::*;
use glam::Vec3;
use std::sync::mpsc;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};
#[cfg(target_os = "linux")]
use winit::platform::wayland::WindowAttributesExtWayland;
#[cfg(target_os = "linux")]
use winit::platform::x11::WindowAttributesExtX11;

fn point_to_segment_dist(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [p[0] - a[0], p[1] - a[1]];
    let ab_len2 = ab[0] * ab[0] + ab[1] * ab[1];
    if ab_len2 < 1e-12 {
        return (ap[0] * ap[0] + ap[1] * ap[1]).sqrt();
    }
    let mut t = (ap[0] * ab[0] + ap[1] * ab[1]) / ab_len2;
    t = t.clamp(0.0, 1.0);
    let closest = [a[0] + t * ab[0], a[1] + t * ab[1]];
    let dx = p[0] - closest[0];
    let dy = p[1] - closest[1];
    (dx * dx + dy * dy).sqrt()
}

#[derive(Default)]
struct InputState {
    action_add: bool,
    action_remove: bool,
    action_pick: bool,
    ctrl_pressed: bool,
    shift_pressed: bool,
    wireframe_mode: bool,
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
    ui_buffer_capacity: usize,
    ui_vertices_count: u32,
    camera_buffer: wgpu::Buffer,
    svo_buffer: wgpu::Buffer,
    svo_capacity: usize,
    svo_bind_group: wgpu::BindGroup,
    svo_bind_group_layout: wgpu::BindGroupLayout,
    palette_buffer: wgpu::Buffer,
    window: Arc<Window>,
    camera: Camera,
    input: InputState,
    octree: Octree,
    play_mode: PlayMode,
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
    collider: PlayerCollider,
    hide_ui: bool,
    bg_color: [f32; 3],
    world_type: WorldType,
    seed: u32,
    cube_edits: Vec<CubeEdit>,
    error_banner: Option<(String, Instant)>,
    fps: f32,
    fps_frame_counter: u32,
    fps_timer: Instant,
    tool_state: ToolState,
    history: HistoryManager,
    orbit_pivot: Vec3,
    mmb_dragging: bool,
    ui_dirty: bool,
    focused_voxel_size: Option<f32>,
    preview_deltas: Vec<VoxelDelta>,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let mut size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            size = winit::dpi::PhysicalSize::new(1280, 720);
        }
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: Default::default(),
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_limits: adapter.limits(),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let mut config = surface.get_default_config(&adapter, size.width, size.height).unwrap();
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);

        let camera = Camera {
            position: Vec3::new(0.0, 16.0, 36.0),
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: -0.30,
            is_ortho: false,
            ortho_size: 32.0,
        };
        let aspect = if config.height > 0 { config.width as f32 / config.height as f32 } else { 1.0 };
        let vp = camera.view_proj(aspect);

        let world_type = WorldType::Empty;
        let seed = 42;
        let cube_edits = Vec::new();
        let octree = generate_world_terrain(world_type, seed, &cube_edits);

        let bg_color = [0.12, 0.14, 0.18];
        let camera_uniform = CameraUniform {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            camera_pos: camera.position.to_array(),
            show_borders: 0.0,
            world_min: octree.world_min.to_array(),
            world_size: octree.world_size,
            tight_min: octree.tight_min.to_array(),
            has_voxels: if octree.total_voxels > 0 { 1.0 } else { 0.0 },
            tight_max: octree.tight_max.to_array(),
            _pad0: 0.0,
            is_ortho: if camera.is_ortho { 1.0 } else { 0.0 },
            ortho_size: camera.ortho_size,
            screen_size: [config.width as f32, config.height as f32],
            bg_color,
            _pad1: 0.0,
        };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let hotbar_colors = PRESET_SWATCHES;
        let palette = Arc::new(RwLock::new(hotbar_colors.to_vec()));
        let mut pal_vec4 = vec![[0.0; 4]; 8192];
        for (i, c) in hotbar_colors.iter().enumerate() {
            pal_vec4[i] = [c[0], c[1], c[2], 1.0];
        }
        let palette_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Palette Storage Buffer"),
            contents: bytemuck::cast_slice(&pal_vec4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

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
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            }),
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
            fragment: Some(wgpu::FragmentState {
                module: &ui_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let mut glb_settings = GlbImportSettings::default();
        glb_settings.max_gpu_nodes = (device.limits().max_storage_buffer_binding_size as usize / 16).min(35_000_000);

        let ui_buffer_capacity = (65536 * std::mem::size_of::<UIVertex>()).max(2_097_152);
        let ui_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UI Buffer"),
            size: ui_buffer_capacity as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut app_state = Self {
            window, surface, device, queue, config, size, render_pipeline, ui_pipeline,
            ui_vertex_buffer, ui_buffer_capacity, ui_vertices_count: 0,
            camera_buffer, svo_buffer, svo_capacity, svo_bind_group, svo_bind_group_layout, palette_buffer, camera,
            input: InputState::default(), octree, play_mode: PlayMode::Flying, selected_slot: 0,
            hotbar_colors, palette, active_menu: ActiveMenu::None, cursor_pos: [0.0, 0.0], active_slider: None, edit_size: 1.0,
            last_target: None, cursor_free: true, gimbal_dragging: false, gimbal_drag_moved: false, prev_cursor_pos: [0.0, 0.0],
            glb_settings, voxelize_rx: None, voxelize_progress: 0.0, voxelize_stage: String::new(), collider: PlayerCollider::default(),
            hide_ui: false, bg_color, world_type, seed, cube_edits, error_banner: None,
            fps: 60.0, fps_frame_counter: 0, fps_timer: Instant::now(),
            tool_state: ToolState::default(), history: HistoryManager::new(64),
            orbit_pivot: Vec3::ZERO, mmb_dragging: false, ui_dirty: true,
            focused_voxel_size: None, preview_deltas: Vec::new(),
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
        for (i, c) in pal.iter().enumerate() { pal_vec4[i] = [c[0], c[1], c[2], 1.0]; }
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
            self.queue.write_buffer(&self.svo_buffer, 0, bytemuck::cast_slice(&self.octree.nodes));
            self.octree.dirty_pages.clear();
            return;
        }
        let elem_size = std::mem::size_of::<OctreeNode>();
        let page_nodes = SVO_PAGE_SIZE;
        let dirty_pages: Vec<usize> = self.octree.dirty_pages.drain().collect();
        for page in dirty_pages {
            let start_node = page * page_nodes;
            if start_node < self.octree.nodes.len() {
                let end_node = (start_node + page_nodes).min(self.octree.nodes.len());
                let offset = (start_node * elem_size) as u64;
                let slice = &self.octree.nodes[start_node..end_node];
                self.queue.write_buffer(&self.svo_buffer, offset, bytemuck::cast_slice(slice));
            }
        }
    }

    pub fn sync_svo_buffer_full(&mut self) {
        self.octree.mark_all_dirty();
        self.sync_svo_buffer();
    }

    pub fn update_orbit_position(&mut self) {
        let cur_dist = (self.camera.position - self.orbit_pivot).length().max(1.0);
        self.camera.position = self.orbit_pivot - self.camera.forward() * cur_dist;
        self.update_camera_buffer();
        self.ui_dirty = true;
    }

    pub fn focus_on_scene(&mut self) {
        let has_voxels = self.octree.total_voxels > 0;
        let min_bound = self.octree.tight_min;
        let max_bound = self.octree.tight_max;
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
        self.orbit_pivot = center;
        self.camera.position = center - self.camera.forward() * required_dist.clamp(8.0, 50000.0);
        self.camera.ortho_size = (half_extents.y.max(half_extents.x / aspect) * 2.2).clamp(16.0, 50000.0);
        self.update_camera_buffer();
        self.ui_dirty = true;
        self.window.request_redraw();
    }

    pub fn toggle_projection(&mut self) {
        self.camera.is_ortho = !self.camera.is_ortho;
        self.update_camera_buffer();
        self.ui_dirty = true;
        self.window.request_redraw();
    }

    pub fn update_camera_buffer(&self) {
        let aspect = if self.config.height > 0 { self.config.width as f32 / self.config.height as f32 } else { 1.0 };
        let vp = self.camera.view_proj(aspect);
        let camera_uniform = CameraUniform {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            camera_pos: self.camera.position.to_array(),
            show_borders: if self.input.wireframe_mode { 1.0 } else { 0.0 },
            world_min: self.octree.world_min.to_array(),
            world_size: self.octree.world_size,
            tight_min: self.octree.tight_min.to_array(),
            has_voxels: if self.octree.total_voxels > 0 { 1.0 } else { 0.0 },
            tight_max: self.octree.tight_max.to_array(),
            _pad0: 0.0,
            is_ortho: if self.camera.is_ortho { 1.0 } else { 0.0 },
            ortho_size: self.camera.ortho_size,
            screen_size: [self.config.width as f32, self.config.height as f32],
            bg_color: self.bg_color,
            _pad1: 0.0,
        };
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));
    }

    pub fn set_menu(&mut self, menu: ActiveMenu) {
        self.active_menu = menu;
        self.active_slider = None;
        self.gimbal_dragging = false;
        self.mmb_dragging = false;
        self.update_camera_buffer();
        self.ui_dirty = true;
        self.window.request_redraw();
    }

    pub fn save_game(&self, filename: &str) -> std::io::Result<()> {
        std::fs::write(
            filename,
            serde_json::to_string_pretty(&SaveData {
                player_pos: self.camera.position.to_array(),
                camera_yaw: self.camera.yaw,
                camera_pitch: self.camera.pitch,
                is_ortho: self.camera.is_ortho,
                ortho_size: self.camera.ortho_size,
                play_mode: self.play_mode,
                world_type: self.world_type,
                seed: self.seed,
                hotbar_colors: self.hotbar_colors,
                palette: self.palette.read().unwrap().clone(),
                octree: self.octree.clone(),
                bg_color: self.bg_color,
            })?,
        )?;
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
        self.hotbar_colors = data.hotbar_colors;
        *self.palette.write().unwrap() = data.palette;
        self.world_type = data.world_type;
        self.seed = data.seed;
        self.bg_color = data.bg_color;
        self.octree = data.octree;
        self.octree.recalculate_voxel_count();
        self.history = HistoryManager::new(64);
        self.sync_svo_buffer_full();
        self.sync_palette_buffer();
        self.update_camera_buffer();
        self.ui_dirty = true;
        self.window.request_redraw();
        Ok(())
    }

    pub fn clear_all_blocks(&mut self) {
        self.cube_edits.clear();
        self.octree = Octree::new();
        self.octree.update_bounds();
        self.history = HistoryManager::new(64);
        self.sync_svo_buffer_full();
        self.update_camera_buffer();
        self.ui_dirty = true;
        self.window.request_redraw();
    }

    pub fn cycle_world_generator(&mut self) {
        self.world_type = self.world_type.next();
        self.cube_edits.clear();
        self.octree = generate_world_terrain(self.world_type, self.seed, &self.cube_edits);
        self.history = HistoryManager::new(64);
        self.sync_svo_buffer_full();
        self.update_camera_buffer();
        self.ui_dirty = true;
        self.window.request_redraw();
    }

    pub fn scale_voxel_size(&mut self, multiply: bool) {
        if multiply { self.edit_size = (self.edit_size * 2.0).min(64.0); }
        else { self.edit_size = (self.edit_size * 0.5).max(MIN_VOXEL_SIZE); }
        self.ui_dirty = true;
        self.window.request_redraw();
    }

    pub fn toggle_play_mode(&mut self) {
        self.play_mode = if self.play_mode == PlayMode::Real { PlayMode::Flying } else { PlayMode::Real };
        self.ui_dirty = true;
        self.window.request_redraw();
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

    fn compute_preview_deltas(&self, target_vec: Vec3) -> Vec<VoxelDelta> {
        let s = self.edit_size;
        let active_mat = (self.selected_slot + 1) as u16;
        match self.tool_state.active_tool {
            ToolType::Pencil | ToolType::Paint => {
                let old_mat = self.octree.query_point(target_vec);
                vec![VoxelDelta { pos: target_vec.to_array(), size: s, old_material: old_mat, new_material: active_mat }]
            }
            ToolType::Sphere => rasterize_sphere(&self.octree, target_vec + Vec3::splat(s * 0.5), self.tool_state.brush_radius, s, active_mat, self.tool_state.hollow),
            ToolType::Cylinder => rasterize_cylinder(&self.octree, target_vec, self.tool_state.brush_radius, self.tool_state.cylinder_height, s, active_mat, self.tool_state.hollow),
            ToolType::Cone => rasterize_cone(&self.octree, target_vec, self.tool_state.brush_radius, self.tool_state.cylinder_height, s, active_mat, self.tool_state.hollow),
            ToolType::Pyramid => rasterize_pyramid(&self.octree, target_vec, self.tool_state.brush_radius, self.tool_state.cylinder_height, s, active_mat, self.tool_state.hollow),
            ToolType::Torus => {
                let minor_r = (self.tool_state.brush_radius * 0.35).max(1.0);
                rasterize_torus(&self.octree, target_vec, self.tool_state.brush_radius, minor_r, s, active_mat, self.tool_state.hollow)
            }
            ToolType::Disc => rasterize_disc(&self.octree, target_vec, self.tool_state.brush_radius, s, active_mat, self.tool_state.hollow),
            ToolType::Box => {
                if let Some(anchor) = self.tool_state.pending_anchor {
                    rasterize_box(&self.octree, anchor, target_vec, s, active_mat, self.tool_state.hollow)
                } else {
                    let old_mat = self.octree.query_point(target_vec);
                    vec![VoxelDelta { pos: target_vec.to_array(), size: s, old_material: old_mat, new_material: active_mat }]
                }
            }
            ToolType::Line => {
                if let Some(anchor) = self.tool_state.pending_anchor {
                    rasterize_line_pipe(&self.octree, anchor, target_vec, self.tool_state.line_radius, s, active_mat)
                } else {
                    let old_mat = self.octree.query_point(target_vec);
                    vec![VoxelDelta { pos: target_vec.to_array(), size: s, old_material: old_mat, new_material: active_mat }]
                }
            }
            ToolType::Replace => rasterize_replace(&self.octree, target_vec + Vec3::splat(s * 0.5), self.tool_state.brush_radius, s, self.octree.query_point(target_vec), active_mat),
            ToolType::Bucket => rasterize_bucket(&self.octree, target_vec, s, active_mat, self.tool_state.bucket_limit),
            ToolType::Select => Vec::new(),
        }
    }

    fn rotate_floating_or_selection(&mut self) {
        if self.tool_state.selection.is_floating {
            for (pos, s, _) in self.tool_state.selection.floating_voxels.iter_mut() {
                let v_center = *pos + Vec3::splat(*s * 0.5);
                let rot_center = Vec3::new(-v_center.z, v_center.y, v_center.x);
                *pos = rot_center - Vec3::splat(*s * 0.5);
            }
            self.ui_dirty = true;
            self.window.request_redraw();
        } else if let Some((min_b, max_b)) = self.tool_state.selection.bounds {
            let center = (min_b + max_b) * 0.5;
            let old_half = (max_b - min_b) * 0.5;
            let new_half = Vec3::new(old_half.z, old_half.y, old_half.x);
            let new_min = center - new_half;
            let new_max = center + new_half;
            let mut removed = Vec::new();
            self.octree.capture_aabb(self.octree.root_index as usize, self.octree.world_min, self.octree.world_size, min_b, max_b, &mut removed);
            if removed.is_empty() {
                self.tool_state.selection.bounds = Some((new_min, new_max));
                self.ui_dirty = true;
                self.window.request_redraw();
                return;
            }
            for (p, s, _) in &removed { self.octree.insert_cube_world(*p, *s, 0, false); }
            let mut added = Vec::with_capacity(removed.len());
            for (p, s, m) in removed.iter() {
                let v_center_rel = (*p + Vec3::splat(*s * 0.5)) - center;
                let rot_center = Vec3::new(-v_center_rel.z, v_center_rel.y, v_center_rel.x);
                let new_corner = center + rot_center - Vec3::splat(*s * 0.5);
                let snapped = Vec3::new(
                    (new_corner.x / *s).round() * *s,
                    (new_corner.y / *s).round() * *s,
                    (new_corner.z / *s).round() * *s,
                );
                self.octree.insert_cube_world(snapped, *s, *m, false);
                added.push((snapped, *s, *m));
            }
            self.octree.collapse(self.octree.root_index as usize);
            self.octree.recalculate_voxel_count();
            self.history.record(HistoryAction { removed, added: added.clone() });
            self.tool_state.selection.bounds = Some((new_min, new_max));
            self.tool_state.selection.captured_voxels = added;
            self.sync_svo_buffer_full();
            self.ui_dirty = true;
            self.window.request_redraw();
        }
    }

    fn copy_selection(&mut self) {
        if let Some((min_b, max_b)) = self.tool_state.selection.bounds {
            let center = (min_b + max_b) * 0.5;
            let mut voxels = Vec::new();
            self.octree.capture_aabb(self.octree.root_index as usize, self.octree.world_min, self.octree.world_size, min_b, max_b, &mut voxels);
            if !voxels.is_empty() {
                self.tool_state.clipboard = voxels.into_iter().map(|(p, s, m)| (p - center, s, m)).collect();
                self.ui_dirty = true;
            }
        }
    }

    fn paste_clipboard(&mut self) {
        if !self.tool_state.clipboard.is_empty() {
            self.tool_state.selection.floating_voxels = self.tool_state.clipboard.clone();
            self.tool_state.selection.is_floating = true;
            self.tool_state.gizmo.dragging_handle = None;
            self.tool_state.gizmo.snapshot.clear();
            self.tool_state.gizmo.accumulated_move = Vec3::ZERO;
            self.tool_state.gizmo.accumulated_angle = 0.0;
            self.tool_state.gizmo.accumulated_scale = 1.0;
            self.ui_dirty = true;
            self.window.request_redraw();
        }
    }

    fn grab_selection(&mut self) {
        if let Some((min_b, max_b)) = self.tool_state.selection.bounds {
            let center = (min_b + max_b) * 0.5;
            let mut voxels = Vec::new();
            self.octree.capture_aabb(self.octree.root_index as usize, self.octree.world_min, self.octree.world_size, min_b, max_b, &mut voxels);
            if voxels.is_empty() { return; }
            for (p, s, _) in &voxels { self.octree.insert_cube_world(*p, *s, 0, false); }
            self.octree.collapse(self.octree.root_index as usize);
            self.octree.recalculate_voxel_count();
            self.history.record(HistoryAction { removed: voxels.clone(), added: Vec::new() });
            self.sync_svo_buffer_full();
            self.tool_state.selection.floating_voxels = voxels.into_iter().map(|(p, s, m)| (p - center, s, m)).collect();
            self.tool_state.selection.is_floating = true;
            self.tool_state.selection.bounds = None;
            self.tool_state.gizmo.dragging_handle = None;
            self.tool_state.gizmo.snapshot.clear();
            self.tool_state.gizmo.accumulated_move = Vec3::ZERO;
            self.tool_state.gizmo.accumulated_angle = 0.0;
            self.tool_state.gizmo.accumulated_scale = 1.0;
            self.ui_dirty = true;
            self.window.request_redraw();
        }
    }

    fn clear_selection_voxels(&mut self) {
        if let Some((min_b, max_b)) = self.tool_state.selection.bounds.take() {
            let mut removed = Vec::new();
            self.octree.capture_aabb(self.octree.root_index as usize, self.octree.world_min, self.octree.world_size, min_b, max_b, &mut removed);
            if !removed.is_empty() {
                for (p, s, _) in &removed { self.octree.insert_cube_world(*p, *s, 0, false); }
                self.octree.collapse(self.octree.root_index as usize);
                self.octree.recalculate_voxel_count();
                self.history.record(HistoryAction { removed, added: Vec::new() });
                self.sync_svo_buffer_full();
            }
            self.ui_dirty = true;
            self.window.request_redraw();
        }
    }

    fn clear_gizmo_state(&mut self) {
        self.tool_state.gizmo.dragging_handle = None;
        self.tool_state.gizmo.hover_handle = None;
        self.tool_state.gizmo.snapshot.clear();
        self.tool_state.gizmo.accumulated_move = Vec3::ZERO;
        self.tool_state.gizmo.accumulated_angle = 0.0;
        self.tool_state.gizmo.accumulated_scale = 1.0;
    }

    fn execute_tool_action(&mut self, is_removal: bool) {
        let target_vec = match self.last_target {
            Some(t) => Vec3::from(t),
            None => return,
        };
        if self.tool_state.selection.is_floating {
            let mut added = Vec::with_capacity(self.tool_state.selection.floating_voxels.len());
            for &(rel, s, m) in &self.tool_state.selection.floating_voxels {
                let v_pos = target_vec + rel;
                self.octree.insert_cube_world(v_pos, s, m, false);
                added.push((v_pos, s, m));
            }
            self.octree.collapse(self.octree.root_index as usize);
            self.octree.recalculate_voxel_count();
            self.history.record(HistoryAction { removed: Vec::new(), added });
            self.tool_state.selection.is_floating = false;
            self.clear_gizmo_state();
            self.sync_svo_buffer_full();
            self.ui_dirty = true;
            return;
        }
        if self.tool_state.active_tool == ToolType::Select {
            if is_removal {
                if self.tool_state.selection.anchor.is_some() {
                    self.tool_state.selection.anchor = None;
                } else {
                    self.tool_state.selection.bounds = None;
                    self.tool_state.selection.captured_voxels.clear();
                }
                self.ui_dirty = true;
                return;
            }

            if let Some(anchor) = self.tool_state.selection.anchor.take() {
                let min_b = anchor.min(target_vec) - Vec3::splat(1e-4);
                let max_b = anchor.max(target_vec) + Vec3::splat(self.edit_size + 1e-4);
                let mut captured = Vec::new();
                self.octree.capture_aabb(self.octree.root_index as usize, self.octree.world_min, self.octree.world_size, min_b, max_b, &mut captured);
                self.tool_state.selection.bounds = Some((min_b, max_b));
                self.tool_state.selection.captured_voxels = captured;
            } else {
                self.tool_state.selection.anchor = Some(target_vec);
            }
            self.ui_dirty = true;
            return;
        }
        if matches!(self.tool_state.active_tool, ToolType::Box | ToolType::Line) && self.tool_state.pending_anchor.is_none() {
            self.tool_state.pending_anchor = Some(target_vec);
            self.ui_dirty = true;
            return;
        }
        let mat = if is_removal { 0 } else { self.get_or_create_material(self.hotbar_colors[self.selected_slot]) };
        let deltas = self.compute_preview_deltas(target_vec);
        if deltas.is_empty() { return; }
        let mut aabb_min = Vec3::splat(f32::MAX);
        let mut aabb_max = Vec3::splat(f32::MIN);
        for d in &deltas {
            let p = Vec3::from(d.pos);
            aabb_min = aabb_min.min(p);
            aabb_max = aabb_max.max(p + Vec3::splat(d.size));
        }
        let mut removed = Vec::new();
        self.octree.capture_aabb(self.octree.root_index as usize, self.octree.world_min, self.octree.world_size, aabb_min, aabb_max, &mut removed);
        for mut d in deltas {
            d.new_material = mat;
            if d.old_material != d.new_material {
                apply_deltas(&mut self.octree, &[d], true);
            }
        }
        let mut added = Vec::new();
        self.octree.capture_aabb(self.octree.root_index as usize, self.octree.world_min, self.octree.world_size, aabb_min, aabb_max, &mut added);
        if self.tool_state.pending_anchor.is_some() {
            self.tool_state.pending_anchor = None;
        }
        self.history.record(HistoryAction { removed, added });
        self.sync_svo_buffer();
        self.update_camera_buffer();
        self.ui_dirty = true;
    }

    fn update_ui(&mut self) {
        let aspect = if self.config.height > 0 { self.config.width as f32 / self.config.height as f32 } else { 1.0 };
        let total_voxels = self.octree.total_voxels;
        let error_msg = self.error_banner.as_ref().map(|(msg, _)| msg.as_str());
        let size_str = if self.edit_size < 0.001 {
            format!("RES: {:.1e}", self.edit_size)
        } else if self.edit_size < 1.0 {
            format!("RES: 1/{} ({:.4})", (1.0 / self.edit_size).round() as u32, self.edit_size)
        } else {
            format!("RES: {:.0}X{:.0}", self.edit_size, self.edit_size)
        };
        let frame_ms = if self.fps > 0.0 { 1000.0 / self.fps } else { 0.0 };
        let hud_status = format!(
            "{:.0} FPS ({:.1}MS) | MODE: {} | PROJ: {} | WORLD: {} | VOXELS: {} | {}",
            self.fps, frame_ms,
            if self.play_mode == PlayMode::Flying { "FLY" } else { "REAL" },
            if self.camera.is_ortho { "ORTHO" } else { "PERSP" },
            self.world_type.name(),
            format_voxel_count(total_voxels),
            size_str
        );
        let tool_param_str = match self.tool_state.active_tool {
            ToolType::Cylinder | ToolType::Cone | ToolType::Pyramid => {
                format!("RAD: {:.0} | HGT: {:.0} | {}", self.tool_state.brush_radius, self.tool_state.cylinder_height, if self.tool_state.hollow { "HOLLOW" } else { "SOLID" })
            }
            ToolType::Sphere | ToolType::Disc | ToolType::Torus => {
                format!("RAD: {:.0} | {}", self.tool_state.brush_radius, if self.tool_state.hollow { "HOLLOW" } else { "SOLID" })
            }
            ToolType::Replace => format!("RAD: {:.0}", self.tool_state.brush_radius),
            ToolType::Line    => format!("PIPE: {:.1}", self.tool_state.line_radius),
            ToolType::Box     => format!("{}", if self.tool_state.hollow { "HOLLOW" } else { "SOLID" }),
            ToolType::Bucket  => format!("MAX: {}", self.tool_state.bucket_limit),
            ToolType::Pencil | ToolType::Paint => size_str.to_string(),
            ToolType::Select  => format!("GIZMO: {}", self.tool_state.gizmo.mode.label()),
        };
        let hud_tools = format!(
            "TOOL: {} | {} | CLIP: {} | UNDO: {}",
            self.tool_state.active_tool.name(),
            tool_param_str,
            self.tool_state.clipboard.len(),
            self.history.undo_stack.len()
        );
        let (est_count, est_mb) = self.glb_settings.estimate_cost();
        let glb_cost_str = format!(
            "EST: ~{} VOXELS ({:.0} MB SVO) | HW LIMIT: {}",
            format_voxel_count(est_count as usize), est_mb, format_voxel_count(self.glb_settings.max_gpu_nodes)
        );
        let verts = build_ui_vertices(
            self.selected_slot, self.active_menu, &self.hotbar_colors, self.play_mode, self.camera.is_ortho, self.world_type,
            self.edit_size, self.last_target, aspect, self.camera.forward(), self.camera.right(), self.camera.up(),
            self.cursor_free, &self.glb_settings, self.voxelize_progress, &self.voxelize_stage, self.camera.view_proj(aspect),
            error_msg, &self.tool_state, &hud_status, &hud_tools, &size_str, &glb_cost_str, self.bg_color,
            self.focused_voxel_size, &self.preview_deltas, self.camera.position, &self.octree,
        );
        let raw_bytes: &[u8] = bytemuck::cast_slice(&verts);
        if raw_bytes.len() > self.ui_buffer_capacity {
            self.ui_buffer_capacity = (raw_bytes.len() * 2).max(self.ui_buffer_capacity * 2);
            self.ui_vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("UI Buffer (Resized)"),
                size: self.ui_buffer_capacity as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        self.ui_vertices_count = verts.len() as u32;
        if !raw_bytes.is_empty() {
            self.queue.write_buffer(&self.ui_vertex_buffer, 0, raw_bytes);
        }
        self.ui_dirty = false;
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.update_camera_buffer();
            self.ui_dirty = true;
        }
    }

    fn update(&mut self, _dt: f32) {
        self.fps_frame_counter += 1;
        let elapsed = self.fps_timer.elapsed().as_secs_f32();
        if elapsed >= 0.5 {
            self.fps = self.fps_frame_counter as f32 / elapsed;
            self.fps_frame_counter = 0;
            self.fps_timer = Instant::now();
            self.ui_dirty = true;
        }
        if let Some((_, time)) = self.error_banner {
            if time.elapsed().as_secs() > 7 {
                self.error_banner = None;
                self.ui_dirty = true;
            }
        }
        let mut messages = Vec::new();
        if let Some(ref rx) = self.voxelize_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    VoxelizeMsg::Progress { percent, stage } => {
                        self.voxelize_progress = percent;
                        self.voxelize_stage = stage;
                        self.ui_dirty = true;
                    }
                    VoxelizeMsg::Done(res) => { messages.push(VoxelizeMsg::Done(res)); }
                }
            }
        }
        for msg in messages {
            match msg {
                VoxelizeMsg::Done(Ok(voxels)) => {
                    if let Some((first_pos, first_size, _)) = voxels.first() {
                        let mut min_bound = *first_pos;
                        let mut max_bound = *first_pos + Vec3::splat(*first_size);
                        for (pos, size, _) in &voxels {
                            min_bound = min_bound.min(*pos);
                            max_bound = max_bound.max(*pos + Vec3::splat(*size));
                        }
                        let total_size = (max_bound - min_bound).max_element();
                        self.octree.ensure_bounds(min_bound, total_size);
                    }
                    for (pos, size, mat) in voxels {
                        let old_mat = self.octree.query_point(pos);
                        if old_mat != mat {
                            self.octree.insert_cube_world(pos, size, mat, false);
                        }
                    }
                    self.octree.collapse(self.octree.root_index as usize);
                    self.octree.recalculate_voxel_count();
                    self.sync_svo_buffer_full();
                    self.sync_palette_buffer();
                    self.voxelize_rx = None;
                    self.set_menu(ActiveMenu::None);
                    self.focus_on_scene();
                    return;
                }
                VoxelizeMsg::Done(Err(err)) => {
                    eprintln!("GLB voxelization failed: {err}");
                    self.error_banner = Some((err, Instant::now()));
                    self.voxelize_rx = None;
                    self.set_menu(ActiveMenu::ImportParams);
                    return;
                }
                _ => {}
            }
        }
        if self.active_menu != ActiveMenu::None {
            if self.ui_dirty { self.update_ui(); }
            return;
        }

        let aspect = if self.config.height > 0 { self.config.width as f32 / self.config.height as f32 } else { 1.0 };
        let (ray_orig, ray_dir) = self.camera.ray_from_ndc(self.cursor_pos[0], self.cursor_pos[1], aspect);
        let hit = raycast_octree(&self.octree, ray_orig, ray_dir, 3000.0);
        let s = self.edit_size;
        let prev_target = self.last_target;
        if let Some(ref h) = hit {
            self.focused_voxel_size = Some(h.voxel_size);
            let p = if self.input.action_remove
                || matches!(self.tool_state.active_tool, ToolType::Paint | ToolType::Replace | ToolType::Bucket | ToolType::Select)
            {
                h.hit_pos - h.normal * (s * 0.5)
            } else {
                h.hit_pos + h.normal * (s * 0.5)
            };
            self.last_target = Some([(p.x / s).floor() * s, (p.y / s).floor() * s, (p.z / s).floor() * s]);
        } else {
            self.focused_voxel_size = None;
            if ray_dir.y.abs() > 1e-5 && (-ray_orig.y / ray_dir.y) > 0.0 && (-ray_orig.y / ray_dir.y) < 15000.0 {
                let t = -ray_orig.y / ray_dir.y;
                let p = ray_orig + ray_dir * t;
                self.last_target = Some([(p.x / s).floor() * s, 0.0, (p.z / s).floor() * s]);
            } else {
                let p = self.orbit_pivot;
                self.last_target = Some([(p.x / s).floor() * s, (p.y / s).floor() * s, (p.z / s).floor() * s]);
            }
        }
        if prev_target != self.last_target || self.ui_dirty {
            if let Some(target) = self.last_target {
                self.preview_deltas = self.compute_preview_deltas(Vec3::from(target));
            } else {
                self.preview_deltas.clear();
            }
            self.ui_dirty = true;
        }

        // --- Unified gizmo hover detection ---
        if self.tool_state.selection.is_floating && self.tool_state.gizmo.dragging_handle.is_none() {
            let pivot = self.last_target.map(Vec3::from).unwrap_or(self.orbit_pivot);
            let dist = (pivot - self.camera.position).length().max(1.0);
            let gizmo_len = dist * 0.12;
            let vp = self.camera.view_proj(aspect);
            let cur = [self.cursor_pos[0] * aspect, self.cursor_pos[1]];

            let mut best: Option<(GizmoTransformMode, TransformAxis, f32)> = None;
            let axes = [TransformAxis::X, TransformAxis::Y, TransformAxis::Z];

            // 1. Scale handles: cubes at 0.48 * gizmo_len
            if self.tool_state.gizmo.mode == GizmoTransformMode::All || self.tool_state.gizmo.mode == GizmoTransformMode::Scale {
                for axis in axes {
                    let box_p = pivot + axis.dir() * (gizmo_len * 0.48);
                    let v = vp * glam::Vec4::new(box_p.x, box_p.y, box_p.z, 1.0);
                    if v.w > 0.05 {
                        let ndc = [v.x / v.w * aspect, v.y / v.w];
                        let d = ((cur[0] - ndc[0]).powi(2) + (cur[1] - ndc[1]).powi(2)).sqrt();
                        if d < 0.045 && best.map_or(true, |(_, _, bd)| d < bd) {
                            best = Some((GizmoTransformMode::Scale, axis, d));
                        }
                    }
                }
            }

            // 2. Rotate handles: rings at 0.80 * gizmo_len
            if self.tool_state.gizmo.mode == GizmoTransformMode::All || self.tool_state.gizmo.mode == GizmoTransformMode::Rotate {
                let ring_r = gizmo_len * 0.80;
                let segments = 32;
                for axis in axes {
                    let up = if axis.dir().y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
                    let u = axis.dir().cross(up).normalize();
                    let v_axis = axis.dir().cross(u).normalize();
                    let mut prev_ndc: Option<[f32; 2]> = None;
                    for i in 0..=segments {
                        let t = (i as f32 / segments as f32) * std::f32::consts::TAU;
                        let p = pivot + u * (ring_r * t.cos()) + v_axis * (ring_r * t.sin());
                        let v = vp * glam::Vec4::new(p.x, p.y, p.z, 1.0);
                        if v.w > 0.05 {
                            let pt_ndc = [v.x / v.w * aspect, v.y / v.w];
                            if let Some(p0) = prev_ndc {
                                let d = point_to_segment_dist(cur, p0, pt_ndc);
                                if d < 0.038 && best.map_or(true, |(_, _, bd)| d < bd) {
                                    best = Some((GizmoTransformMode::Rotate, axis, d));
                                }
                            }
                            prev_ndc = Some(pt_ndc);
                        } else {
                            prev_ndc = None;
                        }
                    }
                }
            }

            // 3. Move handles: arrows from 0.80 * gizmo_len to 1.25 * gizmo_len
            if self.tool_state.gizmo.mode == GizmoTransformMode::All || self.tool_state.gizmo.mode == GizmoTransformMode::Move {
                for axis in axes {
                    let p0 = pivot + axis.dir() * (gizmo_len * 0.80);
                    let p1 = pivot + axis.dir() * (gizmo_len * 1.25);
                    let v0 = vp * glam::Vec4::new(p0.x, p0.y, p0.z, 1.0);
                    let v1 = vp * glam::Vec4::new(p1.x, p1.y, p1.z, 1.0);
                    if v0.w > 0.05 && v1.w > 0.05 {
                        let a_ndc = [v0.x / v0.w * aspect, v0.y / v0.w];
                        let b_ndc = [v1.x / v1.w * aspect, v1.y / v1.w];
                        let d = point_to_segment_dist(cur, a_ndc, b_ndc);
                        if d < 0.040 && best.map_or(true, |(_, _, bd)| d < bd) {
                            best = Some((GizmoTransformMode::Move, axis, d));
                        }
                    }
                }
            }

            let new_hover = best.map(|(m, a, _)| (m, a));
            if new_hover != self.tool_state.gizmo.hover_handle {
                self.tool_state.gizmo.hover_handle = new_hover;
                self.ui_dirty = true;
            }
        } else if !self.tool_state.selection.is_floating {
            if self.tool_state.gizmo.hover_handle.is_some() {
                self.tool_state.gizmo.hover_handle = None;
                self.ui_dirty = true;
            }
        }

        // --- Apply drag based on selected handle type ---
        if let Some((mode, axis)) = self.tool_state.gizmo.dragging_handle {
            let pivot = self.tool_state.gizmo.drag_start_world;
            let cam_dir = (pivot - self.camera.position).normalize();
            let axis_dir = axis.dir();
            let plane_n = match mode {
                GizmoTransformMode::Rotate => axis_dir,
                _ => {
                    let pn = axis_dir.cross(cam_dir).cross(axis_dir);
                    if pn.length_squared() > 1e-6 { pn.normalize() } else { cam_dir }
                }
            };

            let (ray_o0, ray_d0) = self.camera.ray_from_ndc(
                self.tool_state.gizmo.drag_start_cursor[0],
                self.tool_state.gizmo.drag_start_cursor[1],
                aspect,
            );
            let (ray_o1, ray_d1) = self.camera.ray_from_ndc(self.cursor_pos[0], self.cursor_pos[1], aspect);
            let denom0 = plane_n.dot(ray_d0);
            let denom1 = plane_n.dot(ray_d1);

            if denom0.abs() > 1e-6 && denom1.abs() > 1e-6 {
                let t0 = (pivot - ray_o0).dot(plane_n) / denom0;
                let t1 = (pivot - ray_o1).dot(plane_n) / denom1;
                let p0 = ray_o0 + ray_d0 * t0;
                let p1 = ray_o1 + ray_d1 * t1;
                let world_delta = p1 - p0;

                let snap = self.input.ctrl_pressed;
                let s = self.edit_size;

                match mode {
                    GizmoTransformMode::Move => {
                        let mut delta_along = world_delta.dot(axis_dir);
                        if snap { delta_along = (delta_along / s).round() * s; }
                        let move_vec = axis_dir * delta_along;
                        self.tool_state.gizmo.accumulated_move = move_vec;
                        self.tool_state.selection.floating_voxels = self.tool_state.gizmo.snapshot
                            .iter()
                            .map(|&(rel, sz, m)| (rel + move_vec, sz, m))
                            .collect();
                    }
                    GizmoTransformMode::Rotate => {
                        let v0 = (p0 - pivot) - axis_dir * (p0 - pivot).dot(axis_dir);
                        let v1 = (p1 - pivot) - axis_dir * (p1 - pivot).dot(axis_dir);
                        let angle = if v0.length_squared() > 1e-9 && v1.length_squared() > 1e-9 {
                            let a = v0.normalize();
                            let b = v1.normalize();
                            let cos_t = a.dot(b).clamp(-1.0, 1.0);
                            let sin_t = axis_dir.dot(a.cross(b));
                            sin_t.atan2(cos_t)
                        } else { 0.0 };
                        let mut angle = angle;
                        if snap {
                            let step = std::f32::consts::PI / 12.0; // 15°
                            angle = (angle / step).round() * step;
                        }
                        self.tool_state.gizmo.accumulated_angle = angle;
                        let (sin_a, cos_a) = angle.sin_cos();
                        self.tool_state.selection.floating_voxels = self.tool_state.gizmo.snapshot
                            .iter()
                            .map(|&(rel, sz, m)| {
                                let r = rel;
                                let k = axis_dir;
                                let rotated = r * cos_a + k.cross(r) * sin_a + k * k.dot(r) * (1.0 - cos_a);
                                let snapped = Vec3::new(
                                    (rotated.x / s).round() * s,
                                    (rotated.y / s).round() * s,
                                    (rotated.z / s).round() * s,
                                );
                                (snapped, sz, m)
                            })
                            .collect();
                    }
                    GizmoTransformMode::Scale => {
                        let d0 = (p0 - pivot).dot(axis_dir);
                        let d1 = (p1 - pivot).dot(axis_dir);
                        let mut factor = if d0.abs() > 1e-4 { (d1 / d0).abs() } else { 1.0 };
                        if factor < 0.1 { factor = 0.1; }
                        if snap {
                            factor = (factor / 0.5).round() * 0.5;
                            if factor < 0.5 { factor = 0.5; }
                        }
                        self.tool_state.gizmo.accumulated_scale = factor;
                        self.tool_state.selection.floating_voxels = self.tool_state.gizmo.snapshot
                            .iter()
                            .map(|&(rel, sz, m)| {
                                let scaled = rel * factor;
                                let snapped = Vec3::new(
                                    (scaled.x / s).round() * s,
                                    (scaled.y / s).round() * s,
                                    (scaled.z / s).round() * s,
                                );
                                (snapped, sz, m)
                            })
                            .collect();
                    }
                    GizmoTransformMode::All => {}
                }
                self.ui_dirty = true;
            }
        }

        if self.input.action_pick {
            if let Some(ref h) = hit {
                let pal = self.palette.read().unwrap();
                if let Some(&color) = pal.get((h.material - 1) as usize) {
                    self.hotbar_colors[self.selected_slot] = color;
                    drop(pal);
                    self.ui_dirty = true;
                }
            }
            self.input.action_pick = false;
        } else if self.input.action_add {
            self.execute_tool_action(false);
            self.input.action_add = false;
        } else if self.input.action_remove {
            self.execute_tool_action(true);
            self.input.action_remove = false;
        }
        if self.ui_dirty { self.update_ui(); }
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
                            r: self.bg_color[0] as f64,
                            g: self.bg_color[1] as f64,
                            b: self.bg_color[2] as f64,
                            a: 1.0,
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

struct App {
    state: Option<State>,
    last_frame: Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            #[allow(unused_mut)]
            let mut window_attributes = Window::default_attributes()
                .with_title("Voxel Studio - SVO Ray-Marching Engine")
                .with_inner_size(LogicalSize::new(1280.0, 720.0))
                .with_visible(true);
            #[cfg(target_os = "linux")]
            {
                window_attributes = WindowAttributesExtWayland::with_name(window_attributes, "octree_voxels", "octree_voxels");
                window_attributes = WindowAttributesExtX11::with_name(window_attributes, "octree_voxels", "octree_voxels");
            }
            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
            let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
            window.set_cursor_visible(true);
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
                    let mx = (position.x as f32 / state.size.width as f32) * 2.0 - 1.0;
                    let my = 1.0 - (position.y as f32 / state.size.height as f32) * 2.0;
                    state.cursor_pos = [mx, my];
                    if state.mmb_dragging {
                        let dx = mx - state.prev_cursor_pos[0];
                        let dy = my - state.prev_cursor_pos[1];
                        if dx.abs() > 0.0001 || dy.abs() > 0.0001 {
                            if state.input.shift_pressed {
                                let right = state.camera.right();
                                let up = state.camera.up();
                                let pan_dist = if state.camera.is_ortho {
                                    state.camera.ortho_size * 0.5
                                } else {
                                    (state.camera.position - state.orbit_pivot).length().max(2.0) * 0.5
                                };
                                let pan = (-right * dx - up * dy) * pan_dist;
                                state.camera.position += pan;
                                state.orbit_pivot += pan;
                            } else if state.input.ctrl_pressed {
                                let zoom_delta = dy * if state.camera.is_ortho {
                                    state.camera.ortho_size * 0.8
                                } else {
                                    (state.camera.position - state.orbit_pivot).length().max(2.0) * 0.8
                                };
                                if state.camera.is_ortho {
                                    state.camera.ortho_size = (state.camera.ortho_size - zoom_delta).clamp(0.5, 50000.0);
                                } else {
                                    let dir = state.camera.forward();
                                    let cur_dist = (state.camera.position - state.orbit_pivot).length();
                                    let new_dist = (cur_dist - zoom_delta).clamp(1.0, 50000.0);
                                    state.camera.position = state.orbit_pivot - dir * new_dist;
                                }
                            } else {
                                state.camera.yaw += dx * 3.5;
                                state.camera.pitch = (state.camera.pitch + dy * 3.5).clamp(-1.56, 1.56);
                                state.update_orbit_position();
                            }
                            state.update_camera_buffer();
                            state.ui_dirty = true;
                            state.window.request_redraw();
                        }
                    } else if state.gimbal_dragging {
                        let dx = mx - state.prev_cursor_pos[0];
                        let dy = my - state.prev_cursor_pos[1];
                        if dx.abs() > 0.0005 || dy.abs() > 0.0005 {
                            state.gimbal_drag_moved = true;
                            state.camera.yaw += dx * 3.8;
                            state.camera.pitch = (state.camera.pitch + dy * 3.8).clamp(-1.56, 1.56);
                            state.update_orbit_position();
                            state.ui_dirty = true;
                            state.window.request_redraw();
                        }
                    } else if state.active_menu == ActiveMenu::Edit {
                        if let Some(channel) = state.active_slider {
                            state.hotbar_colors[state.selected_slot][channel] = ((mx - -0.32) / (0.18 - -0.32)).clamp(0.0, 1.0);
                            state.ui_dirty = true;
                            state.window.request_redraw();
                        }
                    } else if state.active_menu == ActiveMenu::BgColorModal {
                        if let Some(channel) = state.active_slider {
                            state.bg_color[channel] = ((mx - -0.34) / (0.08 - -0.34)).clamp(0.0, 1.0);
                            state.update_camera_buffer();
                            state.ui_dirty = true;
                            state.window.request_redraw();
                        }
                    } else if state.active_menu == ActiveMenu::None {
                        let mut hovered = None;
                        for (i, &tool) in ALL_TOOLS.iter().enumerate() {
                            let (tx0, ty0, tx1, ty1) = get_left_tool_btn_bounds(i);
                            if mx >= tx0 && mx <= tx1 && my >= ty0 && my <= ty1 {
                                hovered = Some(tool);
                                break;
                            }
                        }
                        if state.tool_state.hovered_tool != hovered {
                            state.tool_state.hovered_tool = hovered;
                            state.ui_dirty = true;
                        }
                        state.window.request_redraw();
                    }
                    state.prev_cursor_pos = [mx, my];
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let step = match delta {
                        MouseScrollDelta::LineDelta(_, y) => if y > 0.0 { -1 } else if y < 0.0 { 1 } else { 0 },
                        MouseScrollDelta::PixelDelta(pos) => if pos.y > 0.0 { -1 } else if pos.y < 0.0 { 1 } else { 0 },
                    };
                    if step != 0 {
                        if state.camera.is_ortho {
                            state.camera.ortho_size = (state.camera.ortho_size * if step < 0 { 0.88 } else { 1.14 }).clamp(0.5, 50000.0);
                        } else {
                            let cur_dist = (state.camera.position - state.orbit_pivot).length().max(1.0);
                            let new_dist = (cur_dist + (if step < 0 { -2.0 } else { 2.0 })).clamp(1.0, 50000.0);
                            state.camera.position = state.orbit_pivot - state.camera.forward() * new_dist;
                        }
                        state.update_camera_buffer();
                        state.ui_dirty = true;
                        state.window.request_redraw();
                    }
                }
                WindowEvent::KeyboardInput { event: key_event, .. } => {
                    let is_pressed = key_event.state == ElementState::Pressed;
                    if key_event.physical_key == PhysicalKey::Code(KeyCode::ControlLeft) || key_event.physical_key == PhysicalKey::Code(KeyCode::ControlRight) {
                        state.input.ctrl_pressed = is_pressed;
                    }
                    if key_event.physical_key == PhysicalKey::Code(KeyCode::ShiftLeft) || key_event.physical_key == PhysicalKey::Code(KeyCode::ShiftRight) {
                        state.input.shift_pressed = is_pressed;
                    }
                    if state.input.ctrl_pressed && is_pressed {
                        let is_c = match &key_event.logical_key {
                            winit::keyboard::Key::Character(s) => s.eq_ignore_ascii_case("c"),
                            _ => key_event.physical_key == PhysicalKey::Code(KeyCode::KeyC),
                        };
                        let is_v = match &key_event.logical_key {
                            winit::keyboard::Key::Character(s) => s.eq_ignore_ascii_case("v"),
                            _ => key_event.physical_key == PhysicalKey::Code(KeyCode::KeyV),
                        };
                        let is_z = match &key_event.logical_key {
                            winit::keyboard::Key::Character(s) => s.eq_ignore_ascii_case("z"),
                            _ => key_event.physical_key == PhysicalKey::Code(KeyCode::KeyZ),
                        };
                        let is_y = match &key_event.logical_key {
                            winit::keyboard::Key::Character(s) => s.eq_ignore_ascii_case("y"),
                            _ => key_event.physical_key == PhysicalKey::Code(KeyCode::KeyY),
                        };
                        if is_c { state.copy_selection(); return; }
                        if is_v { state.paste_clipboard(); return; }
                        if is_z {
                            if state.input.shift_pressed {
                                if let Some(action) = state.history.redo_stack.pop_back() {
                                    for &(p, sz, _) in &action.removed { state.octree.insert_cube_world(p, sz, 0, false); }
                                    for &(p, sz, m) in &action.added { state.octree.insert_cube_world(p, sz, m, false); }
                                    state.octree.collapse(state.octree.root_index as usize);
                                    state.octree.recalculate_voxel_count();
                                    state.history.undo_stack.push_back(action);
                                    state.sync_svo_buffer_full();
                                    state.ui_dirty = true;
                                    state.window.request_redraw();
                                }
                            } else if let Some(action) = state.history.undo_stack.pop_back() {
                                for &(p, sz, _) in &action.added { state.octree.insert_cube_world(p, sz, 0, false); }
                                for &(p, sz, m) in &action.removed { state.octree.insert_cube_world(p, sz, m, false); }
                                state.octree.collapse(state.octree.root_index as usize);
                                state.octree.recalculate_voxel_count();
                                state.history.redo_stack.push_back(action);
                                state.sync_svo_buffer_full();
                                state.ui_dirty = true;
                                state.window.request_redraw();
                            }
                            return;
                        }
                        if is_y {
                            if let Some(action) = state.history.redo_stack.pop_back() {
                                for &(p, sz, _) in &action.removed { state.octree.insert_cube_world(p, sz, 0, false); }
                                for &(p, sz, m) in &action.added { state.octree.insert_cube_world(p, sz, m, false); }
                                state.octree.collapse(state.octree.root_index as usize);
                                state.octree.recalculate_voxel_count();
                                state.history.undo_stack.push_back(action);
                                state.sync_svo_buffer_full();
                                state.ui_dirty = true;
                                state.window.request_redraw();
                            }
                            return;
                        }
                    }
                    if is_pressed {
                        let step_angle = std::f32::consts::PI / 12.0;
                        match key_event.physical_key {
                            PhysicalKey::Code(KeyCode::Numpad4) => { state.camera.yaw += step_angle; state.update_orbit_position(); state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::Numpad6) => { state.camera.yaw -= step_angle; state.update_orbit_position(); state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::Numpad8) => { state.camera.pitch = (state.camera.pitch - step_angle).clamp(-1.56, 1.56); state.update_orbit_position(); state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::Numpad2) => { state.camera.pitch = (state.camera.pitch + step_angle).clamp(-1.56, 1.56); state.update_orbit_position(); state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::Numpad9) => { state.camera.yaw += std::f32::consts::PI; state.camera.pitch = -state.camera.pitch; state.update_orbit_position(); state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::Numpad1) => { state.camera.yaw = if state.input.ctrl_pressed { std::f32::consts::FRAC_PI_2 } else { -std::f32::consts::FRAC_PI_2 }; state.camera.pitch = 0.0; state.update_orbit_position(); state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::Numpad3) => { state.camera.yaw = if state.input.ctrl_pressed { 0.0 } else { std::f32::consts::PI }; state.camera.pitch = 0.0; state.update_orbit_position(); state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::Numpad7) => { state.camera.yaw = -std::f32::consts::FRAC_PI_2; state.camera.pitch = if state.input.ctrl_pressed { 1.56 } else { -1.56 }; state.update_orbit_position(); state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::Numpad5) | PhysicalKey::Code(KeyCode::KeyP) => { state.toggle_projection(); return; }
                            PhysicalKey::Code(KeyCode::NumpadDecimal) | PhysicalKey::Code(KeyCode::Period) => { state.focus_on_scene(); return; }
                            PhysicalKey::Code(KeyCode::NumpadAdd) | PhysicalKey::Code(KeyCode::Equal) => {
                                if state.camera.is_ortho { state.camera.ortho_size = (state.camera.ortho_size * 0.85).clamp(0.5, 50000.0); }
                                else {
                                    let cur_dist = (state.camera.position - state.orbit_pivot).length().max(1.0);
                                    let new_dist = (cur_dist * 0.85).clamp(1.0, 50000.0);
                                    state.camera.position = state.orbit_pivot - state.camera.forward() * new_dist;
                                }
                                state.update_camera_buffer(); state.ui_dirty = true; state.window.request_redraw(); return;
                            }
                            PhysicalKey::Code(KeyCode::NumpadSubtract) | PhysicalKey::Code(KeyCode::Minus) => {
                                if state.camera.is_ortho { state.camera.ortho_size = (state.camera.ortho_size * 1.15).clamp(0.5, 50000.0); }
                                else {
                                    let cur_dist = (state.camera.position - state.orbit_pivot).length().max(1.0);
                                    let new_dist = (cur_dist * 1.15).clamp(1.0, 50000.0);
                                    state.camera.position = state.orbit_pivot - state.camera.forward() * new_dist;
                                }
                                state.update_camera_buffer(); state.ui_dirty = true; state.window.request_redraw(); return;
                            }
                            _ => {}
                        }
                        match key_event.physical_key {
                            PhysicalKey::Code(KeyCode::KeyS) => { state.tool_state.active_tool = ToolType::Select; state.ui_dirty = true; state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::KeyG) if !state.input.ctrl_pressed => { state.grab_selection(); return; }
                            PhysicalKey::Code(KeyCode::KeyR) if !state.input.ctrl_pressed && state.active_menu == ActiveMenu::None => {
                                if state.input.shift_pressed {
                                    state.rotate_floating_or_selection();
                                } else {
                                    state.scale_voxel_size(true);
                                }
                                state.ui_dirty = true;
                                state.window.request_redraw();
                                return;
                            }
                            PhysicalKey::Code(KeyCode::KeyW) if state.active_menu == ActiveMenu::None => {
                                state.tool_state.gizmo.mode = state.tool_state.gizmo.mode.next();
                                state.tool_state.gizmo.dragging_handle = None;
                                state.tool_state.gizmo.accumulated_move = Vec3::ZERO;
                                state.tool_state.gizmo.accumulated_angle = 0.0;
                                state.tool_state.gizmo.accumulated_scale = 1.0;
                                state.ui_dirty = true;
                                state.window.request_redraw();
                                return;
                            }
                            PhysicalKey::Code(KeyCode::Delete) | PhysicalKey::Code(KeyCode::Backspace) => { state.clear_selection_voxels(); return; }
                            _ => {}
                        }
                        match key_event.physical_key {
                            PhysicalKey::Code(KeyCode::KeyV) => { state.tool_state.active_tool = ToolType::Pencil; state.tool_state.pending_anchor = None; state.ui_dirty = true; state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::KeyO) => { state.tool_state.active_tool = ToolType::Sphere; state.tool_state.pending_anchor = None; state.ui_dirty = true; state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::KeyY) => { state.tool_state.active_tool = ToolType::Cylinder; state.tool_state.pending_anchor = None; state.ui_dirty = true; state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::KeyU) => { state.tool_state.active_tool = ToolType::Disc; state.tool_state.pending_anchor = None; state.ui_dirty = true; state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::KeyB) => { state.tool_state.active_tool = ToolType::Box; state.tool_state.pending_anchor = None; state.ui_dirty = true; state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::KeyL) => { state.tool_state.active_tool = ToolType::Line; state.tool_state.pending_anchor = None; state.ui_dirty = true; state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::KeyJ) => { state.tool_state.active_tool = ToolType::Cone; state.tool_state.pending_anchor = None; state.ui_dirty = true; state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::KeyN) => { state.tool_state.active_tool = ToolType::Pyramid; state.tool_state.pending_anchor = None; state.ui_dirty = true; state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::KeyT) => { state.tool_state.active_tool = ToolType::Torus; state.tool_state.pending_anchor = None; state.ui_dirty = true; state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::KeyK) => { state.tool_state.active_tool = ToolType::Paint; state.tool_state.pending_anchor = None; state.ui_dirty = true; state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::KeyI) => { state.tool_state.active_tool = ToolType::Bucket; state.tool_state.pending_anchor = None; state.ui_dirty = true; state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::KeyH) => { state.tool_state.hollow = !state.tool_state.hollow; state.ui_dirty = true; state.window.request_redraw(); return; }
                            PhysicalKey::Code(KeyCode::KeyX) => {
                                state.input.wireframe_mode = !state.input.wireframe_mode;
                                state.update_camera_buffer();
                                state.ui_dirty = true;
                                state.window.request_redraw();
                                return;
                            }
                            PhysicalKey::Code(KeyCode::KeyC) if state.active_menu == ActiveMenu::None && !state.input.ctrl_pressed => {
                                state.input.action_pick = true;
                                state.window.request_redraw();
                                return;
                            }
                            PhysicalKey::Code(KeyCode::BracketLeft) => {
                                match state.tool_state.active_tool {
                                    ToolType::Sphere | ToolType::Disc | ToolType::Torus | ToolType::Replace => {
                                        state.tool_state.brush_radius = (state.tool_state.brush_radius - 1.0).max(1.0);
                                    }
                                    ToolType::Cylinder | ToolType::Cone | ToolType::Pyramid => {
                                        if state.input.shift_pressed {
                                            state.tool_state.cylinder_height = (state.tool_state.cylinder_height - 1.0).max(1.0);
                                        } else {
                                            state.tool_state.brush_radius = (state.tool_state.brush_radius - 1.0).max(1.0);
                                        }
                                    }
                                    ToolType::Line => state.tool_state.line_radius = (state.tool_state.line_radius - 0.5).max(0.0),
                                    ToolType::Bucket => state.tool_state.bucket_limit = (state.tool_state.bucket_limit / 2).max(64),
                                    ToolType::Pencil | ToolType::Paint => state.scale_voxel_size(false),
                                    _ => {}
                                }
                                state.ui_dirty = true; state.window.request_redraw(); return;
                            }
                            PhysicalKey::Code(KeyCode::BracketRight) => {
                                match state.tool_state.active_tool {
                                    ToolType::Sphere | ToolType::Disc | ToolType::Torus | ToolType::Replace => {
                                        state.tool_state.brush_radius = (state.tool_state.brush_radius + 1.0).min(32.0);
                                    }
                                    ToolType::Cylinder | ToolType::Cone | ToolType::Pyramid => {
                                        if state.input.shift_pressed {
                                            state.tool_state.cylinder_height = (state.tool_state.cylinder_height + 1.0).min(64.0);
                                        } else {
                                            state.tool_state.brush_radius = (state.tool_state.brush_radius + 1.0).min(32.0);
                                        }
                                    }
                                    ToolType::Line => state.tool_state.line_radius = (state.tool_state.line_radius + 0.5).min(16.0),
                                    ToolType::Bucket => state.tool_state.bucket_limit = (state.tool_state.bucket_limit * 2).min(8192),
                                    ToolType::Pencil | ToolType::Paint => state.scale_voxel_size(true),
                                    _ => {}
                                }
                                state.ui_dirty = true; state.window.request_redraw(); return;
                            }
                            _ => {}
                        }
                        let is_m_key = match &key_event.logical_key {
                            winit::keyboard::Key::Character(s) => s.eq_ignore_ascii_case("m"),
                            _ => key_event.physical_key == PhysicalKey::Code(KeyCode::KeyM) || key_event.physical_key == PhysicalKey::Code(KeyCode::Semicolon),
                        };
                        if is_m_key { state.toggle_play_mode(); return; }
                        if key_event.physical_key == PhysicalKey::Code(KeyCode::KeyE) {
                            if state.active_menu == ActiveMenu::Edit { state.set_menu(ActiveMenu::None); }
                            else if state.active_menu == ActiveMenu::None { state.set_menu(ActiveMenu::Edit); }
                            return;
                        }
                        if key_event.physical_key == PhysicalKey::Code(KeyCode::F1) {
                            state.hide_ui = !state.hide_ui;
                            if state.hide_ui { state.active_menu = ActiveMenu::None; }
                            state.ui_dirty = true;
                            state.window.request_redraw();
                            return;
                        }
                        if key_event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                            if state.hide_ui {
                                state.hide_ui = false;
                                state.ui_dirty = true;
                                state.window.request_redraw();
                                return;
                            }
                            if state.tool_state.gizmo.dragging_handle.is_some() {
                                state.tool_state.selection.floating_voxels = state.tool_state.gizmo.snapshot.clone();
                                state.clear_gizmo_state();
                                state.ui_dirty = true;
                                state.window.request_redraw();
                                return;
                            }
                            if state.tool_state.selection.is_floating {
                                state.tool_state.selection.is_floating = false;
                                state.clear_gizmo_state();
                                state.ui_dirty = true;
                                state.window.request_redraw();
                                return;
                            }
                            if state.tool_state.selection.anchor.is_some() {
                                state.tool_state.selection.anchor = None;
                                state.ui_dirty = true;
                                state.window.request_redraw();
                                return;
                            }
                            if state.tool_state.selection.bounds.is_some() {
                                state.tool_state.selection.bounds = None;
                                state.tool_state.selection.captured_voxels.clear();
                                state.ui_dirty = true;
                                state.window.request_redraw();
                                return;
                            }
                            if state.tool_state.pending_anchor.is_some() {
                                state.tool_state.pending_anchor = None;
                                state.ui_dirty = true;
                                state.window.request_redraw();
                                return;
                            }
                            match state.active_menu {
                                ActiveMenu::None | ActiveMenu::ImportParams | ActiveMenu::Controls => state.set_menu(ActiveMenu::Pause),
                                ActiveMenu::BgColorModal => state.set_menu(ActiveMenu::Pause),
                                ActiveMenu::Voxelizing => {},
                                _ => state.set_menu(ActiveMenu::None),
                            }
                            return;
                        }
                        match key_event.physical_key {
                            PhysicalKey::Code(KeyCode::F5) => { let _ = state.save_game("world_save.json"); },
                            PhysicalKey::Code(KeyCode::F9) => { let _ = state.load_game("world_save.json"); },
                            PhysicalKey::Code(KeyCode::KeyF) => state.scale_voxel_size(false),
                            PhysicalKey::Code(KeyCode::Digit1) => { state.selected_slot = 0; state.ui_dirty = true; state.window.request_redraw(); },
                            PhysicalKey::Code(KeyCode::Digit2) => { state.selected_slot = 1; state.ui_dirty = true; state.window.request_redraw(); },
                            PhysicalKey::Code(KeyCode::Digit3) => { state.selected_slot = 2; state.ui_dirty = true; state.window.request_redraw(); },
                            PhysicalKey::Code(KeyCode::Digit4) => { state.selected_slot = 3; state.ui_dirty = true; state.window.request_redraw(); },
                            PhysicalKey::Code(KeyCode::Digit5) => { state.selected_slot = 4; state.ui_dirty = true; state.window.request_redraw(); },
                            PhysicalKey::Code(KeyCode::Digit6) => { state.selected_slot = 5; state.ui_dirty = true; state.window.request_redraw(); },
                            PhysicalKey::Code(KeyCode::Digit7) => { state.selected_slot = 6; state.ui_dirty = true; state.window.request_redraw(); },
                            PhysicalKey::Code(KeyCode::Digit8) => { state.selected_slot = 7; state.ui_dirty = true; state.window.request_redraw(); },
                            PhysicalKey::Code(KeyCode::Digit9) => { state.selected_slot = 8; state.ui_dirty = true; state.window.request_redraw(); },
                            PhysicalKey::Code(KeyCode::Digit0) => { state.selected_slot = 9; state.ui_dirty = true; state.window.request_redraw(); },
                            _ => {}
                        }
                    }
                }
                WindowEvent::MouseInput { state: element_state, button, .. } => {
                    let [mx, my] = state.cursor_pos;
                    if button == MouseButton::Middle {
                        state.mmb_dragging = element_state == ElementState::Pressed;
                        state.window.request_redraw();
                        return;
                    }
                    if button == MouseButton::Left {
                        if element_state == ElementState::Pressed {
                            if state.tool_state.selection.is_floating
                                && state.tool_state.gizmo.hover_handle.is_some()
                                && state.active_menu == ActiveMenu::None
                            {
                                let handle = state.tool_state.gizmo.hover_handle.unwrap();
                                state.tool_state.gizmo.dragging_handle = Some(handle);
                                state.tool_state.gizmo.drag_start_cursor = [mx, my];
                                state.tool_state.gizmo.snapshot = state.tool_state.selection.floating_voxels.clone();
                                state.tool_state.gizmo.accumulated_move = Vec3::ZERO;
                                state.tool_state.gizmo.accumulated_angle = 0.0;
                                state.tool_state.gizmo.accumulated_scale = 1.0;
                                if let Some(target) = state.last_target {
                                    state.tool_state.gizmo.drag_start_world = Vec3::from(target);
                                }
                                state.window.request_redraw();
                                return;
                            }

                            let (bx0, by0, bx1, by1) = get_focus_button_bounds(aspect);
                            if mx >= bx0 && mx <= bx1 && my >= by0 && my <= by1 { state.focus_on_scene(); state.window.request_redraw(); return; }
                            let (px0, py0, px1, py1) = get_proj_button_bounds(aspect);
                            if mx >= px0 && mx <= px1 && my >= py0 && my <= py1 { state.toggle_projection(); state.window.request_redraw(); return; }
                            let (rx0, ry0, rx1, ry1) = get_render_button_bounds(aspect);
                            if mx >= rx0 && mx <= rx1 && my >= ry0 && my <= ry1 {
                                state.hide_ui = true;
                                state.active_menu = ActiveMenu::None;
                                state.ui_dirty = true;
                                state.window.request_redraw();
                                return;
                            }
                            if ((mx - GIZMO_CENTER_X) * aspect).powi(2) + (my - GIZMO_CENTER_Y).powi(2) <= (GIZMO_RADIUS + 0.02).powi(2) {
                                state.gimbal_dragging = true;
                                state.gimbal_drag_moved = false;
                                return;
                            }
                            let num_slots = 10;
                            let slot_w = 0.054;
                            let slot_gap = 0.009;
                            let total_w = num_slots as f32 * slot_w + (num_slots - 1) as f32 * slot_gap;
                            let start_x = -total_w / 2.0;
                            if my <= -0.84 && mx >= start_x - 0.02 && mx <= (start_x + total_w + 0.02) {
                                for i in 0..10 {
                                    let x0 = start_x + i as f32 * (slot_w + slot_gap);
                                    let x1 = x0 + slot_w;
                                    if mx >= x0 && mx <= x1 {
                                        state.selected_slot = i;
                                        state.ui_dirty = true;
                                        state.window.request_redraw();
                                        return;
                                    }
                                }
                                return;
                            }

                            if state.active_menu == ActiveMenu::None {
                                // 1. Outils cliquables
                                for (i, &tool) in ALL_TOOLS.iter().enumerate() {
                                    let (tx0, ty0, tx1, ty1) = get_left_tool_btn_bounds(i);
                                    if mx >= tx0 && mx <= tx1 && my >= ty0 && my <= ty1 {
                                        state.tool_state.active_tool = tool;
                                        state.tool_state.pending_anchor = None;
                                        if let Some(target) = state.last_target {
                                            state.preview_deltas = state.compute_preview_deltas(Vec3::from(target));
                                        }
                                        state.ui_dirty = true;
                                        state.window.request_redraw();
                                        return;
                                    }
                                }

                                // 2. Paramètres contextuels dynamiques
                                let is_minus = |x0: f32, x1: f32| mx >= x0 && mx <= x0 + 0.032;
                                let is_plus  = |x0: f32, x1: f32| mx >= x1 - 0.032 && mx <= x1;
                                let in_slot  = |x0: f32, y0: f32, x1: f32, y1: f32| mx >= x0 && mx <= x1 && my >= y0 && my <= y1;

                                let (s0_x0, s0_y0, s0_x1, s0_y1) = get_left_param_slot_bounds(0);
                                let (s1_x0, s1_y0, s1_x1, s1_y1) = get_left_param_slot_bounds(1);
                                let (s2_x0, s2_y0, s2_x1, s2_y1) = get_left_param_slot_bounds(2);

                                match state.tool_state.active_tool {
                                    ToolType::Cylinder | ToolType::Cone | ToolType::Pyramid => {
                                        if in_slot(s0_x0, s0_y0, s0_x1, s0_y1) {
                                            if is_minus(s0_x0, s0_x1) { state.tool_state.brush_radius = (state.tool_state.brush_radius - 1.0).max(1.0); }
                                            else if is_plus(s0_x0, s0_x1) { state.tool_state.brush_radius = (state.tool_state.brush_radius + 1.0).min(32.0); }
                                            state.ui_dirty = true; state.window.request_redraw(); return;
                                        }
                                        if in_slot(s1_x0, s1_y0, s1_x1, s1_y1) {
                                            if is_minus(s1_x0, s1_x1) { state.tool_state.cylinder_height = (state.tool_state.cylinder_height - 1.0).max(1.0); }
                                            else if is_plus(s1_x0, s1_x1) { state.tool_state.cylinder_height = (state.tool_state.cylinder_height + 1.0).min(64.0); }
                                            state.ui_dirty = true; state.window.request_redraw(); return;
                                        }
                                        if in_slot(s2_x0, s2_y0, s2_x1, s2_y1) {
                                            state.tool_state.hollow = !state.tool_state.hollow;
                                            state.ui_dirty = true; state.window.request_redraw(); return;
                                        }
                                    }
                                    ToolType::Sphere | ToolType::Disc | ToolType::Torus => {
                                        if in_slot(s0_x0, s0_y0, s0_x1, s0_y1) {
                                            if is_minus(s0_x0, s0_x1) { state.tool_state.brush_radius = (state.tool_state.brush_radius - 1.0).max(1.0); }
                                            else if is_plus(s0_x0, s0_x1) { state.tool_state.brush_radius = (state.tool_state.brush_radius + 1.0).min(32.0); }
                                            state.ui_dirty = true; state.window.request_redraw(); return;
                                        }
                                        if in_slot(s1_x0, s1_y0, s1_x1, s1_y1) {
                                            state.tool_state.hollow = !state.tool_state.hollow;
                                            state.ui_dirty = true; state.window.request_redraw(); return;
                                        }
                                    }
                                    ToolType::Replace => {
                                        if in_slot(s0_x0, s0_y0, s0_x1, s0_y1) {
                                            if is_minus(s0_x0, s0_x1) { state.tool_state.brush_radius = (state.tool_state.brush_radius - 1.0).max(1.0); }
                                            else if is_plus(s0_x0, s0_x1) { state.tool_state.brush_radius = (state.tool_state.brush_radius + 1.0).min(32.0); }
                                            state.ui_dirty = true; state.window.request_redraw(); return;
                                        }
                                    }
                                    ToolType::Line => {
                                        if in_slot(s0_x0, s0_y0, s0_x1, s0_y1) {
                                            if is_minus(s0_x0, s0_x1) { state.tool_state.line_radius = (state.tool_state.line_radius - 0.5).max(0.0); }
                                            else if is_plus(s0_x0, s0_x1) { state.tool_state.line_radius = (state.tool_state.line_radius + 0.5).min(16.0); }
                                            state.ui_dirty = true; state.window.request_redraw(); return;
                                        }
                                    }
                                    ToolType::Box => {
                                        if in_slot(s0_x0, s0_y0, s0_x1, s0_y1) {
                                            state.tool_state.hollow = !state.tool_state.hollow;
                                            state.ui_dirty = true; state.window.request_redraw(); return;
                                        }
                                    }
                                    ToolType::Pencil | ToolType::Paint => {
                                        if in_slot(s0_x0, s0_y0, s0_x1, s0_y1) {
                                            if is_minus(s0_x0, s0_x1) { state.scale_voxel_size(false); }
                                            else if is_plus(s0_x0, s0_x1) { state.scale_voxel_size(true); }
                                            return;
                                        }
                                    }
                                    ToolType::Bucket => {
                                        if in_slot(s0_x0, s0_y0, s0_x1, s0_y1) {
                                            if is_minus(s0_x0, s0_x1) { state.tool_state.bucket_limit = (state.tool_state.bucket_limit / 2).max(64); }
                                            else if is_plus(s0_x0, s0_x1) { state.tool_state.bucket_limit = (state.tool_state.bucket_limit * 2).min(8192); }
                                            state.ui_dirty = true; state.window.request_redraw(); return;
                                        }
                                    }
                                    ToolType::Select => {
                                        if in_slot(s0_x0, s0_y0, s0_x1, s0_y1) {
                                            state.tool_state.gizmo.mode = state.tool_state.gizmo.mode.next();
                                            state.clear_gizmo_state();
                                            state.ui_dirty = true; state.window.request_redraw(); return;
                                        }
                                        if in_slot(s1_x0, s1_y0, s1_x1, s1_y1) {
                                            if state.tool_state.selection.bounds.is_some() && !state.tool_state.selection.is_floating {
                                                state.grab_selection();
                                            }
                                            return;
                                        }
                                    }
                                }

                                // 3. Éviter tout placement de voxel accidentel lors du clic sur l'UI gauche
                                if mx >= LEFT_PALETTE_X0 - 0.01 && mx <= LEFT_PALETTE_X1 + 0.01 && my <= LEFT_PALETTE_TOP_Y + 0.01 {
                                    return;
                                }
                            }
                        } else if element_state == ElementState::Released {
                            if state.gimbal_dragging {
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
                                            state.update_orbit_position();
                                            state.window.request_redraw();
                                            return;
                                        }
                                    }
                                }
                                return;
                            }
                            if state.tool_state.gizmo.dragging_handle.is_some() {
                                state.clear_gizmo_state();
                                state.ui_dirty = true;
                                state.window.request_redraw();
                                return;
                            }
                        }
                    }
                    match state.active_menu {
                        ActiveMenu::Edit => {
                            if button == MouseButton::Left && element_state == ElementState::Pressed {
                                let sw_w = 0.082; let sw_gap = 0.015; let sw_tot = 10.0 * sw_w + 9.0 * sw_gap; let s_start_x = -sw_tot / 2.0;
                                for i in 0..10 {
                                    if mx >= s_start_x + i as f32 * (sw_w + sw_gap) && mx <= s_start_x + i as f32 * (sw_w + sw_gap) + sw_w && my >= 0.43 && my <= 0.52 {
                                        state.selected_slot = i; state.ui_dirty = true; state.window.request_redraw(); return;
                                    }
                                }
                                if mx >= -0.34 && mx <= 0.20 {
                                    state.active_slider = if my >= 0.30 && my <= 0.38 { Some(0) } else if my >= 0.23 && my <= 0.31 { Some(1) } else if my >= 0.16 && my <= 0.24 { Some(2) } else { None };
                                    if let Some(channel) = state.active_slider {
                                        state.hotbar_colors[state.selected_slot][channel] = ((mx - -0.32) / (0.18 - -0.32)).clamp(0.0, 1.0);
                                        state.ui_dirty = true; state.window.request_redraw(); return;
                                    }
                                }
                                let pw_w = 0.076; let pw_gap = 0.012; let pw_tot = 10.0 * pw_w + 9.0 * pw_gap; let pw_start_x = -pw_tot / 2.0;
                                for (i, &preset_col) in PRESET_SWATCHES.iter().enumerate() {
                                    if mx >= pw_start_x + i as f32 * (pw_w + pw_gap) && mx <= pw_start_x + i as f32 * (pw_w + pw_gap) + pw_w && my >= 0.05 && my <= 0.11 {
                                        state.hotbar_colors[state.selected_slot] = preset_col; state.ui_dirty = true; state.window.request_redraw(); return;
                                    }
                                }
                                if mx >= -0.40 && mx <= -0.22 && my >= -0.10 && my <= -0.02 { state.scale_voxel_size(false); return; }
                                if mx >= 0.22 && mx <= 0.40 && my >= -0.10 && my <= -0.02 { state.scale_voxel_size(true); return; }
                            } else if button == MouseButton::Left {
                                state.active_slider = None;
                            }
                        }
                        ActiveMenu::Pause => {
                            if button == MouseButton::Left && element_state == ElementState::Pressed {
                                if mx >= -0.30 && mx <= 0.30 && my >= 0.48 && my <= 0.56 { state.prompt_native_file_dialog(); return; }
                                if mx >= -0.30 && mx <= 0.30 && my >= 0.38 && my <= 0.46 { state.toggle_play_mode(); return; }
                                if mx >= -0.30 && mx <= 0.30 && my >= 0.28 && my <= 0.36 { state.toggle_projection(); return; }
                                if mx >= -0.30 && mx <= 0.30 && my >= 0.18 && my <= 0.26 { state.cycle_world_generator(); return; }
                                if mx >= -0.30 && mx <= 0.30 && my >= 0.08 && my <= 0.16 { state.clear_all_blocks(); return; }
                                if mx >= -0.30 && mx <= -0.02 && my >= -0.02 && my <= 0.06 { let _ = state.save_game("world_save.json"); return; }
                                if mx >= 0.02 && mx <= 0.30 && my >= -0.02 && my <= 0.06 { let _ = state.load_game("world_save.json"); return; }
                                if mx >= -0.30 && mx <= 0.30 && my >= -0.19 && my <= -0.11 { state.set_menu(ActiveMenu::BgColorModal); return; }
                                if mx >= -0.30 && mx <= 0.30 && my >= -0.29 && my <= -0.21 { state.hide_ui = true; state.active_menu = ActiveMenu::None; state.ui_dirty = true; state.window.request_redraw(); return; }
                                if mx >= -0.30 && mx <= 0.30 && my >= -0.39 && my <= -0.31 { state.set_menu(ActiveMenu::Controls); return; }
                                if mx >= -0.30 && mx <= 0.30 && my >= -0.51 && my <= -0.43 { state.set_menu(ActiveMenu::None); return; }
                                if mx >= -0.30 && mx <= 0.30 && my >= -0.63 && my <= -0.55 { event_loop.exit(); return; }
                            }
                        }
                        ActiveMenu::BgColorModal => {
                            if button == MouseButton::Left && element_state == ElementState::Pressed {
                                if mx >= -0.36 && mx <= 0.12 {
                                    state.active_slider = if my >= 0.22 && my <= 0.30 { Some(0) }
                                    else if my >= 0.15 && my <= 0.23 { Some(1) }
                                    else if my >= 0.08 && my <= 0.16 { Some(2) }
                                    else { None };
                                    if let Some(channel) = state.active_slider {
                                        state.bg_color[channel] = ((mx - -0.34) / (0.08 - -0.34)).clamp(0.0, 1.0);
                                        state.update_camera_buffer();
                                        state.ui_dirty = true;
                                        state.window.request_redraw();
                                        return;
                                    }
                                }
                                let bg_w = 0.068; let bg_gap = 0.008; let bg_tot = 8.0 * bg_w + 7.0 * bg_gap; let bg_start_x = -bg_tot / 2.0;
                                let bg_y0 = -0.10; let bg_y1 = -0.04;
                                if my >= bg_y0 && my <= bg_y1 {
                                    for (i, &col) in PRESET_BG_COLORS.iter().enumerate() {
                                        let x0 = bg_start_x + i as f32 * (bg_w + bg_gap);
                                        let x1 = x0 + bg_w;
                                        if mx >= x0 && mx <= x1 {
                                            state.bg_color = col;
                                            state.update_camera_buffer();
                                            state.ui_dirty = true;
                                            state.window.request_redraw();
                                            return;
                                        }
                                    }
                                }
                                if mx >= -0.22 && mx <= 0.22 && my >= -0.23 && my <= -0.15 {
                                    state.set_menu(ActiveMenu::Pause);
                                    return;
                                }
                            } else if button == MouseButton::Left {
                                state.active_slider = None;
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
                                    if mx >= wx0 + 0.04 && mx <= wx0 + 0.14 { state.glb_settings.target_height = (state.glb_settings.target_height - 4.0).max(4.0); state.ui_dirty = true; state.window.request_redraw(); return; }
                                    if mx >= wx1 - 0.14 && mx <= wx1 - 0.04 { state.glb_settings.target_height = (state.glb_settings.target_height + 4.0).min(128.0); state.ui_dirty = true; state.window.request_redraw(); return; }
                                }
                                if my >= -0.01 && my <= 0.07 {
                                    if mx >= wx0 + 0.04 && mx <= wx0 + 0.14 { state.glb_settings.voxel_size = (state.glb_settings.voxel_size * 0.5).max(MIN_VOXEL_SIZE); state.ui_dirty = true; state.window.request_redraw(); return; }
                                    if mx >= wx1 - 0.14 && mx <= wx1 - 0.04 { state.glb_settings.voxel_size *= 2.0; state.ui_dirty = true; state.window.request_redraw(); return; }
                                }
                                if my >= -0.21 && my <= -0.13 {
                                    if mx >= wx0 + 0.04 && mx <= wx0 + 0.14 { state.glb_settings.palette_size = (state.glb_settings.palette_size / 2).max(2); state.ui_dirty = true; state.window.request_redraw(); return; }
                                    if mx >= wx1 - 0.14 && mx <= wx1 - 0.04 { state.glb_settings.palette_size = (state.glb_settings.palette_size * 2).min(8192); state.ui_dirty = true; state.window.request_redraw(); return; }
                                }
                                if mx >= wx0 + 0.04 && mx <= wx1 - 0.04 && my >= -0.39 && my <= -0.31 {
                                    state.glb_settings.place_at_aim = !state.glb_settings.place_at_aim; state.ui_dirty = true; state.window.request_redraw(); return; }
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
                            if element_state == ElementState::Pressed && !state.mmb_dragging && !state.hide_ui && state.tool_state.gizmo.dragging_handle.is_none() {
                                match button {
                                    MouseButton::Left => state.input.action_add = true,
                                    MouseButton::Right => state.input.action_remove = true,
                                    _ => {}
                                }
                                state.window.request_redraw();
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
                    if state.voxelize_rx.is_some() || state.mmb_dragging || state.gimbal_dragging
                        || state.tool_state.gizmo.dragging_handle.is_some() || state.error_banner.is_some()
                    {
                        state.window.request_redraw();
                    }
                }
                _ => {}
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {}
}

pub fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App { state: None, last_frame: Instant::now() };
    event_loop.run_app(&mut app).unwrap();
}