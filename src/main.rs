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
use std::sync::Arc;
use std::time::Instant;
use std::collections::{HashMap, HashSet};
use std::sync::mpsc;
use glam::{Vec3, Mat4};
use block_mesh::{greedy_quads, GreedyQuadsBuffer, Voxel, VoxelVisibility, MergeVoxel, RIGHT_HANDED_Y_UP_CONFIG};
use ndshape::{ConstShape, ConstShape3u32};
use noise::{NoiseFn, Fbm, Perlin};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
// Ajout du padding pour le chunk (32 + 2 = 34) afin d'éviter le "wrapping" des index aux bordures
type ChunkShape = ConstShape3u32<34, 34, 34>; 
type ChunkPos = (i32, i32, i32);

// --- OCTREE & VOXELS ---

#[derive(Clone, Copy, Default)]
pub struct OctreeNode {
    pub child_mask: u8,
    pub material_id: u16,
    pub child_pointer: u32,
}

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
        let mut half_size = 1 << depth;
        for _ in 0..depth {
            half_size >>= 1;
            let mut octant = 0;
            if x >= half_size { octant |= 1; x -= half_size; }
            if y >= half_size { octant |= 2; y -= half_size; }
            if z >= half_size { octant |= 4; z -= half_size; }

            if self.nodes[current_idx].child_pointer == 0 {
                let new_ptr = self.nodes.len() as u32;
                self.nodes.resize(self.nodes.len() + 8, OctreeNode::default());
                self.nodes[current_idx].child_pointer = new_ptr;
            }
            self.nodes[current_idx].child_mask |= 1 << octant;
            current_idx = (self.nodes[current_idx].child_pointer + octant) as usize;
        }
        self.nodes[current_idx].material_id = material;
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

// --- GENERATION CHUNK ---

struct MeshPayload {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

fn generate_chunk(chunk_pos: ChunkPos) -> MeshPayload {
    let mut octree = Octree::new();
    let fbm = Fbm::<Perlin>::new(42);
    let world_x_offset = chunk_pos.0 * 32;
    let world_z_offset = chunk_pos.2 * 32;

    for x in 0..32 {
        for z in 0..32 {
            let world_x = (x as i32 + world_x_offset) as f64 * 0.05;
            let world_z = (z as i32 + world_z_offset) as f64 * 0.05;
            let noise_val = fbm.get([world_x, world_z]);
            let height = ((noise_val + 1.0) * 12.0).clamp(0.0, 31.0) as u32;

            for y in 0..=height {
                let material = if y == height { 2 } else { 1 };
                octree.insert(x, y, z, 5, material);
            }
        }
    }

    let mut voxels = vec![Block(0); ChunkShape::SIZE as usize];
    // Décalage de [1, 1, 1] pour éviter l'out-of-bounds et générer un culling valide via le padding vide
    octree.flatten_into(octree.root_index as usize, 1, 1, 1, 32, &mut voxels);

    let mut buffer = GreedyQuadsBuffer::new(voxels.len());
    // greedy_quads(min, max) attend un max INCLUSIF et applique lui-même un padding(-1)
    // interne. Le contenu réel occupe [1..33) dans le chunk 34^3, donc il faut donner
    // [0..34) pour que toute la zone de contenu soit mouchée (les faces bordures contre
    // le padding vide sont conservées, ce qui évite d'avoir besoin des chunks voisins).
    greedy_quads(&voxels, &ChunkShape {}, [0, 0, 0], [33, 33, 33], &RIGHT_HANDED_Y_UP_CONFIG.faces, &mut buffer);

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let offset_x = (chunk_pos.0 * 32) as f32;
    let offset_y = (chunk_pos.1 * 32) as f32;
    let offset_z = (chunk_pos.2 * 32) as f32;

    for (group, face) in buffer.quads.groups.iter().zip(RIGHT_HANDED_Y_UP_CONFIG.faces.into_iter()) {
        let n = face.signed_normal();
        let normal = [n.x as f32, n.y as f32, n.z as f32];

        for quad in group.into_iter() {
            let start_index = vertices.len() as u32;

            // 1. On cherche le voxel adjacent du côté positif de la frontière (+1)
            let pos1 = quad.minimum;
            let mut pos2 = quad.minimum;
            if n.x != 0 { pos2[0] += 1; }
            if n.y != 0 { pos2[1] += 1; }
            if n.z != 0 { pos2[2] += 1; }

            let id1 = voxels[ChunkShape::linearize(pos1) as usize].0;
            let id2 = voxels[ChunkShape::linearize(pos2) as usize].0;

            // Sélection du voxel solide
            let mat_id = if id1 != 0 { id1 } else { id2 };
            let color = match mat_id { 2 => [0.2, 0.7, 0.3], _ => [0.5, 0.5, 0.5] };

            for corner in face.quad_mesh_positions(quad, 1.0) {
                vertices.push(Vertex {
                    position: [
                        corner[0] - 1.0 + offset_x, 
                        corner[1] - 1.0 + offset_y, 
                        corner[2] - 1.0 + offset_z
                    ],
                    normal,
                    color,
                });
            }
            
            // 2. On conserve l'ordre natif CCW sans condition .swap()
            let quad_indices = face.quad_mesh_indices(start_index);
            indices.extend_from_slice(&quad_indices);
        }
    }
    MeshPayload { vertices, indices }
}

// --- GESTIONNAIRE DE CHUNKS ---

struct RenderChunk {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

struct ChunkManager {
    loaded_chunks: HashMap<ChunkPos, RenderChunk>,
    loading_chunks: HashSet<ChunkPos>,
    tx: mpsc::Sender<(ChunkPos, MeshPayload)>,
    rx: mpsc::Receiver<(ChunkPos, MeshPayload)>,
    render_distance: i32,
}

impl ChunkManager {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self { loaded_chunks: HashMap::new(), loading_chunks: HashSet::new(), tx, rx, render_distance: 3 }
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
                    rayon::spawn(move || {
                        let payload = generate_chunk(pos);
                        let _ = tx_clone.send((pos, payload));
                    });
                }
            }
        }

        while let Ok((pos, payload)) = self.rx.try_recv() {
            self.loading_chunks.remove(&pos);
            // wgpu panics on zero-size buffers — skip chunks with no visible geometry
            if payload.vertices.is_empty() || payload.indices.is_empty() {
                continue;
            }
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::cast_slice(&payload.vertices), usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::cast_slice(&payload.indices), usage: wgpu::BufferUsages::INDEX,
            });
            self.loaded_chunks.insert(pos, RenderChunk { vertex_buffer, index_buffer, num_indices: payload.indices.len() as u32 });
        }

        self.loaded_chunks.retain(|pos, _| {
            (pos.0 - p_x).abs() <= self.render_distance + 1 && (pos.2 - p_z).abs() <= self.render_distance + 1
        });
    }
}

// --- STRUCTURES RENDU ---

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

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
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform { view_proj: [[f32; 4]; 4] }

struct Camera { position: Vec3, yaw: f32, pitch: f32 }
impl Camera {
    fn view_proj(&self, aspect: f32) -> Mat4 {
        let (sin_p, cos_p) = self.pitch.sin_cos();
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let dir = Vec3::new(cos_y * cos_p, sin_p, sin_y * cos_p).normalize();
        let view = glam::camera::rh::view::look_at_mat4(self.position, self.position + dir, Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective((60.0_f32).to_radians(), aspect, 0.1, 500.0);
        proj * view
    }
}

#[derive(Default)]
struct InputState { forward: bool, backward: bool, left: bool, right: bool, up: bool, down: bool }

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    depth_texture_view: wgpu::TextureView,
    window: Arc<Window>,
    camera: Camera,
    input: InputState,
    chunk_manager: ChunkManager,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let mut size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            size = winit::dpi::PhysicalSize::new(1280, 720);
        }

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }).await.expect("Adaptateur compatible introuvable");

        let info = adapter.get_info();
        println!("GPU actif : {} ({:?}) via {:?}", info.name, info.device_type, info.backend);

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.unwrap();

        let mut config = surface.get_default_config(&adapter, size.width, size.height)
            .expect("La surface n'est pas supportée par cet adaptateur");

        let surface_caps = surface.get_capabilities(&adapter);
        if surface_caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            config.alpha_mode = wgpu::CompositeAlphaMode::Opaque;
        }
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);

        let camera = Camera { position: Vec3::new(16.0, 45.0, 48.0), yaw: -std::f32::consts::FRAC_PI_2, pitch: -0.4 };
        let aspect = config.width as f32 / config.height as f32;
        let camera_uniform = CameraUniform { view_proj: camera.view_proj(aspect).to_cols_array_2d() };
        
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"), contents: bytemuck::cast_slice(&[camera_uniform]), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::VERTEX, count: None,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            }], label: None,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() }], label: None,
        });

        let depth_texture_view = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, label: None, view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor::default());

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[Some(&camera_bind_group_layout)], immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None, layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[Some(Vertex::desc())] },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { 
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back), // Backface Culling correctement activé
                ..Default::default() 
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT, depth_write_enabled: Some(true), depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(), multiview_mask: None, cache: None,
        });

        let mut chunk_manager = ChunkManager::new();
        let initial_payload = generate_chunk((0, 0, 0));
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&initial_payload.vertices), usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&initial_payload.indices), usage: wgpu::BufferUsages::INDEX,
        });
        chunk_manager.loaded_chunks.insert((0, 0, 0), RenderChunk {
            vertex_buffer, index_buffer, num_indices: initial_payload.indices.len() as u32,
        });

        Self {
            window, surface, device, queue, config, size, render_pipeline,
            camera_buffer, camera_bind_group, depth_texture_view, camera,
            input: InputState::default(), chunk_manager,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            self.depth_texture_view = self.device.create_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d { width: self.config.width, height: self.config.height, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: DEPTH_FORMAT, 
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT, label: None, view_formats: &[],
            }).create_view(&wgpu::TextureViewDescriptor::default());
        }
    }

    fn update(&mut self, dt: f32) {
        let speed = 40.0 * dt;
        let (sin_y, cos_y) = self.camera.yaw.sin_cos();
        let forward = Vec3::new(cos_y, 0.0, sin_y).normalize();
        let right = Vec3::new(-sin_y, 0.0, cos_y).normalize();

        if self.input.forward { self.camera.position += forward * speed; }
        if self.input.backward { self.camera.position -= forward * speed; }
        if self.input.right { self.camera.position += right * speed; }
        if self.input.left { self.camera.position -= right * speed; }
        if self.input.up { self.camera.position.y += speed; }
        if self.input.down { self.camera.position.y -= speed; }

        let aspect = if self.config.height > 0 { self.config.width as f32 / self.config.height as f32 } else { 1.0 };
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[CameraUniform { view_proj: self.camera.view_proj(aspect).to_cols_array_2d() }]));
        
        self.chunk_manager.update(self.camera.position, &self.device);
    }

    fn render(&mut self) {
        let current_win_size = self.window.inner_size();
        if current_win_size.width > 0 && current_win_size.height > 0 {
            if current_win_size.width != self.config.width || current_win_size.height != self.config.height {
                self.resize(current_win_size);
            }
        }

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
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { 
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.4, g: 0.7, b: 0.95, a: 1.0 }),
                        store: wgpu::StoreOp::Store 
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None, 
                occlusion_query_set: None, 
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            
            for chunk in self.chunk_manager.loaded_chunks.values() {
                render_pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                render_pass.set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..chunk.num_indices, 0, 0..1);
            }
        }

        drop(view);
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
                .with_title("Voxel Octree Engine")
                .with_inner_size(LogicalSize::new(1280.0, 720.0))
                .with_visible(true);

            #[cfg(target_os = "linux")]
            {
                window_attributes = WindowAttributesExtWayland::with_name(window_attributes, "octree_voxels", "octree_voxels");
                window_attributes = WindowAttributesExtX11::with_name(window_attributes, "octree_voxels", "octree_voxels");
            }

            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
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
                WindowEvent::KeyboardInput { event: key_event, .. } => {
                    if key_event.physical_key == PhysicalKey::Code(KeyCode::Escape) { event_loop.exit(); }
                    let is_pressed = key_event.state == ElementState::Pressed;
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
                WindowEvent::Resized(size) => {
                    state.resize(size);
                    state.window.request_redraw();
                }
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
            if let DeviceEvent::MouseMotion { delta } = event {
                state.camera.yaw += (delta.0 as f32) * 0.002;
                state.camera.pitch -= (delta.1 as f32) * 0.002;
                state.camera.pitch = state.camera.pitch.clamp(-1.5, 1.5);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Vérifie que pour chaque face, le winding produit par `quad_mesh_indices`
    /// est CCW vu de l'extérieur (la normale calculée du triangle pointe dans le
    /// sens de la normale de la face), sinon la face serait éliminée par le culling.
    #[test]
    fn all_faces_are_wound_front_facing() {
        for face in RIGHT_HANDED_Y_UP_CONFIG.faces {
            let quad = block_mesh::UnorientedQuad {
                minimum: [10, 10, 10],
                width: 1,
                height: 1,
            };
            let corners = face.quad_mesh_positions(&quad, 1.0).map(|c| Vec3::from(c));
            let n = face.signed_normal();
            let normal = Vec3::new(n.x as f32, n.y as f32, n.z as f32);
            let idx = face.quad_mesh_indices(0);
            for tri in idx.chunks(3) {
                let (a, b, c) = (corners[tri[0] as usize], corners[tri[1] as usize], corners[tri[2] as usize]);
                let tri_normal = (b - a).cross(c - a);
                assert!(
                    tri_normal.dot(normal) > 0.0,
                    "Face {:?} : triangle {:?} tourné dans le mauvais sens (triangle normal = {:?})",
                    normal, tri, tri_normal
                );
            }
        }
    }

    /// Vérifie que le mesh couvre TOUT le chunk jusqu'aux bords (position 32) :
    /// avec l'ancien appel greedy_quads([1,1,1],[32,32,32]), la colonne/rangee 32
    /// (voxel monde 31) n'etait jamais mouchée, d'ou les vides entre chunks.
    #[test]
    fn mesh_reaches_chunk_boundaries() {
        let payload = generate_chunk((0, 0, 0));
        assert!(!payload.vertices.is_empty());
        let (mut max_x, mut max_y, mut max_z) = (0.0f32, 0.0f32, 0.0f32);
        for v in &payload.vertices {
            max_x = max_x.max(v.position[0]);
            max_y = max_y.max(v.position[1]);
            max_z = max_z.max(v.position[2]);
        }
        assert!((max_x - 32.0).abs() < 1e-3, "Le mesh doit atteindre x=32, max_x = {max_x}");
        assert!((max_z - 32.0).abs() < 1e-3, "Le mesh doit atteindre z=32, max_z = {max_z}");
        assert!(max_y > 1.0, "Le mesh doit contenir des couches superieures, max_y = {max_y}");
    }
}