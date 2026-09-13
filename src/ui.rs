use crate::types::*;
use glam::{Mat4, Vec3, Vec4};

pub const PRESET_SWATCHES: [[f32; 3]; 10] = [
    [0.95, 0.95, 0.95], [0.10, 0.10, 0.12], [0.90, 0.20, 0.20], [0.95, 0.50, 0.15],
    [0.95, 0.85, 0.15], [0.20, 0.75, 0.25], [0.15, 0.80, 0.85], [0.20, 0.45, 0.90],
    [0.65, 0.25, 0.85], [0.55, 0.35, 0.20],
];
pub const PRESET_BG_COLORS: [[f32; 3]; 8] = [
    [0.12, 0.14, 0.18], [0.24, 0.26, 0.30], [0.08, 0.18, 0.32], [0.55, 0.68, 0.82],
    [0.50, 0.28, 0.28], [0.16, 0.24, 0.18], [0.85, 0.82, 0.76], [0.02, 0.02, 0.03],
];
pub const GIZMO_CENTER_X: f32 = 0.86;
pub const GIZMO_CENTER_Y: f32 = 0.76;
pub const GIZMO_RADIUS: f32 = 0.11;
pub const LEFT_PALETTE_X0: f32 = -0.985;
pub const LEFT_PALETTE_X1: f32 = -0.835;
pub const LEFT_PALETTE_TOP_Y: f32 = 0.74;
pub const LEFT_TOOL_BTN_H: f32 = 0.039;
pub const LEFT_TOOL_BTN_GAP: f32 = 0.004;
pub const HOTBAR_LABELS: [&str; 10] = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"];
pub const ALL_TOOLS: [ToolType; 13] = [
    ToolType::Pencil, ToolType::Sphere, ToolType::Cylinder, ToolType::Disc,
    ToolType::Box, ToolType::Line, ToolType::Cone, ToolType::Pyramid,
    ToolType::Torus, ToolType::Paint, ToolType::Replace, ToolType::Bucket,
    ToolType::Select,
];

#[derive(Clone, Copy)]
pub struct GizmoAxis {
    pub dir: Vec3,
    pub name: &'static str,
    pub color: [f32; 4],
    pub yaw: f32,
    pub pitch: f32,
    pub is_positive: bool,
}

pub fn get_gizmo_axes() -> [GizmoAxis; 6] {
    [
        GizmoAxis { dir: Vec3::X,  name: "X",  color: [0.90, 0.20, 0.25, 1.0], yaw: std::f32::consts::PI, pitch: 0.0, is_positive: true },
        GizmoAxis { dir: -Vec3::X, name: "-X", color: [0.45, 0.20, 0.20, 0.75], yaw: 0.0, pitch: 0.0, is_positive: false },
        GizmoAxis { dir: Vec3::Y,  name: "Y",  color: [0.30, 0.80, 0.20, 1.0], yaw: -std::f32::consts::FRAC_PI_2, pitch: -1.56, is_positive: true },
        GizmoAxis { dir: -Vec3::Y, name: "-Y", color: [0.20, 0.45, 0.20, 0.75], yaw: -std::f32::consts::FRAC_PI_2, pitch: 1.56, is_positive: false },
        GizmoAxis { dir: Vec3::Z,  name: "Z",  color: [0.18, 0.55, 0.95, 1.0], yaw: -std::f32::consts::FRAC_PI_2, pitch: 0.0, is_positive: true },
        GizmoAxis { dir: -Vec3::Z, name: "-Z", color: [0.20, 0.30, 0.50, 0.75], yaw: std::f32::consts::FRAC_PI_2, pitch: 0.0, is_positive: false },
    ]
}

pub fn get_left_tool_btn_bounds(index: usize) -> (f32, f32, f32, f32) {
    let y1 = LEFT_PALETTE_TOP_Y - index as f32 * (LEFT_TOOL_BTN_H + LEFT_TOOL_BTN_GAP);
    let y0 = y1 - LEFT_TOOL_BTN_H;
    (LEFT_PALETTE_X0, y0, LEFT_PALETTE_X1, y1)
}
pub fn get_left_radius_controls_bounds() -> (f32, f32, f32, f32) {
    let y1 = LEFT_PALETTE_TOP_Y - 13.0 * (LEFT_TOOL_BTN_H + LEFT_TOOL_BTN_GAP) - 0.004;
    let y0 = y1 - LEFT_TOOL_BTN_H;
    (LEFT_PALETTE_X0, y0, LEFT_PALETTE_X1, y1)
}
pub fn get_left_mode_btn_bounds() -> (f32, f32, f32, f32) {
    let rad_y0 = LEFT_PALETTE_TOP_Y - 13.0 * (LEFT_TOOL_BTN_H + LEFT_TOOL_BTN_GAP) - 0.004 - LEFT_TOOL_BTN_H;
    let y1 = rad_y0 - LEFT_TOOL_BTN_GAP;
    let y0 = y1 - LEFT_TOOL_BTN_H;
    (LEFT_PALETTE_X0, y0, LEFT_PALETTE_X1, y1)
}
pub fn get_focus_button_bounds(aspect: f32) -> (f32, f32, f32, f32) {
    let disc_rx = (GIZMO_RADIUS + 0.02) / aspect;
    let btn_w = 0.075 / aspect;
    let btn_h = 0.046;
    let x1 = GIZMO_CENTER_X - disc_rx - 0.015;
    let x0 = x1 - btn_w;
    let y0 = GIZMO_CENTER_Y + 0.008;
    let y1 = y0 + btn_h;
    (x0, y0, x1, y1)
}
pub fn get_proj_button_bounds(aspect: f32) -> (f32, f32, f32, f32) {
    let disc_rx = (GIZMO_RADIUS + 0.02) / aspect;
    let btn_w = 0.075 / aspect;
    let btn_h = 0.046;
    let x1 = GIZMO_CENTER_X - disc_rx - 0.015;
    let x0 = x1 - btn_w;
    let y1 = GIZMO_CENTER_Y - 0.008;
    let y0 = y1 - btn_h;
    (x0, y0, x1, y1)
}
pub fn get_render_button_bounds(aspect: f32) -> (f32, f32, f32, f32) {
    let disc_rx = (GIZMO_RADIUS + 0.02) / aspect;
    let btn_w = 0.075 / aspect;
    let btn_h = 0.046;
    let x1 = GIZMO_CENTER_X - disc_rx - 0.015;
    let x0 = x1 - btn_w;
    let y1 = GIZMO_CENTER_Y - 0.008 - btn_h - 0.008;
    let y0 = y1 - btn_h;
    (x0, y0, x1, y1)
}

pub fn format_voxel_count(count: usize) -> String {
    if count >= 1_000_000 { format!("{:.2}M", count as f64 / 1_000_000.0) }
    else if count >= 1_000 { format!("{:.1}K", count as f64 / 1_000.0) }
    else { count.to_string() }
}

pub fn add_quad(verts: &mut Vec<UIVertex>, x0: f32, y0: f32, x1: f32, y1: f32, color: [f32; 4]) {
    verts.extend_from_slice(&[
        UIVertex { position: [x0, y0], color }, UIVertex { position: [x1, y0], color }, UIVertex { position: [x1, y1], color },
        UIVertex { position: [x0, y0], color }, UIVertex { position: [x1, y1], color }, UIVertex { position: [x0, y1], color },
    ]);
}
pub fn add_line(verts: &mut Vec<UIVertex>, x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32, aspect: f32, color: [f32; 4]) {
    let dx = (x1 - x0) * aspect;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-5 { return; }
    let half_t = thickness * 0.5;
    let nx = (-dy / len) * half_t / aspect;
    let ny = (dx / len) * half_t;
    verts.extend_from_slice(&[
        UIVertex { position: [x0 - nx, y0 - ny], color }, UIVertex { position: [x1 - nx, y1 - ny], color }, UIVertex { position: [x1 + nx, y1 + ny], color },
        UIVertex { position: [x0 - nx, y0 - ny], color }, UIVertex { position: [x1 + nx, y1 + ny], color }, UIVertex { position: [x0 + nx, y0 + ny], color },
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
        '#' => [0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010],
        _   => [0; 7],
    }
}

fn draw_glyph_raw(verts: &mut Vec<UIVertex>, glyph: &[u8; 7], x: f32, y: f32, pw: f32, ph: f32, color: [f32; 4]) {
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

pub fn draw_text(verts: &mut Vec<UIVertex>, text: &str, start_x: f32, start_y: f32, scale: f32, aspect: f32, color: [f32; 4]) {
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

pub fn draw_text_centered(verts: &mut Vec<UIVertex>, text: &str, cx: f32, cy: f32, scale: f32, aspect: f32, color: [f32; 4]) {
    let ph = 0.0032 * scale;
    let pw = ph / aspect;
    let total_w = (text.len() as f32 * 6.0 - 1.0) * pw;
    let total_h = 7.0 * ph;
    draw_text(verts, text, cx - total_w / 2.0, cy - total_h / 2.0, scale, aspect, color);
}

fn clip_line_segment(v0: &mut Vec4, v1: &mut Vec4, near_w: f32) -> bool {
    if v0.w < near_w && v1.w < near_w { return false; }
    if v0.w < near_w {
        let t = (near_w - v0.w) / (v1.w - v0.w);
        *v0 = *v0 + t * (*v1 - *v0);
    } else if v1.w < near_w {
        let t = (near_w - v1.w) / (v0.w - v1.w);
        *v1 = *v1 + t * (*v0 - *v1);
    }
    true
}

fn clip_line_2d(p0: &mut [f32; 2], p1: &mut [f32; 2], bound: f32) -> bool {
    let mut t0 = 0.0f32;
    let mut t1 = 1.0f32;
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let p = [-dx, dx, -dy, dy];
    let q = [p0[0] + bound, bound - p0[0], p0[1] + bound, bound - p0[1]];
    for i in 0..4 {
        if p[i].abs() < 1e-6 {
            if q[i] < 0.0 { return false; }
        } else {
            let r = q[i] / p[i];
            if p[i] < 0.0 {
                if r > t1 { return false; }
                if r > t0 { t0 = r; }
            } else {
                if r < t0 { return false; }
                if r < t1 { t1 = r; }
            }
        }
    }
    *p0 = [p0[0] + t0 * dx, p0[1] + t0 * dy];
    *p1 = [p0[0] + t1 * dx, p0[1] + t1 * dy];
    true
}

pub fn draw_quad_3d(verts: &mut Vec<UIVertex>, pts: [Vec3; 4], view_proj: Mat4, color: [f32; 4]) {
    let mut ndc = [[0.0f32; 2]; 4];
    for (i, p) in pts.iter().enumerate() {
        let v = view_proj * Vec4::new(p.x, p.y, p.z, 1.0);
        if v.w < 0.05 { return; }
        ndc[i] = [v.x / v.w, v.y / v.w];
        if ndc[i][0].abs() > 2.0 || ndc[i][1].abs() > 2.0 { return; }
    }
    verts.extend_from_slice(&[
        UIVertex { position: ndc[0], color }, UIVertex { position: ndc[1], color }, UIVertex { position: ndc[2], color },
        UIVertex { position: ndc[0], color }, UIVertex { position: ndc[2], color }, UIVertex { position: ndc[3], color },
    ]);
}

pub fn draw_box_wireframe(verts: &mut Vec<UIVertex>, min_p: Vec3, max_p: Vec3, aspect: f32, view_proj: Mat4, color: [f32; 4]) {
    let expand = (max_p - min_p) * 0.002;
    let p0 = min_p - expand;
    let p1 = max_p + expand;
    let corners = [
        Vec3::new(p0.x, p0.y, p0.z), Vec3::new(p1.x, p0.y, p0.z),
        Vec3::new(p0.x, p1.y, p0.z), Vec3::new(p1.x, p1.y, p0.z),
        Vec3::new(p0.x, p0.y, p1.z), Vec3::new(p1.x, p0.y, p1.z),
        Vec3::new(p0.x, p1.y, p1.z), Vec3::new(p1.x, p1.y, p1.z),
    ];
    let face_col = [color[0], color[1], color[2], color[3] * 0.12];
    let faces = [
        [corners[0], corners[1], corners[3], corners[2]],
        [corners[4], corners[5], corners[7], corners[6]],
        [corners[0], corners[4], corners[6], corners[2]],
        [corners[1], corners[5], corners[7], corners[3]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[2], corners[3], corners[7], corners[6]],
    ];
    for face in faces { draw_quad_3d(verts, face, view_proj, face_col); }
    let edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];
    for (i0, i1) in edges {
        let mut v0 = view_proj * Vec4::new(corners[i0].x, corners[i0].y, corners[i0].z, 1.0);
        let mut v1 = view_proj * Vec4::new(corners[i1].x, corners[i1].y, corners[i1].z, 1.0);
        if clip_line_segment(&mut v0, &mut v1, 0.05) {
            let mut ndc0 = [v0.x / v0.w, v0.y / v0.w];
            let mut ndc1 = [v1.x / v1.w, v1.y / v1.w];
            if clip_line_2d(&mut ndc0, &mut ndc1, 1.3) {
                add_line(verts, ndc0[0], ndc0[1], ndc1[0], ndc1[1], 0.0035, aspect, color);
            }
        }
    }
}

pub fn draw_cube_preview(
    verts: &mut Vec<UIVertex>, pos: Vec3, size: f32, aspect: f32, view_proj: Mat4,
    face_col: [f32; 4], edge_col: [f32; 4],
) {
    let p0 = pos;
    let p1 = pos + Vec3::splat(size);
    let corners = [
        Vec3::new(p0.x, p0.y, p0.z), Vec3::new(p1.x, p0.y, p0.z),
        Vec3::new(p0.x, p1.y, p0.z), Vec3::new(p1.x, p1.y, p0.z),
        Vec3::new(p0.x, p0.y, p1.z), Vec3::new(p1.x, p0.y, p1.z),
        Vec3::new(p0.x, p1.y, p1.z), Vec3::new(p1.x, p1.y, p1.z),
    ];
    let faces = [
        [corners[0], corners[1], corners[3], corners[2]],
        [corners[4], corners[5], corners[7], corners[6]],
        [corners[0], corners[4], corners[6], corners[2]],
        [corners[1], corners[5], corners[7], corners[3]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[2], corners[3], corners[7], corners[6]],
    ];
    for face in faces { draw_quad_3d(verts, face, view_proj, face_col); }
    let edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];
    for (i0, i1) in edges {
        let mut v0 = view_proj * Vec4::new(corners[i0].x, corners[i0].y, corners[i0].z, 1.0);
        let mut v1 = view_proj * Vec4::new(corners[i1].x, corners[i1].y, corners[i1].z, 1.0);
        if clip_line_segment(&mut v0, &mut v1, 0.05) {
            let mut ndc0 = [v0.x / v0.w, v0.y / v0.w];
            let mut ndc1 = [v1.x / v1.w, v1.y / v1.w];
            if clip_line_2d(&mut ndc0, &mut ndc1, 1.3) {
                add_line(verts, ndc0[0], ndc0[1], ndc1[0], ndc1[1], 0.002, aspect, edge_col);
            }
        }
    }
}

// ---------- Transform Gizmo drawing ----------

fn project_ndc(p: Vec3, vp: Mat4) -> Option<[f32; 2]> {
    let v = vp * Vec4::new(p.x, p.y, p.z, 1.0);
    if v.w < 0.05 { return None; }
    let ndc = [v.x / v.w, v.y / v.w];
    if ndc[0].abs() > 2.0 || ndc[1].abs() > 2.0 { return None; }
    Some(ndc)
}

fn draw_arrow(
    verts: &mut Vec<UIVertex>, from: Vec3, to: Vec3,
    vp: Mat4, aspect: f32, color: [f32; 4], thickness: f32, head_size: f32,
) {
    if let (Some(a), Some(b)) = (project_ndc(from, vp), project_ndc(to, vp)) {
        add_line(verts, a[0], a[1], b[0], b[1], thickness, aspect, color);
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len > 1e-5 {
            let ux = dx / len;
            let uy = dy / len;
            let px = -uy;
            let py = ux;
            let h = head_size;
            let c = 0.5;
            let p1 = [b[0] - ux * h + px * h * c, b[1] - uy * h + py * h * c];
            let p2 = [b[0] - ux * h - px * h * c, b[1] - uy * h - py * h * c];
            add_line(verts, b[0], b[1], p1[0], p1[1], thickness, aspect, color);
            add_line(verts, b[0], b[1], p2[0], p2[1], thickness, aspect, color);
        }
    }
}

fn draw_ring(
    verts: &mut Vec<UIVertex>, center: Vec3, axis: Vec3, radius: f32,
    vp: Mat4, aspect: f32, color: [f32; 4], segments: usize,
) {
    let up = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let u = axis.cross(up).normalize();
    let v = axis.cross(u).normalize();
    let mut prev: Option<[f32; 2]> = None;
    for i in 0..=segments {
        let t = (i as f32 / segments as f32) * std::f32::consts::TAU;
        let p = center + u * (radius * t.cos()) + v * (radius * t.sin());
        if let Some(ndc) = project_ndc(p, vp) {
            if let Some(p_ndc) = prev {
                add_line(verts, p_ndc[0], p_ndc[1], ndc[0], ndc[1], 0.004, aspect, color);
            }
            prev = Some(ndc);
        } else {
            prev = None;
        }
    }
}

pub fn draw_transform_gizmo(
    verts: &mut Vec<UIVertex>,
    pivot: Vec3,
    mode: GizmoTransformMode,
    hover: Option<(GizmoTransformMode, TransformAxis)>,
    dragging: Option<(GizmoTransformMode, TransformAxis)>,
    camera_pos: Vec3,
    vp: Mat4,
    aspect: f32,
    accumulated_move: Vec3,
) {
    let dist = (pivot - camera_pos).length().max(1.0);
    let arm = dist * 0.12;
    let p = pivot + accumulated_move;

    let mut axes = [TransformAxis::X, TransformAxis::Y, TransformAxis::Z];
    let cam_to_p = (camera_pos - p).normalize();
    axes.sort_by(|a, b| {
        let da = a.dir().dot(cam_to_p);
        let db = b.dir().dot(cam_to_p);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });

    for axis in axes {
        let ax_col = axis.color(false, false);

        // 1. Scale Handles (Cubes on inner axis stems)
        if mode == GizmoTransformMode::All || mode == GizmoTransformMode::Scale {
            let is_hover = hover == Some((GizmoTransformMode::Scale, axis));
            let is_drag = dragging == Some((GizmoTransformMode::Scale, axis));
            let col = if is_drag { [1.0, 1.0, 0.2, 1.0] } else if is_hover { [1.0, 1.0, 1.0, 1.0] } else { ax_col };
            let tip = p + axis.dir() * (arm * 0.48);
            if let (Some(a), Some(b)) = (project_ndc(p, vp), project_ndc(tip, vp)) {
                add_line(verts, a[0], a[1], b[0], b[1], 0.003, aspect, col);
            }
            let cube_sz = arm * 0.09;
            let cube_min = tip - Vec3::splat(cube_sz * 0.5);
            draw_cube_preview(verts, cube_min, cube_sz, aspect, vp, col, [col[0], col[1], col[2], 1.0]);
        }

        // 2. Rotate Handles (Full 3D planar rings)
        if mode == GizmoTransformMode::All || mode == GizmoTransformMode::Rotate {
            let is_hover = hover == Some((GizmoTransformMode::Rotate, axis));
            let is_drag = dragging == Some((GizmoTransformMode::Rotate, axis));
            let col = if is_drag { [1.0, 1.0, 0.2, 1.0] } else if is_hover { [1.0, 1.0, 1.0, 1.0] } else { ax_col };
            let ring_r = arm * 0.80;
            draw_ring(verts, p, axis.dir(), ring_r, vp, aspect, col, 48);
        }

        // 3. Move Handles (Outer arrows with heads)
        if mode == GizmoTransformMode::All || mode == GizmoTransformMode::Move {
            let is_hover = hover == Some((GizmoTransformMode::Move, axis));
            let is_drag = dragging == Some((GizmoTransformMode::Move, axis));
            let col = if is_drag { [1.0, 1.0, 0.2, 1.0] } else if is_hover { [1.0, 1.0, 1.0, 1.0] } else { ax_col };
            let from = p + axis.dir() * (arm * 0.80);
            let to = p + axis.dir() * (arm * 1.25);
            draw_arrow(verts, from, to, vp, aspect, col, 0.0055, 0.026);
        }
    }

    if let Some(c) = project_ndc(p, vp) {
        let r = 0.009;
        add_quad(verts, c[0] - r / aspect, c[1] - r, c[0] + r / aspect, c[1] + r, [1.0, 1.0, 1.0, 0.9]);
    }
}

// ---------- Main UI builder ----------

pub fn build_ui_vertices(
    selected_slot: usize, active_menu: ActiveMenu, hotbar_colors: &[[f32; 3]; 10],
    play_mode: PlayMode, is_ortho: bool, world_type: WorldType,
    edit_size: f32, target_pos: Option<[f32; 3]>, aspect: f32,
    camera_forward: Vec3, camera_right: Vec3, camera_up: Vec3,
    _cursor_free: bool, glb_settings: &GlbImportSettings, progress_val: f32,
    progress_stage: &str, view_proj: Mat4, error_banner: Option<&str>,
    tool_state: &ToolState, hud_status: &str, hud_tools: &str,
    size_str: &str, glb_cost_str: &str, bg_color: [f32; 3],
    focused_voxel_size: Option<f32>, preview_deltas: &[VoxelDelta],
    camera_pos: Vec3, octree: &crate::engine::Octree,
) -> Vec<UIVertex> {
    let mut verts = Vec::new();
    let g_cx = GIZMO_CENTER_X; let g_cy = GIZMO_CENTER_Y; let g_rad = GIZMO_RADIUS;
    let disc_rx = (g_rad + 0.018) / aspect;
    let disc_ry = g_rad + 0.018;
    add_quad(&mut verts, g_cx - disc_rx - 0.003, g_cy - disc_ry - 0.003, g_cx + disc_rx + 0.003, g_cy + disc_ry + 0.003, [0.25, 0.30, 0.38, 0.6]);
    add_quad(&mut verts, g_cx - disc_rx, g_cy - disc_ry, g_cx + disc_rx, g_cy + disc_ry, [0.08, 0.10, 0.14, 0.70]);

    // Focus [.]
    let (bx0, by0, bx1, by1) = get_focus_button_bounds(aspect);
    add_quad(&mut verts, bx0 - 0.003, by0 - 0.003, bx1 + 0.003, by1 + 0.003, [0.35, 0.40, 0.50, 0.8]);
    add_quad(&mut verts, bx0, by0, bx1, by1, [0.12, 0.15, 0.22, 0.90]);
    draw_text_centered(&mut verts, "[.]", (bx0 + bx1) / 2.0, (by0 + by1) / 2.0, 1.05, aspect, [0.3, 0.9, 1.0, 1.0]);

    // Projection [ISO] / [PER]
    let (px0, py0, px1, py1) = get_proj_button_bounds(aspect);
    let proj_border = if is_ortho { [0.3, 0.85, 1.0, 0.9] } else { [0.35, 0.40, 0.50, 0.8] };
    let proj_bg = if is_ortho { [0.18, 0.45, 0.65, 0.95] } else { [0.12, 0.15, 0.22, 0.90] };
    let proj_text = if is_ortho { "ISO" } else { "PER" };
    add_quad(&mut verts, px0 - 0.003, py0 - 0.003, px1 + 0.003, py1 + 0.003, proj_border);
    add_quad(&mut verts, px0, py0, px1, py1, proj_bg);
    draw_text_centered(&mut verts, proj_text, (px0 + px1) / 2.0, (py0 + py1) / 2.0, 1.0, aspect, [1.0, 1.0, 1.0, 1.0]);

    // Render [REN]
    let (rx0, ry0, rx1, ry1) = get_render_button_bounds(aspect);
    add_quad(&mut verts, rx0 - 0.003, ry0 - 0.003, rx1 + 0.003, ry1 + 0.003, [0.35, 0.40, 0.50, 0.8]);
    add_quad(&mut verts, rx0, ry0, rx1, ry1, [0.12, 0.15, 0.22, 0.90]);
    draw_text_centered(&mut verts, "REN", (rx0 + rx1) / 2.0, (ry0 + ry1) / 2.0, 1.0, aspect, [0.3, 0.95, 0.5, 1.0]);

    let mut axes_projected: Vec<(GizmoAxis, f32, f32, f32)> = get_gizmo_axes().into_iter().map(|ax| (ax, ax.dir.dot(camera_right), ax.dir.dot(camera_up), ax.dir.dot(camera_forward))).collect();
    axes_projected.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
    for (ax, sx, sy, _) in &axes_projected {
        let tip_x = g_cx + (sx * g_rad) / aspect;
        let tip_y = g_cy + (sy * g_rad);
        if ax.is_positive {
            add_line(&mut verts, g_cx, g_cy, tip_x, tip_y, 0.0045, aspect, [ax.color[0], ax.color[1], ax.color[2], 0.85]);
            let node_r = 0.021;
            add_quad(&mut verts, tip_x - node_r / aspect, tip_y - node_r, tip_x + node_r / aspect, tip_y + node_r, ax.color);
            draw_text_centered(&mut verts, ax.name, tip_x, tip_y, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
        } else {
            add_line(&mut verts, g_cx, g_cy, tip_x, tip_y, 0.0025, aspect, [ax.color[0], ax.color[1], ax.color[2], 0.35]);
            let node_r = 0.010;
            add_quad(&mut verts, tip_x - node_r / aspect, tip_y - node_r, tip_x + node_r / aspect, tip_y + node_r, ax.color);
        }
    }

    // Hotbar
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
        draw_text_centered(&mut verts, HOTBAR_LABELS[i], (x0 + x1) / 2.0, y_top + 0.02, 1.0, aspect, [0.9, 0.9, 0.9, 0.9]);
    }

    if active_menu == ActiveMenu::None {
        // --- Left Tools HUD Palette ---
        for (i, &tool) in ALL_TOOLS.iter().enumerate() {
            let (tx0, ty0, tx1, ty1) = get_left_tool_btn_bounds(i);
            let is_active = tool_state.active_tool == tool;
            let border_col = if is_active { [1.0, 0.85, 0.2, 1.0] } else { [0.25, 0.30, 0.38, 0.7] };
            let bg_col = if is_active { [0.22, 0.35, 0.50, 0.95] } else { [0.10, 0.12, 0.16, 0.85] };

            add_quad(&mut verts, tx0 - 0.002, ty0 - 0.002, tx1 + 0.002, ty1 + 0.002, border_col);
            add_quad(&mut verts, tx0, ty0, tx1, ty1, bg_col);

            let text_col = if is_active { [1.0, 1.0, 1.0, 1.0] } else { [0.80, 0.85, 0.90, 0.9] };
            draw_text_centered(&mut verts, tool.short_name(), (tx0 + tx1) * 0.5, (ty0 + ty1) * 0.5, 0.95, aspect, text_col);
        }

        // --- Radius Controls [-] R [+] ---
        let (rx0, ry0, rx1, ry1) = get_left_radius_controls_bounds();
        let rad_minus_x1 = rx0 + 0.032;
        let rad_plus_x0 = rx1 - 0.032;
        add_quad(&mut verts, rx0 - 0.002, ry0 - 0.002, rx1 + 0.002, ry1 + 0.002, [0.25, 0.30, 0.38, 0.7]);
        add_quad(&mut verts, rx0, ry0, rx1, ry1, [0.10, 0.12, 0.16, 0.85]);
        add_quad(&mut verts, rx0, ry0, rad_minus_x1, ry1, [0.18, 0.22, 0.30, 0.9]);
        draw_text_centered(&mut verts, "-", (rx0 + rad_minus_x1) * 0.5, (ry0 + ry1) * 0.5, 1.0, aspect, [1.0, 1.0, 1.0, 1.0]);
        draw_text_centered(&mut verts, &format!("R:{:.0}", tool_state.brush_radius), (rad_minus_x1 + rad_plus_x0) * 0.5, (ry0 + ry1) * 0.5, 0.85, aspect, [0.3, 0.9, 1.0, 1.0]);
        add_quad(&mut verts, rad_plus_x0, ry0, rx1, ry1, [0.18, 0.22, 0.30, 0.9]);
        draw_text_centered(&mut verts, "+", (rad_plus_x0 + rx1) * 0.5, (ry0 + ry1) * 0.5, 1.0, aspect, [1.0, 1.0, 1.0, 1.0]);

        // --- Mode Toggle (SOLID / HOLLOW) ---
        let (mx0, my0, mx1, my1) = get_left_mode_btn_bounds();
        let mode_border = if tool_state.hollow { [0.95, 0.60, 0.20, 0.9] } else { [0.25, 0.30, 0.38, 0.7] };
        let mode_bg = if tool_state.hollow { [0.35, 0.22, 0.12, 0.95] } else { [0.10, 0.12, 0.16, 0.85] };
        add_quad(&mut verts, mx0 - 0.002, my0 - 0.002, mx1 + 0.002, my1 + 0.002, mode_border);
        add_quad(&mut verts, mx0, my0, mx1, my1, mode_bg);
        draw_text_centered(&mut verts, if tool_state.hollow { "HOLLOW" } else { "SOLID" }, (mx0 + mx1) * 0.5, (my0 + my1) * 0.5, 0.85, aspect, [1.0, 1.0, 1.0, 1.0]);

        // Floating voxels preview + transform gizmo
        if tool_state.selection.is_floating {
            if let Some(pos) = target_pos {
                let p_anchor = Vec3::from(pos);
                let mut f_min = Vec3::splat(f32::MAX);
                let mut f_max = Vec3::splat(f32::MIN);
                let mut inside_count = 0;
                for &(rel, s, _) in &tool_state.selection.floating_voxels {
                    let v_pos = p_anchor + rel;
                    f_min = f_min.min(v_pos);
                    f_max = f_max.max(v_pos + Vec3::splat(s));
                    let is_inside = octree.query_point(v_pos) != 0;
                    if is_inside {
                        inside_count += 1;
                        draw_cube_preview(
                            &mut verts, v_pos, s, aspect, view_proj,
                            [1.0, 0.20, 0.15, 0.65], [1.0, 0.85, 0.20, 0.98],
                        );
                    } else {
                        draw_cube_preview(
                            &mut verts, v_pos, s, aspect, view_proj,
                            [0.2, 0.8, 1.0, 0.35], [0.3, 0.9, 1.0, 0.85],
                        );
                    }
                }
                if f_min.x <= f_max.x {
                    let box_border = if inside_count > 0 { [1.0, 0.4, 0.2, 0.9] } else { [0.2, 0.9, 1.0, 0.9] };
                    draw_box_wireframe(&mut verts, f_min, f_max, aspect, view_proj, box_border);
                }

                let pivot = Vec3::from(pos);
                draw_transform_gizmo(
                    &mut verts, pivot, tool_state.gizmo.mode,
                    tool_state.gizmo.hover_handle, tool_state.gizmo.dragging_handle,
                    camera_pos, view_proj, aspect, tool_state.gizmo.accumulated_move,
                );

                let snap_hint = if tool_state.gizmo.dragging_handle.is_some() {
                    let (mode, _) = tool_state.gizmo.dragging_handle.unwrap();
                    if mode == GizmoTransformMode::Rotate { "CTRL = SNAP 15DEG" }
                    else { "CTRL = SNAP 1 VOXEL" }
                } else {
                    "DRAG ARROWS (MOVE) | RINGS (ROT) | BOXES (SCALE) | [W] FILTER"
                };
                draw_text(
                    &mut verts,
                    &format!("GIZMO: {} | {} | [LMB] PLACE | [ESC] CANCEL",
                             tool_state.gizmo.mode.label(), snap_hint),
                    -0.76, 0.66, 0.95, aspect, [0.2, 0.9, 1.0, 1.0],
                );
            }
        }

        // Render completed selection bounds
        if let Some((b_min, b_max)) = tool_state.selection.bounds {
            draw_box_wireframe(&mut verts, b_min, b_max, aspect, view_proj, [0.3, 0.9, 1.0, 0.95]);
        }

        // Active selection drag box or initial cursor box
        if tool_state.active_tool == ToolType::Select && !tool_state.selection.is_floating {
            if let Some(anchor) = tool_state.selection.anchor {
                if let Some(t_pos) = target_pos {
                    let target_vec = Vec3::from(t_pos);
                    let b_min = anchor.min(target_vec);
                    let b_max = anchor.max(target_vec) + Vec3::splat(edit_size);
                    draw_box_wireframe(&mut verts, b_min, b_max, aspect, view_proj, [1.0, 0.85, 0.2, 0.9]);
                }
            } else if let Some(t_pos) = target_pos {
                draw_cube_preview(
                    &mut verts, Vec3::from(t_pos), edit_size, aspect, view_proj,
                    [0.3, 0.9, 1.0, 0.25], [0.3, 0.9, 1.0, 0.85],
                );
            }
        }

        // Preview deltas with inside vs outside color differentiation
        if target_pos.is_some() && !tool_state.selection.is_floating && tool_state.active_tool != ToolType::Select {
            let cur_col = hotbar_colors[selected_slot];
            let max_voxels_rendered = 2500;
            let step = if preview_deltas.len() > max_voxels_rendered {
                (preview_deltas.len() / max_voxels_rendered) + 1
            } else { 1 };
            let mut inside_count = 0;
            for delta in preview_deltas.iter().step_by(step) {
                let pos = Vec3::from(delta.pos);
                let size = delta.size;
                if delta.old_material == 0 {
                    draw_cube_preview(
                        &mut verts, pos, size, aspect, view_proj,
                        [cur_col[0], cur_col[1], cur_col[2], 0.32],
                        [cur_col[0], cur_col[1], cur_col[2], 0.85],
                    );
                } else {
                    inside_count += 1;
                    draw_cube_preview(
                        &mut verts, pos, size, aspect, view_proj,
                        [1.0, 0.20, 0.15, 0.65],
                        [1.0, 0.85, 0.20, 0.98],
                    );
                }
            }
            if inside_count > 0 {
                draw_text(&mut verts, &format!("PENETRATING: {} VOXELS INSIDE OBJECT", inside_count * step), -0.76, 0.74, 1.0, aspect, [1.0, 0.35, 0.2, 1.0]);
            }
        }
    }

    if let Some(err) = error_banner {
        add_quad(&mut verts, -0.85, 0.68, 0.85, 0.80, [0.70, 0.12, 0.12, 0.95]);
        draw_text_centered(&mut verts, err, 0.0, 0.74, 0.95, aspect, [1.0, 1.0, 1.0, 1.0]);
    }

    match active_menu {
        ActiveMenu::None => {
            let aim_str = if let Some(sz) = focused_voxel_size {
                if sz < 1.0 { format!("AIM VOXEL: 1/{} ({:.4})", (1.0 / sz).round() as u32, sz) }
                else { format!("AIM VOXEL: {:.1}", sz) }
            } else { "AIM VOXEL: NONE".to_string() };
            let status_line = format!("{} | {}", hud_status, aim_str);
            draw_text(&mut verts, &status_line, -0.96, 0.92, 1.22, aspect, [1.0, 1.0, 1.0, 0.95]);
            draw_text(&mut verts, hud_tools, -0.96, 0.86, 1.0, aspect, [0.3, 0.9, 1.0, 0.95]);
            draw_text(&mut verts, "[S] SEL  [G] GRAB  [R] RES*2  [SHIFT+R] ROT  [W] GIZMO  [CTRL+C/V] COPY/PASTE  [DEL] CLR  [C] PICK", -0.96, 0.80, 0.82, aspect, [0.95, 0.85, 0.35, 0.95]);
        }
        ActiveMenu::Edit => {
            add_quad(&mut verts, -1.0, -1.0, 1.0, 1.0, [0.03, 0.04, 0.06, 0.75]);
            add_quad(&mut verts, -0.566, -0.586, 0.566, 0.656, [0.25, 0.35, 0.50, 1.0]);
            add_quad(&mut verts, -0.56, -0.58, 0.56, 0.65, [0.10, 0.12, 0.16, 0.98]);
            draw_text_centered(&mut verts, "EDIT STUDIO - PALETTE & OCTREE (E)", 0.0, 0.58, 1.3, aspect, [1.0, 0.9, 0.2, 1.0]);
            let sw_w = 0.082; let sw_gap = 0.015; let sw_tot = 10.0 * sw_w + 9.0 * sw_gap; let s_start_x = -sw_tot / 2.0;
            for (i, &rgb) in hotbar_colors.iter().enumerate() {
                let sx0 = s_start_x + i as f32 * (sw_w + sw_gap);
                let sx1 = sx0 + sw_w;
                if selected_slot == i { add_quad(&mut verts, sx0 - 0.008, 0.422, sx1 + 0.008, 0.528, [1.0, 0.9, 0.1, 1.0]); }
                else { add_quad(&mut verts, sx0 - 0.004, 0.426, sx1 + 0.004, 0.524, [0.22, 0.24, 0.30, 1.0]); }
                add_quad(&mut verts, sx0, 0.43, sx1, 0.52, [rgb[0], rgb[1], rgb[2], 1.0]);
                draw_text_centered(&mut verts, HOTBAR_LABELS[i], (sx0 + sx1) / 2.0, 0.542, 1.0, aspect, [0.8, 0.8, 0.8, 0.9]);
            }
            let [cur_r, cur_g, cur_b] = hotbar_colors[selected_slot];
            add_quad(&mut verts, 0.24, 0.18, 0.46, 0.37, [0.25, 0.28, 0.35, 1.0]);
            add_quad(&mut verts, 0.248, 0.188, 0.452, 0.362, [cur_r, cur_g, cur_b, 1.0]);
            draw_text_centered(&mut verts, "ACTIVE COLOR", 0.35, 0.39, 1.0, aspect, [0.85, 0.85, 0.85, 0.9]);
            for (lbl, val, y0, y1, bar_col) in [("R", cur_r, 0.32, 0.36, [0.90, 0.25, 0.25, 1.0]), ("G", cur_g, 0.25, 0.29, [0.25, 0.85, 0.30, 1.0]), ("B", cur_b, 0.18, 0.22, [0.25, 0.50, 0.95, 1.0])] {
                draw_text_centered(&mut verts, lbl, -0.37, (y0 + y1) / 2.0, 1.1, aspect, bar_col);
                add_quad(&mut verts, -0.32, y0, 0.18, y1, [0.18, 0.20, 0.25, 1.0]);
                let filled_x = -0.32 + val * 0.50;
                add_quad(&mut verts, -0.32, y0, filled_x, y1, bar_col);
                add_quad(&mut verts, filled_x - 0.010, y0 - 0.006, filled_x + 0.010, y1 + 0.006, [1.0, 1.0, 1.0, 1.0]);
            }
            draw_text_centered(&mut verts, "QUICK PALETTE CHIPS", 0.0, 0.135, 1.0, aspect, [0.75, 0.75, 0.8, 0.9]);
            let pw_w = 0.076; let pw_gap = 0.012; let pw_tot = 10.0 * pw_w + 9.0 * pw_gap; let pw_start_x = -pw_tot / 2.0;
            for (i, &rgb) in PRESET_SWATCHES.iter().enumerate() {
                let px0 = pw_start_x + i as f32 * (pw_w + pw_gap);
                add_quad(&mut verts, px0 - 0.003, 0.047, px0 + pw_w + 0.003, 0.113, [0.3, 0.3, 0.35, 1.0]);
                add_quad(&mut verts, px0, 0.05, px0 + pw_w, 0.11, [rgb[0], rgb[1], rgb[2], 1.0]);
            }
            add_quad(&mut verts, -0.40, -0.10, -0.22, -0.02, [0.35, 0.40, 0.55, 1.0]);
            draw_text_centered(&mut verts, "/ 2 (F)", -0.31, -0.06, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            draw_text_centered(&mut verts, &format!("CURRENT: {size_str}"), 0.0, -0.06, 1.25, aspect, [1.0, 0.85, 0.2, 1.0]);
            add_quad(&mut verts, 0.22, -0.10, 0.40, -0.02, [0.35, 0.40, 0.55, 1.0]);
            draw_text_centered(&mut verts, "* 2 (R)", 0.31, -0.06, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.22, -0.25, 0.22, -0.17, [0.20, 0.50, 0.30, 1.0]);
            draw_text_centered(&mut verts, "DONE (PRESS E)", 0.0, -0.21, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
        }
        ActiveMenu::Pause => {
            add_quad(&mut verts, -1.0, -1.0, 1.0, 1.0, [0.03, 0.04, 0.06, 0.80]);
            add_quad(&mut verts, -0.426, -0.766, 0.426, 0.726, [0.45, 0.45, 0.50, 1.0]);
            add_quad(&mut verts, -0.42, -0.76, 0.42, 0.72, [0.12, 0.13, 0.17, 0.98]);
            draw_text_centered(&mut verts, "PAUSE / SYSTEM MENU", 0.0, 0.62, 1.3, aspect, [0.95, 0.95, 0.95, 1.0]);
            add_quad(&mut verts, -0.30, 0.48, 0.30, 0.56, [0.20, 0.55, 0.75, 1.0]);
            draw_text_centered(&mut verts, ">> IMPORT 3D MODEL (GLB) <<", 0.0, 0.52, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, 0.38, 0.30, 0.46, [0.25, 0.35, 0.55, 1.0]);
            draw_text_centered(&mut verts, if play_mode == PlayMode::Flying { "PLAY MODE: FLYING (M)" } else { "PLAY MODE: REAL (M)" }, 0.0, 0.42, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, 0.28, 0.30, 0.36, [0.22, 0.40, 0.55, 1.0]);
            draw_text_centered(&mut verts, if is_ortho { "VIEW: ORTHOGRAPHIC (P)" } else { "VIEW: PERSPECTIVE (P)" }, 0.0, 0.32, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, 0.18, 0.30, 0.26, [0.35, 0.25, 0.50, 1.0]);
            draw_text_centered(&mut verts, &format!("WORLD: {}", world_type.name()), 0.0, 0.22, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, 0.08, 0.30, 0.16, [0.60, 0.30, 0.20, 1.0]);
            draw_text_centered(&mut verts, "CLEAR SCENE", 0.0, 0.12, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, -0.02, -0.02, 0.06, [0.25, 0.45, 0.35, 1.0]);
            draw_text_centered(&mut verts, "SAVE (F5)", -0.16, 0.02, 1.0, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, 0.02, -0.02, 0.30, 0.06, [0.35, 0.45, 0.25, 1.0]);
            draw_text_centered(&mut verts, "LOAD (F9)", 0.16, 0.02, 1.0, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, -0.19, 0.30, -0.11, [0.22, 0.38, 0.52, 1.0]);
            add_quad(&mut verts, -0.285, -0.175, -0.215, -0.125, [0.4, 0.45, 0.55, 1.0]);
            add_quad(&mut verts, -0.280, -0.170, -0.220, -0.130, [bg_color[0], bg_color[1], bg_color[2], 1.0]);
            draw_text_centered(&mut verts, "BACKGROUND COLOR (SLIDERS)...", 0.04, -0.15, 0.95, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, -0.29, 0.30, -0.21, [0.18, 0.45, 0.32, 1.0]);
            draw_text_centered(&mut verts, "RENDER MODE / HIDE UI (F1)", 0.0, -0.25, 1.05, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, -0.39, 0.30, -0.31, [0.25, 0.40, 0.55, 1.0]);
            draw_text_centered(&mut verts, "CONTROLS", 0.0, -0.35, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, -0.51, 0.30, -0.43, [0.20, 0.55, 0.30, 1.0]);
            draw_text_centered(&mut verts, "RESUME (ESC)", 0.0, -0.47, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.30, -0.63, 0.30, -0.55, [0.55, 0.20, 0.20, 1.0]);
            draw_text_centered(&mut verts, "QUIT TO DESKTOP", 0.0, -0.59, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
        }
        ActiveMenu::BgColorModal => {
            add_quad(&mut verts, -1.0, -1.0, 1.0, 1.0, [0.02, 0.03, 0.05, 0.70]);
            add_quad(&mut verts, -0.406, -0.426, 0.406, 0.426, [0.35, 0.45, 0.60, 1.0]);
            add_quad(&mut verts, -0.400, -0.420, 0.400, 0.420, [0.10, 0.12, 0.16, 0.98]);
            draw_text_centered(&mut verts, "BACKGROUND COLOR PICKER", 0.0, 0.35, 1.25, aspect, [1.0, 0.9, 0.2, 1.0]);
            let [cur_r, cur_g, cur_b] = bg_color;
            let s_x0 = -0.34;
            let s_x1 = 0.08;
            add_quad(&mut verts, 0.165, 0.105, 0.355, 0.295, [0.40, 0.45, 0.55, 1.0]);
            add_quad(&mut verts, 0.170, 0.110, 0.350, 0.290, [cur_r, cur_g, cur_b, 1.0]);
            draw_text_centered(&mut verts, "PREVIEW", 0.26, 0.315, 0.9, aspect, [0.85, 0.85, 0.85, 1.0]);
            draw_text_centered(&mut verts, &format!("#{:02X}{:02X}{:02X}", (cur_r * 255.0).round() as u8, (cur_g * 255.0).round() as u8, (cur_b * 255.0).round() as u8), 0.26, 0.075, 0.85, aspect, [0.8, 0.85, 0.9, 1.0]);
            for (lbl, val, y0, y1, bar_col) in [
                ("R", cur_r, 0.24, 0.28, [0.90, 0.25, 0.25, 1.0]),
                ("G", cur_g, 0.17, 0.21, [0.25, 0.85, 0.30, 1.0]),
                ("B", cur_b, 0.10, 0.14, [0.25, 0.50, 0.95, 1.0]),
            ] {
                draw_text_centered(&mut verts, lbl, -0.365, (y0 + y1) * 0.5, 1.1, aspect, bar_col);
                add_quad(&mut verts, s_x0, y0, s_x1, y1, [0.18, 0.20, 0.25, 1.0]);
                let filled_x = s_x0 + val * (s_x1 - s_x0);
                add_quad(&mut verts, s_x0, y0, filled_x, y1, bar_col);
                add_quad(&mut verts, filled_x - 0.008, y0 - 0.005, filled_x + 0.008, y1 + 0.005, [1.0, 1.0, 1.0, 1.0]);
                draw_text(&mut verts, &format!("{:.0}%", val * 100.0), s_x1 + 0.015, (y0 + y1) * 0.5 - 0.012, 0.85, aspect, [0.85, 0.85, 0.85, 1.0]);
            }
            draw_text_centered(&mut verts, "QUICK PRESETS", 0.0, 0.01, 0.95, aspect, [0.8, 0.85, 0.9, 1.0]);
            let bg_w = 0.068; let bg_gap = 0.008; let bg_tot = 8.0 * bg_w + 7.0 * bg_gap; let bg_start_x = -bg_tot / 2.0;
            let bg_y0 = -0.10; let bg_y1 = -0.04;
            for (i, &col) in PRESET_BG_COLORS.iter().enumerate() {
                let x0 = bg_start_x + i as f32 * (bg_w + bg_gap);
                let x1 = x0 + bg_w;
                add_quad(&mut verts, x0, bg_y0, x1, bg_y1, [col[0], col[1], col[2], 1.0]);
            }
            add_quad(&mut verts, -0.22, -0.23, 0.22, -0.15, [0.20, 0.55, 0.35, 1.0]);
            draw_text_centered(&mut verts, "APPLY & CLOSE (ESC)", 0.0, -0.19, 1.05, aspect, [1.0, 1.0, 1.0, 1.0]);
        }
        ActiveMenu::Controls => {
            add_quad(&mut verts, -1.0, -1.0, 1.0, 1.0, [0.03, 0.04, 0.06, 0.95]);
            draw_text_centered(&mut verts, "BLENDER-COMPLIANT CONTROLS", 0.0, 0.60, 1.4, aspect, [1.0, 1.0, 1.0, 1.0]);
            draw_text_centered(&mut verts, "MMB = ORBIT  |  SHIFT + MMB = PAN  |  CTRL + MMB / WHEEL = ZOOM", 0.0, 0.42, 0.95, aspect, [0.3, 0.9, 1.0, 1.0]);
            draw_text_centered(&mut verts, "NUMPAD 2 / 4 / 6 / 8 = ORBIT (15 DEG)  |  NUMPAD 1/3/7/9 = VIEWS", 0.0, 0.32, 0.95, aspect, [0.3, 0.9, 1.0, 1.0]);
            draw_text_centered(&mut verts, "CTRL + Z = UNDO  |  CTRL + SHIFT + Z / CTRL + Y = REDO", 0.0, 0.22, 0.95, aspect, [1.0, 0.85, 0.3, 1.0]);
            draw_text_centered(&mut verts, "V: PEN | O: SPH | Y: CYL | U: DSC | B: BOX | L: LIN | J: CON | N: PYR | T: TOR | S: SEL", 0.0, 0.12, 0.80, aspect, [0.9, 0.9, 0.9, 1.0]);
            draw_text_centered(&mut verts, "G = GRAB SELECTION  |  R = VOXEL SIZE *2  |  SHIFT+R = ROTATE 90DEG  |  DEL = CLEAR", 0.0, 0.02, 0.95, aspect, [0.3, 1.0, 0.5, 1.0]);
            draw_text_centered(&mut verts, "W = CYCLE GIZMO (MOVE/ROTATE/SCALE)  |  CTRL + DRAG = SNAP (1 VOXEL / 15DEG)", 0.0, -0.08, 0.95, aspect, [0.9, 0.9, 0.3, 1.0]);
            draw_text_centered(&mut verts, "CTRL + C = COPY SELECTION  |  CTRL + V = PASTE CLIPBOARD", 0.0, -0.18, 0.95, aspect, [0.9, 0.9, 0.9, 1.0]);
            draw_text_centered(&mut verts, "LMB = APPLY/PLACE  |  RMB = ERASE  |  C = PICK COLOR AT CURSOR", 0.0, -0.28, 0.95, aspect, [0.9, 0.9, 0.9, 1.0]);
            add_quad(&mut verts, -0.30, -0.66, 0.30, -0.56, [0.45, 0.22, 0.22, 1.0]);
            draw_text_centered(&mut verts, "BACK (ESC)", 0.0, -0.61, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
        }
        ActiveMenu::ImportParams => {
            add_quad(&mut verts, -1.0, -1.0, 1.0, 1.0, [0.02, 0.03, 0.05, 0.85]);
            add_quad(&mut verts, -0.426, -0.686, 0.426, 0.626, [0.30, 0.45, 0.65, 1.0]);
            add_quad(&mut verts, -0.42, -0.68, 0.42, 0.62, [0.08, 0.10, 0.14, 0.98]);
            draw_text_centered(&mut verts, "GLB VOXEL IMPORT SETTINGS", 0.0, 0.54, 1.25, aspect, [0.3, 0.9, 1.0, 1.0]);
            let file_label = glb_settings.selected_file.as_ref().and_then(|f| f.file_name()).map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "NO FILE SELECTED".into());
            let short_file = if file_label.len() > 18 { format!("{}...", &file_label[..15]) } else { file_label };
            add_quad(&mut verts, -0.38, 0.38, 0.20, 0.46, [0.05, 0.06, 0.09, 1.0]);
            draw_text(&mut verts, &format!("FILE: {short_file}"), -0.36, 0.42, 0.95, aspect, [0.85, 0.85, 0.4, 1.0]);
            add_quad(&mut verts, 0.22, 0.38, 0.38, 0.46, [0.25, 0.35, 0.50, 1.0]);
            draw_text_centered(&mut verts, "CHANGE", 0.30, 0.42, 0.95, aspect, [1.0, 1.0, 1.0, 1.0]);
            draw_text(&mut verts, "TARGET HEIGHT (BLOCKS):", -0.38, 0.29, 1.0, aspect, [0.8, 0.8, 0.8, 1.0]);
            add_quad(&mut verts, -0.38, 0.19, -0.28, 0.27, [0.25, 0.30, 0.40, 1.0]);
            draw_text_centered(&mut verts, "- 4", -0.33, 0.23, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            draw_text_centered(&mut verts, &format!("{:.0} BLOCKS", glb_settings.target_height), 0.0, 0.23, 1.1, aspect, [0.3, 0.9, 1.0, 1.0]);
            add_quad(&mut verts, 0.28, 0.19, 0.38, 0.27, [0.25, 0.30, 0.40, 1.0]);
            draw_text_centered(&mut verts, "+ 4", 0.33, 0.23, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            draw_text(&mut verts, "VOXEL RESOLUTION:", -0.38, 0.09, 1.0, aspect, [0.8, 0.8, 0.8, 1.0]);
            add_quad(&mut verts, -0.38, -0.01, -0.28, 0.07, [0.25, 0.30, 0.40, 1.0]);
            draw_text_centered(&mut verts, "/ 2", -0.33, 0.03, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            draw_text_centered(&mut verts, &format!("{:.3}", glb_settings.voxel_size), 0.0, 0.03, 1.1, aspect, [0.3, 0.9, 1.0, 1.0]);
            add_quad(&mut verts, 0.28, -0.01, 0.38, 0.07, [0.25, 0.30, 0.40, 1.0]);
            draw_text_centered(&mut verts, "* 2", 0.33, 0.03, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            draw_text(&mut verts, "MAX PALETTE SIZE:", -0.38, -0.11, 1.0, aspect, [0.8, 0.8, 0.8, 1.0]);
            add_quad(&mut verts, -0.38, -0.21, -0.28, -0.13, [0.25, 0.30, 0.40, 1.0]);
            draw_text_centered(&mut verts, "/ 2", -0.33, -0.17, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            draw_text_centered(&mut verts, &format!("{}", glb_settings.palette_size), 0.0, -0.17, 1.1, aspect, [0.3, 0.9, 1.0, 1.0]);
            add_quad(&mut verts, 0.28, -0.21, 0.38, -0.13, [0.25, 0.30, 0.40, 1.0]);
            draw_text_centered(&mut verts, "* 2", 0.33, -0.17, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            draw_text(&mut verts, "PLACEMENT ANCHOR:", -0.38, -0.29, 1.0, aspect, [0.8, 0.8, 0.8, 1.0]);
            add_quad(&mut verts, -0.38, -0.39, 0.38, -0.31, [0.18, 0.24, 0.34, 1.0]);
            draw_text_centered(&mut verts, if glb_settings.place_at_aim { "CROSSHAIR / RAYCAST AIM" } else { "AT PLAYER POSITION" }, 0.0, -0.35, 1.0, aspect, [1.0, 1.0, 1.0, 1.0]);
            let cost_col = if glb_settings.is_safe() { [0.3, 0.9, 0.4, 1.0] } else { [0.95, 0.25, 0.2, 1.0] };
            draw_text_centered(&mut verts, glb_cost_str, 0.0, -0.42, 0.85, aspect, cost_col);
            let btn_col = if glb_settings.selected_file.is_some() && glb_settings.is_safe() { [0.20, 0.60, 0.30, 1.0] } else { [0.35, 0.20, 0.20, 0.8] };
            add_quad(&mut verts, -0.38, -0.54, 0.38, -0.44, btn_col);
            draw_text_centered(&mut verts, if glb_settings.is_safe() { "VOXELIZE & INSERT (SOLID)" } else { "TOO DENSE (REDUCE SETTINGS)" }, 0.0, -0.49, 1.0, aspect, [1.0, 1.0, 1.0, 1.0]);
            add_quad(&mut verts, -0.38, -0.66, 0.38, -0.56, [0.45, 0.22, 0.22, 1.0]);
            draw_text_centered(&mut verts, "CANCEL (ESC)", 0.0, -0.61, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
        }
        ActiveMenu::Voxelizing => {
            add_quad(&mut verts, -1.0, -1.0, 1.0, 1.0, [0.02, 0.03, 0.05, 0.88]);
            add_quad(&mut verts, -0.446, -0.246, 0.446, 0.266, [0.25, 0.45, 0.70, 1.0]);
            add_quad(&mut verts, -0.44, -0.24, 0.44, 0.26, [0.08, 0.10, 0.15, 0.98]);
            draw_text_centered(&mut verts, "VOXELIZING 3D MODEL", 0.0, 0.18, 1.25, aspect, [0.3, 0.9, 1.0, 1.0]);
            draw_text_centered(&mut verts, progress_stage, 0.0, 0.09, 0.95, aspect, [0.85, 0.85, 0.9, 1.0]);
            add_quad(&mut verts, -0.384, -0.054, 0.384, 0.044, [0.20, 0.25, 0.35, 1.0]);
            add_quad(&mut verts, -0.38, -0.05, 0.38, 0.04, [0.04, 0.05, 0.07, 1.0]);
            let fill_w = 0.76 * progress_val.clamp(0.0, 1.0);
            if fill_w > 0.001 { add_quad(&mut verts, -0.38, -0.05, -0.38 + fill_w, 0.04, [0.20, 0.75, 0.90, 1.0]); }
            draw_text_centered(&mut verts, &format!("{:.0}%", (progress_val * 100.0).clamp(0.0, 100.0)), 0.0, -0.10, 1.1, aspect, [1.0, 1.0, 1.0, 1.0]);
            draw_text_centered(&mut verts, "SOLID SVO GENERATOR ACTIVE", 0.0, -0.18, 0.85, aspect, [0.5, 0.8, 0.6, 0.9]);
        }
    }
    verts
}