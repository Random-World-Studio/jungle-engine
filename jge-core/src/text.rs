//! 文本渲染（字体栅格化 → 纹理资源）。
//!
//! 本模块用于把一段 UTF-8 文本按指定字体栅格化为一张 PNG 纹理，并以 [`ResourceHandle`](crate::resource::ResourceHandle)
//! 的形式返回，便于直接挂载到 [`Material`](crate::game::component::material::Material) 上被 `Shape` 渲染。
//!
//! 当前实现目标偏“引擎内生成贴图”：
//! - 输出为 PNG bytes（由渲染路径通过 `image` 解码并上传 GPU 纹理）
//! - 支持从系统字体或资源字体文件（ttf/otf/ttc）加载
//! - 支持按颜色与偏移绘制阴影

use std::{collections::HashMap, io::Cursor, sync::Arc};

use anyhow::{Context, anyhow};
use image::{DynamicImage, ImageFormat, RgbaImage};

use crate::resource::{Resource, ResourceHandle};

/// 字体来源。
///
/// - [`Font::System`]：通过系统字体名称查找字体。
/// - [`Font::Resource`]：使用资源句柄指向的字体文件（ttf/otf/ttc），并用名称从字体集里选中具体字体。
#[derive(Debug, Clone)]
pub enum Font {
    /// 按“系统字体名称”查找字体。
    ///
    /// 该名称通常是字体家族名（family name），例如："DejaVu Sans"。
    System(String),

    /// 从资源句柄读取字体文件字节，并在字体文件（可能是 ttc 字体集）中按名称选择字体。
    ///
    /// 第二个字段为字体名称（family/full name），用于在字体集中挑选正确 face。
    Resource(ResourceHandle, String),
}

/// 文本阴影。
///
/// 阴影采用“再次绘制同样字形”的方式实现：
/// - 先绘制阴影，再绘制正文
/// - `offset_px` 的坐标系与输出纹理一致：x+ 向右、y+ 向下
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextShadow {
    pub offset_px: (i32, i32),
    pub color: [u8; 4],
}

/// 文本渲染配置。
#[derive(Debug, Clone)]
pub struct TextRenderOptions {
    /// 字号（像素）。
    pub font_size_px: f32,

    /// 前景色（RGBA）。
    pub color: [u8; 4],

    /// 输出纹理的像素内边距。
    pub padding_px: u32,

    /// 可选的阴影。
    pub shadow: Option<TextShadow>,
}

impl Default for TextRenderOptions {
    fn default() -> Self {
        Self {
            font_size_px: 32.0,
            color: [255, 255, 255, 255],
            padding_px: 2,
            shadow: None,
        }
    }
}

/// 文本贴图（包含输出纹理与尺寸）。
#[derive(Debug, Clone)]
pub struct TextTexture {
    pub resource: ResourceHandle,
    pub width: u32,
    pub height: u32,
}

/// 把文本栅格化为一张 PNG 纹理资源（`ResourceHandle`）。
///
/// 该返回值可直接用于 [`crate::game::component::material::Material`] 的 `resource` 字段；
/// 渲染路径会按“普通图片”解码并创建 `wgpu::Texture`。
///
/// 注意：若要把整张纹理映射到一个矩形 `Shape` 上，你仍需为对应三角形设置 UV（`MaterialPatch`）。
pub fn render_text_to_material_resource(
    text: &str,
    font: &Font,
    options: TextRenderOptions,
) -> anyhow::Result<ResourceHandle> {
    Ok(render_text_to_material_texture(text, font, options)?.resource)
}

/// 把文本栅格化为一张 PNG 纹理资源，并返回纹理尺寸（像素）。
pub fn render_text_to_material_texture(
    text: &str,
    font: &Font,
    options: TextRenderOptions,
) -> anyhow::Result<TextTexture> {
    let fontdue = load_fontdue_font(font)?;

    let (width, height, rgba) = rasterize_text_rgba(text, &fontdue, &options)?;
    let image = RgbaImage::from_raw(width, height, rgba)
        .ok_or_else(|| anyhow!("failed to create rgba image buffer"))?;

    let mut bytes = Vec::new();
    DynamicImage::ImageRgba8(image)
        .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
        .context("failed to encode text texture as png")?;

    Ok(TextTexture {
        resource: Resource::from_memory(bytes),
        width,
        height,
    })
}

fn rasterize_text_rgba(
    text: &str,
    font: &fontdue::Font,
    options: &TextRenderOptions,
) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    use fontdue::layout::{CoordinateSystem, Layout, LayoutSettings, TextStyle};

    // 空文本也返回一个最小纹理，避免 0x0 导致下游创建纹理失败。
    if text.is_empty() {
        return Ok((1, 1, vec![0, 0, 0, 0]));
    }

    let mut layout = Layout::new(CoordinateSystem::PositiveYDown);
    layout.reset(&LayoutSettings {
        x: 0.0,
        y: 0.0,
        ..LayoutSettings::default()
    });

    let fonts: [&fontdue::Font; 1] = [font];
    layout.append(&fonts, &TextStyle::new(text, options.font_size_px, 0));

    let padding = options.padding_px as i32;

    // 计算包围盒（layout glyph 的坐标系：PositiveYDown，x/y 是左上角）。
    let mut min_x = 0.0f32;
    let mut min_y = 0.0f32;
    let mut max_x = 0.0f32;
    let mut max_y = 0.0f32;

    let shadow = options.shadow;
    let shadow_dx = shadow.map(|s| s.offset_px.0).unwrap_or(0) as f32;
    let shadow_dy = shadow.map(|s| s.offset_px.1).unwrap_or(0) as f32;

    for g in layout.glyphs() {
        let x0 = g.x;
        let y0 = g.y;
        let x1 = g.x + g.width as f32;
        let y1 = g.y + g.height as f32;
        min_x = min_x.min(x0);
        min_y = min_y.min(y0);
        max_x = max_x.max(x1);
        max_y = max_y.max(y1);

        if shadow.is_some() {
            min_x = min_x.min(x0 + shadow_dx);
            min_y = min_y.min(y0 + shadow_dy);
            max_x = max_x.max(x1 + shadow_dx);
            max_y = max_y.max(y1 + shadow_dy);
        }
    }

    let content_width = (max_x - min_x).ceil().max(1.0) as i32;
    let content_height = (max_y - min_y).ceil().max(1.0) as i32;

    let width = (content_width + padding * 2).max(1) as u32;
    let height = (content_height + padding * 2).max(1) as u32;

    let origin_x = (-min_x).round() as i32 + padding;
    let origin_y = (-min_y).round() as i32 + padding;

    let mut rgba = vec![0u8; width as usize * height as usize * 4];

    let mut target = DrawRgbaTarget {
        rgba: &mut rgba,
        width,
        height,
    };

    // 先绘制阴影，再绘制正文。
    if let Some(shadow) = options.shadow {
        draw_layout_pass(
            &layout,
            font,
            &mut target,
            (origin_x + shadow.offset_px.0, origin_y + shadow.offset_px.1),
            shadow.color,
        );
    }

    draw_layout_pass(
        &layout,
        font,
        &mut target,
        (origin_x, origin_y),
        options.color,
    );

    Ok((width, height, rgba))
}

struct DrawRgbaTarget<'a> {
    rgba: &'a mut [u8],
    width: u32,
    height: u32,
}

fn draw_layout_pass(
    layout: &fontdue::layout::Layout,
    font: &fontdue::Font,
    target: &mut DrawRgbaTarget<'_>,
    (origin_x, origin_y): (i32, i32),
    color: [u8; 4],
) {
    for g in layout.glyphs() {
        if g.width == 0 || g.height == 0 {
            continue;
        }

        let (metrics, bitmap) = font.rasterize_config(g.key);
        if metrics.width == 0 || metrics.height == 0 {
            continue;
        }

        let start_x = g.x.round() as i32 + origin_x;
        let start_y = g.y.round() as i32 + origin_y;

        for y in 0..metrics.height as i32 {
            let dst_y = start_y + y;
            if dst_y < 0 || dst_y >= target.height as i32 {
                continue;
            }

            for x in 0..metrics.width as i32 {
                let dst_x = start_x + x;
                if dst_x < 0 || dst_x >= target.width as i32 {
                    continue;
                }

                let src_alpha = bitmap[(y as usize) * metrics.width + (x as usize)];
                if src_alpha == 0 {
                    continue;
                }

                // 将 glyph alpha 与输入 alpha 相乘。
                let src_a = (src_alpha as u32 * color[3] as u32 / 255) as u8;
                if src_a == 0 {
                    continue;
                }

                let idx = ((dst_y as u32 * target.width + dst_x as u32) as usize) * 4;
                let dst_r = target.rgba[idx] as u32;
                let dst_g = target.rgba[idx + 1] as u32;
                let dst_b = target.rgba[idx + 2] as u32;
                let dst_a = target.rgba[idx + 3] as u32;

                let sa = src_a as u32;
                let inv = 255u32.saturating_sub(sa);

                let out_a = sa + dst_a * inv / 255;
                let out_r = (color[0] as u32 * sa + dst_r * inv) / 255;
                let out_g = (color[1] as u32 * sa + dst_g * inv) / 255;
                let out_b = (color[2] as u32 * sa + dst_b * inv) / 255;

                target.rgba[idx] = out_r as u8;
                target.rgba[idx + 1] = out_g as u8;
                target.rgba[idx + 2] = out_b as u8;
                target.rgba[idx + 3] = out_a as u8;
            }
        }
    }
}

fn load_fontdue_font(font: &Font) -> anyhow::Result<fontdue::Font> {
    let (bytes, collection_index): (Arc<[u8]>, u32) = match font {
        Font::System(name) => load_system_font_bytes(name)?,
        Font::Resource(handle, face_name) => {
            let bytes = load_bytes_from_resource(handle)?;
            let index = find_face_index_in_bytes(bytes.as_ref(), face_name)?;
            (bytes, index)
        }
    };

    let settings = fontdue::FontSettings {
        collection_index,
        ..fontdue::FontSettings::default()
    };

    fontdue::Font::from_bytes(bytes.as_ref(), settings)
        .map_err(|err| anyhow!(err))
        .context("failed to parse font bytes")
}

fn load_bytes_from_resource(handle: &ResourceHandle) -> anyhow::Result<Arc<[u8]>> {
    let bytes: Arc<[u8]> = {
        let mut guard = handle.write();
        if guard.data_loaded() {
            guard
                .try_get_data_arc()
                .ok_or_else(|| anyhow!("resource reports cached but missing data"))?
        } else {
            guard.get_data_arc()
        }
    };
    Ok(bytes)
}

struct SystemFontCache {
    db: fontdb::Database,
    by_name: HashMap<String, (Arc<[u8]>, u32)>,
}

fn system_font_cache() -> &'static crate::sync::Mutex<SystemFontCache> {
    use std::sync::OnceLock;

    static CACHE: OnceLock<crate::sync::Mutex<SystemFontCache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut db = fontdb::Database::new();
        db.load_system_fonts();
        crate::sync::Mutex::new(SystemFontCache {
            db,
            by_name: HashMap::new(),
        })
    })
}

fn load_system_font_bytes(name: &str) -> anyhow::Result<(Arc<[u8]>, u32)> {
    use fontdb::{Family, Query};

    let cache_mutex = system_font_cache();
    let mut cache = cache_mutex.lock();

    if let Some((bytes, face_index)) = cache.by_name.get(name) {
        return Ok((bytes.clone(), *face_index));
    }

    let query = Query {
        families: &[Family::Name(name)],
        ..Query::default()
    };

    let id = cache
        .db
        .query(&query)
        .ok_or_else(|| anyhow!("system font not found: {name}"))?;

    let mut out_bytes: Option<Arc<[u8]>> = None;
    let mut out_index: u32 = 0;

    cache.db.with_face_data(id, |data, face_index| {
        out_bytes = Some(Arc::<[u8]>::from(data.to_vec()));
        out_index = face_index;
    });

    let bytes = out_bytes.ok_or_else(|| anyhow!("failed to read system font data: {name}"))?;

    cache
        .by_name
        .insert(name.to_string(), (bytes.clone(), out_index));

    Ok((bytes, out_index))
}

fn find_face_index_in_bytes(bytes: &[u8], face_name: &str) -> anyhow::Result<u32> {
    // ttf-parser 支持 TTC collection。
    let face_count = ttf_parser::fonts_in_collection(bytes).unwrap_or(1);

    let mut available = Vec::new();

    for index in 0..face_count {
        let face = ttf_parser::Face::parse(bytes, index)
            .map_err(|_| anyhow!("failed to parse font face at index {index}"))?;

        let mut matched = false;
        for name in face.names() {
            if let Some(s) = name.to_string()
                && !s.is_empty()
            {
                let is_match = s.eq_ignore_ascii_case(face_name);

                // 收集可用名称，便于排错。
                if available.len() < 64 {
                    available.push(s.clone());
                }

                if is_match {
                    matched = true;
                    break;
                }
            }
        }

        if matched {
            return Ok(index);
        }
    }

    Err(anyhow!(
        "font face not found in font bytes: {face_name} (available names sample: {:?})",
        available
    ))
}
