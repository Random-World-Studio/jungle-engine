use anyhow::ensure;
use nalgebra::{Matrix4, Vector2, Vector3};

/// 将 Scene2D 的顶点位置转换为 NDC 坐标。
pub(in crate::game::system::render) fn scene2d_vertex_to_ndc(
    framebuffer_size: (u32, u32),
    vertex: &Vector3<f32>,
    offset: &Vector2<f32>,
    pixels_per_unit: f32,
) -> Vector3<f32> {
    let width = framebuffer_size.0.max(1) as f32;
    let height = framebuffer_size.1.max(1) as f32;
    let half_width = width * 0.5;
    let half_height = height * 0.5;

    let x_pixels = (vertex.x - offset.x) * pixels_per_unit;
    let y_pixels = (vertex.y - offset.y) * pixels_per_unit;

    let x_ndc = if half_width > 0.0 {
        x_pixels / half_width
    } else {
        0.0
    };
    let y_ndc = if half_height > 0.0 {
        y_pixels / half_height
    } else {
        0.0
    };

    Vector3::new(x_ndc, y_ndc, vertex.z)
}

/// 为纹理上传对齐 RGBA 行数据：将每行补齐到 `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`。
pub(in crate::game::system::render) fn pad_rgba_data(
    data: Vec<u8>,
    width: u32,
    height: u32,
) -> anyhow::Result<(Vec<u8>, u32)> {
    ensure!(
        width > 0 && height > 0,
        "texture dimensions must be positive"
    );

    let bytes_per_pixel = 4usize;
    let width_usize = width as usize;
    let height_usize = height as usize;
    let unpadded_bytes_per_row = width_usize * bytes_per_pixel;
    let expected_len = unpadded_bytes_per_row * height_usize;
    ensure!(
        data.len() == expected_len,
        "RGBA data length {} does not match expected {} for {}x{} texture",
        data.len(),
        expected_len,
        width,
        height,
    );

    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let padded_bytes_per_row = ((unpadded_bytes_per_row + alignment - 1) / alignment) * alignment;

    if padded_bytes_per_row == unpadded_bytes_per_row {
        return Ok((data, unpadded_bytes_per_row as u32));
    }

    let mut padded = vec![0u8; padded_bytes_per_row * height_usize];
    for row in 0..height_usize {
        let src_start = row * unpadded_bytes_per_row;
        let dst_start = row * padded_bytes_per_row;
        let src_end = src_start + unpadded_bytes_per_row;
        padded[dst_start..dst_start + unpadded_bytes_per_row]
            .copy_from_slice(&data[src_start..src_end]);
    }

    Ok((padded, padded_bytes_per_row as u32))
}

/// 将 OpenGL 风格的裁剪空间 Z（NDC Z ∈ [-1, 1]）转换为 wgpu/WebGPU 风格（NDC Z ∈ [0, 1]）。
pub(in crate::game::system::render) fn opengl_to_wgpu_matrix() -> Matrix4<f32> {
    Matrix4::new(
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0,
    )
}

pub(in crate::game::system::render) fn cast_slice_f32(data: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector4;

    #[test]
    fn pad_rgba_data_returns_error_on_mismatched_length() {
        let result = pad_rgba_data(vec![0u8; 3], 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn pad_rgba_data_pads_rows_to_alignment() {
        // width=1 => 4 bytes/row, alignment=256 => padded row 256.
        let width = 1u32;
        let height = 2u32;
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let (padded, bytes_per_row) = pad_rgba_data(data, width, height).unwrap();

        assert_eq!(
            bytes_per_row as usize,
            wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize
        );
        assert_eq!(padded.len(), bytes_per_row as usize * height as usize);

        assert_eq!(&padded[0..4], &[1, 2, 3, 4]);
        assert_eq!(
            &padded[bytes_per_row as usize..bytes_per_row as usize + 4],
            &[5, 6, 7, 8]
        );
    }

    #[test]
    fn scene2d_vertex_to_ndc_maps_pixels_to_unit_square() {
        let framebuffer = (100, 100);
        let offset = Vector2::new(0.0, 0.0);
        let pixels_per_unit = 1.0;

        let v = Vector3::new(50.0, 0.0, 0.25);
        let ndc = scene2d_vertex_to_ndc(framebuffer, &v, &offset, pixels_per_unit);
        assert!((ndc.x - 1.0).abs() < 1.0e-6);
        assert!((ndc.y - 0.0).abs() < 1.0e-6);
        assert!((ndc.z - 0.25).abs() < 1.0e-6);

        let v2 = Vector3::new(25.0, 25.0, -1.0);
        let ndc2 = scene2d_vertex_to_ndc(framebuffer, &v2, &offset, pixels_per_unit);
        assert!((ndc2.x - 0.5).abs() < 1.0e-6);
        assert!((ndc2.y - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn opengl_to_wgpu_matrix_maps_z_range() {
        let m = opengl_to_wgpu_matrix();

        let v_near = Vector4::new(0.0, 0.0, -1.0, 1.0);
        let v_far = Vector4::new(0.0, 0.0, 1.0, 1.0);

        let out_near = m * v_near;
        let out_far = m * v_far;

        assert!((out_near.z - 0.0).abs() < 1.0e-6);
        assert!((out_far.z - 1.0).abs() < 1.0e-6);
    }
}
