use super::transform::Transform;
use super::{component, component_impl};
use crate::Aabb3;
use crate::game::entity::Entity;
use nalgebra::{Matrix4, Vector3, Vector4};

/// AABB 裁剪边界组件。
///
/// 挂载该组件的实体会在渲染时形成一个“裁剪盒”，该实体及其整棵子树的渲染结果都会被裁剪到该 AABB 内。
///
/// 备注：引擎的 Scene2D 也处在 3D 世界空间中，因此该组件仅使用 [`Aabb3`]。
///
/// 坐标语义：
/// - `local_bounds` 定义在该实体的局部空间。
/// - 渲染侧会结合实体的 world matrix，将其转换成 world-space 的轴对齐包围盒。
#[component(Transform)]
#[derive(Debug, Clone, Copy)]
pub struct AabbBorder {
    entity_id: Option<Entity>,
    local_bounds: Aabb3,
}

impl Default for AabbBorder {
    fn default() -> Self {
        Self::new(Aabb3::default())
    }
}

#[component_impl]
impl AabbBorder {
    /// 使用局部空间 AABB 创建裁剪边界。
    #[default(Aabb3::default())]
    pub fn new(local_bounds: Aabb3) -> Self {
        Self {
            entity_id: None,
            local_bounds,
        }
    }

    pub fn local_bounds(&self) -> Aabb3 {
        self.local_bounds
    }

    pub fn set_local_bounds(&mut self, bounds: Aabb3) {
        self.local_bounds = bounds;
    }

    /// 计算该裁剪边界在 world-space 下的轴对齐 AABB。
    ///
    /// 注意：即使实体存在旋转/缩放，该函数仍返回 world-space 的轴对齐包围盒（8 个角点变换后再取 min/max）。
    pub async fn world_bounds(entity: Entity) -> Option<Aabb3> {
        let border = entity.get_component::<AabbBorder>().await?;
        let local = border.local_bounds;
        drop(border);

        let world = Transform::world_matrix(entity).await?;
        Some(transform_aabb3(&world, &local))
    }
}

fn transform_aabb3(world: &Matrix4<f32>, aabb: &Aabb3) -> Aabb3 {
    let corners = [
        Vector3::new(aabb.min.x, aabb.min.y, aabb.min.z),
        Vector3::new(aabb.min.x, aabb.min.y, aabb.max.z),
        Vector3::new(aabb.min.x, aabb.max.y, aabb.min.z),
        Vector3::new(aabb.min.x, aabb.max.y, aabb.max.z),
        Vector3::new(aabb.max.x, aabb.min.y, aabb.min.z),
        Vector3::new(aabb.max.x, aabb.min.y, aabb.max.z),
        Vector3::new(aabb.max.x, aabb.max.y, aabb.min.z),
        Vector3::new(aabb.max.x, aabb.max.y, aabb.max.z),
    ];

    let mut min = Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

    for c in corners {
        let v = world * Vector4::new(c.x, c.y, c.z, 1.0);
        min.x = min.x.min(v.x);
        min.y = min.y.min(v.y);
        min.z = min.z.min(v.z);
        max.x = max.x.max(v.x);
        max.y = max.y.max(v.y);
        max.z = max.z.max(v.z);
    }

    Aabb3::new(min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::component::node::Node;
    use crate::game::component::transform::Transform;
    use crate::game::entity::Entity;
    use nalgebra::{Rotation3, Translation3};

    fn transform_corners(world: &Matrix4<f32>, aabb: &Aabb3) -> Aabb3 {
        let corners = [
            Vector3::new(aabb.min.x, aabb.min.y, aabb.min.z),
            Vector3::new(aabb.min.x, aabb.min.y, aabb.max.z),
            Vector3::new(aabb.min.x, aabb.max.y, aabb.min.z),
            Vector3::new(aabb.min.x, aabb.max.y, aabb.max.z),
            Vector3::new(aabb.max.x, aabb.min.y, aabb.min.z),
            Vector3::new(aabb.max.x, aabb.min.y, aabb.max.z),
            Vector3::new(aabb.max.x, aabb.max.y, aabb.min.z),
            Vector3::new(aabb.max.x, aabb.max.y, aabb.max.z),
        ];

        let mut min = Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        for c in corners {
            let v = world * Vector4::new(c.x, c.y, c.z, 1.0);
            min.x = min.x.min(v.x);
            min.y = min.y.min(v.y);
            min.z = min.z.min(v.z);
            max.x = max.x.max(v.x);
            max.y = max.y.max(v.y);
            max.z = max.z.max(v.z);
        }

        Aabb3::new(min, max)
    }

    fn assert_aabb_close(a: &Aabb3, b: &Aabb3) {
        let eps = 1.0e-5;
        assert!(
            (a.min.x - b.min.x).abs() <= eps,
            "min.x: {} vs {}",
            a.min.x,
            b.min.x
        );
        assert!(
            (a.min.y - b.min.y).abs() <= eps,
            "min.y: {} vs {}",
            a.min.y,
            b.min.y
        );
        assert!(
            (a.min.z - b.min.z).abs() <= eps,
            "min.z: {} vs {}",
            a.min.z,
            b.min.z
        );
        assert!(
            (a.max.x - b.max.x).abs() <= eps,
            "max.x: {} vs {}",
            a.max.x,
            b.max.x
        );
        assert!(
            (a.max.y - b.max.y).abs() <= eps,
            "max.y: {} vs {}",
            a.max.y,
            b.max.y
        );
        assert!(
            (a.max.z - b.max.z).abs() <= eps,
            "max.z: {} vs {}",
            a.max.z,
            b.max.z
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn world_bounds_applies_transform_matrix() {
        let entity = Entity::new().await.expect("应能创建实体");

        let local = Aabb3::new(Vector3::new(-1.0, -2.0, -3.0), Vector3::new(4.0, 5.0, 6.0));
        let _ = entity
            .register_component(AabbBorder::new(local))
            .await
            .expect("应能注册 AabbBorder");

        let position = Vector3::new(3.0, -1.5, 2.25);
        let rotation = Vector3::new(0.3, -0.8, 0.15);
        let scale = Vector3::new(2.0, 0.5, 1.25);
        if let Some(mut t) = entity.get_component_mut::<Transform>().await {
            t.set_position(position);
            t.set_rotation(rotation);
            t.set_scale(scale);
        } else {
            panic!("AabbBorder 依赖 Transform，应自动注册");
        }

        let got = AabbBorder::world_bounds(entity)
            .await
            .expect("应能计算 world_bounds");

        let expected_world = {
            let translation = Translation3::from(position);
            let rotation = Rotation3::from_euler_angles(rotation.x, rotation.y, rotation.z);
            let scale = Matrix4::new_nonuniform_scaling(&scale);
            translation.to_homogeneous() * rotation.to_homogeneous() * scale
        };
        let expected = transform_corners(&expected_world, &local);

        assert_aabb_close(&got, &expected);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn world_bounds_includes_parent_chain_transforms() {
        let parent = Entity::new().await.expect("应能创建父实体");
        let child = Entity::new().await.expect("应能创建子实体");

        // 挂到父节点下形成层级。
        let attach_future = {
            let mut parent_node = parent
                .get_component_mut::<Node>()
                .await
                .expect("父实体应有 Node");
            parent_node.attach(child)
        };
        attach_future.await.expect("应能 attach child");

        // 父/子 Transform
        let _ = parent
            .register_component(Transform::new())
            .await
            .expect("应能注册 parent Transform");
        let _ = child
            .register_component(Transform::new())
            .await
            .expect("应能注册 child Transform");

        let parent_pos = Vector3::new(10.0, 0.0, -2.0);
        let child_pos = Vector3::new(-1.0, 2.0, 3.0);
        if let Some(mut t) = parent.get_component_mut::<Transform>().await {
            t.set_position(parent_pos);
        }
        if let Some(mut t) = child.get_component_mut::<Transform>().await {
            t.set_position(child_pos);
        }

        let local = Aabb3::new(Vector3::new(-1.0, -1.0, -1.0), Vector3::new(1.0, 1.0, 1.0));
        let _ = child
            .register_component(AabbBorder::new(local))
            .await
            .expect("应能注册 child AabbBorder");

        let got = AabbBorder::world_bounds(child)
            .await
            .expect("应能计算 world_bounds");

        let parent_world = Translation3::from(parent_pos).to_homogeneous();
        let child_world = Translation3::from(child_pos).to_homogeneous();
        let expected_world = parent_world * child_world;
        let expected = transform_corners(&expected_world, &local);

        assert_aabb_close(&got, &expected);
    }
}
