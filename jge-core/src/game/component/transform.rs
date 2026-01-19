use super::node::Node;
use super::renderable::Renderable;
use super::{component, component_impl};
use crate::game::entity::Entity;
use nalgebra::{Matrix4, Rotation3, Translation3, Vector3};

#[component(Node, Renderable)]
#[derive(Debug, Clone)]
/// 世界变换组件（平移/旋转/缩放）。
///
/// `Transform` 描述实体在世界空间中的位置、旋转（欧拉角，弧度制，按 X-Y-Z 顺序）与缩放。
/// 渲染时通常会用该矩阵把 `Shape` 的局部顶点变换到世界空间。
///
/// 依赖：该组件依赖 `Node` 与 `Renderable`（注册 `Transform` 时会按需自动补齐）。
///
/// # 示例
///
/// ```no_run
/// use jge_core::game::{component::transform::Transform, entity::Entity};
/// use nalgebra::Vector3;
///
/// # fn main() -> anyhow::Result<()> {
/// let e = Entity::new()?;
/// e.register_component(Transform::new())?;
/// if let Some(mut t) = e.get_component_mut::<Transform>() {
///     t.set_position(Vector3::new(1.0, 2.0, 3.0));
/// }
/// Ok(())
/// # }
/// ```
pub struct Transform {
    entity_id: Option<Entity>,
    position: Vector3<f32>,
    rotation: Vector3<f32>,
    scale: Vector3<f32>,
}

#[component_impl]
impl Transform {
    /// 创建一个默认的变换组件，位置为原点、旋转为零（弧度），缩放为单位向量。
    #[default()]
    pub fn new() -> Self {
        Self {
            entity_id: None,
            position: Vector3::zeros(),
            rotation: Vector3::zeros(),
            scale: Vector3::repeat(1.0),
        }
    }

    /// 使用指定的位置、旋转（弧度制欧拉角，按 X-Y-Z 顺序）与缩放创建组件。
    pub fn with_components(
        position: Vector3<f32>,
        rotation: Vector3<f32>,
        scale: Vector3<f32>,
    ) -> Self {
        Self {
            entity_id: None,
            position,
            rotation,
            scale,
        }
    }

    /// 当前世界坐标。
    pub fn position(&self) -> Vector3<f32> {
        self.position
    }

    /// 设置世界坐标。
    pub fn set_position(&mut self, position: Vector3<f32>) {
        self.position = position;
    }

    /// 按增量平移。
    pub fn translate(&mut self, delta: Vector3<f32>) {
        self.position += delta;
    }

    /// 当前旋转（弧度制欧拉角，按 X-Y-Z 顺序）。
    pub fn rotation(&self) -> Vector3<f32> {
        self.rotation
    }

    /// 设置欧拉角旋转（弧度制，按 X-Y-Z 顺序）。
    pub fn set_rotation(&mut self, rotation: Vector3<f32>) {
        self.rotation = rotation;
    }

    /// 按增量旋转（弧度制）。
    pub fn rotate(&mut self, delta: Vector3<f32>) {
        self.rotation += delta;
    }

    /// 当前缩放系数。
    pub fn scale(&self) -> Vector3<f32> {
        self.scale
    }

    /// 设置缩放系数。
    pub fn set_scale(&mut self, scale: Vector3<f32>) {
        self.scale = scale;
    }

    /// 按增量缩放。
    pub fn rescale(&mut self, delta: Vector3<f32>) {
        self.scale += delta;
    }

    /// 计算齐次变换矩阵：平移 × 旋转 × 缩放。
    pub fn matrix(&self) -> Matrix4<f32> {
        let translation = Translation3::from(self.position);
        let rotation =
            Rotation3::from_euler_angles(self.rotation.x, self.rotation.y, self.rotation.z);
        let scale = Matrix4::new_nonuniform_scaling(&self.scale);
        translation.to_homogeneous() * rotation.to_homogeneous() * scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{component::node::Node, entity::Entity};
    use nalgebra::{Point3, Rotation3, Translation3, Vector3, Vector4};
    use std::f32::consts::FRAC_PI_2;

    fn detach_node(entity: Entity) {
        if let Some(mut node) = entity.get_component_mut::<Node>() {
            let _ = node.detach();
        }
    }

    fn clear_components(entity: &Entity) {
        let _ = entity.unregister_component::<Transform>();
        let _ = entity.unregister_component::<Renderable>();
        detach_node(*entity);
        let _ = entity.unregister_component::<Node>();
    }

    #[test]
    fn transform_requires_node_dependency() {
        let entity = Entity::new().expect("应能创建实体");
        clear_components(&entity);

        let inserted = entity
            .register_component(Transform::new())
            .expect("缺少 Node/Renderable 时应自动注册依赖");
        assert!(inserted.is_none());

        assert!(
            entity.get_component::<Node>().is_some(),
            "Node 应被自动注册"
        );
        assert!(
            entity.get_component::<Renderable>().is_some(),
            "Renderable 应被注册"
        );

        let previous = entity
            .register_component(Transform::new())
            .expect("重复插入应返回旧的 Transform");
        assert!(previous.is_some());

        let _ = entity.unregister_component::<Transform>();
        let _ = entity.unregister_component::<Renderable>();
        let _ = entity.unregister_component::<Node>();
    }

    #[test]
    fn transform_defaults_to_identity() {
        let transform = Transform::new();
        assert_eq!(transform.position(), Vector3::zeros());
        assert_eq!(transform.rotation(), Vector3::zeros());
        assert_eq!(transform.scale(), Vector3::repeat(1.0));
    }

    #[test]
    fn transform_mutators_update_components() {
        let mut transform = Transform::new();
        transform.translate(Vector3::new(1.0, 2.0, 3.0));
        transform.rotate(Vector3::new(0.1, -0.2, 0.3));
        transform.rescale(Vector3::new(1.0, 0.0, -0.5));

        assert_eq!(transform.position(), Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(transform.rotation(), Vector3::new(0.1, -0.2, 0.3));
        assert_eq!(transform.scale(), Vector3::new(2.0, 1.0, 0.5));
    }

    #[test]
    fn transform_matrix_combines_components() {
        let transform = Transform::with_components(
            Vector3::new(3.0, -2.0, 5.0),
            Vector3::new(0.0, 0.0, FRAC_PI_2),
            Vector3::new(2.0, 1.0, 1.0),
        );
        let matrix = transform.matrix();

        // 基于显式的平移 × 旋转 × 缩放顺序计算期望值，以验证矩阵生成逻辑。
        let vector = Vector4::new(1.0, 0.0, 0.0, 1.0);
        let result = matrix * vector;

        let rotation = Rotation3::from_euler_angles(0.0, 0.0, FRAC_PI_2);
        let scaled = Vector3::new(1.0, 0.0, 0.0).component_mul(&transform.scale());
        let rotated = rotation * scaled;
        let translation = Translation3::from(transform.position());
        let expected = translation.vector + rotated;

        assert!((result.x - expected.x).abs() < 1e-5);
        assert!((result.y - expected.y).abs() < 1e-5);
        assert!((result.z - expected.z).abs() < 1e-5);
        assert!((result.w - 1.0).abs() < 1e-5);

        // 对 Point3 应用矩阵，并再次与显式计算结果对比。
        let point = Point3::new(0.0, 1.0, 0.0);
        let transformed_h = matrix * point.to_homogeneous();
        let transformed = Point3::from_homogeneous(transformed_h).expect("齐次坐标 w 不应为 0");

        let scaled_point = point.coords.component_mul(&transform.scale());
        let rotated_point = rotation * scaled_point;
        let expected_point = translation.vector + rotated_point;

        assert!((transformed.x - expected_point.x).abs() < 1e-5);
        assert!((transformed.y - expected_point.y).abs() < 1e-5);
        assert!((transformed.z - expected_point.z).abs() < 1e-5);
    }
}
