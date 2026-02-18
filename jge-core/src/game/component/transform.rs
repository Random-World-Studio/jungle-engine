use super::node::Node;
use super::renderable::Renderable;
use super::{component, component_impl};
use crate::game::entity::Entity;
use nalgebra::{Matrix3, Matrix4, Rotation3, Translation3, Vector3};
use std::collections::HashSet;

#[component(Node, Renderable)]
#[derive(Debug, Clone)]
/// 变换组件（平移/旋转/缩放）。
///
/// `Transform` 描述实体在**局部空间（相对父 `Node`）**中的位置、旋转（欧拉角，弧度制，按 X-Y-Z 顺序）与缩放。
///
/// - 若实体没有父节点，则该变换等价于世界变换。
/// - 若父链上存在多个 `Transform`，渲染与摄像机等系统会沿 `Node` 父链做矩阵叠乘得到世界矩阵。
///
/// 渲染时通常会用（world）矩阵把 `Shape` 的局部顶点变换到世界空间。
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
/// let rt = tokio::runtime::Runtime::new()?;
/// rt.block_on(async {
///     let e = Entity::new().await?;
///     e.register_component(Transform::new()).await?;
///     if let Some(mut t) = e.get_component_mut::<Transform>().await {
///         t.set_position(Vector3::new(1.0, 2.0, 3.0));
///     }
///     Ok::<(), anyhow::Error>(())
/// })?;
/// Ok(())
/// # }
/// ```
pub struct Transform {
    entity_id: Option<Entity>,
    position: Vector3<f32>,
    rotation: Vector3<f32>,
    scale: Vector3<f32>,
}

impl Default for Transform {
    fn default() -> Self {
        Self::new()
    }
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

/// 层级变换（基于 `Node` 父链合成 world 结果）。
impl Transform {
    /// 计算实体的世界空间矩阵（沿 `Node` 父链叠乘：`world = parent_world * local`）。
    ///
    /// 规则：
    /// - 缺少 `Node`：返回 `None`（无法遍历层级）。
    /// - 缺少 `Transform`：视为单位矩阵（即只继承父级）。
    /// - 若出现父链环：返回 `None`。
    pub async fn world_matrix(entity: Entity) -> Option<Matrix4<f32>> {
        let mut chain: Vec<Matrix4<f32>> = Vec::new();
        let mut visited: HashSet<Entity> = HashSet::new();

        let mut current = Some(entity);
        while let Some(id) = current {
            if !visited.insert(id) {
                return None;
            }

            if let Some(transform_guard) = id.get_component::<Transform>().await {
                chain.push(transform_guard.matrix());
            } else {
                chain.push(Matrix4::identity());
            }

            let node_guard = id.get_component::<Node>().await?;
            current = node_guard.parent();
        }

        let mut world = Matrix4::identity();
        for local in chain.iter().rev() {
            world = world * local;
        }
        Some(world)
    }

    /// 从 world matrix 提取平移分量。
    pub fn translation_from_matrix(matrix: &Matrix4<f32>) -> Vector3<f32> {
        Vector3::new(matrix[(0, 3)], matrix[(1, 3)], matrix[(2, 3)])
    }

    /// 从 world matrix 提取相机/朝向用的正交基底（忽略缩放）。
    ///
    /// 坐标系约定与 `Camera::orientation_basis` 一致：默认朝向为 -Z。
    pub fn basis_from_matrix(matrix: &Matrix4<f32>) -> crate::game::component::camera::CameraBasis {
        let x = Vector3::new(matrix[(0, 0)], matrix[(1, 0)], matrix[(2, 0)]);
        let y = Vector3::new(matrix[(0, 1)], matrix[(1, 1)], matrix[(2, 1)]);
        let z = Vector3::new(matrix[(0, 2)], matrix[(1, 2)], matrix[(2, 2)]);

        let right = x.try_normalize(1.0e-6).unwrap_or(Vector3::x());
        let up = y.try_normalize(1.0e-6).unwrap_or(Vector3::y());
        // +Z 轴对应列向量 z；默认 forward 为 -Z。
        let forward = (-z)
            .try_normalize(1.0e-6)
            .unwrap_or(Vector3::new(0.0, 0.0, -1.0));

        crate::game::component::camera::CameraBasis { forward, up, right }
    }

    /// 将 world matrix 分解为 (position, euler_rotation_xyz, scale)。
    ///
    /// 仅支持无剪切（shear）的 TRS 矩阵；若缩放分量接近 0，会返回 `None`。
    pub fn decompose_trs(
        matrix: &Matrix4<f32>,
    ) -> Option<(Vector3<f32>, Vector3<f32>, Vector3<f32>)> {
        let position = Self::translation_from_matrix(matrix);

        let m3: Matrix3<f32> = matrix.fixed_view::<3, 3>(0, 0).into_owned();
        let col0 = m3.column(0);
        let col1 = m3.column(1);
        let col2 = m3.column(2);

        let sx = col0.norm();
        let sy = col1.norm();
        let sz = col2.norm();
        if sx <= 1.0e-8 || sy <= 1.0e-8 || sz <= 1.0e-8 {
            return None;
        }
        let scale = Vector3::new(sx, sy, sz);

        let r0 = col0 / sx;
        let r1 = col1 / sy;
        let r2 = col2 / sz;
        let rot_matrix =
            Matrix3::from_columns(&[r0.into_owned(), r1.into_owned(), r2.into_owned()]);
        let rotation = Rotation3::from_matrix_unchecked(rot_matrix);
        let (rx, ry, rz) = rotation.euler_angles();
        let euler = Vector3::new(rx, ry, rz);

        Some((position, euler, scale))
    }

    /// 将实体的“局部 Transform”设置为某个期望的 world 矩阵。
    ///
    /// - 若有父节点：`local = parent_world^{-1} * desired_world`
    /// - 若无父节点：`local = desired_world`
    ///
    /// 返回 `false` 表示无法表示（例如父矩阵不可逆、分解失败、缺失 Node/Transform）。
    pub async fn set_local_from_world_matrix(entity: Entity, desired_world: &Matrix4<f32>) -> bool {
        let Some(node_guard) = entity.get_component::<Node>().await else {
            return false;
        };
        let parent = node_guard.parent();
        drop(node_guard);

        let local = if let Some(parent) = parent {
            let Some(parent_world) = Self::world_matrix(parent).await else {
                return false;
            };
            let Some(inv_parent) = parent_world.try_inverse() else {
                return false;
            };
            inv_parent * desired_world
        } else {
            *desired_world
        };

        let Some((position, rotation, scale)) = Self::decompose_trs(&local) else {
            return false;
        };

        let Some(mut transform_guard) = entity.get_component_mut::<Transform>().await else {
            return false;
        };
        transform_guard.set_position(position);
        transform_guard.set_rotation(rotation);
        transform_guard.set_scale(scale);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{component::node::Node, entity::Entity};
    use nalgebra::{Point3, Rotation3, Translation3, Vector3, Vector4};
    use std::f32::consts::FRAC_PI_2;

    async fn detach_node(entity: Entity) {
        if entity.get_component::<Node>().await.is_some() {
            let detach_future = {
                let mut node = entity
                    .get_component_mut::<Node>()
                    .await
                    .expect("node component disappeared");
                node.detach()
            };
            let _ = detach_future.await;
        }
    }

    async fn clear_components(entity: &Entity) {
        let _ = entity.unregister_component::<Transform>().await;
        let _ = entity.unregister_component::<Renderable>().await;
        detach_node(*entity).await;
        let _ = entity.unregister_component::<Node>().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn transform_requires_node_dependency() {
        let entity = Entity::new().await.expect("应能创建实体");
        clear_components(&entity).await;

        let inserted = entity
            .register_component(Transform::new())
            .await
            .expect("缺少 Node/Renderable 时应自动注册依赖");
        assert!(inserted.is_none());

        assert!(
            entity.get_component::<Node>().await.is_some(),
            "Node 应被自动注册"
        );
        assert!(
            entity.get_component::<Renderable>().await.is_some(),
            "Renderable 应被注册"
        );

        let previous = entity
            .register_component(Transform::new())
            .await
            .expect("重复插入应返回旧的 Transform");
        assert!(previous.is_some());

        let _ = entity.unregister_component::<Transform>().await;
        let _ = entity.unregister_component::<Renderable>().await;
        let _ = entity.unregister_component::<Node>().await;
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
