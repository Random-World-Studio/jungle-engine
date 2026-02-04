use super::{component, component_impl, transform::Transform};
use crate::game::entity::Entity;

/// 通用光照强度（基础光组件）。
///
/// 该组件本身不定义光源类型；它与 [`PointLight`] / [`ParallelLight`] 组合使用：
/// - `Light` 负责强度（`lightness`）
/// - 具体光源组件负责几何/范围
///
/// `lightness` 会被限制为非负有限数。
///
/// # 示例
///
/// ```no_run
/// use jge_core::game::{component::light::{Light, PointLight}, entity::Entity};
///
/// # fn main() -> anyhow::Result<()> {
/// let rt = tokio::runtime::Runtime::new()?;
/// rt.block_on(async {
///     let e = Entity::new().await?;
///     e.register_component(Light::new(2.0)).await?;
///     e.register_component(PointLight::new(10.0)).await?;
///     Ok::<(), anyhow::Error>(())
/// })?;
/// Ok(())
/// # }
/// ```
#[component(Transform)]
#[derive(Debug, Clone)]
pub struct Light {
    entity_id: Option<Entity>,
    lightness: f32,
}

#[component_impl]
impl Light {
    /// 创建基础光组件。
    ///
    /// `lightness` 会被 clamp 到 $[0, +\infty)$。
    #[default(1.0)]
    pub fn new(lightness: f32) -> Self {
        let mut instance = Self {
            entity_id: None,
            lightness: 1.0,
        };
        instance.set_lightness(lightness);
        instance
    }

    /// 获取光照强度。
    pub fn lightness(&self) -> f32 {
        self.lightness
    }

    /// 设置光照强度。
    ///
    /// - 非有限值会被忽略。
    /// - 负数会被 clamp 到 0。
    pub fn set_lightness(&mut self, value: f32) {
        if value.is_finite() {
            self.lightness = value.max(0.0);
        }
    }
}

/// 点光源（位置光）。
///
/// 依赖：必须与 [`Light`] 一起使用。
///
/// 光源位置来自实体的 [`Transform`]。
///
/// 通常会与 [`Light`] 一起挂载到同一个实体上。
#[component(Light)]
#[derive(Debug, Clone)]
pub struct PointLight {
    entity_id: Option<Entity>,
    distance: f32,
}

#[component_impl]
impl PointLight {
    /// 创建点光源。
    ///
    /// `distance` 必须为有限正数，否则会回落到默认值。
    #[default(1.0)]
    pub fn new(distance: f32) -> Self {
        let mut instance = Self {
            entity_id: None,
            distance: 1.0,
        };
        instance.set_distance(distance);
        instance
    }

    /// 获取点光源影响距离（范围）。
    pub fn distance(&self) -> f32 {
        self.distance
    }

    /// 设置点光源影响距离（范围）。
    ///
    /// 仅接受有限正数；否则忽略。
    pub fn set_distance(&mut self, value: f32) {
        if value.is_finite() && value > 0.0 {
            self.distance = value;
        }
    }
}

/// 平行光（方向光）。
///
/// 依赖：必须与 [`Light`] 一起使用。
///
/// 光照方向通常由实体的 [`Transform`] 旋转确定（具体解释由渲染路径定义）。
///
/// 常见用法：与 [`Light`] 一起挂载到实体上作为“方向光”。
#[component(Light)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ParallelLight {
    entity_id: Option<Entity>,
}

#[component_impl]
impl ParallelLight {
    /// 创建平行光组件。
    #[default()]
    pub fn new() -> Self {
        Self { entity_id: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{
        component::{renderable::Renderable, transform::Transform},
        entity::Entity,
    };
    use nalgebra::Vector3;

    #[tokio::test(flavor = "multi_thread")]
    async fn light_defaults_and_clamping() {
        let entity = Entity::new().await.expect("应能创建实体");
        let _ = entity
            .register_component(Renderable::new())
            .await
            .expect("应能插入 Renderable");
        if let Some(mut renderable) = entity.get_component_mut::<Renderable>().await {
            renderable.set_enabled(false);
        }
        let _ = entity
            .register_component(Transform::new())
            .await
            .expect("应能插入 Transform");
        let light = Light::new(1.5);
        let previous = entity
            .register_component(light)
            .await
            .expect("首次插入 Light 不应失败");
        assert!(previous.is_none());

        {
            let mut guard = entity
                .get_component_mut::<Light>()
                .await
                .expect("应能读取 Light 组件");
            guard.set_lightness(-5.0);
        }
        let guard = entity
            .get_component::<Light>()
            .await
            .expect("应能读取 Light 组件");
        assert_eq!(guard.lightness(), 0.0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn point_light_depends_on_light_and_applies_distance() {
        let entity = Entity::new().await.expect("应能创建实体");
        let _ = entity
            .register_component(Renderable::new())
            .await
            .expect("应能插入 Renderable");
        if let Some(mut renderable) = entity.get_component_mut::<Renderable>().await {
            renderable.set_enabled(false);
        }
        let _ = entity
            .register_component(Transform::new())
            .await
            .expect("应能插入 Transform");
        if let Some(mut transform) = entity.get_component_mut::<Transform>().await {
            transform.set_position(Vector3::new(1.0, 2.0, 0.5));
        }
        let _ = entity
            .register_component(Light::new(2.0))
            .await
            .expect("应能插入 Light");
        let light = PointLight::new(5.0);
        let previous = entity
            .register_component(light)
            .await
            .expect("插入 PointLight 时依赖应满足");
        assert!(previous.is_none());

        {
            let mut guard = entity
                .get_component_mut::<PointLight>()
                .await
                .expect("应能读取 PointLight");
            guard.set_distance(-10.0);
        }

        let guard = entity
            .get_component::<PointLight>()
            .await
            .expect("应能读取 PointLight");
        assert!(guard.distance() > 0.0);
        let transform = entity
            .get_component::<Transform>()
            .await
            .expect("应能读取 Transform");
        assert_eq!(transform.position(), Vector3::new(1.0, 2.0, 0.5));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn parallel_light_depends_on_light() {
        let entity = Entity::new().await.expect("应能创建实体");
        let _ = entity
            .register_component(Renderable::new())
            .await
            .expect("应能插入 Renderable");
        if let Some(mut renderable) = entity.get_component_mut::<Renderable>().await {
            renderable.set_enabled(false);
        }
        let _ = entity
            .register_component(Transform::new())
            .await
            .expect("应能插入 Transform");
        let _ = entity
            .register_component(Light::new(0.75))
            .await
            .expect("应能插入 Light");
        let previous = entity
            .register_component(ParallelLight::new())
            .await
            .expect("插入 ParallelLight 时依赖应满足");
        assert!(previous.is_none());
        assert!(entity.get_component::<ParallelLight>().await.is_some());
    }
}
