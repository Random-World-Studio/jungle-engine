use super::{component, component_impl, transform::Transform};
use crate::game::entity::Entity;

#[component(Transform)]
#[derive(Debug, Clone)]
pub struct Light {
    entity_id: Option<Entity>,
    lightness: f32,
}

#[component_impl]
impl Light {
    #[default(1.0)]
    pub fn new(lightness: f32) -> Self {
        let mut instance = Self {
            entity_id: None,
            lightness: 1.0,
        };
        instance.set_lightness(lightness);
        instance
    }

    pub fn lightness(&self) -> f32 {
        self.lightness
    }

    pub fn set_lightness(&mut self, value: f32) {
        if value.is_finite() {
            self.lightness = value.max(0.0);
        }
    }
}

#[component(Light)]
#[derive(Debug, Clone)]
pub struct PointLight {
    entity_id: Option<Entity>,
    distance: f32,
}

#[component_impl]
impl PointLight {
    #[default(1.0)]
    pub fn new(distance: f32) -> Self {
        let mut instance = Self {
            entity_id: None,
            distance: 1.0,
        };
        instance.set_distance(distance);
        instance
    }

    pub fn distance(&self) -> f32 {
        self.distance
    }

    pub fn set_distance(&mut self, value: f32) {
        if value.is_finite() && value > 0.0 {
            self.distance = value;
        }
    }
}

#[component(Light)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ParallelLight {
    entity_id: Option<Entity>,
}

#[component_impl]
impl ParallelLight {
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

    #[test]
    fn light_defaults_and_clamping() {
        let entity = Entity::new().expect("应能创建实体");
        let _ = entity
            .register_component(Renderable::new())
            .expect("应能插入 Renderable");
        if let Some(mut renderable) = entity.get_component_mut::<Renderable>() {
            renderable.set_enabled(false);
        }
        let _ = entity
            .register_component(Transform::new())
            .expect("应能插入 Transform");
        let light = Light::new(1.5);
        let previous = entity
            .register_component(light)
            .expect("首次插入 Light 不应失败");
        assert!(previous.is_none());

        {
            let mut guard = entity
                .get_component_mut::<Light>()
                .expect("应能读取 Light 组件");
            guard.set_lightness(-5.0);
        }
        let guard = entity
            .get_component::<Light>()
            .expect("应能读取 Light 组件");
        assert_eq!(guard.lightness(), 0.0);
    }

    #[test]
    fn point_light_depends_on_light_and_applies_distance() {
        let entity = Entity::new().expect("应能创建实体");
        let _ = entity
            .register_component(Renderable::new())
            .expect("应能插入 Renderable");
        if let Some(mut renderable) = entity.get_component_mut::<Renderable>() {
            renderable.set_enabled(false);
        }
        let _ = entity
            .register_component(Transform::new())
            .expect("应能插入 Transform");
        if let Some(mut transform) = entity.get_component_mut::<Transform>() {
            transform.set_position(Vector3::new(1.0, 2.0, 0.5));
        }
        let _ = entity
            .register_component(Light::new(2.0))
            .expect("应能插入 Light");
        let light = PointLight::new(5.0);
        let previous = entity
            .register_component(light)
            .expect("插入 PointLight 时依赖应满足");
        assert!(previous.is_none());

        {
            let mut guard = entity
                .get_component_mut::<PointLight>()
                .expect("应能读取 PointLight");
            guard.set_distance(-10.0);
        }

        let guard = entity
            .get_component::<PointLight>()
            .expect("应能读取 PointLight");
        assert!(guard.distance() > 0.0);
        let transform = entity
            .get_component::<Transform>()
            .expect("应能读取 Transform");
        assert_eq!(transform.position(), Vector3::new(1.0, 2.0, 0.5));
    }

    #[test]
    fn parallel_light_depends_on_light() {
        let entity = Entity::new().expect("应能创建实体");
        let _ = entity
            .register_component(Renderable::new())
            .expect("应能插入 Renderable");
        if let Some(mut renderable) = entity.get_component_mut::<Renderable>() {
            renderable.set_enabled(false);
        }
        let _ = entity
            .register_component(Transform::new())
            .expect("应能插入 Transform");
        let _ = entity
            .register_component(Light::new(0.75))
            .expect("应能插入 Light");
        let previous = entity
            .register_component(ParallelLight::new())
            .expect("插入 ParallelLight 时依赖应满足");
        assert!(previous.is_none());
        assert!(entity.get_component::<ParallelLight>().is_some());
    }
}
