use tokio::runtime::Runtime;

use crate::game::{
    component::{
        ComponentRead,
        camera::Camera,
        layer::{Layer, LayerTraversalError},
        transform::Transform,
    },
    entity::Entity,
};

pub(in crate::game::system::render) fn try_get_transform(
    runtime: &Runtime,
    entity: Entity,
) -> Option<ComponentRead<Transform>> {
    runtime.block_on(entity.get_component::<Transform>())
}

pub(in crate::game::system::render) fn select_scene3d_camera(
    runtime: &Runtime,
    root: Entity,
    preferred: Option<Entity>,
) -> Result<Option<Entity>, LayerTraversalError> {
    if let Some(candidate) = preferred
        && runtime
            .block_on(candidate.get_component::<Camera>())
            .is_some()
        && try_get_transform(runtime, candidate).is_some()
    {
        return Ok(Some(candidate));
    }

    let ordered = runtime.block_on(Layer::renderable_entities(root))?;
    for entity in ordered {
        if runtime.block_on(entity.get_component::<Camera>()).is_some()
            && try_get_transform(runtime, entity).is_some()
        {
            return Ok(Some(entity));
        }
    }
    Ok(None)
}
