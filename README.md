# Jungle Engine

## Component Workflow Policy

All component attachment and lookup code must flow through the `Entity` API. This keeps lifecycle hooks (`register_dependencies`, shader preloads, etc.) consistent and lets components stay in sync with the scene graph.

### Registering Components

1. Use `entity.register_component(T::new())` (or any other constructor) for every attachment.
2. Never call `Component::insert`, the storage `Component::storage`, or any static helper on component types directly.
3. If a component depends on other primitives (for example `Scene3D` requires `Layer`, `Transform`, and `Renderable`), declare them through `#[component(DependencyA, DependencyB, …)]` or override `register_dependencies`. The macro-generated code now uses `entity.get_component()` to probe existing attachments before inserting defaults.

### Reading Components

1. To read immutable state, call `entity.get_component::<T>()`.
2. For mutable access, call `entity.get_component_mut::<T>()`.
3. Avoid `Component::read`/`Component::write` anywhere outside of `Entity` itself. Direct storage access skips dependency bookkeeping and may desynchronize cached transforms.

### Example

```rust
fn ensure_scene3d(entity: Entity) -> anyhow::Result<()> {
	if entity.get_component::<Scene3D>().is_none() {
		entity.register_component(Scene3D::new())?;
	}

	// Safe mutable access via Entity API.
	if let Some(mut layer) = entity.get_component_mut::<Layer>() {
		layer.attach_shader_from_path(RenderPipelineStage::Vertex, ShaderLanguage::Wgsl, "shaders/3d.vs".into())?;
	}

	Ok(())
}
```

Following this pattern ensures new systems automatically benefit from dependency hooks, macro-generated defaults, and future storage improvements.
