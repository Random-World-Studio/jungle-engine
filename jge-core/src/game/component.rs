//! # 组件工作流约定（Component Workflow Policy）
//!
//! 引擎对“组件的挂载/读取/写入”有一个强约束：**所有组件的依赖注册、挂载与查询，都必须通过 [`Entity`] API 流转**。
//! 这样可以保证生命周期钩子（例如 `register_dependencies`、着色器预热等）一致执行，并让组件能与节点树/场景缓存保持同步。
//!
//! ## 挂载组件（Registering Components）
//!
//! 1. 始终使用 `entity.register_component(T::new()).await?`（或其他构造方式）来挂载组件。
//! 2. 不要在 `Entity` 之外直接调用以下 API：
//!    - `Component::insert`
//!    - `Component::storage`（以及任何基于它的静态辅助方法）
//! 3. 若组件依赖其他基础组件（例如 `Scene3D` 依赖 `Layer`、`Transform`、`Renderable`），请通过 `#[component(DepA, DepB, ...)]`
//!    声明依赖，或在 `register_dependencies` 中显式注册。宏生成的默认实现会先用 `entity.get_component().await` 探测已有依赖，
//!    再按需插入默认值。
//!
//! ## 读取/写入组件（Reading Components）
//!
//! 1. 只读访问使用 `entity.get_component::<T>().await`。
//! 2. 可变访问使用 `entity.get_component_mut::<T>().await`。
//! 3. 避免在 `Entity` 之外使用 `Component::read`/`Component::write`：直接访问存储会绕过依赖记账，并可能导致缓存（例如变换/场景）不同步。
//!
//! ## 关于在 async 中跨 `.await` 持有 guard（重要）
//!
//! `ComponentRead`/`ComponentWrite` 是锁 guard。
//!
//! - 目前它们是 `Send`，因此**在需要 `Future: Send` 的场景**（例如 `tokio::spawn`）里也可以跨 `.await` 存活。
//! - 但依然强烈建议：**尽量缩短持锁时间**，避免在持有写 guard 时执行耗时 `.await`，否则会造成其它任务无法读写该组件。
//! - 更重要的是：**不要在持有 guard 时 `.await` 可能回到 ECS/节点树的 Future**（例如 `Node::attach/detach/set_logic`、或会触发 `GameLogic` 回调的流程），
//!   否则非常容易形成锁顺序反转导致死锁。
//!
//! 推荐模式：先构造要 await 的 Future（或先拷贝出必要数据），立即 `drop(guard)`，再 `await`。
//!
//! ## 示例
//!
//! ```rust
//! use jge_core::game::{component::scene3d::Scene3D, entity::Entity};
//! use jge_core::game::component::layer::{Layer, RenderPipelineStage, ShaderLanguage};
//!
//! async fn ensure_scene3d(entity: Entity) -> anyhow::Result<()> {
//!     if entity.get_component::<Scene3D>().await.is_none() {
//!         entity.register_component(Scene3D::new()).await?;
//!     }
//!
//!     // 通过 Entity API 安全地获取可变引用。
//!     if let Some(mut layer) = entity.get_component_mut::<Layer>().await {
//!         layer.attach_shader_from_path(
//!             RenderPipelineStage::Vertex,
//!             ShaderLanguage::Wgsl,
//!             "shaders/3d.vs".into(),
//!         )?;
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod background;
pub mod camera;
pub mod layer;
pub mod light;
pub mod material;
pub mod node;
pub mod renderable;
pub mod scene2d;
pub mod scene3d;
pub mod shape;
pub mod transform;
pub use jge_macros::{component, component_impl};

use std::{any::type_name, collections::HashMap, fmt, sync::Arc};

use async_trait::async_trait;
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use std::any::TypeId;

use crate::game::entity::{Entity, EntityId};

/// 组件 trait（所有组件的统一接口）。
///
/// 对游戏开发者来说：你通常不需要手动实现这个 trait。
///
/// - 引擎内置组件（如 `Transform`、`Layer`、`Scene3D`）都已实现。
/// - 自定义组件建议优先使用 `#[component]` / `#[component_impl]` 宏生成实现。
/// - **不要**在游戏代码里直接调用 `insert/read/write/remove/storage`。
///   统一使用 [`Entity`] 的 API（`register_component`/`get_component`/`get_component_mut`），并在调用处 `.await`。
#[async_trait]
pub trait Component: Send + Sync + Sized + 'static {
    /// 返回该组件类型的全局存储。
    ///
    /// 这是底层存储入口，一般不直接使用。
    fn storage() -> &'static ComponentStorage<Self>;

    /// 注册依赖组件。
    ///
    /// 当你通过 `Entity::register_component` 注册组件时，会先调用此函数。
    async fn register_dependencies(_entity: Entity) -> Result<(), ComponentDependencyError> {
        Ok(())
    }

    /// 注销依赖组件。
    async fn unregister_dependencies(_entity: Entity) {}

    /// 组件被挂载到实体时触发（生命周期钩子）。
    ///
    /// 该钩子是 async 的：允许组件在挂载阶段访问/初始化其它组件与资源，而无需在同步上下文里做阻塞等待。
    async fn attach_entity(&mut self, _entity: Entity) {}

    /// 组件从实体卸载时触发（生命周期钩子）。
    async fn detach_entity(&mut self) {}

    /// 将组件写入存储。
    ///
    /// 建议通过 `Entity::register_component` 间接调用。
    async fn insert(
        entity: Entity,
        component: Self,
    ) -> Result<Option<Self>, ComponentDependencyError> {
        let mut component = component;
        component.attach_entity(entity).await;
        let previous = Self::storage().insert(entity.id(), component).await;

        let Some(mut existing) = previous else {
            return Ok(None);
        };

        existing.detach_entity().await;
        Ok(Some(existing))
    }

    async fn read(entity: Entity) -> Option<ComponentRead<Self>> {
        Self::storage().get(entity.id()).await
    }

    async fn write(entity: Entity) -> Option<ComponentWrite<Self>> {
        Self::storage().get_mut(entity.id()).await
    }

    async fn remove(entity: Entity) -> Option<Self> {
        let mut component = Self::storage().remove(entity.id()).await?;
        component.detach_entity().await;
        Some(component)
    }
}

#[derive(Debug)]
/// 组件依赖不满足时的错误。
///
/// 常见于 `Entity::register_component(...)`：当某个组件声明了依赖，但依赖组件缺失且无法自动补齐时会返回此错误。
pub struct ComponentDependencyError {
    entity: Entity,
    required: &'static str,
    source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
}

impl PartialEq for ComponentDependencyError {
    fn eq(&self, other: &Self) -> bool {
        self.entity == other.entity && self.required == other.required
    }
}

impl Eq for ComponentDependencyError {}

impl ComponentDependencyError {
    /// 构造一个“缺少某依赖组件”的错误。
    pub fn new(entity: Entity, required: &'static str) -> Self {
        Self {
            entity,
            required,
            source: None,
        }
    }

    pub fn with_source<E>(entity: Entity, required: &'static str, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            entity,
            required,
            source: Some(Box::new(source)),
        }
    }

    /// 发生错误的实体。
    pub fn entity(&self) -> Entity {
        self.entity
    }

    /// 缺失的依赖组件名（类型名字符串）。
    pub fn required(&self) -> &'static str {
        self.required
    }
}

impl fmt::Display for ComponentDependencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "实体 {} 缺少依赖组件 {}",
            self.entity.id(),
            self.required
        )?;
        if let Some(source) = &self.source {
            write!(f, ": {}", source)?;
        }
        Ok(())
    }
}

impl std::error::Error for ComponentDependencyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|error| error.as_ref() as &(dyn std::error::Error + 'static))
    }
}

#[derive(Debug)]
struct SlotValue<C> {
    entity_id: EntityId,
    value: C,
}

type ComponentSlot<C> = Arc<RwLock<Option<SlotValue<C>>>>;
type ComponentEntryMap<C> = HashMap<EntityId, ComponentSlot<C>>;
pub struct ComponentStorage<C> {
    entries: RwLock<ComponentEntryMap<C>>,
}

impl<C: 'static> Default for ComponentStorage<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: 'static> ComponentStorage<C> {
    /// 创建一个空的组件存储。
    ///
    /// 通常只在组件实现中作为 `static OnceLock<ComponentStorage<T>>` 的默认值使用。
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
        }
    }

    /// 遍历所有已存在组件，并将其映射收集成一个 Vec。
    ///
    /// 这是偏底层的枚举接口，通常用于系统层做批处理/收集。
    pub async fn collect_with<T, F>(&self, mut f: F) -> Vec<T>
    where
        F: FnMut(EntityId, &C) -> Option<T>,
    {
        let slots: Vec<ComponentSlot<C>> = {
            let guard = self.entries.read().await;
            guard.values().cloned().collect()
        };
        let mut results = Vec::new();
        for slot in slots {
            let value_guard = slot.read_owned().await;
            let Some(slot_value) = value_guard.as_ref() else {
                continue;
            };
            if let Some(mapped) = f(slot_value.entity_id, &slot_value.value) {
                results.push(mapped);
            }
        }

        results
    }

    /// 插入或替换一个组件值。
    ///
    /// 建议通过 `Entity::register_component` 间接调用。
    pub async fn insert(&self, entity_id: EntityId, component: C) -> Option<C> {
        let slot = {
            let mut guard = self.entries.write().await;
            guard
                .entry(entity_id)
                .or_insert_with(|| Arc::new(RwLock::new(None)))
                .clone()
        };

        let mut value_guard = slot.write_owned().await;
        let previous = value_guard.replace(SlotValue {
            entity_id,
            value: component,
        });
        previous.map(|p| p.value)
    }

    /// 获取组件的只读 guard。
    pub async fn get(&self, entity_id: EntityId) -> Option<ComponentRead<C>> {
        let slot = {
            let guard = self.entries.read().await;
            guard.get(&entity_id).cloned()?
        };
        Some(ComponentRead::new(slot).await)
    }

    /// 获取组件的可写 guard。
    pub async fn get_mut(&self, entity_id: EntityId) -> Option<ComponentWrite<C>> {
        let slot = {
            let guard = self.entries.read().await;
            guard.get(&entity_id).cloned()?
        };
        Some(ComponentWrite::new(slot).await)
    }

    /// 移除组件并返回旧值。
    pub async fn remove(&self, entity_id: EntityId) -> Option<C> {
        let slot = {
            let mut guard = self.entries.write().await;
            guard.remove(&entity_id)?
        };

        let mut value_guard = slot.write_owned().await;
        value_guard.take().map(|p| p.value)
    }
}

/// 组件只读访问 guard。
///
/// 通过 `Entity::get_component::<T>().await` 获得。
/// 该类型实现了 `Deref<Target = T>`，因此可以像 `&T` 一样使用。
pub struct ComponentRead<C: 'static> {
    _slot: Arc<RwLock<Option<SlotValue<C>>>>,
    guard: OwnedRwLockReadGuard<Option<SlotValue<C>>>,
}

/// 组件可写访问 guard。
///
/// 通过 `Entity::get_component_mut::<T>().await` 获得。
/// 该类型实现了 `DerefMut`，因此可以直接修改组件字段/调用 setter。
pub struct ComponentWrite<C: 'static> {
    _slot: Arc<RwLock<Option<SlotValue<C>>>>,
    guard: OwnedRwLockWriteGuard<Option<SlotValue<C>>>,
    mark_render_dirty_on_drop: bool,
}

impl<C: 'static> ComponentRead<C> {
    async fn new(slot: Arc<RwLock<Option<SlotValue<C>>>>) -> Self {
        let guard = Arc::clone(&slot).read_owned().await;
        Self { _slot: slot, guard }
    }

    /// 返回拥有该组件的实体。
    pub fn entity(&self) -> Entity {
        Entity::from(
            self.guard
                .as_ref()
                .expect("组件存储出现不一致状态: 读到空槽位")
                .entity_id,
        )
    }
}

impl<C: 'static> ComponentWrite<C> {
    async fn new(slot: Arc<RwLock<Option<SlotValue<C>>>>) -> Self {
        let guard = Arc::clone(&slot).write_owned().await;
        Self {
            _slot: slot,
            guard,
            mark_render_dirty_on_drop: affects_render_snapshot::<C>(),
        }
    }

    /// 返回拥有该组件的实体。
    pub fn entity(&self) -> Entity {
        Entity::from(
            self.guard
                .as_ref()
                .expect("组件存储出现不一致状态: 写到空槽位")
                .entity_id,
        )
    }
}

impl<C: 'static> Drop for ComponentWrite<C> {
    fn drop(&mut self) {
        if self.mark_render_dirty_on_drop {
            crate::game::system::render::mark_render_snapshot_dirty_for_component::<C>();
        }
    }
}

fn affects_render_snapshot<C: 'static>() -> bool {
    let type_id = TypeId::of::<C>();
    // 这是一个“白名单”集合：只有会影响 RenderSnapshot 构建的组件写入才触发 dirty。
    // 目标是减少无意义的快照重建，同时保持语义正确。
    type_id == TypeId::of::<crate::game::component::background::Background>()
        || type_id == TypeId::of::<crate::game::component::camera::Camera>()
        || type_id == TypeId::of::<crate::game::component::layer::Layer>()
        || type_id == TypeId::of::<crate::game::component::light::Light>()
        || type_id == TypeId::of::<crate::game::component::light::PointLight>()
        || type_id == TypeId::of::<crate::game::component::light::ParallelLight>()
        || type_id == TypeId::of::<crate::game::component::material::Material>()
        || type_id == TypeId::of::<crate::game::component::node::Node>()
        || type_id == TypeId::of::<crate::game::component::renderable::Renderable>()
        || type_id == TypeId::of::<crate::game::component::scene2d::Scene2D>()
        || type_id == TypeId::of::<crate::game::component::scene3d::Scene3D>()
        || type_id == TypeId::of::<crate::game::component::shape::Shape>()
        || type_id == TypeId::of::<crate::game::component::transform::Transform>()
}

impl<C: 'static> std::ops::Deref for ComponentRead<C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        &self
            .guard
            .as_ref()
            .expect("组件存储出现不一致状态: 读到空槽位")
            .value
    }
}

impl<C: 'static> std::ops::Deref for ComponentWrite<C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        &self
            .guard
            .as_ref()
            .expect("组件存储出现不一致状态: 写到空槽位")
            .value
    }
}

impl<C: 'static> std::ops::DerefMut for ComponentWrite<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self
            .guard
            .as_mut()
            .expect("组件存储出现不一致状态: 写到空槽位")
            .value
    }
}

pub async fn require_component<C: Component>(
    entity: Entity,
) -> Result<(), ComponentDependencyError> {
    if entity.get_component::<C>().await.is_some() {
        Ok(())
    } else {
        Err(ComponentDependencyError::new(entity, type_name::<C>()))
    }
}

#[cfg(test)]
mod tests {
    use super::{component, component_impl, *};
    use std::error::Error as StdError;
    use std::io::{Error as IoError, ErrorKind};

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[component]
    #[derive(Debug, PartialEq)]
    struct TestComponent {
        value: i32,
    }

    #[component_impl(TestComponent)]
    impl TestComponent {
        #[default(0)]
        fn new(value: i32) -> Self {
            Self { value }
        }
    }

    #[component]
    #[derive(Debug, PartialEq)]
    struct AlternateDefaultComponent {
        created_by: &'static str,
    }

    #[component_impl(AlternateDefaultComponent)]
    impl AlternateDefaultComponent {
        #[default()]
        fn build() -> Self {
            Self {
                created_by: "build",
            }
        }
    }

    #[component]
    #[derive(Debug)]
    struct FailingDefaultComponent;

    #[component_impl(FailingDefaultComponent)]
    impl FailingDefaultComponent {
        #[default()]
        fn create() -> Result<Self, IoError> {
            Err(IoError::other("default failure"))
        }
    }

    #[test]
    fn component_guards_are_send_and_sync_when_component_is_threadsafe() {
        assert_send::<TestComponent>();
        assert_sync::<TestComponent>();
        assert_send::<ComponentRead<TestComponent>>();
        assert_send::<ComponentWrite<TestComponent>>();
    }

    fn assert_future_send<F: std::future::Future + Send>(future: F) -> F {
        future
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn component_write_can_cross_await_in_send_future() {
        let entity = Entity::new().await.expect("应能创建实体");
        entity
            .register_component(TestComponent { value: 0 })
            .await
            .expect("应能注册测试组件");

        let fut = async move {
            let mut guard = entity
                .get_component_mut::<TestComponent>()
                .await
                .expect("应能获取写 guard");
            tokio::task::yield_now().await;
            guard.value += 1;
        };

        let fut = assert_future_send(fut);
        fut.await;

        let guard = entity
            .get_component::<TestComponent>()
            .await
            .expect("应能获取读 guard");
        assert_eq!(guard.value, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_and_read_component() {
        let entity = Entity::new().await.expect("应能创建实体");

        assert!(
            entity
                .register_component(TestComponent::new(5))
                .await
                .expect("首次插入组件不应触发依赖错误")
                .is_none()
        );

        let component = entity
            .get_component::<TestComponent>()
            .await
            .expect("应当能读取到刚插入的组件");
        assert_eq!(component.value, 5);
        drop(component);

        assert!(
            entity
                .unregister_component::<TestComponent>()
                .await
                .is_some()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn insert_replaces_existing_value() {
        let entity = Entity::new().await.expect("应能创建实体");

        assert!(
            entity
                .register_component(TestComponent::new(7))
                .await
                .expect("首次插入组件不应触发依赖错误")
                .is_none()
        );
        let previous = entity
            .register_component(TestComponent::new(9))
            .await
            .expect("重复插入不应触发依赖错误")
            .expect("重复插入应返回旧的组件值");
        assert_eq!(previous.value, 7);

        let component = entity
            .get_component::<TestComponent>()
            .await
            .expect("更新后仍应能读取组件");
        assert_eq!(component.value, 9);
        drop(component);

        assert!(
            entity
                .unregister_component::<TestComponent>()
                .await
                .is_some()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn get_mut_allows_in_place_mutation() {
        let entity = Entity::new().await.expect("应能创建实体");
        entity
            .register_component(TestComponent::new(11))
            .await
            .expect("插入组件不应触发依赖错误");

        {
            let mut component = entity
                .get_component_mut::<TestComponent>()
                .await
                .expect("应当能获取可写引用");
            component.value = 42;
        }

        let component = entity
            .get_component::<TestComponent>()
            .await
            .expect("写入后组件仍应存在");
        assert_eq!(component.value, 42);
        drop(component);

        assert!(
            entity
                .unregister_component::<TestComponent>()
                .await
                .is_some()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn remove_clears_component_slot() {
        let entity = Entity::new().await.expect("应能创建实体");
        entity
            .register_component(TestComponent::new(21))
            .await
            .expect("插入组件不应触发依赖错误");

        let removed = entity
            .unregister_component::<TestComponent>()
            .await
            .expect("应当能移除组件");
        assert_eq!(removed.value, 21);
        assert!(entity.get_component::<TestComponent>().await.is_none());
        assert!(
            entity
                .unregister_component::<TestComponent>()
                .await
                .is_none()
        );
    }

    #[component(TestComponent)]
    #[derive(Debug, Default)]
    struct NeedsTestComponent;

    #[component(AlternateDefaultComponent)]
    #[derive(Debug, Default)]
    struct NeedsAlternateDefault;

    #[component(FailingDefaultComponent)]
    #[derive(Debug, Default)]
    struct NeedsFailingDefault;

    #[tokio::test(flavor = "multi_thread")]
    async fn component_dependency_checks_for_required_components() {
        let entity = Entity::new().await.expect("应能创建实体");

        let inserted = entity
            .register_component(NeedsTestComponent)
            .await
            .expect("依赖应能通过默认构造自动满足");
        assert!(inserted.is_none());

        let dependency = entity
            .get_component::<TestComponent>()
            .await
            .expect("应自动注册缺失的依赖组件");
        assert_eq!(dependency.value, 0);
        drop(dependency);

        let previous = entity
            .register_component(TestComponent::new(3))
            .await
            .expect("更新依赖组件不应触发错误")
            .expect("应当获取到默认注册的依赖组件");
        assert_eq!(previous.value, 0);

        let previous_needs = entity
            .register_component(NeedsTestComponent)
            .await
            .expect("依赖已满足时仍应允许注册组件");
        assert!(previous_needs.is_some());

        let _ = entity.unregister_component::<NeedsTestComponent>().await;
        let _ = entity.unregister_component::<TestComponent>().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn dependency_supports_non_new_default_methods() {
        let entity = Entity::new().await.expect("应能创建实体");

        let inserted = entity
            .register_component(NeedsAlternateDefault)
            .await
            .expect("依赖应能通过默认方法自动满足");
        assert!(inserted.is_none());

        let dependency = entity
            .get_component::<AlternateDefaultComponent>()
            .await
            .expect("依赖组件应被默认方法创建");
        assert_eq!(dependency.created_by, "build");
        drop(dependency);

        let _ = entity.unregister_component::<NeedsAlternateDefault>().await;
        let _ = entity
            .unregister_component::<AlternateDefaultComponent>()
            .await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn dependency_default_errors_propagate_sources() {
        let entity = Entity::new().await.expect("应能创建实体");

        let err = entity
            .register_component(NeedsFailingDefault)
            .await
            .expect_err("失败的默认方法应向上传播错误");
        assert_eq!(err.entity(), entity);
        assert_eq!(err.required(), type_name::<FailingDefaultComponent>());

        let source = StdError::source(&err).expect("错误源应被保留");
        let source = source
            .downcast_ref::<IoError>()
            .expect("错误源应是 std::io::Error");
        assert_eq!(source.kind(), ErrorKind::Other);
        assert_eq!(source.to_string(), "default failure");

        assert!(
            entity
                .get_component::<FailingDefaultComponent>()
                .await
                .is_none()
        );
        assert!(
            entity
                .get_component::<NeedsFailingDefault>()
                .await
                .is_none()
        );
    }
}
