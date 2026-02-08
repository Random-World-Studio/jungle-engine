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
use tokio::sync::{Mutex, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use crate::game::entity::{Entity, EntityId};

const CHUNK_SIZE: usize = 16;

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

struct ComponentStorageInner<C> {
    chunks: Vec<Arc<ComponentChunk<C>>>,
    entity_to_location: HashMap<EntityId, ComponentLocation>,
    chunks_with_free_slots: Vec<usize>,
}

impl<C> Default for ComponentStorageInner<C> {
    fn default() -> Self {
        Self {
            chunks: Vec::new(),
            entity_to_location: HashMap::new(),
            chunks_with_free_slots: Vec::new(),
        }
    }
}

#[derive(Clone, Copy)]
struct ComponentLocation {
    chunk: usize,
    slot: usize,
}

#[derive(Debug)]
struct SlotValue<C> {
    entity_id: EntityId,
    value: C,
}

struct ComponentSlot<C> {
    value: Arc<RwLock<Option<SlotValue<C>>>>,
}

impl<C> ComponentSlot<C> {
    fn new() -> Self {
        Self {
            value: Arc::new(RwLock::new(None)),
        }
    }

    async fn replace(&self, entity_id: EntityId, value: C) -> Option<C> {
        let mut guard = self.value.write().await;
        let previous = guard.replace(SlotValue { entity_id, value });
        previous.map(|p| p.value)
    }

    async fn take(&self) -> Option<C> {
        let mut guard = self.value.write().await;
        guard.take().map(|p| p.value)
    }

    async fn read_owned(&self) -> Option<OwnedRwLockReadGuard<Option<SlotValue<C>>>> {
        let guard = Arc::clone(&self.value).read_owned().await;
        if guard.is_some() { Some(guard) } else { None }
    }

    async fn write_owned(&self) -> Option<OwnedRwLockWriteGuard<Option<SlotValue<C>>>> {
        let guard = Arc::clone(&self.value).write_owned().await;
        if guard.is_some() { Some(guard) } else { None }
    }
}

struct ComponentChunk<C> {
    slots: Vec<Arc<ComponentSlot<C>>>,
    freelist: Mutex<Vec<usize>>,
}

impl<C> ComponentChunk<C> {
    fn new() -> Self {
        let slots = (0..CHUNK_SIZE)
            .map(|_| Arc::new(ComponentSlot::new()))
            .collect();
        let freelist = Mutex::new((0..CHUNK_SIZE).rev().collect());
        Self { slots, freelist }
    }

    async fn allocate(&self, entity_id: EntityId, value: C) -> (usize, Option<C>, bool) {
        let mut freelist = self.freelist.lock().await;
        let slot = freelist.pop().expect("尝试在已满的块中分配槽位");
        let has_more_free = !freelist.is_empty();
        drop(freelist);
        let previous = self.slot(slot).replace(entity_id, value).await;
        debug_assert!(previous.is_none(), "从空闲槽位分配时不应存在旧值");
        (slot, previous, has_more_free)
    }

    async fn replace(&self, slot: usize, entity_id: EntityId, value: C) -> Option<C> {
        self.slot(slot).replace(entity_id, value).await
    }

    async fn release(&self, slot: usize) -> (Option<C>, bool) {
        let removed = self.slot(slot).take().await;
        if removed.is_some() {
            let mut freelist = self.freelist.lock().await;
            let was_full = freelist.is_empty();
            freelist.push(slot);
            (removed, was_full)
        } else {
            (None, false)
        }
    }

    fn slot(&self, index: usize) -> Arc<ComponentSlot<C>> {
        self.slots.get(index).expect("槽位索引越界").clone()
    }
}

pub struct ComponentStorage<C> {
    inner: RwLock<ComponentStorageInner<C>>,
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
            inner: RwLock::new(ComponentStorageInner::default()),
        }
    }

    /// 遍历所有已存在组件，并将其映射收集成一个 Vec。
    ///
    /// 这是偏底层的枚举接口，通常用于系统层做批处理/收集。
    pub async fn collect_with<T, F>(&self, mut f: F) -> Vec<T>
    where
        F: FnMut(EntityId, &C) -> Option<T>,
    {
        let slots: Vec<Arc<ComponentSlot<C>>> = {
            let guard = self.inner.read().await;
            guard
                .chunks
                .iter()
                .flat_map(|chunk| chunk.slots.iter().cloned())
                .collect()
        };

        let mut results = Vec::new();
        for slot in slots {
            let Some(value_guard) = slot.read_owned().await else {
                continue;
            };
            let Some(slot_value) = value_guard.as_ref() else {
                continue;
            };
            if let Some(mapped) = f(slot_value.entity_id, &slot_value.value) {
                results.push(mapped);
            }
        }

        results
    }

    /// 按 chunk 分组遍历并收集结果。
    ///
    /// 当你希望利用“分块”天然分组做并行/局部性优化时可用。
    pub async fn collect_chunks_with<T, F>(&self, mut f: F) -> Vec<Vec<T>>
    where
        F: FnMut(EntityId, &C) -> Option<T>,
    {
        let chunks: Vec<Arc<ComponentChunk<C>>> = {
            let guard = self.inner.read().await;
            guard.chunks.clone()
        };

        let mut results: Vec<Vec<T>> = Vec::new();
        for chunk in &chunks {
            let mut chunk_results = Vec::new();
            for slot in &chunk.slots {
                let Some(value_guard) = slot.read_owned().await else {
                    continue;
                };
                let Some(slot_value) = value_guard.as_ref() else {
                    continue;
                };
                if let Some(mapped) = f(slot_value.entity_id, &slot_value.value) {
                    chunk_results.push(mapped);
                }
            }
            if !chunk_results.is_empty() {
                results.push(chunk_results);
            }
        }

        results
    }

    /// 插入或替换一个组件值。
    ///
    /// 建议通过 `Entity::register_component` 间接调用。
    pub async fn insert(&self, entity_id: EntityId, component: C) -> Option<C> {
        // fast path: existing component
        if let Some(location) = {
            let guard = self.inner.read().await;
            guard.entity_to_location.get(&entity_id).copied()
        } {
            let chunk = {
                let guard = self.inner.read().await;
                guard.chunks[location.chunk].clone()
            };
            return chunk.replace(location.slot, entity_id, component).await;
        }

        // slow path: allocate a new slot
        let (chunk_index, _chunk, slot, previous, has_more_free) = {
            let mut guard = self.inner.write().await;

            // re-check under write lock (in case another task inserted)
            if let Some(location) = guard.entity_to_location.get(&entity_id).copied() {
                let chunk = guard.chunks[location.chunk].clone();
                drop(guard);
                let previous = chunk.replace(location.slot, entity_id, component).await;
                return previous;
            }

            let (chunk_index, chunk) = match guard.chunks_with_free_slots.pop() {
                Some(index) => {
                    let chunk = guard.chunks[index].clone();
                    (index, chunk)
                }
                None => {
                    let index = guard.chunks.len();
                    let chunk = Arc::new(ComponentChunk::new());
                    guard.chunks.push(chunk.clone());
                    (index, chunk)
                }
            };

            // allocate outside of the inner lock to reduce contention
            drop(guard);

            let (slot, previous, has_more_free) = chunk.allocate(entity_id, component).await;
            (chunk_index, chunk, slot, previous, has_more_free)
        };

        debug_assert!(previous.is_none(), "从空闲槽位分配时不应存在旧值");

        let mut guard = self.inner.write().await;
        guard.entity_to_location.insert(
            entity_id,
            ComponentLocation {
                chunk: chunk_index,
                slot,
            },
        );
        if has_more_free {
            guard.chunks_with_free_slots.push(chunk_index);
        }
        None
    }

    /// 获取组件的只读 guard。
    pub async fn get(&self, entity_id: EntityId) -> Option<ComponentRead<C>> {
        let (chunk, slot) = {
            let guard = self.inner.read().await;
            let location = guard.entity_to_location.get(&entity_id)?;
            (guard.chunks[location.chunk].clone(), location.slot)
        };
        ComponentRead::new(chunk, slot).await
    }

    /// 获取组件的可写 guard。
    pub async fn get_mut(&self, entity_id: EntityId) -> Option<ComponentWrite<C>> {
        let (chunk, slot) = {
            let guard = self.inner.read().await;
            let location = guard.entity_to_location.get(&entity_id)?;
            (guard.chunks[location.chunk].clone(), location.slot)
        };
        ComponentWrite::new(chunk, slot).await
    }

    /// 移除组件并返回旧值。
    pub async fn remove(&self, entity_id: EntityId) -> Option<C> {
        let (chunk, location) = {
            let mut guard = self.inner.write().await;
            let location = guard.entity_to_location.remove(&entity_id)?;
            let chunk = guard.chunks[location.chunk].clone();
            (chunk, location)
        };

        let (removed, became_available) = chunk.release(location.slot).await;
        if became_available {
            let mut guard = self.inner.write().await;
            guard.chunks_with_free_slots.push(location.chunk);
        }
        removed
    }
}

/// 组件只读访问 guard。
///
/// 通过 `Entity::get_component::<T>().await` 获得。
/// 该类型实现了 `Deref<Target = T>`，因此可以像 `&T` 一样使用。
pub struct ComponentRead<C: 'static> {
    _chunk: Arc<ComponentChunk<C>>,
    _slot: Arc<ComponentSlot<C>>,
    guard: OwnedRwLockReadGuard<Option<SlotValue<C>>>,
}

/// 组件可写访问 guard。
///
/// 通过 `Entity::get_component_mut::<T>().await` 获得。
/// 该类型实现了 `DerefMut`，因此可以直接修改组件字段/调用 setter。
pub struct ComponentWrite<C: 'static> {
    _chunk: Arc<ComponentChunk<C>>,
    _slot: Arc<ComponentSlot<C>>,
    guard: OwnedRwLockWriteGuard<Option<SlotValue<C>>>,
}

impl<C: 'static> ComponentRead<C> {
    async fn new(chunk: Arc<ComponentChunk<C>>, index: usize) -> Option<Self> {
        let slot = chunk.slot(index);
        let guard = slot.read_owned().await?;
        Some(Self {
            _chunk: chunk,
            _slot: slot,
            guard,
        })
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
    async fn new(chunk: Arc<ComponentChunk<C>>, index: usize) -> Option<Self> {
        let slot = chunk.slot(index);
        let guard = slot.write_owned().await?;
        Some(Self {
            _chunk: chunk,
            _slot: slot,
            guard,
        })
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

    #[component]
    #[derive(Debug, PartialEq, Eq)]
    struct ChunkProbeGroupComponent {
        index: usize,
    }

    #[component_impl(ChunkProbeGroupComponent)]
    impl ChunkProbeGroupComponent {
        #[default(0)]
        fn new(index: usize) -> Self {
            Self { index }
        }
    }

    #[component]
    #[derive(Debug, PartialEq, Eq)]
    struct ChunkProbeSkipComponent {
        index: usize,
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

    #[component_impl(ChunkProbeSkipComponent)]
    impl ChunkProbeSkipComponent {
        #[default(0)]
        fn new(index: usize) -> Self {
            Self { index }
        }
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

    #[tokio::test(flavor = "multi_thread")]
    async fn collect_chunks_with_groups_values_by_chunk() {
        let total = CHUNK_SIZE + 2;
        let mut entities = Vec::with_capacity(total);
        for _ in 0..total {
            entities.push(Entity::new().await.expect("应能创建实体"));
        }

        for (index, entity) in entities.iter().enumerate() {
            entity
                .register_component(ChunkProbeGroupComponent::new(index))
                .await
                .expect("插入组件不应触发依赖错误");
        }

        let chunks = ChunkProbeGroupComponent::storage()
            .collect_chunks_with(|_entity_id, component| Some(component.index))
            .await;

        assert_eq!(chunks.len(), 2, "应跨越 CHUNK_SIZE 形成两个 chunk");
        assert_eq!(
            chunks[0].len(),
            CHUNK_SIZE,
            "第一个 chunk 应填满 CHUNK_SIZE 个槽位"
        );
        assert_eq!(chunks[1].len(), 2, "第二个 chunk 应只包含剩余 2 个组件");

        // 清理，避免影响后续测试。
        for entity in entities.drain(..) {
            let _ = entity
                .unregister_component::<ChunkProbeGroupComponent>()
                .await;
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn collect_chunks_with_skips_empty_chunks() {
        let total = CHUNK_SIZE + 1;
        let mut entities = Vec::with_capacity(total);
        for _ in 0..total {
            entities.push(Entity::new().await.expect("应能创建实体"));
        }

        for (index, entity) in entities.iter().enumerate() {
            entity
                .register_component(ChunkProbeSkipComponent::new(index))
                .await
                .expect("插入组件不应触发依赖错误");
        }

        // 移除第一个 chunk 的所有组件，使其成为空 chunk。
        for entity in &entities[..CHUNK_SIZE] {
            let _ = entity
                .unregister_component::<ChunkProbeSkipComponent>()
                .await;
        }

        let chunks = ChunkProbeSkipComponent::storage()
            .collect_chunks_with(|_entity_id, component| Some(component.index))
            .await;

        assert_eq!(chunks.len(), 1, "空 chunk 应被跳过");
        assert_eq!(
            chunks[0],
            vec![CHUNK_SIZE],
            "应只剩下第二个 chunk 的最后一个组件"
        );

        // 清理。
        let _ = entities[CHUNK_SIZE]
            .unregister_component::<ChunkProbeSkipComponent>()
            .await;
    }
}
