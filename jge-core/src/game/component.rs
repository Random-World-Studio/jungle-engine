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

use parking_lot::{
    ArcRwLockReadGuard, ArcRwLockWriteGuard, Mutex, RawRwLock, RwLock, RwLockUpgradableReadGuard,
};
use std::{
    any::type_name,
    collections::HashMap,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::game::entity::Entity;

const CHUNK_SIZE: usize = 64;

/// 所有组件都使用分块存储，每块包含固定数量的槽位，以便于后续向量化和并行优化。
pub trait Component: Sized + 'static {
    fn storage() -> &'static ComponentStorage<Self>;

    fn register_dependencies(_entity: Entity) -> Result<(), ComponentDependencyError> {
        Ok(())
    }

    fn unregister_dependencies(_entity: Entity) {}

    fn attach_entity(&mut self, _entity: Entity) {}

    fn detach_entity(&mut self) {}

    fn insert(entity: Entity, component: Self) -> Result<Option<Self>, ComponentDependencyError> {
        let mut component = component;
        component.attach_entity(entity);
        let previous = Self::storage().insert(entity.id(), component);
        Ok(previous.map(|mut existing| {
            existing.detach_entity();
            existing
        }))
    }

    fn read(entity: Entity) -> Option<ComponentRead<Self>> {
        Self::storage().get(entity.id())
    }

    fn write(entity: Entity) -> Option<ComponentWrite<Self>> {
        Self::storage().get_mut(entity.id())
    }

    fn remove(entity: Entity) -> Option<Self> {
        Self::storage().remove(entity.id()).map(|mut component| {
            component.detach_entity();
            component
        })
    }
}

#[derive(Debug)]
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

    pub fn entity(&self) -> Entity {
        self.entity
    }

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
    entity_to_location: HashMap<u64, ComponentLocation>,
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

const EMPTY_ENTITY_ID: u64 = u64::MAX;

struct ComponentSlot<C> {
    value: Arc<RwLock<Option<C>>>,
    entity_id: AtomicU64,
}

impl<C> ComponentSlot<C> {
    fn new() -> Self {
        Self {
            value: Arc::new(RwLock::new(None)),
            entity_id: AtomicU64::new(EMPTY_ENTITY_ID),
        }
    }

    fn replace(&self, entity_id: u64, value: C) -> Option<C> {
        let mut guard = self.value.write();
        let previous = guard.replace(value);
        self.entity_id.store(entity_id, Ordering::Relaxed);
        previous
    }

    fn take(&self) -> Option<C> {
        let mut guard = self.value.write();
        let removed = guard.take();
        if removed.is_some() {
            self.entity_id.store(EMPTY_ENTITY_ID, Ordering::Relaxed);
        }
        removed
    }

    fn try_read(&self) -> Option<ArcRwLockReadGuard<RawRwLock, Option<C>>> {
        let guard = self.value.try_read_arc()?;
        if guard.is_some() { Some(guard) } else { None }
    }

    fn try_write(&self) -> Option<ArcRwLockWriteGuard<RawRwLock, Option<C>>> {
        let guard = self.value.try_write_arc()?;
        if guard.is_some() { Some(guard) } else { None }
    }

    fn entity_id(&self) -> u64 {
        self.entity_id.load(Ordering::Relaxed)
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

    fn allocate(&self, entity_id: u64, value: C) -> (usize, Option<C>, bool) {
        let mut freelist = self.freelist.lock();
        let slot = freelist.pop().expect("尝试在已满的块中分配槽位");
        let has_more_free = !freelist.is_empty();
        drop(freelist);
        let previous = self.slot(slot).replace(entity_id, value);
        debug_assert!(previous.is_none(), "从空闲槽位分配时不应存在旧值");
        (slot, previous, has_more_free)
    }

    fn replace(&self, slot: usize, entity_id: u64, value: C) -> Option<C> {
        self.slot(slot).replace(entity_id, value)
    }

    fn release(&self, slot: usize) -> (Option<C>, bool) {
        let removed = self.slot(slot).take();
        if removed.is_some() {
            let mut freelist = self.freelist.lock();
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

impl<C> Default for ComponentStorage<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C> ComponentStorage<C> {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(ComponentStorageInner::default()),
        }
    }

    pub fn collect_with<T, F>(&self, mut f: F) -> Vec<T>
    where
        F: FnMut(u64, &C) -> Option<T>,
    {
        let guard = self.inner.read();
        let mut results = Vec::new();

        for chunk in &guard.chunks {
            for slot in &chunk.slots {
                let value_guard = slot.value.read();
                if let Some(component) = value_guard.as_ref() {
                    let entity_id = slot.entity_id.load(Ordering::Relaxed);
                    debug_assert!(entity_id != EMPTY_ENTITY_ID, "组件槽位缺少实体标识");
                    if let Some(mapped) = f(entity_id, component) {
                        results.push(mapped);
                    }
                }
            }
        }

        results
    }

    pub fn insert(&self, entity_id: u64, component: C) -> Option<C> {
        let guard = self.inner.upgradable_read();
        if let Some(location) = guard.entity_to_location.get(&entity_id).copied() {
            let chunk = guard.chunks[location.chunk].clone();
            drop(guard);
            return chunk.replace(location.slot, entity_id, component);
        }

        let mut guard = RwLockUpgradableReadGuard::upgrade(guard);
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

        let (slot, previous, has_more_free) = chunk.allocate(entity_id, component);
        debug_assert!(previous.is_none(), "从空闲槽位分配时不应存在旧值");
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

    pub fn get(&'static self, entity_id: u64) -> Option<ComponentRead<C>> {
        let (chunk, slot) = {
            let guard = self.inner.read();
            let location = guard.entity_to_location.get(&entity_id)?;
            (guard.chunks[location.chunk].clone(), location.slot)
        };
        ComponentRead::new(chunk, slot)
    }

    pub fn get_mut(&'static self, entity_id: u64) -> Option<ComponentWrite<C>> {
        let (chunk, slot) = {
            let guard = self.inner.read();
            let location = guard.entity_to_location.get(&entity_id)?;
            (guard.chunks[location.chunk].clone(), location.slot)
        };
        ComponentWrite::new(chunk, slot)
    }

    pub fn remove(&self, entity_id: u64) -> Option<C> {
        let guard = self.inner.upgradable_read();
        let location = match guard.entity_to_location.get(&entity_id).copied() {
            Some(location) => location,
            None => return None,
        };
        let chunk = guard.chunks[location.chunk].clone();
        let mut guard = RwLockUpgradableReadGuard::upgrade(guard);
        guard.entity_to_location.remove(&entity_id);
        let (removed, became_available) = chunk.release(location.slot);
        if became_available {
            guard.chunks_with_free_slots.push(location.chunk);
        }
        removed
    }
}

pub struct ComponentRead<C: 'static> {
    _chunk: Arc<ComponentChunk<C>>,
    _slot: Arc<ComponentSlot<C>>,
    guard: ArcRwLockReadGuard<RawRwLock, Option<C>>,
}

pub struct ComponentWrite<C: 'static> {
    _chunk: Arc<ComponentChunk<C>>,
    _slot: Arc<ComponentSlot<C>>,
    guard: ArcRwLockWriteGuard<RawRwLock, Option<C>>,
}

impl<C: 'static> ComponentRead<C> {
    fn new(chunk: Arc<ComponentChunk<C>>, index: usize) -> Option<Self> {
        let slot = chunk.slot(index);
        let guard = slot.try_read()?;
        Some(Self {
            _chunk: chunk,
            _slot: slot,
            guard,
        })
    }

    pub fn entity(&self) -> Entity {
        Entity::from(self._slot.entity_id())
    }
}

impl<C: 'static> ComponentWrite<C> {
    fn new(chunk: Arc<ComponentChunk<C>>, index: usize) -> Option<Self> {
        let slot = chunk.slot(index);
        let guard = slot.try_write()?;
        Some(Self {
            _chunk: chunk,
            _slot: slot,
            guard,
        })
    }

    pub fn entity(&self) -> Entity {
        Entity::from(self._slot.entity_id())
    }
}

impl<C: 'static> std::ops::Deref for ComponentRead<C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        self.guard
            .as_ref()
            .expect("组件存储出现不一致状态: 读到空槽位")
    }
}

impl<C: 'static> std::ops::Deref for ComponentWrite<C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        self.guard
            .as_ref()
            .expect("组件存储出现不一致状态: 写到空槽位")
    }
}

impl<C: 'static> std::ops::DerefMut for ComponentWrite<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard
            .as_mut()
            .expect("组件存储出现不一致状态: 写到空槽位")
    }
}

pub fn require_component<C: Component>(entity: Entity) -> Result<(), ComponentDependencyError> {
    if entity.get_component::<C>().is_some() {
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
            Err(IoError::new(ErrorKind::Other, "default failure"))
        }
    }

    #[test]
    fn insert_and_read_component() {
        let entity = Entity::new().expect("应能创建实体");

        assert!(
            entity
                .register_component(TestComponent::new(5))
                .expect("首次插入组件不应触发依赖错误")
                .is_none()
        );

        let component = entity
            .get_component::<TestComponent>()
            .expect("应当能读取到刚插入的组件");
        assert_eq!(component.value, 5);
        drop(component);

        assert!(entity.unregister_component::<TestComponent>().is_some());
    }

    #[test]
    fn insert_replaces_existing_value() {
        let entity = Entity::new().expect("应能创建实体");

        assert!(
            entity
                .register_component(TestComponent::new(7))
                .expect("首次插入组件不应触发依赖错误")
                .is_none()
        );
        let previous = entity
            .register_component(TestComponent::new(9))
            .expect("重复插入不应触发依赖错误")
            .expect("重复插入应返回旧的组件值");
        assert_eq!(previous.value, 7);

        let component = entity
            .get_component::<TestComponent>()
            .expect("更新后仍应能读取组件");
        assert_eq!(component.value, 9);
        drop(component);

        assert!(entity.unregister_component::<TestComponent>().is_some());
    }

    #[test]
    fn get_mut_allows_in_place_mutation() {
        let entity = Entity::new().expect("应能创建实体");
        entity
            .register_component(TestComponent::new(11))
            .expect("插入组件不应触发依赖错误");

        {
            let mut component = entity
                .get_component_mut::<TestComponent>()
                .expect("应当能获取可写引用");
            component.value = 42;
        }

        let component = entity
            .get_component::<TestComponent>()
            .expect("写入后组件仍应存在");
        assert_eq!(component.value, 42);
        drop(component);

        assert!(entity.unregister_component::<TestComponent>().is_some());
    }

    #[test]
    fn remove_clears_component_slot() {
        let entity = Entity::new().expect("应能创建实体");
        entity
            .register_component(TestComponent::new(21))
            .expect("插入组件不应触发依赖错误");

        let removed = entity
            .unregister_component::<TestComponent>()
            .expect("应当能移除组件");
        assert_eq!(removed.value, 21);
        assert!(entity.get_component::<TestComponent>().is_none());
        assert!(entity.unregister_component::<TestComponent>().is_none());
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

    #[test]
    fn component_dependency_checks_for_required_components() {
        let entity = Entity::new().expect("应能创建实体");

        let inserted = entity
            .register_component(NeedsTestComponent::default())
            .expect("依赖应能通过默认构造自动满足");
        assert!(inserted.is_none());

        let dependency = entity
            .get_component::<TestComponent>()
            .expect("应自动注册缺失的依赖组件");
        assert_eq!(dependency.value, 0);
        drop(dependency);

        let previous = entity
            .register_component(TestComponent::new(3))
            .expect("更新依赖组件不应触发错误")
            .expect("应当获取到默认注册的依赖组件");
        assert_eq!(previous.value, 0);

        let previous_needs = entity
            .register_component(NeedsTestComponent::default())
            .expect("依赖已满足时仍应允许注册组件");
        assert!(previous_needs.is_some());

        let _ = entity.unregister_component::<NeedsTestComponent>();
        let _ = entity.unregister_component::<TestComponent>();
    }

    #[test]
    fn dependency_supports_non_new_default_methods() {
        let entity = Entity::new().expect("应能创建实体");

        let inserted = entity
            .register_component(NeedsAlternateDefault::default())
            .expect("依赖应能通过默认方法自动满足");
        assert!(inserted.is_none());

        let dependency = entity
            .get_component::<AlternateDefaultComponent>()
            .expect("依赖组件应被默认方法创建");
        assert_eq!(dependency.created_by, "build");
        drop(dependency);

        let _ = entity.unregister_component::<NeedsAlternateDefault>();
        let _ = entity.unregister_component::<AlternateDefaultComponent>();
    }

    #[test]
    fn dependency_default_errors_propagate_sources() {
        let entity = Entity::new().expect("应能创建实体");

        let err = entity
            .register_component(NeedsFailingDefault::default())
            .expect_err("失败的默认方法应向上传播错误");
        assert_eq!(err.entity(), entity);
        assert_eq!(err.required(), type_name::<FailingDefaultComponent>());

        let source = StdError::source(&err).expect("错误源应被保留");
        let source = source
            .downcast_ref::<IoError>()
            .expect("错误源应是 std::io::Error");
        assert_eq!(source.kind(), ErrorKind::Other);
        assert_eq!(source.to_string(), "default failure");

        assert!(entity.get_component::<FailingDefaultComponent>().is_none());
        assert!(entity.get_component::<NeedsFailingDefault>().is_none());
    }
}
