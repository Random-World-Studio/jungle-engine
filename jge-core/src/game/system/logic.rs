use anyhow::Result;
use async_trait::async_trait;
use std::ops::Deref;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::event::Event;
use crate::game::entity::Entity;

/// 游戏逻辑（可附加到 `Node` 上）。
///
/// - `new_boxed`：异步构造函数，返回一个 `Box<dyn GameLogic>` 的实例。
/// - `on_attach`：当节点被挂载到父节点下时调用（异步）。
/// - `update`：每个游戏刻调用一次（异步）。
/// - `on_render`：每帧调用一次（同步）。
/// - `on_event`：发生事件时调用（异步）。
/// - `on_detach`：节点被卸载时调用（异步）。
#[async_trait]
pub trait GameLogic: Send + Sync {
    /// 构造器。默认返回一个空实现。
    fn new_boxed(_e: Entity) -> Result<Box<dyn GameLogic>>
    where
        Self: Sized,
    {
        Ok(Box::new(EmptyGameLogic {}))
    }

    /// 当刚被挂载到父节点下时调用，默认不做任何事。
    fn on_attach(&mut self, _e: Entity) -> Result<()> {
        Ok(())
    }

    /// 每个游戏刻调用一次（异步），默认不做任何事。
    async fn update(&mut self, _e: Entity, _delta: std::time::Duration) -> Result<()> {
        Ok(())
    }

    /// 每帧调用一次（异步），提供当帧耗时，默认空实现。
    async fn on_render(&mut self, _e: Entity, _delta: std::time::Duration) -> Result<()> {
        Ok(())
    }

    /// 信号触发时调用（异步），默认不做任何事。
    async fn on_event(&mut self, _e: Entity, _event: &Event) -> Result<()> {
        Ok(())
    }

    /// 当节点被卸载时调用（异步），默认不做任何事。
    fn on_detach(&mut self, _e: Entity) -> Result<()> {
        Ok(())
    }
}

/// 一个空实现，作为 `GameLogic` 的默认占位符。
struct EmptyGameLogic {}

#[async_trait]
impl GameLogic for EmptyGameLogic {}

/// 用于在 `Node` 上持有可变的 `GameLogic` 实例。
///
/// 这是一个带内部共享可变性的句柄类型：`Arc<Mutex<Box<dyn GameLogic>>>`。
/// 你通常只需要用 [`GameLogicHandle::new`] 创建它。
#[derive(Clone)]
pub struct GameLogicHandle(Arc<Mutex<Box<dyn GameLogic>>>);

impl GameLogicHandle {
    /// 创建一个新的逻辑句柄。
    pub fn new(logic: impl GameLogic + 'static) -> Self {
        Self(Arc::new(Mutex::new(Box::new(logic))))
    }
}

impl std::fmt::Debug for GameLogicHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("GameLogicHandle").finish()
    }
}

impl Deref for GameLogicHandle {
    type Target = Arc<Mutex<Box<dyn GameLogic>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex as StdMutex};
    use tokio::runtime::Builder;

    struct CounterLogic {
        counter: Arc<StdMutex<u32>>,
    }

    #[async_trait]
    impl GameLogic for CounterLogic {
        async fn update(&mut self, _e: Entity, _delta: std::time::Duration) -> Result<()> {
            *self.counter.lock().unwrap() += 1;
            Ok(())
        }
    }

    #[test]
    fn new_boxed_default_returns_ok() {
        struct AnyLogic;

        #[async_trait]
        impl GameLogic for AnyLogic {}

        let entity = Entity::from(0);
        let boxed = <AnyLogic as GameLogic>::new_boxed(entity);
        assert!(boxed.is_ok());
    }

    #[test]
    fn handle_clone_shares_same_logic_instance() {
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let counter = Arc::new(StdMutex::new(0u32));
            let handle = GameLogicHandle::new(CounterLogic {
                counter: counter.clone(),
            });
            let cloned = handle.clone();

            {
                let mut logic = handle.lock().await;
                logic.update(Entity::from(1), std::time::Duration::from_millis(1))
                    .await
                    .unwrap();
            }
            {
                let mut logic = cloned.lock().await;
                logic.update(Entity::from(1), std::time::Duration::from_millis(1))
                    .await
                    .unwrap();
            }

            assert_eq!(*counter.lock().unwrap(), 2);
        });
    }

    #[test]
    fn debug_includes_type_name() {
        struct DummyLogic;

        #[async_trait]
        impl GameLogic for DummyLogic {}

        let handle = GameLogicHandle::new(DummyLogic);

        let dbg = format!("{handle:?}");
        assert!(dbg.contains("GameLogicHandle"));
    }
}
