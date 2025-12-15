use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;

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
    /// 异步构造器。默认返回一个空实现。
    async fn new_boxed(_entity_id: u64) -> Result<Box<dyn GameLogic>>
    where
        Self: Sized,
    {
        Ok(Box::new(EmptyGameLogic {}))
    }

    /// 当刚被挂载到父节点下时调用，默认不做任何事。
    async fn on_attach(&mut self, _entity_id: u64) -> Result<()> {
        Ok(())
    }

    /// 每个游戏刻调用一次（异步），默认不做任何事。
    async fn update(&mut self, _delta: std::time::Duration) -> Result<()> {
        Ok(())
    }

    /// 每帧调用一次（同步），提供当帧耗时，默认空实现。
    fn on_render(&mut self, _delta: std::time::Duration) {}

    /// 发生事件时调用（异步），默认不做任何事。
    async fn on_event(&mut self, _event: &str) -> Result<()> {
        Ok(())
    }

    /// 当节点被卸载时调用（异步），默认不做任何事。
    async fn on_detach(&mut self, _entity_id: u64) -> Result<()> {
        Ok(())
    }
}

/// 一个空实现，作为 `GameLogic` 的默认占位符。
struct EmptyGameLogic {}

#[async_trait]
impl GameLogic for EmptyGameLogic {}

/// 类型便利：用于在 `Node` 上持有可变的 `GameLogic` 实例。
pub type GameLogicHandle = Arc<Mutex<Box<dyn GameLogic>>>;
