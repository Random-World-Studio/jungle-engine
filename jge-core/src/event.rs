use std::{any::Any, sync::Arc};

pub use winit::{
    event::{DeviceEvent, ElementState, KeyEvent, WindowEvent},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
};

/// 引擎事件。
///
/// 当前以 `Custom` 为主：允许游戏侧把任意自定义类型作为事件载荷传入逻辑层。
///
/// 事件载荷使用 `Arc` 承载，便于在多个 `GameLogic` 之间广播（无需要求载荷实现 `Clone`）。
#[derive(Clone)]
pub enum Event {
    Custom(Arc<dyn Any + Send + Sync>),
}

impl Event {
    /// 从任意自定义类型构造事件。
    pub fn custom<T>(value: T) -> Self
    where
        T: Any + Send + Sync,
    {
        Self::Custom(Arc::new(value))
    }

    /// 尝试把 `Custom` 事件的载荷按类型借用出来。
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        match self {
            Self::Custom(payload) => payload.as_ref().downcast_ref::<T>(),
        }
    }
}

/// 将 `winit` 的窗口事件映射为引擎事件。
///
/// 游戏侧可实现该 trait，把 `WindowEvent` 转成自己的事件类型并包进 [`Event::Custom`]。
pub trait WindowEventMapper: Send + Sync {
    fn map_window_event(&mut self, event: &WindowEvent) -> Option<Event>;
}

impl<F> WindowEventMapper for F
where
    F: FnMut(&WindowEvent) -> Option<Event> + Send + Sync,
{
    fn map_window_event(&mut self, event: &WindowEvent) -> Option<Event> {
        (self)(event)
    }
}

/// 默认映射器：忽略所有窗口事件。
pub struct NoopWindowEventMapper;

impl WindowEventMapper for NoopWindowEventMapper {
    fn map_window_event(&mut self, _event: &WindowEvent) -> Option<Event> {
        None
    }
}

/// 将 `winit` 的设备事件映射为引擎事件。
///
/// 典型用途：读取 `DeviceEvent::MouseMotion` 的相对位移，用于第一人称视角控制。
pub trait DeviceEventMapper: Send + Sync {
    fn map_device_event(&mut self, event: &DeviceEvent) -> Option<Event>;
}

impl<F> DeviceEventMapper for F
where
    F: FnMut(&DeviceEvent) -> Option<Event> + Send + Sync,
{
    fn map_device_event(&mut self, event: &DeviceEvent) -> Option<Event> {
        (self)(event)
    }
}

/// 默认映射器：忽略所有设备事件。
pub struct NoopDeviceEventMapper;

impl DeviceEventMapper for NoopDeviceEventMapper {
    fn map_device_event(&mut self, _event: &DeviceEvent) -> Option<Event> {
        None
    }
}
