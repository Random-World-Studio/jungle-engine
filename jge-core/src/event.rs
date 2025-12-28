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
    /// 自定义事件：使用 `Arc<dyn Any + Send + Sync>` 承载任意用户载荷。
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
    /// 将 `winit::WindowEvent` 转换为引擎 [`Event`]。
    ///
    /// 返回 `None` 表示忽略该事件。
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
    /// 将 `winit::DeviceEvent` 转换为引擎 [`Event`]。
    ///
    /// 返回 `None` 表示忽略该事件。
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use winit::dpi::PhysicalSize;

    #[derive(Debug, PartialEq, Eq)]
    struct TestPayload {
        value: u32,
    }

    #[test]
    fn custom_event_downcasts_to_original_type() {
        let event = Event::custom(TestPayload { value: 42 });
        let payload = event
            .downcast_ref::<TestPayload>()
            .expect("应能 downcast 到原始类型");
        assert_eq!(payload.value, 42);

        assert!(event.downcast_ref::<u64>().is_none(), "错误类型应返回 None");
    }

    #[test]
    fn cloned_custom_event_shares_payload() {
        let event = Event::custom(TestPayload { value: 7 });
        let cloned = event.clone();

        let a = event.downcast_ref::<TestPayload>().unwrap();
        let b = cloned.downcast_ref::<TestPayload>().unwrap();
        assert!(ptr::eq(a, b), "Event::clone 应共享同一份 Arc 载荷");
    }

    #[test]
    fn window_event_mapper_closure_is_used() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_mapper = calls.clone();
        let mut mapper = |event: &WindowEvent| {
            calls_for_mapper.fetch_add(1, Ordering::Relaxed);
            match event {
                WindowEvent::Resized(size) => Some(Event::custom(size.width)),
                _ => None,
            }
        };

        let event = WindowEvent::Resized(PhysicalSize::new(800u32, 600u32));
        let mapped =
            WindowEventMapper::map_window_event(&mut mapper, &event).expect("应能映射窗口事件");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(mapped.downcast_ref::<u32>(), Some(&800));
    }

    #[test]
    fn device_event_mapper_closure_is_used() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_mapper = calls.clone();
        let mut mapper = |_event: &DeviceEvent| {
            calls_for_mapper.fetch_add(1, Ordering::Relaxed);
            Some(Event::custom("device"))
        };

        let event = DeviceEvent::Added;
        let mapped =
            DeviceEventMapper::map_device_event(&mut mapper, &event).expect("应能映射设备事件");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(mapped.downcast_ref::<&'static str>(), Some(&"device"));
    }

    #[test]
    fn noop_mappers_return_none() {
        let mut window = NoopWindowEventMapper;
        let mut device = NoopDeviceEventMapper;

        let window_event = WindowEvent::Resized(PhysicalSize::new(1u32, 1u32));
        assert!(window.map_window_event(&window_event).is_none());

        let device_event = DeviceEvent::Added;
        assert!(device.map_device_event(&device_event).is_none());
    }
}
