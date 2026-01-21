use std::{any::Any, fmt::Debug, sync::Arc};

pub use winit::{
    event::{DeviceEvent, ElementState, KeyEvent, WindowEvent},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
};

/// 引擎事件。
///
/// 当前以 `Custom` 为主：允许游戏侧把任意自定义类型作为事件载荷传入逻辑层。
///
/// 事件载荷使用 `Arc` 承载，便于在多个 `GameLogic` 之间广播（无需要求载荷实现 `Clone`）。
///
/// # 示例：自定义事件载荷
///
/// ```
/// #[derive(Debug, PartialEq, Eq)]
/// struct MyEvent {
///     id: u32,
/// }
///
/// let e = ::jge_core::event::Event::custom(MyEvent { id: 1 });
/// assert_eq!(e.downcast_ref::<MyEvent>().unwrap().id, 1);
/// ```
#[derive(Clone)]
pub enum Event {
    CloseRequested,
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
            _ => None,
        }
    }
}

impl Debug for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CloseRequested => write!(f, "CloseRequested"),
            Self::Custom(arg) => f.debug_tuple("Custom").field(arg).finish(),
        }
    }
}

/// 将 `winit` 的窗口事件映射为引擎事件。
///
/// 游戏侧可实现该 trait，把 `WindowEvent` 转成自己的事件类型并包进 [`Event::Custom`]。
///
/// 同一个映射器也可以选择性地处理 `DeviceEvent`（例如 `MouseMotion` 的相对位移），以便在
/// 平台/窗口系统限制导致 `CursorMoved` 不可靠时，仍能获得稳定的相对输入。
///
/// # 示例：用闭包实现映射器
///
/// ```
/// use ::jge_core::event::{Event, EventMapper, WindowEvent};
///
/// let mut mapper = |evt: &WindowEvent| {
///     // 这里仅做演示：真实项目中你通常会匹配特定按键/鼠标事件。
///     let _ = evt;
///     Some(Event::custom("hello".to_string()))
/// };
///
/// // 通过 trait 调用：
/// let _ = EventMapper::map_window_event(&mut mapper, &WindowEvent::Focused(true));
/// ```
pub trait EventMapper: Send + Sync {
    /// 将 `winit::WindowEvent` 转换为引擎 [`Event`]。
    ///
    /// 返回 `None` 表示忽略该事件。
    fn map_window_event(&mut self, _event: &WindowEvent) -> Option<Event> {
        None
    }

    /// 将 `winit::DeviceEvent` 转换为引擎 [`Event`]。
    ///
    /// 默认忽略所有设备事件；需要时可覆写此方法（典型用例：`DeviceEvent::MouseMotion`）。
    fn map_device_event(&mut self, _event: &DeviceEvent) -> Option<Event> {
        None
    }
}

impl<F> EventMapper for F
where
    F: FnMut(&WindowEvent) -> Option<Event> + Send + Sync,
{
    fn map_window_event(&mut self, event: &WindowEvent) -> Option<Event> {
        (self)(event)
    }
}

/// 默认映射器：忽略所有窗口事件。
pub struct NoopEventMapper;

impl EventMapper for NoopEventMapper {}

/// 把“窗口事件映射”和“设备事件映射”组合成一个 [`EventMapper`]。
///
/// 适用于需要同时处理 `WindowEvent` 与 `DeviceEvent` 的场景（例如鼠标视角控制）。
pub struct SplitEventMapper<W, D> {
    window: W,
    device: D,
}

impl<W, D> SplitEventMapper<W, D> {
    pub fn new(window: W, device: D) -> Self {
        Self { window, device }
    }
}

impl<W, D> EventMapper for SplitEventMapper<W, D>
where
    W: FnMut(&WindowEvent) -> Option<Event> + Send + Sync,
    D: FnMut(&DeviceEvent) -> Option<Event> + Send + Sync,
{
    fn map_window_event(&mut self, event: &WindowEvent) -> Option<Event> {
        (self.window)(event)
    }

    fn map_device_event(&mut self, event: &DeviceEvent) -> Option<Event> {
        (self.device)(event)
    }
}

/// 便捷构造：创建一个同时处理窗口/设备事件的映射器。
pub fn split_event_mapper<W, D>(window: W, device: D) -> SplitEventMapper<W, D>
where
    W: FnMut(&WindowEvent) -> Option<Event> + Send + Sync,
    D: FnMut(&DeviceEvent) -> Option<Event> + Send + Sync,
{
    SplitEventMapper::new(window, device)
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
    fn event_mapper_closure_is_used() {
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
        let mapped = EventMapper::map_window_event(&mut mapper, &event).expect("应能映射窗口事件");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(mapped.downcast_ref::<u32>(), Some(&800));
    }

    #[test]
    fn noop_mappers_return_none() {
        let mut mapper = NoopEventMapper;

        let window_event = WindowEvent::Resized(PhysicalSize::new(1u32, 1u32));
        assert!(mapper.map_window_event(&window_event).is_none());

        let device_event = DeviceEvent::Added;
        assert!(mapper.map_device_event(&device_event).is_none());
    }
}
