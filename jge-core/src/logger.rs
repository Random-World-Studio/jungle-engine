use tracing::level_filters::LevelFilter;
use tracing_subscriber::{EnvFilter, fmt::time, layer::SubscriberExt, util::SubscriberInitExt};

/// 初始化引擎的日志/追踪（tracing）订阅者。
///
/// - 默认会读取环境变量（由 `tracing_subscriber::EnvFilter` 支持），用于覆盖/追加过滤规则。
/// - Debug 构建下日志更详细；Release 构建下更偏向错误级别。
///
/// 注意：该函数应在应用启动早期调用一次；重复初始化通常会失败或被忽略（取决于 tracing 订阅者实现）。
pub fn init() -> anyhow::Result<()> {
    if cfg!(debug_assertions) {
        tracing_subscriber::Registry::default()
            .with(
                tracing_subscriber::fmt::layer()
                    .with_ansi(true)
                    .with_timer(time::uptime()),
            )
            .with(
                EnvFilter::builder()
                    .with_default_directive(LevelFilter::TRACE.into())
                    .from_env()?
                    .add_directive("jge-core=debug".parse()?)
                    .add_directive("calloop=info".parse()?)
                    .add_directive("winit=warn".parse()?)
                    .add_directive("naga=warn".parse()?)
                    .add_directive("sctk=warn".parse()?)
                    .add_directive("wgpu_hal=error".parse()?)
                    .add_directive("wgpu_core=error".parse()?),
            )
            .init();
    } else {
        tracing_subscriber::Registry::default()
            .with(
                tracing_subscriber::fmt::layer()
                    .with_ansi(true)
                    .with_timer(time::uptime()),
            )
            .with(
                EnvFilter::builder()
                    .with_default_directive(LevelFilter::TRACE.into())
                    .from_env()?
                    .add_directive("jge-core=info".parse()?)
                    .add_directive("calloop=error".parse()?)
                    .add_directive("winit=error".parse()?)
                    .add_directive("naga=error".parse()?)
                    .add_directive("sctk=error".parse()?)
                    .add_directive("wgpu_hal=error".parse()?)
                    .add_directive("wgpu_core=error".parse()?),
            )
            .init();
    }
    Ok(())
}
