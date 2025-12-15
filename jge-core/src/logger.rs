use tracing::level_filters::LevelFilter;
use tracing_subscriber::{EnvFilter, fmt::time, layer::SubscriberExt, util::SubscriberInitExt};

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
