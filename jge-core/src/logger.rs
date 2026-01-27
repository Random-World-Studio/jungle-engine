use std::{panic, thread};

use tracing::level_filters::LevelFilter;
use tracing_subscriber::{EnvFilter, fmt::time, layer::SubscriberExt, util::SubscriberInitExt};

/// 初始化引擎的日志/追踪（tracing）订阅者。
///
/// - 默认会读取环境变量（由 `tracing_subscriber::EnvFilter` 支持），用于覆盖/追加过滤规则。
/// - Debug 构建下日志更详细；Release 构建下更偏向错误级别。
///
/// 注意：该函数应在应用启动早期调用一次；重复初始化通常会失败或被忽略（取决于 tracing 订阅者实现）。
///
/// # 示例
///
/// ```no_run
/// fn main() -> ::anyhow::Result<()> {
///     ::jge_core::logger::init()?;
///     // ... 创建 Game/加载资源/运行主循环
///     Ok(())
/// }
/// ```
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
                    .with_default_directive(LevelFilter::INFO.into())
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

    // 安装 panic hook：panic 时用 error 级别记录信息，方便在日志系统中统一收集。
    // 注意：这里不调用默认 hook，避免其受 `RUST_BACKTRACE` 影响输出 full 堆栈；
    // 我们永远输出“简略堆栈”版本。
    panic::set_hook(Box::new(move |panic_info| {
        let payload = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".to_string()
        };

        let location = panic_info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown location>".to_string());

        let thread_name = thread::current().name().unwrap_or("<unnamed>").to_string();

        let backtrace = format_short_backtrace();

        tracing::error!(
            target: "jge-core",
            panic = %payload,
            thread = %thread_name,
            location = %location,
            "panic occurred\n{}",
            backtrace
        );
    }));
    Ok(())
}

fn format_short_backtrace() -> String {
    let bt = backtrace::Backtrace::new();
    let mut out = String::from("stack backtrace:\n");

    let mut printed = 0usize;
    for frame in bt.frames().iter() {
        if printed >= 64 {
            break;
        }

        let symbol = frame.symbols().first();
        let name = symbol
            .and_then(|s| s.name())
            .map(|n| n.to_string())
            .unwrap_or_else(|| "<unknown>".to_string());

        // 过滤掉 logger 自己的帧，避免 backtrace 里先出现 hook/格式化函数。
        if name.contains("jge_core::logger::format_short_backtrace")
            || name.contains("jge_core::logger::init")
        {
            continue;
        }

        out.push_str(&format!("{:>4}: {}\n", printed, name));
        if let Some(symbol) = symbol {
            if let (Some(file), Some(line)) = (symbol.filename(), symbol.lineno()) {
                if let Some(col) = symbol.colno() {
                    out.push_str(&format!(
                        "             at {}:{}:{}\n",
                        file.display(),
                        line,
                        col
                    ));
                } else {
                    out.push_str(&format!("             at {}:{}\n", file.display(), line));
                }
            }
        }

        printed += 1;
    }

    out
}
