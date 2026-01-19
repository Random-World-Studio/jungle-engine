/// 窗口模式。
pub enum WindowMode {
    Windowed,
    Fullscreen,
}

/// 窗口配置。
///
/// 用于控制窗口标题、初始尺寸、是否开启垂直同步等。
///
/// # 示例
///
/// ```
/// let mut cfg = ::jge_core::config::WindowConfig::default();
/// cfg.title = "My Game".to_string();
/// cfg.width = 1280;
/// cfg.height = 720;
/// cfg.vsync = true;
/// ```
pub struct WindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub mode: WindowMode,
    pub vsync: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: String::from("Jungle Engine"),
            width: 1920,
            height: 1080,
            mode: WindowMode::Windowed,
            vsync: true,
        }
    }
}

/// 引擎运行配置。
///
/// - `window`：窗口相关配置
/// - `escape_closes`：是否允许按下 `Esc` 关闭窗口
/// - `game_tick_ms`：逻辑更新 tick 间隔（毫秒）
///
/// # 示例
///
/// ```
/// let mut cfg = ::jge_core::config::GameConfig::default();
/// cfg.game_tick_ms = 16; // 约 60 tick/s
/// cfg.escape_closes = true;
/// cfg.window.width = 1280;
/// cfg.window.height = 720;
/// ```
pub struct GameConfig {
    pub window: WindowConfig,
    pub escape_closes: bool,
    pub game_tick_ms: u64,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            window: WindowConfig::default(),
            escape_closes: true,
            game_tick_ms: 50,
        }
    }
}
