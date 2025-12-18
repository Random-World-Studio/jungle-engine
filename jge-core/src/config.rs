pub enum WindowMode {
    Windowed,
    Fullscreen,
}

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

pub struct GameConfig {
    pub window: WindowConfig,
    pub escape_closes: bool,
    pub game_tick_ms: usize,
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
