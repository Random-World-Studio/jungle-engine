//! 引擎系统模块。
//!
//! 系统（System）负责对场景/组件进行“跨实体”的处理。
//! - [`logic`]：游戏逻辑抽象与调度（`GameLogic` / `GameLogicHandle`）
//! - [`render`]：渲染系统（按 Layer/场景类型渲染）

pub mod logic;
pub(crate) mod logic_registry;
pub mod render;
