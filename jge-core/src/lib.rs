//! Jungle Engine 的核心库（`jge-core`）。
//!
//! 该 crate 提供引擎运行时（[`Game`]）、事件抽象（[`event`]）、资源系统（[`resource`]）以及 ECS 风格的组件/实体/系统模块（[`game`]）。
//!
//! 大多数游戏项目只需要：
//! - 通过 [`Game`] 创建并运行主循环
//! - 使用 [`game::entity::Entity`] / [`game::component`] API 构建场景
//! - 在 [`game::system::logic::GameLogic`] 中编写游戏逻辑

extern crate self as jge_core;

pub mod config;
pub mod event;
pub mod game;
pub mod logger;
pub mod resource;
mod window;

pub use game::Game;

