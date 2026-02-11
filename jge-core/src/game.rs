//! ECS / 场景树 / 系统模块。
//!
//! 你会在这里找到：
//! - [`entity::Entity`]：实体句柄与组件读写入口
//! - [`component`]：内置组件（渲染、摄像机、灯光、节点树等）与组件访问 guard
//! - [`system`]：逻辑与渲染系统（通常由 [`crate::Game`] 自动驱动）
//!
//! 大多数游戏项目并不需要直接操作系统层；更推荐通过：
//! - [`crate::scene!`] 构建场景树（得到根实体）
//! - 在组件上配置渲染/逻辑能力
//! - 在 [`system::logic::GameLogic`] 中编写每帧更新与事件响应

pub mod component;
pub mod entity;
pub(crate) mod spatial;
pub mod system;

pub(crate) mod reachability;
mod runtime;

pub use runtime::Game;
