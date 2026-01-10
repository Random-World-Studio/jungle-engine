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

/// 场景 DSL 宏：用声明式语法构建一棵 `Node` 树，并返回绑定集。
///
/// 这个宏由 `jge-core` 重导出，因此游戏项目只需要依赖 `jge-core` 即可使用。
///
/// # 两种用法
///
/// - **内联 DSL**：`scene! { ... }`
/// - **从文件加载**：`scene!("path/to/file.jgs")`（在编译期读取文件内容并生成同样的构建代码）
///
/// ## 文件路径与自动重编译
///
/// - 默认（stable）：相对路径会以调用方 crate 的 `CARGO_MANIFEST_DIR/src` 为基准解析。
/// - 可选（nightly + 启用 `jge-core/callsite-relative-paths`）：相对路径会以 **宏调用点所在源代码文件的父目录** 为基准解析（语义与 `include_str!("...")` 一致）。
/// - 文件会作为编译依赖被追踪：修改 `.jgs` 内容会触发重新编译。
///   （实现方式是宏展开时注入了 `include_str!(...)` 依赖，不依赖任何不稳定编译器特性。）
///
/// # 基本结构（概览）
///
/// - 根必须且仅能有一个 `node { ... }`。
/// - `node` 支持可选项（顺序不限）：
///   - `node "name" { ... }`：节点名（任意 `Into<String>` 表达式）
///   - `node (id = <expr>) { ... }`：指定实体 id
///   - `node ... as ident { ... }`：把实体绑定到局部变量名（在同一 `scene!` 块内可见）
/// - 在 `node` 体内：
///   - `+ CompExpr;`：把组件注册到当前实体（等价于 `e.register_component(CompExpr)`）
///   - `+ CompExpr => |e, c| { ... };`：注册前配置组件（`c` 为 `&T`）
///   - `+ CompExpr => |e, mut c| { ... };`：注册前配置组件（`c` 为 `&mut T`）
///   - `resource(name = path, ...) |e, c| { ... }`：在配置闭包前注入资源句柄
///   - `with(...) { ... }`：在块内自动执行 `get_component/get_component_mut` 并提供引用
///
/// `scene!` 展开后是一段普通 Rust 语句块：**严格按书写顺序执行**，并遵循 Rust 的作用域规则。
///
/// # 示例
///
/// ```no_run
/// fn build_root() -> ::anyhow::Result<::jge_core::game::entity::Entity> {
///     let bindings = ::jge_core::scene! {
///         node "root" as root {
///             node "camera" as camera {
///                 + ::jge_core::game::component::renderable::Renderable::new();
///                 + ::jge_core::game::component::transform::Transform::new();
///                 + ::jge_core::game::component::camera::Camera::new();
///             }
///
///             // 写命令式逻辑时用 with() { ... }。
///             with() {
///                 let _ = (e, root, camera);
///                 Ok(())
///             }
///         }
///     }?;
///
///     Ok(bindings.root)
/// }
/// ```
///
/// 从文件加载（推荐 `.jgs`）：
///
/// ```ignore
/// use jge_core::scene;
///
/// fn build() -> anyhow::Result<()> {
///     let _bindings = scene!("assets/levels/intro.jgs")?;
///     Ok(())
/// }
/// ```
pub use jge_macros::scene;
