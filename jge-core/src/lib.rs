//! Jungle Engine 的核心库（`jge-core`）。
//!
//! 该 crate 提供引擎运行时（[`Game`]）、事件抽象（[`event`]）、资源系统（[`mod@resource`]）以及 ECS 风格的组件/实体/系统模块（[`game`]）。
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
pub mod text;
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
///   - `* LogicExpr;`：为当前节点挂载 `GameLogic`（宏会在内部把逻辑设置与 `attach` 阶段的生命周期回调衔接起来）
///
/// `scene!` 会展开成一段普通 Rust 语句块，但为了支持 `as ident` 的前向引用，展开是“分阶段”的：
/// - 先创建所有节点实体并填充 `as ident` 绑定；
/// - 再执行节点初始化语句（`node "name"` / `+ ...` / `with(...) { ... }` / `* ...;`）；
/// - 最后按 DSL 中的声明顺序把子节点 `attach` 成树（保证每个节点的子节点顺序与声明一致）。
///
/// 注意：由于 `attach` 在最后阶段才发生，在 `with(...) { ... }` 等初始化块内，节点的父子关系尚未建立。
/// 如果你依赖 `Node::parent/children` 或 `GameLogic::on_attach` 的时机，请把相关逻辑放在宏返回之后（或在运行时由逻辑系统驱动）。
///
/// # 示例
///
/// ```no_run
/// async fn build_root() -> ::anyhow::Result<::jge_core::game::entity::Entity> {
///     // `scene!` 返回 Future，需要在某个 async 上下文中 `.await`。
///     // 在实际游戏项目中，推荐通过 `Game::block_on` 或 `Game::spawn` 来驱动这些 async 构建任务。
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
///     }
///     .await?;
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
///     // scene! 现在返回 Future，因此需要在 async runtime 中 await
///     // let _bindings = scene!("assets/levels/intro.jgs").await?;
///     Ok(())
/// }
/// ```
pub use jge_macros::scene;

/// 资源 DSL 宏：用 YAML 描述一棵“资源树”，并在编译期展开成一串资源注册调用：
/// [`resource::Resource::register`](crate::resource::Resource::register) /
/// [`resource::Resource::register_dir`](crate::resource::Resource::register_dir)。
///
/// 该宏由 `jge-core` 重导出，因此游戏项目只需要依赖 `jge-core` 即可使用。
///
/// # 输入形式
///
/// - **内联 YAML**：`resource!(r#"..."#)`
/// - **从文件读取**：`resource!("path/to/resources.yaml")`
///
/// 当参数字符串以 `.yaml`/`.yml` 结尾时，宏会把它当作文件路径解析：
/// - 文件 **必须存在**，否则在编译期报错。
/// - 文件内容会作为编译依赖被追踪（修改 YAML 会触发重新编译）。
///
/// # 路径规则
///
/// - 资源逻辑路径：由 YAML 的目录层级拼接而成，使用 `/` 作为分隔符，例如 `textures/ui/button.png`。
/// - `from` 相对路径：
///   - `embed`：
///     - **内联 YAML**：以 **宏调用点源代码文件的父目录** 为基准解析。
///     - **从文件读取 YAML**：以该 **YAML 文件所在目录** 为基准解析。
///   - `fs` / `dir`：若 `from` 不是绝对路径，则在**运行时**按进程的当前工作目录（cwd）解析。
///
/// # YAML 语法（精简 BNF）
///
/// 顶层必须是一个列表：
///
/// ```text
/// - <dir_name>: [ <node>... ]
/// - <res_name>: embed|fs
///   from: <path>     # embed/fs 必须
/// - <res_name>: txt
///   txt: |\n...      # txt 必须（必须是 YAML 字符串标量；推荐用 | 块标量）
/// - <res_name>: bin
///   bin: |\n...      # bin 必须（多个空格/换行分隔的两位十六进制字节，不含 0x 前缀）
/// - <res_name>: bin
///   bin: |\n...      # bin 必须（多个空格/换行分隔的两位十六进制字节，不含 0x 前缀）
/// - <res_name>: dir
///   from: <dir_path>  # dir 必须（文件系统路径；宏只注册“运行时目录映射”节点，不会在编译期遍历/检查目录内容）
/// ```
///
/// 约束：
/// - 目录名/资源名（每个路径段）不能为空，且 **不允许包含 `/`**。
/// - 同一个宏展开内的逻辑路径不能重复。
/// - `embed`/`fs` 不允许出现 `txt` 字段；`txt` 不允许出现 `from` 字段。
///
/// # 五种资源类型
///
/// - `embed`：编译期嵌入（`include_bytes!`），适合纹理/着色器等随二进制分发的资源。
/// - `fs`：磁盘懒加载（`Resource::from_file`），适合开发期热改或超大文件。
/// - `txt`：内联文本（写在 YAML 里），适合小配置/小片段。
/// - `bin`：内联二进制（写在 YAML 里），适合小型二进制 blob（例如调试用纹理/自定义表）。
/// - `dir`：运行时目录映射（`Resource::register_dir`）。
///   - 不会在编译期/注册期扫描磁盘。
///   - 访问 `dir/...` 下的具体文件时（[`resource::Resource::from`](crate::resource::Resource::from)），才会按需探测并惰性注册为 `fs` 资源。
///   - 对映射目录调用 [`resource::Resource::list_children`](crate::resource::Resource::list_children) 会读取磁盘目录并把一级子项懒注册到资源树。
///
/// # 示例：内联 YAML
///
/// ```no_run
/// fn register_resources_inline() -> ::anyhow::Result<()> {
///     ::jge_core::resource!(r#"
/// - textures:
///   - hello.txt: txt
///     txt: |
///       hello
///       world
/// "#)?;
///     Ok(())
/// }
/// ```
///
/// # 示例：从文件读取
///
/// ```ignore
/// fn register_resources_from_file() -> ::anyhow::Result<()> {
///     // 说明：
///     // - YAML 文件路径（assets/resources.yaml）仍然相对“本行所在源文件的父目录”。
///     // - embed 的 `from:` 相对路径仍相对 YAML 文件自身所在目录。
///     // - fs/dir 的 `from:` 相对路径改为运行时按 cwd 解析。
///     ::jge_core::resource!("assets/resources.yaml")?;
///     Ok(())
/// }
/// ```
pub use jge_macros::resource;

#[cfg(test)]
mod scene_macro_tests;
