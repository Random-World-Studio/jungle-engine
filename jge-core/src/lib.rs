//! Jungle Engine 的核心库（`jge-core`）。
//!
//! 该 crate 提供引擎运行时（[`Game`]）、事件抽象（[`event`]）、资源系统（[`mod@resource`]）以及 ECS 风格的组件/实体/系统模块（[`game`]）。
//!
//! 大多数游戏项目只需要：
//! - 通过 [`Game`] 创建并运行主循环
//! - 使用 [`game::entity::Entity`] / [`game::component`] API 构建场景
//! - 在 [`game::system::logic::GameLogic`] 中编写游戏逻辑

extern crate self as jge_core;

pub use async_trait::async_trait;

pub mod aabb;
pub mod config;
pub mod event;
pub mod game;
pub mod logger;
pub mod resource;
pub mod scenes;
pub mod text;
mod window;

/// 同步锁封装（对外暴露），用于跨线程共享 winit Window 等同步 API。
///
/// 约定：引擎对外的窗口共享句柄使用这里的 `Mutex`，避免把具体依赖（parking_lot）散落到各个 crate。
pub mod sync {
    pub use parking_lot::{Mutex, MutexGuard};
}

pub use aabb::{Aabb2, Aabb3};
pub use game::Game;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProgressFrame {
    /// 表示接下来进入第 i 阶段（共 n 阶段）。
    ///
    /// 约定：i 从 0 开始计数；n > 0。
    Phase(usize, usize),
    /// 当前阶段进度（范围建议为 0.0 到 1.0，单调不减）。
    Progress(f64),
}

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
/// - 文件会作为编译依赖被追踪：修改 `.jgs` 内容会触发重新编译。
///   （实现方式是宏展开时注入了 `include_str!(...)` 依赖，不依赖任何不稳定编译器特性。）
///
/// # 基本结构（概览）
///
/// - （可选）在最外层声明 `progress(i/n) sender_expr;`，用于汇报构造进度。
///   - 语法有两种：
///     - `progress sender_expr;`（缺省阶段为 `0/1`）
///     - `progress(i/n) sender_expr;`（显式阶段；`i`/`n` 都是 `usize` 表达式；整数字面量会在上下文中推导为 `usize`；`i` 从 `0` 开始计数；`n > 0`）
///   - `sender_expr` 类型为 `tokio::sync::mpsc::Sender<ProgressFrame>`（会被 move 进宏生成的 future；如需复用请传 `sender.clone()`）
///   - 宏会发送：`ProgressFrame::Phase(i, n)` + 若干 `ProgressFrame::Progress(p)`（开始时会尽力先发 `Progress(0.0)`；结束时会尽力发到 `Progress(1.0)`）
///   - 若 `i == n - 1`（最后阶段）：宏会在成功返回前额外发送 `ProgressFrame::Phase(n, n)` 作为“阶段完成”标记（当前实现会在最后一次进度 tick 之后发送）
///   - 可靠性：best-effort（接收端关闭/队列满等导致发送失败会被忽略，不会让场景构造失败）
/// - 你可以写空输入 `scene! {}` / `scene!()` 作为 no-op（返回 `Ok(())`），用于“可选场景”。
/// - 若不是空输入：根必须且仅能有一个 `node { ... }`。
/// - `node` 支持可选项（顺序不限）：
///   - `node "name" { ... }`：节点名（任意 `Into<String>` 表达式）
///   - `node (id = <expr>) { ... }`：指定实体 id
///   - `node ... as ident { ... }`：把实体绑定到局部变量名（在同一 `scene!` 块内可见）
/// - 在 `node` 体内：
///   - `+ CompExpr;`：把组件注册到当前实体（等价于 `e.register_component(CompExpr).await?`）
///   - `+ CompExpr => |e, c| { ... };`：注册前配置组件（`c` 为 `&T`）
///   - `+ CompExpr => |e, mut c| { ... };`：注册前配置组件（`c` 为 `&mut T`）
///   - `resource(name = path, ...) |e, c| { ... }`：在配置闭包前注入资源句柄
///   - `with(...) { ... }`：在块内自动执行 `get_component/get_component_mut`（内部会 `.await`）并提供引用
///   - `* LogicExpr;`：为当前节点挂载 `GameLogic`（宏会在内部把逻辑设置与 `attach` 阶段的生命周期回调衔接起来）
///   - `<expr> node;`：挂载外部已构造好的子节点树。
///     - `<expr>` 可以是另一个 `scene!` 的返回值（匿名 `SceneBindings`），也可以是 [`game::entity::Entity`]。
///     - 若 `<expr>` 为 `SceneBindings`：`destroy().await` 会自动级联调用该嵌套场景的 `destroy().await`。
///     - 若 `<expr>` 为 `Entity`：不会尝试嵌套 destroy（因为 `Entity` 本身不支持 destroy）。
///
/// `scene!` 会展开成一段普通 Rust 语句块，但为了支持 `as ident` 的前向引用，展开是“分阶段”的：
/// - 先创建所有节点实体并填充 `as ident` 绑定；
/// - 再执行节点初始化语句（`node "name"` / `+ ...` / `with(...) { ... }` / `* ...;`）；
/// - 最后按 DSL 中的声明顺序把子节点 `attach` 成树（保证每个节点的子节点顺序与声明一致）。
///
/// 注意：由于 `attach` 在最后阶段才发生，在 `with(...) { ... }` 等初始化块内，节点的父子关系尚未建立。
/// 如果你依赖 `Node::parent/children` 或 `GameLogic::on_attach` 的时机，请把相关逻辑放在宏返回之后（或在运行时由逻辑系统驱动）。
///
/// ## 示例：进度接收端打印日志
///
/// ```no_run
/// use jge_core::ProgressFrame;
/// use tracing::info;
///
/// # async fn demo() -> anyhow::Result<()> {
/// let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressFrame>(64);
/// tokio::spawn(async move {
///     while let Some(frame) = rx.recv().await {
///         match frame {
///             ProgressFrame::Phase(i, n) => info!(phase_i = i, phase_n = n, "scene build phase"),
///             ProgressFrame::Progress(p) => info!(progress = p, "scene build progress"),
///         }
///     }
/// });
///
/// let _bindings = jge_core::scene! {
///     progress(0/1) tx;
///     node "root" {
///         node "a" { }
///         node "b" { }
///     }
/// }
/// .await?;
/// # Ok(())
/// # }
/// ```
///
/// ## 销毁语义：`SceneBindings::destroy().await`
///
/// `scene!` 返回的 `SceneBindings` 提供 `destroy()` 方法（async），用于销毁本次构建出来的场景；调用时需要 `destroy().await`。
///
/// - `destroy().await` 会对场景中每个实体，卸载 DSL 中显式声明的 `+ CompExpr;` 组件。
/// - 依赖关系：[`game::entity::Entity::unregister_component`] 会调用组件的 `unregister_dependencies` 钩子。
/// - 幂等：重复调用 `destroy().await` 不会报错。
///
/// ## 组合子场景：外部构造 + `<expr> node;`
///
/// 你可以先在外部构造一个子场景（拿到它的 `SceneBindings`），再把该 bindings 作为表达式传入并挂载到父场景：
///
/// ```no_run
/// use jge_core::scene;
///
/// # async fn demo() -> anyhow::Result<()> {
/// let child = scene! {
///     node "child_root" {
///         + jge_core::game::component::transform::Transform::new();
///     }
/// }
/// .await?;
///
/// let parent = scene! {
///     node "parent_root" {
///         child node;
///     }
/// }
/// .await?;
///
/// // destroy 会级联 child.destroy()
/// parent.destroy().await;
/// Ok(())
/// # }
/// ```
///
/// 如果你只想“挂载某个外部实体”，也可以传入 `Entity`；此时 destroy 不会触碰该实体上的组件。
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
///   - `embed` / `embeddir`：
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
/// - <res_name>: embed|embeddir|fs
///   from: <path>     # embed/fs 必须
/// - <res_name>: embeddir
///   from: <dir_path> # embeddir 必须（宏在编译期递归扫描目录，并把每个文件按 embed 方式注册到 <res_name>/... 下）
/// - <res_name>: txt
///   txt: |\n...      # txt 必须（必须是 YAML 字符串标量；推荐用 | 块标量）
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
/// - `embeddir`：编译期递归嵌入目录（对目录内每个文件生成 `include_bytes!` 并注册到对应逻辑路径）。
///   - 注意：目录内新增/删除文件是否会自动触发重新编译，取决于构建系统对目录变更的追踪；如遇到不刷新，请手动触发一次重新编译。
///   - 若你不想看到编译期提示，可在调用点加 `#[allow(unused_must_use)]` 包裹该宏调用，或设置环境变量 `JGE_RESOURCE_EMBEDDIR_SILENCE=1`。
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
