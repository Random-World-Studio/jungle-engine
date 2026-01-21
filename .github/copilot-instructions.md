# Jungle Engine（jungle-engine）Copilot 指南（偏引擎开发）

## Repo 地图

- Rust workspace（Edition 2024）：`jge-core`（引擎）、`jge-macros`（proc-macro）、`jge-tpl`/`jge-cli`（bin）。

## 运行时架构（改代码时保持一致）

- 入口 `jge_core::Game`（`jge-core/src/game.rs`）：winit `EventLoop` + 内部 Tokio runtime。
  - 固定 tick：`spawn_update_loop` 以 `GameConfig.game_tick_ms` 驱动 `GameLogic::update`；一次 tick 会等待本轮全部 chunk 完成，避免积压。
  - 事件分发：`EventMapper` 把 `winit::{WindowEvent, DeviceEvent}` 转成 `jge_core::event::Event`（`jge-core/src/event.rs`），再广播到逻辑层。
  - 渲染：`GameWindow::render` 收集 Layer 根并逐层渲染；遇到挂 `Layer` 的实体会停止向下遍历，避免嵌套 Layer 重复渲染（`jge-core/src/window.rs`，并与 `Layer::renderable_entities` 的策略保持一致）。
- 图形后端：wgpu 禁用 OpenGL（`Backends::GL` 被移除），Linux 通常要求 Vulkan 驱动。

## 引擎内部约定（易踩坑）

- `logger::init()` 会安装 panic hook 并输出“简略 backtrace”（`jge-core/src/logger.rs`）；不要在库内部重复初始化订阅者。
- 资源系统是全局树：`ResourceHandle = Arc<RwLock<_>>`，注册/查找靠逻辑路径（`jge-core/src/resource.rs`）。
  - 引擎内置 shader 通过 `include_bytes!` 预注册在 `Resource::resources()`。
- 改动任何“遍历/收集”逻辑时，优先对齐现有单测语义（例如 Layer 的“跳过嵌套 Layer 子树”）。

## 宏与组件系统（修改宏/组件时必读）

- `scene!`（`jge-macros/src/scene_macro.rs`）：
  - 顶层必须且只能有一个根 `node { ... }`；语句严格按书写顺序执行。
  - `with(...) { ... };` 缺组件会直接返回错误（用于“引擎侧约束”而不是可选查询）。
  - rust-analyzer 环境会短路：文件读取/解析在编辑器里可能“看起来能过但实际 build 会失败”。
- `resource!`（`jge-macros/src/resource_macro.rs`）：
  - 内联 YAML：`from:` 相对宏调用点源文件目录；从文件读取 YAML：`from:` 相对 YAML 文件目录。
- `#[component]` / `#[component_impl]`（`jge-macros/src/component_attr.rs`、`.../component_impl_attr.rs`）：
  - `#[component(Deps...)]` 会在注册时确保依赖组件存在（通过 dep 的 `__jge_component_default`）。
  - `#[component]` 不支持泛型；struct 含 `entity_id` 字段会自动 attach/detach。

## 开发工作流

- 跑示例/手动验证：`cargo run -p jge-tpl`（输入/窗口边界条件在这里最容易复现）。
- 运行 CLI：`cargo run -p jge-cli`（目前较轻量）。
- 核心回归：`cargo test -p jge-core`（包含大量语义单测与 doctest）。
- 格式化：`cargo fmt` / `cargo fmt --check`。

## 快速定位（引擎开发优先看）

- 主循环/调度：`jge-core/src/game.rs`
- 事件抽象与映射器：`jge-core/src/event.rs`
- 渲染入口与 Layer 根收集：`jge-core/src/window.rs`
- ECS 与组件实现：`jge-core/src/game/component/`、`jge-core/src/game/system/`
- 宏实现：`jge-macros/src/*_macro.rs`、`jge-macros/src/component_*_attr.rs`
