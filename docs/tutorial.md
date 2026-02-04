# Jungle Engine 教程：从零搭建一个可运行项目

## 1. 环境要求

- Rust stable（需要支持 Edition 2024，建议使用最新 stable）
- 图形后端：当前未启用 OpenGL；运行机器需要支持 Vulkan / DX / Metal 等现代后端并安装驱动

## 2. 创建项目并添加依赖

在你自己的任意目录执行：

```bash
cargo new my-game
cd my-game
```

添加依赖：

```bash
cargo add anyhow                            # 统一的错误处理
cargo add async-trait                       # 允许你在 trait 中直接编写 async fn 函数
cargo add tokio --features full             # 异步运行时
cargo add tracing --features async-await    # 日志系统
cargo add nalgebra                          # 引擎使用的线性代数数据类型及其相关运算（教程示例会用到 Vector2/Vector3）
```

再添加引擎依赖：

```bash
cargo add jge-core --git https://gitlab.suthby.org/infrastructurepack/jungle-engine.git
```

如果你希望锁定到某个 commit：

```bash
cargo add jge-core --git https://gitlab.suthby.org/infrastructurepack/jungle-engine.git --rev <commit_sha>
```

> 说明：示例代码会使用 `tokio` / `async-trait` / `tracing` 来编写异步逻辑与日志。

## 3. 写一个最小可渲染场景

编辑 `src/main.rs`：

```rust
use anyhow::Context;
use nalgebra::Vector3;

use jge_core::{
    Game,
    config::GameConfig,
    game::{component::node::Node, entity::Entity},
    game::component::{
        camera::Camera,
        light::{Light, ParallelLight, PointLight},
        scene3d::Scene3D,
        shape::Shape,
        transform::Transform,
    },
    logger,
    scene,
};

fn main() -> anyhow::Result<()> {
    logger::init()?;

    // Game 创建前需要一个 root Entity；Entity/ECS API 是 async 的。
    // 这里用一个临时 runtime 仅用于“启动引导”创建 root。
    // Game 启动后请优先使用 `game.block_on` / `game.spawn` 驱动引擎相关 async 任务。
    let bootstrap_rt = tokio::runtime::Runtime::new()?;
    let engine_root = bootstrap_rt.block_on(Entity::new())?;
    let game = Game::new(GameConfig::default(), engine_root)?;

    // 在启动主循环前，用 Game 的 runtime 构建并挂载场景子树。
    game.block_on(build_scene(engine_root)).context("构建场景失败")?;
    game.run()
}

async fn build_scene(engine_root: Entity) -> anyhow::Result<()> {
    let bindings = scene! {
        node "scene_root" as scene_root {
            // 这个节点是一个 3D layer：挂 Scene3D 后会自动补齐 Layer/Renderable/Transform
            node "layer" as layer {
                + Scene3D::new();

                node "camera" as camera {
                    + Camera::new();
                    with(mut transform: Transform) {
                        // 右手系：默认 forward 为 -Z
                        transform.set_position(Vector3::new(0.0, 2.0, 6.0));
                        transform.set_rotation(Vector3::new(-0.25, 0.0, 0.0));
                        Ok(())
                    }
                }

                // 绑定摄像机（`scene!` 的 `as` 绑定支持前向引用/非强顺序）
                with(mut scene: Scene3D) {
                    scene.bind_camera(camera).await.context("绑定摄像机失败")?;
                    Ok(())
                }
                with(scene: Scene3D) {
                    scene
                        .sync_camera_transform()
                        .await
                        .context("同步摄像机变换失败")?;
                    Ok(())
                }

                // 一个地面（两个三角形）
                node "ground" {
                    + Shape::from_triangles(vec![
                        [
                            Vector3::new(-1.0, 0.0, -1.0),
                            Vector3::new(1.0, 0.0, -1.0),
                            Vector3::new(-1.0, 0.0, 1.0),
                        ],
                        [
                            Vector3::new(1.0, 0.0, -1.0),
                            Vector3::new(1.0, 0.0, 1.0),
                            Vector3::new(-1.0, 0.0, 1.0),
                        ],
                    ]);
                    with(mut transform: Transform) {
                        transform.set_scale(Vector3::new(6.0, 1.0, 6.0));
                        Ok(())
                    }
                }

                // 一个三角形
                node "triangle" {
                    + Shape::from_triangles(vec![[
                        Vector3::new(0.0, 0.8, 0.0),
                        Vector3::new(-0.6, -0.4, 0.0),
                        Vector3::new(0.6, -0.4, 0.0),
                    ]]);
                    with(mut transform: Transform) {
                        transform.set_position(Vector3::new(0.0, 0.6, -2.5));
                        Ok(())
                    }
                }

                // 点光源
                node "point_light" {
                    + PointLight::new(12.0);
                    with(mut transform: Transform, mut light: Light) {
                        transform.set_position(Vector3::new(4.0, 3.0, 4.0));
                        light.set_lightness(1.0);
                        Ok(())
                    }
                }

                // 平行光
                node "sun" {
                    + ParallelLight::new();
                    with(mut transform: Transform, mut light: Light) {
                        transform.set_rotation(Vector3::new(-0.3, -std::f32::consts::PI, 0.0));
                        light.set_lightness(0.7);
                        Ok(())
                    }
                }
            }
        }
    }
    .await?;

    let attach_future = {
        let mut root_node = engine_root
            .get_component_mut::<Node>()
            .await
            .expect("引擎根实体应持有 Node 组件");
        root_node.attach(bindings.scene_root)
    };
    attach_future.await.context("挂载场景根节点失败")?;

    Ok(())
}
```

运行：

```bash
cargo run
```

> 关于 `scene!` 的执行语义（与“代码书写顺序”不同）：
>
> - `as ident` 绑定是“先收集并填充”的，因此你可以在任意位置引用同一个 `scene!` 块里导出的实体名（包括前向引用）。
> - 节点树的 `attach` 会在宏的最后阶段集中完成：在 `with(...) { ... }` 等初始化块内，`Node::parent/children` 关系尚未建立。

### （可选）销毁场景：`SceneBindings::destroy().await`

`scene!` 返回的 `SceneBindings` 绑定集会额外提供一个 `destroy()` 方法（async），用于销毁本次构建出来的场景：调用时需要 `destroy().await`。

- 销毁的定义：对场景中每个实体，卸载你在 DSL 中显式注册的组件（也就是 `+ CompExpr;` 那些）。
- 依赖关系：`Entity::unregister_component` 会调用组件的 `unregister_dependencies` 钩子。
- 幂等：重复调用 `destroy().await` 不会报错。

### （可选）构造进度上报：`progress tx;`

当场景比较大时，你可能希望在构造期间汇报进度（例如加载关卡、生成复杂几何、批量注册组件）。
`scene!` 支持在最外层可选地声明：

```rust
progress tx;
```

其中 `tx` 是一个 `tokio::sync::mpsc::Sender<f64>` 变量。宏会在构造流程推进时发送进度值：

- 范围：`0.0` 到 `1.0`
- 语义：单调不减，最后会尽力发送 `1.0`
- 可靠性：发送是 best-effort（接收端关闭时会被忽略，不会导致构造失败）

示例：接收端打印日志

```rust
use tracing::info;

let (progress_tx, mut progress_rx) = tokio::sync::mpsc::channel::<f64>(64);
tokio::spawn(async move {
    while let Some(p) = progress_rx.recv().await {
        info!(progress = p, "scene build progress");
    }
});

let _bindings = scene! {
    progress progress_tx;
    node "root" {
        node "a" { }
        node "b" { }
    }
}
.await?;
```

> 说明：进度的“步数划分”是宏的内部实现细节（创建/初始化/挂载等步骤），不要依赖具体发送次数；只建议用来做 UI/日志展示。

## 4.（可选）把场景 DSL 放到 `.jgs` 文件

你也可以把 DSL 写到文件里，然后用 `scene!("...")` 加载：

```rust
// 该宏返回 Future，因此需要在 async 上下文中 `.await`。
// 在本教程的结构里，建议放进 `game.block_on(async { ... })` 或 `game.spawn(async { ... })` 里执行。
let bindings = scene!("assets/levels/intro.jgs").await?;
```

## 5. 添加游戏逻辑（GameLogic）

引擎里“每帧更新 + 处理事件”的核心抽象是 `GameLogic`：

- `update(entity, delta)`：每帧回调
- `on_event(entity, event)`：每个事件回调
- `on_attach(entity)`：当节点被挂载到父节点下时回调（异步）
- `on_detach(entity)`：当节点从父节点脱离时回调（异步）

通常你会把逻辑挂在某个 Node 上：

```rust
use async_trait::async_trait;
use std::time::Duration;

use jge_core::event::Event;
use jge_core::game::{
    component::{node::Node, transform::Transform},
    entity::Entity,
    system::logic::GameLogic,
};

#[derive(Debug)]
struct Rotator {
    speed: f32,
}

#[async_trait]
impl GameLogic for Rotator {
    async fn on_event(&mut self, _entity: Entity, _event: &Event) -> anyhow::Result<()> {
        Ok(())
    }

    async fn update(&mut self, entity: Entity, delta: Duration) -> anyhow::Result<()> {
        let Some(mut transform) = entity.get_component_mut::<Transform>().await else {
            return Ok(());
        };

        let dt = delta.as_secs_f32();
        let mut rot = transform.rotation();
        rot.y += self.speed * dt;
        transform.set_rotation(rot);
        Ok(())
    }
}

// 在 scene! 里挂逻辑：
async fn attach_logic_example() -> anyhow::Result<Entity> {
    let bindings = jge_core::scene! {
        node "root" as root {
            node "spin" as spin {
                // 推荐语法：直接用 `*` 给当前节点挂逻辑
                * Rotator { speed: 1.2 };
            }
        }
    }
    .await?;
    Ok(bindings.root)
}
```

## 6. 事件映射（把窗口/输入转换成你的游戏事件）

引擎允许你把底层事件（例如 winit 的键盘/鼠标/窗口事件）映射为你自己的“游戏事件”，再由 `GameLogic` 统一处理。

做法是：

1) 在 `Game::new(...).with_*_event_mapper(...)` 中把系统事件转换成 `Event::custom(...)`
2) 在 `GameLogic::on_event` 里对 `Event` 做 `downcast_ref::<YourEvent>()`

最小示例（只演示按键与关闭请求）：

```rust
use jge_core::event::{ElementState, Event, Key, KeyEvent, NamedKey, WindowEvent};

#[derive(Debug)]
enum InputEvent {
    JumpPressed,
}

fn wire_event_mapping(game: jge_core::Game) -> jge_core::Game {
    game.with_event_mapper(|event: &WindowEvent| match event {
        WindowEvent::KeyboardInput { event, .. } => {
            if event.state != ElementState::Pressed {
                return None;
            }
            match &event.logical_key {
                Key::Named(NamedKey::Space) => Some(Event::custom(InputEvent::JumpPressed)),
                _ => None,
            }
        }
        WindowEvent::CloseRequested => Some(Event::CloseRequested),
        _ => None,
    })
}

// 在 GameLogic 里接收：
fn on_event(event: &Event) {
    if let Some(input) = event.downcast_ref::<InputEvent>() {
        match input {
            InputEvent::JumpPressed => {
                // ...
            }
        }
    }
}
```

更完整的“鼠标相对移动 / 滚轮 / WASD”等映射可以直接参考仓库里的示例：
`jge-tpl/src/main.rs`。

## 7. 材质（Material）与资源（Resource）

- 先用 `resource!` 宏注册纹理/着色器等资源
- 再在 `scene!` 里把纹理句柄注入到 `Material` 组件

### 7.1 资源系统（resource! 宏）

引擎提供 `resource!` 宏，用一份 YAML 描述“资源树”，并在编译期展开为一串注册调用。

推荐把 YAML 放在 `src/` 下（和调用点源文件同目录更直观）。例如创建：

- `src/resources.yaml`
- `src/resource/shaders/background_horizon.fs`
- `src/resource/bamboo.png`

然后在 `src/resources.yaml` 写：

```yaml
- shaders:
  - demo:
    - background_horizon.fs: embed
      from: "resource/shaders/background_horizon.fs"
- textures:
  - bamboo.png: embed
    from: "resource/bamboo.png"
```

在 `src/main.rs` 里注册（通常在 `Game::new(...)` 之前调用一次）：

```rust
fn register_resources() -> anyhow::Result<()> {
        // YAML 文件路径相对当前源文件所在目录（此处是 src/）。
        jge_core::resource!("resources.yaml")?;
        Ok(())
}
```

### 7.2 资源类型

- `embed`：编译期嵌入单个文件（`include_bytes!`）。
- `embeddir`：编译期递归嵌入目录内所有文件（每个文件按 `embed` 方式注册到对应逻辑路径）。
    - 注意：目录内新增/删除文件不一定会自动触发重新编译/重新展开；如发现资源列表未更新，请手动触发一次重新编译。
    - 编译期提示说明：该宏会故意触发一个带自定义文案的 `unused_must_use` 警告作为提示（并不代表你的代码真的写错了）。
    - 关闭编译期提示：在调用点包一层 `#[allow(unused_must_use)] { jge_core::resource!(...) ?; }`，或设置环境变量 `JGE_RESOURCE_EMBEDDIR_SILENCE=1`。
- `fs`：磁盘懒加载（`Resource::from_file`），适合开发期热改或超大文件。
- `dir`：运行时目录映射（注册目录节点；访问子路径时才按需从磁盘惰性注册）。
- `txt`：内联文本（写在 YAML 里）。注意：`txt` 必须是 **YAML 字符串标量**，推荐使用 `|` 块标量：
- `bin`：内联二进制（写在 YAML 里；以空格/换行分隔的两位十六进制字节，且不含 0x 前缀）。

```yaml
- config:
    - hello.txt: txt
        txt: |
            hello
            world
```

### 7.3 路径规则（重点）

这里有两层“相对路径”，基准不同：

1) `resource!("...")` 这个 **YAML 文件路径**：相对“宏调用点所在源文件”的目录解析。

2) YAML 中每个节点的 `from:` **资源文件/目录路径**：

- 当使用 `resource!(r#"..."#)` **内联 YAML** 时，`from:`（`embed`/`embeddir`）相对“宏调用点所在源文件”的目录。
- 当使用 `resource!("path/to/resources.yaml")` **从文件读取 YAML** 时，`from:`（`embed`/`embeddir`）相对该 YAML 文件自身所在目录。

这一设计的好处是：当你的资源声明放在单独的 YAML 文件里时，你可以把 **YAML 与它引用的资源文件按相对布局一起移动/复用**（例如把整个资源目录从 `assets/` 移到 `content/`）。
注意：如果你只移动 YAML 而不移动对应资源文件，那么 `from:` 的相对路径会随之改变，通常会导致资源找不到。

### 7.4 材质（Material）使用纹理资源

`Material` 组件需要一个纹理资源句柄（`ResourceHandle`）以及一组 UV patch（每个三角形对应 3 个 UV 坐标）。
你可以先从“只挂纹理资源，UV 之后再补”开始：

1) 先注册纹理资源：

```rust
register_resources()?;
```

2) 在 `scene!` 里挂 `Material`（用 `resource(...)` 把句柄注入到配置闭包里）：

```rust
use jge_core::game::component::{material::Material, shape::Shape};

fn build_with_material() -> anyhow::Result<jge_core::game::entity::Entity> {
    let bindings = jge_core::scene! {
        node "root" as root {
            node "mesh" {
                + Shape::from_triangles(vec![
                    [
                        nalgebra::Vector3::new(0.0, 0.5, 0.0),
                        nalgebra::Vector3::new(-0.5, -0.5, 0.0),
                        nalgebra::Vector3::new(0.5, -0.5, 0.0),
                    ],
                ]);

                + Material::new(texture, Vec::new())
                    => resource(texture = "textures/bamboo.png")
                    |_, mut mat| -> anyhow::Result<()> {
                        // 初学阶段可以先不设置 UV（regions 为空）。
                        // 想贴图时再用 mat.set_regions(...) 填每个三角形的 UV patch。
                        let _ = &mut mat;
                        Ok(())
                    };
            }
        }
    }?;
    Ok(bindings.root)
}
```

仓库示例里包含“立方体 UV patch 计算”的完整写法，也可参考：`jge-tpl/src/main.rs`。

## 8. 自定义组件（用 #[component] / #[component_impl]）

如果你想给实体挂自己的数据（比如生命值、标签、计时器、状态机等），可以写一个自定义组件：

```rust
use jge_core::game::component::{component, component_impl, transform::Transform};

// 这个组件依赖 Transform：注册 Health 时会确保 Transform 已存在。
#[component(Transform)]
#[derive(Debug, Clone)]
pub struct Health {
    hp: i32,
}

#[component_impl]
impl Health {
    // 如果有其它组件依赖本组件，那么在那个组件注册时将使用 Health::new(100) 构造本组件
    #[default(100)]
    pub fn new(hp: i32) -> Self {
        Self { hp }
    }

    pub fn hp(&self) -> i32 {
        self.hp
    }

    pub fn damage(&mut self, amount: i32) {
        self.hp = (self.hp - amount).max(0);
    }
}
```

目前可以通过 `Entity::get_component::<Health>().await` / `Entity::get_component_mut::<Health>().await` 来访问自定义组件实例。

例如在你的 GameLogic 中在某个事件发生时访问：

```rust
if let Some(mut health) = entity.get_component_mut::<Health>().await {
    health.damage(10);
    println!("Entity {:?} 受伤，剩余生命值：{}", entity.id(), health.hp());
}
```

## 9. jge-core API 入口速查（面向游戏开发者）

这一节是一个“去翻源码/Rustdoc 之前的导航”。只列使用入口，不讲底层原理。

### 9.1 运行与窗口

- `jge_core::Game`：创建并运行游戏主循环（见本教程第 3 节）。
- `jge_core::config::GameConfig`：窗口/渲染相关配置。
- `jge_core::logger::init()`：初始化日志。

### 9.2 场景构建（scene!）

- `jge_core::scene! { ... }`：用 DSL 声明节点树并在节点上挂组件。
- 典型模式：
    - `+ SomeComponent::new()`：注册组件
    - `with(mut c: SomeComponent) { ... }`：对组件做一次性配置

### 9.3 ECS/组件访问（Entity）

- `jge_core::game::entity::Entity`：实体句柄。
    - `Entity::new().await` 创建实体（自动带 `Node`）。
    - `entity.register_component(T::new()).await` 注册组件（会自动补齐依赖）。
    - `entity.get_component::<T>().await` 只读访问。
    - `entity.get_component_mut::<T>().await` 可写访问。

> 提示：由于 `Game::new` 需要一个 root Entity，而 `Entity::new().await` 是 async，
> 你可以在创建 `Game` 之前用临时 runtime `block_on` 一次创建 root；
> 创建 `Game` 之后请优先用 `game.block_on` / `game.spawn`。

### 9.4 渲染基础工作流（Layer / Scene2D / Scene3D）

- `jge_core::game::component::layer::Layer`：把一棵节点子树声明为一个渲染层。
- `jge_core::game::component::scene2d::Scene2D`：2D 渲染路径。
- `jge_core::game::component::scene3d::Scene3D`：3D 渲染路径（可绑定摄像机）。
- `Scene2D` 的坐标约定：
    - **视口中心为屏幕原点**（NDC 的 (0,0)），默认 `offset=(0,0)` 时世界坐标 (0,0) 落在视口中心。
    - `offset` 表示“视口中心对应的世界坐标”；`pixels_per_unit` 表示世界单位到像素的缩放。
    - x 向右为正，y 向上为正（注意这与 `LayerViewport` 的归一化坐标原点在左上角是两个不同概念）。
    - 深度（z）约定：开启深度测试并使用 `z` 作为深度值，且 **`z` 必须在 `[0,1]`**；`z` 越大越靠前。
- 常用组合：
    - `Renderable + Transform + Shape (+ Material)`：让实体“能被绘制”。
    - `Camera + Transform`：摄像机实体。
    - `Light + PointLight/ParallelLight (+ Transform)`：光源实体。

补充：如果你需要在运行时实时获取“当前屏幕（或视口）内可见的世界坐标范围”，可以使用：

- `Scene2D::visible_world_bounds()`：获取当前 Layer 的可见范围；如果该 Layer 设置了 `LayerViewport`（分屏/裁剪），会自动读取并应用。
    - 该函数内部使用引擎维护的窗口/渲染目标像素尺寸（在窗口 resize 后的下一帧渲染会自动更新）；尺寸尚未初始化时会返回 `None`。

### 9.5 资源系统（resource!）

- `jge_core::resource!("resources.yaml")`：注册资源树。
- `jge_core::resource::ResourceHandle`：资源句柄（常用于 `Material`、shader 等组件配置）。

### 9.6 事件与逻辑

- `jge_core::event::Event`：统一事件类型（支持 `Event::custom(T)` + downcast）。
- `jge_core::game::system::logic::GameLogic`：每帧更新与事件回调接口。
- `jge_core::game::component::node::Node::set_logic(...)`：把逻辑挂到节点上（返回 Future，需要 `.await`；见本教程第 5 节）。

### 9.7 本次文档覆盖范围与后续可补点

本仓库当前已重点补齐/增强了以下“高频入口”的 Rustdoc（用途 + 工作流 + 最小示例）：

- ECS/组件访问：`Entity`、`Component`、`ComponentRead/Write`、`Node`
- 渲染组件：`Layer`、`Scene2D/Scene3D`、`Renderable`、`Transform`、`Camera`、`Shape`、`Material`、`Light`、`Background`

如果你打算继续完善文档，通常最有价值的下一批是：

- `jge-macros`：把 `#[component]` / `#[component_impl]` / `scene!` / `resource!` 的边界条件、常见错误信息做成“FAQ 小节”。
- `jge-cli`：把创建项目/生成模板的命令行参数整理进 README，并配一条“从 0 到能跑”的命令序列。
- `jge-tpl`：把示例里的渲染/资源/输入组织方式，提炼成“推荐目录结构 + 最小工程骨架”。
