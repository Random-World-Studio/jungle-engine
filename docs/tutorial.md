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

    let root = build_scene().context("构建场景失败")?;

    let game = Game::new(GameConfig::default(), root)?;
    game.run()
}

fn build_scene() -> anyhow::Result<jge_core::game::entity::Entity> {
    let bindings = scene! {
        node "root" as root {
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

                // 绑定摄像机（依赖执行顺序：camera 在此已可见）
                with(mut scene: Scene3D) {
                    scene.bind_camera(camera).context("绑定摄像机失败")?;
                    Ok(())
                }
                with(scene: Scene3D) {
                    scene.sync_camera_transform().context("同步摄像机变换失败")?;
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
    }?;

    Ok(bindings.root)
}
```

运行：

```bash
cargo run
```

## 4.（可选）把场景 DSL 放到 `.jgs` 文件

你也可以把 DSL 写到文件里，然后用 `scene!("...")` 加载：

```rust
let bindings = scene!("assets/levels/intro.jgs")?;
```

## 5. 添加游戏逻辑（GameLogic）

引擎里“每帧更新 + 处理事件”的核心抽象是 `GameLogic`：

- `update(entity, delta)`：每帧回调
- `on_event(entity, event)`：每个事件回调

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
        let Some(mut transform) = entity.get_component_mut::<Transform>() else {
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
fn attach_logic_example() -> anyhow::Result<Entity> {
    let bindings = jge_core::scene! {
        node "root" as root {
            node "spin" as spin {
                with(mut node: Node) {
                    node.set_logic(Rotator { speed: 1.2 });
                    Ok(())
                }
            }
        }
    }?;
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

### 7.2 三种资源类型

- `embed`：编译期嵌入（`include_bytes!`），适合纹理/着色器等随二进制分发的资源。
- `fs`：磁盘懒加载（`Resource::from_file`），适合开发期热改或超大文件。
- `txt`：内联文本（写在 YAML 里）。注意：`txt` 必须是 **YAML 字符串标量**，推荐使用 `|` 块标量：

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

2) YAML 中每个节点的 `from:` **资源文件路径**：

- 当使用 `resource!(r#"..."#)` **内联 YAML** 时，`from:` 相对“宏调用点所在源文件”的目录。
- 当使用 `resource!("path/to/resources.yaml")` **从文件读取 YAML** 时，`from:` 相对该 YAML 文件自身所在目录。

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

目前可以通过 `Entity::get_component::<Health>()` / `Entity::get_component_mut::<Health>()` 来访问自定义组件实例。

例如在你的 GameLogic 中在某个事件发生时访问：

```rust
if let Some(mut health) = entity.get_component_mut::<Health>() {
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
    - `Entity::new()` 创建实体（自动带 `Node`）。
    - `entity.register_component(T::new())` 注册组件（会自动补齐依赖）。
    - `entity.get_component::<T>()` 只读访问。
    - `entity.get_component_mut::<T>()` 可写访问。

### 9.4 渲染基础工作流（Layer / Scene2D / Scene3D）

- `jge_core::game::component::layer::Layer`：把一棵节点子树声明为一个渲染层。
- `jge_core::game::component::scene2d::Scene2D`：2D 渲染路径。
- `jge_core::game::component::scene3d::Scene3D`：3D 渲染路径（可绑定摄像机）。
- 常用组合：
    - `Renderable + Transform + Shape (+ Material)`：让实体“能被绘制”。
    - `Camera + Transform`：摄像机实体。
    - `Light + PointLight/ParallelLight (+ Transform)`：光源实体。

### 9.5 资源系统（resource!）

- `jge_core::resource!("resources.yaml")`：注册资源树。
- `jge_core::resource::ResourceHandle`：资源句柄（常用于 `Material`、shader 等组件配置）。

### 9.6 事件与逻辑

- `jge_core::event::Event`：统一事件类型（支持 `Event::custom(T)` + downcast）。
- `jge_core::game::system::logic::GameLogic`：每帧更新与事件回调接口。
- `jge_core::game::component::node::Node::set_logic(...)`：把逻辑挂到节点上（见本教程第 5 节）。

### 9.7 本次文档覆盖范围与后续可补点

本仓库当前已重点补齐/增强了以下“高频入口”的 Rustdoc（用途 + 工作流 + 最小示例）：

- ECS/组件访问：`Entity`、`Component`、`ComponentRead/Write`、`Node`
- 渲染组件：`Layer`、`Scene2D/Scene3D`、`Renderable`、`Transform`、`Camera`、`Shape`、`Material`、`Light`、`Background`

如果你打算继续完善文档，通常最有价值的下一批是：

- `jge-macros`：把 `#[component]` / `#[component_impl]` / `scene!` / `resource!` 的边界条件、常见错误信息做成“FAQ 小节”。
- `jge-cli`：把创建项目/生成模板的命令行参数整理进 README，并配一条“从 0 到能跑”的命令序列。
- `jge-tpl`：把示例里的渲染/资源/输入组织方式，提炼成“推荐目录结构 + 最小工程骨架”。
