# Jungle Engine 教程（从零搭建一个可运行的项目）

本教程面向第一次接触本仓库的开发者，目标是：**从零创建一个“独立的 Rust crate”（不在本 workspace 内），并一步一步搭建出能开窗、能响应输入、能挂逻辑、能渲染的最小项目**。

> 说明：当前 `jge-cli` 仍是占位（仅输出 `Hello, world!`）。本教程会教你在独立项目里通过远程 git 依赖使用 `jge-core` 与 `jge-macros`。

## 1.5 先用 2 分钟理解 ECS（没接触过也能继续看）

很多游戏引擎会用 ECS（Entity-Component-System）来组织“场景里的东西”。你可以把它理解成一种把**数据**和**行为**拆开管理的方式：

- **Entity（实体）**：只是一个 id / 句柄，本身不带“属性”。它的意义是“把一组组件聚在一起”。
- **Component（组件）**：挂在实体上的数据块（例如：位置 `Transform`、几何 `Shape`、材质 `Material`、相机 `Camera`）。组件通常只存数据，不负责驱动流程。
- **System（系统）**：遍历并处理“拥有某些组件组合”的实体，执行行为（例如：每帧更新位置、处理输入事件、收集可渲染三角形并提交到 GPU）。

这种拆分的好处是：

- 组合更灵活：一个实体“是什么”由它挂了哪些组件决定（同一套组件可以拼出不同类型对象）。
- 数据局部性更好：系统往往只关心某几类组件，可以更高效地批处理。

在 Jungle Engine 里，你会在教程里不断遇到这些对应关系：

- 你用 `Entity::new()` 创建实体，然后用 `entity.register_component(...)` 给它挂组件。
- `Node` 是一种组件，用来把实体组织成树（场景层级）。
- `GameLogic` 是一种系统：引擎会在 tick/事件发生时调用它，让你驱动实体变化。

你不需要一次性掌握 ECS 的所有理论；跟着第 3、5、7、8、9 章动手做几次，你会很自然地把这些概念串起来。

## 1. 仓库结构（你在看什么）

这是一个 Rust workspace，包含：

- `jge-core/`：引擎核心库（窗口、渲染、组件系统、事件与逻辑调度等）
- `jge-macros/`：过程宏，提供 `#[component]` / `#[component_impl]`
- `jge-tpl/`：模板/演示项目（可以作为“实现参考”，但本教程不会教你去运行它）
- `jge-cli/`：CLI（目前未实现）

本教程会新建一个独立 crate（下面用 `my-game` 作为名字）。

---

## 2. 一步一步搭建 my-game

这一节是主线：每完成一步，你都应该能编译/运行并看到进展。

### 2.1 环境要求

- Rust toolchain：需要支持 **Edition 2024** 的 stable（建议使用最新 stable）
- 图形后端：当前**未启用 OpenGL**；运行机器需要支持 **Vulkan / DirectX / Metal** 等现代图形后端，并已正确安装显卡驱动

### 2.2 创建一个 crate

在你自己的任意目录执行：

```bash
cargo new my-game
cd my-game
```

### 2.3 通过远程 git 依赖 jge-core

> 把下面的 `<ENGINE_GIT_URL>` 替换成本仓库地址。

先添加常用依赖：

```bash
cargo add anyhow                            # 统一的错误处理
cargo add async-trait                       # 允许你在 trait 中直接编写 async fn 函数
cargo add tokio --features full             # 异步运行时
cargo add tracing --features async-await    # 日志系统
cargo add nalgebra                          # 引擎使用的线性代数数据类型及其相关运算（教程示例会用到 Vector2/Vector3）
```

再添加引擎依赖：

```bash
cargo add jge-core --git <ENGINE_GIT_URL>
```

如果你希望锁定到某个 commit：

```bash
cargo add jge-core --git <ENGINE_GIT_URL> --rev <commit_sha>
```

> 说明：示例代码会使用 `tokio` / `async-trait` / `tracing` 来编写异步逻辑与日志。

### 2.4 写出第一版 main：开窗并进入事件循环

编辑 `my-game/src/main.rs`：

```rust
use anyhow::Context;

use jge_core::{

    Game,
    config::GameConfig,
    game::entity::Entity,
    logger,
};

fn main() -> anyhow::Result<()> {

    logger::init()?;


    // 从“只有一个根实体”的最小世界开始。
    let root = Entity::new().context("创建根实体失败")?;


    let game = Game::new(GameConfig::default(), root)?;
    game.run()
}
```

运行：

```bash
cargo run
```

如果这一步能弹出窗口（哪怕黑屏），说明你已经完成了：

- 正确接入 `jge-core`
- 成功创建 `Game`
- 进入引擎主循环

接下来，我们会把“黑屏窗口”逐步变成一个真正的场景，并在这个过程中引出你必须理解的几个概念。

---

## 3. 核心概念：Entity / Component / Node 树（你在第 2.4 步已经用上了）

到第 2.4 步时你创建了 `Entity::new()`，这会立刻引出一个关键事实：**Jungle Engine 的场景组织是以 Node 树为骨架，而 Node 本身是一个组件。**

### 3.1 Entity 是“组件容器”

`Entity` 是一个轻量句柄（本质是 id），组件通过类型静态存储进行挂载。

- `Entity::new()`：创建实体，并**自动挂载 `Node` 组件**（用于形成场景树）
- `entity.register_component(T::new())`：给实体挂组件
- `entity.get_component::<T>()` / `entity.get_component_mut::<T>()`：读写组件

### 3.2 强制的“组件工作流”规范（非常重要）

文档中已明确要求：**所有组件挂载与读取必须通过 `Entity` API**，不要直接操作 `Component::insert`/`storage` 等底层接口。

原因是：组件依赖补齐与生命周期行为都依赖这条标准路径；绕开它容易造成行为不一致。

---

## 4. Game：事件循环、tick 与渲染（我们马上要开始改造）

核心类型 `Game` 负责驱动主循环。典型启动流程就是你在第 2.4 步写的那四行：

1) 构造根实体 `root: Entity`
2) `Game::new(GameConfig::default(), root)`
3) （可选）设置窗口初始化回调、事件映射器
4) `game.run()` 进入引擎主循环

### 4.1 GameConfig / WindowConfig

- `WindowConfig { title, width, height, mode, vsync }`
- `GameConfig { window, escape_closes, game_tick_ms }`

其中 `game_tick_ms` 控制逻辑 update 的 tick 间隔（默认 50ms）。

现在我们回到“搭建主线”：下一步是让你的程序“动起来”。

---

## 5. GameLogic：把行为挂到 Node 上（第一个可观察的变化）

逻辑 trait `GameLogic` 允许你把行为挂到某个 `Node` 上。你可以实现：

- `async fn update(&mut self, e: Entity, delta: Duration)`：每 tick 调一次
- `async fn on_event(&mut self, e: Entity, event: &Event)`：发生事件时调用
- 以及 `on_attach/on_detach/on_render` 等

### 5.1 在 my-game 里挂一个最小逻辑

把下面的逻辑挂到 `root` 的 `Node` 上：

```rust
use async_trait::async_trait;

use jge_core::game::{

    component::node::Node,
    entity::Entity,
    system::logic::GameLogic,
};

struct PrintLogic;

#[async_trait]
impl GameLogic for PrintLogic {

    async fn update(&mut self, _e: Entity, _delta: std::time::Duration) -> anyhow::Result<()> {
        tracing::info!(target = "my-game", "tick");
        Ok(())
    }
}

// main 中创建 root 之后：
root.get_component_mut::<Node>().unwrap().set_logic(PrintLogic);
```

重新运行后，你应该能看到日志持续输出，这证明：

- `Game` 的 tick loop 在跑
- 你挂在 Node 上的 logic 会被调度

下一步：让输入事件进到你的 logic。

---

## 6. 事件映射：WindowEvent / DeviceEvent → 引擎 Event（把输入“喂”给逻辑）

引擎提供两条“翻译通道”，把平台事件映射为引擎事件：

- `with_window_event_mapper(...)`：把窗口事件转成引擎 `Event`
- `with_device_event_mapper(...)`：把设备事件转成引擎 `Event`

引擎事件目前以 `Custom` 为主：

- `Event::custom(MyEvent { ... })`
- 在逻辑里用 `event.downcast_ref::<MyEvent>()` 拿回你的类型

### 6.1 在 my-game 里做一个最小输入事件

定义一个自定义事件，并在 mapper 里把键盘输入映射成它：

```rust
#[derive(Debug)]
enum MyInput {

    PressedSpace,
}

// 构造 game 时加上 mapper：
let game = Game::new(GameConfig::default(), root)?

    .with_window_event_mapper(|event| {
        use jge_core::event::{ElementState, Event, Key, NamedKey, WindowEvent};


        match event {
            WindowEvent::KeyboardInput { event, .. }
                if matches!(event.state, ElementState::Pressed)
                    && matches!(&event.logical_key, Key::Named(NamedKey::Space)) =>
            {
                Some(Event::custom(MyInput::PressedSpace))
            }
            _ => None,
        }
    });
```

然后在你的 `GameLogic::on_event` 里处理：

```rust
use jge_core::event::Event;

async fn on_event(&mut self, _e: Entity, event: &Event) -> anyhow::Result<()> {

    if let Some(MyInput::PressedSpace) = event.downcast_ref::<MyInput>() {
        tracing::info!(target = "my-game", "space pressed");
    }
    Ok(())
}
```

到这里你已经具备了：

- 一个可以跑的世界（root + Node）
- 一个持续更新的逻辑（update）
- 一条输入事件通路（mapper → Event → on_event）

下一步就是：让画面从“黑屏”变成“有东西可渲染”。

---

## 7. Layer / Scene：怎么让东西被渲染出来（把黑屏变成可见内容）

要“看到画面”，你需要在 `Node` 树上提供一个渲染层入口：

- 渲染系统（RenderSystem）会从根实体开始遍历 `Node` 树
- **遇到带 `Layer` 组件的实体就视为渲染层根**，并渲染这一层

因此要“看到画面”，最少需要：

- 树上某个实体挂 `Layer`
- 该 layer 里有可渲染内容（比如 Scene3D/Renderable/Material/Shape 等组合）

### 7.1 这一章的目标：你要“亲手搭出”一个最小 3D 场景

在开始写代码之前，先记住一个最小清单（按重要性从高到低）：

1) 一棵 `Node` 树（你已经有了：`Entity::new()` 会自动挂 `Node`）
2) 树上存在一个 `Layer` 根（渲染入口）
3) 这个 layer 下有一个场景组件（`Scene3D` 或 `Scene2D`）
4) 这个场景里至少有一个“可渲染实体”（需要 `Renderable + Transform + Shape`）
5) 3D 场景需要一个摄像机（`Camera` + `Transform`），并绑定到 `Scene3D`

下面我们用 3D 场景演示（2D 的思路类似）。

### 7.2 创建一个 Scene3D 作为 layer 根节点

在 `main()` 中创建 `root` 后，按下面步骤创建 `scene` 实体，并把它挂到 `root` 下面：

```rust
use anyhow::Context;

use jge_core::game::{
    component::{
        node::Node,
        scene3d::Scene3D,
    },
    entity::Entity,
};

fn build_scene3d_root(root: Entity) -> anyhow::Result<Entity> {
    let scene = Entity::new().context("创建 Scene3D 实体失败")?;

    // Scene3D 会自动补齐依赖（Layer / Renderable / Transform）。
    let _ = scene
        .register_component(Scene3D::new())
        .context("挂载 Scene3D 失败")?;

    // 把 scene 挂到 root 的 Node 树下面，形成层级关系。
    root.get_component_mut::<Node>()
        .expect("root 应持有 Node")
        .attach(scene)
        .context("把 scene 挂到 root 失败")?;

    Ok(scene)
}
```

你现在已经满足了“渲染入口”的最小条件：树上出现了一个 `Layer` 根（它通过 `Scene3D` 的依赖补齐被自动注册）。

### 7.3 创建摄像机并绑定到 Scene3D

`Scene3D` 需要一个摄像机实体：

```rust
use jge_core::game::component::{
    camera::Camera,
    node::Node,
    renderable::Renderable,
    transform::Transform,
};
use nalgebra::Vector3;

fn spawn_camera(scene: Entity) -> anyhow::Result<Entity> {
    let camera = Entity::new().context("创建 Camera 实体失败")?;

    let _ = camera
        .register_component(Renderable::new())
        .context("挂 Renderable 失败")?;
    let _ = camera
        .register_component(Transform::new())
        .context("挂 Transform 失败")?;
    let _ = camera
        .register_component(Camera::new())
        .context("挂 Camera 失败")?;

    // 坐标系：右手系，-Z 是“前”。因此把相机放到 +Z，看向原点方向。
    {
        let mut t = camera
            .get_component_mut::<Transform>()
            .expect("camera 应持有 Transform");
        t.set_position(Vector3::new(0.0, 1.0, 3.0));
    }

    scene
        .get_component_mut::<Node>()
        .expect("scene 应持有 Node")
        .attach(camera)
        .context("把 camera 挂到 scene 失败")?;

    scene
        .get_component_mut::<Scene3D>()
        .expect("scene 应持有 Scene3D")
        .bind_camera(camera)
        .context("绑定摄像机失败")?;

    Ok(camera)
}
```

此时 3D 场景已经具备“观察者”。接下来我们创建一个真正能渲染的实体。

### 7.4 创建第一个可渲染实体：一个三角形

要让实体参与渲染收集，需要满足：

- 有 `Renderable` 且 enabled（默认就是 enabled）
- 有 `Transform`（用于变换到世界坐标）
- 有 `Shape`（提供三角形几何）

```rust
use jge_core::game::component::{
    node::Node,
    renderable::Renderable,
    shape::Shape,
    transform::Transform,
};
use nalgebra::Vector3;

fn spawn_triangle(scene: Entity) -> anyhow::Result<Entity> {
    let tri = Entity::new().context("创建三角形实体失败")?;

    let _ = tri
        .register_component(Renderable::new())
        .context("挂 Renderable 失败")?;
    let _ = tri
        .register_component(Transform::new())
        .context("挂 Transform 失败")?;

    // 一个位于原点附近的三角形（面朝 +Z 或 -Z 取决于顶点顺序；背面剔除开启）。
    let shape = Shape::from_triangles(vec![[
        Vector3::new(-0.6, 0.0, 0.0),
        Vector3::new(0.6, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
    ]]);
    let _ = tri.register_component(shape).context("挂 Shape 失败")?;

    scene
        .get_component_mut::<Node>()
        .expect("scene 应持有 Node")
        .attach(tri)
        .context("把三角形挂到 scene 失败")?;

    Ok(tri)
}
```

### 7.5（可选但推荐）加一个点光源，让 3D 更容易观察

Scene3D 的光源也是“实体 + 组件”，并且需要位于同一棵 layer 树下。

```rust
use jge_core::game::component::{
    light::{Light, PointLight},
    node::Node,
    renderable::Renderable,
    transform::Transform,
};
use nalgebra::Vector3;

fn spawn_point_light(scene: Entity) -> anyhow::Result<Entity> {
    let light = Entity::new().context("创建灯光实体失败")?;

    let _ = light
        .register_component(Renderable::new())
        .context("挂 Renderable 失败")?;
    let _ = light
        .register_component(Transform::new())
        .context("挂 Transform 失败")?;
    let _ = light
        .register_component(Light::new(1.0))
        .context("挂 Light 失败")?;
    let _ = light
        .register_component(PointLight::new(8.0))
        .context("挂 PointLight 失败")?;

    {
        let mut t = light
            .get_component_mut::<Transform>()
            .expect("light 应持有 Transform");
        t.set_position(Vector3::new(2.0, 3.0, 2.0));
    }

    scene
        .get_component_mut::<Node>()
        .expect("scene 应持有 Node")
        .attach(light)
        .context("把 light 挂到 scene 失败")?;

    Ok(light)
}
```

### 7.6 把这些步骤串起来：在 main() 里调用

现在你可以在 `main()` 中这样组装：

```rust
let root = Entity::new()?;

let scene = build_scene3d_root(root)?;
let _camera = spawn_camera(scene)?;
let _tri = spawn_triangle(scene)?;
let _light = spawn_point_light(scene)?;

let game = Game::new(GameConfig::default(), root)?;
game.run()
```

如果你仍然是黑屏，优先按下面顺序检查：

1) `scene` 是否真的挂在 `root` 的 `Node` 下（`Node::attach` 是否返回 Ok）
2) `scene` 是否成功挂载 `Scene3D`（它会自动补齐 `Layer`）
3) 是否调用了 `bind_camera`，且 camera 有 `Camera + Transform`
4) 三角形实体是否有 `Renderable + Transform + Shape`

---

## 8. 材质与资源：给实体贴图（贴图/Shader 等）

资源系统允许你用“路径”作为 key，注册与读取二进制资源：

- `ResourcePath::from("textures/bamboo.png")`：用类似路径的分段表示资源 key
- `ResourceHandle<T = Resource>`：资源句柄类型（默认 `T = Resource`），用于共享与按需加载
- `Resource::from_memory(bytes)` / `Resource::from_file(path)`：直接创建 `ResourceHandle`
- `Resource::register(path, handle)`：注册资源（你也可以直接写 `Resource::register(path, Resource::from_memory(bytes))`）
- `Resource::from(path)`：按路径取资源句柄

> 说明：当前 API 里 `Resource::from_*` 会直接返回 `ResourceHandle`，因此你不需要再手写 `Arc::new(RwLock::new(...))`。

### 8.1 这一章的目标：给你的三角形“贴一张图”

你已经在第 7 章里创建了一个三角形实体。现在我们做两件事：

1) 把一张 png 注册进资源系统（`Resource::register`）
2) 给三角形实体挂 `Material` 组件，并提供 UV（`regions`）

> 提醒：如果你暂时不挂 `Material`，引擎会使用默认材质（你仍然应该能看到几何体）。

### 8.2 准备一张 png（你自己的素材）

在 `my-game` 项目里新建一个素材文件，例如：

- `my-game/assets/texture.png`

你可以放任意 png（自己画、自己导出、或用工具生成都可以）。教程不依赖示例工程素材。

### 8.3 注册资源：ResourcePath → ResourceHandle

```rust
use jge_core::resource::{Resource, ResourcePath};

fn register_texture() -> anyhow::Result<jge_core::resource::ResourceHandle> {
    // 资源路径是逻辑 key，不要求与磁盘路径一致。
    let path = ResourcePath::from("textures/texture.png");

    Resource::register(
        path.clone(),
        Resource::from_memory(Vec::from(include_bytes!("../assets/texture.png"))),
    )?;

    let handle = Resource::from(path).expect("资源注册后应能取回句柄");
    Ok(handle)
}
```

### 8.4 给实体挂 Material：resource + UV regions

`Material` 的关键字段有两个：

- `resource`：贴图资源句柄
- `regions`：每个三角形对应一组三个 UV（按顶点顺序）

对我们第 7.4 节的单个三角形来说，`regions` 只需要 1 个 patch：

```rust
use jge_core::game::component::material::Material;
use nalgebra::Vector2;

fn apply_material(tri: Entity, texture: jge_core::resource::ResourceHandle) -> anyhow::Result<()> {
    // 这个三角形只有 1 个面，所以 regions 只放 1 个 patch。
    // patch[vertex_index] 对应 triangle[vertex_index]。
    let patch = [
        Vector2::new(0.0, 1.0),
        Vector2::new(1.0, 1.0),
        Vector2::new(0.5, 0.0),
    ];

    let material = Material::new(texture, vec![patch]);
    let _ = tri.register_component(material)?;
    Ok(())
}
```

### 8.5 两个三角形（一个四边形）时 regions 怎么写？

引擎在渲染时会用 `triangle_index` 去索引 `regions[triangle_index]`：

- 第 0 个三角形 → `regions[0]`
- 第 1 个三角形 → `regions[1]`

因此当你用两个三角形拼一个矩形时，你需要提供 2 个 patch（哪怕 UV 一样也要写两份）。

### 8.6 常见问题：贴图失败怎么办？

如果你看到类似“failed to prepare 3D material texture, fallback to default”的日志，通常是：

- `include_bytes!` 路径不对（相对路径是相对当前 `src/main.rs` 所在文件）
- png 文件不是有效图片或被损坏
- 你忘记先 `Resource::register` 就直接 `Resource::from` 取句柄

---

## 9. 组件依赖与宏：#[component] / #[component_impl]（当你开始写自己的组件类型时）

过程宏解决一个现实问题：你在组装场景时，经常需要“保证某些组件必然存在”。

### 9.0 独立项目里如何使用 jge-macros

- **通过 `jge-core` 的 re-export 使用宏**

`jge-core` 在组件模块里 re-export 了 `component/component_impl`，可以直接使用：

```rust
use jge_core::game::component::{component, component_impl};
```

### 9.1 组件声明与依赖自动补齐

- `#[component(DepA, DepB, ...)]`：声明组件依赖
- 当你 `entity.register_component(MyComponent::new())` 时，如果依赖缺失，会自动尝试为实体补齐依赖组件

### 9.2 组件默认构造（给依赖补齐用）

- `#[component_impl]` 标记组件的固有 `impl` 块
- 在其中使用 `#[default(...)]` 标记一个“默认构造”函数（参数默认值从属性里给出）

这样依赖补齐时，宏就能生成一个“默认构造”路径来补齐缺失组件。

### 9.3 用一个“自定义组件”把旋转效果写出来（完整示例）

这一小节的目标：你自己定义一个组件 `Spin`，把它挂到实体上，再写一个逻辑让实体每帧旋转。

#### 9.3.1 定义组件：Spin（依赖 Transform）

```rust
use jge_core::game::component::{component, component_impl, transform::Transform};
use jge_core::game::entity::Entity;

#[component(Transform)]
#[derive(Debug, Clone)]
pub struct Spin {
    entity_id: Option<Entity>,
    speed: f32,
}

#[component_impl]
impl Spin {
    #[default(1.0)]
    pub fn new(speed: f32) -> Self {
        Self {
            entity_id: None,
            speed,
        }
    }

    pub fn speed(&self) -> f32 {
        self.speed
    }
}
```

#### 9.3.2 把 Spin 挂到你的三角形实体上

```rust
let _ = tri.register_component(Spin::new(1.5))?;
```

#### 9.3.3 写一个逻辑：如果实体有 Spin，就修改 Transform

```rust
use async_trait::async_trait;
use jge_core::game::system::logic::GameLogic;
use jge_core::game::{component::transform::Transform, entity::Entity};
use nalgebra::Vector3;

struct SpinLogic;

#[async_trait]
impl GameLogic for SpinLogic {
    async fn update(&mut self, e: Entity, delta: std::time::Duration) -> anyhow::Result<()> {
        let Some(spin) = e.get_component::<Spin>() else {
            return Ok(());
        };
        let speed = spin.speed();
        drop(spin);

        if let Some(mut t) = e.get_component_mut::<Transform>() {
            let mut r = t.rotation();
            r.y += speed * delta.as_secs_f32();
            t.set_rotation(Vector3::new(r.x, r.y, r.z));
        }
        Ok(())
    }
}

// 把逻辑挂到 tri 自己的 Node 上（这样 update 的 e 就是 tri）。
tri.get_component_mut::<jge_core::game::component::node::Node>()
    .expect("tri 应持有 Node")
    .set_logic(SpinLogic);
```

这个例子展示了三件事：

1) 你可以用 `#[component(...)]` 声明依赖，避免漏挂必需组件
2) 你可以把“数据”（Spin）与“行为”（SpinLogic）分离
3) 你可以只通过 `Entity` API 去读写组件，符合引擎约定

---

## 10. 常见问题（FAQ）

### 10.1 为什么我的实体渲染不出来？

按这个顺序排查：

- 根实体是否有 `Node`（`Entity::new()` 会自动挂）
- 树上是否有任意实体挂了 `Layer`
- `Layer` 下是否存在可渲染内容（Scene/Renderable/Material/Shape 等组合）
