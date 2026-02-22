pub mod file_selector;

use crate::game::entity::Entity;
use ::core::future::Future;
use ::core::pin::Pin;
use ::std::sync::Arc;

/// 支持把“一个场景/子树值”转换成可挂载的根节点实体。
///
/// 该 trait 主要用于 `scene!` DSL 的组合能力：`<expr> node;`。
///
/// # 示例
///
/// ```no_run
/// use anyhow::Context as _;
/// use jge_core::{scene, scenes::SceneRoot};
///
/// # async fn demo() -> anyhow::Result<()> {
/// let child = scene! {
///     node "child" { }
/// }
/// .await?;
///
/// // 既可以从 bindings 取根实体：
/// let child_root = child.scene_root();
/// let _ = child_root;
///
/// // 也可以直接用 Entity：
/// // let e = jge_core::game::entity::Entity::new().await?;
/// // let _ = e.scene_root();
/// Ok(())
/// # }
/// ```
pub trait SceneRoot {
    fn scene_root(&self) -> Entity;
}

impl SceneRoot for Entity {
    fn scene_root(&self) -> Entity {
        *self
    }
}

impl<T: SceneRoot + ?Sized> SceneRoot for &T {
    fn scene_root(&self) -> Entity {
        (*self).scene_root()
    }
}

/// 表示一个“可被销毁”的场景 bindings（通常是 `scene!` 返回值）。
///
/// 该 trait 是 object-safe 的，用于在外层 `SceneBindings` 内部保存
/// 一组“嵌套 scene”的 destroy 链。
///
/// `scene!` 会为它生成的 `SceneBindings` 自动实现该 trait。
pub trait SceneDestroy: Send + Sync {
    fn destroy_boxed<'a>(&'a self) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
}

/// 额外的“销毁钩子”（可访问 bindings 上下文）。
///
/// 默认 `scene!` 的 destroy 只会卸载 DSL 中显式声明的组件，并级联销毁嵌套 scene bindings。
/// 若你的逻辑在运行时动态创建/挂载了其他实体（例如运行时生成的特效、子树、临时 UI），
/// 可以注册一个或多个 destroy hook，把清理逻辑挂在 bindings 上。
///
/// 该 trait 是 object-safe 的。
pub trait SceneDestroyHook: Send + Sync {
    fn destroy_boxed<'a>(
        &'a self,
        bindings: &'a dyn SceneBinding,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
}

/// 把一个 async 销毁回调包装成 [`SceneDestroyHook`]。
///
/// 由于回调通常需要借用 `bindings`（例如读取 `binding("xxx")`），这里要求回调返回
/// **带同一生命周期的 boxed future**。
pub struct DestroyHook {
    f: Box<
        dyn for<'a> Fn(&'a dyn SceneBinding) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>
            + Send
            + Sync,
    >,
}

impl SceneDestroyHook for DestroyHook {
    fn destroy_boxed<'a>(
        &'a self,
        bindings: &'a dyn SceneBinding,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        (self.f)(bindings)
    }
}

/// 便捷函数：把 closure 转成可注册的 destroy hook。
///
/// 这里要求 closure 返回 `Pin<Box<dyn Future + 'a>>`（通常写成 `Box::pin(async move { ... })`），
/// 以确保 future 的生命周期严格受 `bindings: &'a dyn SceneBinding` 这个借用约束。
///
/// 这是为了允许你在 hook 内部安全地使用 `bindings`（例如调用 [`SceneBinding::binding`]），
/// 并避免把回调错误地约束为必须返回 `'static` future。
///
/// # 示例
///
/// ```no_run
/// use jge_core::scenes::{destroy_hook, SceneBinding as _};
///
/// # async fn demo() -> anyhow::Result<()> {
/// let bindings = jge_core::scene! { node "root" { } }.await?;
///
/// bindings.register_destroy_hook(destroy_hook(|b| {
///     Box::pin(async move {
///         // 在这里可以访问 bindings 上下文。
///         let _root = b.binding("root");
///     })
/// }));
///
/// Ok(())
/// # }
/// ```
pub fn destroy_hook<F>(f: F) -> Box<dyn SceneDestroyHook>
where
    F: for<'a> Fn(&'a dyn SceneBinding) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>
        + Send
        + Sync
        + 'static,
{
    Box::new(DestroyHook { f: Box::new(f) })
}

/// `scene!` 的返回值应实现的 bindings trait。
///
/// 该 trait 是 object-safe 的，使得 `scene!` 的返回值可以用动态 trait 对象传递：
///
/// ```no_run
/// use jge_core::scenes::SceneBinding;
///
/// # async fn demo() -> anyhow::Result<()> {
/// let bindings = jge_core::scene! { node "root" { } }.await?;
/// let boxed: Box<dyn SceneBinding> = Box::new(bindings);
/// let _root = boxed.scene_root();
/// boxed.destroy_boxed().await;
/// Ok(())
/// # }
/// ```
pub trait SceneBinding: SceneRoot + SceneDestroy {
    /// 获取某个 `as name` 绑定对应的实体。
    ///
    /// - `name = "root"` 一定可用。
    /// - 其他 name 由 `scene!` 展开生成。
    fn binding(&self, name: &str) -> Option<Entity>;

    /// 返回当前 bindings 支持的名字列表（包含 `"root"`）。
    fn binding_names(&self) -> &'static [&'static str];

    /// 注册一个额外的“销毁钩子”。
    ///
    /// 默认 `scene!` 的 destroy 只会卸载 DSL 中显式声明的组件，并级联销毁嵌套 scene bindings。
    /// 若你的逻辑在运行时动态创建/挂载了其他实体（例如运行时生成的特效、子树、临时 UI），
    /// 可以把清理逻辑注册到这里，保证 `destroy_boxed()/destroy().await` 时也会被执行。
    ///
    /// - 支持多次注册（按注册顺序执行）。
    /// - 建议通过 [`destroy_hook`] 包装 async closure（通常写成 `Box::pin(async move { ... })`）。
    fn register_destroy_hook(&self, hook: Box<dyn SceneDestroyHook>);
}

impl<T: SceneRoot + ?Sized> SceneRoot for Box<T> {
    fn scene_root(&self) -> Entity {
        (**self).scene_root()
    }
}

impl<T: SceneDestroy + ?Sized> SceneDestroy for Box<T> {
    fn destroy_boxed<'a>(&'a self) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        (**self).destroy_boxed()
    }
}

impl<T: SceneRoot + ?Sized> SceneRoot for Arc<T> {
    fn scene_root(&self) -> Entity {
        (**self).scene_root()
    }
}

impl<T: SceneDestroy + ?Sized> SceneDestroy for Arc<T> {
    fn destroy_boxed<'a>(&'a self) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        (**self).destroy_boxed()
    }
}

/// 把某个值转换为“可选的嵌套 destroy”。
///
/// - 对 `Entity`：返回 `None`（实体本身不支持 `destroy()`）
/// - 对实现了 [`SceneDestroy`] 的类型：返回 `Some(Box<dyn SceneDestroy>)`
///
/// 该 trait 的用途是让 `<expr> node;` 同时支持：
/// - 传入另一个 `scene!` 的返回值：外层 `destroy().await` 会级联调用内层 destroy
/// - 传入 `Entity`：只做挂载，不会误删/误卸载外部实体上的组件
pub trait SceneMaybeDestroy {
    fn into_optional_destroy(self) -> Option<Box<dyn SceneDestroy>>;
}

impl SceneMaybeDestroy for Entity {
    fn into_optional_destroy(self) -> Option<Box<dyn SceneDestroy>> {
        None
    }
}

impl<T> SceneMaybeDestroy for T
where
    T: SceneDestroy + 'static,
{
    fn into_optional_destroy(self) -> Option<Box<dyn SceneDestroy>> {
        Some(Box::new(self))
    }
}
