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
