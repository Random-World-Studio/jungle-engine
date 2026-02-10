pub mod file_selector;

use crate::game::entity::Entity;
use ::core::future::Future;
use ::core::pin::Pin;

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
