#![allow(clippy::type_complexity)]

use anyhow::Context as _;

use crate::ProgressFrame;
use crate::game::component::node::Node;
use crate::game::component::renderable::Renderable;
use crate::game::component::transform::Transform;
use crate::game::entity::Entity;
use crate::scenes::SceneBinding;

#[tokio::test(flavor = "multi_thread")]
async fn scene_macro_allows_forward_reference_to_as_binding_and_preserves_child_order()
-> anyhow::Result<()> {
    let bindings = crate::scene! {
        node "root" {
            node "a" as a {
                // 这里刻意在 DSL 中引用后面才声明的 `b`。
                with(_n: Node) {
                    let _b_node = b
                        .get_component::<Node>()
                        .await
                        .with_context(|| "b should exist before init")?;
                    Ok(())
                };
            }
            node "b" as b { }
        }
    }
    .await?;

    let root_node = bindings
        .root
        .get_component::<Node>()
        .await
        .with_context(|| "root should have Node")?;

    assert_eq!(root_node.children(), &[bindings.a, bindings.b]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scene_macro_allows_empty_input_as_noop() -> anyhow::Result<()> {
    crate::scene! {}.await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scene_macro_can_report_progress() -> anyhow::Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressFrame>(64);
    let progress_tx = tx.clone();

    let _bindings = crate::scene! {
        progress progress_tx;
        node "root" {
            node "a" { }
            node "b" { }
        }
    }
    .await?;

    // 关闭发送端，确保 receiver 能 drain 完成。
    drop(tx);

    let mut frames = Vec::new();
    while let Some(frame) = rx.recv().await {
        frames.push(frame);
    }

    assert!(!frames.is_empty(), "progress 应至少发送一次");

    // 第一帧应为阶段声明（默认 0/1）。
    assert_eq!(frames[0], ProgressFrame::Phase(0, 1));

    // Progress 值序列：应从 0.0 单调不减到 1.0。
    let values: Vec<f64> = frames
        .iter()
        .filter_map(|f| match f {
            ProgressFrame::Progress(p) => Some(*p),
            _ => None,
        })
        .collect();

    assert!(
        (values[0] - 0.0).abs() <= f64::EPSILON,
        "progress 第一个值应为 0.0"
    );
    assert!(
        (values[values.len() - 1] - 1.0).abs() <= 1e-12,
        "progress 最后一个值应为 1.0"
    );
    assert!(
        values.iter().all(|v| *v >= 0.0 && *v <= 1.0),
        "progress 值必须在 [0, 1]"
    );
    assert!(
        values.windows(2).all(|w| w[0] <= w[1]),
        "progress 必须单调不减"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scene_macro_progress_can_specify_phase() -> anyhow::Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressFrame>(64);

    let progress_tx = tx.clone();

    let _bindings = crate::scene! {
        progress(2/5) progress_tx;
        node "root" { }
    }
    .await?;

    drop(tx);

    let first = rx.recv().await.expect("should receive at least one frame");
    assert_eq!(first, ProgressFrame::Phase(2, 5));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scene_macro_progress_can_use_usize_expr() -> anyhow::Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressFrame>(64);
    let progress_tx = tx.clone();

    let i: usize = 2;
    let n: usize = 5;

    let _bindings = crate::scene! {
        progress(i/n) progress_tx;
        node "root" { }
    }
    .await?;

    drop(tx);

    let first = rx.recv().await.expect("should receive at least one frame");
    assert_eq!(first, ProgressFrame::Phase(2, 5));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scene_macro_progress_last_phase_sends_terminal_phase() -> anyhow::Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressFrame>(64);
    let progress_tx = tx.clone();

    let _bindings = crate::scene! {
        progress(4/5) progress_tx;
        node "root" { }
    }
    .await?;

    drop(tx);

    let mut frames = Vec::new();
    while let Some(frame) = rx.recv().await {
        frames.push(frame);
    }

    assert!(
        frames.iter().any(|f| *f == ProgressFrame::Phase(5, 5)),
        "最后阶段应在结尾发送 Phase(5,5)"
    );
    assert_eq!(
        frames.last().copied(),
        Some(ProgressFrame::Phase(5, 5)),
        "Phase(5,5) 应为最后一帧"
    );
    Ok(())
}

#[test]
fn scene_macro_progress_runtime_n_zero_panics() {
    let rt = tokio::runtime::Runtime::new().expect("create tokio runtime");

    let result = std::panic::catch_unwind(|| {
        rt.block_on(async {
            let (tx, _rx) = tokio::sync::mpsc::channel::<ProgressFrame>(1);
            let progress_tx = tx.clone();

            let i: usize = 0;
            let n: usize = 0;

            // n 不是字面量时，宏会在运行期 assert!(n>0)。
            let _ = crate::scene! {
                progress(i/n) progress_tx;
                node "root" { }
            }
            .await;
        })
    });

    assert!(result.is_err(), "n==0 时应 panic");
}

#[tokio::test(flavor = "multi_thread")]
async fn scene_macro_bindings_destroy_unregisters_explicit_components() -> anyhow::Result<()> {
    let bindings = crate::scene! {
        node "root" {
            + Transform::new();
        }
    }
    .await?;

    assert!(
        bindings.root.get_component::<Transform>().await.is_some(),
        "Transform 应已注册"
    );
    bindings.destroy().await;

    assert!(
        bindings.root.get_component::<Transform>().await.is_none(),
        "destroy() 后 Transform 应被卸载"
    );
    // 幂等：重复 destroy 不应出错。
    bindings.destroy().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scene_macro_bindings_destroy_unregisters_components_for_all_entities() -> anyhow::Result<()>
{
    let bindings = crate::scene! {
        node "root" {
            + Renderable::new();
            node "child" as child {
                + Transform::new();
                + Renderable::new();
            }
        }
    }
    .await?;

    assert!(bindings.root.get_component::<Renderable>().await.is_some());
    assert!(bindings.root.get_component::<Transform>().await.is_none());
    assert!(bindings.child.get_component::<Transform>().await.is_some());
    assert!(bindings.child.get_component::<Renderable>().await.is_some());

    bindings.destroy().await;

    assert!(bindings.root.get_component::<Renderable>().await.is_none());
    assert!(bindings.child.get_component::<Transform>().await.is_none());
    assert!(bindings.child.get_component::<Renderable>().await.is_none());

    // destroy() 不应影响默认 Node 的存在（实体句柄仍有效）。
    assert!(bindings.root.get_component::<Node>().await.is_some());
    assert!(bindings.child.get_component::<Node>().await.is_some());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scene_macro_bindings_destroy_is_send_and_spawnable() -> anyhow::Result<()> {
    let bindings = crate::scene! {
        node "root" {
            + Transform::new();
        }
    }
    .await?;

    let bindings = std::sync::Arc::new(bindings);
    let handle = tokio::spawn({
        let bindings = bindings.clone();
        async move {
            bindings.destroy().await;
        }
    });

    handle.await.expect("destroy task should complete");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scene_macro_destroy_hooks_are_executed() -> anyhow::Result<()> {
    use crate::scenes::SceneBinding as _;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let bindings = crate::scene! {
        node "root" {
            + Transform::new();
        }
    }
    .await?;

    let hits = Arc::new(AtomicUsize::new(0));
    let hits2 = hits.clone();
    bindings.register_destroy_hook(crate::scenes::destroy_hook(move |b| {
        let hits = hits2.clone();
        Box::pin(async move {
            assert!(b.binding("root").is_some());
            hits.fetch_add(1, Ordering::SeqCst);
        })
    }));

    bindings.destroy().await;
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "destroy hook should run once"
    );

    // destroy() 允许重复调用；hook 列表应被 drain，避免重复执行。
    bindings.destroy().await;
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "destroy hook should not rerun"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scene_macro_external_scene_bindings_node_can_attach_and_destroy_cascades()
-> anyhow::Result<()> {
    let bindings = crate::scene! {
        node "root" {
            crate::scene! {
                node "child_root" {
                    + Transform::new();
                }
            }
            .await? node;
        }
    }
    .await?;

    let root_node = bindings
        .root
        .get_component::<Node>()
        .await
        .expect("root should have Node");
    assert_eq!(root_node.children().len(), 1);

    let child_root = root_node.children()[0];
    assert!(
        child_root.get_component::<Transform>().await.is_some(),
        "nested scene should have registered Transform"
    );

    bindings.destroy().await;

    assert!(
        child_root.get_component::<Transform>().await.is_none(),
        "outer destroy() should cascade into nested scene bindings destroy()"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scene_macro_external_entity_node_does_not_force_nested_destroy() -> anyhow::Result<()> {
    let external = Entity::new().await?;
    external.register_component(Transform::new()).await?;

    let bindings = crate::scene! {
        node "root" {
            external node;
        }
    }
    .await?;

    bindings.destroy().await;

    assert!(
        external.get_component::<Transform>().await.is_some(),
        "Entity passed to `<expr> node;` should not be force-destroyed"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scene_macro_bindings_can_be_passed_as_dyn_scene_binding() -> anyhow::Result<()> {
    let bindings = crate::scene! {
        node "root" {
            node "child" as child {
                + Transform::new();
            }
        }
    }
    .await?;

    let boxed: Box<dyn SceneBinding> = Box::new(bindings);

    assert!(boxed.binding_names().contains(&"root"));
    assert!(boxed.binding_names().contains(&"child"));

    let root = boxed.scene_root();
    let child = boxed.binding("child").expect("child binding should exist");

    assert!(root.get_component::<Node>().await.is_some());
    assert!(child.get_component::<Transform>().await.is_some());

    boxed.destroy_boxed().await;
    assert!(child.get_component::<Transform>().await.is_none());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scene_macro_with_requires_component() -> anyhow::Result<()> {
    let result = crate::scene! {
        node "root" {
            // Transform 没有被注册，因此 with 绑定应失败。
            with(_t: Transform) {
                Ok(())
            };
        }
    }
    .await;

    let err = match result {
        Ok(_) => anyhow::bail!("with() should error if component is missing"),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("scene!: with 缺少组件"),
        "error should mention missing component, got: {msg}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn scene_macro_attach_happens_after_init() -> anyhow::Result<()> {
    let bindings = crate::scene! {
        node "root" as _root {
            node "a" as a { }

            // init 阶段执行：此时树关系尚未建立。
            with(n: Node) {
                assert!(n.children().is_empty(), "root should have no children during init");

                let a_node = a
                    .get_component::<Node>()
                    .await
                    .with_context(|| "a should have Node")?;
                assert!(a_node.parent().is_none(), "a should have no parent during init");
                Ok(())
            };
        }
    }
    .await?;

    let root_node = bindings
        .root
        .get_component::<Node>()
        .await
        .with_context(|| "root should have Node")?;
    assert_eq!(root_node.children(), &[bindings.a]);

    let a_node = bindings
        .a
        .get_component::<Node>()
        .await
        .with_context(|| "a should have Node")?;
    assert_eq!(a_node.parent(), Some(bindings.root));

    // 避免遗留显式注册组件；这里没显式组件但仍保持 destroy 可用。
    bindings.destroy().await;
    Ok(())
}
