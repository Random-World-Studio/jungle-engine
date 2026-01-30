use anyhow::Context as _;

use crate::game::component::node::Node;

#[tokio::test]
async fn scene_macro_allows_forward_reference_to_as_binding_and_preserves_child_order()
-> anyhow::Result<()> {
    let bindings = crate::scene! {
        node "root" {
            node "a" as a {
                // 这里刻意在 DSL 中引用后面才声明的 `b`。
                with(_n: Node) {
                    let _b_node = b
                        .get_component::<Node>()
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
        .with_context(|| "root should have Node")?;

    assert_eq!(root_node.children(), &[bindings.a, bindings.b]);
    Ok(())
}

#[tokio::test]
async fn scene_macro_can_report_progress() -> anyhow::Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<f64>(64);
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

    let mut values = Vec::new();
    while let Some(v) = rx.recv().await {
        values.push(v);
    }

    assert!(!values.is_empty(), "progress 应至少发送一次");
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
