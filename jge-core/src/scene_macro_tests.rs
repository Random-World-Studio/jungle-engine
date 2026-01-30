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
