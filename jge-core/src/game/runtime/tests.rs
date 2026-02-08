use super::*;
use crate::game::component::node::Node;
use crate::game::system::logic::GameLogic;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

struct TrackingLogic {
    label: &'static str,
    events: Arc<StdMutex<Vec<&'static str>>>,
}

#[async_trait::async_trait]
impl GameLogic for TrackingLogic {
    async fn on_attach(&mut self, _e: Entity) -> anyhow::Result<()> {
        self.events.lock().unwrap().push(if self.label == "root" {
            "root_attach"
        } else {
            "child_attach"
        });
        Ok(())
    }

    async fn on_detach(&mut self, _e: Entity) -> anyhow::Result<()> {
        self.events.lock().unwrap().push(if self.label == "root" {
            "root_detach"
        } else {
            "child_detach"
        });
        Ok(())
    }
}

#[test]
fn game_drop_detaches_tree_and_calls_root_on_detach() {
    let events = Arc::new(StdMutex::new(Vec::new()));

    let setup_rt = tokio::runtime::Runtime::new().expect("should create setup runtime");
    let (root, child) = setup_rt.block_on(async {
        let root = Entity::new().await.expect("should create root entity");
        root.register_component(Node::new("root").unwrap())
            .await
            .expect("should register root Node");
        {
            let mut node = root
                .get_component_mut::<Node>()
                .await
                .expect("should get root Node mut guard");
            node.set_logic(TrackingLogic {
                label: "root",
                events: events.clone(),
            })
            .await;
        }

        let child = Entity::new().await.expect("should create child entity");
        child
            .register_component(Node::new("child").unwrap())
            .await
            .expect("should register child Node");
        {
            let mut node = child
                .get_component_mut::<Node>()
                .await
                .expect("should get child Node mut guard");
            node.set_logic(TrackingLogic {
                label: "child",
                events: events.clone(),
            })
            .await;
        }

        (root, child)
    });
    drop(setup_rt);

    let game = Game::new(GameConfig::default(), root).expect("should create game");

    // 等待 root 的 on_attach 实际执行完毕（按实现它可能被 spawn）。
    game.block_on(async {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if events.lock().unwrap().contains(&"root_attach") {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("root on_attach should complete");
    });

    // 驱动 attach，确保 on_attach 完成。
    game.block_on(async {
        tokio::time::timeout(Duration::from_secs(2), async {
            let attach_future = {
                let mut root_node = root
                    .get_component_mut::<Node>()
                    .await
                    .expect("should get root Node mut guard");
                root_node.attach(child)
            };
            attach_future.await.unwrap();
        })
        .await
        .expect("attach should not block");
    });

    game.block_on(async {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if events.lock().unwrap().contains(&"child_attach") {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("child on_attach should complete");
    });

    let (drop_tx, drop_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        drop(game);
        let _ = drop_tx.send(());
    });
    drop_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("dropping Game should not block");

    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let log = events.lock().unwrap();
        if log.contains(&"child_detach") && log.contains(&"root_detach") {
            break;
        }
        drop(log);
        assert!(Instant::now() < deadline, "on_detach should complete");
        std::thread::yield_now();
    }

    let log = events.lock().unwrap().clone();
    assert_eq!(
        log.as_slice(),
        &["root_attach", "child_attach", "child_detach", "root_detach"]
    );
}
