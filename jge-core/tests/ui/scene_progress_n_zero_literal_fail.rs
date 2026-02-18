use jge_core::{ProgressFrame, scene};

fn main() {
    let (tx, _rx) = tokio::sync::mpsc::channel::<ProgressFrame>(1);

    let _fut = scene! {
        progress(0/0) tx;
        node "root" { }
    };
}
