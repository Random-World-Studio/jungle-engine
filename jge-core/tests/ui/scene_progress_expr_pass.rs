use jge_core::{ProgressFrame, scene};

fn main() {
    let (tx, _rx) = tokio::sync::mpsc::channel::<ProgressFrame>(1);

    let i: usize = 0;
    let n: usize = 1;

    // 只要能通过编译即可。
    let _fut = scene! {
        progress(i/n) tx;
        node "root" { }
    };
}
