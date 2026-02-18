#[test]
fn scene_macro_ui_compile_fail() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/scene_progress_expr_pass.rs");
    t.compile_fail("tests/ui/scene_progress_n_zero_literal_fail.rs");
    t.compile_fail("tests/ui/scene_two_roots_fail.rs");
    t.compile_fail("tests/ui/scene_invalid_item_fail.rs");
}
