#[test]
fn scene_macro_ui_compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/scene_two_roots_fail.rs");
    t.compile_fail("tests/ui/scene_invalid_item_fail.rs");
}
