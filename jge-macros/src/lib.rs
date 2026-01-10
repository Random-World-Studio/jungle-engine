#![cfg_attr(feature = "callsite_relative_paths", feature(proc_macro_span))]

use proc_macro::TokenStream;

mod component_attr;
mod component_impl_attr;
mod scene_macro;

#[proc_macro_attribute]
pub fn component(args: TokenStream, input: TokenStream) -> TokenStream {
    component_attr::expand_component(args, input)
}

#[proc_macro_attribute]
pub fn component_impl(args: TokenStream, input: TokenStream) -> TokenStream {
    component_impl_attr::expand_component_impl(args, input)
}

#[proc_macro]
pub fn scene(input: TokenStream) -> TokenStream {
    scene_macro::expand_scene(input)
}
