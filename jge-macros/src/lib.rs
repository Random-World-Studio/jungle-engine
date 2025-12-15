use heck::ToShoutySnakeCase;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Attribute, DeriveInput, Expr, FnArg, ImplItem, ItemImpl, Pat, PatIdent, Path, ReturnType,
    Token, Type, parse::Parser, parse_macro_input, punctuated::Punctuated,
};

#[proc_macro_attribute]
pub fn component(args: TokenStream, input: TokenStream) -> TokenStream {
    let args_stream = proc_macro2::TokenStream::from(args);
    let dependencies: Vec<Path> = if args_stream.is_empty() {
        Vec::new()
    } else {
        match Punctuated::<Path, Token![,]>::parse_terminated.parse2(args_stream) {
            Ok(list) => list.into_iter().collect(),
            Err(err) => return err.to_compile_error().into(),
        }
    };

    let derive_input = parse_macro_input!(input as DeriveInput);
    let struct_item = derive_input.clone();

    if !derive_input.generics.params.is_empty() {
        return syn::Error::new_spanned(
            derive_input.generics,
            "#[component] does not yet support generic component types",
        )
        .to_compile_error()
        .into();
    }

    let struct_ident = &derive_input.ident;
    let vis = &derive_input.vis;

    let storage_name = struct_ident.to_string().to_shouty_snake_case();
    let storage_ident = format_ident!("__{}_STORAGE", storage_name);

    let dependency_checks = dependencies.iter().map(|dep| {
        quote! {
            ::jge_core::game::component::require_component::<#dep>(entity)?;
        }
    });

    let dependency_function = if dependencies.is_empty() {
        quote! {}
    } else {
        quote! {
            fn ensure_dependencies(entity: ::jge_core::game::entity::Entity) -> Result<(), ::jge_core::game::component::ComponentDependencyError> {
                #(#dependency_checks)*
                Ok(())
            }
        }
    };

    let expanded = if matches!(derive_input.data, syn::Data::Struct(_)) {
        quote! {
            #struct_item

            #[allow(non_upper_case_globals)]
            #vis static #storage_ident: ::std::sync::OnceLock<::jge_core::game::component::ComponentStorage<#struct_ident>> = ::std::sync::OnceLock::new();

            impl ::jge_core::game::component::Component for #struct_ident {
                fn storage() -> &'static ::jge_core::game::component::ComponentStorage<Self> {
                    #storage_ident.get_or_init(::jge_core::game::component::ComponentStorage::new)
                }

                #dependency_function
            }
        }
    } else {
        syn::Error::new_spanned(struct_ident, "#[component] can only be used with structs")
            .to_compile_error()
    };

    TokenStream::from(expanded)
}

#[proc_macro_attribute]
pub fn component_impl(_args: TokenStream, input: TokenStream) -> TokenStream {
    let mut item = parse_macro_input!(input as ItemImpl);

    if item.trait_.is_some() {
        return syn::Error::new_spanned(item.impl_token, "#[component_impl] 只能用于固有 impl 块")
            .to_compile_error()
            .into();
    }

    let self_ty = (*item.self_ty).clone();
    let type_path = match self_ty {
        Type::Path(ref path) => path.path.clone(),
        _ => {
            return syn::Error::new_spanned(item.self_ty, "#[component_impl] 仅支持简单路径类型")
                .to_compile_error()
                .into();
        }
    };

    let mut default_info: Option<(Vec<(PatIdent, Type)>, Vec<Expr>)> = None;

    for impl_item in item.items.iter_mut() {
        if let ImplItem::Fn(method) = impl_item {
            if let Some((attr, index)) = extract_default_attribute(&method.attrs) {
                if default_info.is_some() {
                    return syn::Error::new_spanned(
                        attr,
                        "同一 impl 中只能存在一个 #[default] 标记",
                    )
                    .to_compile_error()
                    .into();
                }

                method.attrs.remove(index);

                if method.sig.receiver().is_some() {
                    return syn::Error::new_spanned(
                        method.sig.clone(),
                        "#[default] 不能用于带 self 参数的方法",
                    )
                    .to_compile_error()
                    .into();
                }

                match &method.sig.output {
                    ReturnType::Type(_, ty) => {
                        if !matches!(**ty, Type::Path(ref p) if p.path.is_ident("Self")) {
                            return syn::Error::new_spanned(
                                ty,
                                "#[default] 仅支持返回 Self 的函数",
                            )
                            .to_compile_error()
                            .into();
                        }
                    }
                    ReturnType::Default => {
                        return syn::Error::new_spanned(
                            method.sig.clone(),
                            "#[default] 仅支持显式返回 Self 的函数",
                        )
                        .to_compile_error()
                        .into();
                    }
                }

                let defaults: Punctuated<Expr, Token![,]> =
                    match attr.parse_args_with(Punctuated::<Expr, Token![,]>::parse_terminated) {
                        Ok(values) => values,
                        Err(err) => return err.to_compile_error().into(),
                    };

                let mut params = Vec::new();
                for input in &method.sig.inputs {
                    match input {
                        FnArg::Receiver(receiver) => {
                            return syn::Error::new_spanned(
                                receiver,
                                "#[default] 函数不能拥有 self 参数",
                            )
                            .to_compile_error()
                            .into();
                        }
                        FnArg::Typed(pat_type) => {
                            let pat = match &*pat_type.pat {
                                Pat::Ident(ident) => ident.clone(),
                                other => {
                                    return syn::Error::new_spanned(
                                        other,
                                        "#[default] 函数的参数必须是标识符模式",
                                    )
                                    .to_compile_error()
                                    .into();
                                }
                            };
                            params.push((pat, (*pat_type.ty).clone()));
                        }
                    }
                }

                if params.len() != defaults.len() {
                    return syn::Error::new_spanned(
                        method.sig.clone(),
                        "#[default] 的实参数量必须与函数参数一致",
                    )
                    .to_compile_error()
                    .into();
                }

                default_info = Some((params, defaults.into_iter().collect()));
            }
        }
    }

    let registration = if let Some((params, defaults)) = default_info {
        let ensure_ident = format_ident!(
            "__jge_component_default_ensure_{}",
            type_path
                .segments
                .last()
                .map(|seg| seg.ident.to_string())
                .unwrap_or_else(|| "component".into())
        );
        let module_ident = format_ident!(
            "__jge_component_defaults_module_{}",
            type_path
                .segments
                .last()
                .map(|seg| seg.ident.to_string())
                .unwrap_or_else(|| "component".into())
        );

        let ensure_statements =
            params
                .into_iter()
                .zip(defaults.into_iter())
                .map(|((pat, ty), expr)| {
                    let ident = &pat.ident;
                    quote! {
                        if <#ty as ::jge_core::game::component::Component>::read(entity).is_none() {
                            let #pat: #ty = #expr;
                            let _ = entity.register_component(#ident)?;
                        }
                    }
                });

        quote! {
            #[allow(non_snake_case)]
            mod #module_ident {
                use super::*;

                pub(super) fn #ensure_ident(entity: ::jge_core::game::entity::Entity) -> ::std::result::Result<(), ::jge_core::game::component::ComponentDependencyError> {
                    #(#ensure_statements)*
                    Ok(())
                }

                ::inventory::submit! {
                    ::jge_core::game::component::ComponentDefaultDescriptor {
                        type_id: ::std::any::TypeId::of::<#type_path>(),
                        ensure: |entity| #ensure_ident(entity),
                    }
                }
            }
        }
    } else {
        quote! {}
    };

    TokenStream::from(quote! {
        #item
        #registration
    })
}

fn extract_default_attribute(attrs: &[Attribute]) -> Option<(Attribute, usize)> {
    attrs
        .iter()
        .enumerate()
        .find(|(_, attr)| attr.path().is_ident("default"))
        .map(|(index, attr)| (attr.clone(), index))
}
