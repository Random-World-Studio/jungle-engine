use heck::ToShoutySnakeCase;
use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Attribute, DeriveInput, Expr, GenericArgument, ImplItem, ItemImpl, Path, PathArguments,
    ReturnType, Token, Type, parse::Parser, parse_macro_input, punctuated::Punctuated,
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
        let generics = derive_input.generics.to_token_stream().to_string();
        return syn::Error::new_spanned(
            derive_input.generics,
            format!(
                "#[component] 目前不支持带泛型参数的组件类型，检测到泛型参数：{}",
                generics
            ),
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
            if <#dep as ::jge_core::game::component::Component>::read(entity).is_none() {
                let component = #dep::__jge_component_default(entity)?;
                let _ = entity.register_component(component)?;
            }
        }
    });

    let dependency_function = if dependencies.is_empty() {
        quote! {}
    } else {
        quote! {
            fn register_dependencies(entity: ::jge_core::game::entity::Entity) -> Result<(), ::jge_core::game::component::ComponentDependencyError> {
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

    if !matches!(&*item.self_ty, Type::Path(_)) {
        return syn::Error::new_spanned(item.self_ty, "#[component_impl] 仅支持简单路径类型")
            .to_compile_error()
            .into();
    }

    let mut default_info: Option<(syn::Ident, Vec<Expr>, DefaultReturnKind)> = None;

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
                    let signature = method.sig.to_token_stream().to_string();
                    return syn::Error::new_spanned(
                        method.sig.clone(),
                        format!("#[default] 不能用于带 self 参数的方法：{}", signature),
                    )
                    .to_compile_error()
                    .into();
                }

                let return_kind = match &method.sig.output {
                    ReturnType::Type(_, ty) => match analyze_default_return_type(ty) {
                        Ok(kind) => kind,
                        Err(err) => return err.to_compile_error().into(),
                    },
                    ReturnType::Default => {
                        let signature = method.sig.to_token_stream().to_string();
                        return syn::Error::new_spanned(
                            method.sig.clone(),
                            format!(
                                "#[default] 所在函数必须显式返回 Self 或 Result<Self, E>：{}",
                                signature
                            ),
                        )
                        .to_compile_error()
                        .into();
                    }
                };

                let defaults: Punctuated<Expr, Token![,]> =
                    match attr.parse_args_with(Punctuated::<Expr, Token![,]>::parse_terminated) {
                        Ok(values) => values,
                        Err(err) => return err.to_compile_error().into(),
                    };

                if method.sig.inputs.len() != defaults.len() {
                    let expected = method.sig.inputs.len();
                    let provided = defaults.len();
                    return syn::Error::new_spanned(
                        method.sig.clone(),
                        format!(
                            "#[default] 参数数量不匹配：函数需要 {} 个参数，但属性提供了 {} 个默认值",
                            expected, provided
                        ),
                    )
                    .to_compile_error()
                    .into();
                }

                default_info = Some((
                    method.sig.ident.clone(),
                    defaults.into_iter().collect(),
                    return_kind,
                ));
            }
        }
    }

    if let Some((method_ident, defaults, return_kind)) = default_info {
        let helper_tokens = match return_kind {
            DefaultReturnKind::Direct => quote! {
                #[doc(hidden)]
                pub(crate) fn __jge_component_default(entity: ::jge_core::game::entity::Entity) -> ::std::result::Result<Self, ::jge_core::game::component::ComponentDependencyError> {
                    Ok(Self::#method_ident(#(#defaults),*))
                }
            },
            DefaultReturnKind::Result => quote! {
                #[doc(hidden)]
                pub(crate) fn __jge_component_default(entity: ::jge_core::game::entity::Entity) -> ::std::result::Result<Self, ::jge_core::game::component::ComponentDependencyError> {
                    let component = Self::#method_ident(#(#defaults),*)
                        .map_err(|error| ::jge_core::game::component::ComponentDependencyError::with_source(
                            entity,
                            ::core::any::type_name::<Self>(),
                            error,
                        ))?;
                    Ok(component)
                }
            },
        };

        let helper_fn: ImplItem = match syn::parse2(helper_tokens) {
            Ok(item) => item,
            Err(err) => return err.to_compile_error().into(),
        };

        item.items.push(helper_fn);
    }

    TokenStream::from(quote! {
        #item
    })
}

fn extract_default_attribute(attrs: &[Attribute]) -> Option<(Attribute, usize)> {
    attrs
        .iter()
        .enumerate()
        .find(|(_, attr)| attr.path().is_ident("default"))
        .map(|(index, attr)| (attr.clone(), index))
}

enum DefaultReturnKind {
    Direct,
    Result,
}

fn analyze_default_return_type(ty: &Type) -> Result<DefaultReturnKind, syn::Error> {
    if is_self_type(ty) {
        return Ok(DefaultReturnKind::Direct);
    }

    if is_result_of_self(ty) {
        return Ok(DefaultReturnKind::Result);
    }

    let ty_tokens = ty.to_token_stream().to_string();
    Err(syn::Error::new_spanned(
        ty,
        format!(
            "#[default] 所在函数必须返回 Self 或 Result<Self, E>，当前返回类型为 {}",
            ty_tokens
        ),
    ))
}

fn is_self_type(ty: &Type) -> bool {
    matches!(ty, Type::Path(type_path) if type_path.qself.is_none() && type_path.path.is_ident("Self"))
}

fn is_result_of_self(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => {
            if let Some(segment) = type_path.path.segments.last() {
                if segment.ident == "Result" {
                    if let PathArguments::AngleBracketed(args) = &segment.arguments {
                        if let Some(first) = args.args.first() {
                            if let GenericArgument::Type(first_type) = first {
                                return is_self_type(first_type);
                            }
                        }
                    }
                }
            }
            false
        }
        _ => false,
    }
}
