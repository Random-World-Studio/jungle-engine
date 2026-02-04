use heck::ToShoutySnakeCase;
use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{DeriveInput, Path, Token, parse::Parser, parse_macro_input, punctuated::Punctuated};

pub fn expand_component(args: TokenStream, input: TokenStream) -> TokenStream {
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
            if entity.get_component::<#dep>().await.is_none() {
                let component = #dep::__jge_component_default(entity)?;
                let _ = entity.register_component(component).await?;
            }
        }
    });

    let dependency_function = if dependencies.is_empty() {
        quote! {}
    } else {
        quote! {
            async fn register_dependencies(entity: ::jge_core::game::entity::Entity) -> Result<(), ::jge_core::game::component::ComponentDependencyError> {
                #(#dependency_checks)*
                Ok(())
            }
        }
    };

    let mut has_entity_field = false;
    if let syn::Data::Struct(struct_data) = &derive_input.data {
        for field in struct_data.fields.iter() {
            if let Some(ident) = &field.ident {
                if ident == "entity_id" {
                    has_entity_field = true;
                    break;
                }
            }
        }
    }

    let attach_entity_fn = if has_entity_field {
        quote! {
            fn attach_entity(&mut self, entity: ::jge_core::game::entity::Entity) {
                self.entity_id = ::std::option::Option::Some(entity);
            }

            fn detach_entity(&mut self) {
                self.entity_id = ::std::option::Option::None;
            }
        }
    } else {
        quote! {
            fn attach_entity(&mut self, entity: ::jge_core::game::entity::Entity) {
                let _ = entity;
            }

            fn detach_entity(&mut self) {}
        }
    };

    let expanded = if matches!(derive_input.data, syn::Data::Struct(_)) {
        quote! {
            #struct_item

            #[allow(non_upper_case_globals)]
            #vis static #storage_ident: ::std::sync::OnceLock<::jge_core::game::component::ComponentStorage<#struct_ident>> = ::std::sync::OnceLock::new();

            #[jge_core::async_trait]
            impl ::jge_core::game::component::Component for #struct_ident {
                fn storage() -> &'static ::jge_core::game::component::ComponentStorage<Self> {
                    #storage_ident.get_or_init(::jge_core::game::component::ComponentStorage::new)
                }

                #dependency_function

                #attach_entity_fn
            }
        }
    } else {
        syn::Error::new_spanned(struct_ident, "#[component] can only be used with structs")
            .to_compile_error()
    };

    TokenStream::from(expanded)
}
