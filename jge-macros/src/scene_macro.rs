use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use quote::{format_ident, quote};
use syn::parse_macro_input;

use std::path::PathBuf;

// ===== scene! DSL =====

mod scene_dsl {
    use super::*;
    use quote::quote_spanned;
    use syn::braced;
    use syn::spanned::Spanned;
    use syn::{
        Expr, Ident, Token, parenthesized,
        parse::{Parse, ParseStream},
        punctuated::Punctuated,
    };

    syn::custom_keyword!(node);
    syn::custom_keyword!(with);
    syn::custom_keyword!(resource);

    pub struct SceneInput {
        pub root: NodeDecl,
    }

    impl Parse for SceneInput {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            if input.is_empty() {
                return Err(input.error("scene! 需要且仅需要一个根 node {...}"));
            }
            let root: NodeDecl = input.parse()?;
            if !input.is_empty() {
                return Err(input.error("scene! 顶层只允许一个根 node"));
            }
            Ok(Self { root })
        }
    }

    pub struct NodeDecl {
        pub name: Option<Expr>,
        pub id: Option<Expr>,
        pub bind: Option<Ident>,
        pub items: Vec<NodeItem>,
        pub span: proc_macro2::Span,
    }

    impl Parse for NodeDecl {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let node_kw: node = input.parse()?;
            let span = node_kw.span();

            // 可选名称：`node "name" ...`（也允许任意 expr，但通常用字符串）
            let mut name = if input.peek(syn::token::Paren)
                || input.peek(Token![as])
                || input.peek(syn::token::Brace)
            {
                None
            } else {
                Some(input.parse::<Expr>()?)
            };

            // 兼容 `node "name" as ident { ... }`：syn 会把它解析成 `ExprCast`，
            // 从而“吞掉” DSL 的 `as ident` 绑定。
            // 这里把 `"name" as ident` 重新解释为：name = "name"，bind = ident。
            let mut bind_from_name: Option<Ident> = None;
            if let Some(Expr::Cast(expr_cast)) = &name {
                if let syn::Type::Path(type_path) = &*expr_cast.ty {
                    if type_path.qself.is_none() && type_path.path.segments.len() == 1 {
                        bind_from_name = Some(type_path.path.segments[0].ident.clone());
                        name = Some((*expr_cast.expr).clone());
                    }
                }
            }

            // 可选 (id = expr)
            let id = if input.peek(syn::token::Paren) {
                let content;
                parenthesized!(content in input);
                let id_ident: Ident = content.parse()?;
                if id_ident != "id" {
                    return Err(syn::Error::new_spanned(
                        id_ident,
                        "node(...) 目前仅支持 (id = <expr>)",
                    ));
                }
                content.parse::<Token![=]>()?;
                let expr: Expr = content.parse()?;
                if !content.is_empty() {
                    // 允许多余逗号
                    let _ = content.parse::<Token![,]>();
                    if !content.is_empty() {
                        return Err(content.error("(id = <expr>) 后不应包含其他内容"));
                    }
                }
                Some(expr)
            } else {
                None
            };

            // 可选 as ident
            let mut bind = if input.peek(Token![as]) {
                input.parse::<Token![as]>()?;
                Some(input.parse::<Ident>()?)
            } else {
                None
            };

            if bind.is_some() && bind_from_name.is_some() {
                return Err(syn::Error::new(
                    span,
                    "scene!: node 绑定重复：请只使用一次 `as ident`",
                ));
            }
            if bind.is_none() {
                bind = bind_from_name;
            }

            let content;
            braced!(content in input);
            let mut items = Vec::new();
            while !content.is_empty() {
                items.push(content.parse::<NodeItem>()?);
            }

            Ok(Self {
                name,
                id,
                bind,
                items,
                span,
            })
        }
    }

    pub enum NodeItem {
        Child(NodeDecl),
        With(WithItem),
        Component(ComponentItem),
    }

    impl Parse for NodeItem {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            if input.peek(node) {
                return Ok(Self::Child(input.parse()?));
            }
            if input.peek(with) {
                return Ok(Self::With(input.parse()?));
            }
            if input.peek(Token![+]) {
                return Ok(Self::Component(input.parse()?));
            }

            Err(input.error("node 体内只允许：node ...、with(...) {...}、+ Component ...;"))
        }
    }

    pub struct WithItem {
        pub bindings: Vec<WithBinding>,
        pub block: syn::Block,
        pub span: proc_macro2::Span,
    }

    impl Parse for WithItem {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let kw: with = input.parse()?;
            let span = kw.span();

            let content;
            parenthesized!(content in input);
            let bindings: Punctuated<WithBinding, Token![,]> =
                content.parse_terminated(WithBinding::parse, Token![,])?;

            let block: syn::Block = input.parse()?;
            let _ = input.parse::<Token![;]>();

            Ok(Self {
                bindings: bindings.into_iter().collect(),
                block,
                span,
            })
        }
    }

    pub struct WithBinding {
        pub is_mut: bool,
        pub name: Ident,
        pub ty: syn::Type,
    }

    fn wants_mut_ref_in_component_config(closure: &syn::ExprClosure) -> bool {
        let mut iter = closure.inputs.iter();
        let _ = iter.next();
        let Some(second) = iter.next() else {
            return true;
        };

        match second {
            syn::Pat::Ident(pat) => pat.mutability.is_some(),
            syn::Pat::Type(pat) => match pat.ty.as_ref() {
                syn::Type::Reference(r) => r.mutability.is_some(),
                _ => false,
            },
            syn::Pat::Reference(pat) => pat.mutability.is_some(),
            _ => false,
        }
    }

    fn strip_mut_from_second_param(closure: &syn::ExprClosure) -> syn::Result<syn::ExprClosure> {
        let mut rewritten = closure.clone();
        let Some(second) = rewritten.inputs.iter_mut().nth(1) else {
            return Ok(rewritten);
        };

        match second {
            syn::Pat::Ident(pat) => {
                pat.mutability = None;
                Ok(rewritten)
            }
            syn::Pat::Type(pat) => match pat.pat.as_mut() {
                syn::Pat::Ident(ident) => {
                    ident.mutability = None;
                    Ok(rewritten)
                }
                _ => Ok(rewritten),
            },
            syn::Pat::Reference(pat) => {
                pat.mutability = None;
                Ok(rewritten)
            }
            _ => Ok(rewritten),
        }
    }

    impl Parse for WithBinding {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let is_mut = if input.peek(Token![mut]) {
                input.parse::<Token![mut]>()?;
                true
            } else {
                false
            };
            let name: Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            let ty: syn::Type = input.parse()?;
            Ok(Self { is_mut, name, ty })
        }
    }

    pub struct ComponentItem {
        pub expr: Expr,
        pub config: Option<ComponentConfig>,
        pub span: proc_macro2::Span,
    }

    impl Parse for ComponentItem {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let plus: Token![+] = input.parse()?;
            let span = plus.span();
            let expr: Expr = input.parse()?;
            let config = if input.peek(Token![=>]) {
                input.parse::<Token![=>]>()?;
                Some(input.parse::<ComponentConfig>()?)
            } else {
                None
            };
            input.parse::<Token![;]>()?;
            Ok(Self { expr, config, span })
        }
    }

    pub struct ComponentConfig {
        pub resources: Vec<(Ident, Expr)>,
        pub closure: syn::ExprClosure,
    }

    impl Parse for ComponentConfig {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let mut resources = Vec::new();
            if input.peek(resource) {
                let _: resource = input.parse()?;
                let content;
                parenthesized!(content in input);
                let assigns: Punctuated<ResourceAssign, Token![,]> =
                    content.parse_terminated(ResourceAssign::parse, Token![,])?;
                resources = assigns.into_iter().map(|a| (a.name, a.path_expr)).collect();
            }

            let closure: syn::ExprClosure = input.parse()?;
            Ok(Self { resources, closure })
        }
    }

    struct ResourceAssign {
        name: Ident,
        path_expr: Expr,
    }

    impl Parse for ResourceAssign {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let name: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            let path_expr: Expr = input.parse()?;
            Ok(Self { name, path_expr })
        }
    }

    pub fn expand(scene: SceneInput) -> syn::Result<proc_macro2::TokenStream> {
        let core_crate = match crate_name("jge-core") {
            // 在 rustdoc doctest 中，宏会在一个“临时测试 crate”里展开；此时 `crate::...`
            // 不再指向 `jge-core`，会导致路径解析失败。
            // `jge-core` 在 crate 根声明了 `extern crate self as jge_core;`，因此即使在
            // crate 自身内部也可以可靠地用 `::jge_core::...` 指向自己。
            Ok(FoundCrate::Itself) => quote!(::jge_core),
            Ok(FoundCrate::Name(name)) => {
                let ident = format_ident!("{}", name);
                quote!(::#ident)
            }
            Err(_) => {
                // fallback：在 workspace 内通常就是 ::jge_core
                quote!(::jge_core)
            }
        };

        let entity_ty = quote!(#core_crate::game::entity::Entity);
        let node_ty = quote!(#core_crate::game::component::node::Node);

        let mut stmts: Vec<proc_macro2::TokenStream> = Vec::new();
        let mut binds: Vec<Ident> = Vec::new();
        let mut counter: usize = 0;

        let root_var = format_ident!("__jge_scene_root");
        expand_node(
            &scene.root,
            None,
            &root_var,
            &mut counter,
            &mut stmts,
            &mut binds,
            &core_crate,
            &entity_ty,
            &node_ty,
        )?;

        // 生成 bindings 结构体（局部类型）
        // 注意：`proc_macro2::Ident` 的相等性会受 span 影响；这里按名字去重。
        // 同时跳过保留字段 `root`（SceneBindings 固定包含 root）。
        let mut unique_binds = Vec::<Ident>::new();
        let mut seen = ::std::collections::HashSet::<String>::new();
        for b in binds {
            let name = b.to_string();
            if name == "root" {
                continue;
            }
            if seen.insert(name) {
                unique_binds.push(b);
            }
        }

        let bindings_struct = {
            let fields = unique_binds.iter().map(|b| quote!(pub #b: #entity_ty));
            quote! {
                #[allow(non_camel_case_types)]
                pub struct SceneBindings {
                    pub root: #entity_ty,
                    #(#fields,)*
                }
            }
        };

        let bindings_ctor_fields = unique_binds.iter().map(|b| quote!(#b: #b));

        Ok(quote! {{
            use ::anyhow::Context as _;
            #bindings_struct

            #(#stmts)*

            ::core::result::Result::<SceneBindings, ::anyhow::Error>::Ok(SceneBindings {
                root: #root_var,
                #(#bindings_ctor_fields,)*
            })
        }})
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_node(
        node: &NodeDecl,
        parent: Option<&Ident>,
        out_var: &Ident,
        counter: &mut usize,
        stmts: &mut Vec<proc_macro2::TokenStream>,
        binds: &mut Vec<Ident>,
        core_crate: &proc_macro2::TokenStream,
        entity_ty: &proc_macro2::TokenStream,
        node_ty: &proc_macro2::TokenStream,
    ) -> syn::Result<()> {
        let span = node.span;
        let create_stmt = if let Some(id_expr) = &node.id {
            quote_spanned! {span=>
                let #out_var: #entity_ty = #core_crate::game::entity::Entity::new_with_id(#id_expr)
                    .with_context(|| "scene!: 创建实体失败")?;
            }
        } else {
            quote_spanned! {span=>
                let #out_var: #entity_ty = #core_crate::game::entity::Entity::new()
                    .with_context(|| "scene!: 创建实体失败")?;
            }
        };
        stmts.push(create_stmt);

        // 可选：设置节点名称（若用户提供 name，则覆盖默认名）
        if let Some(name_expr) = &node.name {
            stmts.push(quote_spanned! {span=>
                {
                    let mut __node = #out_var
                        .get_component_mut::<#node_ty>()
                        .with_context(|| "scene!: 实体缺少 Node 组件（无法设置名称）")?;
                    __node
                        .set_name(#name_expr)
                        .with_context(|| "scene!: 设置 Node 名称失败")?;
                }
            });
        }

        // 可选：挂到父节点
        if let Some(parent_var) = parent {
            stmts.push(quote_spanned! {span=>
                {
                    let mut __parent_node = #parent_var
                        .get_component_mut::<#node_ty>()
                        .with_context(|| "scene!: 父节点缺少 Node 组件（无法挂载子节点）")?;
                    __parent_node
                        .attach(#out_var)
                        .with_context(|| "scene!: 挂载子节点失败")?;
                }
            });
        }

        // 可选：as 绑定
        if let Some(bind) = &node.bind {
            binds.push(bind.clone());
            stmts.push(quote_spanned! {span=>
                let #bind: #entity_ty = #out_var;
            });
        }

        // body items
        for item in &node.items {
            match item {
                NodeItem::Child(child) => {
                    *counter += 1;
                    let child_var = format_ident!("__jge_scene_e{}", *counter);
                    expand_node(
                        child,
                        Some(out_var),
                        &child_var,
                        counter,
                        stmts,
                        binds,
                        core_crate,
                        entity_ty,
                        node_ty,
                    )?;
                }
                NodeItem::With(with_item) => {
                    let with_span = with_item.span;
                    let block = &with_item.block;
                    let binding_stmts = with_item.bindings.iter().map(|b| {
                        let name = &b.name;
                        let ty = &b.ty;
                        *counter += 1;
                        let guard = format_ident!("__jge_with_guard{}", *counter);

                        if b.is_mut {
                            quote_spanned! {with_span=>
                                let mut #guard = e
                                    .get_component_mut::<#ty>()
                                    .with_context(|| format!("scene!: with 缺少组件：{}", stringify!(#ty)))?;
                                let #name: &mut #ty = &mut *#guard;
                            }
                        } else {
                            quote_spanned! {with_span=>
                                let #guard = e
                                    .get_component::<#ty>()
                                    .with_context(|| format!("scene!: with 缺少组件：{}", stringify!(#ty)))?;
                                let #name: &#ty = &*#guard;
                            }
                        }
                    });

                    stmts.push(quote_spanned! {with_span=>
                        {
                            let e: #entity_ty = #out_var;
                            let _ = e;
                            #(#binding_stmts)*
                            (|| -> ::anyhow::Result<()> #block)()?;
                        }
                    });
                }
                NodeItem::Component(component) => {
                    let comp_span = component.span;
                    let comp_expr = &component.expr;
                    if let Some(cfg) = &component.config {
                        let closure = &cfg.closure;
                        let resources = cfg.resources.iter().map(|(name, path_expr)| {
                            quote_spanned! {comp_span=>
                                let #name = {
                                    let __path: #core_crate::resource::ResourcePath = (#path_expr).into();
                                    #core_crate::resource::Resource::from(__path)
                                        .ok_or_else(|| ::anyhow::anyhow!("scene!: 资源未注册：{}", stringify!(#path_expr)))?
                                };
                            }
                        });

                        if cfg.closure.inputs.len() != 2 {
                            return Err(syn::Error::new_spanned(
                                closure,
                                "scene!: 组件配置闭包必须是 |e, c| 或 |e, mut c|（两个参数）",
                            ));
                        }

                        let wants_mut_ref = wants_mut_ref_in_component_config(closure);

                        // For `|e, mut c|` configs, `mut` is a DSL signal for `&mut C`.
                        // The binding itself does not need to be mutable; strip it to avoid
                        // triggering `unused_mut` lints in downstream crates.
                        let closure_for_call = if wants_mut_ref {
                            strip_mut_from_second_param(closure)?
                        } else {
                            closure.clone()
                        };

                        *counter += 1;
                        let comp_tmp = format_ident!("__jge_scene_comp{}", *counter);
                        let apply_fn = format_ident!("__jge_scene_apply_cfg{}", *counter);
                        if wants_mut_ref {
                            stmts.push(quote_spanned! {comp_span=>
                                {
                                    #(#resources)*
                                    let mut #comp_tmp = #comp_expr;
                                    fn #apply_fn<C, F>(e: #entity_ty, c: &mut C, f: F) -> ::anyhow::Result<()>
                                    where
                                        F: FnOnce(#entity_ty, &mut C) -> ::anyhow::Result<()>,
                                    {
                                        f(e, c)
                                    }
                                    #apply_fn(#out_var, &mut #comp_tmp, #closure_for_call)?;
                                    let _ = #out_var
                                        .register_component(#comp_tmp)
                                        .with_context(|| format!("scene!: 注册组件失败：{}", stringify!(#comp_expr)))?;
                                }
                            });
                        } else {
                            stmts.push(quote_spanned! {comp_span=>
                                {
                                    #(#resources)*
                                    let mut #comp_tmp = #comp_expr;
                                    fn #apply_fn<C, F>(e: #entity_ty, c: &C, f: F) -> ::anyhow::Result<()>
                                    where
                                        F: FnOnce(#entity_ty, &C) -> ::anyhow::Result<()>,
                                    {
                                        f(e, c)
                                    }
                                    #apply_fn(#out_var, &#comp_tmp, #closure_for_call)?;
                                    let _ = #out_var
                                        .register_component(#comp_tmp)
                                        .with_context(|| format!("scene!: 注册组件失败：{}", stringify!(#comp_expr)))?;
                                }
                            });
                        }
                    } else {
                        stmts.push(quote_spanned! {comp_span=>
                            {
                                let _ = #out_var
                                    .register_component(#comp_expr)
                                    .with_context(|| format!("scene!: 注册组件失败：{}", stringify!(#comp_expr)))?;
                            }
                        });
                    }
                }
            }
        }

        Ok(())
    }
}

pub fn expand_scene(input: TokenStream) -> TokenStream {
    let input2 = proc_macro2::TokenStream::from(input.clone());

    let callsite_file = PathBuf::from(input.clone().into_iter().next().unwrap().span().file());

    // 支持：scene!("path/to/file.jgs") —— 编译期从文件读取 DSL。
    if let Ok(path_lit) = syn::parse2::<syn::LitStr>(input2.clone()) {
        return expand_scene_from_file(path_lit, callsite_file);
    }

    let parsed = parse_macro_input!(input as scene_dsl::SceneInput);
    match scene_dsl::expand(parsed) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_scene_from_file(path_lit: syn::LitStr, callsite_file: PathBuf) -> TokenStream {
    let raw_path = path_lit.value();
    let path = PathBuf::from(&raw_path);

    // 关键：让 Cargo/rustc 自动追踪该文件，并在文件变更时触发重编译。
    // 我们在展开代码里注入一个 `include_str!` 的“哑引用”，它属于调用方 crate，
    // 因此 rustc 会把该文件记录为编译依赖。
    let dep_include = if path.is_absolute() {
        quote! {
            const _: &str = include_str!(#path_lit);
        }
    } else {
        if cfg!(feature = "callsite_relative_paths") {
            // 让依赖追踪语义与调用点一致：相对路径以源文件目录为基准。
            quote! {
                const _: &str = include_str!(#path_lit);
            }
        } else {
            // stable 默认：相对路径以调用方 crate 的 `src/` 为基准。
            // 这匹配绝大多数调用点都位于 `src/*.rs` 的场景（相对路径语义接近“当前源文件目录”）。
            quote! {
                const _: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/", #path_lit));
            }
        }
    };

    let resolved = if path.is_absolute() {
        path
    } else {
        let base = callsite_file
            .parent()
            .map(PathBuf::from)
            .or_else(|| std::env::var("CARGO_MANIFEST_DIR").ok().map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("."));
        base.join(path)
    };

    let src = match std::fs::read_to_string(&resolved) {
        Ok(content) => content,
        Err(err) => {
            return syn::Error::new(
                path_lit.span(),
                format!(
                    "scene!(\"...\") 无法读取场景文件：{}\n解析为路径：{}\n错误：{}",
                    raw_path,
                    resolved.display(),
                    err
                ),
            )
            .to_compile_error()
            .into();
        }
    };

    let parsed = match syn::parse_str::<scene_dsl::SceneInput>(&src) {
        Ok(parsed) => parsed,
        Err(err) => {
            return syn::Error::new(
                path_lit.span(),
                format!(
                    "scene!(\"...\") 解析场景文件失败：{}\n解析为路径：{}\n解析错误：{}",
                    raw_path,
                    resolved.display(),
                    err
                ),
            )
            .to_compile_error()
            .into();
        }
    };

    match scene_dsl::expand(parsed) {
        Ok(tokens) => quote!({
            #dep_include
            #tokens
        })
        .into(),
        Err(err) => err.to_compile_error().into(),
    }
}
