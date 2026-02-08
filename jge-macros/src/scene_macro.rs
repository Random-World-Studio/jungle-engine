use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use quote::{format_ident, quote};
use syn::parse_macro_input;

use std::path::PathBuf;

fn is_rust_analyzer_env() -> bool {
    // rust-analyzer 的 proc-macro 展开运行在 VS Code extension host 环境里。
    // 在该环境下我们不做任何文件读取/解析，避免“路径不存在”的误报污染编辑器诊断。
    std::env::var("ELECTRON_RUN_AS_NODE").ok().as_deref() == Some("1")
        && std::env::var("VSCODE_CRASH_REPORTER_PROCESS_TYPE")
            .ok()
            .as_deref()
            == Some("extensionHost")
        && std::env::var("VSCODE_IPC_HOOK").ok().is_some()
}

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
    syn::custom_keyword!(progress);

    #[derive(Clone)]
    struct ProgressCtx {
        tx: Ident,
        step: Ident,
        total: Ident,
    }

    fn progress_tick(ctx: &ProgressCtx, span: proc_macro2::Span) -> proc_macro2::TokenStream {
        let tx = &ctx.tx;
        let step = &ctx.step;
        let total = &ctx.total;
        quote_spanned! {span=>
            #step += 1;
            if let Some(__tx) = &#tx {
                let __p = (#step as f64) / (#total as f64);
                let _ = __tx.send(__p).await;
            }
        }
    }

    pub struct SceneInput {
        pub progress_tx: Option<Ident>,
        pub root: NodeDecl,
    }

    impl Parse for SceneInput {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            if input.is_empty() {
                return Err(input.error("scene! 需要且仅需要一个根 node {...}"));
            }

            // 可选：顶层进度汇报 `progress tx;`
            let progress_tx = if input.peek(progress) {
                let _: progress = input.parse()?;
                let tx: Ident = input.parse()?;
                input.parse::<Token![;]>()?;
                Some(tx)
            } else {
                None
            };

            let root: NodeDecl = input.parse()?;
            if !input.is_empty() {
                return Err(input.error("scene! 顶层只允许：可选的 `progress tx;` + 一个根 node"));
            }
            Ok(Self { progress_tx, root })
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
            if let Some(Expr::Cast(expr_cast)) = &name
                && let syn::Type::Path(type_path) = &*expr_cast.ty
                && type_path.qself.is_none()
                && type_path.path.segments.len() == 1
            {
                bind_from_name = Some(type_path.path.segments[0].ident.clone());
                name = Some((*expr_cast.expr).clone());
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
        Logic(LogicItem),
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
            if input.peek(Token![*]) {
                return Ok(Self::Logic(input.parse()?));
            }

            Err(input.error(
                "node 体内只允许：node ...、with(...) {...}、+ Component ...;、* LogicExpr;",
            ))
        }
    }

    pub struct LogicItem {
        pub expr: Expr,
        pub span: proc_macro2::Span,
    }

    impl Parse for LogicItem {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let star: Token![*] = input.parse()?;
            let span = star.span();
            let expr: Expr = input.parse()?;
            input.parse::<Token![;]>()?;
            Ok(Self { expr, span })
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

        let node_count = count_nodes(&scene.root);
        let edge_count = count_edges(&scene.root);
        let init_count = count_init_steps(&scene.root);
        let progress_total_steps = node_count + edge_count + init_count;

        let progress_tx_ident = format_ident!("__jge_scene_progress_tx");
        let progress_step_ident = format_ident!("__jge_scene_progress_step");
        let progress_total_ident = format_ident!("__jge_scene_progress_total");

        let progress_ctx = scene.progress_tx.as_ref().map(|_| ProgressCtx {
            tx: progress_tx_ident.clone(),
            step: progress_step_ident.clone(),
            total: progress_total_ident.clone(),
        });

        let mut create_stmts: Vec<proc_macro2::TokenStream> = Vec::new();
        let mut init_stmts: Vec<proc_macro2::TokenStream> = Vec::new();
        let mut binds: Vec<Ident> = Vec::new();
        let mut counter: usize = 0;

        // (entity, destroy_ops_var) for every node in this scene.
        // destroy ops are recorded only for *explicitly registered* components in the DSL.
        let mut destroy_pairs: Vec<(Ident, Ident)> = Vec::new();

        collect_binds(&scene.root, &mut binds);

        let root_var = format_ident!("__jge_scene_root");
        let attach_stmts = expand_node(
            &scene.root,
            &root_var,
            &mut counter,
            &mut create_stmts,
            &mut init_stmts,
            &mut destroy_pairs,
            &core_crate,
            &entity_ty,
            &node_ty,
            progress_ctx.as_ref(),
        )?;

        // 预声明 `as ident` 绑定，允许后续初始化阶段（with/组件配置等）跨节点前向引用。
        let bind_decls: Vec<proc_macro2::TokenStream> = binds
            .iter()
            .map(|b| {
                let span = b.span();
                quote_spanned! {span=>
                    let #b: #entity_ty;
                }
            })
            .collect();

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
                    #[doc(hidden)]
                    pub __jge_scene_destroy: ::std::vec::Vec<(
                        #entity_ty,
                        ::std::vec::Vec<
                            fn(
                                #entity_ty,
                            ) -> ::core::pin::Pin<
                                ::std::boxed::Box<
                                    dyn ::core::future::Future<Output = ()>
                                        + ::core::marker::Send,
                                >,
                            >,
                        >,
                    )>,
                }
            }
        };

        let bindings_ctor_fields = unique_binds.iter().map(|b| quote!(#b: #b));

        let destroy_ctor_items = destroy_pairs
            .iter()
            .map(|(entity_var, destroy_var)| quote!((#entity_var, #destroy_var)));

        let bindings_impl = quote! {
            impl SceneBindings {
                /// 销毁本次 `scene!` 构建出来的所有实体：
                ///
                /// - 语义：对每个实体，按 DSL 中显式声明的 `+ CompExpr;` 列表卸载组件；
                /// - 依赖：`Entity::unregister_component` 会调用组件的 `unregister_dependencies` 钩子；
                ///   是否会卸载依赖组件取决于组件实现策略；
                /// - 幂等：可重复调用，多次调用不会报错。
                pub async fn destroy(&self) {
                    for (e, ops) in &self.__jge_scene_destroy {
                        for op in ops {
                            op(*e).await;
                        }
                    }
                }
            }
        };

        let progress_prelude = if let Some(tx_ident) = &scene.progress_tx {
            quote! {
                let _: ::tokio::sync::mpsc::Sender<f64> = #tx_ident.clone();
                let #progress_total_ident: usize = #progress_total_steps;
                let mut #progress_step_ident: usize = 0;
                let #progress_tx_ident: ::core::option::Option<::tokio::sync::mpsc::Sender<f64>> =
                    ::core::option::Option::Some(#tx_ident.clone());

                if let ::core::option::Option::Some(__tx) = &#progress_tx_ident {
                    let _ = __tx.send(0.0).await;
                }
            }
        } else {
            quote! {}
        };

        Ok(quote! {
            async move {
                use ::anyhow::Context as _;

                #[allow(unused_macros)]
                macro_rules! __jge_scene_log_err {
                    (
                        $phase:expr,
                        $node:expr,
                        $entity_id:expr,
                        $parent_id:expr,
                        $child_id:expr,
                        $component:expr,
                        $resource_expr:expr,
                        $resource_path:expr,
                        $message:expr,
                        $res:expr
                    ) => {
                        __jge_scene_log_err!(
                            $phase,
                            $node,
                            ::core::option::Option::None::<#core_crate::game::entity::Entity>,
                            $entity_id,
                            $parent_id,
                            $child_id,
                            $component,
                            $resource_expr,
                            $resource_path,
                            $message,
                            $res
                        )
                    };

                    (
                        $phase:expr,
                        $node:expr,
                        $entity_for_name:expr,
                        $entity_id:expr,
                        $parent_id:expr,
                        $child_id:expr,
                        $component:expr,
                        $resource_expr:expr,
                        $resource_path:expr,
                        $message:expr,
                        $res:expr
                    ) => {
                        {
                            match ($res) {
                                ::core::result::Result::Ok(__ok) => ::core::result::Result::Ok(__ok),
                                ::core::result::Result::Err(__e) => {
                                    let __node_name: ::core::option::Option<::std::string::String> = match $entity_for_name {
                                        ::core::option::Option::Some(__e2) => __e2
                                            .get_component::<#core_crate::game::component::node::Node>()
                                            .await
                                            .map(|n| n.name().to_string()),
                                        ::core::option::Option::None => ::core::option::Option::None,
                                    };

                                    #core_crate::logger::__scene_log_error(
                                        #core_crate::logger::SceneLogErrorContext {
                                            phase: $phase,
                                            node: $node,
                                            node_name: __node_name,
                                            entity_id: $entity_id,
                                            parent_id: $parent_id,
                                            child_id: $child_id,
                                            component: $component,
                                            resource_expr: $resource_expr,
                                            resource_path: $resource_path,
                                            file: file!(),
                                            line: line!(),
                                            column: column!(),
                                            message: $message,
                                        },
                                        &__e,
                                    );
                                    ::core::result::Result::Err(__e)
                                }
                            }
                        }
                    };
                }
                #bindings_struct
                #bindings_impl

                #progress_prelude

                #(#bind_decls)*
                #(#create_stmts)*
                #(#init_stmts)*
                #attach_stmts

                ::core::result::Result::<SceneBindings, ::anyhow::Error>::Ok(SceneBindings {
                    root: #root_var,
                    #(#bindings_ctor_fields,)*
                    __jge_scene_destroy: ::std::vec![#(#destroy_ctor_items),*],
                })
            }
        })
    }

    fn count_nodes(node: &NodeDecl) -> usize {
        1 + node
            .items
            .iter()
            .map(|item| match item {
                NodeItem::Child(child) => count_nodes(child),
                _ => 0,
            })
            .sum::<usize>()
    }

    fn count_edges(node: &NodeDecl) -> usize {
        let direct_children = node
            .items
            .iter()
            .filter(|item| matches!(item, NodeItem::Child(_)))
            .count();
        direct_children
            + node
                .items
                .iter()
                .map(|item| match item {
                    NodeItem::Child(child) => count_edges(child),
                    _ => 0,
                })
                .sum::<usize>()
    }

    fn count_init_steps(node: &NodeDecl) -> usize {
        let mut count = 0;
        if node.name.is_some() {
            count += 1;
        }
        for item in &node.items {
            match item {
                NodeItem::Child(child) => count += count_init_steps(child),
                NodeItem::With(_) | NodeItem::Component(_) | NodeItem::Logic(_) => count += 1,
            }
        }
        count
    }

    fn collect_binds(node: &NodeDecl, out: &mut Vec<Ident>) {
        if let Some(bind) = &node.bind {
            out.push(bind.clone());
        }
        for item in &node.items {
            if let NodeItem::Child(child) = item {
                collect_binds(child, out);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_node(
        node: &NodeDecl,
        out_var: &Ident,
        counter: &mut usize,
        create_stmts: &mut Vec<proc_macro2::TokenStream>,
        init_stmts: &mut Vec<proc_macro2::TokenStream>,
        destroy_pairs: &mut Vec<(Ident, Ident)>,
        core_crate: &proc_macro2::TokenStream,
        entity_ty: &proc_macro2::TokenStream,
        node_ty: &proc_macro2::TokenStream,
        progress: Option<&ProgressCtx>,
    ) -> syn::Result<proc_macro2::TokenStream> {
        let span = node.span;

        let node_ctx_var = format_ident!("{}_scene_ctx", out_var);
        let mut ctx_parts: Vec<proc_macro2::TokenStream> = Vec::new();
        ctx_parts.push(quote!("var="));
        ctx_parts.push(quote!(stringify!(#out_var)));
        if let Some(bind) = &node.bind {
            ctx_parts.push(quote!(", as="));
            ctx_parts.push(quote!(stringify!(#bind)));
        }
        if let Some(name_expr) = &node.name {
            ctx_parts.push(quote!(", name_expr="));
            ctx_parts.push(quote!(stringify!(#name_expr)));
        }
        if let Some(id_expr) = &node.id {
            ctx_parts.push(quote!(", id_expr="));
            ctx_parts.push(quote!(stringify!(#id_expr)));
        }
        let node_ctx_expr = quote!(concat!(#(#ctx_parts),*));

        let create_stmt = if let Some(id_expr) = &node.id {
            quote_spanned! {span=>
                let #out_var: #entity_ty = __jge_scene_log_err!(
                    "create_entity",
                    #node_ctx_expr,
                    ::core::option::Option::None,
                    ::core::option::Option::None,
                    ::core::option::Option::None,
                    ::core::option::Option::None,
                    ::core::option::Option::None,
                    ::core::option::Option::None::<&#core_crate::resource::ResourcePath>,
                    "scene!: 创建实体失败",
                    #core_crate::game::entity::Entity::new_with_id(#id_expr)
                        .await
                        .with_context(|| "scene!: 创建实体失败")
                )?;
            }
        } else {
            quote_spanned! {span=>
                let #out_var: #entity_ty = __jge_scene_log_err!(
                    "create_entity",
                    #node_ctx_expr,
                    ::core::option::Option::None,
                    ::core::option::Option::None,
                    ::core::option::Option::None,
                    ::core::option::Option::None,
                    ::core::option::Option::None,
                    ::core::option::Option::None::<&#core_crate::resource::ResourcePath>,
                    "scene!: 创建实体失败",
                    #core_crate::game::entity::Entity::new()
                        .await
                        .with_context(|| "scene!: 创建实体失败")
                )?;
            }
        };
        create_stmts.push(create_stmt);

        create_stmts.push(quote_spanned! {span=>
            let #node_ctx_var: &'static str = #node_ctx_expr;
        });

        // 每个实体一份“显式组件卸载操作”列表。
        let destroy_var = format_ident!("{}_destroy", out_var);
        destroy_pairs.push((out_var.clone(), destroy_var.clone()));
        create_stmts.push(quote_spanned! {span=>
            let mut #destroy_var: ::std::vec::Vec<
                fn(
                    #entity_ty,
                ) -> ::core::pin::Pin<
                    ::std::boxed::Box<
                        dyn ::core::future::Future<Output = ()> + ::core::marker::Send,
                    >,
                >,
            > = ::std::vec::Vec::new();
        });

        // 可选：as 绑定（这里用“赋值初始化”，允许在创建阶段结束后再使用该名字）
        if let Some(bind) = &node.bind {
            create_stmts.push(quote_spanned! {span=>
                #bind = #out_var;
            });
        }

        // 进度：每创建一个实体算一步。
        if let Some(ctx) = progress {
            create_stmts.push(progress_tick(ctx, span));
        }

        // 初始化阶段：设置节点名称（若用户提供 name，则覆盖默认名）
        if let Some(name_expr) = &node.name {
            let tick = progress.map(|ctx| progress_tick(ctx, span));
            init_stmts.push(quote_spanned! {span=>
                {
                    let mut __node = __jge_scene_log_err!(
                        "set_name_get_node",
                        #node_ctx_var,
                        ::core::option::Option::Some(#out_var),
                        ::core::option::Option::Some(#out_var.id()),
                        ::core::option::Option::None,
                        ::core::option::Option::None,
                        ::core::option::Option::None,
                        ::core::option::Option::None,
                        ::core::option::Option::None::<&#core_crate::resource::ResourcePath>,
                        "scene!: 实体缺少 Node 组件（无法设置名称）",
                        #out_var
                            .get_component_mut::<#node_ty>()
                            .await
                            .with_context(|| "scene!: 实体缺少 Node 组件（无法设置名称）")
                    )?;
                    __jge_scene_log_err!(
                        "set_name",
                        #node_ctx_var,
                        ::core::option::Option::Some(#out_var),
                        ::core::option::Option::Some(#out_var.id()),
                        ::core::option::Option::None,
                        ::core::option::Option::None,
                        ::core::option::Option::None,
                        ::core::option::Option::None,
                        ::core::option::Option::None::<&#core_crate::resource::ResourcePath>,
                        "scene!: 设置 Node 名称失败",
                        __node
                            .set_name(#name_expr)
                            .with_context(|| "scene!: 设置 Node 名称失败")
                    )?;
                    #tick
                }
            });
        }

        // 子节点实体变量（用于最终集中建树，顺序必须与 DSL 声明一致）
        let mut child_vars_in_order: Vec<(Ident, proc_macro2::Span)> = Vec::new();
        let mut child_attach_tokens: Vec<proc_macro2::TokenStream> = Vec::new();

        // body items
        for item in &node.items {
            match item {
                NodeItem::Child(child) => {
                    *counter += 1;
                    let child_var = format_ident!("__jge_scene_e{}", *counter);
                    let attach_subtree = expand_node(
                        child,
                        &child_var,
                        counter,
                        create_stmts,
                        init_stmts,
                        destroy_pairs,
                        core_crate,
                        entity_ty,
                        node_ty,
                        progress,
                    )?;
                    child_vars_in_order.push((child_var, child.span));
                    child_attach_tokens.push(attach_subtree);
                }
                NodeItem::With(with_item) => {
                    let with_span = with_item.span;
                    let block = &with_item.block;
                    let tick = progress.map(|ctx| progress_tick(ctx, with_span));
                    let binding_stmts = with_item.bindings.iter().map(|b| {
                        let name = &b.name;
                        let ty = &b.ty;
                        *counter += 1;
                        let guard = format_ident!("__jge_with_guard{}", *counter);

                        if b.is_mut {
                            quote_spanned! {with_span=>
                                let mut #guard = __jge_scene_log_err!(
                                    "with_bind",
                                    #node_ctx_var,
                                    ::core::option::Option::Some(e),
                                    ::core::option::Option::Some(e.id()),
                                    ::core::option::Option::None,
                                    ::core::option::Option::None,
                                    ::core::option::Option::Some(stringify!(#ty)),
                                    ::core::option::Option::None,
                                    ::core::option::Option::None::<&#core_crate::resource::ResourcePath>,
                                    "scene!: with 缺少组件",
                                    e.get_component_mut::<#ty>()
                                        .await
                                        .with_context(|| format!(
                                            "scene!: with 缺少组件：{} (bind `{}`)",
                                            stringify!(#ty),
                                            stringify!(#name),
                                        ))
                                )?;
                                let #name: &mut #ty = &mut *#guard;
                            }
                        } else {
                            quote_spanned! {with_span=>
                                let #guard = __jge_scene_log_err!(
                                    "with_bind",
                                    #node_ctx_var,
                                    ::core::option::Option::Some(e),
                                    ::core::option::Option::Some(e.id()),
                                    ::core::option::Option::None,
                                    ::core::option::Option::None,
                                    ::core::option::Option::Some(stringify!(#ty)),
                                    ::core::option::Option::None,
                                    ::core::option::Option::None::<&#core_crate::resource::ResourcePath>,
                                    "scene!: with 缺少组件",
                                    e.get_component::<#ty>()
                                        .await
                                        .with_context(|| format!(
                                            "scene!: with 缺少组件：{} (bind `{}`)",
                                            stringify!(#ty),
                                            stringify!(#name),
                                        ))
                                )?;
                                let #name: &#ty = &*#guard;
                            }
                        }
                    });

                    init_stmts.push(quote_spanned! {with_span=>
                        {
                            let e: #entity_ty = #out_var;
                            let _ = e;
                            #(#binding_stmts)*
                            __jge_scene_log_err!(
                                "with_block",
                                #node_ctx_var,
                                ::core::option::Option::Some(e),
                                ::core::option::Option::Some(e.id()),
                                ::core::option::Option::None,
                                ::core::option::Option::None,
                                ::core::option::Option::None,
                                ::core::option::Option::None,
                                ::core::option::Option::None::<&#core_crate::resource::ResourcePath>,
                                "scene!: with 块执行失败",
                                (async move #block).await
                            )?;
                            #tick
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
                                    match #core_crate::resource::Resource::from(__path.clone()) {
                                        ::core::option::Option::Some(__res) => ::core::result::Result::Ok(__res),
                                        ::core::option::Option::None => {
                                            let __e = ::anyhow::anyhow!(
                                                "scene!: 资源未注册：{}",
                                                __path.join("/")
                                            );
                                            let __node_name = #out_var
                                                .get_component::<#node_ty>()
                                                .await
                                                .map(|n| n.name().to_string());
                                            #core_crate::logger::__scene_log_error(
                                                #core_crate::logger::SceneLogErrorContext {
                                                    phase: "resource_lookup",
                                                    node: #node_ctx_var,
                                                    node_name: __node_name,
                                                    entity_id: ::core::option::Option::Some(#out_var.id()),
                                                    parent_id: ::core::option::Option::None,
                                                    child_id: ::core::option::Option::None,
                                                    component: ::core::option::Option::Some(stringify!(#comp_expr)),
                                                    resource_expr: ::core::option::Option::Some(stringify!(#path_expr)),
                                                    resource_path: ::core::option::Option::Some(&__path),
                                                    file: file!(),
                                                    line: line!(),
                                                    column: column!(),
                                                    message: "scene!: 资源未注册",
                                                },
                                                &__e,
                                            );
                                            ::core::result::Result::Err(__e)
                                        }
                                    }?
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
                        let unreg_fn = format_ident!("__jge_scene_unreg_for{}", *counter);
                        if wants_mut_ref {
                            let tick = progress.map(|ctx| progress_tick(ctx, comp_span));
                            init_stmts.push(quote_spanned! {comp_span=>
                                {
                                    #(#resources)*
                                    let mut #comp_tmp = #comp_expr;
                                    fn #unreg_fn<C: #core_crate::game::component::Component>(_: &C) -> fn(
                                        #entity_ty,
                                    ) -> ::core::pin::Pin<
                                        ::std::boxed::Box<
                                            dyn ::core::future::Future<Output = ()>
                                                + ::core::marker::Send,
                                        >,
                                    > {
                                        |e: #entity_ty| {
                                            ::std::boxed::Box::pin(async move {
                                                let _ = e.unregister_component::<C>().await;
                                            })
                                        }
                                    }
                                    #destroy_var.push(#unreg_fn(&#comp_tmp));
                                    fn #apply_fn<C, F>(e: #entity_ty, c: &mut C, f: F) -> ::anyhow::Result<()>
                                    where
                                        F: FnOnce(#entity_ty, &mut C) -> ::anyhow::Result<()>,
                                    {
                                        f(e, c)
                                    }
                                    __jge_scene_log_err!(
                                        "component_config",
                                        #node_ctx_var,
                                        ::core::option::Option::Some(#out_var),
                                        ::core::option::Option::Some(#out_var.id()),
                                        ::core::option::Option::None,
                                        ::core::option::Option::None,
                                        ::core::option::Option::Some(stringify!(#comp_expr)),
                                        ::core::option::Option::None,
                                        ::core::option::Option::None::<&#core_crate::resource::ResourcePath>,
                                        "scene!: 组件配置执行失败",
                                        #apply_fn(#out_var, &mut #comp_tmp, #closure_for_call)
                                    )?;
                                    let _ = __jge_scene_log_err!(
                                        "register_component",
                                        #node_ctx_var,
                                        ::core::option::Option::Some(#out_var),
                                        ::core::option::Option::Some(#out_var.id()),
                                        ::core::option::Option::None,
                                        ::core::option::Option::None,
                                        ::core::option::Option::Some(stringify!(#comp_expr)),
                                        ::core::option::Option::None,
                                        ::core::option::Option::None::<&#core_crate::resource::ResourcePath>,
                                        "scene!: 注册组件失败",
                                        #out_var
                                            .register_component(#comp_tmp)
                                            .await
                                            .with_context(|| format!(
                                                "scene!: 注册组件失败：{}",
                                                stringify!(#comp_expr)
                                            ))
                                    )?;
                                    #tick
                                }
                            });
                        } else {
                            let tick = progress.map(|ctx| progress_tick(ctx, comp_span));
                            init_stmts.push(quote_spanned! {comp_span=>
                                {
                                    #(#resources)*
                                    let mut #comp_tmp = #comp_expr;
                                    fn #unreg_fn<C: #core_crate::game::component::Component>(_: &C) -> fn(
                                        #entity_ty,
                                    ) -> ::core::pin::Pin<
                                        ::std::boxed::Box<
                                            dyn ::core::future::Future<Output = ()>
                                                + ::core::marker::Send,
                                        >,
                                    > {
                                        |e: #entity_ty| {
                                            ::std::boxed::Box::pin(async move {
                                                let _ = e.unregister_component::<C>().await;
                                            })
                                        }
                                    }
                                    #destroy_var.push(#unreg_fn(&#comp_tmp));
                                    fn #apply_fn<C, F>(e: #entity_ty, c: &C, f: F) -> ::anyhow::Result<()>
                                    where
                                        F: FnOnce(#entity_ty, &C) -> ::anyhow::Result<()>,
                                    {
                                        f(e, c)
                                    }
                                    __jge_scene_log_err!(
                                        "component_config",
                                        #node_ctx_var,
                                        ::core::option::Option::Some(#out_var),
                                        ::core::option::Option::Some(#out_var.id()),
                                        ::core::option::Option::None,
                                        ::core::option::Option::None,
                                        ::core::option::Option::Some(stringify!(#comp_expr)),
                                        ::core::option::Option::None,
                                        ::core::option::Option::None::<&#core_crate::resource::ResourcePath>,
                                        "scene!: 组件配置执行失败",
                                        #apply_fn(#out_var, &#comp_tmp, #closure_for_call)
                                    )?;
                                    let _ = __jge_scene_log_err!(
                                        "register_component",
                                        #node_ctx_var,
                                        ::core::option::Option::Some(#out_var),
                                        ::core::option::Option::Some(#out_var.id()),
                                        ::core::option::Option::None,
                                        ::core::option::Option::None,
                                        ::core::option::Option::Some(stringify!(#comp_expr)),
                                        ::core::option::Option::None,
                                        ::core::option::Option::None::<&#core_crate::resource::ResourcePath>,
                                        "scene!: 注册组件失败",
                                        #out_var
                                            .register_component(#comp_tmp)
                                            .await
                                            .with_context(|| format!(
                                                "scene!: 注册组件失败：{}",
                                                stringify!(#comp_expr)
                                            ))
                                    )?;
                                    #tick
                                }
                            });
                        }
                    } else {
                        let tick = progress.map(|ctx| progress_tick(ctx, comp_span));
                        *counter += 1;
                        let comp_tmp = format_ident!("__jge_scene_comp{}", *counter);
                        let unreg_fn = format_ident!("__jge_scene_unreg_for{}", *counter);
                        init_stmts.push(quote_spanned! {comp_span=>
                            {
                                let #comp_tmp = #comp_expr;
                                fn #unreg_fn<C: #core_crate::game::component::Component>(_: &C) -> fn(
                                    #entity_ty,
                                ) -> ::core::pin::Pin<
                                    ::std::boxed::Box<
                                        dyn ::core::future::Future<Output = ()>
                                            + ::core::marker::Send,
                                    >,
                                > {
                                    |e: #entity_ty| {
                                        ::std::boxed::Box::pin(async move {
                                            let _ = e.unregister_component::<C>().await;
                                        })
                                    }
                                }
                                #destroy_var.push(#unreg_fn(&#comp_tmp));
                                let _ = __jge_scene_log_err!(
                                    "register_component",
                                    #node_ctx_var,
                                    ::core::option::Option::Some(#out_var),
                                    ::core::option::Option::Some(#out_var.id()),
                                    ::core::option::Option::None,
                                    ::core::option::Option::None,
                                    ::core::option::Option::Some(stringify!(#comp_expr)),
                                    ::core::option::Option::None,
                                    ::core::option::Option::None::<&#core_crate::resource::ResourcePath>,
                                    "scene!: 注册组件失败",
                                    #out_var
                                        .register_component(#comp_tmp)
                                        .await
                                        .with_context(|| format!(
                                            "scene!: 注册组件失败：{}",
                                            stringify!(#comp_expr)
                                        ))
                                )?;
                                #tick
                            }
                        });
                    }
                }
                NodeItem::Logic(logic_item) => {
                    let logic_span = logic_item.span;
                    let logic_expr = &logic_item.expr;
                    let tick = progress.map(|ctx| progress_tick(ctx, logic_span));
                    init_stmts.push(quote_spanned! {logic_span=>
                        {
                            let __set_logic_future = {
                                let mut __node = __jge_scene_log_err!(
                                    "set_logic_get_node",
                                    #node_ctx_var,
                                    ::core::option::Option::Some(#out_var),
                                    ::core::option::Option::Some(#out_var.id()),
                                    ::core::option::Option::None,
                                    ::core::option::Option::None,
                                    ::core::option::Option::None,
                                    ::core::option::Option::None,
                                    ::core::option::Option::None::<&#core_crate::resource::ResourcePath>,
                                    "scene!: 实体缺少 Node 组件（无法挂载 GameLogic）",
                                    #out_var
                                        .get_component_mut::<#node_ty>()
                                        .await
                                        .with_context(|| "scene!: 实体缺少 Node 组件（无法挂载 GameLogic）")
                                )?;
                                __node.set_logic(#logic_expr)
                            };
                            __set_logic_future.await;
                            #tick
                        }
                    });
                }
            }
        }

        // 最终集中建树：只保证“子节点添加顺序”与 DSL 声明一致。
        // 这里按“父 -> 直接子 -> 子树”的顺序 attach，确保树在 attach 完成后稳定可遍历。
        let direct_attach = child_vars_in_order.iter().map(|(child_var, child_span)| {
            let child_span = *child_span;
            let tick = progress.map(|ctx| progress_tick(ctx, child_span));
            quote_spanned! {child_span=>
                {
                    let __attach_future = {
                        let mut __parent_node = __jge_scene_log_err!(
                            "attach_get_parent_node",
                            #node_ctx_var,
                            ::core::option::Option::Some(#out_var),
                            ::core::option::Option::Some(#out_var.id()),
                            ::core::option::Option::Some(#out_var.id()),
                            ::core::option::Option::Some(#child_var.id()),
                            ::core::option::Option::None,
                            ::core::option::Option::None,
                            ::core::option::Option::None::<&#core_crate::resource::ResourcePath>,
                            "scene!: 父节点缺少 Node 组件（无法挂载子节点）",
                            #out_var
                                .get_component_mut::<#node_ty>()
                                .await
                                .with_context(|| "scene!: 父节点缺少 Node 组件（无法挂载子节点）")
                        )?;
                        __parent_node.attach(#child_var)
                    };
                    __jge_scene_log_err!(
                        "attach_child",
                        #node_ctx_var,
                        ::core::option::Option::Some(#out_var),
                        ::core::option::Option::Some(#out_var.id()),
                        ::core::option::Option::Some(#out_var.id()),
                        ::core::option::Option::Some(#child_var.id()),
                        ::core::option::Option::None,
                        ::core::option::Option::None,
                        ::core::option::Option::None::<&#core_crate::resource::ResourcePath>,
                        "scene!: 挂载子节点失败",
                        __attach_future.await.with_context(|| "scene!: 挂载子节点失败")
                    )?;
                    #tick
                }
            }
        });

        Ok(quote! {
            #(#direct_attach)*
            #(#child_attach_tokens)*
        })
    }
}

pub fn expand_scene(input: TokenStream) -> TokenStream {
    // 支持空输入：scene!() / scene!{} / scene! —— 直接生成 Ok(())。
    // 目的：允许调用方统一在末尾写 `?`，并在“可选场景”情况下把该宏当作 no-op。
    if input.is_empty() {
        return quote! {
            async move {
                ::core::result::Result::<(), ::anyhow::Error>::Ok(())
            }
        }
        .into();
    }

    let input2 = proc_macro2::TokenStream::from(input.clone());

    let callsite_file = PathBuf::from(input.clone().into_iter().next().unwrap().span().file());

    // 支持：scene!("path/to/file.jgs") —— 编译期从文件读取 DSL。
    if let Ok(path_lit) = syn::parse2::<syn::LitStr>(input2.clone()) {
        if is_rust_analyzer_env() {
            // RA 环境下：不读取文件、不解析 DSL、不生成节点树，仅返回最小占位实现。
            // 这能让编辑器不再报“路径不存在”，但 bindings 字段信息不会反映真实 .jgs。
            return quote! {
                async move {
                    ::core::result::Result::<_, ::anyhow::Error>::Ok({
                        struct SceneBindings {
                            pub root: ::jge_core::game::entity::Entity,
                            #[doc(hidden)]
                            pub __jge_scene_destroy: ::std::vec::Vec<(
                                ::jge_core::game::entity::Entity,
                                ::std::vec::Vec<
                                    fn(
                                        ::jge_core::game::entity::Entity,
                                    ) -> ::core::pin::Pin<
                                        ::std::boxed::Box<
                                            dyn ::core::future::Future<Output = ()>
                                                + ::core::marker::Send,
                                        >,
                                    >,
                                >,
                            )>,
                        }
                        impl SceneBindings {
                            pub async fn destroy(&self) {
                                // rust-analyzer placeholder: no-op
                            }
                        }
                        SceneBindings {
                            root: ::jge_core::game::entity::Entity::new()
                                .await
                                .expect("rust-analyzer placeholder should create entity"),
                            __jge_scene_destroy: ::std::vec::Vec::new(),
                        }
                    })
                }
            }
            .into();
        }
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
        // 让依赖追踪语义与调用点一致：相对路径以源文件目录为基准。
        quote! {
            const _: &str = include_str!(#path_lit);
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
