use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use quote::{format_ident, quote};
use syn::{LitStr, parse2};

use std::collections::HashSet;
use std::path::PathBuf;

fn is_rust_analyzer_env() -> bool {
    // rust-analyzer 的 proc-macro 扩展通常运行在 VS Code extension host 进程环境里，
    // 该环境会携带一组稳定的 VS Code/Electron 相关变量。这里仅用用户提供的变量名判断，
    // 避免误伤正常 cargo build/test 进程。
    std::env::var("ELECTRON_RUN_AS_NODE").ok().as_deref() == Some("1")
        && std::env::var("VSCODE_CRASH_REPORTER_PROCESS_TYPE")
            .ok()
            .as_deref()
            == Some("extensionHost")
        && std::env::var("VSCODE_IPC_HOOK").ok().is_some()
}

fn resolve_from_for_yaml_file(from: &str, yaml_dir: Option<&PathBuf>) -> String {
    let Some(yaml_dir) = yaml_dir else {
        return from.to_string();
    };

    let from_path = PathBuf::from(from);
    if from_path.is_absolute() {
        return from.to_string();
    }
    if yaml_dir.as_os_str().is_empty() {
        return from.to_string();
    }

    yaml_dir.join(from_path).to_string_lossy().into_owned()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResourceKind {
    Embed,
    Fs,
    Txt,
    Bin,
    Dir,
}

impl ResourceKind {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "embed" => Some(Self::Embed),
            "fs" => Some(Self::Fs),
            "txt" => Some(Self::Txt),
            "bin" => Some(Self::Bin),
            "dir" => Some(Self::Dir),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::Fs => "fs",
            Self::Txt => "txt",
            Self::Bin => "bin",
            Self::Dir => "dir",
        }
    }
}

#[derive(Debug)]
struct ResourceEntry {
    logical_path: String,
    kind: ResourceKind,
    from: Option<String>,
    txt: Option<String>,
    bin: Option<Vec<u8>>,
    // dir 类型不直接携带数据，from 字段指向目录根路径
}

pub fn expand_resource(input: TokenStream) -> TokenStream {
    // rust-analyzer 会在分析期执行 proc-macro 展开；该阶段经常无法可靠访问磁盘路径，
    // 容易出现“路径不存在”的误报。为避免污染编辑器诊断，这里在 RA 环境下直接短路：
    // 不做任何 YAML 解析 / 文件读取 / include_* 注入。
    if is_rust_analyzer_env() {
        return quote! {{
            ::core::result::Result::<(), ::anyhow::Error>::Ok(())
        }}
        .into();
    }

    // 支持空输入：resource!() / resource!{} / resource! —— 直接生成 Ok(())。
    // 目的：允许调用方统一在末尾写 `?`，并在“可选资源表”情况下把该宏当作 no-op。
    if input.is_empty() {
        return quote! {{
            ::core::result::Result::<(), ::anyhow::Error>::Ok(())
        }}
        .into();
    }

    let input2 = proc_macro2::TokenStream::from(input.clone());

    let callsite_file = match input.clone().into_iter().next() {
        Some(tt) => PathBuf::from(tt.span().file()),
        None => {
            return syn::Error::new(
                proc_macro2::Span::call_site(),
                "resource!: 需要一个 YAML 字符串字面量，或一个 .yaml/.yml 文件路径字符串字面量",
            )
            .to_compile_error()
            .into();
        }
    };

    // 注意：proc-macro span 给出的 file 可能是相对路径（例如 "src/foo.rs"）。
    // 我们需要把它解析为“调用点源文件所在目录”的绝对路径，用于在宏展开期访问文件系统。
    let callsite_dir = {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));

        let callsite_file_fs = if callsite_file.is_absolute() {
            callsite_file.clone()
        } else {
            // rustc/proc-macro 提供的 span file 有两种常见形式：
            // - "src/foo.rs"（相对 crate 根）
            // - "crate-name/src/foo.rs"（相对 workspace 根）
            // 这里优先按 crate 根拼接；若不存在再尝试按 workspace 根拼接。
            let joined = manifest_dir.join(&callsite_file);
            if joined.exists() {
                joined
            } else {
                let alt = manifest_dir.parent().map(|p| p.join(&callsite_file));
                if let Some(alt) = alt
                    && alt.exists()
                {
                    alt
                } else {
                    joined
                }
            }
        };

        callsite_file_fs
            .parent()
            .map(PathBuf::from)
            .unwrap_or(manifest_dir)
    };

    let lit = match parse2::<LitStr>(input2.clone()) {
        Ok(lit) => lit,
        Err(_) => {
            return syn::Error::new(
                proc_macro2::Span::call_site(),
                "resource!: 当前仅支持一个字符串字面量参数：resource!(r#\"...yaml...\"#) 或 resource!(\"assets/resources.yaml\")",
            )
            .to_compile_error()
            .into();
        }
    };

    let raw = lit.value();

    // 选择输入来源：
    // - 若字符串像文件路径（.yaml/.yml）：读取该文件（必须存在）
    // - 否则视为内联 YAML
    //
    // 重要：当从文件读取 YAML 时，`from:` 的相对路径应当相对该 YAML 文件的目录。
    // 内联 YAML 则维持原语义（相对宏调用点源文件）。
    let (yaml_src, dep_include, yaml_dir_for_from) =
        match maybe_read_yaml_file(&lit, &raw, &callsite_dir) {
            Ok(v) => v,
            Err(err) => return err.to_compile_error().into(),
        };

    let doc: serde_yaml::Value = match serde_yaml::from_str(&yaml_src) {
        Ok(v) => v,
        Err(err) => {
            return syn::Error::new(lit.span(), format!("resource!: YAML 解析失败：{}", err))
                .to_compile_error()
                .into();
        }
    };

    let mut entries = Vec::<ResourceEntry>::new();
    if let Err(err) = parse_root(
        &doc,
        &callsite_dir,
        yaml_dir_for_from.as_ref(),
        &mut entries,
    ) {
        return err.to_compile_error().into();
    }

    // 检查逻辑路径重复（提前给出更清晰的错误）。
    let mut seen = HashSet::<String>::new();
    for e in &entries {
        if !seen.insert(e.logical_path.clone()) {
            return syn::Error::new(
                lit.span(),
                format!("resource!: 逻辑路径重复：{}", e.logical_path),
            )
            .to_compile_error()
            .into();
        }
    }

    let core_crate = match crate_name("jge-core") {
        Ok(FoundCrate::Itself) => quote!(::jge_core),
        Ok(FoundCrate::Name(name)) => {
            let ident = format_ident!("{}", name);
            quote!(::#ident)
        }
        Err(_) => quote!(::jge_core),
    };

    let register_stmts = entries.iter().map(|e| {
        let logical_path = LitStr::new(&e.logical_path, proc_macro2::Span::call_site());
        match e.kind {
            ResourceKind::Embed => {
                let from = e.from.as_ref().expect("embed must have from");
                let from_lit = LitStr::new(from, proc_macro2::Span::call_site());
                quote! {
                    #core_crate::resource::Resource::register(
                        #core_crate::resource::ResourcePath::from(#logical_path),
                        #core_crate::resource::Resource::from_memory(::std::vec::Vec::from(include_bytes!(#from_lit))),
                    )
                    .with_context(|| format!("resource!: 注册资源失败（{}:{}）", #logical_path, #from_lit))?;
                }
            }
            ResourceKind::Fs => {
                let from = e.from.as_ref().expect("fs must have from");
                let from_lit = LitStr::new(from, proc_macro2::Span::call_site());
                quote! {
                    {
                        let __file = ::std::path::Path::new(file!());
                        // 在某些环境下 file!() 可能包含包目录前缀（例如 "jge-core/src/foo.rs"）。
                        // 这里去掉该前缀，保证与 CARGO_MANIFEST_DIR 拼接后路径正确。
                        let __file = __file.strip_prefix(env!("CARGO_PKG_NAME")).unwrap_or(__file);
                        let __base = ::std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(__file);
                        let __dir = __base
                            .parent()
                            .ok_or_else(|| ::anyhow::anyhow!("resource!: 无法解析调用点文件父目录：{}", file!()))?;
                        let __path = __dir.join(#from_lit);
                        #core_crate::resource::Resource::register(
                            #core_crate::resource::ResourcePath::from(#logical_path),
                            #core_crate::resource::Resource::from_file(__path.as_path()),
                        )
                        .with_context(|| format!("resource!: 注册资源失败（{}:{}）", #logical_path, #from_lit))?;
                    }
                }
            }
            ResourceKind::Txt => {
                let txt = e.txt.as_ref().expect("txt must have txt");
                let txt_lit = LitStr::new(txt, proc_macro2::Span::call_site());
                quote! {
                    #core_crate::resource::Resource::register(
                        #core_crate::resource::ResourcePath::from(#logical_path),
                        #core_crate::resource::Resource::from_memory(::std::vec::Vec::from((#txt_lit).as_bytes())),
                    )
                    .with_context(|| format!("resource!: 注册资源失败（{}:txt）", #logical_path))?;
                }
            }
            ResourceKind::Bin => {
                let bin = e.bin.as_ref().expect("bin must have bin");
                let bytes = bin.iter().map(|b| quote!(#b));
                quote! {
                    #core_crate::resource::Resource::register(
                        #core_crate::resource::ResourcePath::from(#logical_path),
                        #core_crate::resource::Resource::from_memory(::std::vec![#(#bytes),*]),
                    )
                    .with_context(|| format!("resource!: 注册资源失败（{}:bin）", #logical_path))?;
                }
            }
            ResourceKind::Dir => {
                // dir entries are expanded at parse time into concrete fs entries;
                // there should be no direct Dir entries to register here. Emit no-op.
                quote! {}
            }
        }
    });

    quote! {{
        use ::anyhow::Context as _;
        #dep_include
        #(#register_stmts)*
        ::core::result::Result::<(), ::anyhow::Error>::Ok(())
    }}
    .into()
}

fn looks_like_yaml_path(s: &str) -> bool {
    let s = s.trim();
    s.ends_with(".yaml") || s.ends_with(".yml")
}

fn maybe_read_yaml_file(
    lit: &LitStr,
    raw: &str,
    callsite_dir: &PathBuf,
) -> Result<(String, proc_macro2::TokenStream, Option<PathBuf>), syn::Error> {
    if !looks_like_yaml_path(raw) {
        return Ok((raw.to_string(), quote! {}, None));
    }

    let path = PathBuf::from(raw);
    let path_is_absolute = path.is_absolute();
    let resolved = if path_is_absolute {
        path.clone()
    } else {
        callsite_dir.join(&path)
    };

    if !resolved.exists() {
        return Err(syn::Error::new(
            lit.span(),
            format!(
                "resource!(\"...\") YAML 文件不存在：{}\n解析为路径：{}",
                raw,
                resolved.display()
            ),
        ));
    }

    let dep_include = quote! {
        const _: &str = include_str!(#lit);
    };

    let src = std::fs::read_to_string(&resolved).map_err(|err| {
        syn::Error::new(
            lit.span(),
            format!(
                "resource!(\"...\") 无法读取 YAML 文件：{}\n解析为路径：{}\n错误：{}",
                raw,
                resolved.display(),
                err
            ),
        )
    })?;

    let yaml_dir_for_from = if path_is_absolute {
        resolved.parent().map(PathBuf::from)
    } else {
        // 这里返回“相对调用点源文件目录”的 YAML 目录路径：
        // - resource!("a/b.yaml") => yaml_dir = "a"
        // - resource!("b.yaml")   => yaml_dir = ""（空）
        PathBuf::from(raw)
            .parent()
            .map(PathBuf::from)
            .or_else(|| Some(PathBuf::new()))
    };

    Ok((src, dep_include, yaml_dir_for_from))
}

fn parse_root(
    doc: &serde_yaml::Value,
    callsite_dir: &PathBuf,
    yaml_dir_for_from: Option<&PathBuf>,
    out: &mut Vec<ResourceEntry>,
) -> Result<(), syn::Error> {
    let Some(seq) = doc.as_sequence() else {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "resource!: YAML 顶层必须是列表（以 '-' 开头的序列）",
        ));
    };

    let mut prefix = Vec::<String>::new();
    for node in seq {
        parse_node(node, callsite_dir, yaml_dir_for_from, &mut prefix, out)?;
    }
    Ok(())
}

fn parse_node(
    node: &serde_yaml::Value,
    callsite_dir: &PathBuf,
    yaml_dir_for_from: Option<&PathBuf>,
    prefix: &mut Vec<String>,
    out: &mut Vec<ResourceEntry>,
) -> Result<(), syn::Error> {
    let Some(map) = node.as_mapping() else {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "resource!: 每个列表元素必须是一个 map（目录或资源节点）",
        ));
    };

    // 目录：只有一个 key，且 value 是 sequence
    if map.len() == 1 {
        let (k, v) = map.iter().next().unwrap();
        if let Some(dir_name) = as_plain_key(k) {
            if let Some(children) = v.as_sequence() {
                validate_segment(&dir_name)?;
                prefix.push(dir_name);
                for child in children {
                    parse_node(child, callsite_dir, yaml_dir_for_from, prefix, out)?;
                }
                prefix.pop();
                return Ok(());
            }
        }
    }

    // 资源：一个“资源名: kind” + 根据 kind 的 from/txt
    let mut res_name: Option<String> = None;
    let mut kind: Option<ResourceKind> = None;
    let mut from: Option<String> = None;
    let mut txt: Option<String> = None;
    let mut bin: Option<Vec<u8>> = None;

    for (k, v) in map {
        let Some(key) = as_plain_key(k) else { continue };
        match key.as_str() {
            "from" => {
                from = v
                    .as_str()
                    .map(|s| resolve_from_for_yaml_file(s, yaml_dir_for_from));
            }
            "txt" => {
                txt = parse_txt_value(v)?;
            }
            "bin" => {
                bin = parse_bin_value(v)?;
            }
            _ => {
                // 资源名键
                if res_name.is_some() {
                    return Err(syn::Error::new(
                        proc_macro2::Span::call_site(),
                        "resource!: 资源节点只能包含一个资源名键（除 from/txt 外）",
                    ));
                }
                let Some(vs) = v.as_str() else {
                    return Err(syn::Error::new(
                        proc_macro2::Span::call_site(),
                        format!(
                            "resource!: 资源类型必须是字符串（embed/fs/txt/bin/dir），但 {key} 的值不是字符串"
                        ),
                    ));
                };
                let Some(kd) = ResourceKind::parse(vs) else {
                    return Err(syn::Error::new(
                        proc_macro2::Span::call_site(),
                        format!("resource!: 未知资源类型：{vs}（仅支持 embed/fs/txt/bin/dir）"),
                    ));
                };
                validate_segment(&key)?;
                res_name = Some(key);
                kind = Some(kd);
            }
        }
    }

    let Some(res_name) = res_name else {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "resource!: 资源节点缺少资源名（例如：- foo.png: embed）",
        ));
    };
    let Some(kind) = kind else {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "resource!: 资源节点缺少类型（embed/fs/txt/bin/dir）",
        ));
    };

    match kind {
        ResourceKind::Embed | ResourceKind::Fs => {
            if from.as_deref().unwrap_or("").is_empty() {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("resource!: {res_name}: {} 需要 from 字段", kind.as_str()),
                ));
            }
            if txt.is_some() {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("resource!: {res_name}: {} 不允许 txt 字段", kind.as_str()),
                ));
            }
            if bin.is_some() {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("resource!: {res_name}: {} 不允许 bin 字段", kind.as_str()),
                ));
            }
        }
        ResourceKind::Txt => {
            if txt.as_deref().unwrap_or("").is_empty() {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("resource!: {res_name}: txt 需要 txt 字段",),
                ));
            }
            if from.is_some() {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("resource!: {res_name}: txt 不允许 from 字段"),
                ));
            }
            if bin.is_some() {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("resource!: {res_name}: txt 不允许 bin 字段"),
                ));
            }
        }
        ResourceKind::Bin => {
            if bin.as_deref().unwrap_or_default().is_empty() {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("resource!: {res_name}: bin 需要 bin 字段"),
                ));
            }
            if from.is_some() {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("resource!: {res_name}: bin 不允许 from 字段"),
                ));
            }
            if txt.is_some() {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("resource!: {res_name}: bin 不允许 txt 字段"),
                ));
            }
        }
        ResourceKind::Dir => {
            // dir 在这里不直接产生单个 entry：宏在编译期读取目录结构，
            // 并为目录下的每个文件注册为对应的 fs 资源条目（保持子目录层级）。
            let from_path = from.as_deref().unwrap_or("");
            if from_path.is_empty() {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("resource!: {res_name}: dir 需要 from 字段"),
                ));
            }
            let dir = std::path::PathBuf::from(from_path);
            let dir_fs_path = if dir.is_absolute() {
                dir
            } else {
                callsite_dir.join(dir)
            };

            if !dir_fs_path.exists() || !dir_fs_path.is_dir() {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!(
                        "resource!: {res_name}: dir 指定的路径不存在或不是目录：{}",
                        from_path
                    ),
                ));
            }

            // 遍历目录并把文件注册为 fs 资源，逻辑路径以当前节点为根。
            fn push_dir_entries(
                base_dir: &std::path::Path,
                callsite_dir: &std::path::Path,
                rel_prefix: &Vec<String>,
                out: &mut Vec<ResourceEntry>,
            ) -> Result<(), syn::Error> {
                let mut entries = std::fs::read_dir(base_dir)
                    .map_err(|e| {
                        syn::Error::new(
                            proc_macro2::Span::call_site(),
                            format!("resource!: 无法读取目录 {}: {}", base_dir.display(), e),
                        )
                    })?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| {
                        syn::Error::new(
                            proc_macro2::Span::call_site(),
                            format!("resource!: 读取目录项失败: {}", e),
                        )
                    })?;

                entries.sort_by_key(|e| e.file_name().to_string_lossy().into_owned());

                for entry in entries {
                    let path = entry.path();
                    let name = entry.file_name().into_string().map_err(|_| {
                        syn::Error::new(
                            proc_macro2::Span::call_site(),
                            "resource!: 目录项名称不是有效 Unicode",
                        )
                    })?;
                    if path.is_dir() {
                        let mut next = rel_prefix.clone();
                        next.push(name);
                        push_dir_entries(&path, callsite_dir, &next, out)?;
                    } else if path.is_file() {
                        let mut segs = rel_prefix.clone();
                        segs.push(name);
                        let logical = segs.join("/");

                        let from = match path.strip_prefix(callsite_dir) {
                            Ok(rel) => rel.to_string_lossy().into_owned(),
                            Err(_) => path.to_string_lossy().into_owned(),
                        };

                        out.push(ResourceEntry {
                            logical_path: logical,
                            kind: ResourceKind::Fs,
                            from: Some(from),
                            txt: None,
                            bin: None,
                        });
                    }
                }
                Ok(())
            }

            let mut base = prefix.clone();
            base.push(res_name.clone());
            push_dir_entries(&dir_fs_path, callsite_dir, &base, out)?;
            // 已展开目录为具体文件条目，返回即可。
            return Ok(());
        }
    }

    let mut segments = prefix.clone();
    segments.push(res_name);

    let logical_path = segments.join("/");

    out.push(ResourceEntry {
        logical_path,
        kind,
        from,
        txt,
        bin,
    });

    Ok(())
}

fn as_plain_key(v: &serde_yaml::Value) -> Option<String> {
    v.as_str().map(|s| s.to_string())
}

fn validate_segment(seg: &str) -> Result<(), syn::Error> {
    if seg.is_empty() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "resource!: 路径段不能为空",
        ));
    }
    if seg.contains('/') {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("resource!: 路径段不允许包含 '/': {seg}"),
        ));
    }
    Ok(())
}

fn parse_txt_value(v: &serde_yaml::Value) -> Result<Option<String>, syn::Error> {
    if let Some(s) = v.as_str() {
        return Ok(Some(s.to_string()));
    }

    Err(syn::Error::new(
        proc_macro2::Span::call_site(),
        "resource!: txt 字段必须是 YAML 字符串（请使用 `|` 块标量）",
    ))
}

fn parse_bin_value(v: &serde_yaml::Value) -> Result<Option<Vec<u8>>, syn::Error> {
    if let Some(s) = v.as_str() {
        let bytes = parse_hex_byte_blob(s)?;
        return Ok(Some(bytes));
    }

    Err(syn::Error::new(
        proc_macro2::Span::call_site(),
        "resource!: bin 字段必须是 YAML 字符串（请使用 `|` 块标量）",
    ))
}

fn parse_hex_byte_blob(s: &str) -> Result<Vec<u8>, syn::Error> {
    let mut out = Vec::<u8>::new();
    for token in s.split_whitespace() {
        if token.starts_with("0x") || token.starts_with("0X") {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!(
                    "resource!: bin 字节不允许带 0x 前缀（应为两位十六进制，例如 ff），但遇到：{token}"
                ),
            ));
        }

        if token.len() != 2 {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("resource!: bin 字节必须是两位十六进制（00..ff），但遇到：{token}"),
            ));
        }

        let byte = u8::from_str_radix(token, 16).map_err(|_| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("resource!: bin 字节必须是十六进制（00..ff），但遇到：{token}"),
            )
        })?;
        out.push(byte);
    }

    if out.is_empty() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "resource!: bin 内容不能为空（需要至少一个字节，例如：00 ff 7a）",
        ));
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hex_byte_blob_accepts_whitespace_separated_bytes() {
        let bytes = parse_hex_byte_blob("00 ff 7A\n10\t20").unwrap();
        assert_eq!(bytes, vec![0x00, 0xff, 0x7a, 0x10, 0x20]);
    }

    #[test]
    fn parse_hex_byte_blob_rejects_empty() {
        assert!(parse_hex_byte_blob("\n  \t").is_err());
    }

    #[test]
    fn parse_hex_byte_blob_rejects_0x_prefix() {
        assert!(parse_hex_byte_blob("0x00").is_err());
    }

    #[test]
    fn parse_hex_byte_blob_rejects_non_hex() {
        assert!(parse_hex_byte_blob("gg").is_err());
    }

    #[test]
    fn parse_hex_byte_blob_rejects_wrong_length() {
        assert!(parse_hex_byte_blob("0").is_err());
        assert!(parse_hex_byte_blob("000").is_err());
    }
}
