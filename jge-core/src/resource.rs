use std::{
    fmt::Display,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use parking_lot::RwLock;

/// 资源本体。
///
/// `Resource` 负责持有资源的二进制数据，并可选择性地关联文件系统路径用于延迟加载/缓存。
/// 通常不直接持有该类型，而是通过 [`ResourceHandle`]（`Arc<RwLock<_>>`）共享访问。
#[derive(Debug)]
pub struct Resource {
    fs_path: Option<String>,
    cached: bool, // fs_path 为 None 时，表示资源仅存在于内存中，cached 总为 true
    data: Arc<[u8]>,
}

/// 资源路径（逻辑路径）。
///
/// 资源系统使用“分段路径”组织资源，类似 `shaders/3d.vs` 这种以 `/` 分隔的层级。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResourcePath(Vec<String>);

impl ResourcePath {
    /// 从路径段创建一个资源路径。
    ///
    /// # 示例
    ///
    /// ```
    /// let p = ::jge_core::resource::ResourcePath::new(vec!["textures".into(), "ui".into(), "a.png".into()]);
    /// assert_eq!(p.join("/"), "textures/ui/a.png");
    /// ```
    pub fn new<S: Into<Vec<String>>>(segments: S) -> Self {
        Self(segments.into())
    }

    /// 返回资源路径的分段切片。
    pub fn segments(&self) -> &[String] {
        &self.0
    }

    /// 返回路径段数量。
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// 是否为空路径（没有任何段）。
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// 用指定分隔符拼接为一个字符串。
    ///
    /// 注意：资源系统通常使用 `/` 作为逻辑分隔符。
    pub fn join(&self, separator: &str) -> String {
        self.0.join(separator)
    }
}

impl From<&str> for ResourcePath {
    fn from(value: &str) -> Self {
        let segments = value
            .split('/')
            .filter(|segment| !segment.is_empty())
            .map(|segment| segment.to_string())
            .collect::<Vec<String>>();
        Self(segments)
    }
}

impl From<String> for ResourcePath {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}

impl From<Vec<String>> for ResourcePath {
    fn from(value: Vec<String>) -> Self {
        Self(value)
    }
}

impl<'a> FromIterator<&'a str> for ResourcePath {
    fn from_iter<T: IntoIterator<Item = &'a str>>(iter: T) -> Self {
        let segments = iter.into_iter().map(|s| s.to_string()).collect();
        Self(segments)
    }
}

impl FromIterator<String> for ResourcePath {
    fn from_iter<T: IntoIterator<Item = String>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl IntoIterator for ResourcePath {
    type Item = String;
    type IntoIter = std::vec::IntoIter<String>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a ResourcePath {
    type Item = &'a String;
    type IntoIter = std::slice::Iter<'a, String>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl AsRef<[String]> for ResourcePath {
    fn as_ref(&self) -> &[String] {
        &self.0
    }
}

/// 资源句柄。
///
/// - 具备共享所有权（`Arc`）与并发读写（`RwLock`）。
/// - 支持类型参数 `T`，默认 `T = Resource`，便于未来扩展为不同资源类型。
pub type ResourceHandle<T = Resource> = Arc<RwLock<T>>;

enum NodeData {
    Directory(RwLock<Vec<ResourceNode>>),
    Resource(ResourceHandle),
}

/// 资源树节点。
///
/// 资源系统内部会把 [`ResourcePath`] 解析为分段路径，并构建一棵目录/资源混合的树。
/// 该类型主要用于内部存储结构；游戏侧通常只需要通过 [`Resource::register`] 与 [`Resource::from`] 访问资源。
pub struct ResourceNode {
    name: String,
    data: NodeData,

    /// 若存在，则该目录节点表示一个“运行时从文件系统映射的子树”。
    ///
    /// - 相对路径：按进程当前工作目录（cwd）解析。
    /// - 绝对路径：直接使用。
    fs_root: Option<PathBuf>,
}

/// 资源系统错误。
#[derive(Debug)]
pub enum ResourceError {
    /// 资源路径冲突：尝试在“已经是资源”的路径下继续注册子路径，或与已有节点类型不一致。
    PathConflict,
}

impl Display for ResourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResourceError::PathConflict => {
                write!(f, "资源路径冲突：尝试在已有资源路径下注册新资源")
            }
        }
    }
}

impl std::error::Error for ResourceError {}

impl Resource {
    fn new_empty() -> Self {
        Self {
            fs_path: None,
            cached: true,
            data: Arc::<[u8]>::from(Vec::new()),
        }
    }

    fn new_memory(data: Vec<u8>) -> Self {
        Self {
            fs_path: None,
            cached: true,
            data: Arc::<[u8]>::from(data),
        }
    }

    fn new_file(path: &Path) -> Self {
        Self {
            fs_path: Some(path.to_string_lossy().to_string()),
            cached: false,
            data: Arc::<[u8]>::from(Vec::new()),
        }
    }

    /// 创建一个空资源句柄。
    pub fn empty() -> ResourceHandle {
        Arc::new(RwLock::new(Self::new_empty()))
    }

    /// 从内存字节创建资源句柄（已缓存）。
    pub fn from_memory(data: Vec<u8>) -> ResourceHandle {
        Arc::new(RwLock::new(Self::new_memory(data)))
    }

    /// 从磁盘路径创建资源句柄（懒加载，首次读取时缓存）。
    pub fn from_file(path: &Path) -> ResourceHandle {
        Arc::new(RwLock::new(Self::new_file(path)))
    }

    fn resources() -> &'static RwLock<Vec<ResourceNode>> {
        static RESOURCES: OnceLock<RwLock<Vec<ResourceNode>>> = OnceLock::new();
        RESOURCES.get_or_init(|| {
            let res = RwLock::new(Vec::new());
            {
                let mut root = res.write();
                Self::register_inner_in_list(
                    &mut root,
                    &ResourcePath::from("shaders/2d.vs").0,
                    0,
                    Resource::from_memory(Vec::from(include_bytes!("resource/shaders/2d.vs"))),
                )
                .unwrap();
                Self::register_inner_in_list(
                    &mut root,
                    &ResourcePath::from("shaders/2d.fs").0,
                    0,
                    Resource::from_memory(Vec::from(include_bytes!("resource/shaders/2d.fs"))),
                )
                .unwrap();
                Self::register_inner_in_list(
                    &mut root,
                    &ResourcePath::from("shaders/3d.vs").0,
                    0,
                    Resource::from_memory(Vec::from(include_bytes!("resource/shaders/3d.vs"))),
                )
                .unwrap();
                Self::register_inner_in_list(
                    &mut root,
                    &ResourcePath::from("shaders/3d.fs").0,
                    0,
                    Resource::from_memory(Vec::from(include_bytes!("resource/shaders/3d.fs"))),
                )
                .unwrap();
                Self::register_inner_in_list(
                    &mut root,
                    &ResourcePath::from("shaders/background.vs").0,
                    0,
                    Resource::from_memory(Vec::from(include_bytes!(
                        "resource/shaders/background.vs"
                    ))),
                )
                .unwrap();
                Self::register_inner_in_list(
                    &mut root,
                    &ResourcePath::from("shaders/background.fs").0,
                    0,
                    Resource::from_memory(Vec::from(include_bytes!(
                        "resource/shaders/background.fs"
                    ))),
                )
                .unwrap();
            }
            res
        })
    }

    fn register_inner_in_list(
        list: &mut Vec<ResourceNode>,
        segments: &[String],
        index: usize,
        resource: ResourceHandle,
    ) -> Result<(), ResourceError> {
        let seg = segments.get(index).ok_or(ResourceError::PathConflict)?;

        let existing_pos = list.iter().position(|n| n.name == seg.as_str());
        let is_leaf = index + 1 == segments.len();

        if is_leaf {
            if existing_pos.is_some() {
                return Err(ResourceError::PathConflict);
            }
            list.push(ResourceNode {
                name: seg.clone(),
                data: NodeData::Resource(resource),
                fs_root: None,
            });
            return Ok(());
        }

        let dir_pos = if let Some(pos) = existing_pos {
            pos
        } else {
            list.push(ResourceNode {
                name: seg.clone(),
                data: NodeData::Directory(RwLock::new(Vec::new())),
                fs_root: None,
            });
            list.len() - 1
        };

        let node = &mut list[dir_pos];
        let NodeData::Directory(children) = &mut node.data else {
            return Err(ResourceError::PathConflict);
        };
        let mut guard = children.write();
        Self::register_inner_in_list(&mut guard, segments, index + 1, resource)
    }

    /// 注册资源句柄到全局资源树。
    ///
    /// 若路径与已有节点发生冲突（例如 `a/b` 已经是资源，但又尝试注册 `a/b/c`），将返回 [`ResourceError::PathConflict`]。
    pub fn register(path: ResourcePath, resource: ResourceHandle) -> Result<(), ResourceError> {
        let ResourcePath(segments) = path;
        let mut root = Self::resources().write();
        Self::register_inner_in_list(&mut root, &segments, 0, resource)
    }

    /// 注册一个“运行时从文件系统映射的目录子树”。
    ///
    /// 语义：
    /// - 只注册目录节点本身，不会在编译期/注册期遍历文件系统。
    /// - 运行时：
    ///   - [`Self::from`] 在命中该目录（或其后代）时，会按需检查磁盘并惰性创建资源节点。
    ///   - [`Self::list_children`] 会读取该目录并把一级子项懒注册到资源树。
    pub fn register_dir(path: ResourcePath, fs_root: PathBuf) -> Result<(), ResourceError> {
        fn find_or_insert_dir_node_mut<'a>(
            list: &'a mut Vec<ResourceNode>,
            segment: &str,
        ) -> Result<&'a mut ResourceNode, ResourceError> {
            if let Some(pos) = list.iter().position(|n| n.name == segment) {
                return Ok(&mut list[pos]);
            }
            list.push(ResourceNode {
                name: segment.to_string(),
                data: NodeData::Directory(RwLock::new(Vec::new())),
                fs_root: None,
            });
            Ok(list.last_mut().expect("just pushed"))
        }

        fn register_dir_in_list(
            list: &mut Vec<ResourceNode>,
            segments: &[String],
            index: usize,
            fs_root: &PathBuf,
        ) -> Result<(), ResourceError> {
            let seg = segments.get(index).ok_or(ResourceError::PathConflict)?;
            let node = find_or_insert_dir_node_mut(list, seg)?;

            if index + 1 == segments.len() {
                match &node.data {
                    NodeData::Directory(_) => {
                        node.fs_root = Some(fs_root.clone());
                        return Ok(());
                    }
                    NodeData::Resource(_) => return Err(ResourceError::PathConflict),
                }
            }

            let NodeData::Directory(children) = &node.data else {
                return Err(ResourceError::PathConflict);
            };
            let mut guard = children.write();
            register_dir_in_list(&mut guard, segments, index + 1, fs_root)
        }

        let ResourcePath(segments) = path;
        if segments.is_empty() {
            // 不允许把根当成映射目录（避免把整个资源树绑定到 cwd）。
            return Err(ResourceError::PathConflict);
        }
        let mut root = Self::resources().write();
        register_dir_in_list(&mut root, &segments, 0, &fs_root)
    }

    fn ensure_child_from_fs(
        list: &mut Vec<ResourceNode>,
        parent_fs_root: &Path,
        child_name: &str,
    ) -> Option<usize> {
        let child_path = parent_fs_root.join(child_name);
        if child_path.is_dir() {
            list.push(ResourceNode {
                name: child_name.to_string(),
                data: NodeData::Directory(RwLock::new(Vec::new())),
                fs_root: Some(child_path),
            });
            return Some(list.len() - 1);
        }
        if child_path.is_file() {
            list.push(ResourceNode {
                name: child_name.to_string(),
                data: NodeData::Resource(Resource::from_file(child_path.as_path())),
                fs_root: None,
            });
            return Some(list.len() - 1);
        }
        None
    }

    fn maybe_inherit_fs_root(
        node: &mut ResourceNode,
        parent_fs_root: Option<&Path>,
    ) -> Option<PathBuf> {
        if let Some(root) = &node.fs_root {
            return Some(root.clone());
        }
        let parent = parent_fs_root?;
        let candidate = parent.join(&node.name);
        if candidate.is_dir() {
            node.fs_root = Some(candidate.clone());
            Some(candidate)
        } else {
            None
        }
    }

    fn from_in_list(
        list: &mut Vec<ResourceNode>,
        parent_fs_root: Option<PathBuf>,
        segments: &[String],
        index: usize,
    ) -> Option<ResourceHandle> {
        if index >= segments.len() {
            return None;
        }

        let seg = &segments[index];

        let mut node_index = list.iter().position(|n| n.name == seg.as_str());
        if node_index.is_none()
            && let Some(root) = parent_fs_root.as_deref()
        {
            node_index = Self::ensure_child_from_fs(list, root, seg);
        }
        let node_index = node_index?;

        let node = &mut list[node_index];
        let next_root = if matches!(node.data, NodeData::Directory(_)) {
            Self::maybe_inherit_fs_root(node, parent_fs_root.as_deref())
        } else {
            None
        };

        match &mut node.data {
            NodeData::Directory(children) => {
                if index + 1 == segments.len() {
                    return None;
                }
                let mut guard = children.write();
                Self::from_in_list(&mut guard, next_root, segments, index + 1)
            }
            NodeData::Resource(handle) => {
                if index + 1 == segments.len() {
                    Some(Arc::clone(handle))
                } else {
                    None
                }
            }
        }
    }

    /// 按资源路径获取已注册的资源句柄。
    ///
    /// - 返回 `None` 表示该路径未注册，或路径命中目录但不是最终资源节点。
    pub fn from(path: ResourcePath) -> Option<ResourceHandle> {
        let ResourcePath(segments) = path;
        let mut root = Self::resources().write();
        Self::from_in_list(&mut root, None, &segments, 0)
    }

    /// 列出指定目录资源路径下的直接子项名称（不区分资源类型）。
    ///
    /// 返回 `None` 表示该路径未注册或路径不是目录。
    pub fn list_children(path: ResourcePath) -> Option<Vec<String>> {
        fn list_children_in_list(
            list: &mut Vec<ResourceNode>,
            parent_fs_root: Option<PathBuf>,
            segments: &[String],
            index: usize,
        ) -> Option<Vec<String>> {
            if index == segments.len() {
                // 当前 list 对应目标目录。
                let mut names: Vec<String> = list.iter().map(|n| n.name.clone()).collect();

                if let Some(root) = parent_fs_root.as_deref()
                    && let Ok(rd) = std::fs::read_dir(root)
                {
                    for entry in rd.flatten() {
                        let Ok(name) = entry.file_name().into_string() else {
                            continue;
                        };
                        if !names.iter().any(|n| n == &name) {
                            // 懒注册一级节点（若磁盘项存在但不是文件/目录，ensure 会返回 None）
                            let _ = Resource::ensure_child_from_fs(list, root, &name);
                            names.push(name);
                        }
                    }
                }

                names.sort_unstable();
                names.dedup();
                return Some(names);
            }

            let seg = &segments[index];
            let mut node_index = list.iter().position(|n| n.name == seg.as_str());
            if node_index.is_none() {
                // list_children 需要目录：若磁盘上存在对应目录，则惰性创建。
                if let Some(root) = parent_fs_root.as_deref() {
                    let p = root.join(seg);
                    if p.is_dir() {
                        list.push(ResourceNode {
                            name: seg.clone(),
                            data: NodeData::Directory(RwLock::new(Vec::new())),
                            fs_root: Some(p),
                        });
                        node_index = Some(list.len() - 1);
                    }
                }
            }
            let node_index = node_index?;

            let node = &mut list[node_index];
            let next_root = Resource::maybe_inherit_fs_root(node, parent_fs_root.as_deref());
            let NodeData::Directory(children) = &node.data else {
                return None;
            };
            let mut guard = children.write();
            list_children_in_list(&mut guard, next_root, segments, index + 1)
        }

        let ResourcePath(segments) = path;
        let mut root = Self::resources().write();
        list_children_in_list(&mut root, None, &segments, 0)
    }

    pub fn data_loaded(&self) -> bool {
        if let Some(_fspath) = &self.fs_path {
            self.cached
        } else {
            true
        }
    }

    /// 在调用前应先结合 `data_loaded` 判断是否已经缓存。
    ///
    /// - 当资源已缓存时，返回内部数据的引用，避免写锁竞争。
    /// - 若尚未加载，将返回 `None`，调用方应退回到 [`Self::get_data`].
    pub fn try_get_data(&self) -> Option<&[u8]> {
        self.cached.then_some(self.data.as_ref())
    }

    /// 当资源已缓存时，返回共享的字节缓存（零拷贝，clone 为 O(1)）。
    ///
    /// 适合在拿到数据后立即释放锁，然后在锁外进行解码/解析等耗时操作。
    pub fn try_get_data_arc(&self) -> Option<Arc<[u8]>> {
        self.cached.then(|| Arc::clone(&self.data))
    }

    /// 确保资源已加载并返回内部数据。
    ///
    /// - 当资源尚未缓存时，会尝试从文件系统读取并缓存。
    /// - 调用该方法需要可变引用，建议在调用前先尝试 [`Self::try_get_data`]
    ///   以减少不必要的写锁获取。
    pub fn get_data(&mut self) -> &[u8] {
        if let Some(fspath) = &self.fs_path
            && !self.cached
        {
            if let Ok(bytes) = std::fs::read(fspath) {
                self.data = Arc::<[u8]>::from(bytes);
            }
            self.cached = true;
        }
        self.data.as_ref()
    }

    /// 确保资源已加载并返回共享字节缓存（零拷贝）。
    pub fn get_data_arc(&mut self) -> Arc<[u8]> {
        let _ = self.get_data();
        Arc::clone(&self.data)
    }

    /// 预加载资源确保其数据已缓存。
    ///
    /// 调用该方法会尝试加载资源数据并缓存，避免后续访问时的延迟。
    pub fn preload(&mut self) {
        let _ = self.get_data();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::{Mutex, OnceLock};
    use tempfile::NamedTempFile;

    fn cwd_mutex() -> &'static Mutex<()> {
        static CWD_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
        CWD_MUTEX.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn resource_path_from_str_parses_segments() {
        let path = ResourcePath::from("/aaa/bbb/ccc");
        let segments: Vec<&str> = path.segments().iter().map(|s| s.as_str()).collect();
        assert_eq!(segments, ["aaa", "bbb", "ccc"]);
        assert_eq!(path.join("/"), "aaa/bbb/ccc");
    }

    #[test]
    fn from_memory_prefers_try_get_data() {
        let resource = Resource::from_memory(b"hello".to_vec());
        let resource_guard = resource.read();
        assert!(resource_guard.data_loaded(), "内存资源应默认已加载");

        let maybe_data = if resource_guard.data_loaded() {
            resource_guard
                .try_get_data()
                .expect("已缓存资源应能直接读取")
        } else {
            unreachable!("from_memory 应始终已缓存");
        };

        assert_eq!(maybe_data, &b"hello".to_vec());
    }

    #[test]
    fn from_file_lazy_loads_then_caches() {
        let temp = NamedTempFile::new().expect("应能创建临时文件");
        fs::write(temp.path(), b"content").expect("应能写入测试文件");

        let resource = Resource::from_file(temp.path());
        {
            let resource_guard = resource.read();
            assert!(!resource_guard.data_loaded(), "文件资源初始不应缓存");
        }

        let mut resource_guard = resource.write();
        // 首次尝试应返回 None，随后退回到 get_data 触发加载。
        assert!(
            resource_guard.try_get_data().is_none(),
            "未缓存时应返回 None"
        );

        let bytes_ptr = {
            let bytes = if resource_guard.data_loaded() {
                resource_guard
                    .try_get_data()
                    .expect("缓存存在时应能直接读取")
            } else {
                resource_guard.get_data()
            };
            assert_eq!(bytes, b"content");
            bytes.as_ptr()
        };
        assert!(resource_guard.data_loaded(), "访问后应标记为已缓存");

        // 再次访问应命中缓存，不需写锁。
        let cached_ptr = resource_guard
            .try_get_data()
            .expect("缓存应在首次加载后保持可用")
            .as_ptr();
        assert_eq!(bytes_ptr, cached_ptr);
    }

    #[test]
    fn resource_macro_bin_registers_bytes() -> anyhow::Result<()> {
        crate::resource!(
            r#"
- test_resources:
  - inline_blob: bin
    bin: |
      00 ff 7a
      10 20
"#
        )?;

        let handle = Resource::from(ResourcePath::from("test_resources/inline_blob"))
            .expect("bin resource should be registered");
        let guard = handle.read();
        let bytes = guard
            .try_get_data()
            .expect("memory resource should be loaded");
        assert_eq!(bytes, &[0x00, 0xff, 0x7a, 0x10, 0x20]);
        Ok(())
    }

    #[test]
    fn resource_macro_dir_registers_files_and_list_children_works() -> anyhow::Result<()> {
        // `resource!` 的 fs/dir `from` 语义是“运行时相对 cwd”。
        // 为避免不同执行方式导致 cwd 不同，这里串行化并临时切到 crate 根目录。
        let _lock = cwd_mutex().lock().expect("cwd mutex poisoned");
        let original_cwd = std::env::current_dir()?;
        std::env::set_current_dir(env!("CARGO_MANIFEST_DIR"))?;

        let test_result: anyhow::Result<()> = (|| {
            crate::resource!("resource_testdata/resources_dir.yaml")?;

            let hello = Resource::from(ResourcePath::from("dir_test/bundle/hello.txt"))
                .expect("dir file should be registered");

            let hello_fs_path = hello
                .read()
                .fs_path
                .clone()
                .expect("fs resource should have fs_path");
            assert!(
                std::path::Path::new(&hello_fs_path).exists(),
                "expected hello.txt fs path to exist, got: {hello_fs_path}"
            );
            assert_eq!(std::fs::read(&hello_fs_path)?, b"hello\n".to_vec());

            {
                let mut guard = hello.write();
                let bytes = if guard.data_loaded() {
                    guard
                        .try_get_data()
                        .expect("cached data should be available")
                } else {
                    guard.get_data()
                };
                assert_eq!(bytes, b"hello\n");
            }

            let world = Resource::from(ResourcePath::from("dir_test/bundle/nested/world.txt"))
                .expect("nested dir file should be registered");
            {
                let mut guard = world.write();
                let bytes = if guard.data_loaded() {
                    guard
                        .try_get_data()
                        .expect("cached data should be available")
                } else {
                    guard.get_data()
                };
                assert_eq!(bytes, b"world\n");
            }

            assert_eq!(
                Resource::list_children(ResourcePath::from("dir_test/bundle")).unwrap(),
                vec!["hello.txt".to_string(), "nested".to_string()]
            );
            assert_eq!(
                Resource::list_children(ResourcePath::from("dir_test/bundle/nested")).unwrap(),
                vec!["world.txt".to_string()]
            );

            Ok(())
        })();

        // 尽力恢复 cwd；若失败则返回错误（保持测试环境干净）。
        std::env::set_current_dir(original_cwd)?;
        test_result
    }

    #[test]
    #[allow(unused_must_use)]
    fn resource_macro_embeddir_embeds_files_recursively() -> anyhow::Result<()> {
        crate::resource!("resource_testdata/resources_embeddir.yaml")?;

        let hello = Resource::from(ResourcePath::from("embeddir_test/bundle/hello.txt"))
            .expect("embeddir file should be registered");
        {
            let guard = hello.read();
            assert!(guard.fs_path.is_none(), "embeddir should embed in memory");
            let bytes = guard
                .try_get_data()
                .expect("embedded resource should be loaded");
            assert_eq!(bytes, b"hello\n");
        }

        let world = Resource::from(ResourcePath::from("embeddir_test/bundle/nested/world.txt"))
            .expect("embeddir nested file should be registered");
        {
            let guard = world.read();
            let bytes = guard
                .try_get_data()
                .expect("embedded resource should be loaded");
            assert_eq!(bytes, b"world\n");
        }

        assert_eq!(
            Resource::list_children(ResourcePath::from("embeddir_test/bundle")).unwrap(),
            vec!["hello.txt".to_string(), "nested".to_string()]
        );
        assert_eq!(
            Resource::list_children(ResourcePath::from("embeddir_test/bundle/nested")).unwrap(),
            vec!["world.txt".to_string()]
        );

        Ok(())
    }
}
