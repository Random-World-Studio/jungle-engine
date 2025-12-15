use std::{
    fmt::Display,
    mem,
    path::Path,
    sync::{Arc, OnceLock},
};

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

#[derive(Debug)]
pub struct Resource {
    fs_path: Option<String>,
    cached: bool, // fs_path 为 None 时，表示资源仅存在于内存中，cached 总为 true
    data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResourcePath(Vec<String>);

impl ResourcePath {
    pub fn new<S: Into<Vec<String>>>(segments: S) -> Self {
        Self(segments.into())
    }

    pub fn segments(&self) -> &[String] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

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
pub type ResourceHandle = Arc<RwLock<Resource>>;

enum NodeData {
    Directory(RwLock<Vec<Box<ResourceNode>>>),
    Resource(ResourceHandle),
}

pub struct ResourceNode {
    name: String,
    data: NodeData,
}

#[derive(Debug)]
pub enum ResourceError {
    PathConflict(Resource),
}

impl Display for ResourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResourceError::PathConflict(_) => {
                write!(f, "资源路径冲突：尝试在已有资源路径下注册新资源")
            }
        }
    }
}

impl std::error::Error for ResourceError {}

impl Resource {
    pub fn from_memory(data: Vec<u8>) -> Self {
        Self {
            fs_path: None,
            cached: true,
            data,
        }
    }

    pub fn from_file(path: &Path) -> Self {
        Self {
            fs_path: Some(path.to_string_lossy().to_string()),
            cached: false,
            data: Vec::new(),
        }
    }

    fn resources() -> &'static RwLock<Vec<Box<ResourceNode>>> {
        static RESOURCES: OnceLock<RwLock<Vec<Box<ResourceNode>>>> = OnceLock::new();
        RESOURCES.get_or_init(|| {
            let res = RwLock::new(Vec::new());
            Self::register_inner(
                unsafe { mem::transmute(res.write()) },
                ResourcePath::from("shaders/2d.vs"),
                Resource::from_memory(Vec::from(include_bytes!("resource/shaders/2d.vs"))),
            )
            .unwrap();
            Self::register_inner(
                unsafe { mem::transmute(res.write()) },
                ResourcePath::from("shaders/2d.fs"),
                Resource::from_memory(Vec::from(include_bytes!("resource/shaders/2d.fs"))),
            )
            .unwrap();
            Self::register_inner(
                unsafe { mem::transmute(res.write()) },
                ResourcePath::from("shaders/3d.vs"),
                Resource::from_memory(Vec::from(include_bytes!("resource/shaders/3d.vs"))),
            )
            .unwrap();
            Self::register_inner(
                unsafe { mem::transmute(res.write()) },
                ResourcePath::from("shaders/3d.fs"),
                Resource::from_memory(Vec::from(include_bytes!("resource/shaders/3d.fs"))),
            )
            .unwrap();
            res
        })
    }

    fn register_inner(
        mut list: RwLockWriteGuard<'static, Vec<Box<ResourceNode>>>,
        path: ResourcePath,
        resource: Resource,
    ) -> Result<(), ResourceError> {
        let transmute = |r: RwLockWriteGuard<Vec<Box<ResourceNode>>>| -> RwLockWriteGuard<'static, Vec<Box<ResourceNode>>> {
            unsafe { mem::transmute(r) }
        };

        let l = path.len();
        for (i, s) in path.into_iter().enumerate() {
            let mut node = None;
            for n in list.iter() {
                if n.name == s {
                    node = Some(&n.data);
                    break;
                }
            }

            if let None = node {
                if i + 1 == l {
                    list.push(Box::new(ResourceNode {
                        name: s.clone(),
                        data: NodeData::Resource(Arc::new(RwLock::new(resource))),
                    }));
                    return Ok(());
                } else {
                    list.push(Box::new(ResourceNode {
                        name: s.clone(),
                        data: NodeData::Directory(RwLock::new(Vec::new())),
                    }));
                    node = Some(&list.last().unwrap().data);
                };
            }

            match node.unwrap() {
                NodeData::Directory(resource_nodes) => list = transmute(resource_nodes.write()),
                NodeData::Resource(_) => break,
            }
        }

        Err(ResourceError::PathConflict(resource))
    }

    pub fn register(path: ResourcePath, resource: Resource) -> Result<(), ResourceError> {
        Self::register_inner(Self::resources().write(), path, resource)
    }

    pub fn from(path: ResourcePath) -> Option<ResourceHandle> {
        let transmute = |r: RwLockReadGuard<Vec<Box<ResourceNode>>>| -> RwLockReadGuard<'static, Vec<Box<ResourceNode>>>{
            unsafe { mem::transmute(r) }
        };

        let mut list = transmute(Self::resources().read());

        let l = path.len();
        for (i, s) in path.into_iter().enumerate() {
            let mut node = None;
            for n in list.iter() {
                if n.name == s {
                    node = Some(&n.data);
                    break;
                }
            }

            if let None = node {
                return None;
            }

            match node.unwrap() {
                NodeData::Directory(resource_nodes) => list = transmute(resource_nodes.read()),
                NodeData::Resource(resource) => {
                    if i + 1 == l {
                        return Some(Arc::clone(resource));
                    } else {
                        return None;
                    }
                }
            }
        }
        None
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
    pub fn try_get_data(&self) -> Option<&'static Vec<u8>> {
        if self.cached {
            unsafe { Some(mem::transmute(&self.data)) }
        } else {
            None
        }
    }

    /// 确保资源已加载并返回内部数据。
    ///
    /// - 当资源尚未缓存时，会尝试从文件系统读取并缓存。
    /// - 调用该方法需要可变引用，建议在调用前先尝试 [`Self::try_get_data`]
    ///   以减少不必要的写锁获取。
    pub fn get_data(&mut self) -> &'static Vec<u8> {
        if let Some(fspath) = &self.fs_path {
            if !self.cached {
                if let Ok(bytes) = std::fs::read(fspath) {
                    self.data = bytes;
                }
                self.cached = true;
            }
        }
        unsafe { mem::transmute(&self.data) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;

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
        assert!(resource.data_loaded(), "内存资源应默认已加载");

        let maybe_data = if resource.data_loaded() {
            resource.try_get_data().expect("已缓存资源应能直接读取")
        } else {
            unreachable!("from_memory 应始终已缓存");
        };

        assert_eq!(maybe_data, &b"hello".to_vec());
    }

    #[test]
    fn from_file_lazy_loads_then_caches() {
        let temp = NamedTempFile::new().expect("应能创建临时文件");
        fs::write(temp.path(), b"content").expect("应能写入测试文件");

        let mut resource = Resource::from_file(temp.path());
        assert!(!resource.data_loaded(), "文件资源初始不应缓存");

        // 首次尝试应返回 None，随后退回到 get_data 触发加载。
        assert!(resource.try_get_data().is_none(), "未缓存时应返回 None");

        let bytes = if resource.data_loaded() {
            resource.try_get_data().expect("缓存存在时应能直接读取")
        } else {
            resource.get_data()
        };
        assert_eq!(bytes, &b"content".to_vec());
        assert!(resource.data_loaded(), "访问后应标记为已缓存");

        // 再次访问应命中缓存，不需写锁。
        let cached = resource.try_get_data().expect("缓存应在首次加载后保持可用");
        assert!(std::ptr::eq(bytes, cached));
    }
}
