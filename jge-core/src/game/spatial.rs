use nalgebra::Vector3;

use crate::Aabb3;

/// 摄像机视锥（以摄像机位姿定义）。
///
/// 坐标约定与 `Scene3D` 一致：
/// - 先把世界点转换到摄像机空间：
///   - x = dot(p - pos, right)
///   - y = dot(p - pos, up)
///   - z = dot(p - pos, forward)
/// - 视锥裁剪：
///   - near <= z <= far
///   - |x| <= z * horizontal_tan
///   - |y| <= z * vertical_tan
#[derive(Debug, Clone, Copy)]
pub(crate) struct Frustum {
    position: Vector3<f32>,
    right: Vector3<f32>,
    up: Vector3<f32>,
    forward: Vector3<f32>,
    horizontal_tan: f32,
    vertical_tan: f32,
    near: f32,
    far: f32,
}

impl Frustum {
    pub(crate) fn new(
        position: Vector3<f32>,
        right: Vector3<f32>,
        up: Vector3<f32>,
        forward: Vector3<f32>,
        horizontal_tan: f32,
        vertical_tan: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self {
            position,
            right,
            up,
            forward,
            horizontal_tan,
            vertical_tan,
            near,
            far,
        }
    }

    pub(crate) fn intersects_aabb(&self, aabb: &Aabb3) -> bool {
        // 把 aabb 的 8 个角点转换到摄像机空间后，得到一个摄像机空间的 AABB，
        // 再用与 `triangle_intersects_frustum` 同构的不等式做“保守相交测试”。
        //
        // 这里宁可多放过一些（false positive），也不要误剔除（false negative）。

        let corners = aabb_corners(aabb);

        let mut min = Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for corner in corners {
            let v = self.world_to_camera(corner);
            min.x = min.x.min(v.x);
            min.y = min.y.min(v.y);
            min.z = min.z.min(v.z);
            max.x = max.x.max(v.x);
            max.y = max.y.max(v.y);
            max.z = max.z.max(v.z);
        }

        // 任何一个平面把整个盒子“完全排除”时，才认为不相交。
        if max.z < self.near {
            return false;
        }
        if min.z > self.far {
            return false;
        }

        // Right: x <= z * tan
        // 若 box 内所有点都满足 x > z*tan，则完全在右侧外面。
        // 对线性函数 f = x - z*tan，在 box 上的最小值出现在 (min_x, max_z)。
        if min.x - max.z * self.horizontal_tan > 0.0 {
            return false;
        }

        // Left: x >= -z * tan  <=>  x + z*tan >= 0
        // 若 box 内所有点都满足 x < -z*tan，则完全在左侧外面。
        // 对 g = x + z*tan，在 box 上的最大值出现在 (max_x, max_z)。
        if max.x + max.z * self.horizontal_tan < 0.0 {
            return false;
        }

        // Top: y <= z * tan
        if min.y - max.z * self.vertical_tan > 0.0 {
            return false;
        }

        // Bottom: y >= -z * tan  <=> y + z*tan >= 0
        if max.y + max.z * self.vertical_tan < 0.0 {
            return false;
        }

        true
    }

    fn world_to_camera(&self, world: Vector3<f32>) -> Vector3<f32> {
        let offset = world - self.position;
        Vector3::new(
            offset.dot(&self.right),
            offset.dot(&self.up),
            offset.dot(&self.forward),
        )
    }
}

#[derive(Debug, Clone)]
pub(crate) struct OctreeIndex {
    root: usize,
    nodes: Vec<OctreeNode>,
}

#[derive(Debug, Clone)]
struct OctreeNode {
    bounds: Aabb3,
    children: [Option<usize>; 8],
    items: Vec<usize>,
    depth: u8,
}

impl OctreeIndex {
    pub(crate) fn build(item_bounds: &[Option<Aabb3>]) -> Option<Self> {
        const MAX_DEPTH: u8 = 8;
        const MAX_ITEMS_PER_NODE: usize = 16;

        let mut bounds: Option<Aabb3> = None;
        for b in item_bounds.iter().flatten() {
            bounds = Some(bounds.map(|acc| acc.union(b)).unwrap_or(*b));
        }
        let bounds = bounds?;

        let root_bounds = make_cube(bounds).expanded(1.0e-4);

        let mut nodes = Vec::new();
        nodes.push(OctreeNode {
            bounds: root_bounds,
            children: [None; 8],
            items: Vec::new(),
            depth: 0,
        });
        let mut index = Self { root: 0, nodes };

        for (item, b) in item_bounds.iter().enumerate() {
            let Some(b) = b else {
                continue;
            };
            if !aabb3_is_finite(b) {
                continue;
            }
            index.insert(item, b, MAX_DEPTH, MAX_ITEMS_PER_NODE);
        }

        Some(index)
    }

    pub(crate) fn query_aabb(&self, query: &Aabb3) -> Vec<usize> {
        let mut out = Vec::new();
        self.query_aabb_into(self.root, query, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    pub(crate) fn query_frustum(&self, frustum: &Frustum) -> Vec<usize> {
        let mut out = Vec::new();
        self.query_frustum_into(self.root, frustum, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    fn query_aabb_into(&self, node_index: usize, query: &Aabb3, out: &mut Vec<usize>) {
        let node = &self.nodes[node_index];
        if !node.bounds.intersects(query) {
            return;
        }

        out.extend_from_slice(&node.items);
        for child in node.children.iter().flatten() {
            self.query_aabb_into(*child, query, out);
        }
    }

    fn query_frustum_into(&self, node_index: usize, frustum: &Frustum, out: &mut Vec<usize>) {
        let node = &self.nodes[node_index];
        if !frustum.intersects_aabb(&node.bounds) {
            return;
        }

        out.extend_from_slice(&node.items);
        for child in node.children.iter().flatten() {
            self.query_frustum_into(*child, frustum, out);
        }
    }

    fn insert(&mut self, item: usize, bounds: &Aabb3, max_depth: u8, max_items: usize) {
        self.insert_into(self.root, item, bounds, max_depth, max_items);
    }

    fn insert_into(
        &mut self,
        node_index: usize,
        item: usize,
        bounds: &Aabb3,
        max_depth: u8,
        max_items: usize,
    ) {
        let node_bounds = self.nodes[node_index].bounds;
        if !node_bounds.intersects(bounds) {
            return;
        }

        let depth = self.nodes[node_index].depth;

        if depth >= max_depth {
            self.nodes[node_index].items.push(item);
            return;
        }

        // 若已有子节点，尽量下沉。
        if self.nodes[node_index].children.iter().any(|c| c.is_some()) {
            for child in self.nodes[node_index].children.iter().flatten().copied() {
                if self.nodes[child].bounds.contains_aabb(bounds) {
                    self.insert_into(child, item, bounds, max_depth, max_items);
                    return;
                }
            }

            // 放不进任何一个完整子块（跨越边界），留在当前节点。
            self.nodes[node_index].items.push(item);
            return;
        }

        // 无子节点时，先塞进本节点；超过阈值再分裂。
        self.nodes[node_index].items.push(item);
        if self.nodes[node_index].items.len() <= max_items {
            return;
        }

        self.split(node_index);
    }

    fn split(&mut self, node_index: usize) {
        let bounds = self.nodes[node_index].bounds;
        let depth = self.nodes[node_index].depth;

        let children_bounds = split_bounds(bounds);
        let mut child_indices: [Option<usize>; 8] = [None; 8];
        for (i, child_bound) in children_bounds.into_iter().enumerate() {
            let idx = self.nodes.len();
            self.nodes.push(OctreeNode {
                bounds: child_bound,
                children: [None; 8],
                items: Vec::new(),
                depth: depth + 1,
            });
            child_indices[i] = Some(idx);
        }

        let items = std::mem::take(&mut self.nodes[node_index].items);
        self.nodes[node_index].children = child_indices;

        // 注意：split 不知道每个 item 的 bounds，所以这里只做结构创建。
        // item 的重分配在 build 阶段通过 insert 的“包含子块则下沉”路径完成。
        self.nodes[node_index].items = items;
    }
}

fn aabb3_is_finite(aabb: &Aabb3) -> bool {
    [
        aabb.min.x, aabb.min.y, aabb.min.z, aabb.max.x, aabb.max.y, aabb.max.z,
    ]
    .iter()
    .all(|v| v.is_finite())
}

fn make_cube(bounds: Aabb3) -> Aabb3 {
    let center = bounds.center();
    let size = bounds.size();
    let half = 0.5 * size.x.max(size.y).max(size.z);
    let half = half.max(1.0e-3);
    Aabb3::new(
        center - Vector3::new(half, half, half),
        center + Vector3::new(half, half, half),
    )
}

fn split_bounds(bounds: Aabb3) -> [Aabb3; 8] {
    let center = bounds.center();
    let min = bounds.min;
    let max = bounds.max;

    // 八个象限的排列：bit0=x, bit1=y, bit2=z；0 取 min..center，1 取 center..max。
    let mut out = [bounds; 8];
    for i in 0..8 {
        let (x0, x1) = if (i & 1) == 0 {
            (min.x, center.x)
        } else {
            (center.x, max.x)
        };
        let (y0, y1) = if (i & 2) == 0 {
            (min.y, center.y)
        } else {
            (center.y, max.y)
        };
        let (z0, z1) = if (i & 4) == 0 {
            (min.z, center.z)
        } else {
            (center.z, max.z)
        };
        out[i] = Aabb3::new(Vector3::new(x0, y0, z0), Vector3::new(x1, y1, z1));
    }
    out
}

fn aabb_corners(aabb: &Aabb3) -> [Vector3<f32>; 8] {
    let min = aabb.min;
    let max = aabb.max;
    [
        Vector3::new(min.x, min.y, min.z),
        Vector3::new(max.x, min.y, min.z),
        Vector3::new(min.x, max.y, min.z),
        Vector3::new(max.x, max.y, min.z),
        Vector3::new(min.x, min.y, max.z),
        Vector3::new(max.x, min.y, max.z),
        Vector3::new(min.x, max.y, max.z),
        Vector3::new(max.x, max.y, max.z),
    ]
}
