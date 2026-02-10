use std::any::TypeId;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::Mutex;

use super::snapshot;

static RENDER_SNAPSHOT_DIRTY: AtomicBool = AtomicBool::new(true);

static SNAPSHOT_REBUILD_COUNT: AtomicU64 = AtomicU64::new(0);
static SNAPSHOT_REBUILD_TOTAL_NS: AtomicU64 = AtomicU64::new(0);
static SNAPSHOT_REBUILD_LAST_NS: AtomicU64 = AtomicU64::new(0);
static SNAPSHOT_REBUILD_MAX_NS: AtomicU64 = AtomicU64::new(0);

const PROFILE_ENV: &str = "JGE_PROFILE";
const PROFILE_ENV_ENABLED_VALUE: &str = "1";

static SNAPSHOT_REBUILD_SAMPLING_ENABLED: OnceLock<bool> = OnceLock::new();
static SNAPSHOT_REBUILD_SAMPLES_NS: OnceLock<Mutex<Vec<u64>>> = OnceLock::new();

fn snapshot_sampling_enabled() -> bool {
    *SNAPSHOT_REBUILD_SAMPLING_ENABLED.get_or_init(|| {
        if !cfg!(debug_assertions) {
            return false;
        }
        std::env::var(PROFILE_ENV)
            .ok()
            .is_some_and(|v| v.trim() == PROFILE_ENV_ENABLED_VALUE)
    })
}

fn snapshot_samples() -> &'static Mutex<Vec<u64>> {
    SNAPSHOT_REBUILD_SAMPLES_NS.get_or_init(|| Mutex::new(Vec::new()))
}

pub(crate) fn take_render_snapshot_dirty() -> bool {
    RENDER_SNAPSHOT_DIRTY.swap(false, Ordering::AcqRel)
}

pub(crate) fn mark_render_snapshot_dirty_for_component<C: 'static>() {
    RENDER_SNAPSHOT_DIRTY.store(true, Ordering::Release);

    // 组件写入/替换/卸载时，若可能影响 Layer 边界或节点结构，需清理遍历缓存。
    // 这里用 TypeId 白名单保持实现简单。
    let type_id = TypeId::of::<C>();
    if snapshot::component_requires_traversal_cache_invalidation(type_id) {
        snapshot::invalidate_traversal_caches();
    }
}

pub(crate) fn record_snapshot_rebuild(duration: Duration) {
    let ns = duration.as_nanos().min(u128::from(u64::MAX)) as u64;
    SNAPSHOT_REBUILD_COUNT.fetch_add(1, Ordering::Relaxed);
    SNAPSHOT_REBUILD_TOTAL_NS.fetch_add(ns, Ordering::Relaxed);
    SNAPSHOT_REBUILD_LAST_NS.store(ns, Ordering::Relaxed);

    // 维护 max（无锁，允许竞争）。
    let mut current = SNAPSHOT_REBUILD_MAX_NS.load(Ordering::Relaxed);
    while ns > current {
        match SNAPSHOT_REBUILD_MAX_NS.compare_exchange_weak(
            current,
            ns,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(v) => current = v,
        }
    }

    // 仅在 debug + JGE_PROFILE=1 时采样，避免在正常运行时引入锁与排序开销。
    if snapshot_sampling_enabled() {
        let mut guard = snapshot_samples().lock();
        guard.push(ns);
        // 与 RenderProfiler 保持同样的上限，避免无限增长。
        const MAX_SAMPLES: usize = 5000;
        if guard.len() > MAX_SAMPLES {
            let drain = guard.len() - MAX_SAMPLES;
            guard.drain(0..drain);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SnapshotRebuildStats {
    pub(crate) count: u64,
    pub(crate) total_ns: u64,
    pub(crate) last_ns: u64,
    pub(crate) max_ns: u64,
    pub(crate) p99_ns: u64,
}

pub(crate) fn snapshot_rebuild_stats() -> SnapshotRebuildStats {
    let p99_ns = if snapshot_sampling_enabled() {
        let guard = snapshot_samples().lock();
        if guard.is_empty() {
            0
        } else {
            let mut sorted = guard.clone();
            drop(guard);
            sorted.sort_unstable();
            percentile_nearest_rank(&sorted, 0.99)
        }
    } else {
        0
    };

    SnapshotRebuildStats {
        count: SNAPSHOT_REBUILD_COUNT.load(Ordering::Relaxed),
        total_ns: SNAPSHOT_REBUILD_TOTAL_NS.load(Ordering::Relaxed),
        last_ns: SNAPSHOT_REBUILD_LAST_NS.load(Ordering::Relaxed),
        max_ns: SNAPSHOT_REBUILD_MAX_NS.load(Ordering::Relaxed),
        p99_ns,
    }
}

fn percentile_nearest_rank(sorted_ns: &[u64], p: f64) -> u64 {
    if sorted_ns.is_empty() {
        return 0;
    }
    let p = p.clamp(0.0, 1.0);
    let n = sorted_ns.len();
    let rank_1_based = (p * (n as f64)).ceil().max(1.0) as usize;
    let idx = rank_1_based.saturating_sub(1).min(n - 1);
    sorted_ns[idx]
}
