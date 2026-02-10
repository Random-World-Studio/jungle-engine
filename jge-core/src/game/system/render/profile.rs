use std::{
    collections::HashMap,
    fs,
    path::Path,
    time::{Duration, Instant},
};

use tokio::runtime::Runtime;
use tracing::info;

use crate::game::{
    component::node::Node,
    entity::{Entity, EntityId},
};

const PROFILE_ENV: &str = "JGE_PROFILE";
const PROFILE_ENV_ENABLED_VALUE: &str = "1";
const REPORT_PATH: &str = "jge-profile.md";

// 为避免 debug profiling 下内存无限增长，保留最近 N 帧样本。
const MAX_SAMPLES_PER_ENTITY: usize = 5000;

#[derive(Debug, Default)]
pub(in crate::game::system::render) struct RenderProfiler {
    enabled: bool,
    frame_index: u64,

    // 节点名缓存：避免在生成报告时访问 ECS（异步），同时 Drop 时也能安全输出。
    node_names: HashMap<EntityId, String>,

    // 当前帧累计：一个实体在一帧内可能被渲染多次（例如多次 draw / 多个渲染阶段），这里按实体聚合。
    frame_accumulated_ns: HashMap<EntityId, u64>,

    // 历史样本：每帧一个值（ns）。
    samples_ns: HashMap<EntityId, Vec<u64>>,
}

impl RenderProfiler {
    pub(in crate::game::system::render) fn from_env() -> Self {
        // 要求：仅 debug 模式可用。
        if !cfg!(debug_assertions) {
            return Self::default();
        }

        let enabled = std::env::var(PROFILE_ENV)
            .ok()
            .is_some_and(|v| v.trim() == PROFILE_ENV_ENABLED_VALUE);

        Self {
            enabled,
            frame_index: 0,
            node_names: HashMap::new(),
            frame_accumulated_ns: HashMap::new(),
            samples_ns: HashMap::new(),
        }
    }

    pub(in crate::game::system::render) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(in crate::game::system::render) fn begin_frame(&mut self) {
        if !self.enabled {
            return;
        }
        self.frame_index = self.frame_index.wrapping_add(1);
        self.frame_accumulated_ns.clear();
    }

    pub(in crate::game::system::render) fn end_frame(&mut self) {
        if !self.enabled {
            return;
        }

        for (entity_id, ns) in self.frame_accumulated_ns.drain() {
            let entry = self.samples_ns.entry(entity_id).or_default();
            entry.push(ns);
            if entry.len() > MAX_SAMPLES_PER_ENTITY {
                // 简单截断：保留后半段（最近样本）。
                let drain_count = entry.len() - MAX_SAMPLES_PER_ENTITY;
                entry.drain(0..drain_count);
            }
        }
    }

    fn record_entity_duration(&mut self, entity: Entity, duration: Duration) {
        if !self.enabled {
            return;
        }

        let ns = duration.as_nanos().min(u128::from(u64::MAX)) as u64;
        *self.frame_accumulated_ns.entry(entity.id()).or_insert(0) += ns;
    }

    pub(in crate::game::system::render) fn entity_scope(
        &mut self,
        runtime: &Runtime,
        entity: Entity,
    ) -> EntityRenderScope {
        if self.enabled {
            let entity_id = entity.id();
            if let std::collections::hash_map::Entry::Vacant(entry) =
                self.node_names.entry(entity_id)
                && let Some(node) = runtime.block_on(entity.get_component::<Node>())
            {
                entry.insert(node.name().to_string());
            }
        }
        EntityRenderScope::new(self, entity)
    }

    fn build_report(&self) -> Report {
        let mut rows: Vec<ReportRow> = Vec::new();

        for (entity_id, samples) in &self.samples_ns {
            if samples.is_empty() {
                continue;
            }

            let mut sorted = samples.clone();
            sorted.sort_unstable();

            let count = sorted.len();
            let sum_ns: u128 = sorted.iter().map(|v| *v as u128).sum();
            let avg_ns = (sum_ns / (count as u128)) as u64;
            let min_ns = *sorted.first().unwrap();
            let max_ns = *sorted.last().unwrap();

            let p50_ns = percentile_nearest_rank(&sorted, 0.50);
            let p99_ns = percentile_nearest_rank(&sorted, 0.99);
            let p999_ns = percentile_nearest_rank(&sorted, 0.999);

            let node_name = self.node_names.get(entity_id).cloned().unwrap_or_default();

            rows.push(ReportRow {
                entity_id: *entity_id,
                node_name,
                samples: count,
                avg_ns,
                min_ns,
                max_ns,
                p50_ns,
                p99_ns,
                p999_ns,
            });
        }

        rows.sort_by_key(|r| std::cmp::Reverse(r.avg_ns));

        Report { rows }
    }

    fn write_markdown_report(&self) {
        if !self.enabled {
            return;
        }

        let report = self.build_report();
        let markdown = report.to_markdown();

        if let Err(err) = fs::write(Path::new(REPORT_PATH), markdown) {
            info!(target = "jge-profile", error = %err, "failed to write profiling report");
            return;
        }

        // 日志输出 Top 10。
        for (index, row) in report.rows.iter().take(10).enumerate() {
            info!(
                target = "jge-profile",
                rank = index + 1,
                entity_id = %row.entity_id,
                node = row.node_name,
                samples = row.samples,
                avg_ms = ns_to_ms(row.avg_ns),
                min_ms = ns_to_ms(row.min_ns),
                max_ms = ns_to_ms(row.max_ns),
                p50_ms = ns_to_ms(row.p50_ns),
                p99_ms = ns_to_ms(row.p99_ns),
                p999_ms = ns_to_ms(row.p999_ns),
                "entity render cpu profile"
            );
        }

        info!(
            target = "jge-profile",
            path = REPORT_PATH,
            "profiling report written"
        );
    }
}

impl Drop for RenderProfiler {
    fn drop(&mut self) {
        // 仅在 debug + 显式启用时输出。
        if !self.enabled {
            return;
        }
        // 尽可能把最后一帧也刷进去（如果调用者忘了 end_frame）。
        // 这里不会覆盖已有样本，只会把当前累计写入 samples。
        if !self.frame_accumulated_ns.is_empty() {
            self.end_frame();
        }
        self.write_markdown_report();
    }
}

pub(in crate::game::system::render) struct EntityRenderScope {
    profiler: *mut RenderProfiler,
    entity: Entity,
    start: Option<Instant>,
}

impl EntityRenderScope {
    fn new(profiler: &mut RenderProfiler, entity: Entity) -> Self {
        let enabled = profiler.enabled();
        Self {
            profiler: profiler as *mut RenderProfiler,
            entity,
            start: enabled.then(Instant::now),
        }
    }
}

impl Drop for EntityRenderScope {
    fn drop(&mut self) {
        let Some(start) = self.start else {
            return;
        };
        // SAFETY: scope 生命周期受调用点约束，profiler 存在于 RenderSystem 缓存中，
        // 渲染线程单线程使用；这里只做一次累加写入。
        unsafe {
            if let Some(profiler) = self.profiler.as_mut() {
                profiler.record_entity_duration(self.entity, start.elapsed());
            }
        }
    }
}

#[derive(Debug)]
struct Report {
    rows: Vec<ReportRow>,
}

#[derive(Debug)]
struct ReportRow {
    entity_id: EntityId,
    node_name: String,
    samples: usize,
    avg_ns: u64,
    min_ns: u64,
    max_ns: u64,
    p50_ns: u64,
    p99_ns: u64,
    p999_ns: u64,
}

impl Report {
    fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# JGE Render Profile\n\n");
        out.push_str("启用方式（仅 debug）：`JGE_PROFILE=1`。统计为 CPU 侧每实体每帧渲染耗时（ns 聚合）。\n\n");
        out.push_str("| Rank | EntityId | Node | Samples | Avg (ms) | Min (ms) | Max (ms) | p50 (ms) | p99 (ms) | p999 (ms) |\n");
        out.push_str("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|\n");

        for (index, row) in self.rows.iter().enumerate() {
            let node = if row.node_name.is_empty() {
                "-".to_string()
            } else {
                row.node_name.replace('|', "\\|")
            };

            out.push_str(&format!(
                "| {} | {} | {} | {} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} |\n",
                index + 1,
                row.entity_id,
                node,
                row.samples,
                ns_to_ms(row.avg_ns),
                ns_to_ms(row.min_ns),
                ns_to_ms(row.max_ns),
                ns_to_ms(row.p50_ns),
                ns_to_ms(row.p99_ns),
                ns_to_ms(row.p999_ns),
            ));
        }

        out
    }
}

fn ns_to_ms(ns: u64) -> f64 {
    (ns as f64) / 1_000_000.0
}

fn percentile_nearest_rank(sorted_ns: &[u64], p: f64) -> u64 {
    if sorted_ns.is_empty() {
        return 0;
    }
    let p = p.clamp(0.0, 1.0);
    let n = sorted_ns.len();

    // Nearest-rank: ceil(p*n) (1-based) -> convert to 0-based.
    let rank_1_based = (p * (n as f64)).ceil().max(1.0) as usize;
    let idx = rank_1_based.saturating_sub(1).min(n - 1);
    sorted_ns[idx]
}
