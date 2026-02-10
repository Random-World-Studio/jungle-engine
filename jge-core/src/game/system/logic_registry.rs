use std::{collections::HashMap, sync::OnceLock};

use parking_lot::RwLock;

use crate::game::{entity::EntityId, system::logic::GameLogicHandle};

static LOGICS: OnceLock<RwLock<HashMap<EntityId, GameLogicHandle>>> = OnceLock::new();

fn logics() -> &'static RwLock<HashMap<EntityId, GameLogicHandle>> {
    LOGICS.get_or_init(|| RwLock::new(HashMap::new()))
}

pub(crate) fn set(entity: EntityId, logic: GameLogicHandle) {
    logics().write().insert(entity, logic);
}

pub(crate) fn remove(entity: EntityId) {
    logics().write().remove(&entity);
}

pub(crate) fn collect_chunks(chunk_size: usize) -> Vec<Vec<(EntityId, GameLogicHandle)>> {
    let pairs: Vec<(EntityId, GameLogicHandle)> = {
        let guard = logics().read();
        guard.iter().map(|(id, h)| (*id, h.clone())).collect()
    };

    if pairs.is_empty() {
        return Vec::new();
    }

    let chunk_size = chunk_size.max(1);
    pairs
        .chunks(chunk_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}
