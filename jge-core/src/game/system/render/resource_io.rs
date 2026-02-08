use std::sync::Arc;

use anyhow::{Context, anyhow};

use crate::resource::ResourceHandle;

pub(in crate::game::system::render) fn load_bytes_arc(
    handle: &ResourceHandle,
    context_label: &'static str,
) -> anyhow::Result<Arc<[u8]>> {
    {
        let guard = handle.read();
        if guard.data_loaded() {
            if let Some(bytes) = guard.try_get_data_arc() {
                return Ok(bytes);
            }
            return Err(anyhow!(
                "{context_label}: resource reports cached but missing data"
            ));
        }
    }

    let mut guard = handle.write();
    Ok(guard.get_data_arc())
}

pub(in crate::game::system::render) fn load_utf8_string(
    handle: &ResourceHandle,
    context_label: &'static str,
) -> anyhow::Result<String> {
    let bytes = load_bytes_arc(handle, context_label)?;
    let source = std::str::from_utf8(bytes.as_ref())
        .with_context(|| format!("{context_label}: bytes are not valid UTF-8"))?;
    Ok(source.to_owned())
}
