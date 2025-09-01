pub struct ModelManager {
    cfg: EngineConfig,
}

impl ModelManager {
    pub fn new(cfg: EngineConfig) -> Self;

    pub async fn ensure_whisper_model(
        &self,
        model: &str,
        progress: Option<&super::engine::ProgressFn>,
        is_cancelled: Option<&dyn Fn() -> bool + Send + Sync>,
    ) -> eyre::Result<std::path::PathBuf> {
        // Move your robust validation, size checks, symlink target checks, retry, and
        // stale lock cleanup here. Use cfg.cache_dir to decide paths.
        todo!()
    }

    pub async fn ensure_diarize_models(
        &self,
        seg_url: &str,
        emb_url: &str,
        progress: Option<&super::engine::ProgressFn>,
        is_cancelled: Option<&dyn Fn() -> bool + Send + Sync>,
    ) -> eyre::Result<(std::path::PathBuf, std::path::PathBuf)> {
        todo!()
    }

    pub fn delete_whisper_model(&self, model: &str) -> eyre::Result<()> { todo!() }
    pub fn cleanup_stale_locks(&self) -> eyre::Result<()> { todo!() }
}