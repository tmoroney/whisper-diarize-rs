// src/engine.rs
use std::path::PathBuf;

pub type ProgressFn = dyn Fn(i32) + Send + Sync;

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub cache_dir: PathBuf,
    pub use_gpu: bool,
    pub gpu_device: Option<i32>,
    pub enable_dtw: bool,
}

pub struct Callbacks<'a> {
    pub transcribe_progress: Option<&'a ProgressFn>,
    pub diarize_progress: Option<&'a ProgressFn>,
    pub is_cancelled: Option<&'a dyn Fn() -> bool + Send + Sync>,
}

pub struct Engine {
    cfg: EngineConfig,
    models: crate::model_manager::ModelManager,
}

impl Engine {
    pub fn new(cfg: EngineConfig) -> Self {
        Self { models: crate::model_manager::ModelManager::new(cfg.clone()), cfg }
    }

    pub async fn transcribe(
        &self,
        whisper_model: &str,
        opts: crate::TranscribeOptions,
        diarize: Option<crate::DiarizeOptions>,
        cb: Callbacks<'_>,
    ) -> eyre::Result<crate::Transcript> {
        // 1) Ensure/download models with self.models using cfg.cache_dir
        // 2) Create whisper context (transcribe::create_context)
        // 3) Run pipeline (transcribe::run_pipeline)
        // 4) If diarize, run diarize::process_diarization and merge
        unimplemented!()
    }

    pub async fn delete_whisper_model(&self, model_name: &str) -> eyre::Result<()> {
        unimplemented!()
    }
}