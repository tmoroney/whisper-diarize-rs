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
    pub is_cancelled: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
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
        options: crate::TranscribeOptions,
        cb: Callbacks<'_>,
    ) -> eyre::Result<Vec<crate::Segment>> {
        // Ensure/download Whisper model
        let _model_path = self
            .models
            .ensure_whisper_model(whisper_model, cb.transcribe_progress, cb.is_cancelled)
            .await?;

        // If diarization is requested, ensure diarization models
        if let Some(true) = options.enable_diarize {
            let seg_url = "https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/segmentation-3.0.onnx";
            let emb_url = "https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/wespeaker_en_voxceleb_CAM++.onnx";
            let _ = self
                .models
                .ensure_diarize_models(seg_url, emb_url, cb.diarize_progress, cb.is_cancelled)
                .await?;
        }

        // TODO: wire up actual transcription/diarization pipeline
        Ok(Vec::new())
    }

    pub async fn delete_whisper_model(&self, model_name: &str) -> eyre::Result<()> {
        self.models.delete_whisper_model(model_name)
    }
}