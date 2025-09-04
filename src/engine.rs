use std::path::PathBuf;
use eyre::eyre;
use crate::audio;
use whisper_rs;
use crate::transcribe::create_context;
use crate::types::SpeechSegment;
use crate::types::DiarizeOptions;

pub type ProgressFn = dyn Fn(i32) + Send + Sync;

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub cache_dir: PathBuf, // Cache directory for downloaded models
    pub use_gpu: bool, // Enable GPU acceleration
    pub gpu_device: Option<i32>, // GPU device id, default 0
    pub enable_dtw: bool, // Enable DTW for better word timestamps (works best for larger models)
    pub vad_model_path: Option<String>, // Path to Voice Activity Detection (VAD) model
}

pub struct Callbacks<'a> {
    pub transcribe_progress: Option<&'a ProgressFn>,
    pub diarize_progress: Option<&'a ProgressFn>,
    pub is_cancelled: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
}

pub struct Engine {
    cfg: EngineConfig,
    models: crate::model_manager::ModelManager,
    diarize_segment_model_path: Option<PathBuf>,
    diarize_embedding_model_path: Option<PathBuf>,
}

impl Engine {
    pub fn new(cfg: EngineConfig) -> Self {
        Self { 
            models: crate::model_manager::ModelManager::new(cfg.clone()), 
            cfg,
            diarize_segment_model_path: None,
            diarize_embedding_model_path: None,
        }
    }

    pub async fn transcribe(
        &mut self,
        audio_path: &str,
        options: crate::TranscribeOptions,
        cb: Callbacks<'_>,
    ) -> eyre::Result<Vec<crate::Segment>> {
        if !std::path::PathBuf::from(audio_path.clone()).exists() {
            eyre::bail!("audio file doesn't exist")
        }

        // Ensure/download Whisper model
        let _model_path = self
            .models
            .ensure_whisper_model(&options.model, cb.transcribe_progress, cb.is_cancelled)
            .await?;

        let original_samples = crate::audio::read_wav(&audio_path)?;

        let mut speech_segments: Vec<SpeechSegment> = Vec::new();
        let mut num_samples: usize = 0;
        let mut diarize_options: Option<DiarizeOptions> = None;

        if let Some(true) = options.enable_diarize {
            let seg_url = "https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/segmentation-3.0.onnx";
            let emb_url = "https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/wespeaker_en_voxceleb_CAM++.onnx";
            let (seg_path, emb_path) = self
                .models
                .ensure_diarize_models(seg_url, emb_url, cb.diarize_progress, cb.is_cancelled)
                .await?;
            self.diarize_segment_model_path = Some(seg_path);
            self.diarize_embedding_model_path = Some(emb_path);

            diarize_options = Some(DiarizeOptions {
                segment_model_path: self.diarize_segment_model_path.clone().unwrap().to_string_lossy().to_string(),
                embedding_model_path: self.diarize_embedding_model_path.clone().unwrap().to_string_lossy().to_string(),
                threshold: options.advanced.as_ref().unwrap().diarize_threshold.unwrap_or(0.5),
                max_speakers: options.max_speakers.unwrap_or(2),
            });

            // Get speech segments as an iterator and collect them all
            let diarize_segments_iter =
                pyannote_rs::get_segments(&original_samples, 16000, &self.diarize_segment_model_path.as_ref().unwrap()).map_err(|e| eyre!("{:?}", e))?;
            let mut diarize_segments: Vec<pyannote_rs::Segment> = Vec::new();
            for segment_result in diarize_segments_iter {
                let segment = segment_result.map_err(|e| eyre!("{:?}", e))?;
                diarize_segments.push(segment);
            }

            // Convert integer samples to float + cast to SpeechSegment
            for segment in diarize_segments.iter_mut() {
                speech_segments.push(SpeechSegment {
                    start: segment.start,
                    end: segment.end,
                    samples: segment.samples.clone(),
                });
            }
        } else if let Some(true) = options.enable_vad && let Some(ref vad_model_path) = self.cfg.vad_model_path {
            speech_segments = crate::vad::get_segments(&vad_model_path, &original_samples)
                .map_err(|e| eyre!("{:?}", e))?;
        }
        else {
            speech_segments = vec![SpeechSegment {
                start: 0.0,
                end: original_samples.len() as f64 / 16000.0,
                samples: original_samples.clone(),
            }];
        }

        let num_samples = speech_segments.iter().map(|s| s.samples.len()).sum();

        let ctx = crate::transcribe::create_context(
            _model_path.as_path(),
            &options.model,
            self.cfg.gpu_device,
            Some(self.cfg.use_gpu),
            Some(self.cfg.enable_dtw),
            Some(num_samples),
        )
        .map_err(|e| eyre!("Failed to create Whisper context: {}", e))?;

        crate::transcribe::run_transcription_pipeline(
            ctx,
            speech_segments,
            options,
            diarize_options,
            None,
            None,
            None,
        )
        .await
    }

    pub async fn delete_whisper_model(&self, model_name: &str) -> eyre::Result<()> {
        self.models.delete_whisper_model(model_name)
    }
}