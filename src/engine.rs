use std::path::PathBuf;
use eyre::eyre;
use crate::types::{SpeechSegment, DiarizeOptions, LabeledProgressFn, NewSegmentFn};

// callback type aliases are defined in crate::types

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub cache_dir: PathBuf, // Cache directory for downloaded models
    pub use_gpu: bool, // Enable GPU acceleration
    pub gpu_device: Option<i32>, // GPU device id, default 0
    pub enable_dtw: bool, // Enable DTW for better word timestamps (works best for larger models)
    pub vad_model_path: Option<String>, // Path to Voice Activity Detection (VAD) model
    pub diarize_segment_model_path: Option<String>, // Optional path to diarization segmentation model; if None, it will be downloaded
    pub diarize_embedding_model_path: Option<String>, // Optional path to diarization embedding model; if None, it will be downloaded
}

pub struct Callbacks<'a> {
    // Unified progress callback: receives percent and a label describing the stage
    pub progress: Option<&'a LabeledProgressFn>,
    pub new_segment_callback: Option<&'a NewSegmentFn>,
    pub is_cancelled: Option<Box<dyn Fn() -> bool + Send + Sync + 'static>>,
}

impl<'a> Default for Callbacks<'a> {
    fn default() -> Self {
        Self {
            progress: None,
            new_segment_callback: None,
            is_cancelled: None,
        }
    }
}

pub struct Engine {
    cfg: EngineConfig,
    models: crate::model_manager::ModelManager,
}

impl Engine {
    pub fn new(cfg: EngineConfig) -> Self {
        Self {
            models: crate::model_manager::ModelManager::new(cfg.cache_dir.clone()),
            cfg,
        }
    }

    pub async fn transcribe_audio(
        &mut self,
        audio_path: &str,
        options: crate::TranscribeOptions,
        cb: Option<Callbacks<'_>>,
    ) -> eyre::Result<Vec<crate::Segment>> {
        let cb = cb.unwrap_or_default();
        if !std::path::PathBuf::from(audio_path).exists() {
            eyre::bail!("audio file doesn't exist")
        }

        // Ensure/download Whisper model
        let _model_path = self
            .models
            .ensure_whisper_model(&options.model, cb.progress, cb.is_cancelled.as_deref())
            .await?;

        let original_samples = crate::audio::read_wav(&audio_path)?;

        let mut speech_segments: Vec<SpeechSegment> = Vec::new();
        let mut diarize_options: Option<DiarizeOptions> = None;

        if let Some(true) = options.enable_diarize {
            let seg_url = "https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/segmentation-3.0.onnx";
            let emb_url = "https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/wespeaker_en_voxceleb_CAM++.onnx";

            // Ensure/download diarization models if not provided
            let (seg_path, emb_path) = match (&self.cfg.diarize_segment_model_path, &self.cfg.diarize_embedding_model_path) {
                (Some(seg), Some(emb)) => (PathBuf::from(seg), PathBuf::from(emb)),
                _ => self
                    .models
                    .ensure_diarize_models(seg_url, emb_url, cb.progress, cb.is_cancelled.as_deref())
                    .await?,
            };

            // Set diarize options
            let threshold = options.advanced.as_ref().and_then(|a| a.diarize_threshold).unwrap_or(0.5);
            diarize_options = Some(DiarizeOptions {
                segment_model_path: seg_path.to_string_lossy().to_string(),
                embedding_model_path: emb_path.to_string_lossy().to_string(),
                threshold,
                max_speakers: match options.max_speakers {
                    Some(0) | None => usize::MAX,
                    Some(n) => n,
                },
            });

            // Consume the lazy pyannote_rs iterator: the for-loop calls `next()` under the hood,
            // forcing evaluation as we go. Each yielded pyannote_rs::Segment is converted into
            // our SpeechSegment and appended to `speech_segments` immediately.
            let diarize_segments_iter = pyannote_rs::get_segments(&original_samples, 16000, &seg_path)
                .map_err(|e| eyre!("{:?}", e))?;
            for seg_res in diarize_segments_iter {
                let seg = seg_res.map_err(|e| eyre!("{:?}", e))?;
                speech_segments.push(SpeechSegment { start: seg.start, end: seg.end, samples: seg.samples });
            }
        } else if let Some(true) = options.enable_vad {
            // Use provided VAD model path if present; otherwise download via ModelManager
            let vad_model_path: PathBuf = if let Some(ref p) = self.cfg.vad_model_path {
                PathBuf::from(p)
            } else {
                self
                    .models
                    .ensure_vad_model(cb.progress, cb.is_cancelled.as_deref())
                    .await?
            };

            // `vad::get_segments` expects a &str path; convert from PathBuf
            let vad_model_path_str = vad_model_path.to_string_lossy().to_string();
            speech_segments = crate::vad::get_segments(&vad_model_path_str, &original_samples)
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

        println!("Transcribing {} segments", speech_segments.len());

        let ctx = crate::transcribe::create_context(
            _model_path.as_path(),
            &options.model,
            self.cfg.gpu_device,
            Some(self.cfg.use_gpu),
            Some(self.cfg.enable_dtw),
            Some(num_samples),
        )
        .map_err(|e| eyre!("Failed to create Whisper context: {}", e))?;

        // Capture translation options before moving `options` into the pipeline
        let translate_to = options.translate_target.clone();
        let from_lang = options.lang.clone().unwrap_or_else(|| "auto".to_string());
        let whisper_to_en = options.whisper_to_english.unwrap_or(false);

        let mut segments = crate::transcribe::run_transcription_pipeline(
            ctx,
            speech_segments,
            options,
            diarize_options,
            cb.progress,
            cb.new_segment_callback,
            cb.is_cancelled,
        )
        .await?;

        if !whisper_to_en {
            if let Some(to_lang) = translate_to.as_deref() {
                crate::translate::translate_segments(segments.as_mut_slice(), &from_lang, to_lang, cb.progress)
                    .await
                    .map_err(|e| eyre!("{}", e))?;
            }
        }

        Ok(segments)
    }

    pub async fn delete_whisper_model(&self, model_name: &str) -> eyre::Result<()> {
        self.models.delete_whisper_model(model_name)
    }
}