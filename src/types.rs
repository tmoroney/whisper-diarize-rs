use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Default)]
pub struct AdvancedTranscribe {
    pub sampling_strategy: Option<String>, // "beam_search" or "greedy"
    pub best_of_or_beam_size: Option<i32>, // The maximum width of the beam. Higher values are better (to a point) at the cost of exponential CPU time. Defaults to 5 in whisper.cpp. Will be clamped to at least 1.
    pub n_threads: Option<i32>, // Number of threads used for decoding. Defaults to min(4, std::thread::hardware_concurrency()).
    pub temperature: Option<f32>, // Temperature for sampling. Defaults to 0.7.
    pub max_text_ctx: Option<i32>, // The maximum number of tokens to keep in the text context. Defaults to 16000.
    pub init_prompt: Option<String>, // Initial prompt for the model.
}

// TranscribeOptions references AdvancedTranscribe optionally
#[derive(Clone, Debug)]
pub struct TranscribeOptions {
    pub audio_path: String,
    pub offset: Option<f64>,
    pub model: String,
    pub lang: Option<String>,
    pub translate: Option<bool>,
    pub enable_dtw: Option<bool>,
    pub enable_diarize: Option<bool>,
    pub max_speakers: Option<usize>,
    pub vad_model_path: Option<String>,
    pub advanced: Option<AdvancedTranscribe>, // Optional knobs
}

#[derive(Clone, Debug)]
pub struct DiarizeOptions {
    pub segment_model_path: String,
    pub embedding_model_path: String,
    pub threshold: f32,
    pub max_speakers: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WordTimestamp {
    pub word: String,
    pub start: f64,
    pub end: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probability: Option<f32>,
}

// Transcribe function will return a list of segments
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<WordTimestamp>>,
}