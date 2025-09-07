pub mod audio;
pub mod engine;
pub mod model_manager;
pub mod transcribe;
pub mod vad;
pub mod types;
pub mod translate;
pub mod utils;
pub mod formatting;

// Re-exports (crate users only need these)
pub use engine::{Engine, EngineConfig, Callbacks};
pub use vad::get_segments;
pub use types::{TranscribeOptions, Segment, WordTimestamp, SubtitleCue};
pub use model_manager::ModelManager;
pub use utils::{get_translate_languages, get_whisper_languages};
pub use formatting::{PostProcessConfig, process_segments};