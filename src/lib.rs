pub mod audio;
pub mod engine;
pub mod model_manager;
pub mod transcribe;
pub mod vad;
pub mod types;

// Re-exports (crate users only need these)
pub use engine::{Engine, EngineConfig, Callbacks};
pub use vad::get_segments;
pub use types::{TranscribeOptions, Segment, WordTimestamp};
pub use model_manager::ModelManager;