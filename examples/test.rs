use whisper_diarize_rs::{Engine, EngineConfig};

fn main() -> Result<(), eyre::Report> {
    let audio_path = std::env::args().nth(1).expect("Please specify audio file");
    let engine = Engine::new(EngineConfig {
        cache_dir: "./cache".into(),
        use_gpu: true,
        gpu_device: None,
        enable_dtw: true,
        vad_model_path: None,
        diarize_segment_model_path: None,
        diarize_embedding_model_path: None,
    });

    engine.transcribe(&audio_path, crate::TranscribeOptions::default(), crate::Callbacks::default())?;

    Ok(())
}