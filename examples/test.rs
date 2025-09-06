use whisper_diarize_rs::{Engine, EngineConfig, TranscribeOptions, Callbacks, Segment};
use eyre::Result;

#[tokio::main]
async fn main() -> Result<(), eyre::Report> {
    whisper_rs::install_logging_hooks();
    let audio_path = std::env::args().nth(1).expect("Please specify audio file");
    let mut engine = Engine::new(EngineConfig {
        cache_dir: "./cache".into(),
        use_gpu: true,
        gpu_device: None,
        enable_dtw: true,
        vad_model_path: None,
        diarize_segment_model_path: None,
        diarize_embedding_model_path: None,
    });

    let mut options = TranscribeOptions::default();
    options.model = "base.en".into();
    options.lang = Some("en".into());
    options.enable_vad = Some(true);
    options.enable_diarize = Some(true);

    // progress callback: prints integer percent updates from the transcription pipeline
    fn on_new_segment(segment: &Segment) { println!("new segment: {}", segment.text); }
    fn on_progress(p: i32) { println!("progress: {}%", p); }
    fn on_download_progress(p: i32, _label: &str) { println!("download progress: {}% for {}", p, _label); }
    let callbacks = Callbacks {
        download_progress: Some(&on_download_progress),
        transcribe_progress: Some(&on_progress),
        new_segment_callback: Some(&on_new_segment),
        is_cancelled: None,
    };

    let segments = engine
        .transcribe_audio(&audio_path, options, Some(callbacks))
        .await?;

    println!("Transcribed {} segments", segments.len());

    Ok(())
}