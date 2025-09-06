use whisper_diarize_rs::{Engine, EngineConfig, TranscribeOptions};

#[tokio::main]
async fn main() -> Result<(), eyre::Report> {
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

    let segments = engine
        .transcribe(&audio_path, TranscribeOptions::default(), None)
        .await?;

    println!("Transcribed {} segments", segments.len());

    // print all segments
    for segment in segments {
        println!("{}", segment.text);
    }

    Ok(())
}