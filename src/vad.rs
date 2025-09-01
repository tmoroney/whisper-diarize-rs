use whisper_rs::{WhisperVadContext, WhisperVadContextParams, WhisperVadParams};
use eyre::Result;

/// Detect speech segments with Silero VAD via whisper-rs.
/// `samples` must be mono f32 at 16_000 Hz in [-1.0, 1.0].
pub fn detect_speech_segments(vad_model: &str, samples: &[f32]) -> Result<Vec<(usize, usize)>> {
    // 1) Configure the VAD execution context (CPU is fine; GPU here means CUDA-only).
    let ctx = WhisperVadContextParams::new();

    // 2) Create the VAD context with the Silero model path
    let mut vad = WhisperVadContext::new(vad_model, ctx)?; // segments_from_samples needs &mut self.

    // 3) Tune VAD behavior (defaults are reasonable; adjust if needed)
    let vadp = WhisperVadParams::new();
    // Examples:
    // vadp.set_threshold(0.5);
    // vadp.set_min_speech_duration(250);    // ms
    // vadp.set_min_silence_duration(100);   // ms
    // vadp.set_speech_pad(30);              // ms
    // vadp.set_samples_overlap(0.10);       // seconds of overlap between segments
    // vadp.set_max_speech_duration(f32::MAX);
    // (See docs for meanings / defaults - https://docs.rs/whisper_rs/latest/whisper_rs/struct.WhisperVadParams.html)

    // 4) Run the whole pipeline
    let segs = vad.segments_from_samples(vadp, samples)?;

    // 5) Convert centiseconds → sample indices at 16 kHz, clamp, and drop degenerate ranges
    let n = samples.len() as f32;
    const SR: f32 = 16_000.0;

    let out: Vec<(usize, usize)> = segs
        .map(|s| {
            let start = ((s.start / 100.0) * SR).round().clamp(0.0, n) as usize;
            let end   = ((s.end   / 100.0) * SR).round().clamp(0.0, n) as usize;
            (start, end)
        })
        .filter(|(a, b)| b > a)
        .collect();

    Ok(out)
}