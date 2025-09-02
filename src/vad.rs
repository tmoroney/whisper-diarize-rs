use whisper_rs::{WhisperVadContext, WhisperVadContextParams, WhisperVadParams};
use crate::types::SpeechSegment;
use eyre::Result;

/// Detect speech segments with Silero VAD via whisper-rs.
/// `samples` must be mono f32 at 16_000 Hz in [-1.0, 1.0].
pub fn detect_speech_segments(vad_model: &str, samples: &[f32]) -> Result<Vec<SpeechSegment>> {
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

    // 5) Convert VAD centiseconds to seconds, derive clamped sample indices at 16 kHz,
    //    and collect segments with properly converted i16 audio samples.
    let n = samples.len();
    const SR: f32 = 16_000.0;
    let n_f32 = n as f32;

    let out: Vec<SpeechSegment> = segs
        .map(|s| {
            // VAD returns centiseconds; convert to seconds for API outputs
            let start_sec = (s.start as f64) / 100.0;
            let end_sec = (s.end as f64) / 100.0;

            // Derive clamped sample indices
            let start_idx = ((start_sec as f32 * SR).round()).clamp(0.0, n_f32) as usize;
            let end_idx = ((end_sec as f32 * SR).round()).clamp(0.0, n_f32) as usize;

            // Convert f32 samples in [-1,1] to i16 with clamping
            let seg_samples: Vec<i16> = if end_idx > start_idx {
                samples[start_idx..end_idx]
                    .iter()
                    .map(|&x| {
                        let v = (x * 32767.0).round();
                        v.clamp(-32768.0, 32767.0) as i16
                    })
                    .collect()
            } else {
                Vec::new()
            };

            SpeechSegment { start: start_sec, end: end_sec, samples: seg_samples }
        })
        .filter(|seg| seg.end > seg.start && !seg.samples.is_empty())
        .collect();

    Ok(out)
}