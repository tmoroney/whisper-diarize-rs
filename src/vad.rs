use whisper_rs::{WhisperVadContext, WhisperVadContextParams, WhisperVadParams};
use crate::types::SpeechSegment;
use eyre::Result;

/// Detect speech segments with Silero VAD via whisper-rs. Input `int_samples` must be mono i16 at 16_000 Hz.
pub fn get_segments(vad_model: &str, int_samples: &[i16]) -> Result<Vec<SpeechSegment>> {
    // Convert entire integer buffer to f32 for VAD processing
    let mut samples = vec![0.0f32; int_samples.len()];
    whisper_rs::convert_integer_to_float_audio(&int_samples, &mut samples)?;

    // 1) Configure the VAD execution context (CPU is fine; GPU here means CUDA-only).
    let ctx = WhisperVadContextParams::new();

    // 2) Create the VAD context with the Silero model path
    let mut vad = WhisperVadContext::new(vad_model, ctx)?; // segments_from_samples needs &mut self.

    // 3) Tune VAD behavior (defaults are reasonable; adjust if needed)
    let mut vadp = WhisperVadParams::new();
    vadp.set_min_silence_duration(200); // ms
    // vadp.set_threshold(0.5);
    // vadp.set_min_speech_duration(250);    // ms
    // vadp.set_speech_pad(30);              // ms
    // vadp.set_samples_overlap(0.10);       // seconds of overlap between segments
    // vadp.set_max_speech_duration(f32::MAX);
    // (See docs for meanings / defaults - https://docs.rs/whisper_rs/latest/whisper_rs/struct.WhisperVadParams.html)

    // 4) Run the whole pipeline
    let segs = vad.segments_from_samples(vadp, &samples)?;

    // 5) Convert VAD centiseconds to seconds, derive clamped sample indices at 16 kHz,
    //    and collect segments with integer (i16) samples sliced from the original buffer.
    let n = int_samples.len();
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

            // Extract i16 samples from the original buffer
            let seg_samples: Vec<i16> = if end_idx > start_idx {
                int_samples[start_idx..end_idx].to_vec()
            } else {
                Vec::new()
            };

            SpeechSegment { start: start_sec, end: end_sec, samples: seg_samples }
        })
        .filter(|seg| seg.end > seg.start && !seg.samples.is_empty())
        .collect();

    Ok(out)
}