use crate::types::{Segment, TranscribeOptions};
use eyre::Result;
use std::path::PathBuf;
use whisper_rs::{FullParams, SamplingStrategy};

fn setup_params(options: &TranscribeOptions) -> FullParams {
    // Determine the beam size or best_of value, defaulting to 5
    let mut beam_size_or_best_of = options.sampling_bestof_or_beam_size.unwrap_or(5).max(1);

    // Decide on the sampling strategy
    let sampling_strategy = match options.sampling_strategy.as_deref() {
        Some("greedy") => SamplingStrategy::Greedy {
            best_of: beam_size_or_best_of,
        },
        _ => SamplingStrategy::BeamSearch {
            beam_size: beam_size_or_best_of,
            patience: -1.0,
        },
    };
    tracing::debug!("sampling strategy: {:?}", sampling_strategy);

    let mut params = FullParams::new(sampling_strategy);

    // Basic config
    params.set_print_special(false);
    params.set_print_progress(true);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(true);
    params.set_token_timestamps(true);

    // Set input language
    if let Some(ref lang) = options.lang {
        params.set_language(Some(lang));
    }

    // Set translation options
    if options.translate.unwrap_or(false) {
        params.set_translate(true);
    }

    // Optional temperature (only greedy sampling supports temperature > 0)
    if options.sampling_strategy.as_deref() == Some("greedy") {
        if let Some(temp) = options.temperature {
            params.set_temperature(temp);
        }
    }

    // Optional max text context
    if let Some(ctx) = options.max_text_ctx {
        params.set_n_max_text_ctx(ctx);
    }

    // Optional initial prompt
    if let Some(ref prompt) = options.init_prompt {
        params.set_initial_prompt(prompt);
    }

    // Optional thread count
    if let Some(threads) = options.n_threads {
        params.set_n_threads(threads);
    }

    params
}

/// Estimate a safe DTW working-set size (in bytes) for whisper.cpp DTW.
/// Pass the result to `DtwParameters { dtw_mem_size, .. }`.
fn calculate_dtw_mem_size(num_samples: usize) -> usize {
    // Frame geometry at 16 kHz: 10 ms per frame → 160 samples per frame
    const FRAME_SAMPLES: usize = 160;
    let num_frames = (num_samples + FRAME_SAMPLES - 1) / FRAME_SAMPLES; // ceil division

    // Memory model bits
    const BYTES_F32: usize = 4;
    const BYTES_I32: usize = 4;

    // Rolling buffers + auxiliaries (cost, prev, scratch, etc.)
    // Use 4 lanes to leave headroom on long segments/presets.
    const LANES: usize = 4;

    // Dynamic band: narrow for short audio, wider for long audio.
    // Keeps quality while bounding memory.
    let band_frames = match num_frames {
        0..=15_000 => 96,    // ≤150 s
        15_001..=45_000 => 128, // 150–450 s
        _ => 160,            // >450 s
    };

    // Core DP working set (float costs) plus an int32 backtrack-ish buffer
    let dp_bytes = num_frames
        .saturating_mul(band_frames)
        .saturating_mul(LANES)
        .saturating_mul(BYTES_F32);

    let bt_bytes = num_frames
        .saturating_mul(BYTES_I32); // rough backtrack/indices budget

    // Fixed baseline for internal scratch
    const BASELINE_MB: usize = 24;
    let base_bytes = BASELINE_MB * 1024 * 1024;

    // Total and clamps
    let total = base_bytes
        .saturating_add(dp_bytes)
        .saturating_add(bt_bytes);

    let min_bytes = 24 * 1024 * 1024;   // 24 MB floor
    let max_bytes = 768 * 1024 * 1024;  // 768 MB ceiling
    let clamped = total.clamp(min_bytes, max_bytes);

    // Align up to 8 MB so we never round *down* below requirement
    const ALIGN: usize = 8 * 1024 * 1024;
    (clamped + (ALIGN - 1)) & !(ALIGN - 1)
}

fn create_context(
    model_path: &Path,
    model_name: &str,
    gpu_device: Option<i32>,
    use_gpu: Option<bool>,
    enable_dtw: Option<bool>,
    num_samples: Option<usize>,
) -> Result<WhisperContext> {
    tracing::debug!("open model...");
    if !model_path.exists() {
        bail!("whisper file doesn't exist")
    }
    let mut ctx_params = WhisperContextParameters::default();

    // Force disable GPU if explicitly disabled
    if let Some(false) = use_gpu {
        ctx_params.use_gpu = false;
    }

    // Set GPU device if explicitly specified
    if let Some(gpu_device) = gpu_device {
        ctx_params.gpu_device = gpu_device; // GPU device id, default 0
    }

    // Set DTW parameters if enabled
    if let Some(true) = enable_dtw {
        ctx_params.flash_attn(false); // DTW requires flash_attn off
        let model_preset = match model_name {
            "tiny.en" => DtwModelPreset::TinyEn,
            "tiny" => DtwModelPreset::Tiny,
            "base.en" => DtwModelPreset::BaseEn,
            "base" => DtwModelPreset::Base,
            "small.en" => DtwModelPreset::SmallEn,
            "small" => DtwModelPreset::Small,
            "medium.en" => DtwModelPreset::MediumEn,
            "medium" => DtwModelPreset::Medium,
            "large-v3" => DtwModelPreset::LargeV3,
            "large-v3-turbo" => DtwModelPreset::LargeV3Turbo,
            _ => DtwModelPreset::Small, // Defaulting to Small
        };

        let dtw_mem_size = calculate_dtw_mem_size(num_samples.unwrap_or(0));
        ctx_params.dtw_parameters(DtwParameters {
            mode: DtwMode::ModelPreset { model_preset },
            dtw_mem_size,
        });
    } else {
        // Only enable flash attention if GPU is active and DTW is disabled
        if use_gpu == Some(true) {
            ctx_params.flash_attn(true);
        }
    }

    println!("gpu device: {:?}", ctx_params.gpu_device);
    println!("use gpu: {:?}", ctx_params.use_gpu);
    println!("DTW enabled: {}", enable_dtw.unwrap_or(false));
    println!("flash attn: {}", ctx_params.flash_attn);
    println!("num samples: {}", num_samples.unwrap_or(0));
    let model_path = model_path
        .to_str()
        .ok_or_eyre("can't convert model option to str")?;
    tracing::debug!("creating whisper context with model path {}", model_path);
    let ctx_unwind_result = catch_unwind(AssertUnwindSafe(|| {
        WhisperContext::new_with_params(model_path, ctx_params).context("failed to open model")
    }));
    match ctx_unwind_result {
        Err(error) => {
            bail!("create whisper context crash: {:?}", error)
        }
        Ok(ctx_result) => {
            let ctx = ctx_result?;
            tracing::debug!("created context successfuly");
            Ok(ctx)
        }
    }
}

// Pass in path to normalised mono 16k PCM16 audio file
pub async fn run_transcription_pipeline<R: Runtime>(
    model_path: PathBuf,
    options: TranscribeOptions,
    progress_callback: Option<Box<dyn Fn(i32) + Send + Sync>>,
    new_segment_callback: Option<Box<dyn Fn(Segment) + Send>>,
    abort_callback: Option<Box<dyn Fn() -> bool + Send>>,
    diarize_options: Option<DiarizeOptions>,
    additional_ffmpeg_args: Option<Vec<String>>,
    enable_diarize: Option<bool>,
) -> Result<Transcript> {
    tracing::debug!("Transcribe called with {:?}", options);

    if !PathBuf::from(options.path.clone()).exists() {
        bail!("audio file doesn't exist")
    }

    // Read the mono wav file
    let original_samples = audio::read_wav(options.path.clone().into())
        .context("failed to decode normalized WAV to PCM samples")?;

    // Convert to f32 samples for whisper.cpp
    let mut samples = vec![0.0f32; original_samples.len()];
    whisper_rs::convert_integer_to_float_audio(&original_samples, &mut samples)?;

    // Create whisper.cpp context
    let ctx = create_context(
        model_path.as_path(),
        &options.model, // requires model name for dtw preset
        options.gpu_device,
        options.enable_gpu,
        options.enable_dtw,
        Some(samples.len()),
    )
    .map_err(|e| format!("Failed to create Whisper context: {}", e))?;

    let mut state = ctx.create_state().context("failed to create state")?;
    let mut params = setup_params(&options);
    
    let st = std::time::Instant::now();

    // DEFINE ABORT CALLBACK
    if let Some(abort_callback) = abort_callback {
        params.set_abort_callback_safe(abort_callback);
    }

    // DEFINE PROGRESS CALLBACK
    params.set_progress_callback_safe(|progress| {
        if let Ok(mut cb) = PROGRESS_CALLBACK.lock() {
            if let Some(cb) = cb.as_mut() {
                cb(progress);
            }
        }
    });

    state.full(params, &samples).context("failed to transcribe")?;
    let _et = std::time::Instant::now();

    tracing::debug!("getting segments count...");
    let num_segments = state.full_n_segments();
    if num_segments == 0 {
        bail!("no segments found!")
    }
    tracing::debug!("found {} sentence segments", num_segments);

    // Counters for statistics
    let mut empty_segments = 0usize;
    let mut total_chars = 0usize;

    // List for subtitle segments
    let mut segments: Vec<Segment> = Vec::with_capacity(num_segments as usize);

    for (seg_idx, seg) in state.as_iter().enumerate() { // iterator over `WhisperSegment`
        let mut text = seg.to_str_lossy().map(|c| c.into_owned()).unwrap_or_default();
        text = text.trim_start().to_string(); // remove Whisper’s typical leading space

        // Filter out [BLANK_AUDIO]
        if text == "[BLANK_AUDIO]" {
            continue;
        }

        // For the very first segment (or if prev ended with .?!), capitalize:
        if seg_idx == 0 /* or your prev-ender test */ {
            text = sentence_case_first_alpha(&text);
        }

        // quick t0/t1 preview (centiseconds → seconds)
        let t0_frames = seg.start_timestamp();
        let t1_frames = seg.end_timestamp();
        let approx_start = cs_to_s(t0_frames);
        let approx_end   = cs_to_s(t1_frames);
        let preview: String = text.chars().take(40).collect();
    
        tracing::debug!(
            "Seg {} approx [{:.2}-{:.2}] text_len={} preview={:?}",
            seg_idx, approx_start, approx_end, text.len(), preview
        );
    
        if text.trim().is_empty() {
            empty_segments += 1;
            tracing::warn!(
                "Seg {} has empty/whitespace text in [{:.2}-{:.2}]",
                seg_idx, approx_start, approx_end
            );
        }
    
        // word timestamps (DTW if enabled at context creation)
        let mut word_timestamps = get_word_timestamps_from_segment(&seg, options.enable_dtw.unwrap_or(false));
        let (seg_start, seg_end, words_opt) = if word_timestamps.is_empty() {
            tracing::debug!(
                "Seg {} word_timestamps empty; falling back to segment bounds [{:.2}-{:.2}]",
                seg_idx, approx_start, approx_end
            );
            (approx_start, approx_end, None)
        } else {
            if seg_idx == 0 {
                word_timestamps.first_mut().unwrap().word = sentence_case_first_alpha(&word_timestamps.first().unwrap().word);
            }
            let s = word_timestamps.first().map(|w| w.start).unwrap_or(approx_start);
            let e = word_timestamps.last().map(|w| w.end).unwrap_or(s);
            tracing::debug!(
                "Seg {} word_timestamps count={} bounds [{:.2}-{:.2}]",
                seg_idx, word_timestamps.len(), s, e
            );
            (s, e, Some(word_timestamps))
        };
    
        // prevent slight overlaps with previous segment
        if let Some(last) = segments.last_mut() {
            if last.end > seg_start {
                last.end = seg_start;
            }
            if let Some(words) = &mut last.words {
                if let Some(last_word) = words.last_mut() {
                    if last_word.end > last.end {
                        last_word.end = last.end;
                    }
                }
            }
        }

        total_chars += text.len();
    
        segments.push(Segment {
            speaker_id: None,
            start: seg_start,
            end: seg_end,
            text,
            words: words_opt,
        });
    }

    tracing::info!(
        "Transcription summary: segments={}, empty_segments={}, total_chars={}",
        num_segments, empty_segments, total_chars
    );
    if empty_segments == num_segments as usize {
        tracing::warn!("All segments are empty/whitespace. Upstream audio or decoding may be silent/corrupted.");
    }

    let offset = options.offset.unwrap_or(0.0);

    // loop through and offset to each word and segment, then round
    for segment in segments.iter_mut() {
        segment.start = round_to_places(segment.start + offset, 3);
        segment.end = round_to_places(segment.end + offset, 3);
        if let Some(words) = &mut segment.words {
            for word in words.iter_mut() {
                word.start = round_to_places(word.start + offset, 3);
                word.end = round_to_places(word.end + offset, 3);
            }
        }
    }

    return Ok(segments);
}