use crate::types::{SpeechSegment, Segment, WordTimestamp, TranscribeOptions, DiarizeOptions, LabeledProgressFn, NewSegmentFn};
use eyre::{Result, bail, WrapErr, OptionExt};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperSegment, DtwParameters, DtwMode, DtwModelPreset};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Mutex;
use eyre::eyre;
use crate::utils::{cs_to_s, calculate_dtw_mem_size};

type ProgressCallbackType = once_cell::sync::Lazy<Mutex<Option<Box<dyn Fn(i32) + Send + Sync>>>>;
static PROGRESS_CALLBACK: ProgressCallbackType = once_cell::sync::Lazy::new(|| Mutex::new(None));

// Global cancellation state
pub static SHOULD_CANCEL: once_cell::sync::Lazy<Mutex<bool>> =
    once_cell::sync::Lazy::new(|| Mutex::new(false));

// Latest progress values updated from hot compute loops (blocking threads)
// Emission to the frontend is throttled via a periodic Tokio task.

fn setup_params(options: &TranscribeOptions) -> FullParams {
    // Determine the beam size or best_of value, defaulting to 5
    let beam_size_or_best_of = options.advanced.as_ref().and_then(|a| a.best_of_or_beam_size).unwrap_or(5).max(1);

    // Decide on the sampling strategy
    let sampling_strategy = match options.advanced.as_ref().and_then(|a| a.sampling_strategy.as_deref()) {
        Some("greedy") => SamplingStrategy::Greedy {
            best_of: beam_size_or_best_of,
        },
        _ => SamplingStrategy::BeamSearch {
            beam_size: beam_size_or_best_of,
            patience: -1.0,
        },
    };
    tracing::debug!("sampling strategy: {:?}", sampling_strategy);

    // Create initial params
    let mut params = FullParams::new(sampling_strategy);

    // Basic config
    params.set_print_special(false);
    params.set_print_progress(true);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(true);
    params.set_token_timestamps(true);
    params.set_single_segment(true); // Works better for speech segments

    // Set input language
    if let Some(ref lang) = options.lang {
        params.set_language(Some(lang));
    }

    // Set translation options (Whisper built-in to English)
    if options.whisper_to_english.unwrap_or(false) {
        params.set_translate(true);
    }

    if let Some(advanced) = options.advanced.as_ref() {
        if let Some(temp) = advanced.temperature {
            params.set_temperature(temp);
        }

        // Optional temperature (only greedy sampling supports temperature > 0)
        if advanced.sampling_strategy.as_deref() == Some("greedy") {
            if let Some(temp) = advanced.temperature {
                params.set_temperature(temp);
            }
        }

        // Optional max text context
        if let Some(ctx) = advanced.max_text_ctx {
            params.set_n_max_text_ctx(ctx);
        }

        // Optional initial prompt
        if let Some(ref prompt) = advanced.init_prompt {
            params.set_initial_prompt(prompt);
        }

        // Optional thread count
        if let Some(threads) = advanced.n_threads {
            params.set_n_threads(threads);
        }
    }

    params
}

pub fn create_context(
    model_path: &Path,
    model_name: &str,
    gpu_device: Option<i32>,
    use_gpu: Option<bool>,
    enable_dtw: Option<bool>,
    enable_flash_attn: Option<bool>,
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
        println!("dtw mem size: {} MB", dtw_mem_size / 1024 / 1024);
        ctx_params.dtw_parameters(DtwParameters {
            mode: DtwMode::ModelPreset { model_preset },
            dtw_mem_size,
        });
    } else {
        // Enable flash attention if DTW is disabled (and GPU is available)
        if enable_flash_attn.unwrap_or(true) && use_gpu.unwrap_or(true) {
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

// Returns true if `s` is *only* a control marker like "[_BEG_]" or "[_TT_320]".
fn is_whole_control_token(s: &str) -> bool {
    let t = s.trim_matches('\0').trim();
    if !(t.starts_with("[_") && t.ends_with(']')) { return false; }
    // ensure inner is all A–Z / 0–9 / '_' (how whisper.cpp prints its markers)
    let inner = &t[2..t.len()-1];
    !inner.is_empty() && inner.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

// Strips embedded control markers from tokens (e.g., remove "[_TT_320]" from middle of text)
fn strip_embedded_control_markers(s: &str) -> String {
    let mut result = String::new();
    let mut i = 0;
    let chars: Vec<char> = s.chars().collect();
    while i < chars.len() {
        if i + 1 < chars.len() && chars[i] == '[' && chars[i + 1] == '_' {
            // Find the closing ']'
            let mut j = i + 2;
            while j < chars.len() && chars[j] != ']' {
                j += 1;
            }
            if j < chars.len() {
                // Check if it's a valid control marker
                let marker: String = chars[i..=j].iter().collect();
                if is_whole_control_token(&marker) {
                    // Skip this marker
                    i = j + 1;
                    continue;
                }
            }
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

fn get_word_timestamps(seg: &WhisperSegment) -> Vec<WordTimestamp> {
    #[derive(Clone)]
    struct Tok {
        text: String,
        p: f32,
        t0: f64,
        t1: f64,
        anchor: Option<f64>,
    }

    let n = seg.n_tokens() as usize;
    let mut toks: Vec<Tok> = Vec::with_capacity(n);

    for i in 0..n {
        if let Some(tok) = seg.get_token(i as i32) {
            let raw = tok.to_str_lossy().map(|c| c.into_owned()).unwrap_or_default();

            // Skip whole control tokens like "[_BEG_]" or "[_TT_320]"
            if is_whole_control_token(&raw) {
                continue;
            }

            // Remove embedded control markers that hitchhike inside printable tokens
            let clean = strip_embedded_control_markers(&raw);

            // Skip if nothing printable remains
            if clean.trim_matches('\0').trim().is_empty() {
                continue;
            }

            let td = tok.token_data();
            // Use DTW anchor only if present (>= 0). Whisper uses -1 when DTW is not computed.
            let anchor = if td.t_dtw >= 0 { Some(cs_to_s(td.t_dtw)) } else { None };

            toks.push(Tok {
                text: clean,
                p: td.p,
                t0: cs_to_s(td.t0),
                t1: cs_to_s(td.t1),
                anchor,
            });
        }
    }

    if toks.is_empty() {
        return Vec::new();
    }

    // Token bounds via DTW midpoints when anchors exist; fallback to t0/t1.
    let mut bounds = Vec::with_capacity(toks.len());
    for i in 0..toks.len() {
        let a_prev = i.checked_sub(1).and_then(|j| toks.get(j)).and_then(|t| t.anchor);
        let a_here = toks[i].anchor;
        let a_next = toks.get(i + 1).and_then(|t| t.anchor);

        let start = match (a_prev, a_here) {
            (Some(l), Some(c)) => 0.5 * (l + c),
            _ => toks[i].t0,
        };
        let end = match (a_here, a_next) {
            (Some(c), Some(r)) => 0.5 * (c + r),
            _ => toks[i].t1,
        };
        bounds.push((start, end));
    }

    // Group into words using "leading space/newline starts a new word".
    let mut words = Vec::<WordTimestamp>::new();
    let mut cur = String::new();
    let mut ps: Vec<f32> = Vec::new();
    let mut w_start = bounds[0].0;
    let mut w_end = bounds[0].1;
    let mut started = false;

    for (i, t) in toks.iter().enumerate() {
        let s = t.text.as_str();
        let new_word_boundary = s.starts_with(' ') || s.starts_with('\n');

        if new_word_boundary && started {
            let w = cur.trim();
            if !w.is_empty() {
                let p = (!ps.is_empty()).then(|| ps.iter().copied().sum::<f32>() / ps.len() as f32);
                words.push(WordTimestamp { word: w.to_string(), start: w_start, end: w_end, probability: p });
            }
            cur.clear();
            ps.clear();
            started = false;
        }

        if !started {
            w_start = bounds[i].0;
            started = true;
        }
        w_end = bounds[i].1;
        cur.push_str(s);
        ps.push(t.p);
    }

    if started {
        let w = cur.trim();
        if !w.is_empty() {
            let p = (!ps.is_empty()).then(|| ps.iter().copied().sum::<f32>() / ps.len() as f32);
            words.push(WordTimestamp { word: w.to_string(), start: w_start, end: w_end, probability: p });
        }
    }

    words
}

// Pass in path to normalised mono 16k PCM16 audio file
pub async fn run_transcription_pipeline(
    ctx: WhisperContext,
    speech_segments: Vec<SpeechSegment>,
    options: TranscribeOptions,
    diarize_options: Option<DiarizeOptions>,
    progress_callback: Option<&LabeledProgressFn>,
    new_segment_callback: Option<&NewSegmentFn>,
    abort_callback: Option<Box<dyn Fn() -> bool + Send + Sync>>,
) -> Result<Vec<Segment>> {
    tracing::debug!("Transcribe called with {:?}", options);

    // Create Whisper state
    let mut state = ctx.create_state().context("failed to create state")?;
    let mut params = setup_params(&options);

    // Initialize diarize components if diarize is enabled
    let mut embedding_manager: Option<pyannote_rs::EmbeddingManager> = None;
    let mut extractor: Option<pyannote_rs::EmbeddingExtractor> = None;
    if let Some(ref diarize_options) = diarize_options {
        embedding_manager = Some(pyannote_rs::EmbeddingManager::new(diarize_options.max_speakers));
        extractor = Some(pyannote_rs::EmbeddingExtractor::new(&diarize_options.embedding_model_path)
            .map_err(|e| eyre!("{:?}", e))?);
    }

    // DEFINE ABORT CALLBACK
    if let Some(abort_callback) = abort_callback {
        params.set_abort_callback_safe(abort_callback);
    }

    // DEFINE PROGRESS CALLBACK (no-op bridge; per-segment progress is emitted below)
    params.set_progress_callback_safe(|progress| {
        if let Ok(mut cb) = PROGRESS_CALLBACK.lock() {
            if let Some(cb) = cb.as_mut() { cb(progress); }
        }
    });

    let mut empty_segments = 0;
    let mut total_chars = 0;

    // Apply this offset directly when producing segment and word timestamps
    let user_offset = options.offset.unwrap_or(0.0);

    // List for subtitle segments
    let mut segments: Vec<Segment> = Vec::with_capacity(speech_segments.len());
    let mut previous_text: Option<String> = None;

    for (i, speech_segment) in speech_segments.iter().enumerate() {
        let original_samples = speech_segment.samples.clone();

        // Convert float samples back to integer samples for embedding
        let mut samples = vec![0.0f32; original_samples.len()];
        whisper_rs::convert_integer_to_float_audio(&original_samples, &mut samples)?;

        // Set initial prompt if available (borrow to avoid moving out of Option)
        if let Some(ref previous_text) = previous_text {
            params.set_initial_prompt(previous_text);
        }

        // Transcribe the segment
        state.full(params.clone(), &samples).context("failed to transcribe")?;
        let num_segments = state.full_n_segments();
        tracing::debug!("found {} sentence segments", num_segments);

        // Base offset for this chunk relative to the full audio timeline,
        // including any user-specified global offset
        let base_offset = speech_segment.start + user_offset;

        for seg in state.as_iter() {
            // Get the transcribed text from the state
            let mut text: String = seg.to_str().unwrap().to_string();
            text = text.trim_start().to_string(); // remove Whisper's typical leading space

            // Use the segment's start/end times (convert from centiseconds to seconds)
            // and offset by the speech segment's start to get absolute times
            let approx_start = base_offset + cs_to_s(seg.start_timestamp());
            let approx_end = base_offset + cs_to_s(seg.end_timestamp());
    
            tracing::debug!(
                "Seg approx [{:.2}-{:.2}] text_len={} text={:?}",
                approx_start, approx_end, text.len(), text
            );
    
            if text.trim().is_empty() {
                empty_segments += 1;
                tracing::warn!(
                    "Seg has empty/whitespace text in [{:.2}-{:.2}]",
                    approx_start, approx_end
                );
            }
        
            let mut word_timestamps: Vec<WordTimestamp> = Vec::new();
            if let Some(true) = options.word_timestamps {
                word_timestamps = get_word_timestamps(&seg);
            }

            // if word timestamps are empty, fall back to segment bounds
            let (seg_start, seg_end, words_opt) = if word_timestamps.is_empty() {
                tracing::debug!(
                    "Seg word_timestamps empty; falling back to segment bounds [{:.2}-{:.2}]",
                    approx_start, approx_end
                );
                (approx_start, approx_end, None)
            } else {
                // Offset word timestamps to absolute timeline
                for w in &mut word_timestamps {
                    w.start += base_offset;
                    w.end += base_offset;
                }
                let s = word_timestamps.first().map(|w| w.start).unwrap_or(approx_start);
                let e = word_timestamps.last().map(|w| w.end).unwrap_or(s);
                tracing::debug!(
                    "Seg word_timestamps count={} bounds [{:.2}-{:.2}]",
                    word_timestamps.len(), s, e
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

            // Embedding and speaker identification (speaker diarization) - if enabled
            let mut speaker_id = None;
            if num_segments > 0 && let Some(ref diarize_options) = diarize_options {
                // Compute embedding
                let extractor = extractor.as_mut().unwrap();
                let embedding_result = match extractor.compute(&original_samples) {
                    Ok(result) => Some(result.collect()),
                    Err(error) => {
                        tracing::error!("error: {:?}", error);
                        tracing::trace!(
                            "start = {:.2}, end = {:.2}, speaker = ?",
                            seg.start_timestamp(),
                            seg.end_timestamp()
                        );
                        None
                    }
                };

                // Find speaker
                let embedding_manager = embedding_manager.as_mut().unwrap();
                let speaker = if let Some(embedding_result) = embedding_result {
                    if embedding_manager.get_all_speakers().len() == diarize_options.max_speakers {
                        embedding_manager
                            .get_best_speaker_match(embedding_result)
                            .map(|r| r.to_string())
                            .unwrap_or("?".into())
                    } else {
                        embedding_manager
                            .search_speaker(embedding_result, diarize_options.threshold)
                            .map(|r| r.to_string())
                            .unwrap_or("?".into())
                    }
                } else {
                    "?".into()
                };
                speaker_id = Some(speaker);
            }

            total_chars += text.len();

            // Update previous_text before moving `text` into the Segment (or None if empty)
            previous_text = (!text.trim().is_empty()).then(|| text.clone());

            let segment = Segment {
                speaker_id,
                start: seg_start,
                end: seg_end,
                text,
                words: words_opt,
            };

            // Emit new segment to callback
            if let Some(cb) = new_segment_callback {
                cb(&segment);
            }

            // Emit progress update to callback
            if let Some(progress_callback) = progress_callback {
                tracing::trace!("progress: {} * {} / 100", i, speech_segments.len());
                let progress = ((i + 1) as f64 / speech_segments.len() as f64 * 100.0) as i32;
                progress_callback(progress, "Transcribing audio");
            }
            segments.push(segment);
        }
    }

    // TODO: Add post-processing to format segments according to formatting options

    tracing::debug!("Empty segments: {}", empty_segments);
    tracing::debug!("Total characters: {}", total_chars);
    tracing::debug!("Segments: {}", segments.len());

    // Clear progress bridge to avoid dangling references beyond this async call
    if let Ok(mut slot) = PROGRESS_CALLBACK.lock() { *slot = None; }

    return Ok(segments);
}