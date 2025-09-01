use crate::types::Segment;
use eyre::Result;
use std::path::PathBuf;

// Pass in path to normalised mono 16k PCM16 audio file
pub async fn run_transcription_pipeline<R: Runtime>(
    ctx: WhisperContext,
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

    let original_samples = wav::read_wav(options.path.clone().into())
        .context("failed to decode normalized WAV to PCM samples")?;

    let mut samples = vec![0.0f32; original_samples.len()];
    whisper_rs::convert_integer_to_float_audio(&original_samples, &mut samples)?;

    let mut state = ctx.create_state().context("failed to create state")?;
    let mut params = setup_params(&options);
    
    let st = std::time::Instant::now();

    // TODO: Change transcription loop to use vad or speech_segments iterator instead of full and remove special diarization progress handling

    // Only set up Whisper's internal progress callback when diarization is NOT enabled
    // When diarization is enabled, we handle progress manually in the diarization loop above
    if let Some(callback) = progress_callback {
        let mut guard = PROGRESS_CALLBACK.lock().map_err(|e| eyre!("{:?}", e))?;
        let internal_progress_callback = move |progress: i32| callback(progress);
        *guard = Some(Box::new(internal_progress_callback));
    }

    if let Some(abort_callback) = abort_callback {
        params.set_abort_callback_safe(abort_callback);
    }

    if PROGRESS_CALLBACK
        .lock()
        .map_err(|e| eyre!("{:?}", e))?
        .as_ref()
        .is_some()
    {
        params.set_progress_callback_safe(|progress| {
            if let Ok(mut cb) = PROGRESS_CALLBACK.lock() {
                if let Some(cb) = cb.as_mut() {
                    cb(progress);
                }
            }
        });
    }

    state.full(params, &samples).context("failed to transcribe")?;
    let _et = std::time::Instant::now();

    tracing::debug!("getting segments count...");
    let num_segments = state.full_n_segments();
    if num_segments == 0 {
        bail!("no segments found!")
    }
    tracing::debug!("found {} sentence segments", num_segments);

    let mut segments: Vec<Segment> = Vec::with_capacity(num_segments as usize);
    let mut empty_segments = 0usize;
    let mut total_chars = 0usize;

    for (seg_idx, seg) in state.as_iter().enumerate() { // iterator over `WhisperSegment`
        // segment text
        // If you want “Have” instead of “ have”, do a tiny sentence-case pass:
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

        total_chars += text.len();

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