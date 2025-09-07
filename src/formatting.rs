// Subtitle post-processing utilities for Whisper-style outputs (DTW+VAD already applied)
// Focus: natural line breaks, punctuation/pauses-aware grouping, CPL/CPS enforcement,
// word-edge clamping and tiny-word merging.
//
// Input types are provided by the user (WordTimestamp, Segment). We add:
// - PostProcessConfig: knobs for caps and thresholds
// - SubtitleCue: finalized two-line subtitle unit ready for rendering/exports
// - process_segments(): main entrypoint
//
// Notes:
// * We assume segments.words are in chronological order and include basic punctuation as standalone tokens or
//   attached to words (we handle both by extracting trailing punctuation).
// * If you have a frame-level VAD mask, you can plug it into `SilenceOracle` to refine clamping; otherwise we
//   rely on inter-word gaps and simple thresholds.

use serde::{Deserialize, Serialize};
use crate::types::{WordTimestamp, Segment, SubtitleCue};

/// Internal working token type used during processing.
#[derive(Clone, Debug)]
struct Tok {
    pub word: String,
    pub punc: String,
    pub start: f64,
    pub end: f64,
    pub prob: Option<f32>,
    pub speaker: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostProcessConfig {
    /// Max characters per rendered line (CPL)
    pub max_chars_per_line: usize, // e.g., 38
    /// Max lines per subtitle cue (commonly 2)
    pub max_lines: usize,          // e.g., 2
    /// Characters-per-second cap; we’ll split further if exceeded
    pub cps_cap: f64,              // e.g., 17.0
    /// If a pause between words >= this, we consider it a strong split candidate
    pub split_gap_sec: f64,        // e.g., 0.5
    /// Only allow comma-based breaks if the line would exceed this length otherwise
    pub comma_min_chars_before_allow: usize, // e.g., 55
    /// Minimum duration for a single word (merged if below)
    pub min_word_dur: f64,         // e.g., 0.10
    /// Minimum duration per subtitle cue
    pub min_sub_dur: f64,          // e.g., 1.0
    /// Maximum duration per subtitle cue
    pub max_sub_dur: f64,          // e.g., 6.0
    /// Optional soft cap on words per line (0 disables)
    pub soft_max_words_per_line: usize, // e.g., 10
}

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self {
            max_chars_per_line: 38,
            max_lines: 1,
            cps_cap: 17.0,
            split_gap_sec: 0.5,
            comma_min_chars_before_allow: 55,
            min_word_dur: 0.10,
            min_sub_dur: 1.0,
            max_sub_dur: 6.0,
            soft_max_words_per_line: 0,
        }
    }
}

/// Optional oracle to refine silence decisions; connect your VAD here if available.
pub trait SilenceOracle {
    /// Return true if [t0, t1] is considered non-speech (or mostly so)
    fn is_silence(&self, t0: f64, t1: f64) -> bool;
}

/// Dummy oracle that always returns false (no extra silence knowledge).
pub struct NoSilence;
impl SilenceOracle for NoSilence { fn is_silence(&self, _t0: f64, _t1: f64) -> bool { false } }

/// Oracle backed by a list of speech intervals (start,end in seconds).
/// `is_silence([t0,t1])` returns true if the interval does not overlap any speech range.
#[derive(Clone, Debug)]
pub struct VadMaskOracle {
    pub mask: Vec<(f64, f64)>,
}

impl VadMaskOracle {
    pub fn new(mut mask: Vec<(f64, f64)>) -> Self {
        // Ensure mask is normalized: sort and drop empty/negative ranges
        mask.retain(|(s, e)| e > s);
        mask.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Self { mask }
    }
}

impl SilenceOracle for VadMaskOracle {
    fn is_silence(&self, t0: f64, t1: f64) -> bool {
        if t1 <= t0 { return true; }
        // If any speech interval overlaps [t0, t1], it's not silence
        for (s0, s1) in &self.mask {
            if *s1 <= t0 { continue; }
            if *s0 >= t1 { break; }
            // overlap exists
            if *s1 > t0 && *s0 < t1 { return false; }
        }
        true
    }
}

/// Main entry: post-process whisper segments into readable subtitle cues.
pub fn process_segments(
    segments: &[Segment],
    cfg: &PostProcessConfig,
    oracle: Option<&dyn SilenceOracle>,
) -> Vec<SubtitleCue> {
    let oracle = oracle.unwrap_or(&NoSilence);

    // 1) Collect words from all segments, keep speaker_id continuity.
    let mut all: Vec<(Option<String>, WordTimestamp)> = Vec::new();
    for seg in segments {
        let speaker = seg.speaker_id.clone();
        if let Some(ws) = &seg.words {
            for w in ws {
                all.push((speaker.clone(), w.clone()));
            }
        } else {
            // fallback: treat the whole segment as one word if needed
            if !seg.text.trim().is_empty() {
                all.push((speaker.clone(), WordTimestamp {
                    word: seg.text.clone(), start: seg.start, end: seg.end, probability: None,
                }));
            }
        }
    }
    if all.is_empty() { return Vec::new(); }

    // 2) Normalize tokens: separate trailing punctuation for split logic.

    let mut toks: Vec<Tok> = Vec::with_capacity(all.len());
    for (speaker, w) in all.into_iter() {
        let (core, punc) = split_trailing_punct(&w.word);
        toks.push(Tok {
            word: core.to_string(),
            punc: punc.to_string(),
            start: w.start,
            end: w.end,
            prob: w.probability,
            speaker,
        });
    }

    // 3) Clamp tiny words and adjust boundaries using gaps and (optional) silence oracle.
    clamp_and_merge_tiny_words(&mut toks, cfg, oracle);

    // 4) Partition into groups by strong punctuation and long gaps.
    let groups = split_into_groups(&toks, cfg);

    // 5) For each group, create 1..N cues respecting CPL/CPS, pauses, commas.
    let mut cues: Vec<SubtitleCue> = Vec::new();
    for g in groups {
        let mut i = 0;
        while i < g.len() {
            // Grow a window that respects max duration and CPS; then split into up to max_lines.
            let (j, cue) = build_cue(&g, i, cfg);
            cues.push(cue);
            i = j;
        }
    }

    cues
}

// === Implementation details ===

fn split_trailing_punct(s: &str) -> (&str, &str) {
    let bytes = s.as_bytes();
    let mut cut = s.len();
    // Rough ASCII/Latin punctuation set; fine for most cases. Add more if you need.
    // Note: don’t strip apostrophes in contractions.
    let is_punc = |c: u8| match c as char {
        '.' | '!' | '?' | ',' | ';' | ':' | '…' | '。' | '！' | '？' | '、' | '，' | '—' | '–' | ')' | ']' | '}' | '"' => true,
        _ => false,
    };
    for (idx, &b) in bytes.iter().enumerate().rev() {
        if is_punc(b) { cut = idx; } else { break; }
    }
    if cut < s.len() { (&s[..cut], &s[cut..]) } else { (s, "") }
}

fn is_terminal_punct(p: &str) -> bool {
    matches!(p, "." | "!" | "?" | "…" | "。" | "！" | "？")
}

fn is_comma_like(p: &str) -> bool { matches!(p, "," | "，" | "、" | ";") }

fn clamp_and_merge_tiny_words(toks: &mut Vec<Tok>, cfg: &PostProcessConfig, oracle: &dyn SilenceOracle) {
    if toks.is_empty() { return; }

    // First pass: clamp boundaries against neighbors and silence.
    for i in 0..toks.len() {
        // Clamp to min duration
        let mut dur = toks[i].end - toks[i].start;
        if dur < cfg.min_word_dur {
            let grow = (cfg.min_word_dur - dur) / 2.0;
            toks[i].start -= grow;
            toks[i].end += grow;
        }
        // Avoid crossing neighbor midpoints
        if i > 0 {
            let mid = 0.5 * (toks[i - 1].end + toks[i].start);
            toks[i - 1].end = toks[i - 1].end.min(mid);
            toks[i].start = toks[i].start.max(mid);
        }
        if i + 1 < toks.len() {
            let mid = 0.5 * (toks[i].end + toks[i + 1].start);
            toks[i].end = toks[i].end.min(mid);
            toks[i + 1].start = toks[i + 1].start.max(mid);
        }
        // Snap tiny interior silences to edges if oracle says it’s silence
        let pad = 0.02;
        if oracle.is_silence(toks[i].start - pad, toks[i].start) {
            toks[i].start += pad;
        }
        if oracle.is_silence(toks[i].end, toks[i].end + pad) {
            toks[i].end -= pad;
        }
    }

    // Second pass: merge very tiny words with neighbors (prefer next)
    let mut out: Vec<Tok> = Vec::with_capacity(toks.len());
    let mut i = 0;
    while i < toks.len() {
        let dur = toks[i].end - toks[i].start;
        if dur < cfg.min_word_dur && i + 1 < toks.len() {
            // merge i into i+1
            let mut next = toks[i + 1].clone();
            let merged_word = join_tokens(&toks[i], &next);
            next.word = merged_word.0;
            next.punc = merged_word.1;
            next.start = toks[i].start.min(next.start);
            out.push(next);
            i += 2;
        } else if dur < cfg.min_word_dur && i > 0 {
            // merge into previous
            let mut prev = out.pop().unwrap();
            let merged_word = join_tokens(&prev, &toks[i]);
            prev.word = merged_word.0;
            prev.punc = merged_word.1;
            prev.end = prev.end.max(toks[i].end);
            out.push(prev);
            i += 1;
        } else {
            out.push(toks[i].clone());
            i += 1;
        }
    }
    *toks = out;
}

fn join_tokens(a: &Tok, b: &Tok) -> (String, String) {
    let mut s = String::new();
    if !a.word.is_empty() { s.push_str(&a.word); }
    if !a.punc.is_empty() { s.push_str(&a.punc); }
    if !a.word.is_empty() && !b.word.is_empty() { s.push(' '); }
    s.push_str(&b.word);
    let p = b.punc.clone();
    (s, p)
}

fn split_into_groups(toks: &[Tok], cfg: &PostProcessConfig) -> Vec<Vec<Tok>> {
    let mut groups: Vec<Vec<Tok>> = Vec::new();
    let mut cur: Vec<Tok> = Vec::new();
    for (i, t) in toks.iter().enumerate() {
        cur.push(t.clone());
        let strong_p = is_terminal_punct(t.punc.as_str());
        let long_gap = i + 1 < toks.len() && (toks[i + 1].start - t.end) >= cfg.split_gap_sec;
        if strong_p || long_gap {
            if !cur.is_empty() { groups.push(std::mem::take(&mut cur)); }
        }
    }
    if !cur.is_empty() { groups.push(cur); }
    groups
}

fn build_cue(group: &[Tok], start_idx: usize, cfg: &PostProcessConfig) -> (usize, SubtitleCue) {
    // Expand j while respecting max_sub_dur and a soft CPS cap; we’ll further split into lines later.
    let mut j = start_idx + 1;
    loop {
        let w_slice = &group[start_idx..j];
        let (t0, t1, chars) = slice_stats(w_slice);
        let dur = (t1 - t0).max(0.001);
        let cps = chars as f64 / dur;

        let next_ok = j < group.len()
            && dur < cfg.max_sub_dur
            && (cps <= cfg.cps_cap || (chars as usize) < cfg.max_chars_per_line * cfg.max_lines);
        if next_ok { j += 1; } else { break; }
    }

    let w_slice = &group[start_idx..j];
    let (t0, t1, _chars) = slice_stats(w_slice);

    // Decide line split(s)
    let lines = split_into_lines(w_slice, cfg);
    let speaker = w_slice.first().and_then(|t| t.speaker.clone());

    let words = w_slice.iter().map(|t| WordTimestamp { word: render_token(t), start: t.start, end: t.end, probability: t.prob }).collect();

    let cue = SubtitleCue { start: t0.max(0.0), end: t1, lines, words, speaker_id: speaker };
    (j, cue)
}

fn render_token(t: &Tok) -> String {
    let mut s = t.word.clone();
    s.push_str(&t.punc);
    s
}

fn slice_stats(slice: &[Tok]) -> (f64, f64, usize) {
    let t0 = slice.first().map(|t| t.start).unwrap_or(0.0);
    let t1 = slice.last().map(|t| t.end).unwrap_or(t0);
    let chars: usize = slice
        .iter()
        .map(|t| t.word.len() + t.punc.len())
        .sum::<usize>()
        + slice.len().saturating_sub(1); // spaces between words
    (t0, t1, chars)
}

fn split_into_lines(slice: &[Tok], cfg: &PostProcessConfig) -> Vec<String> {
    if slice.is_empty() { return vec![String::new()]; }
    if cfg.max_lines <= 1 { return vec![render_slice(slice)]; }

    // Prepare candidate split indices k (between words): 1..slice.len()-1
    let mut cands: Vec<usize> = Vec::new();
    for k in 1..slice.len() {
        let left = &slice[..k];
        let right = &slice[k..];
        // Prefer terminal punctuation on the left
        let left_term = slice[k - 1].punc.as_str();
        let is_term = is_terminal_punct(left_term);
        // Long pause
        let gap = right.first().unwrap().start - left.last().unwrap().end;
        let long_gap = gap >= cfg.split_gap_sec;
        // Comma allowed only if line would be long otherwise
        let comma_ok = is_comma_like(left_term)
            && slice_chars(slice) >= cfg.comma_min_chars_before_allow;
        // Always include at least a few fallback cands
        if is_term || long_gap || comma_ok || k % 2 == 0 || k == slice.len() / 2 {
            cands.push(k);
        }
    }
    if cands.is_empty() { return vec![render_slice(slice)]; }

    // Score candidates and choose best
    let mut best_k = cands[0];
    let mut best_score = f64::INFINITY;
    for &k in &cands {
        let lchars = slice_chars(&slice[..k]);
        let rchars = slice_chars(&slice[k..]);
        let ltext = render_slice(&slice[..k]);
        let rtext = render_slice(&slice[k..]);
        let lwords = k;
        let rwords = slice.len() - k;

        let len_pen = length_penalty(lchars, cfg.max_chars_per_line)
            + length_penalty(rchars, cfg.max_chars_per_line);

        // Soft word-per-line penalty, if enabled
        let word_pen = if cfg.soft_max_words_per_line > 0 {
            soft_cap_penalty(lwords, cfg.soft_max_words_per_line)
                + soft_cap_penalty(rwords, cfg.soft_max_words_per_line)
        } else { 0.0 };

        // Syntax-ish penalty: discourage splits that separate short function words from their head
        let syntax_pen = syntax_penalty(&ltext, &rtext);

        // Break quality bonus
        let left_term = slice[k - 1].punc.as_str();
        let is_term = is_terminal_punct(left_term) as i32;
        let is_comma = is_comma_like(left_term) as i32;
        let gap = slice[k].start - slice[k - 1].end;
        let long_gap = (gap >= cfg.split_gap_sec) as i32;
        let bonus = (-0.6 * is_term as f64) + (-0.3 * long_gap as f64) + (0.15 * is_comma as f64);

        let score = len_pen + word_pen + syntax_pen + bonus;
        if score < best_score { best_score = score; best_k = k; }
    }

    let left = render_slice(&slice[..best_k]);
    let right = render_slice(&slice[best_k..]);

    vec![left, right]
}

fn render_slice(slice: &[Tok]) -> String {
    let mut s = String::new();
    for (i, t) in slice.iter().enumerate() {
        if i > 0 { s.push(' '); }
        s.push_str(&t.word);
        s.push_str(&t.punc);
    }
    s
}

fn slice_chars(slice: &[Tok]) -> usize {
    slice.iter().map(|t| t.word.len() + t.punc.len()).sum::<usize>() + slice.len().saturating_sub(1)
}

fn length_penalty(chars: usize, cap: usize) -> f64 {
    if chars <= cap { 0.0 } else { let d = (chars - cap) as f64; 0.02 * d * d }
}

fn soft_cap_penalty(v: usize, cap: usize) -> f64 {
    if v <= cap { 0.0 } else { let d = (v - cap) as f64; 0.01 * d * d }
}

fn syntax_penalty(left: &str, right: &str) -> f64 {
    // Very lightweight heuristics: penalize if right starts with a short function word
    // or if left ends with a short function word ("I", "to", "a", etc.).
    // This helps avoid splits like "I think I | would like to".
    const SHORT_FUNCT: &[&str] = &[
        "i", "to", "a", "the", "and", "or", "of", "in", "on", "for", "with", "at",
    ];
    let starts_bad = right.split_whitespace().next()
        .map(|w| SHORT_FUNCT.contains(&w.to_lowercase().as_str()))
        .unwrap_or(false);
    let ends_bad = left.split_whitespace().last()
        .map(|w| SHORT_FUNCT.contains(&w.to_lowercase().as_str()))
        .unwrap_or(false);
    let mut pen = 0.0;
    if starts_bad { pen += 0.3; }
    if ends_bad { pen += 0.25; }
    pen
}

// --- Example usage (remove or adapt in your app) ---
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_split() {
        let cfg = PostProcessConfig::default();
        let words = vec![
            Tok { word: "I".into(), punc: "".into(), start: 0.00, end: 0.10, prob: None, speaker: None },
            Tok { word: "think".into(), punc: "".into(), start: 0.10, end: 0.38, prob: None, speaker: None },
            Tok { word: "I".into(), punc: "".into(), start: 0.50, end: 0.60, prob: None, speaker: None },
            Tok { word: "would".into(), punc: "".into(), start: 0.60, end: 0.80, prob: None, speaker: None },
            Tok { word: "like".into(), punc: "".into(), start: 0.80, end: 0.95, prob: None, speaker: None },
            Tok { word: "to".into(), punc: ".".into(), start: 0.95, end: 1.10, prob: None, speaker: None },
        ];

        // Build a pseudo segment and run
        let seg = Segment { start: 0.0, end: 1.1, text: String::new(), speaker_id: None, words: Some(words.iter().map(|t| WordTimestamp{word: format!("{}{}", t.word, t.punc), start: t.start, end: t.end, probability: None}).collect()) };
        let cues = process_segments(&[seg], &cfg, None);
        assert!(!cues.is_empty());
        // Expect two lines split as "I think" | "I would like to."
        let lines = &cues[0].lines;
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("I think"));
    }
}
