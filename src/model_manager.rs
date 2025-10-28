use crate::types::{LabeledProgressFn, ProgressType};
use eyre::{bail, eyre, Context, Result};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::api::Progress as HubProgress;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;
use once_cell::sync::Lazy;

// Global download state to ensure only one download runs at a time
static ACTIVE_DOWNLOAD: Lazy<Mutex<Option<Arc<CancellationToken>>>> = Lazy::new(|| Mutex::new(None));

// Generation counter to invalidate old progress callbacks
static DOWNLOAD_GENERATION: AtomicU64 = AtomicU64::new(0);

// Internal progress adapter for hf-hub that forwards percentage to an optional callback
struct DownloadProgress<'a> {
    // percentage = offset + (current/total) * scale
    offset: f32,
    scale: f32,
    current: usize,
    total: usize,
    progress_cb: Option<&'a LabeledProgressFn>,
    label: &'a str,
    is_cancelled: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    on_cancel_cleanup: Option<Box<dyn Fn() + Send + Sync + 'a>>,
    generation: u64,
    cancel_token: Arc<CancellationToken>,
}

impl<'a> DownloadProgress<'a> {
    fn new(
        progress_cb: Option<&'a LabeledProgressFn>,
        is_cancelled: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
        offset: f32,
        scale: f32,
        on_cancel_cleanup: Option<Box<dyn Fn() + Send + Sync + 'a>>,
        label: &'a str,
        cancel_token: Arc<CancellationToken>,
    ) -> Self {
        Self {
            offset,
            scale,
            current: 0,
            total: 0,
            progress_cb,
            is_cancelled,
            on_cancel_cleanup,
            label,
            generation: DOWNLOAD_GENERATION.load(Ordering::Relaxed),
            cancel_token,
        }
    }

    /// Check if this download should stop (cancelled by user or superseded by new download)
    fn should_stop(&self) -> bool {
        // User cancelled
        if let Some(is_cancelled) = self.is_cancelled {
            if is_cancelled() {
                return true;
            }
        }
        
        // Cancelled by newer download
        if self.cancel_token.is_cancelled() {
            return true;
        }
        
        // Superseded by newer generation
        if self.generation != DOWNLOAD_GENERATION.load(Ordering::Relaxed) {
            return true;
        }
        
        false
    }

    fn emit(&self) {
        if self.should_stop() {
            return;
        }
        
        if let (Some(cb), total) = (self.progress_cb, self.total) {
            let pct = if total == 0 {
                self.offset
            } else {
                self.offset + (self.current as f32 / self.total as f32) * self.scale
            };
            cb(pct as i32, ProgressType::Download, self.label);
        }
    }
    
    /// Handle cancellation: cancel token and run cleanup
    fn handle_cancellation(&self) {
        self.cancel_token.cancel();
        if let Some(ref f) = self.on_cancel_cleanup {
            f();
        }
    }
}

impl<'a> HubProgress for DownloadProgress<'a> {
    fn init(&mut self, size: usize, _filename: &str) {
        self.total = size;
        self.current = 0;
        self.emit();
    }

    fn update(&mut self, size: usize) {
        // Check if user cancelled and handle cleanup
        if let Some(is_cancelled) = self.is_cancelled {
            if is_cancelled() {
                self.handle_cancellation();
                return;
            }
        }

        self.current += size;
        self.emit();
    }

    fn finish(&mut self) {
        // intentionally no-op; caller will manage final 100% emission if needed
    }
}

pub struct ModelManager {
    cache_dir: PathBuf,
}

impl ModelManager {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    fn model_cache_dir(&self) -> Result<PathBuf> {
        let dir = self.cache_dir.clone();
        if !dir.exists() {
            fs::create_dir_all(&dir).context("Failed to create model cache directory")?;
        }
        Ok(dir)
    }


    // ---- Public API ----
    pub async fn ensure_whisper_model(
        &self,
        model: &str,
        progress: Option<&LabeledProgressFn>,
        is_cancelled: Option<&(dyn Fn() -> bool + Send + Sync)>,
    ) -> Result<PathBuf> {
        eprintln!("ensure_whisper_model called for model: {}", model);
        
        // Early cancellation
        if let Some(is_cancelled) = is_cancelled {
            if is_cancelled() {
                self.cleanup_stale_locks().ok();
                bail!("Model download cancelled");
            }
        }

        let filename = format!("ggml-{}.bin", model);
        eprintln!("Looking for filename: {}", filename);

        // On macOS with CoreML feature, main model is 0-70%; otherwise 0-100%
        #[cfg(feature = "coreml")]
        let needs_coreml = cfg!(target_os = "macos");
        #[cfg(not(feature = "coreml"))]
        let needs_coreml = false;

        let model_path = if needs_coreml {
            // 0..70 for main model
            self.ensure_hub_model(
                "ggerganov/whisper.cpp",
                &filename,
                progress,
                is_cancelled,
                0.0,
                70.0,
                &format!("Downloading {}", model),
            )
            .await?
        } else {
            self.ensure_hub_model(
                "ggerganov/whisper.cpp",
                &filename,
                progress,
                is_cancelled,
                0.0,
                100.0,
                &format!("Downloading {}", model),
            )
            .await?
        };

        // If enabled, fetch CoreML encoder as well (zip then extract)
        #[cfg(feature = "coreml")]
        {
            if cfg!(target_os = "macos") {
                let coreml_file = format!("ggml-{}-encoder.mlmodelc.zip", model);

                // Fast path: if the extracted CoreML encoder directory already exists in cache,
                // skip downloading the zip entirely.
                let extracted_name = coreml_file.trim_end_matches(".zip");
                if let Ok(cache_root) = self.model_cache_dir() {
                    let base = cache_root
                        .join("models--ggerganov--whisper.cpp")
                        .join("snapshots");
                    if base.exists() {
                        if let Ok(entries) = fs::read_dir(&base) {
                            for entry in entries.flatten() {
                                let snap = entry.path();
                                if !snap.is_dir() { continue; }
                                let extracted_path = snap.join(extracted_name);
                                if extracted_path.exists() {
                                    // Do NOT emit progress here to avoid starting progress
                                    // when everything is already cached.
                                    return Ok(model_path);
                                }
                            }
                        }
                    }
                }

                // 70..90 for the CoreML archive download. If it fails (e.g., network), log and continue
                // with the main model instead of failing the entire operation.
                let coreml_zip_path = match self
                    .ensure_hub_model(
                        "ggerganov/whisper.cpp",
                        &coreml_file,
                        progress,
                        is_cancelled,
                        70.0,
                        20.0,
                        "Downloading CoreML encoder",
                    )
                    .await
                {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!(
                            "Warning: CoreML encoder download failed ({}). Proceeding without CoreML encoder.",
                            e
                        );
                        if let Some(cb) = progress { cb(100, ProgressType::Download, "Failed to download CoreML encoder"); }
                        return Ok(model_path);
                    }
                };

                // Progress at 90% (download done, start extracting)
                if let Some(cb) = progress { cb(90, ProgressType::Download, "Extracting CoreML encoder"); }

                // Extract to same directory as the cached zip
                let extract_dir = coreml_zip_path
                    .parent()
                    .ok_or_else(|| eyre!("Failed to get parent directory for CoreML zip"))?;
                let extracted_name = coreml_file.trim_end_matches(".zip");
                let extracted_path = extract_dir.join(extracted_name);

                if !extracted_path.exists() {
                    let file = fs::File::open(&coreml_zip_path)
                        .context("Failed to open CoreML zip")?;
                    let mut archive = zip::ZipArchive::new(file)
                        .context("Failed to read CoreML zip archive")?;

                    let total = archive.len() as u64;
                    let mut count = 0u64;
                    for i in 0..archive.len() {
                        let mut file = archive.by_index(i).context("Failed to access zip entry")?;
                        let outpath = match file.enclosed_name() {
                            Some(path) => extract_dir.join(path),
                            None => continue,
                        };
                        if (&*file.name()).ends_with('/') {
                            fs::create_dir_all(&outpath).ok();
                        } else {
                            if let Some(p) = outpath.parent() { fs::create_dir_all(p).ok(); }
                            let mut outfile = fs::File::create(&outpath)
                                .context("Failed to create extracted file")?;
                            std::io::copy(&mut file, &mut outfile)
                                .context("Failed to extract file")?;
                        }
                        count += 1;
                        if let Some(cb) = progress {
                            let pct = 90.0 + (count as f32 / total as f32) * 10.0;
                            cb(pct as i32, ProgressType::Download, "Extracting CoreML encoder");
                        }
                    }

                    // After extraction, delete the zip and its blob target (if symlinked)
                    let _ = remove_snapshot_file_and_blob(&coreml_zip_path);
                }

                // Final completion
                if let Some(cb) = progress { cb(100, ProgressType::Download, "Extracted CoreML encoder"); }
            }
        }

        Ok(model_path)
    }

    /// Ensure the Silero VAD model exists locally. If not, download via hf-hub.
    /// Uses the ggml-org/whisper-vad repository and the file `ggml-silero-v5.1.2.bin`.
    pub async fn ensure_vad_model(
        &self,
        progress: Option<&LabeledProgressFn>,
        is_cancelled: Option<&(dyn Fn() -> bool + Send + Sync)>,
    ) -> Result<PathBuf> {
        self
            .ensure_hub_model(
                "ggml-org/whisper-vad",
                "ggml-silero-v5.1.2.bin",
                progress,
                is_cancelled,
                0.0,
                100.0,
                "Downloading VAD Model",
            )
            .await
    }

    pub async fn ensure_diarize_models(
        &mut self,
        seg_url: &str,
        emb_url: &str,
        progress: Option<&LabeledProgressFn>,
        is_cancelled: Option<&(dyn Fn() -> bool + Send + Sync)>,
    ) -> Result<(PathBuf, PathBuf)> {
        if let Some(is_cancelled) = is_cancelled { if is_cancelled() { bail!("Cancelled"); } }

        let model_dir = self.model_cache_dir()?;
        let seg_name = url_filename(seg_url).ok_or_else(|| eyre!("Invalid seg_url"))?;
        let emb_name = url_filename(emb_url).ok_or_else(|| eyre!("Invalid emb_url"))?;

        let seg_path = model_dir.join(&seg_name);
        if !seg_path.exists() {
            if let Some(cb) = progress { cb(5, ProgressType::Download, "Downloading Diarize Models"); }
            download_to(&seg_path, seg_url).await?;
            if let Some(cb) = progress { cb(50, ProgressType::Download, "Downloading Diarize Models"); }
        }

        if let Some(is_cancelled) = is_cancelled { if is_cancelled() { bail!("Cancelled"); } }

        let emb_path = model_dir.join(&emb_name);
        if !emb_path.exists() {
            if let Some(cb) = progress { cb(55, ProgressType::Download, "Downloading Diarize Models"); }
            download_to(&emb_path, emb_url).await?;
            if let Some(cb) = progress { cb(100, ProgressType::Download, "Downloaded Diarize Models"); }
        }

        Ok((seg_path, emb_path))
    }

    pub fn delete_whisper_model(&self, model: &str) -> Result<()> {
        let cache_dir = self.model_cache_dir()?;
        if !cache_dir.exists() { return Ok(()); }

        let patterns = vec![
            format!("ggml-{}.bin", model),
            format!("ggml-{}-encoder.mlmodelc", model),
            format!("ggml-{}-encoder.mlmodelc.zip", model),
        ];

        let mut stack = vec![cache_dir];
        let mut deleted_any = false;
        while let Some(dir) = stack.pop() {
            for entry in fs::read_dir(&dir).context("Failed to read cache dir")? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    if patterns.iter().any(|p| p == name) {
                        let _ = remove_snapshot_file_and_blob(&path);
                        deleted_any = true;
                    }
                }
            }
        }

        if !deleted_any {
            bail!("No files found for model '{}'", model);
        }
        Ok(())
    }

    pub fn cleanup_stale_locks(&self) -> Result<()> {
        let root = self.model_cache_dir()?;
        if !root.exists() { return Ok(()); }

        let mut stack = vec![root];
        while let Some(dir) = stack.pop() {
            for entry in fs::read_dir(&dir).context("Failed to read cache dir")? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    if name.ends_with(".lock") || name.ends_with(".incomplete") || name.ends_with(".part") {
                        if let Err(e) = fs::remove_file(&path) {
                            // Log but don't fail - some files might be in use
                            eprintln!("Failed to remove {}: {}", path.display(), e);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// List all cached Whisper models in the cache directory.
    /// Returns a vector of model names (e.g., "tiny", "base", "small").
    pub fn list_cached_models(&self) -> Result<Vec<String>> {
        let cache_dir = self.model_cache_dir()?;
        if !cache_dir.exists() { return Ok(vec![]); }

        // Only look in the Whisper repository directory
        let whisper_repo_dir = cache_dir.join("models--ggerganov--whisper.cpp").join("snapshots");
        if !whisper_repo_dir.exists() { return Ok(vec![]); }

        let mut models = std::collections::HashSet::new();

        // Iterate through snapshot directories
        for snapshot_entry in fs::read_dir(&whisper_repo_dir).context("Failed to read snapshots dir")? {
            let snapshot_entry = snapshot_entry?;
            let snapshot_path = snapshot_entry.path();
            if !snapshot_path.is_dir() { continue; }

            // Look for ggml-{model}.bin files in this snapshot
            for file_entry in fs::read_dir(&snapshot_path).context("Failed to read snapshot dir")? {
                let file_entry = file_entry?;
                let path = file_entry.path();
                if path.is_file() {
                    if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                        if name.starts_with("ggml-") && name.ends_with(".bin") {
                            // Extract model name from "ggml-{model}.bin"
                            let model_part = name.strip_prefix("ggml-").unwrap_or("");
                            let model_name = model_part.strip_suffix(".bin").unwrap_or("");
                            if !model_name.is_empty() {
                                models.insert(model_name.to_string());
                            }
                        }
                    }
                }
            }
        }

        let mut result: Vec<String> = models.into_iter().collect();
        result.sort(); // Sort for consistent ordering
        Ok(result)
    }

    /// Delete a cached Whisper model by name.
    /// Returns true if successfully deleted, false if model doesn't exist or deletion failed.
    pub fn delete_cached_model(&self, model_name: &str) -> bool {
        match self.delete_whisper_model(model_name) {
            Ok(()) => true,
            Err(_) => false,
        }
    }

    /// Setup for a new download: cancel previous download and create new token
    fn setup_new_download(&self) -> Result<Arc<CancellationToken>> {
        let mut active = ACTIVE_DOWNLOAD.lock().unwrap();
        
        // Cancel and cleanup previous download if it exists
        if let Some(old_token) = active.take() {
            eprintln!("Cancelling previous download and cleaning up partial files");
            old_token.cancel();
            self.cleanup_stale_locks().ok();
        }
        
        // Create new token for this download
        let new_token = Arc::new(CancellationToken::new());
        *active = Some(new_token.clone());
        Ok(new_token)
    }

    /// Downloads a model file from HuggingFace Hub with caching and progress support.
    /// 
    /// This function ensures only one download runs at a time by:
    /// 1. Cancelling any previous download
    /// 2. Cleaning up partial files from cancelled downloads
    /// 3. Creating a new cancellation token for this download
    async fn ensure_hub_model(
        &self,
        repo_id: &str,
        filename: &str,
        progress: Option<&LabeledProgressFn>,
        is_cancelled: Option<&(dyn Fn() -> bool + Send + Sync)>,
        offset: f32,
        scale: f32,
        label: &str,
    ) -> Result<PathBuf> {
        // Setup: Cancel old download, create new token, cleanup partial files
        let cancel_token = self.setup_new_download()?;
        
        // Increment generation counter to invalidate any stale callbacks
        DOWNLOAD_GENERATION.fetch_add(1, Ordering::Relaxed);
        
        // Early cancellation
        if let Some(is_cancelled) = is_cancelled {
            if is_cancelled() {
                self.cleanup_stale_locks().ok();
                cancel_token.cancel();
                bail!("Download cancelled");
            }
        }

        // Clean up any stale locks and partial files before starting
        self.cleanup_stale_locks().ok();

        let cache_dir = self.model_cache_dir()?;

        // Fast path: if a valid cached file exists under snapshots, return it immediately to avoid
        // hitting the network. We do this conservatively and validate before returning.
        if let Some(cached) = self.find_cached_file(repo_id, filename)? {
            if validate_model_file(&cached).is_ok() {
                // Do NOT emit progress here; caller requested to only start progress
                // reporting if a download actually occurs.
                return Ok(cached);
            }
        }

        // Check cancellation again before starting download
        if let Some(is_cancelled) = is_cancelled {
            if is_cancelled() {
                self.cleanup_stale_locks().ok();
                bail!("Download cancelled before starting");
            }
        }

        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir)
            .build()
            .with_context(|| format!("Failed to build hf-hub API for repo '{}'", repo_id))?;

        let repo = api.model(repo_id.to_string());

        // Always use progress adapter; it will no-op if no callback provided
        let prog = DownloadProgress::new(
            progress,
            is_cancelled,
            offset,
            scale,
            Some(Box::new({
                let this = self;
                move || { this.cleanup_stale_locks().ok(); }
            })),
            label,
            cancel_token.clone(),
        );

        let path = repo
            .download_with_progress(filename, prog)
            .with_context(|| format!("Failed to download '{}' from '{}'", filename, repo_id))?;

        // Validate the downloaded/cached file; if invalid, remove and retry once
        if let Err(e) = validate_model_file(&path) {
            eprintln!(
                "Model file validation failed after initial retrieval ({}). Attempting one re-download...",
                e
            );
            let _ = remove_snapshot_file_and_blob(&path);
            self.cleanup_stale_locks().ok();

            let prog2 = DownloadProgress::new(progress, is_cancelled, offset, scale, None, label, cancel_token.clone());
            let path2 = repo
                .download_with_progress(filename, prog2)
                .with_context(|| format!("Failed to re-download '{}' from '{}'", filename, repo_id))?;
            validate_model_file(&path2)
                .with_context(|| format!("Model validation failed for '{}' from '{}'", filename, repo_id))?;

            if let Some(cb) = progress { cb((offset + scale) as i32, ProgressType::Download, label); }
            return Ok(path2);
        }

        if let Some(cb) = progress { cb((offset + scale) as i32, ProgressType::Download, label); }
        Ok(path)
    }

    // Attempt to locate a cached file in the hf-hub cache layout without performing any network requests.
    // Cache layout: <cache_root>/models--{owner}--{repo}/snapshots/<rev>/{filename}
    fn find_cached_file(&self, repo_id: &str, filename: &str) -> Result<Option<PathBuf>> {
        let cache_root = self.model_cache_dir()?;
        let mut parts = repo_id.splitn(2, '/');
        let owner = parts.next().unwrap_or("");
        let repo = parts.next().unwrap_or("");
        if owner.is_empty() || repo.is_empty() {
            return Ok(None);
        }
        let base = cache_root.join(format!("models--{}--{}", owner, repo)).join("snapshots");
        if !base.exists() { return Ok(None); }
        for entry in fs::read_dir(&base).context("Failed to read snapshots dir")? {
            let entry = entry?;
            let snap = entry.path();
            if !snap.is_dir() { continue; }
            let candidate = snap.join(filename);
            if candidate.exists() {
                return Ok(Some(candidate));
            }
        }
        Ok(None)
    }
}

// ---- Helpers to validate and clean cached model files ----
fn resolve_symlink_target(path: &Path) -> Result<PathBuf> {
    let metadata = fs::symlink_metadata(path).context("symlink_metadata failed")?;
    if metadata.file_type().is_symlink() {
        let target = fs::read_link(path).context("read_link failed")?;
        let base = path.parent().ok_or_else(|| eyre!("Failed to get parent directory"))?;
        Ok(if target.is_absolute() { target } else { base.join(target) })
    } else {
        Ok(path.to_path_buf())
    }
}

fn validate_model_file(path: &Path) -> Result<()> {
    let blob_path = resolve_symlink_target(path)?;
    if !blob_path.exists() {
        bail!("Model blob target does not exist: {}", blob_path.display());
    }
    let md = fs::metadata(&blob_path).context("metadata failed")?;
    // Note: Some valid models (e.g., Silero VAD) are quite small (< 1 MB). Use a conservative
    // lower bound to catch obviously truncated files while permitting small, valid models.
    const MIN_BYTES: u64 = 100_000; // 100 KB
    if md.len() < MIN_BYTES {
        bail!("Model blob seems too small ({} bytes): {}", md.len(), blob_path.display());
    }
    let mut f = fs::File::open(&blob_path).context("open failed")?;
    let mut buf = [0u8; 16];
    let _ = f.read(&mut buf).context("read failed")?;
    Ok(())
}

fn remove_snapshot_file_and_blob(path: &Path) -> Result<()> {
    if !path.exists() { return Ok(()); }
    let metadata = fs::symlink_metadata(path).context("symlink_metadata failed")?;
    if metadata.file_type().is_symlink() {
        let target_path = fs::read_link(path).context("read_link failed")?;
        let base = path.parent().ok_or_else(|| eyre!("Failed to get parent directory"))?;
        let blob_path = if target_path.is_absolute() { target_path } else { base.join(target_path) };
        if blob_path.exists() { let _ = fs::remove_file(&blob_path); }
        let _ = fs::remove_file(path);
    } else if metadata.is_dir() {
        let _ = fs::remove_dir_all(path);
    } else {
        let _ = fs::remove_file(path);
    }
    Ok(())
}

fn url_filename(url: &str) -> Option<String> {
    url.rsplit('/').next().map(|s| s.to_string())
}

async fn download_to(dest_path: &Path, url: &str) -> Result<()> {
    if let Some(parent) = dest_path.parent() { fs::create_dir_all(parent).ok(); }
    let resp = reqwest::get(url).await.context("Failed to GET url")?;
    if !resp.status().is_success() {
        bail!("Failed to download '{}': status {}", url, resp.status());
    }
    let bytes = resp.bytes().await.context("Failed to read body bytes")?;
    let mut f = fs::File::create(dest_path).context("Failed to create destination file")?;
    std::io::copy(&mut bytes.as_ref(), &mut f).context("Failed to write file")?;
    Ok(())
}