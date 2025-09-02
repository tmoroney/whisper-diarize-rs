use crate::engine::{EngineConfig, ProgressFn};
use eyre::{bail, eyre, Context, Result};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::api::Progress as HubProgress;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

// Internal progress adapter for hf-hub that forwards percentage to an optional callback
struct DownloadProgress<'a> {
    // percentage = offset + (current/total) * scale
    offset: f32,
    scale: f32,
    current: usize,
    total: usize,
    progress_cb: Option<&'a ProgressFn>,
    is_cancelled: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    on_cancel_cleanup: Option<Box<dyn Fn() + Send + Sync + 'a>>, // e.g., cleanup stale locks
}

impl<'a> DownloadProgress<'a> {
    fn new(
        progress_cb: Option<&'a ProgressFn>,
        is_cancelled: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
        offset: f32,
        scale: f32,
        on_cancel_cleanup: Option<Box<dyn Fn() + Send + Sync + 'a>>,
    ) -> Self {
        Self {
            offset,
            scale,
            current: 0,
            total: 0,
            progress_cb,
            is_cancelled,
            on_cancel_cleanup,
        }
    }

    fn emit(&self) {
        if let (Some(cb), total) = (self.progress_cb, self.total) {
            let pct = if total == 0 {
                self.offset
            } else {
                self.offset + (self.current as f32 / self.total as f32) * self.scale
            };
            cb(pct as i32);
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
        if let Some(is_cancelled) = self.is_cancelled {
            if is_cancelled() {
                if let Some(ref f) = self.on_cancel_cleanup {
                    f();
                }
                // Do not emit further progress
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
    cfg: EngineConfig,
}

impl ModelManager {
    pub fn new(cfg: EngineConfig) -> Self {
        Self { cfg }
    }

    fn model_cache_dir(&self) -> Result<PathBuf> {
        let dir = self.cfg.cache_dir.clone();
        if !dir.exists() {
            fs::create_dir_all(&dir).context("Failed to create model cache directory")?;
        }
        Ok(dir)
    }

    fn snapshots_dir(&self) -> Result<PathBuf> {
        Ok(self
            .model_cache_dir()? // hf-hub cache root
            .join("models--ggerganov--whisper.cpp")
            .join("snapshots"))
    }

    fn blobs_dir(&self) -> Result<PathBuf> {
        Ok(self
            .model_cache_dir()? // hf-hub cache root
            .join("models--ggerganov--whisper.cpp")
            .join("blobs"))
    }

    // ---- Public API ----

    pub async fn ensure_whisper_model(
        &self,
        model: &str,
        progress: Option<&ProgressFn>,
        is_cancelled: Option<&(dyn Fn() -> bool + Send + Sync)>,
    ) -> Result<PathBuf> {
        // Early cancellation
        if let Some(is_cancelled) = is_cancelled {
            if is_cancelled() {
                self.cleanup_stale_locks().ok();
                bail!("Model download cancelled");
            }
        }

        let filename = format!("ggml-{}.bin", model);
        let cache_dir = self.model_cache_dir()?;
        let snapshots_dir = self.snapshots_dir()?;

        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir.clone())
            .with_token(None)
            .build()
            .context("Failed to build hf-hub API")?;
        let repo = api.model("ggerganov/whisper.cpp".to_string());

        // Try to find a valid cached file first
        let mut model_path: Option<PathBuf> = None;
        if snapshots_dir.exists() {
            for entry in fs::read_dir(&snapshots_dir).context("Failed to read snapshots dir")? {
                let entry = entry.context("Failed to read snapshots entry")?;
                let snapshot_dir = entry.path();
                if snapshot_dir.is_dir() {
                    let potential = snapshot_dir.join(&filename);
                    if potential.exists() {
                        if let Err(e) = validate_model_file(&potential) {
                            // Clean invalid symlink/blob and continue searching
                            eprintln!(
                                "Found cached model but validation failed ({}). Removing and continuing...",
                                e
                            );
                            let _ = remove_snapshot_file_and_blob(&potential);
                            continue;
                        }
                        model_path = Some(potential);
                        break;
                    }
                }
            }
        }

        let mut model_path = match model_path {
            Some(p) => p,
            None => {
                if let Some(is_cancelled) = is_cancelled {
                    if is_cancelled() {
                        self.cleanup_stale_locks().ok();
                        bail!("Model download cancelled");
                    }
                }

                // On macOS with CoreML feature, main model is 0-70%; otherwise 0-100%
                #[cfg(feature = "coreml")]
                let needs_coreml = cfg!(target_os = "macos");
                #[cfg(not(feature = "coreml"))]
                let needs_coreml = false;

                if needs_coreml {
                    let mut prog = DownloadProgress::new(
                        progress,
                        is_cancelled,
                        0.0,
                        70.0,
                        Some(Box::new({
                            let this = self;
                            move || {
                                this.cleanup_stale_locks().ok();
                            }
                        })),
                    );
                    let downloaded = repo
                        .download_with_progress(&filename, prog)
                        .context("Failed to download Whisper model")?;
                    // Smooth continuity to 70%
                    if let Some(cb) = progress { cb(70); }
                    downloaded
                } else {
                    let prog = DownloadProgress::new(progress, is_cancelled, 0.0, 100.0, None);
                    let downloaded = repo
                        .download_with_progress(&filename, prog)
                        .context("Failed to download Whisper model")?;
                    if let Some(cb) = progress { cb(100); }
                    downloaded
                }
            }
        };

        // Validate result; if invalid, clean and retry once
        if let Err(e) = validate_model_file(&model_path) {
            eprintln!(
                "Model file validation failed after initial retrieval ({}). Attempting one re-download...",
                e
            );
            let _ = remove_snapshot_file_and_blob(&model_path);
            self.cleanup_stale_locks().ok();

            let prog = DownloadProgress::new(progress, is_cancelled, 0.0, 100.0, None);
            let redownloaded = repo
                .download_with_progress(&filename, prog)
                .context("Failed to re-download Whisper model")?;
            if let Some(cb) = progress { cb(100); }

            if let Err(e2) = validate_model_file(&redownloaded) {
                bail!(eyre!(
                    "Model file appears corrupted or incomplete even after re-download: {}",
                    e2
                ));
            }
            model_path = redownloaded;
        }

        // If enabled, fetch CoreML encoder as well (zip then extract)
        #[cfg(feature = "coreml")]
        {
            if cfg!(target_os = "macos") {
                let coreml_file = format!("ggml-{}-encoder.mlmodelc.zip", model);

                // Check cache first
                let mut coreml_cached = false;
                if snapshots_dir.exists() {
                    for entry in fs::read_dir(&snapshots_dir).context("Failed to read snapshots dir")? {
                        let entry = entry?;
                        let snapshot_dir = entry.path();
                        if snapshot_dir.is_dir() {
                            let potential_zip = snapshot_dir.join(&coreml_file);
                            let extracted_name = coreml_file.trim_end_matches(".zip");
                            let potential_extracted = snapshot_dir.join(extracted_name);
                            if potential_zip.exists() && potential_extracted.exists() {
                                coreml_cached = true;
                                break;
                            }
                        }
                    }
                }

                if !coreml_cached {
                    if let Some(is_cancelled) = is_cancelled {
                        if is_cancelled() {
                            self.cleanup_stale_locks().ok();
                            bail!("Model download cancelled");
                        }
                    }

                    // Emit immediate progress at 70 to maintain continuity
                    if let Some(cb) = progress { cb(70); }

                    let prog = DownloadProgress::new(
                        progress,
                        is_cancelled,
                        70.0,
                        20.0,
                        Some(Box::new({
                            let this = self;
                            move || { this.cleanup_stale_locks().ok(); }
                        })),
                    );
                    let coreml_zip_path = repo
                        .download_with_progress(&coreml_file, prog)
                        .context("Failed to download CoreML encoder")?;

                    // Progress at 90% (download done, start extracting)
                    if let Some(cb) = progress { cb(90); }

                    // Extract to same directory
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
                                cb(pct as i32);
                            }
                        }

                        // After extraction, delete the zip (symlink target and link if needed)
                        let md = fs::symlink_metadata(&coreml_zip_path)
                            .context("Failed to stat CoreML zip")?;
                        if md.file_type().is_symlink() {
                            let target = fs::read_link(&coreml_zip_path)
                                .context("Failed to read CoreML zip symlink")?;
                            let blob = if target.is_absolute() { target } else { extract_dir.join(target) };
                            if blob.exists() { let _ = fs::remove_file(&blob); }
                            let _ = fs::remove_file(&coreml_zip_path);
                        } else {
                            let _ = fs::remove_file(&coreml_zip_path);
                        }
                    }

                    // Final completion
                    if let Some(cb) = progress { cb(100); }
                } else {
                    // Smooth transition if cached
                    if let Some(cb) = progress { cb(70); }
                    if let Some(cb) = progress { cb(90); }
                    if let Some(cb) = progress { cb(100); }
                }
            }
        }

        Ok(model_path)
    }

    pub async fn ensure_diarize_models(
        &self,
        seg_url: &str,
        emb_url: &str,
        progress: Option<&ProgressFn>,
        is_cancelled: Option<&(dyn Fn() -> bool + Send + Sync)>,
    ) -> Result<(PathBuf, PathBuf)> {
        if let Some(is_cancelled) = is_cancelled { if is_cancelled() { bail!("Cancelled"); } }

        let model_dir = self.model_cache_dir()?;
        let seg_name = url_filename(seg_url).ok_or_else(|| eyre!("Invalid seg_url"))?;
        let emb_name = url_filename(emb_url).ok_or_else(|| eyre!("Invalid emb_url"))?;

        let seg_path = model_dir.join(&seg_name);
        if !seg_path.exists() {
            if let Some(cb) = progress { cb(5); }
            download_to(&seg_path, seg_url).await?;
            if let Some(cb) = progress { cb(50); }
        }

        if let Some(is_cancelled) = is_cancelled { if is_cancelled() { bail!("Cancelled"); } }

        let emb_path = model_dir.join(&emb_name);
        if !emb_path.exists() {
            if let Some(cb) = progress { cb(55); }
            download_to(&emb_path, emb_url).await?;
            if let Some(cb) = progress { cb(100); }
        }

        Ok((seg_path, emb_path))
    }

    pub fn delete_whisper_model(&self, model: &str) -> Result<()> {
        let snapshots_dir = self.snapshots_dir()?;
        if !snapshots_dir.exists() { return Ok(()); }

        let file_patterns = vec![
            format!("ggml-{}.bin", model),
            format!("ggml-{}-encoder.mlmodelc", model),
            format!("ggml-{}-encoder.mlmodelc.zip", model),
        ];

        let mut deleted_any = false;
        for entry in fs::read_dir(&snapshots_dir).context("Failed to read snapshots dir")? {
            let entry = entry?;
            let snapshot_dir = entry.path();
            if !snapshot_dir.is_dir() { continue; }
            for pattern in &file_patterns {
                let file_path = snapshot_dir.join(pattern);
                if !file_path.exists() { continue; }
                let md = fs::symlink_metadata(&file_path).context("stat failed")?;
                if md.file_type().is_symlink() {
                    let target = fs::read_link(&file_path).context("read_link failed")?;
                    let blob = if target.is_absolute() { target } else { snapshot_dir.join(target) };
                    if blob.exists() { let _ = fs::remove_file(&blob); }
                    let _ = fs::remove_file(&file_path);
                } else if md.is_dir() {
                    let _ = fs::remove_dir_all(&file_path);
                } else {
                    let _ = fs::remove_file(&file_path);
                }
                deleted_any = true;
            }
        }

        if !deleted_any {
            bail!("No files found for model '{}'", model);
        }
        Ok(())
    }

    pub fn cleanup_stale_locks(&self) -> Result<()> {
        let blobs_dir = self.blobs_dir()?;
        if !blobs_dir.exists() { return Ok(()); }
        for entry in fs::read_dir(&blobs_dir).context("Failed to read blobs dir")? {
            let entry = entry?;
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                if name.ends_with(".lock") || name.ends_with(".incomplete") || name.ends_with(".part") {
                    let _ = fs::remove_file(&path);
                }
            }
        }
        Ok(())
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
    if md.len() < 1_000_000 {
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