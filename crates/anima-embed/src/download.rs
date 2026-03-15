use std::path::{Path, PathBuf};

use tokio::io::AsyncWriteExt;

#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Ensure model files exist in the given directory, downloading if needed.
/// Returns paths to (model.onnx, tokenizer.json).
pub async fn ensure_model_files(
    model_dir: &Path,
    model_url: &str,
    tokenizer_url: &str,
) -> Result<(PathBuf, PathBuf), DownloadError> {
    tokio::fs::create_dir_all(model_dir).await?;

    let model_path = model_dir.join("model.onnx");
    let tokenizer_path = model_dir.join("tokenizer.json");

    if !model_path.exists() {
        tracing::info!("Downloading embedding model to {}", model_path.display());
        download_file(model_url, &model_path).await?;
        tracing::info!("Model downloaded successfully");
    }

    if !tokenizer_path.exists() {
        tracing::info!(
            "Downloading tokenizer to {}",
            tokenizer_path.display()
        );
        download_file(tokenizer_url, &tokenizer_path).await?;
        tracing::info!("Tokenizer downloaded successfully");
    }

    Ok((model_path, tokenizer_path))
}

async fn download_file(url: &str, dest: &Path) -> Result<(), DownloadError> {
    let response = reqwest::get(url).await?.error_for_status()?;
    let bytes = response.bytes().await?;

    let mut file = tokio::fs::File::create(dest).await?;
    file.write_all(&bytes).await?;
    file.flush().await?;

    Ok(())
}
