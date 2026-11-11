use std::path::{Path, PathBuf};

use tokio::io::AsyncWriteExt;

#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// ONNX graphs larger than this are assumed to embed their own weights and
/// are not scanned for external-data references.
const MAX_GRAPH_SCAN_BYTES: u64 = 256 * 1024 * 1024;

/// Ensure model files exist in the given directory, downloading if needed.
/// Returns paths to (model.onnx, tokenizer.json).
///
/// Large ONNX exports keep their weights in a sidecar file next to the graph
/// (`model.onnx.data` or `model.onnx_data`). When the downloaded graph
/// references one, it is fetched from the same remote directory.
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

    ensure_external_data(model_dir, &model_path, model_url).await?;

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

/// Download any external-data sidecar the local graph references and that is
/// not on disk yet. Does nothing for self-contained models.
async fn ensure_external_data(
    model_dir: &Path,
    model_path: &Path,
    model_url: &str,
) -> Result<(), DownloadError> {
    let Some((base_url, remote_name)) = split_url(model_url) else {
        return Ok(());
    };
    if tokio::fs::metadata(model_path).await?.len() > MAX_GRAPH_SCAN_BYTES {
        return Ok(());
    }
    let graph = tokio::fs::read(model_path).await?;

    for sidecar in referenced_sidecars(&graph, remote_name) {
        let dest = model_dir.join(&sidecar);
        if dest.exists() {
            continue;
        }
        tracing::info!(
            "Downloading model weights to {} (this can be several GB)",
            dest.display()
        );
        download_file(&format!("{base_url}/{sidecar}"), &dest).await?;
        tracing::info!("Model weights downloaded successfully");
    }
    Ok(())
}

/// Split a URL into (directory URL, file name), ignoring any query string.
fn split_url(url: &str) -> Option<(&str, &str)> {
    let without_query = url.split(['?', '#']).next().unwrap_or(url);
    let (base, name) = without_query.rsplit_once('/')?;
    if name.is_empty() {
        return None;
    }
    Some((base, name))
}

/// Sidecar file names (`<name>.data`, `<name>_data`) that appear in the graph
/// bytes. ONNX stores external-data locations as plain strings, so a byte
/// search is enough and avoids a protobuf dependency.
fn referenced_sidecars(graph: &[u8], remote_name: &str) -> Vec<String> {
    [format!("{remote_name}.data"), format!("{remote_name}_data")]
        .into_iter()
        .filter(|candidate| contains(graph, candidate.as_bytes()))
        .collect()
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|w| w == needle)
}

/// Stream `url` to `dest`. The body is written to a `.part` file and renamed
/// on success, so an interrupted download is never mistaken for a complete
/// file, and multi-GB models are not buffered in memory.
async fn download_file(url: &str, dest: &Path) -> Result<(), DownloadError> {
    let mut response = reqwest::get(url).await?.error_for_status()?;

    let mut part = dest.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);

    let mut file = tokio::fs::File::create(&part).await?;
    while let Some(chunk) = response.chunk().await? {
        file.write_all(&chunk).await?;
    }
    file.flush().await?;
    drop(file);

    tokio::fs::rename(&part, dest).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_url_strips_query() {
        assert_eq!(
            split_url("https://hf.co/org/repo/resolve/main/onnx/model_int8.onnx?download=true"),
            Some(("https://hf.co/org/repo/resolve/main/onnx", "model_int8.onnx"))
        );
        assert_eq!(split_url("https://hf.co/dir/"), None);
    }

    #[test]
    fn detects_referenced_sidecar() {
        let graph = b"\x08\x07location\x12\x0fmodel.onnx.data\x00weights";
        assert_eq!(referenced_sidecars(graph, "model.onnx"), vec!["model.onnx.data"]);

        let underscore = b"..location..model.onnx_data..";
        assert_eq!(referenced_sidecars(underscore, "model.onnx"), vec!["model.onnx_data"]);
    }

    #[test]
    fn self_contained_graph_has_no_sidecar() {
        assert!(referenced_sidecars(b"plain graph bytes", "model.onnx").is_empty());
    }
}
