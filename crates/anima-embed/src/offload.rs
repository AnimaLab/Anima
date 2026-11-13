//! Keep CPU-bound inference and blocking API calls off the async workers.
//!
//! The server calls the synchronous `Embedder` and `Reranker` methods straight
//! from async handlers and processor tasks. ONNX inference takes tens to
//! hundreds of milliseconds, and a remote call waits on the network; meanwhile
//! the tokio worker running it can serve nothing else. With a few concurrent
//! searches (one per prompt from each connected harness) every worker is busy
//! and the server stops answering, `/health` included.
//!
//! `tokio::task::block_in_place` hands the worker's other tasks to another
//! thread for the duration of the call, without changing any call site.

use crate::model::EmbedError;
use crate::{Embedder, SparseVector};

/// Run `f`, telling tokio first when it would otherwise block a runtime worker.
///
/// `block_in_place` exists only on the multi-thread runtime and panics on the
/// current-thread one, so elsewhere `f` simply runs.
pub fn offload_blocking<T>(f: impl FnOnce() -> T) -> T {
    match tokio::runtime::Handle::try_current() {
        Ok(h) if h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(f)
        }
        _ => f(),
    }
}

/// An `Embedder` whose every call goes through [`offload_blocking`].
pub struct OffloadingEmbedder<E>(pub E);

impl<E: Embedder> Embedder for OffloadingEmbedder<E> {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        offload_blocking(|| self.0.embed(text))
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        offload_blocking(|| self.0.embed_query(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        offload_blocking(|| self.0.embed_batch(texts))
    }

    fn dimension(&self) -> usize {
        self.0.dimension()
    }

    fn embed_with_sparse(&self, text: &str) -> Result<(Vec<f32>, SparseVector), EmbedError> {
        offload_blocking(|| self.0.embed_with_sparse(text))
    }

    fn embed_query_with_sparse(&self, text: &str) -> Result<(Vec<f32>, SparseVector), EmbedError> {
        offload_blocking(|| self.0.embed_query_with_sparse(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Stands in for ONNX inference: holds its thread for a while.
    struct SlowEmbedder(Duration);

    impl Embedder for SlowEmbedder {
        fn embed(&self, _: &str) -> Result<Vec<f32>, EmbedError> {
            std::thread::sleep(self.0);
            Ok(vec![0.0])
        }
        fn embed_query(&self, t: &str) -> Result<Vec<f32>, EmbedError> {
            self.embed(t)
        }
        fn dimension(&self) -> usize {
            1
        }
    }

    /// Two workers, two slow embeddings in flight: a third, trivial task must
    /// still run promptly. Without the offload both workers are held and the
    /// trivial task waits for an embedding to finish.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn slow_inference_does_not_starve_other_tasks() {
        let embedder = std::sync::Arc::new(OffloadingEmbedder(SlowEmbedder(Duration::from_millis(800))));
        // The barrier holds each task on its worker until both have started,
        // so the two embeddings are guaranteed to run on both workers at once.
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let busy: Vec<_> = (0..2)
            .map(|_| {
                let (e, b) = (embedder.clone(), barrier.clone());
                tokio::spawn(async move {
                    b.wait();
                    e.embed("x")
                })
            })
            .collect();
        // Not tokio::time::sleep: its timer is driven by the very workers this
        // test holds, so it would only fire once the embeddings were done. The
        // test body runs on its own thread, outside the worker pool.
        std::thread::sleep(Duration::from_millis(200));

        let start = Instant::now();
        tokio::spawn(async {}).await.unwrap();
        let waited = start.elapsed();

        for b in busy {
            b.await.unwrap().unwrap();
        }
        assert!(waited < Duration::from_millis(400), "trivial task waited {waited:?}");
    }

    #[test]
    fn runs_outside_any_runtime() {
        assert_eq!(offload_blocking(|| 3), 3);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runs_on_the_current_thread_runtime() {
        assert_eq!(offload_blocking(|| 4), 4);
    }
}
