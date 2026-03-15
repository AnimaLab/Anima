use std::path::Path;
use anima_embed::model::{EmbeddingModel, PoolingStrategy};

fn main() {
    let model_path = Path::new("./models/bge-m3/model.onnx");
    let tokenizer_path = Path::new("./models/bge-m3/tokenizer.json");

    let model = EmbeddingModel::load(model_path, tokenizer_path, 1024, PoolingStrategy::Mean, None)
        .expect("Failed to load model");

    let texts: Vec<&str> = vec![
        "User loves cooking Italian food with Sarah",
        "They went hiking in the mountains last Friday",
        "The concert was on May 7th, 2023 at the park",
    ];

    // Single embeddings
    println!("=== Single embed() ===");
    let mut singles = Vec::new();
    for t in &texts {
        let emb = model.embed(t).expect("embed failed");
        println!("  [{}] norm={:.6} dim={}", t,
            emb.iter().map(|x| x*x).sum::<f32>().sqrt(), emb.len());
        singles.push(emb);
    }

    // Batch embedding
    println!("\n=== Batch embed_batch() ===");
    let batch = model.embed_batch(&texts).expect("embed_batch failed");
    for (i, emb) in batch.iter().enumerate() {
        println!("  [{}] norm={:.6} dim={}", texts[i],
            emb.iter().map(|x| x*x).sum::<f32>().sqrt(), emb.len());
    }

    // Compare
    println!("\n=== Cosine similarity (single vs batch) ===");
    for i in 0..texts.len() {
        let dot: f32 = singles[i].iter().zip(batch[i].iter()).map(|(a,b)| a*b).sum();
        let norm_a: f32 = singles[i].iter().map(|x| x*x).sum::<f32>().sqrt();
        let norm_b: f32 = batch[i].iter().map(|x| x*x).sum::<f32>().sqrt();
        let cos = dot / (norm_a * norm_b);
        let max_diff: f32 = singles[i].iter().zip(batch[i].iter())
            .map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        println!("  text[{}]: cosine={:.8} max_abs_diff={:.8}", i, cos, max_diff);
    }
}
