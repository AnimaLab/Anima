/// Scalar quantization: f32 → i8 for L2-normalized embeddings.
///
/// Uses symmetric scaling: `i8_value = round(f32_value * 127.0).clamp(-128, 127)`.
/// Works because L2-normalized vectors have values in [-1, 1].

/// Quantize an f32 embedding to i8 values.
pub fn quantize_embedding(embedding: &[f32]) -> Vec<i8> {
    embedding
        .iter()
        .map(|&v| (v * 127.0).round().clamp(-128.0, 127.0) as i8)
        .collect()
}

/// Quantize an f32 embedding to a byte blob for sqlite-vec int8 columns.
/// Each i8 value is cast to u8 for the blob.
pub fn embedding_to_int8_blob(embedding: &[f32]) -> Vec<u8> {
    embedding
        .iter()
        .map(|&v| (v * 127.0).round().clamp(-128.0, 127.0) as i8 as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_boundary_values() {
        let input = vec![1.0_f32, -1.0, 0.0, 0.5, -0.5];
        let quantized = quantize_embedding(&input);
        assert_eq!(quantized[0], 127);
        assert_eq!(quantized[1], -127);
        assert_eq!(quantized[2], 0);
        assert_eq!(quantized[3], 64);
        assert_eq!(quantized[4], -64);
    }

    #[test]
    fn test_quantize_clamps_out_of_range() {
        let input = vec![1.5_f32, -1.5];
        let quantized = quantize_embedding(&input);
        assert_eq!(quantized[0], 127);
        assert_eq!(quantized[1], -128);
    }

    #[test]
    fn test_int8_blob_length() {
        let input = vec![0.1_f32; 1024];
        let blob = embedding_to_int8_blob(&input);
        assert_eq!(blob.len(), 1024);
    }

    #[test]
    fn test_int8_blob_roundtrip_fidelity() {
        let dim = 1024;
        let input: Vec<f32> = (0..dim).map(|i| ((i as f32) / dim as f32) * 2.0 - 1.0).collect();
        let norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = input.iter().map(|x| x / norm).collect();

        let quantized = quantize_embedding(&normalized);
        let dequantized: Vec<f32> = quantized.iter().map(|&v| v as f32 / 127.0).collect();

        let dot: f32 = normalized.iter().zip(&dequantized).map(|(a, b)| a * b).sum();
        let norm_a: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = dequantized.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine = dot / (norm_a * norm_b);
        assert!(cosine >= 0.99, "Cosine similarity {cosine} < 0.99");
    }

    #[test]
    fn test_quantize_zero_vector() {
        let input = vec![0.0_f32; 8];
        let quantized = quantize_embedding(&input);
        assert!(quantized.iter().all(|&v| v == 0));
        let blob = embedding_to_int8_blob(&input);
        assert!(blob.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_int8_blob_byte_representation() {
        let input = vec![1.0_f32, -1.0, 0.0];
        let blob = embedding_to_int8_blob(&input);
        assert_eq!(blob[0], 127_i8 as u8);
        assert_eq!(blob[1], (-127_i8) as u8);
        assert_eq!(blob[2], 0_i8 as u8);
    }
}
