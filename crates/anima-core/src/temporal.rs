/// Exponential temporal decay.
///
/// decay = exp(-lambda * age_hours)
///
/// With lambda=0.001: half-life ~693 hours (~29 days)
/// With lambda=0.01:  half-life ~69 hours (~3 days)
pub fn exponential_decay(age_hours: f64, lambda: f64) -> f64 {
    (-lambda * age_hours).exp()
}

/// Apply temporal weighting to an RRF score.
///
/// final = rrf * (1 - t_weight + t_weight * decay)
///
/// When t_weight=0: pure RRF (no temporal influence)
/// When t_weight=1: fully modulated by decay
pub fn apply_temporal_weight(rrf_score: f64, decay: f64, temporal_weight: f64) -> f64 {
    rrf_score * (1.0 - temporal_weight + temporal_weight * decay)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── exponential_decay ──────────────────────────────────

    #[test]
    fn decay_at_zero_age_is_1() {
        assert!((exponential_decay(0.0, 0.001) - 1.0).abs() < 1e-10);
        assert!((exponential_decay(0.0, 0.1) - 1.0).abs() < 1e-10);
        assert!((exponential_decay(0.0, 1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn decay_half_life_default_lambda() {
        // lambda=0.001 → half-life = ln(2)/0.001 ≈ 693 hours ≈ 29 days
        let lambda = 0.001;
        let half_life = (2.0_f64).ln() / lambda;
        let d = exponential_decay(half_life, lambda);
        assert!((d - 0.5).abs() < 1e-10,
            "at half-life ({:.1}h), decay should be 0.5, got {}", half_life, d);
    }

    #[test]
    fn decay_half_life_fast_lambda() {
        // lambda=0.01 → half-life ≈ 69.3 hours ≈ ~3 days
        let lambda = 0.01;
        let half_life = (2.0_f64).ln() / lambda;
        let d = exponential_decay(half_life, lambda);
        assert!((d - 0.5).abs() < 1e-10);
        assert!((half_life - 69.3).abs() < 0.1);
    }

    #[test]
    fn decay_monotonically_decreasing() {
        let lambda = 0.001;
        let mut prev = 1.0;
        for hours in [1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0] {
            let d = exponential_decay(hours, lambda);
            assert!(d < prev, "decay should decrease: at {}h got {} >= prev {}", hours, d, prev);
            assert!(d > 0.0, "decay should always be positive");
            prev = d;
        }
    }

    #[test]
    fn decay_very_old_approaches_zero() {
        // 10 years ≈ 87600 hours
        let d = exponential_decay(87600.0, 0.001);
        assert!(d < 0.001, "very old memory should have near-zero decay, got {}", d);
    }

    #[test]
    fn decay_1_hour_old_barely_changes() {
        let d = exponential_decay(1.0, 0.001);
        assert!(d > 0.999, "1-hour-old should barely decay, got {}", d);
    }

    // ─── apply_temporal_weight ──────────────────────────────

    #[test]
    fn temporal_weight_zero_pure_rrf() {
        // t_weight=0 → output = rrf * 1.0 = rrf
        assert!((apply_temporal_weight(0.85, 0.1, 0.0) - 0.85).abs() < 1e-10);
        assert!((apply_temporal_weight(0.5, 0.0, 0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn temporal_weight_one_full_decay() {
        // t_weight=1 → output = rrf * decay
        assert!((apply_temporal_weight(1.0, 0.5, 1.0) - 0.5).abs() < 1e-10);
        assert!((apply_temporal_weight(0.8, 0.25, 1.0) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn temporal_weight_half() {
        // t_weight=0.5 → output = rrf * (0.5 + 0.5 * decay)
        let result = apply_temporal_weight(1.0, 0.0, 0.5);
        // decay=0 → 1.0 * (0.5 + 0.5*0.0) = 0.5
        assert!((result - 0.5).abs() < 1e-10);

        let result2 = apply_temporal_weight(1.0, 1.0, 0.5);
        // decay=1 → 1.0 * (0.5 + 0.5*1.0) = 1.0
        assert!((result2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn temporal_weight_preserves_zero_rrf() {
        // If RRF is 0, temporal weight shouldn't create a score
        assert!((apply_temporal_weight(0.0, 1.0, 0.5) - 0.0).abs() < 1e-10);
        assert!((apply_temporal_weight(0.0, 0.5, 1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn temporal_weight_result_between_rrf_and_zero() {
        // For any valid inputs, result should be in [0, rrf]
        let rrf = 0.75;
        for decay in [0.0, 0.1, 0.5, 0.9, 1.0] {
            for tw in [0.0, 0.2, 0.5, 0.8, 1.0] {
                let result = apply_temporal_weight(rrf, decay, tw);
                assert!(result >= 0.0 && result <= rrf + 1e-10,
                    "result {} out of [0, {}] for decay={}, tw={}", result, rrf, decay, tw);
            }
        }
    }
}
