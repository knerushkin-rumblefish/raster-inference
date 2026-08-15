//! Holds the vendored copy to bit-equality with the crate it was copied from.
//!
//! This is the risk control for the whole RoPE port. The chain computes in
//! Q16.16 and its correctness claim is that a replay reproduces the *exact*
//! bits; a rounding or quadrant-reduction difference between this copy and the
//! reference would not fail loudly, it would just produce a different token.
//! So the copy is compared against upstream directly rather than against a
//! golden file that could be regenerated from the wrong side.
//!
//! Values cross the boundary as raw bits (`to_bits` / `from_bits`), so the test
//! does not depend on both crates resolving to the same `fixed` instance.

use det_num::ops as vendored;
use det_num::types::{Acc as VAcc, Act as VAct};
use raster_inference::shared::numerics::det_num as upstream;

/// The two RoPE configurations this model actually uses, from `config.json`
/// `rope_parameters` — sliding layers rotate the full head, full-attention
/// layers only the first quarter of a double-width head.
const CONFIGS: &[(&str, usize, usize, f32)] = &[
    // (label, rotary_dim, freq_base_dim, base)
    ("sliding", 256, 256, 10_000.0),
    ("full", 128, 512, 1_000_000.0),
];

/// Deterministic pseudo-random activations; no rand dependency, and the same
/// sequence on every run so a failure is reproducible.
fn sample_bits(seed: u64, len: usize) -> Vec<i32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Keep magnitudes in a plausible activation range rather than
            // saturating everything, so rounding paths are actually exercised.
            ((state >> 24) as i32) % (8 << 16)
        })
        .collect()
}

fn check(label: &str, rotary_dim: usize, freq_base_dim: usize, base: f32, position: usize) {
    let width = freq_base_dim.max(rotary_dim);
    let bits = sample_bits((position as u64).wrapping_mul(0x9E37_79B9) ^ width as u64, width);

    let up_base = upstream::f32_to_acc(base);
    let up_in: Vec<upstream::Act> = bits.iter().map(|b| upstream::Act::from_bits(*b)).collect();
    let up_out = upstream::rope_rotate_pairs(&up_in, rotary_dim, freq_base_dim, up_base, position);

    let v_base = VAcc::from_bits(up_base.to_bits());
    let v_in: Vec<VAct> = bits.iter().map(|b| VAct::from_bits(*b)).collect();
    let v_out = vendored::rope_rotate_pairs(&v_in, rotary_dim, freq_base_dim, v_base, position);

    assert_eq!(up_out.len(), v_out.len(), "{label}: length differs");
    for (i, (u, v)) in up_out.iter().zip(v_out.iter()).enumerate() {
        assert_eq!(
            u.to_bits(),
            v.to_bits(),
            "{label}: lane {i} differs at position {position} \
             (upstream {}, vendored {})",
            u.to_bits(),
            v.to_bits()
        );
    }
}

#[test]
fn rope_matches_upstream_for_gemma_configs() {
    for (label, rotary_dim, freq_base_dim, base) in CONFIGS {
        // position 0 is the early-return path; 1 is the first real rotation;
        // the rest span a prompt and then well past one.
        for position in [0usize, 1, 2, 5, 6, 7, 63, 512, 4095] {
            check(label, *rotary_dim, *freq_base_dim, *base, position);
        }
    }
}

#[test]
fn rope_matches_upstream_across_widths() {
    // Small even widths catch off-by-one in the half-dim split that the
    // production sizes (128/256) would hide.
    for rotary_dim in [2usize, 4, 8, 16, 64] {
        for position in [1usize, 3, 17] {
            check("width", rotary_dim, rotary_dim, 10_000.0, position);
        }
    }
}

#[test]
fn integer_rope_base_encodes_as_a_plain_shift() {
    // `model-import` has to put the RoPE base into `LayerParams` as `Acc` bits,
    // but it cannot call `f32_to_acc` — that lives in the part of the upstream
    // module which needs `std` and was deliberately not vendored. Both bases
    // this model uses are exact integers, so `bits = base << 32` is the same
    // value. Asserted rather than assumed, because a silent mismatch here
    // detunes every rotation angle in the model.
    for (_, _, _, base) in CONFIGS {
        let shifted = (*base as i64) << 32;
        assert_eq!(
            upstream::f32_to_acc(*base).to_bits(),
            shifted,
            "base {base} does not encode as a plain shift"
        );
    }
}

#[test]
fn partial_rotary_leaves_the_tail_untouched() {
    // A full-attention layer rotates only `rotary_dim` of a `freq_base_dim`-wide
    // head. Everything past `rotary_dim` must come back byte-identical — that is
    // what "partial" means, and getting it wrong is silent.
    let (rotary_dim, freq_base_dim) = (128usize, 512usize);
    let bits = sample_bits(0xABCD, freq_base_dim);
    let v_in: Vec<VAct> = bits.iter().map(|b| VAct::from_bits(*b)).collect();
    let out = vendored::rope_rotate_pairs(
        &v_in,
        rotary_dim,
        freq_base_dim,
        VAcc::from_bits(upstream::f32_to_acc(1_000_000.0).to_bits()),
        7,
    );
    for i in rotary_dim..freq_base_dim {
        assert_eq!(
            out[i].to_bits(),
            bits[i],
            "lane {i} past rotary_dim was modified"
        );
    }
}
