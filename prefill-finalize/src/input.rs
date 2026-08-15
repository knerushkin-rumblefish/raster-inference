//! Shared Rastered data types and deterministic numerics for the
//! `prefill-finalize` stage.
//!
//! Q16.16 fixed point throughout, integer-only so every tile replays
//! bit-identically in the zkVM (SKILL.md §3). The kernels mirror the shape of
//! `det_kernels::det_hidden_to_logits`, not its quantisation.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use raster::{Bytes, BytesPage, List};
use serde::{Deserialize, Serialize};

/// Fractional bits in the Q16.16 representation.
pub const FRAC_BITS: u32 = 16;

/// One unit (`1.0`) in Q16.16.
pub const ONE: i32 = 1 << FRAC_BITS;

/// Replay-unit page size for the logits projection.
pub const PAGE_SIZE: u64 = 196_608;

// ---------------------------------------------------------------------------
// The pipeline's activation type — field-for-field identical in every stage
// that passes activations on, because the chain links by structural
// commitment.
// ---------------------------------------------------------------------------

/// One prompt position: the token that produced it and its activation row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationRow {
    pub token_id: u32,
    pub values: BytesPage,
}

/// A prompt's activations, in prompt order.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationSequence {
    pub rows: List<ActivationRow>,
    pub errors: List<String>,
    /// This layer's own K/V cache, carried so a later sharing layer can attend
    /// over it. Empty on stages that donate to nobody — which is every stage
    /// except layers 13 and 14 — and ignored by every consumer that is not a
    /// sharing layer.
    pub kv: List<KeyRow>,
}

// ---------------------------------------------------------------------------
// This stage's own types
// ---------------------------------------------------------------------------

/// The output head's scalar parameters plus the final-norm vector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalHeadParams {
    pub hidden_size: u32,
    pub norm_eps: i64,
    /// `0` disables softcapping; otherwise logits become
    /// `softcap · tanh(logit / softcap)`.
    pub softcap: i32,
    pub norm_weights: BytesPage,
}

/// The stage's committed external: the final norm plus the logits projection
/// (vocab × hidden, row-major).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalHead {
    pub params: FinalHeadParams,
    #[page_size = 196_608]
    pub projection: Bytes<196_608>,
}

/// Loop-carried result of walking the prompt to its last position.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalPosition {
    pub found: bool,
    pub token_id: u32,
    pub token_count: u32,
    pub values: BytesPage,
}

/// The final position after the output norm — the vector every logit is a dot
/// product against.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct NormalizedPosition {
    pub values: BytesPage,
}

/// One scored token.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct LogitEntry {
    pub token_id: u32,
    pub value: i32,
}

/// Loop-carried summary of failed rows: how many, and the first message.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ErrorSummary {
    pub count: u32,
    pub first: String,
}

/// Key/value side of one token — the layer's KV cache entry.
///
/// MUST match `KeyRow` in every stage that carries activations: Gemma 3n's last
/// 20 layers attend over an earlier layer's cache, so the cache has to survive a
/// stage boundary, and a chain binds by structural commitment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct KeyRow {
    pub position: u32,
    pub k: BytesPage,
    pub v: BytesPage,
}

/// The stage's authorized output: the prompt's next-token scores.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PrefillLogits {
    pub decode_position: u32,
    pub logits: List<LogitEntry>,
    pub errors: List<String>,
}

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

pub fn pack_i32s(values: &[i32]) -> BytesPage {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    BytesPage::__from_parts(0, 0, bytes)
}

pub fn unpack_i32s(page: &BytesPage) -> core::result::Result<Vec<i32>, String> {
    let bytes = page.as_slice();
    if bytes.len() % 4 != 0 {
        return Err(alloc::format!(
            "page length {} is not i32-aligned",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("chunks_exact(4)")))
        .collect())
}

// ---------------------------------------------------------------------------
// Q16.16 kernels
// ---------------------------------------------------------------------------

fn clamp_i64(value: i64) -> i32 {
    if value > i32::MAX as i64 {
        i32::MAX
    } else if value < i32::MIN as i64 {
        i32::MIN
    } else {
        value as i32
    }
}

/// Q16.16 multiply, saturating.
pub fn mul(a: i32, b: i32) -> i32 {
    clamp_i64((a as i64 * b as i64) >> FRAC_BITS)
}

/// Q16.16 divide, saturating. Dividing by zero yields zero rather than
/// trapping: a tile must not panic on staged data.
pub fn div(a: i32, b: i32) -> i32 {
    if b == 0 {
        return 0;
    }
    clamp_i64(((a as i64) << FRAC_BITS) / b as i64)
}

/// Integer square root: the largest `r` with `r*r <= value`. Deterministic on
/// every target, which `f64::sqrt` would not be.
pub fn isqrt(value: i64) -> i64 {
    if value <= 0 {
        return 0;
    }
    let mut guess = value;
    let mut next = (guess + 1) / 2;
    while next < guess {
        guess = next;
        next = (guess + value / guess) / 2;
    }
    guess
}

/// `e^x` in Q16.16, via `e^x = 2^(x·log2 e)`.
pub fn exp(x: i32) -> i32 {
    const LOG2E: i32 = 94548;
    const A1: i32 = 45426;
    const A2: i32 = 15743;
    const A3: i32 = 3634;

    if x <= -(20 * ONE) {
        return 0;
    }
    if x >= 20 * ONE {
        return i32::MAX;
    }

    let y = mul(x, LOG2E);
    let int_part = y >> FRAC_BITS;
    let frac = y - (int_part << FRAC_BITS);
    let poly = ONE + mul(frac, A1 + mul(frac, A2 + mul(frac, A3)));

    if int_part >= 0 {
        clamp_i64((poly as i64) << int_part)
    } else {
        let shift = (-int_part) as u32;
        if shift >= 31 {
            0
        } else {
            poly >> shift
        }
    }
}

/// Q16.16 dot product, accumulated in i64.
pub fn dot(a: &[i32], b: &[i32]) -> core::result::Result<i32, String> {
    if a.len() != b.len() {
        return Err(alloc::format!(
            "dot width mismatch: {} vs {}",
            a.len(),
            b.len()
        ));
    }
    let mut acc: i64 = 0;
    for (left, right) in a.iter().zip(b) {
        acc += (*left as i64 * *right as i64) >> FRAC_BITS;
    }
    Ok(clamp_i64(acc))
}

/// RMS normalisation: `x[i] = x[i] / rms(x) · weight[i]`.
pub fn rms_norm(row: &mut [i32], weights: &[i32], eps: i64) -> core::result::Result<(), String> {
    if row.len() != weights.len() {
        return Err(alloc::format!(
            "rms norm width mismatch: {} values vs {} weights",
            row.len(),
            weights.len()
        ));
    }
    if row.is_empty() {
        return Err(String::from("rms norm requires at least one value"));
    }

    let mut sum_squares: i64 = 0;
    for value in row.iter() {
        sum_squares += (*value as i64 * *value as i64) >> FRAC_BITS;
    }
    let mean_square = sum_squares / row.len() as i64 + eps;
    let rms = isqrt(mean_square << FRAC_BITS);
    if rms == 0 {
        return Ok(());
    }

    for (value, weight) in row.iter_mut().zip(weights) {
        let weighted = (*value as i64 * *weight as i64) >> FRAC_BITS;
        *value = clamp_i64((weighted << FRAC_BITS) / rms);
    }
    Ok(())
}

/// `tanh(x)` in Q16.16, from [`exp`].
pub fn tanh(x: i32) -> i32 {
    let e2x = exp(x.saturating_mul(2));
    div(e2x.saturating_sub(ONE), e2x.saturating_add(ONE))
}

/// Gemma's final-logit softcapping.
pub fn softcap(value: i32, cap: i32) -> i32 {
    if cap == 0 {
        return value;
    }
    mul(cap, tanh(div(value, cap)))
}
