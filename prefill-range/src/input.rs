//! Shared Rastered data types and deterministic numerics for the
//! `prefill-range` stage.
//!
//! Q16.16 fixed point throughout (`1.0 == 1 << 16`), integer-only so every
//! tile replays bit-identically in the zkVM (SKILL.md §3).

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use raster::{Bytes, BytesPage, List};
use serde::{Deserialize, Serialize};

/// Fractional bits in the Q16.16 representation.
pub const FRAC_BITS: u32 = 16;

/// One unit (`1.0`) in Q16.16.
pub const ONE: i32 = 1 << FRAC_BITS;

/// Replay-unit page size for every weight matrix in this program.
pub const PAGE_SIZE: u64 = 196_608;

// ---------------------------------------------------------------------------
// The pipeline's activation type — field-for-field identical in every stage
// that passes activations on.
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

/// Scalars plus the four (plus Q/K) norm vectors. Materializable — legal `args`.
/// Weight matrices live on [`TransformerLayer`] as `Bytes` regions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct LayerParams {
    pub layer_idx: u32,
    pub hidden_size: u32,
    pub ffn_size: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    /// `0` = full attention; otherwise a key is visible only while
    /// `query.position - key.position < sliding_window`.
    pub sliding_window: u32,
    /// `1/sqrt(head_dim)`, applied to attention scores.
    pub attn_scale: i32,
    /// Applied to the layer's output row; `0` means "not present".
    pub layer_scalar: i32,
    pub norm_eps: i64,
    /// RoPE base as `Acc` (Q32.32) bits — 10 000 on sliding layers, 1 000 000 on
    /// full-attention ones, from `config.json`'s `rope_parameters`.
    pub rope_base: i64,
    /// How many of a head's lanes rotate. The whole head on a sliding layer;
    /// `head_dim × partial_rotary_factor` (a quarter) on a full-attention one,
    /// which is what makes Gemma's "partial rotary" partial.
    pub rotary_dim: u32,
    /// Denominator dimension of the frequency ladder — always `head_dim`, and
    /// *not* the same as `rotary_dim` once the rotation is partial.
    pub rope_freq_base_dim: u32,
    /// Layer this one borrows K/V from, or `-1` when it computes its own.
    ///
    /// Gemma 3n shares KV across its last `num_kv_shared_layers` layers: a
    /// sharing layer projects only Q and attends over the donor's cache, so it
    /// never runs `w_k`/`w_v` at all. `-1` rather than `Option` because the
    /// value crosses a tile boundary and must stay a plain scalar.
    pub kv_donor_layer: i32,
    pub norm_input: BytesPage,
    pub norm_post_attn: BytesPage,
    pub norm_pre_ffw: BytesPage,
    pub norm_post_ffw: BytesPage,
    pub q_norm: BytesPage,
    pub k_norm: BytesPage,
}

/// One transformer layer's committed external.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TransformerLayer {
    pub params: LayerParams,
    #[page_size = 196_608]
    pub w_q: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_k: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_v: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_o: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_gate: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_up: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_down: Bytes<196_608>,
}

/// Key/value side of one token, for the attention scan.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct KeyRow {
    pub position: u32,
    pub k: BytesPage,
    pub v: BytesPage,
}

/// Query side of one token, plus the residual it will be added back to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct QueryRow {
    pub position: u32,
    pub token_id: u32,
    pub q: BytesPage,
    pub residual: BytesPage,
}

/// Pass 1's result: keys and queries as separate lists so the key scan does
/// not rematerialize query-side fields.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct KvSequence {
    pub keys: List<KeyRow>,
    pub queries: List<QueryRow>,
    pub errors: List<String>,
}

/// RMS-normalised activation, ready for Q/K/V matvecs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct QkvPrep {
    pub token_id: u32,
    pub residual: BytesPage,
    pub normed: BytesPage,
    pub error: String,
}

/// Matvec accumulator: packed output vector plus an error if a page failed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ProjAccum {
    pub values: BytesPage,
    pub error: String,
}

/// Streaming softmax state — one running max, sum and weighted accumulator
/// per head.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct AttnState {
    pub started: bool,
    pub max: BytesPage,
    pub sum: BytesPage,
    pub acc: BytesPage,
    pub error: String,
}

/// A packed vector that may carry a failure from an earlier step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PackedVec {
    pub values: BytesPage,
    pub error: String,
}

/// Loop-carried summary of failed rows.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ErrorSummary {
    pub count: u32,
    pub first: String,
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

pub fn mul(a: i32, b: i32) -> i32 {
    clamp_i64((a as i64 * b as i64) >> FRAC_BITS)
}

pub fn div(a: i32, b: i32) -> i32 {
    if b == 0 {
        return 0;
    }
    clamp_i64(((a as i64) << FRAC_BITS) / b as i64)
}

pub fn add_sat(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

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

pub fn gelu(x: i32) -> i32 {
    const GELU_COEFF: i32 = 111542;
    let z = mul(x, GELU_COEFF);
    let sigmoid = div(ONE, ONE.saturating_add(exp(-z)));
    mul(x, sigmoid)
}

pub fn scale_row(row: &mut [i32], scalar: i32) {
    for value in row.iter_mut() {
        *value = mul(*value, scalar);
    }
}

pub fn add_row(dst: &mut [i32], src: &[i32]) -> core::result::Result<(), String> {
    if dst.len() != src.len() {
        return Err(alloc::format!(
            "row add width mismatch: {} vs {}",
            dst.len(),
            src.len()
        ));
    }
    for (value, other) in dst.iter_mut().zip(src) {
        *value = add_sat(*value, *other);
    }
    Ok(())
}

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
