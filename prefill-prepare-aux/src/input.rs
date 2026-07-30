//! Shared Rastered data types and deterministic numerics for the
//! `prefill-prepare-aux` stage.
//!
//! # Numerics
//!
//! Every value is a **Q16.16 fixed-point** integer: `1.0` is `1 << 16`. Tiles
//! must replay bit-identically inside the zkVM, so floats are out — see
//! SKILL.md §3. All kernels here are integer-only, saturating, and free of
//! any platform-dependent behaviour.
//!
//! These are stand-ins with the same *shape* as Gemma's `det_num` kernels
//! (`scale_act`, `add_sat`, `requantize`, `rms_norm_in_place`), not a port of
//! them: this repo is a chain example, and porting the real quantisation
//! belongs with the model weights it was tuned for.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use raster::List;
use serde::{Deserialize, Serialize};

/// Fractional bits in the Q16.16 representation.
pub const FRAC_BITS: u32 = 16;

/// One unit (`1.0`) in Q16.16.
pub const ONE: i32 = 1 << FRAC_BITS;

/// Number of hex chars one value occupies in a packed vector.
pub const HEX_CHARS_PER_VALUE: u32 = 8;

// ---------------------------------------------------------------------------
// Types crossing the stage boundary
// ---------------------------------------------------------------------------

/// One prompt position, from `input_embedding`.
///
/// MUST match `ActivationRow` in `input-embedding` and `prefill-range`
/// field-for-field — the chain links stages by structural commitment, not by
/// Rust type name.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationRow {
    pub token_id: u32,
    pub values_hex: String,
}

/// MUST match `ActivationSequence` in `input-embedding` and `prefill-range`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationSequence {
    pub rows: List<ActivationRow>,
    pub errors: List<String>,
}

/// One row of this layer's PLE embedding table: the per-layer embedding for a
/// token id, packed as `ple_width` Q16.16 values.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleEmbeddingRow {
    pub token_id: u32,
    pub values_hex: String,
}

/// Everything about one PLE layer that a tile needs materialised at once.
///
/// All fields are scalars (the matrices ride as packed hex leaves), so this is
/// `Materializable` and can be a per-call argument. `projection_hex` holds
/// `ple_width × hidden_size` values in row-major order.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleLayerParams {
    pub layer_idx: u32,
    pub hidden_size: u32,
    pub ple_width: u32,
    pub embedding_scale: i32,
    pub projection_scalar: i32,
    pub input_scale: i32,
    pub norm_eps: i64,
    pub projection_hex: String,
    pub norm_weights_hex: String,
}

/// This layer's committed external: its parameters plus its PLE embedding
/// table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleLayer {
    pub params: PleLayerParams,
    pub embedding_rows: List<PleEmbeddingRow>,
}

/// One token's work in flight: its id and its activation row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleTask {
    pub token_id: u32,
    pub activation_hex: String,
}

/// Outcome of scanning this layer's PLE table for a [`PleTask`]'s token id.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleEmbeddingMatch {
    pub found: bool,
    pub values_hex: String,
}

/// The projected + normalised half of one row, between the two compute tiles.
///
/// `error` is non-empty when the row could not be produced. A recur sequence
/// has no fallible form, and `.expect()` in one would panic the program —
/// which publishes nothing at all. Carrying the failure as data keeps it a
/// committed outcome that the check at the end of the sequence can act on.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ProjectedRow {
    pub values_hex: String,
    pub error: String,
}

/// Loop-carried summary of the rows that failed: how many, and the first
/// message (kept small — recur state is re-committed every iteration).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ErrorSummary {
    pub count: u32,
    pub first: String,
}

/// The stage's authorized output: this layer's prefill input rows, one per
/// prompt token, in prompt order.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleLayerInputs {
    pub layer_idx: u32,
    pub rows: List<String>,
    pub errors: List<String>,
}

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

/// Packs Q16.16 values into the hex leaf form: 8 lowercase hex chars each.
pub fn pack_hex(values: &[i32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = Vec::with_capacity(values.len() * HEX_CHARS_PER_VALUE as usize);
    for value in values {
        let bits = *value as u32;
        for shift in (0..HEX_CHARS_PER_VALUE).rev() {
            out.push(HEX[((bits >> (shift * 4)) & 0xf) as usize]);
        }
    }
    String::from_utf8(out).expect("hex packing is always ascii")
}

/// Decodes a packed vector. A malformed leaf is a committed tile error, never a
/// panic: staged data is untrusted until the commitment says otherwise.
pub fn unpack_hex(packed: &str) -> core::result::Result<Vec<i32>, String> {
    let bytes = packed.as_bytes();
    if bytes.len() % HEX_CHARS_PER_VALUE as usize != 0 {
        return Err(alloc::format!(
            "packed vector length {} is not a multiple of {} hex chars",
            bytes.len(),
            HEX_CHARS_PER_VALUE
        ));
    }
    let mut values = Vec::with_capacity(bytes.len() / HEX_CHARS_PER_VALUE as usize);
    for chunk in bytes.chunks_exact(HEX_CHARS_PER_VALUE as usize) {
        let mut bits: u32 = 0;
        for &ch in chunk {
            let nibble = match ch {
                b'0'..=b'9' => ch - b'0',
                b'a'..=b'f' => ch - b'a' + 10,
                _ => {
                    return Err(alloc::format!(
                        "packed vector contains non-hex byte 0x{ch:02x}"
                    ))
                }
            };
            bits = (bits << 4) | u32::from(nibble);
        }
        values.push(bits as i32);
    }
    Ok(values)
}

// ---------------------------------------------------------------------------
// Q16.16 kernels — integer only, saturating, deterministic
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

/// Saturating add — Gemma's `add_sat`.
pub fn add_sat(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

/// Scales every element in place.
pub fn scale_row(row: &mut [i32], scalar: i32) {
    for value in row.iter_mut() {
        *value = mul(*value, scalar);
    }
}

/// Integer square root of a non-negative i64 (Newton, monotone and exact —
/// the largest `r` with `r*r <= value`). Deterministic on every target, which
/// `f64::sqrt` would not be.
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

/// `row × matrix'` where `matrix` is row-major `out_len × row.len()`: one dot
/// product per output element, accumulated in i64 and requantised once.
pub fn matvec(row: &[i32], matrix: &[i32], out_len: usize) -> core::result::Result<Vec<i32>, String> {
    if row.is_empty() {
        return Err(String::from("projection input row is empty"));
    }
    if matrix.len() != out_len * row.len() {
        return Err(alloc::format!(
            "projection matrix has {} values, expected {} ({} x {})",
            matrix.len(),
            out_len * row.len(),
            out_len,
            row.len()
        ));
    }
    let mut out = Vec::with_capacity(out_len);
    for out_idx in 0..out_len {
        let offset = out_idx * row.len();
        let mut acc: i64 = 0;
        for (col, value) in row.iter().enumerate() {
            acc += (*value as i64 * matrix[offset + col] as i64) >> FRAC_BITS;
        }
        out.push(clamp_i64(acc));
    }
    Ok(out)
}

/// RMS normalisation in place: `x[i] = x[i] / rms(x) * weight[i]`, with `eps`
/// added to the mean square before the root.
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

    // mean square, in Q16.16
    let mut sum_squares: i64 = 0;
    for value in row.iter() {
        sum_squares += (*value as i64 * *value as i64) >> FRAC_BITS;
    }
    let mean_square = sum_squares / row.len() as i64 + eps;

    // sqrt of a Q16.16 value: shift up by FRAC_BITS first so the root lands
    // back in Q16.16.
    let rms = isqrt(mean_square << FRAC_BITS);
    if rms == 0 {
        // An all-zero row normalises to itself; weighting it changes nothing.
        return Ok(());
    }

    for (value, weight) in row.iter_mut().zip(weights) {
        let weighted = (*value as i64 * *weight as i64) >> FRAC_BITS;
        *value = clamp_i64((weighted << FRAC_BITS) / rms);
    }
    Ok(())
}
