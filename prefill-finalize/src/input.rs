//! Shared Rastered data types and deterministic numerics for the
//! `prefill-finalize` stage.
//!
//! Q16.16 fixed point throughout, integer-only so every tile replays
//! bit-identically in the zkVM (SKILL.md §3). The kernels mirror the shape of
//! `det_kernels::det_hidden_to_logits`, not its quantisation.

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
// The pipeline's activation type — field-for-field identical in every stage
// that passes activations on, because the chain links by structural
// commitment.
// ---------------------------------------------------------------------------

/// One prompt position: the token that produced it and its activation row.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationRow {
    pub token_id: u32,
    pub values_hex: String,
}

/// A prompt's activations, in prompt order.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationSequence {
    pub rows: List<ActivationRow>,
    pub errors: List<String>,
}

// ---------------------------------------------------------------------------
// This stage's own types
// ---------------------------------------------------------------------------

/// One row of the logits projection: the token it scores and its weights.
///
/// In Gemma this matrix is *tied* to the input embedding, which is why the row
/// layout here is the same as `input-embedding`'s table.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct LogitRow {
    pub token_id: u32,
    pub values_hex: String,
}

/// The output head's scalar parameters.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalHeadParams {
    pub hidden_size: u32,
    pub norm_eps: i64,
    /// `0` disables softcapping; otherwise logits become
    /// `softcap · tanh(logit / softcap)`.
    pub softcap: i32,
    pub norm_weights_hex: String,
}

/// The stage's committed external: the final norm plus the logits projection.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalHead {
    pub params: FinalHeadParams,
    pub rows: List<LogitRow>,
}

/// Loop-carried result of walking the prompt to its last position.
///
/// The fold keeps overwriting `values_hex`, so what survives the loop is the
/// final row — `select!` cannot index "last", and the count arrives for free.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalPosition {
    pub found: bool,
    pub token_id: u32,
    pub token_count: u32,
    pub values_hex: String,
}

/// The final position after the output norm — the vector every logit is a dot
/// product against.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct NormalizedPosition {
    pub values_hex: String,
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

/// The stage's authorized output: the prompt's next-token scores.
///
/// `decode_position` is where decoding resumes — the prompt's token count.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PrefillLogits {
    pub decode_position: u32,
    pub logits: List<LogitEntry>,
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

/// Decodes a packed vector. Malformed staged data is a committed tile outcome,
/// never a panic.
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

/// Saturating add.
pub fn add_sat(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
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

/// `e^x` in Q16.16, via `e^x = 2^(x·log2 e)`: split the exponent into its
/// integer and fractional parts, shift by the former, and evaluate a cubic on
/// the latter. Saturates outside ±20.
pub fn exp(x: i32) -> i32 {
    /// log2(e)
    const LOG2E: i32 = 94548;
    // 2^f ≈ 1 + f·(a1 + f·(a2 + f·a3)) on f ∈ [0,1)
    const A1: i32 = 45426; // 0.693147
    const A2: i32 = 15743; // 0.240227
    const A3: i32 = 3634; // 0.055504

    if x <= -(20 * ONE) {
        return 0;
    }
    if x >= 20 * ONE {
        return i32::MAX;
    }

    let y = mul(x, LOG2E);
    // Arithmetic shift floors toward negative infinity, so the fraction stays
    // in [0,1) for negative exponents too.
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

/// `gelu(x) = x · sigmoid(1.702x)` — the sigmoid approximation, which needs
/// only [`exp`] and stays integer-only.
pub fn gelu(x: i32) -> i32 {
    const GELU_COEFF: i32 = 111542; // 1.702
    let z = mul(x, GELU_COEFF);
    let sigmoid = div(ONE, ONE.saturating_add(exp(-z)));
    mul(x, sigmoid)
}

/// Scales every element in place.
pub fn scale_row(row: &mut [i32], scalar: i32) {
    for value in row.iter_mut() {
        *value = mul(*value, scalar);
    }
}

/// Elementwise saturating add of `src` into `dst`.
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

/// `row × matrix'` where `matrix` is row-major `out_len × row.len()`.
pub fn matvec(
    row: &[i32],
    matrix: &[i32],
    out_len: usize,
) -> core::result::Result<Vec<i32>, String> {
    if row.is_empty() {
        return Err(String::from("projection input row is empty"));
    }
    if matrix.len() != out_len * row.len() {
        return Err(alloc::format!(
            "matrix has {} values, expected {} ({} x {})",
            matrix.len(),
            out_len * row.len(),
            out_len,
            row.len()
        ));
    }
    let mut out = Vec::with_capacity(out_len);
    for out_idx in 0..out_len {
        let offset = out_idx * row.len();
        out.push(dot(row, &matrix[offset..offset + row.len()])?);
    }
    Ok(out)
}

/// RMS normalisation: `x[i] = x[i] / rms(x) · weight[i]`, `eps` added to the
/// mean square before the root.
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

/// `tanh(x)` in Q16.16, from [`exp`]: `tanh(x) = (e^2x - 1) / (e^2x + 1)`.
pub fn tanh(x: i32) -> i32 {
    let e2x = exp(x.saturating_mul(2));
    div(e2x.saturating_sub(ONE), e2x.saturating_add(ONE))
}

/// Gemma's final-logit softcapping: `cap · tanh(value / cap)`, which squashes
/// outliers without clipping. `cap == 0` disables it.
pub fn softcap(value: i32, cap: i32) -> i32 {
    if cap == 0 {
        return value;
    }
    mul(cap, tanh(div(value, cap)))
}
