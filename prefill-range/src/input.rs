//! Shared Rastered data types and deterministic numerics for the
//! `prefill-range` stage.
//!
//! # Numerics
//!
//! Q16.16 fixed point throughout (`1.0 == 1 << 16`), integer-only so every
//! tile replays bit-identically in the zkVM (SKILL.md §3). That constraint is
//! why `exp` here is a polynomial on the fraction bits rather than
//! `f32::exp`, and why the square root is an integer Newton iteration.
//!
//! The kernels mirror the *shape* of `det_kernels::det_layer_prefill`, not its
//! quantisation. Deliberately not modelled (each would be another dimension to
//! decompose, and this is a chain example): RoPE, multi-head attention,
//! sliding windows, KV sharing between layers, and the PLE gate block.

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
// The pipeline's activation type — shared by every stage that passes
// activations on. Field-for-field identical in `input-embedding` and
// `prefill-prepare-aux`: the chain links stages by structural commitment, so
// a layer's output can only feed the next layer if the shapes match exactly.
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

/// One transformer layer's weights, as this stage's committed external.
///
/// Every matrix rides as a packed hex leaf, so the whole layer is
/// `Materializable` and can be a per-call argument. Row-major throughout:
/// `w_q_hex` is `hidden_size × hidden_size`, `w_up_hex` is
/// `ffn_size × hidden_size`, `w_down_hex` is `hidden_size × ffn_size`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct LayerParams {
    pub layer_idx: u32,
    pub hidden_size: u32,
    pub ffn_size: u32,
    /// Query heads. `w_q` is `num_heads · head_dim × hidden_size`.
    pub num_heads: u32,
    /// Key/value heads — fewer than `num_heads` means grouped-query
    /// attention: several query heads share one K/V head.
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
    pub norm_input_hex: String,
    pub norm_post_attn_hex: String,
    pub norm_pre_ffw_hex: String,
    pub norm_post_ffw_hex: String,
    /// Per-head RMS norms applied to Q and K after projection, `head_dim` wide.
    pub q_norm_hex: String,
    pub k_norm_hex: String,
    pub w_q_hex: String,
    pub w_k_hex: String,
    pub w_v_hex: String,
    pub w_o_hex: String,
    pub w_gate_hex: String,
    pub w_up_hex: String,
    pub w_down_hex: String,
}

/// One token's projected attention inputs, plus the residual it will be added
/// back to. `position` is the index the first pass stamped — a recur *tile*
/// can read `input.index()`, which is why the projection pass is a tile and
/// not a sequence.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct KvRow {
    pub position: u32,
    pub token_id: u32,
    pub q_hex: String,
    pub k_hex: String,
    pub v_hex: String,
    pub residual_hex: String,
}

/// The first pass's result: the whole prompt's Q/K/V.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct KvSequence {
    pub rows: List<KvRow>,
    pub errors: List<String>,
}

/// The query side of one attention row, materialized once per token.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct AttnQuery {
    pub position: u32,
    pub token_id: u32,
    pub q_hex: String,
    pub residual_hex: String,
}

/// Streaming softmax state, carried across the key/value scan — one running
/// max, sum and weighted accumulator **per head**, packed into hex leaves.
///
/// This is the largest recur state in the repo: the accumulator is
/// `num_heads · head_dim` wide. It is *fixed* width, not growing — the
/// alternative (collect all scores, then normalise) would need a second list
/// per token and a second pass over it.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct AttnState {
    pub started: bool,
    pub max_hex: String,
    pub sum_hex: String,
    pub acc_hex: String,
    pub error: String,
}

/// The attention block's output for one token, before the MLP.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct AttnOutput {
    pub values_hex: String,
    pub error: String,
}

/// Loop-carried summary of failed rows: how many, and the first message.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ErrorSummary {
    pub count: u32,
    pub first: String,
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
