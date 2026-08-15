//! Shared Rastered data types and deterministic numerics for the
//! `prefill-prepare-aux` stage.
//!
//! Q16.16 fixed-point throughout. Tiles must replay bit-identically inside the
//! zkVM, so floats are out — see SKILL.md §3.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use raster::{Bytes, BytesPage, List};
use serde::{Deserialize, Serialize};

/// Fractional bits in the Q16.16 representation.
pub const FRAC_BITS: u32 = 16;

/// One unit (`1.0`) in Q16.16.
pub const ONE: i32 = 1 << FRAC_BITS;

/// Replay-unit page size for PLE tables and the projection matrix.
pub const PAGE_SIZE: u64 = 196_608;

// ---------------------------------------------------------------------------
// Types crossing the stage boundary
// ---------------------------------------------------------------------------

/// One prompt position, from `input_embedding`.
///
/// MUST match `ActivationRow` in `input-embedding` and `prefill-range`
/// field-for-field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationRow {
    pub token_id: u32,
    pub values: BytesPage,
}

/// MUST match `ActivationSequence` in `input-embedding` and `prefill-range`.
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

/// Scalars plus the PLE projection norm. Materializable — legal `args`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleLayerParams {
    pub layer_idx: u32,
    pub hidden_size: u32,
    pub ple_width: u32,
    pub embedding_scale: i32,
    pub projection_scalar: i32,
    pub input_scale: i32,
    pub norm_eps: i64,
    pub norm_weights: BytesPage,
}

/// This layer's committed external.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleLayer {
    pub params: PleLayerParams,
    #[page_size = 196_608]
    pub embeddings: Bytes<196_608>,
    #[page_size = 196_608]
    pub projection: Bytes<196_608>,
}

/// One token's work in flight.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleTask {
    pub token_id: u32,
    pub activation: BytesPage,
}

/// Matvec accumulator: a packed output vector, plus an error if a page failed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ProjAccum {
    pub values: BytesPage,
    pub error: String,
}

/// The projected + normalised half of one row, between the two compute tiles.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ProjectedRow {
    pub values: BytesPage,
    pub error: String,
}

/// Loop-carried summary of the rows that failed.
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

/// The stage's authorized output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleLayerInputs {
    pub layer_idx: u32,
    pub rows: List<BytesPage>,
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

pub fn unpack_i32s_at(
    page: &BytesPage,
    byte_off: u64,
    n_values: u32,
) -> core::result::Result<Vec<i32>, String> {
    let local = byte_off
        .checked_sub(page.offset())
        .ok_or_else(|| String::from("row offset is before this page"))? as usize;
    let len = n_values as usize * 4;
    let bytes = page.as_slice().get(local..local + len).ok_or_else(|| {
        alloc::format!(
            "row at byte {byte_off} spans past this page (local {local}, need {len}, have {})",
            page.len()
        )
    })?;
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

pub fn add_sat(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

pub fn scale_row(row: &mut [i32], scalar: i32) {
    for value in row.iter_mut() {
        *value = mul(*value, scalar);
    }
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
