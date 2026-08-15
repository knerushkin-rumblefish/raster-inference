//! Phase 4 — `prefill-range`: tiles.
//!
//! One transformer layer over the prompt. Weight matrices are paged byte
//! regions; each matvec is a recur over `.pages`. Attention is two passes:
//! pass 1 drafts K/V, pass 2 attends. The layer loop lives in the chain
//! manifest.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use det_num::ops::rope_rotate_pairs_in_place;
use det_num::{Acc, Act};
use raster::prelude::*;

pub mod input;

use input::*;

/// Guards scalar shapes and norm-vector widths. Matrix byte lengths are
/// checked separately against the `Bytes` regions.
#[tile(kind = iter, description = "Validate this layer's declared shapes")]
pub fn validate_layer_params(params: LayerParams) -> Result<LayerParams> {
    let hidden = params.hidden_size as usize;
    let ffn = params.ffn_size as usize;
    let heads = params.num_heads as usize;
    let kv_heads = params.num_kv_heads as usize;
    let head_dim = params.head_dim as usize;
    if hidden == 0 || ffn == 0 {
        return Err(String::from(
            "transformer layer requires non-zero hidden_size and ffn_size",
        ));
    }
    if heads == 0 || kv_heads == 0 || head_dim == 0 {
        return Err(String::from(
            "transformer layer requires non-zero num_heads, num_kv_heads and head_dim",
        ));
    }
    if heads % kv_heads != 0 {
        return Err(alloc::format!(
            "layer {}: {heads} query heads do not group evenly over {kv_heads} kv heads",
            params.layer_idx
        ));
    }
    // The RoPE kernel asserts these rather than returning, and an assert inside
    // a tile is a panic instead of a committed error. Check them here so a bad
    // fixture fails as data, at a step that can say which layer it was.
    let rotary_dim = params.rotary_dim as usize;
    if rotary_dim % 2 != 0 {
        return Err(alloc::format!(
            "layer {}: rotary_dim {rotary_dim} must be even",
            params.layer_idx
        ));
    }
    if rotary_dim > head_dim {
        return Err(alloc::format!(
            "layer {}: rotary_dim {rotary_dim} exceeds head_dim {head_dim}",
            params.layer_idx
        ));
    }
    let freq_base_dim = params.rope_freq_base_dim as usize;
    if rotary_dim > 0 && (freq_base_dim < 2 || freq_base_dim % 2 != 0) {
        return Err(alloc::format!(
            "layer {}: rope_freq_base_dim {freq_base_dim} must be even and at least 2",
            params.layer_idx
        ));
    }
    if rotary_dim > 0 && params.rope_base <= 0 {
        return Err(alloc::format!(
            "layer {}: rope_base must be positive",
            params.layer_idx
        ));
    }
    if params.norm_eps < 0 {
        return Err(String::from("rms-norm epsilon must be non-negative"));
    }

    let checks: [(&str, &BytesPage, usize); 6] = [
        ("norm_input", &params.norm_input, hidden),
        ("norm_post_attn", &params.norm_post_attn, hidden),
        ("norm_pre_ffw", &params.norm_pre_ffw, hidden),
        ("norm_post_ffw", &params.norm_post_ffw, hidden),
        ("q_norm", &params.q_norm, head_dim),
        ("k_norm", &params.k_norm, head_dim),
    ];
    for (name, page, values) in checks {
        let expected = values * 4;
        if page.len() != expected {
            return Err(alloc::format!(
                "layer {} {name} has {} bytes, expected {expected} ({values} values)",
                params.layer_idx,
                page.len()
            ));
        }
    }
    Ok(params)
}

#[tile(kind = iter, description = "Layer hidden size")]
pub fn hidden_of(params: LayerParams) -> u32 {
    params.hidden_size
}

#[tile(kind = iter, description = "Query projection width")]
pub fn q_out_len(params: LayerParams) -> u32 {
    params.num_heads * params.head_dim
}

#[tile(kind = iter, description = "Key/value projection width")]
pub fn kv_out_len(params: LayerParams) -> u32 {
    params.num_kv_heads * params.head_dim
}

#[tile(kind = iter, description = "FFN width")]
pub fn ffn_of(params: LayerParams) -> u32 {
    params.ffn_size
}

#[tile(kind = iter, description = "Reject a weight region of the wrong byte length")]
pub fn assert_matrix_bytes(byte_len: u64, out_len: u32, in_width: u32) -> Result<u32> {
    let expected = out_len as u64 * in_width as u64 * 4;
    if byte_len == expected {
        Ok(0)
    } else {
        Err(alloc::format!(
            "matrix has {byte_len} bytes, expected {expected} ({out_len} x {in_width} i32s)"
        ))
    }
}

#[tile(kind = iter, description = "Start the position counter at zero")]
pub fn zero_u32() -> u32 {
    0
}

#[tile(kind = iter, description = "Advance the token position")]
pub fn advance_position(position: u32) -> u32 {
    position + 1
}

#[tile(kind = iter, description = "Zero a matvec accumulator")]
pub fn zero_accum(out_len: u32) -> ProjAccum {
    ProjAccum {
        values: pack_i32s(&vec![0; out_len as usize]),
        error: String::new(),
    }
}

/// One page of a row-major matrix: each output row is `in_width` i32s.
#[tile(
    kind = recur,
    description = "MAC one weight page into the output accumulator",
    estimated_cycles = 800_000
)]
pub fn mac_weight_page(
    input: RecurInput<BytesPage>,
    state: RecurState<ProjAccum>,
    row: BytesPage,
    in_width: u32,
) -> RecurState<ProjAccum> {
    let mut state = state;
    if !state.error.is_empty() {
        return state;
    }
    let page = input.into_value();
    if let Err(error) = mac_page(&mut state, &page, &row, in_width) {
        state.error = error;
    }
    state
}

fn mac_page(
    state: &mut ProjAccum,
    page: &BytesPage,
    row: &BytesPage,
    in_width: u32,
) -> core::result::Result<(), String> {
    let row = unpack_i32s(row)?;
    if row.len() != in_width as usize {
        return Err(alloc::format!(
            "activation has {} values, expected in_width {in_width}",
            row.len()
        ));
    }
    if in_width == 0 {
        return Err(String::from("matvec in_width is zero"));
    }
    let weights = unpack_i32s(page)?;
    if weights.len() % row.len() != 0 {
        return Err(String::from("weight page does not contain whole output rows"));
    }
    let stride = in_width as u64 * 4;
    if page.offset() % stride != 0 {
        return Err(String::from("weight page offset is not row-aligned"));
    }
    let mut out = unpack_i32s(&state.values)?;
    let start_row = (page.offset() / stride) as usize;
    let rows_in_page = weights.len() / row.len();
    for local in 0..rows_in_page {
        let out_idx = start_row + local;
        if out_idx >= out.len() {
            return Err(String::from("weight page writes past the accumulator"));
        }
        let w = &weights[local * row.len()..(local + 1) * row.len()];
        out[out_idx] = dot(&row, w)?;
    }
    state.values = pack_i32s(&out);
    Ok(())
}

// ---------------------------------------------------------------------------
// Pass 1 — projections
// ---------------------------------------------------------------------------

/// The RMS-normalised page on a Q/K/V prep. `select!(BytesPage, …)` is only
/// for indexing a `Bytes` region.
#[tile(kind = iter, description = "Normalised page of a Q/K/V prep")]
pub fn qkv_normed(prep: QkvPrep) -> BytesPage {
    prep.normed
}

/// The packed values of a vector that may carry an earlier error.
#[tile(kind = iter, description = "Values page of a packed vector")]
pub fn packed_values(vec: PackedVec) -> BytesPage {
    vec.values
}

/// RMS-normalise one activation row for the Q/K/V matvecs.
#[tile(kind = iter, description = "RMS-normalise one token for Q/K/V")]
pub fn begin_qkv(row: ActivationRow, params: LayerParams) -> QkvPrep {
    match begin_qkv_inner(&row, &params) {
        Ok((residual, normed)) => QkvPrep {
            token_id: row.token_id,
            residual: pack_i32s(&residual),
            normed: pack_i32s(&normed),
            error: String::new(),
        },
        Err(error) => QkvPrep {
            token_id: row.token_id,
            residual: pack_i32s(&[]),
            normed: pack_i32s(&vec![0; params.hidden_size as usize]),
            error,
        },
    }
}

fn begin_qkv_inner(
    row: &ActivationRow,
    params: &LayerParams,
) -> core::result::Result<(Vec<i32>, Vec<i32>), String> {
    let residual = unpack_i32s(&row.values)?;
    let hidden = params.hidden_size as usize;
    if residual.len() != hidden {
        return Err(alloc::format!(
            "activation row has {} values, expected hidden_size {hidden}",
            residual.len()
        ));
    }
    let mut normed = residual.clone();
    rms_norm(&mut normed, &unpack_i32s(&params.norm_input)?, params.norm_eps)?;
    Ok((residual, normed))
}

/// Per-head Q/K norms, then append the key and query rows.
#[tile(kind = iter, description = "Finish Q/K/V and append key and query rows")]
pub fn finish_qkv(
    output: Draft<KvSequence>,
    position: u32,
    prep: QkvPrep,
    q: ProjAccum,
    k: ProjAccum,
    v: ProjAccum,
    params: LayerParams,
) -> Draft<KvSequence> {
    let mut output = output;
    match finish_qkv_inner(position, &prep, &q, &k, &v, &params) {
        Ok((query, key)) => {
            output.queries().push(query);
            output.keys().push(key);
        }
        Err(error) => output
            .errors()
            .push(alloc::format!("token at position {position}: {error}")),
    }
    output
}

/// Rotary position embedding for one head row, in place.
///
/// Delegates to the vendored canonical kernel rather than reimplementing the
/// rotation: `det_num::ops::rope_rotate_pairs_in_place` is a byte-for-byte copy
/// of the reference's, held there by an equivalence test. The only work here is
/// moving between the chain's raw Q16.16 `i32` lanes and the `Act` newtype.
///
/// `position == 0` and `rotary_dim == 0` are no-ops inside the kernel, so this
/// needs no special case for them. The kernel's shape asserts are guaranteed by
/// [`validate_layer_params`], which turns a bad fixture into a committed error
/// before any row reaches here.
fn apply_rope(head: &mut [i32], params: &LayerParams, position: u32) {
    let rotary_dim = params.rotary_dim as usize;
    if rotary_dim == 0 || position == 0 {
        return;
    }
    let mut lanes: Vec<Act> = head.iter().map(|bits| Act::from_bits(*bits)).collect();
    rope_rotate_pairs_in_place(
        &mut lanes,
        rotary_dim,
        params.rope_freq_base_dim as usize,
        Acc::from_bits(params.rope_base),
        position as usize,
    );
    for (lane, rotated) in head.iter_mut().zip(lanes.iter()) {
        *lane = rotated.to_bits();
    }
}

fn finish_qkv_inner(
    position: u32,
    prep: &QkvPrep,
    q: &ProjAccum,
    k: &ProjAccum,
    v: &ProjAccum,
    params: &LayerParams,
) -> core::result::Result<(QueryRow, KeyRow), String> {
    if !prep.error.is_empty() {
        return Err(prep.error.clone());
    }
    if !q.error.is_empty() {
        return Err(q.error.clone());
    }
    if !k.error.is_empty() {
        return Err(k.error.clone());
    }
    if !v.error.is_empty() {
        return Err(v.error.clone());
    }
    let mut q = unpack_i32s(&q.values)?;
    let mut k = unpack_i32s(&k.values)?;
    let v = unpack_i32s(&v.values)?;
    let head_dim = params.head_dim as usize;
    let q_norm = unpack_i32s(&params.q_norm)?;
    let k_norm = unpack_i32s(&params.k_norm)?;
    // Norm then rotate, per head — the reference's order
    // (`transformer_kernels.rs`, q_norm/k_norm before the two `apply_rope`
    // calls). Swapping them changes every score.
    for head in q.chunks_mut(head_dim) {
        rms_norm(head, &q_norm, params.norm_eps)?;
        apply_rope(head, params, position);
    }
    for head in k.chunks_mut(head_dim) {
        rms_norm(head, &k_norm, params.norm_eps)?;
        apply_rope(head, params, position);
    }
    Ok((
        QueryRow {
            position,
            token_id: prep.token_id,
            q: pack_i32s(&q),
            residual: prep.residual.clone(),
        },
        KeyRow {
            position,
            k: pack_i32s(&k),
            v: pack_i32s(&v),
        },
    ))
}

// ---------------------------------------------------------------------------
// Pass 2 — attention and MLP
// ---------------------------------------------------------------------------

/// Materializes one query so the steps that follow can each read it.
#[tile(kind = iter, description = "Open one token's attention query")]
pub fn begin_attention(row: QueryRow) -> QueryRow {
    row
}

#[tile(kind = iter, description = "Empty streaming-softmax state")]
pub fn zero_attn_state() -> AttnState {
    AttnState {
        started: false,
        max: pack_i32s(&[]),
        sum: pack_i32s(&[]),
        acc: pack_i32s(&[]),
        error: String::new(),
    }
}

fn visible(query_position: u32, key_position: u32, sliding_window: u32) -> bool {
    if key_position > query_position {
        return false;
    }
    sliding_window == 0 || query_position - key_position < sliding_window
}

/// Attends one `Block` of keys/values, streaming-softmax style.
#[tile(
    kind = recur,
    description = "Attend one chunk of keys with streaming softmax",
    estimated_cycles = 30000
)]
pub fn attend_kv_chunk(
    input: RecurInput<KeyRow>,
    state: RecurState<AttnState>,
    query: QueryRow,
    params: LayerParams,
    donor_pass: bool,
) -> RecurState<AttnState> {
    let mut state = state;
    if !state.error.is_empty() {
        return state;
    }
    // A sequence cannot branch, so the caller runs this over both key lists —
    // its own and the donor's — and exactly one pass contributes. Gemma 3n's
    // last 20 layers attend over an earlier layer's cache; the rest attend over
    // their own. The two passes chain one `AttnState`, so the streaming softmax
    // is identical to a single pass over whichever list applies.
    if (params.kv_donor_layer >= 0) != donor_pass {
        return state;
    }
    let key = input.into_value();
    if !visible(query.position, key.position, params.sliding_window) {
        return state;
    }
    if let Err(error) = accumulate_key(&mut state, &query, &key, &params) {
        state.error = error;
    }
    state
}

fn accumulate_key(
    state: &mut AttnState,
    query: &QueryRow,
    key: &KeyRow,
    params: &LayerParams,
) -> core::result::Result<(), String> {
    let head_dim = params.head_dim as usize;
    let heads = params.num_heads as usize;
    let kv_heads = params.num_kv_heads as usize;
    let group = heads / kv_heads;

    let q = unpack_i32s(&query.q)?;
    let k = unpack_i32s(&key.k)?;
    let v = unpack_i32s(&key.v)?;
    if q.len() != heads * head_dim || k.len() != kv_heads * head_dim || v.len() != k.len() {
        return Err(alloc::format!(
            "attention width mismatch: q {} k {} v {} for {heads}x{head_dim} over {kv_heads} kv heads",
            q.len(),
            k.len(),
            v.len()
        ));
    }

    let (mut max, mut sum, mut acc) = if state.started {
        (
            unpack_i32s(&state.max)?,
            unpack_i32s(&state.sum)?,
            unpack_i32s(&state.acc)?,
        )
    } else {
        (
            vec![0; heads],
            vec![0; heads],
            vec![0; heads * head_dim],
        )
    };

    for head in 0..heads {
        let kv_head = head / group;
        let q_head = &q[head * head_dim..(head + 1) * head_dim];
        let k_head = &k[kv_head * head_dim..(kv_head + 1) * head_dim];
        let v_head = &v[kv_head * head_dim..(kv_head + 1) * head_dim];
        let score = mul(dot(q_head, k_head)?, params.attn_scale);
        let slot = &mut acc[head * head_dim..(head + 1) * head_dim];

        if !state.started {
            max[head] = score;
            sum[head] = ONE;
            slot.copy_from_slice(v_head);
            continue;
        }

        if score > max[head] {
            let factor = exp(max[head] - score);
            for (value, new) in slot.iter_mut().zip(v_head) {
                *value = add_sat(mul(*value, factor), *new);
            }
            sum[head] = add_sat(mul(sum[head], factor), ONE);
            max[head] = score;
        } else {
            let weight = exp(score - max[head]);
            for (value, new) in slot.iter_mut().zip(v_head) {
                *value = add_sat(*value, mul(*new, weight));
            }
            sum[head] = add_sat(sum[head], weight);
        }
    }

    state.started = true;
    state.max = pack_i32s(&max);
    state.sum = pack_i32s(&sum);
    state.acc = pack_i32s(&acc);
    Ok(())
}

/// Divide each head by its softmax denominator. Does not project through W_o.
#[tile(kind = iter, description = "Normalise streaming-softmax accumulators")]
pub fn softmax_context(state: AttnState, params: LayerParams) -> PackedVec {
    match softmax_context_inner(&state, &params) {
        Ok(values) => PackedVec {
            values: pack_i32s(&values),
            error: String::new(),
        },
        Err(error) => PackedVec {
            values: pack_i32s(&vec![0; (params.num_heads * params.head_dim) as usize]),
            error,
        },
    }
}

fn softmax_context_inner(
    state: &AttnState,
    params: &LayerParams,
) -> core::result::Result<Vec<i32>, String> {
    if !state.error.is_empty() {
        return Err(state.error.clone());
    }
    if !state.started {
        return Err(String::from("attention saw no keys at or before this token"));
    }
    let head_dim = params.head_dim as usize;
    let sums = unpack_i32s(&state.sum)?;
    let mut context = unpack_i32s(&state.acc)?;
    for (head, chunk) in context.chunks_mut(head_dim).enumerate() {
        let denominator = sums.get(head).copied().unwrap_or(0);
        for value in chunk.iter_mut() {
            *value = div(*value, denominator);
        }
    }
    Ok(context)
}

/// Post-attention RMS norm and residual add after the W_o matvec.
#[tile(kind = iter, description = "Add residual after the output projection")]
pub fn attn_residual(
    query: QueryRow,
    context: PackedVec,
    attn_proj: ProjAccum,
    params: LayerParams,
) -> PackedVec {
    if !context.error.is_empty() {
        return PackedVec {
            values: pack_i32s(&[]),
            error: alloc::format!("token at position {}: {}", query.position, context.error),
        };
    }
    if !attn_proj.error.is_empty() {
        return PackedVec {
            values: pack_i32s(&[]),
            error: alloc::format!(
                "token at position {}: {}",
                query.position,
                attn_proj.error
            ),
        };
    }
    match attn_residual_inner(&query, &attn_proj, &params) {
        Ok(values) => PackedVec {
            values: pack_i32s(&values),
            error: String::new(),
        },
        Err(error) => PackedVec {
            values: pack_i32s(&[]),
            error: alloc::format!("token at position {}: {error}", query.position),
        },
    }
}

fn attn_residual_inner(
    query: &QueryRow,
    attn_proj: &ProjAccum,
    params: &LayerParams,
) -> core::result::Result<Vec<i32>, String> {
    let mut attn = unpack_i32s(&attn_proj.values)?;
    rms_norm(
        &mut attn,
        &unpack_i32s(&params.norm_post_attn)?,
        params.norm_eps,
    )?;
    add_row(&mut attn, &unpack_i32s(&query.residual)?)?;
    Ok(attn)
}

/// Pre-FFN RMS norm. Returns the vector the gate/up matvecs consume.
#[tile(kind = iter, description = "RMS-normalise before the MLP")]
pub fn pre_ff_norm(attn: PackedVec, params: LayerParams) -> PackedVec {
    if !attn.error.is_empty() {
        return PackedVec {
            values: pack_i32s(&vec![0; params.hidden_size as usize]),
            error: attn.error,
        };
    }
    match pre_ff_norm_inner(&attn, &params) {
        Ok(values) => PackedVec {
            values: pack_i32s(&values),
            error: String::new(),
        },
        Err(error) => PackedVec {
            values: pack_i32s(&vec![0; params.hidden_size as usize]),
            error,
        },
    }
}

fn pre_ff_norm_inner(
    attn: &PackedVec,
    params: &LayerParams,
) -> core::result::Result<Vec<i32>, String> {
    let mut normed = unpack_i32s(&attn.values)?;
    rms_norm(
        &mut normed,
        &unpack_i32s(&params.norm_pre_ffw)?,
        params.norm_eps,
    )?;
    Ok(normed)
}

/// `gelu(gate) ⊙ up`.
#[tile(kind = iter, description = "Elementwise GELU(gate) * up")]
pub fn gelu_mul(gate: ProjAccum, up: ProjAccum) -> PackedVec {
    if !gate.error.is_empty() {
        return PackedVec {
            values: pack_i32s(&[]),
            error: gate.error,
        };
    }
    if !up.error.is_empty() {
        return PackedVec {
            values: pack_i32s(&[]),
            error: up.error,
        };
    }
    match gelu_mul_inner(&gate, &up) {
        Ok(values) => PackedVec {
            values: pack_i32s(&values),
            error: String::new(),
        },
        Err(error) => PackedVec {
            values: pack_i32s(&[]),
            error,
        },
    }
}

fn gelu_mul_inner(
    gate: &ProjAccum,
    up: &ProjAccum,
) -> core::result::Result<Vec<i32>, String> {
    let mut gate = unpack_i32s(&gate.values)?;
    let up = unpack_i32s(&up.values)?;
    if gate.len() != up.len() {
        return Err(alloc::format!(
            "gate/up width mismatch: {} vs {}",
            gate.len(),
            up.len()
        ));
    }
    for (value, up_value) in gate.iter_mut().zip(&up) {
        *value = mul(gelu(*value), *up_value);
    }
    Ok(gate)
}

/// Post-FFN RMS norm, residual add, optional layer scalar, append the row.
#[tile(kind = iter, description = "Finish the MLP and append the layer output row")]
pub fn finish_mlp(
    output: Draft<ActivationSequence>,
    query: QueryRow,
    residual: PackedVec,
    ff_in: PackedVec,
    ff: ProjAccum,
    params: LayerParams,
    own_key: KeyRow,
) -> Draft<ActivationSequence> {
    let mut output = output;
    match finish_mlp_inner(&residual, &ff_in, &ff, &params) {
        Ok(values) => output.rows().push(ActivationRow {
            token_id: query.token_id,
            values: pack_i32s(&values),
        }),
        Err(error) => output
            .errors()
            .push(alloc::format!("token at position {}: {error}", query.position)),
    }
    // Publish this layer's own cache entry alongside the row. Only layers 13
    // and 14 are ever read back, but the type is uniform across all 35 stages,
    // so every layer carries its own — the alternative is a second program.
    output.kv().push(own_key);
    output
}

fn finish_mlp_inner(
    residual: &PackedVec,
    ff_in: &PackedVec,
    ff: &ProjAccum,
    params: &LayerParams,
) -> core::result::Result<Vec<i32>, String> {
    if !residual.error.is_empty() {
        return Err(residual.error.clone());
    }
    if !ff_in.error.is_empty() {
        return Err(ff_in.error.clone());
    }
    if !ff.error.is_empty() {
        return Err(ff.error.clone());
    }
    let residual_vals = unpack_i32s(&residual.values)?;
    let mut ff = unpack_i32s(&ff.values)?;
    rms_norm(
        &mut ff,
        &unpack_i32s(&params.norm_post_ffw)?,
        params.norm_eps,
    )?;
    add_row(&mut ff, &residual_vals)?;
    if params.layer_scalar != 0 {
        scale_row(&mut ff, params.layer_scalar);
    }
    Ok(ff)
}

// ---------------------------------------------------------------------------
// Failure accounting
// ---------------------------------------------------------------------------

#[tile(kind = recur, description = "Summarise the rows this layer could not produce")]
pub fn summarise_errors(
    input: RecurInput<String>,
    state: RecurState<ErrorSummary>,
) -> RecurState<ErrorSummary> {
    let mut state = state;
    let message = input.into_value();
    if state.count == 0 {
        state.first = message;
    }
    state.count += 1;
    state
}

#[tile(kind = iter, description = "Reject an incomplete layer output")]
pub fn assert_layer_complete(
    projection_errors: u32,
    first_projection_error: String,
    attention_errors: u32,
    first_attention_error: String,
) -> Result<u32> {
    if projection_errors > 0 {
        return Err(alloc::format!(
            "{projection_errors} row(s) failed projection; first: {first_projection_error}"
        ));
    }
    if attention_errors > 0 {
        return Err(alloc::format!(
            "{attention_errors} row(s) failed attention/MLP; first: {first_attention_error}"
        ));
    }
    Ok(0)
}
