//! Phase 4 — `prefill-range`: tiles.
//!
//! One transformer layer over the prompt, following
//! `det_kernels::det_layer_prefill`:
//!
//! ```text
//!   normed  = rms_norm(x, w_input)
//!   q,k,v   = normed × W_q, W_k, W_v
//!   attn    = causal_softmax(q·kᵀ · scale) × v
//!   x       = rms_norm(attn × W_o, w_post_attn) + x
//!   normed  = rms_norm(x, w_pre_ffw)
//!   ff      = (gelu(normed × W_gate) ⊙ (normed × W_up)) × W_down
//!   x       = rms_norm(ff, w_post_ffw) + x
//! ```
//!
//! Attention is why this stage has two passes: a token attends over *every*
//! token's keys, so the K/V sequence has to be a finished, storage-backed list
//! before any attention step can read it. Pass 1 builds it as a draft; pass 2
//! attends over it. Within pass 2 the softmax is streaming (running max and
//! sum), which is what keeps one replay unit to one key chunk.
//!
//! Attention is multi-head with grouped queries: `num_heads` query heads over
//! `num_kv_heads` key/value heads, per-head RMS norms on Q and K, and an
//! optional sliding window. Each head keeps its own streaming-softmax max,
//! sum and accumulator, packed into the recur state.
//!
//! Like `prefill-prepare-aux`, the *layer* loop lives in the chain manifest —
//! one stage instance per layer, each consuming the previous layer's output.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use raster::prelude::*;

pub mod input;

// The glob also brings the derive-generated `…DraftExt` traits into scope.
use input::*;

/// Guards the layer external before any of it is used: every matrix must have
/// the width its declared shape implies.
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
    if params.norm_eps < 0 {
        return Err(String::from("rms-norm epsilon must be non-negative"));
    }

    let checks: [(&str, &String, usize); 13] = [
        ("norm_input", &params.norm_input_hex, hidden),
        ("norm_post_attn", &params.norm_post_attn_hex, hidden),
        ("norm_pre_ffw", &params.norm_pre_ffw_hex, hidden),
        ("norm_post_ffw", &params.norm_post_ffw_hex, hidden),
        ("q_norm", &params.q_norm_hex, head_dim),
        ("k_norm", &params.k_norm_hex, head_dim),
        ("w_q", &params.w_q_hex, heads * head_dim * hidden),
        ("w_k", &params.w_k_hex, kv_heads * head_dim * hidden),
        ("w_v", &params.w_v_hex, kv_heads * head_dim * hidden),
        ("w_o", &params.w_o_hex, hidden * heads * head_dim),
        ("w_gate", &params.w_gate_hex, ffn * hidden),
        ("w_up", &params.w_up_hex, ffn * hidden),
        ("w_down", &params.w_down_hex, hidden * ffn),
    ];
    for (name, packed, values) in checks {
        let expected = values * HEX_CHARS_PER_VALUE as usize;
        if packed.len() != expected {
            return Err(alloc::format!(
                "layer {} {name} has {} hex chars, expected {expected} ({values} values)",
                params.layer_idx,
                packed.len()
            ));
        }
    }
    Ok(params)
}

// ---------------------------------------------------------------------------
// Pass 1 — projections
// ---------------------------------------------------------------------------

/// Normalises one activation row and projects it to Q, K and V.
///
/// A recur *tile*, not a sequence, for one reason: `input.index()` is the
/// token's position, and only a tile can read it. That position is what the
/// causal mask in pass 2 compares against.
#[tile(
    kind = recur,
    description = "Normalise and project one token to Q/K/V",
    estimated_cycles = 40000
)]
pub fn project_qkv_row(
    input: RecurInput<ActivationRow>,
    output: RecurOutput<KvSequence>,
    params: LayerParams,
) -> RecurOutput<KvSequence> {
    let mut output = output;
    let position = input.index() as u32;
    let row = input.into_value();

    match project_row(&row, &params) {
        Ok((q, k, v, residual)) => output.rows().push(KvRow {
            position,
            token_id: row.token_id,
            q_hex: pack_hex(&q),
            k_hex: pack_hex(&k),
            v_hex: pack_hex(&v),
            residual_hex: pack_hex(&residual),
        }),
        Err(error) => output
            .errors()
            .push(alloc::format!("token at position {position}: {error}")),
    }
    output
}

type ProjectedRow = (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>);

fn project_row(row: &ActivationRow, params: &LayerParams) -> core::result::Result<ProjectedRow, String> {
    let residual = unpack_hex(&row.values_hex)?;
    let hidden = params.hidden_size as usize;
    if residual.len() != hidden {
        return Err(alloc::format!(
            "activation row has {} values, expected hidden_size {hidden}",
            residual.len()
        ));
    }

    let mut normed = residual.clone();
    rms_norm(&mut normed, &unpack_hex(&params.norm_input_hex)?, params.norm_eps)?;

    let head_dim = params.head_dim as usize;
    let q_width = params.num_heads as usize * head_dim;
    let kv_width = params.num_kv_heads as usize * head_dim;

    let mut q = matvec(&normed, &unpack_hex(&params.w_q_hex)?, q_width)?;
    let mut k = matvec(&normed, &unpack_hex(&params.w_k_hex)?, kv_width)?;
    let v = matvec(&normed, &unpack_hex(&params.w_v_hex)?, kv_width)?;

    // Q and K are normalised per head, not across the concatenated vector.
    let q_norm = unpack_hex(&params.q_norm_hex)?;
    let k_norm = unpack_hex(&params.k_norm_hex)?;
    for head in q.chunks_mut(head_dim) {
        rms_norm(head, &q_norm, params.norm_eps)?;
    }
    for head in k.chunks_mut(head_dim) {
        rms_norm(head, &k_norm, params.norm_eps)?;
    }
    Ok((q, k, v, residual))
}

// ---------------------------------------------------------------------------
// Pass 2 — attention and MLP
// ---------------------------------------------------------------------------

/// Materializes one K/V row's query side, so the steps that follow can each
/// read it.
#[tile(kind = iter, description = "Open one token's attention query")]
pub fn begin_attention(row: KvRow) -> AttnQuery {
    AttnQuery {
        position: row.position,
        token_id: row.token_id,
        q_hex: row.q_hex,
        residual_hex: row.residual_hex,
    }
}

/// Whether a key is visible to a query: causal always, plus the layer's
/// sliding window when it has one.
fn visible(query_position: u32, key_position: u32, sliding_window: u32) -> bool {
    if key_position > query_position {
        return false;
    }
    sliding_window == 0 || query_position - key_position < sliding_window
}

/// Attends one `Block` of keys/values, streaming-softmax style.
///
/// Keys at positions after the query are skipped — that is the causal mask.
/// The running max keeps `exp` in range without a second pass over the scores,
/// which is what lets one replay unit see one chunk instead of the whole
/// sequence.
#[tile(
    kind = recur,
    description = "Attend one chunk of keys with streaming softmax",
    estimated_cycles = 30000
)]
pub fn attend_kv_chunk(
    input: RecurInput<Block<KvRow>>,
    state: RecurState<AttnState>,
    query: AttnQuery,
    params: LayerParams,
) -> RecurState<AttnState> {
    let mut state = state;
    if !state.error.is_empty() {
        return state;
    }
    for key in input.into_value() {
        if !visible(query.position, key.position, params.sliding_window) {
            continue;
        }
        if let Err(error) = accumulate_key(&mut state, &query, &key, &params) {
            state.error = error;
            return state;
        }
    }
    state
}

fn accumulate_key(
    state: &mut AttnState,
    query: &AttnQuery,
    key: &KvRow,
    params: &LayerParams,
) -> core::result::Result<(), String> {
    let head_dim = params.head_dim as usize;
    let heads = params.num_heads as usize;
    let kv_heads = params.num_kv_heads as usize;
    let group = heads / kv_heads;

    let q = unpack_hex(&query.q_hex)?;
    let k = unpack_hex(&key.k_hex)?;
    let v = unpack_hex(&key.v_hex)?;
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
            unpack_hex(&state.max_hex)?,
            unpack_hex(&state.sum_hex)?,
            unpack_hex(&state.acc_hex)?,
        )
    } else {
        (
            alloc::vec![0; heads],
            alloc::vec![0; heads],
            alloc::vec![0; heads * head_dim],
        )
    };

    for head in 0..heads {
        // Grouped-query attention: several query heads share one K/V head.
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
            // Rescale everything this head accumulated to the new maximum.
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
    state.max_hex = pack_hex(&max);
    state.sum_hex = pack_hex(&sum);
    state.acc_hex = pack_hex(&acc);
    Ok(())
}

/// Finishes the attention block for one token: normalise by the softmax
/// denominator, project through `W_o`, normalise, and add the residual.
#[tile(kind = iter, description = "Finish one token's attention block")]
pub fn finish_attention(query: AttnQuery, state: AttnState, params: LayerParams) -> AttnOutput {
    match finish_attention_row(&query, &state, &params) {
        Ok(values) => AttnOutput {
            values_hex: pack_hex(&values),
            error: String::new(),
        },
        // Failure travels as data: a recur sequence has no fallible form, and
        // `.expect()` in one would panic the program into publishing nothing.
        Err(error) => AttnOutput {
            values_hex: String::new(),
            error: alloc::format!("token at position {}: {error}", query.position),
        },
    }
}

fn finish_attention_row(
    query: &AttnQuery,
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
    let sums = unpack_hex(&state.sum_hex)?;
    let mut context = unpack_hex(&state.acc_hex)?;
    // Each head normalises by its own softmax denominator before the heads are
    // concatenated back into one vector for the output projection.
    for (head, chunk) in context.chunks_mut(head_dim).enumerate() {
        let denominator = sums.get(head).copied().unwrap_or(0);
        for value in chunk.iter_mut() {
            *value = div(*value, denominator);
        }
    }

    let hidden = params.hidden_size as usize;
    let mut attn = matvec(&context, &unpack_hex(&params.w_o_hex)?, hidden)?;
    rms_norm(
        &mut attn,
        &unpack_hex(&params.norm_post_attn_hex)?,
        params.norm_eps,
    )?;
    add_row(&mut attn, &unpack_hex(&query.residual_hex)?)?;
    Ok(attn)
}

/// Runs the MLP block on the attention output and appends the layer's row.
#[tile(
    kind = iter,
    description = "Run the MLP block and append the layer output row",
    estimated_cycles = 60000
)]
pub fn apply_mlp(
    output: Draft<ActivationSequence>,
    query: AttnQuery,
    attn: AttnOutput,
    params: LayerParams,
) -> Draft<ActivationSequence> {
    let mut output = output;
    match mlp_row(&attn, &params) {
        Ok(values) => output.rows().push(ActivationRow {
            token_id: query.token_id,
            values_hex: pack_hex(&values),
        }),
        Err(error) => output
            .errors()
            .push(alloc::format!("token at position {}: {error}", query.position)),
    }
    output
}

fn mlp_row(attn: &AttnOutput, params: &LayerParams) -> core::result::Result<Vec<i32>, String> {
    if !attn.error.is_empty() {
        return Err(attn.error.clone());
    }
    let residual = unpack_hex(&attn.values_hex)?;
    let hidden = params.hidden_size as usize;
    let ffn = params.ffn_size as usize;

    let mut normed = residual.clone();
    rms_norm(
        &mut normed,
        &unpack_hex(&params.norm_pre_ffw_hex)?,
        params.norm_eps,
    )?;

    let mut gate = matvec(&normed, &unpack_hex(&params.w_gate_hex)?, ffn)?;
    let up = matvec(&normed, &unpack_hex(&params.w_up_hex)?, ffn)?;
    for (value, up_value) in gate.iter_mut().zip(&up) {
        *value = mul(gelu(*value), *up_value);
    }

    let mut ff = matvec(&gate, &unpack_hex(&params.w_down_hex)?, hidden)?;
    rms_norm(
        &mut ff,
        &unpack_hex(&params.norm_post_ffw_hex)?,
        params.norm_eps,
    )?;
    add_row(&mut ff, &residual)?;
    if params.layer_scalar != 0 {
        scale_row(&mut ff, params.layer_scalar);
    }
    Ok(ff)
}

// ---------------------------------------------------------------------------
// Failure accounting
// ---------------------------------------------------------------------------

/// Folds a list of recorded failures into a count plus the first message.
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

/// Fails the program when either pass dropped a row, so a short layer output is
/// never published as an authorized output.
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
