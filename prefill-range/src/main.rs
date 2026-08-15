//! Phase 4 — `prefill-range`: sequences.
//!
//! ```text
//!   rows    ──recur seq──▶ [ begin · mac Q · mac K · mac V · finish ] ──▶ KvSequence
//!   queries ──recur seq──▶ [ attend keys · mac O · mlp matvecs     ] ──▶ ActivationSequence
//!   errors  ──recur─────▶ summarise ──▶ assert none
//! ```

use raster::prelude::*;

use prefill_range::input::*;
use prefill_range::*;

/// One prompt position: RMS-norm, project Q/K/V by paging each matrix, stamp
/// the loop-carried position, append key and query rows.
#[sequence(kind = recur)]
fn project_token(
    input: RecurSequenceInput<ActivationRow>,
    state: RecurSequenceState<u32>,
    output: RecurSequenceOutput<KvSequence>,
    params: LayerParams,
    w_q: List<BytesPage>,
    w_k: List<BytesPage>,
    w_v: List<BytesPage>,
) -> (
    RecurSequenceState<u32>,
    RecurSequenceOutput<KvSequence>,
) {
    let prep = call!(begin_qkv, input, params.clone());
    let hidden = call!(hidden_of, params.clone());
    let q_len = call!(q_out_len, params.clone());
    let kv_len = call!(kv_out_len, params.clone());
    let normed = call!(qkv_normed, prep.clone());
    let acc_q = call!(zero_accum, q_len);
    let q = call_recur!(
        tile = mac_weight_page,
        input = w_q,
        state = acc_q,
        args = (normed.clone(), hidden.clone())
    );
    let acc_k = call!(zero_accum, kv_len.clone());
    let k = call_recur!(
        tile = mac_weight_page,
        input = w_k,
        state = acc_k,
        args = (normed.clone(), hidden.clone())
    );
    let acc_v = call!(zero_accum, kv_len);
    let v = call_recur!(
        tile = mac_weight_page,
        input = w_v,
        state = acc_v,
        args = (normed, hidden)
    );
    let output = call!(finish_qkv, output, state.clone(), prep, q, k, v, params);
    let state = call!(advance_position, state);
    (state, output)
}

/// One query: attend over keys, project W_o, run the MLP matvecs, append the
/// layer output row.
#[sequence(kind = recur)]
fn attend_token(
    input: RecurSequenceInput<QueryRow>,
    output: RecurSequenceOutput<ActivationSequence>,
    keys: List<KeyRow>,
    donor_keys: List<KeyRow>,
    params: LayerParams,
    w_o: List<BytesPage>,
    w_gate: List<BytesPage>,
    w_up: List<BytesPage>,
    w_down: List<BytesPage>,
) -> RecurSequenceOutput<ActivationSequence> {
    let query = call!(begin_attention, input);
    let attn_state = call!(zero_attn_state);
    // Both key sets are swept and exactly one contributes — `attend_kv_chunk`
    // drops every row whose pass does not match `kv_donor_layer`. A sequence
    // cannot branch, and chaining the state through two sweeps is equivalent to
    // one sweep over whichever list applies.
    let attended = call_recur!(
        tile = attend_kv_chunk,
        input = keys.clone(),
        state = attn_state,
        args = (query.clone(), params.clone(), false)
    );
    let attended = call_recur!(
        tile = attend_kv_chunk,
        input = donor_keys,
        state = attended,
        args = (query.clone(), params.clone(), true)
    );
    let context = call!(softmax_context, attended, params.clone());
    let context_values = call!(packed_values, context.clone());
    let q_len = call!(q_out_len, params.clone());
    let hidden = call!(hidden_of, params.clone());
    let acc_o = call!(zero_accum, hidden.clone());
    let attn_proj = call_recur!(
        tile = mac_weight_page,
        input = w_o,
        state = acc_o,
        args = (context_values, q_len)
    );
    let residual = call!(
        attn_residual,
        query.clone(),
        context,
        attn_proj,
        params.clone()
    );
    let ff_in = call!(pre_ff_norm, residual.clone(), params.clone());
    let ff_in_values = call!(packed_values, ff_in.clone());
    let ffn = call!(ffn_of, params.clone());
    let acc_gate = call!(zero_accum, ffn.clone());
    let gate = call_recur!(
        tile = mac_weight_page,
        input = w_gate,
        state = acc_gate,
        args = (ff_in_values.clone(), hidden.clone())
    );
    let acc_up = call!(zero_accum, ffn.clone());
    let up = call_recur!(
        tile = mac_weight_page,
        input = w_up,
        state = acc_up,
        args = (ff_in_values, hidden.clone())
    );
    let gated = call!(gelu_mul, gate, up);
    let gated_values = call!(packed_values, gated);
    let acc_down = call!(zero_accum, hidden);
    let ff = call_recur!(
        tile = mac_weight_page,
        input = w_down,
        state = acc_down,
        args = (gated_values, ffn)
    );
    // This token's own cache entry, fetched by an authorized dynamic index:
    // `keys` is built in prompt order by pass 1, so it is index-aligned with
    // the query's position.
    let position = select!(u32, query.clone().position);
    let own_key = select!(KeyRow, keys[position]);
    call!(
        finish_mlp,
        output,
        query,
        residual,
        ff_in,
        ff,
        params,
        own_key
    )
}

/// Phase 4 entrypoint, for one transformer layer.
#[sequence]
fn main(
    activations: ActivationSequence,
    layer: TransformerLayer,
    donor_kv: ActivationSequence,
) -> Result<ActivationSequence> {
    let params = select!(LayerParams, layer.clone().params);
    let params = call!(validate_layer_params, params)?;

    let hidden = call!(hidden_of, params.clone());
    let q_len = call!(q_out_len, params.clone());
    let kv_len = call!(kv_out_len, params.clone());
    let ffn = call!(ffn_of, params.clone());

    let w_q_len = select!(u64, layer.clone().w_q.byte_len);
    call!(assert_matrix_bytes, w_q_len, q_len.clone(), hidden.clone())?;
    let w_k_len = select!(u64, layer.clone().w_k.byte_len);
    call!(assert_matrix_bytes, w_k_len, kv_len.clone(), hidden.clone())?;
    let w_v_len = select!(u64, layer.clone().w_v.byte_len);
    call!(assert_matrix_bytes, w_v_len, kv_len, hidden.clone())?;
    let w_o_len = select!(u64, layer.clone().w_o.byte_len);
    call!(assert_matrix_bytes, w_o_len, hidden.clone(), q_len)?;
    let w_gate_len = select!(u64, layer.clone().w_gate.byte_len);
    call!(assert_matrix_bytes, w_gate_len, ffn.clone(), hidden.clone())?;
    let w_up_len = select!(u64, layer.clone().w_up.byte_len);
    call!(assert_matrix_bytes, w_up_len, ffn.clone(), hidden.clone())?;
    let w_down_len = select!(u64, layer.clone().w_down.byte_len);
    call!(assert_matrix_bytes, w_down_len, hidden, ffn)?;

    let w_q = select!(List<BytesPage>, layer.clone().w_q.pages);
    let w_k = select!(List<BytesPage>, layer.clone().w_k.pages);
    let w_v = select!(List<BytesPage>, layer.clone().w_v.pages);
    let w_o = select!(List<BytesPage>, layer.clone().w_o.pages);
    let w_gate = select!(List<BytesPage>, layer.clone().w_gate.pages);
    let w_up = select!(List<BytesPage>, layer.clone().w_up.pages);
    let w_down = select!(List<BytesPage>, layer.clone().w_down.pages);

    let rows = select!(List<ActivationRow>, activations.rows);
    let pos0 = call!(zero_u32);
    let kv = call_recur_seq!(
        sequence = project_token,
        input = rows,
        state = pos0,
        output = new!(KvSequence),
        args = (params.clone(), w_q, w_k, w_v)
    );
    let keys = select!(List<KeyRow>, kv.clone().keys);
    let queries = select!(List<QueryRow>, kv.clone().queries);
    // The donor's published cache. Bound to `input_embedding` (an empty `kv`)
    // for every layer that computes its own, so the binding is uniform across
    // all 35 stages of this one program.
    let donor_keys = select!(List<KeyRow>, donor_kv.kv);

    let prepared = call_recur_seq!(
        sequence = attend_token,
        input = queries,
        output = new!(ActivationSequence),
        args = (keys, donor_keys, params, w_o, w_gate, w_up, w_down)
    );
    raster::println!("prefill range pass → {:?}", prepared);

    let projection_errors = select!(List<String>, kv.errors);
    let projection_summary = call_recur!(
        tile = summarise_errors,
        input = projection_errors,
        state = ErrorSummary {
            count: 0,
            first: String::new()
        },
        args = ()
    );
    let attention_errors = select!(List<String>, prepared.clone().errors);
    let attention_summary = call_recur!(
        tile = summarise_errors,
        input = attention_errors,
        state = ErrorSummary {
            count: 0,
            first: String::new()
        },
        args = ()
    );

    let projection_count = select!(u32, projection_summary.clone().count);
    let first_projection_error = select!(String, projection_summary.first);
    let attention_count = select!(u32, attention_summary.clone().count);
    let first_attention_error = select!(String, attention_summary.first);
    call!(
        assert_layer_complete,
        projection_count,
        first_projection_error,
        attention_count,
        first_attention_error
    )?;

    Ok(prepared)
}
