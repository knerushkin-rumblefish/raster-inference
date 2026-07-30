//! Phase 4 — `prefill-range`: sequences.
//!
//! ```text
//!   rows   ──recur────▶ project Q/K/V                        ──▶ KvSequence
//!   kv     ──recur seq─▶ [ query · attend kv · finish · mlp ] ──▶ ActivationSequence
//!   errors ──recur────▶ summarise ──▶ assert none
//! ```
//!
//! The K/V sequence is built first and *then* iterated: attention reads every
//! token's keys, so the list has to be finished and storage-backed before the
//! attention pass can take it as a recur-sequence arg.
//!
//! One stage instance per layer. Its input and output are the same type, so
//! instances chain: `input_embedding → prefill_range_l0 → prefill_range_l1 → …`

use raster::prelude::*;

use prefill_range::input::*;
use prefill_range::*;

/// One prompt position: attend over the whole K/V sequence, then run the MLP
/// and append this layer's output row.
#[sequence(kind = recur)]
fn attend_token(
    input: RecurSequenceInput<KvRow>,
    output: RecurSequenceOutput<ActivationSequence>,
    kv_rows: List<KvRow>,
    params: LayerParams,
) -> RecurSequenceOutput<ActivationSequence> {
    let query = call!(begin_attention, input);

    let attended = call_recur!(
        tile = attend_kv_chunk,
        input = kv_rows,
        chunk = 4,
        state = AttnState {
            started: false,
            max_hex: String::new(),
            sum_hex: String::new(),
            acc_hex: String::new(),
            error: String::new()
        },
        args = (query.clone(), params.clone())
    );

    let attn = call!(finish_attention, query.clone(), attended, params.clone());

    call!(apply_mlp, output, query, attn, params)
}

/// Phase 4 entrypoint, for one transformer layer.
///
/// `activations` is bound by the chain to the previous stage's authorized
/// output — `input_embedding` for the first layer, the previous
/// `prefill_range` instance for the rest. `layer` is this instance's committed
/// weights.
#[sequence]
fn main(activations: ActivationSequence, layer: LayerParams) -> Result<ActivationSequence> {
    let params = call!(validate_layer_params, layer)?;
    let rows = select!(List<ActivationRow>, activations.rows);

    // Pass 1: a recur *tile*, so each step can read its own iteration index —
    // that index is the token's position, which the causal mask needs.
    let kv = call_recur!(
        tile = project_qkv_row,
        input = rows,
        output = new!(KvSequence),
        args = (params.clone(),)
    );
    let kv_rows = select!(List<KvRow>, kv.clone().rows);

    // Pass 2: the same list is both the thing iterated and the thing attended
    // over. Both uses are references — nothing is materialized twice.
    let prepared = call_recur_seq!(
        sequence = attend_token,
        input = kv_rows.clone(),
        output = new!(ActivationSequence),
        args = (kv_rows, params)
    );
    raster::println!("prefill range pass → {:?}", prepared);

    // Neither pass can fail the loop it runs in, so both record failures as
    // data; they are folded and checked here.
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
