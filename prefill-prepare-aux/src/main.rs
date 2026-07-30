//! Phase 3 — `prefill-prepare-aux`: sequences.
//!
//! ```text
//!   tokens ──recur seq──▶ [ task · scan PLE table · project · combine ] ──▶ PleLayerInputs
//!   misses ──recur─────▶ count ──▶ assert none
//! ```
//!
//! One stage instance per PLE layer: the layer's parameters and PLE table are
//! its committed external, and the prompt activations come `from` the
//! `input_embedding` stage.

use raster::prelude::*;

use prefill_prepare_aux::input::*;
use prefill_prepare_aux::*;

/// One prompt position: gather its PLE embedding, project its activation row,
/// and append the combined layer input.
#[sequence(kind = recur)]
fn prepare_ple_row(
    input: RecurSequenceInput<ActivationRow>,
    output: RecurSequenceOutput<PleLayerInputs>,
    embedding_rows: List<PleEmbeddingRow>,
    params: PleLayerParams,
    ple_width: u32,
) -> RecurSequenceOutput<PleLayerInputs> {
    let task = call!(begin_ple_task, input);

    let hit = call_recur!(
        tile = scan_ple_embeddings,
        input = embedding_rows,
        chunk = 4,
        state = PleEmbeddingMatch {
            found: false,
            values_hex: String::new()
        },
        args = (task.clone(), ple_width)
    );

    let projected = call!(project_activation_row, task.clone(), params.clone());

    call!(combine_ple_row, output, task, hit, projected, params)
}

/// Phase 3 entrypoint, for one PLE layer.
///
/// `embedded` is bound by the chain to `input_embedding`'s authorized output;
/// `layer` is this stage instance's committed external. The return value is
/// this layer's prefill input rows.
#[sequence]
fn main(embedded: ActivationSequence, layer: PleLayer) -> Result<PleLayerInputs> {
    let params = select!(PleLayerParams, layer.clone().params);
    let params = call!(validate_layer_params, params)?;
    let ple_width = select!(u32, params.clone().ple_width);

    let tokens = select!(List<ActivationRow>, embedded.rows);
    let embedding_rows = select!(List<PleEmbeddingRow>, layer.embedding_rows);

    // `layer_idx` is set once, before the loop, and the draft is threaded in as
    // the loop's output — a recur step has no way to write a scalar field only
    // on the first iteration and stay set-once correct.
    let draft = call!(begin_ple_layer, new!(PleLayerInputs), params.clone());

    let prepared = call_recur_seq!(
        sequence = prepare_ple_row,
        input = tokens,
        output = draft,
        args = (embedding_rows, params, ple_width)
    );
    raster::println!("prefill aux pass → {:?}", prepared);

    // A failed row cannot fail the recur sequence itself, so the failures are
    // collected as data, folded, and checked here.
    let errors = select!(List<String>, prepared.clone().errors);
    let summary = call_recur!(
        tile = summarise_layer_errors,
        input = errors,
        state = ErrorSummary {
            count: 0,
            first: String::new()
        },
        args = ()
    );
    let error_count = select!(u32, summary.clone().count);
    let first_error = select!(String, summary.first);
    call!(assert_layer_complete, error_count, first_error)?;

    Ok(prepared)
}
