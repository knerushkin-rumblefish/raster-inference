//! Phase 3 — `prefill-prepare-aux`: tiles.
//!
//! Gemma's per-layer-embedding (PLE) prefill inputs, for **one** layer:
//!
//! ```text
//!   embedded  = ple_embedding[token_id] * embedding_scale
//!   projected = rms_norm(activation × projection * projection_scalar, w, eps)
//!   row       = (embedded +sat projected) * input_scale
//! ```
//!
//! The native routine loops layers × tokens. Only the token loop lives here:
//! nested recur cannot accumulate (the inner loop finalizes its draft, and a
//! `List<List<T>>` cannot be drafted at all), so the layer dimension is
//! expanded by the chain — one stage instance per PLE layer, each bound to its
//! own committed layer external. That is the same static expansion the
//! pipeline plan uses for decode steps.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use alloc::string::String;
use raster::prelude::*;

pub mod input;

// The glob also brings the derive-generated `…DraftExt` traits into scope.
use input::*;

/// Seeds the output draft with the layer it belongs to.
///
/// `layer_idx` is a set-once scalar, and the loop that follows never knows it
/// is on the first token, so it is written here — before the recur — and the
/// draft is threaded in as that loop's `output`.
#[tile(kind = iter, description = "Open this layer's prefill-input draft")]
pub fn begin_ple_layer(
    output: Draft<PleLayerInputs>,
    params: PleLayerParams,
) -> Draft<PleLayerInputs> {
    let mut output = output;
    output.layer_idx().set(params.layer_idx);
    output
}

/// Materializes one embedded prompt position into a task.
#[tile(kind = iter, description = "Turn an embedded token into a PLE task")]
pub fn begin_ple_task(token: ActivationRow) -> PleTask {
    PleTask {
        token_id: token.token_id,
        activation_hex: token.values_hex,
    }
}

/// Scans one `Block` of this layer's PLE table for the task's token id.
///
/// Same gather as `input-embedding`, and the same limit behind it: `select!`
/// takes literal indexes only, so a row cannot be addressed by a runtime token
/// id and the table has to be walked.
#[tile(kind = recur, description = "Scan a chunk of the PLE table for one token")]
pub fn scan_ple_embeddings(
    input: RecurInput<Block<PleEmbeddingRow>>,
    state: RecurState<PleEmbeddingMatch>,
    task: PleTask,
    ple_width: u32,
) -> RecurState<PleEmbeddingMatch> {
    let mut state = state;
    let expected_len = (ple_width * HEX_CHARS_PER_VALUE) as usize;

    for row in input.into_value() {
        let usable = row.token_id == task.token_id && row.values_hex.len() == expected_len;
        if !state.found && usable {
            state.found = true;
            state.values_hex = row.values_hex;
        }
    }

    state
}

/// Projects one activation row through this layer's PLE projection and
/// normalises it: `rms_norm(activation × projection * projection_scalar)`.
///
/// This is the arithmetic half of the row. It is its own replay unit because
/// the matvec dominates the cost — keeping it apart from the combine step
/// means a fraud proof over the projection does not have to re-run the rest.
#[tile(
    kind = iter,
    description = "Project and normalise one activation row",
    estimated_cycles = 20000
)]
pub fn project_activation_row(task: PleTask, params: PleLayerParams) -> ProjectedRow {
    match project_row(&task, &params) {
        Ok(values) => ProjectedRow {
            values_hex: pack_hex(&values),
            error: String::new(),
        },
        // Failure travels as data, not as a panic: see `ProjectedRow`.
        Err(error) => ProjectedRow {
            values_hex: String::new(),
            error,
        },
    }
}

fn project_row(
    task: &PleTask,
    params: &PleLayerParams,
) -> core::result::Result<alloc::vec::Vec<i32>, String> {
    let activation = unpack_hex(&task.activation_hex)?;
    if activation.len() != params.hidden_size as usize {
        return Err(alloc::format!(
            "activation row for token {} has {} values, expected hidden_size {}",
            task.token_id,
            activation.len(),
            params.hidden_size
        ));
    }

    let projection = unpack_hex(&params.projection_hex)?;
    let norm_weights = unpack_hex(&params.norm_weights_hex)?;

    let mut projected = matvec(&activation, &projection, params.ple_width as usize)?;
    scale_row(&mut projected, params.projection_scalar);
    rms_norm(&mut projected, &norm_weights, params.norm_eps)?;
    Ok(projected)
}

/// Combines the scaled PLE embedding with the projected row and appends the
/// result: `(embedded +sat projected) * input_scale`.
///
/// A token with no PLE row is recorded instead of appended, so the miss is
/// visible to the check at the end of the sequence rather than silently
/// producing a wrong-length layer input.
#[tile(kind = iter, description = "Combine the PLE embedding with the projected row")]
pub fn combine_ple_row(
    output: Draft<PleLayerInputs>,
    task: PleTask,
    hit: PleEmbeddingMatch,
    projected: ProjectedRow,
    params: PleLayerParams,
) -> Draft<PleLayerInputs> {
    let mut output = output;
    match combine_row(&task, &hit, &projected, &params) {
        Ok(values) => output.rows().push(pack_hex(&values)),
        Err(error) => output.errors().push(error),
    }
    output
}

fn combine_row(
    task: &PleTask,
    hit: &PleEmbeddingMatch,
    projected: &ProjectedRow,
    params: &PleLayerParams,
) -> core::result::Result<alloc::vec::Vec<i32>, String> {
    if !projected.error.is_empty() {
        return Err(projected.error.clone());
    }
    if !hit.found {
        return Err(alloc::format!(
            "token {} has no PLE embedding row in layer {}",
            task.token_id,
            params.layer_idx
        ));
    }

    let mut embedded = unpack_hex(&hit.values_hex)?;
    let projected = unpack_hex(&projected.values_hex)?;
    if embedded.len() != projected.len() {
        return Err(alloc::format!(
            "PLE row width mismatch for token {}: {} embedded vs {} projected",
            task.token_id,
            embedded.len(),
            projected.len()
        ));
    }

    scale_row(&mut embedded, params.embedding_scale);
    for (value, projected_value) in embedded.iter_mut().zip(projected) {
        *value = add_sat(*value, projected_value);
    }
    scale_row(&mut embedded, params.input_scale);
    Ok(embedded)
}

/// Folds the recorded failures into a count plus the first message.
#[tile(kind = recur, description = "Summarise the rows this layer could not produce")]
pub fn summarise_layer_errors(
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

/// Fails the program when any row failed, so a short or partial layer input is
/// never published as an authorized output.
#[tile(kind = iter, description = "Reject an incomplete layer input")]
pub fn assert_layer_complete(error_count: u32, first_error: String) -> Result<u32> {
    if error_count == 0 {
        Ok(error_count)
    } else {
        Err(alloc::format!(
            "{error_count} PLE row(s) failed; first: {first_error}"
        ))
    }
}

/// Guards the layer external before any of it is used.
///
/// The descriptor arrives committed, but "committed" only means the bytes are
/// the ones the manifest names — it says nothing about them being coherent, so
/// the widths are checked once, in a tile, rather than trusted per row.
#[tile(kind = iter, description = "Validate this layer's declared widths")]
pub fn validate_layer_params(params: PleLayerParams) -> Result<PleLayerParams> {
    if params.hidden_size == 0 || params.ple_width == 0 {
        return Err(String::from(
            "PLE layer requires non-zero hidden_size and ple_width",
        ));
    }
    let expected_projection =
        (params.ple_width * params.hidden_size * HEX_CHARS_PER_VALUE) as usize;
    if params.projection_hex.len() != expected_projection {
        return Err(alloc::format!(
            "layer {} projection has {} hex chars, expected {} ({} x {} values)",
            params.layer_idx,
            params.projection_hex.len(),
            expected_projection,
            params.ple_width,
            params.hidden_size
        ));
    }
    let expected_norm = (params.ple_width * HEX_CHARS_PER_VALUE) as usize;
    if params.norm_weights_hex.len() != expected_norm {
        return Err(alloc::format!(
            "layer {} norm weights have {} hex chars, expected {}",
            params.layer_idx,
            params.norm_weights_hex.len(),
            expected_norm
        ));
    }
    if params.norm_eps < 0 {
        return Err(String::from("PLE rms-norm epsilon must be non-negative"));
    }
    Ok(params)
}
