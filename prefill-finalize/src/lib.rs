//! Phase 5 — `prefill-finalize`: tiles.
//!
//! The output head, following `det_kernels::det_hidden_to_logits`:
//!
//! ```text
//!   position = last row of the prompt's activations
//!   normed   = rms_norm(position, w_final)
//!   logit[t] = softcap(normed · projection[t])
//! ```
//!
//! Unlike the layer stages this one has no per-layer expansion — there is one
//! output head — and its loop is a single level: the projection is walked once,
//! one `Block` of vocabulary rows per replay unit, with the normalised position
//! as a scalar argument.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use alloc::string::String;
use raster::prelude::*;

pub mod input;

// The glob also brings the derive-generated `…DraftExt` traits into scope.
use input::*;

/// Guards the head external before any of it is used.
#[tile(kind = iter, description = "Validate the output head's declared shapes")]
pub fn validate_final_head(params: FinalHeadParams) -> Result<FinalHeadParams> {
    if params.hidden_size == 0 {
        return Err(String::from("output head requires a non-zero hidden_size"));
    }
    if params.norm_eps < 0 {
        return Err(String::from("rms-norm epsilon must be non-negative"));
    }
    if params.softcap < 0 {
        return Err(String::from(
            "final logit softcap must be non-negative (0 disables it)",
        ));
    }
    let expected = (params.hidden_size * HEX_CHARS_PER_VALUE) as usize;
    if params.norm_weights_hex.len() != expected {
        return Err(alloc::format!(
            "final norm weights have {} hex chars, expected {expected}",
            params.norm_weights_hex.len()
        ));
    }
    Ok(params)
}

/// Walks the prompt to its last position, counting as it goes.
///
/// A fold, because `select!` takes literal indexes only: "the last row" is not
/// addressable, so the state keeps overwriting until the loop ends. The count
/// it accumulates is the decode position.
#[tile(kind = recur, description = "Track the prompt's final position")]
pub fn track_final_position(
    input: RecurInput<ActivationRow>,
    state: RecurState<FinalPosition>,
) -> RecurState<FinalPosition> {
    let mut state = state;
    let row = input.into_value();
    state.found = true;
    state.token_id = row.token_id;
    state.token_count += 1;
    state.values_hex = row.values_hex;
    state
}

/// Applies the final RMS norm to the position every logit is scored against.
#[tile(kind = iter, description = "Normalise the final position")]
pub fn normalize_final_position(
    position: FinalPosition,
    params: FinalHeadParams,
) -> Result<NormalizedPosition> {
    if !position.found {
        return Err(String::from(
            "prefill produced no activation rows; there is no final position to score",
        ));
    }
    let mut values = unpack_hex(&position.values_hex)?;
    if values.len() != params.hidden_size as usize {
        return Err(alloc::format!(
            "final position has {} values, expected hidden_size {}",
            values.len(),
            params.hidden_size
        ));
    }
    rms_norm(
        &mut values,
        &unpack_hex(&params.norm_weights_hex)?,
        params.norm_eps,
    )?;
    Ok(NormalizedPosition {
        values_hex: pack_hex(&values),
    })
}

/// Seeds the output draft with the position decoding resumes from.
///
/// Set-once, so it is written before the loop and the draft is threaded in as
/// the loop's `output`.
#[tile(kind = iter, description = "Open the logits draft at the decode position")]
pub fn begin_logits(
    output: Draft<PrefillLogits>,
    position: FinalPosition,
) -> Draft<PrefillLogits> {
    let mut output = output;
    output.decode_position().set(position.token_count);
    output
}

/// Scores one `Block` of the vocabulary: a dot product per row, then softcap.
///
/// The projection is the largest collection in a model, and it is the recur
/// `input` here — one chunk per replay unit, with the normalised position
/// riding along as a small scalar argument.
#[tile(
    kind = recur,
    description = "Score one chunk of the vocabulary",
    estimated_cycles = 30000
)]
pub fn project_logit_chunk(
    input: RecurInput<Block<LogitRow>>,
    output: RecurOutput<PrefillLogits>,
    normalized: NormalizedPosition,
    params: FinalHeadParams,
) -> RecurOutput<PrefillLogits> {
    let mut output = output;
    let position = match unpack_hex(&normalized.values_hex) {
        Ok(values) => values,
        Err(error) => {
            output.errors().push(error);
            return output;
        }
    };

    for row in input.into_value() {
        match unpack_hex(&row.values_hex) {
            Ok(weights) => match dot(&position, &weights) {
                Ok(value) => output.logits().push(LogitEntry {
                    token_id: row.token_id,
                    value: softcap(value, params.softcap),
                }),
                Err(error) => output
                    .errors()
                    .push(alloc::format!("token {}: {error}", row.token_id)),
            },
            Err(error) => output
                .errors()
                .push(alloc::format!("token {}: {error}", row.token_id)),
        }
    }
    output
}

/// Folds the recorded failures into a count plus the first message.
#[tile(kind = recur, description = "Summarise the logits this head could not score")]
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

/// Fails the program when any vocabulary row went unscored: a logit vector
/// with holes would silently change which token wins.
#[tile(kind = iter, description = "Reject an incomplete logit vector")]
pub fn assert_logits_complete(error_count: u32, first_error: String) -> Result<u32> {
    if error_count == 0 {
        Ok(error_count)
    } else {
        Err(alloc::format!(
            "{error_count} vocabulary row(s) failed to score; first: {first_error}"
        ))
    }
}
