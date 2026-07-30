//! Phase 2 — `input-embedding`: tiles.
//!
//! The second stage of the `raster-chain-inference` chain: it gathers one
//! embedding row per prompt token id, in prompt order.
//!
//! The gather is a scan, not an index lookup. Raster's `select!` takes literal
//! indexes only, so "row number `token_id`" cannot be selected from a runtime
//! value; the stage therefore walks the embedding table and matches on
//! `row.token_id`. Every replay unit stays bounded (one `Block` of rows plus
//! one token id), at the cost of `prompt_tokens × table_rows` iterations. See
//! `README.md` for what a dynamic-index primitive would change.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use alloc::string::String;
use raster::prelude::*;

pub mod input;

// The glob also brings the derive-generated `…DraftExt` traits into scope.
use input::*;

/// Materializes one prompt token id into a lookup query.
///
/// The recur-sequence item handle can only be read once; binding it here gives
/// the two steps that follow a small value they can each read.
#[tile(kind = iter, description = "Turn a prompt token id into a row query")]
pub fn begin_row_lookup(token_id: u32) -> RowQuery {
    RowQuery { token_id }
}

/// Scans one `Block` of embedding rows for the queried token id.
///
/// A row only counts as found when its packed width matches `hidden_size`, so
/// a malformed table row is reported as a miss rather than silently embedding
/// a wrong-width activation.
#[tile(kind = recur, description = "Scan a chunk of embedding rows for one token id")]
pub fn scan_embedding_rows(
    input: RecurInput<Block<EmbeddingRow>>,
    state: RecurState<RowMatch>,
    query: RowQuery,
    hidden_size: u32,
) -> RecurState<RowMatch> {
    let mut state = state;
    let expected_len = (hidden_size * HEX_CHARS_PER_VALUE) as usize;

    for row in input.into_value() {
        let usable = row.token_id == query.token_id && row.values_hex.len() == expected_len;
        if !state.found && usable {
            state.found = true;
            state.values_hex = row.values_hex;
        }
    }

    state
}

/// Appends the gathered activation row, or records the token id as a miss.
///
/// The output grows by one entry per prompt token — the draft pays for the
/// increment instead of re-committing the whole activation list.
#[tile(kind = iter, description = "Append one gathered activation row")]
pub fn append_activation_row(
    output: Draft<ActivationSequence>,
    query: RowQuery,
    hit: RowMatch,
) -> Draft<ActivationSequence> {
    let mut output = output;
    if hit.found {
        output.rows().push(ActivationRow {
            token_id: query.token_id,
            values_hex: hit.values_hex,
        });
    } else {
        output.errors().push(alloc::format!(
            "token {} has no embedding row of the declared width",
            query.token_id
        ));
    }
    output
}

/// Folds the recorded failures into a count plus the first message.
#[tile(kind = recur, description = "Summarise the rows this stage could not produce")]
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

/// Fails the program when any prompt token went unembedded.
///
/// The protocol attests success only, so this is what keeps a partial gather
/// from being published as an authorized output.
#[tile(kind = iter, description = "Reject a run that could not embed every token")]
pub fn assert_all_tokens_embedded(error_count: u32, first_error: String) -> Result<u32> {
    if error_count == 0 {
        Ok(error_count)
    } else {
        Err(alloc::format!(
            "{error_count} prompt token(s) could not be embedded; first: {first_error}"
        ))
    }
}
