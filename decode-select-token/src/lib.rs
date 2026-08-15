//! Phase 6 — `decode_select_token`: tiles.
//!
//! Greedy selection: the highest-scoring logit wins, ties going to the lowest
//! token id. That tie-break is not incidental — it is `argmax_first` in the
//! reference (`casettek/.../routines/decode_select_token/native/tiles.rs`),
//! where a candidate replaces the incumbent only on a *strictly greater* score.
//! Reproducing it exactly is what lets this stage's token be compared against
//! the reference run.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use raster::prelude::*;

pub mod input;

use input::*;

/// Folds one `Block` of logits into the running argmax.
///
/// The block is the recur `input` — the collection being iterated — and the
/// state is three scalars. A `chunk` keeps this at `vocab / chunk` replay units
/// instead of one per token id, without widening what any single unit touches.
#[tile(kind = recur, description = "Fold a chunk of logits into the running argmax")]
pub fn scan_logit_chunk(
    input: RecurInput<Block<LogitEntry>>,
    state: RecurState<ArgmaxState>,
) -> RecurState<ArgmaxState> {
    let mut state = state;
    for entry in input.into_value() {
        // Strictly greater: the first occurrence of a maximum keeps the slot.
        if !state.has_value || entry.value > state.value {
            state.has_value = true;
            state.token_id = entry.token_id;
            state.value = entry.value;
        }
    }
    state
}

/// Publishes the selection, refusing a run that scored no token at all.
///
/// An empty logit list would otherwise finalize to `token_id: 0` — a real token
/// — with nothing recording that it was never chosen. The reference bails on the
/// same condition ("output decode requires at least one logit").
#[tile(kind = iter, description = "Publish the selected token")]
pub fn finish_selection(state: ArgmaxState, decode_position: u32) -> Result<SelectedToken> {
    if !state.has_value {
        return Err(alloc::string::String::from(
            "output decode requires at least one logit to select the next token",
        ));
    }
    Ok(SelectedToken {
        decode_position,
        token_id: state.token_id,
        value: state.value,
    })
}
