//! Phase 6 — `decode_select_token`: sequences.
//!
//! ```text
//!   logits ──recur (chunk = 256)──▶ running argmax ──▶ SelectedToken
//! ```
//!
//! `chunk` is the only tuning knob here. The vocabulary is 262,144 entries, so
//! one replay unit per entry would be 262,144 of them to pick a maximum; a
//! `Block` of 256 makes it 1,024 units of ~2 KB each. The bound is a literal
//! pinned in the CFS, so coarsening this way costs nothing in what a verifier
//! can check.

use raster::prelude::*;

use decode_select_token::input::*;
use decode_select_token::*;

/// Phase 6 entrypoint.
///
/// `logits` is bound by the chain to `prefill_finalize`'s authorized output.
#[sequence]
fn main(logits: PrefillLogits) -> Result<SelectedToken> {
    let decode_position = select!(u32, logits.clone().decode_position);
    let scores = select!(List<LogitEntry>, logits.logits);

    let best = call_recur!(
        tile = scan_logit_chunk,
        input = scores,
        chunk = 256,
        state = ArgmaxState {
            has_value: false,
            token_id: 0,
            value: 0
        },
        args = ()
    );

    let selected = call!(finish_selection, best, decode_position)?;
    Ok(selected)
}
