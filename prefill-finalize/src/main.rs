//! Phase 5 — `prefill-finalize`: sequences.
//!
//! ```text
//!   rows       ──recur──▶ track final position ──▶ normalise
//!   projection ──recur──▶ score chunk                ──▶ PrefillLogits
//!   errors     ──recur──▶ summarise ──▶ assert none
//! ```
//!
//! Every step here is a plain `call_recur!` over a real collection: the prompt
//! rows, then the vocabulary. No nesting, no second collection per item — the
//! only value that rides along is the normalised final position, which is one
//! row wide.

use raster::prelude::*;

use prefill_finalize::input::*;
use prefill_finalize::*;

/// Phase 5 entrypoint.
///
/// `activations` is bound by the chain to the last transformer layer's
/// authorized output; `head` is the committed output head. The return value is
/// the prompt's next-token scores, which is what a decode stage consumes.
#[sequence]
fn main(activations: ActivationSequence, head: FinalHead) -> Result<PrefillLogits> {
    let params = select!(FinalHeadParams, head.clone().params);
    let params = call!(validate_final_head, params)?;

    // Fold the prompt down to its last row. `select!` cannot index "last", so
    // the fold overwrites its state until the loop ends; the token count it
    // accumulates on the way is the decode position.
    let rows = select!(List<ActivationRow>, activations.rows);
    let position = call_recur!(
        tile = track_final_position,
        input = rows,
        state = FinalPosition {
            found: false,
            token_id: 0,
            token_count: 0,
            values_hex: String::new()
        },
        args = ()
    );

    let normalized = call!(normalize_final_position, position.clone(), params.clone())?;

    // `decode_position` is set-once, so the draft is opened before the loop and
    // threaded in as its output.
    let draft = call!(begin_logits, new!(PrefillLogits), position);

    let projection = select!(List<LogitRow>, head.rows);
    let logits = call_recur!(
        tile = project_logit_chunk,
        input = projection,
        chunk = 4,
        output = draft,
        args = (normalized, params)
    );
    raster::println!("prefill finalize → {:?}", logits);

    let errors = select!(List<String>, logits.clone().errors);
    let summary = call_recur!(
        tile = summarise_errors,
        input = errors,
        state = ErrorSummary {
            count: 0,
            first: String::new()
        },
        args = ()
    );
    let error_count = select!(u32, summary.clone().count);
    let first_error = select!(String, summary.first);
    call!(assert_logits_complete, error_count, first_error)?;

    Ok(logits)
}
