//! Phase 2 — `input-embedding`: sequences.
//!
//! ```text
//!   token_ids ──recur seq──▶ [ query · scan embedding rows · append ] ──▶ ActivationSequence
//!   misses    ──recur─────▶ count ──▶ assert none
//! ```
//!
//! The embedding table travels as a recur-sequence arg — an `AuthRef`, never
//! materialized — and the inner `call_recur!` walks it a `Block` at a time
//! with the current token id as a scalar arg.

use raster::prelude::*;

use input_embedding::input::*;
use input_embedding::*;

/// One prompt token: look its row up in the embedding table and append the
/// gathered activation.
#[sequence(kind = recur)]
fn embed_prompt_token(
    input: RecurSequenceInput<u32>,
    output: RecurSequenceOutput<ActivationSequence>,
    rows: List<EmbeddingRow>,
    hidden_size: u32,
) -> RecurSequenceOutput<ActivationSequence> {
    let query = call!(begin_row_lookup, input);

    let hit = call_recur!(
        tile = scan_embedding_rows,
        input = rows,
        chunk = 4,
        state = RowMatch {
            found: false,
            values_hex: String::new()
        },
        args = (query.clone(), hidden_size)
    );

    call!(append_activation_row, output, query, hit)
}

/// Phase 2 entrypoint.
///
/// `prompt` is bound by the chain to `prompt_prepare`'s authorized output;
/// `embedding` is a committed external. The return value is this stage's
/// authorized output.
#[sequence]
fn main(prompt: PromptTokenization, embedding: EmbeddingTable) -> Result<ActivationSequence> {
    let token_ids = select!(List<u32>, prompt.token_ids);
    let hidden_size = select!(u32, embedding.clone().hidden_size);
    let rows = select!(List<EmbeddingRow>, embedding.rows);

    let embedded = call_recur_seq!(
        sequence = embed_prompt_token,
        input = token_ids,
        output = new!(ActivationSequence),
        args = (rows, hidden_size)
    );
    raster::println!("embedding pass → {:?}", embedded);

    // A miss cannot fail the recur sequence itself (a recur sequence has no
    // fallible form), so the misses are collected, folded, and checked here.
    let errors = select!(List<String>, embedded.clone().errors);
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
    call!(assert_all_tokens_embedded, error_count, first_error)?;

    Ok(embedded)
}
