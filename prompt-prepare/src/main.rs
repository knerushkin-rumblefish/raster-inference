//! Phase 1 — `prompt-prepare`: sequences.
//!
//! Orchestration only: these bodies name steps, select data, and bind results.
//! Both passes are recur sequences, because each element needs several tiles
//! *and* a scan of a second collection:
//!
//! ```text
//!   pieces ──recur seq──▶ [ pair · scan merge table · emit · advance ] ──▶ MergedPieces
//!   merged ──recur seq──▶ [ query · scan vocabulary · append        ] ──▶ PromptTokenization
//! ```
//!
//! The second collection (the merge table, the vocabulary) travels as a
//! recur-sequence arg: those are `AuthRef`s, never materialized, and each inner
//! `call_recur!` walks one of them a `Block` at a time with the current item as
//! a scalar arg.

use raster::prelude::*;

use prompt_prepare::input::*;
use prompt_prepare::*;

/// One prompt piece: pair it with the pending token, look the pair up in the
/// merge table, emit the finished token if the merge stopped, and carry the
/// result to the next piece.
#[sequence(kind = recur)]
fn merge_prompt_piece(
    input: RecurSequenceInput<String>,
    state: RecurSequenceState<MergeCursor>,
    output: RecurSequenceOutput<MergedPieces>,
    merges: List<BpeMerge>,
) -> (
    RecurSequenceState<MergeCursor>,
    RecurSequenceOutput<MergedPieces>,
) {
    let step = call!(begin_merge_step, state, input);

    let hit = call_recur!(
        tile = scan_merge_rules,
        input = merges,
        chunk = 4,
        state = MergeMatch {
            matched: false,
            rank: 0,
            merged: String::new()
        },
        args = (step.clone(),)
    );

    let output = call!(emit_merged_piece, output, step.clone(), hit.clone());
    let state = call!(advance_merge_cursor, step, hit);

    (state, output)
}

/// One merged piece: resolve it against the vocabulary and append its id.
#[sequence(kind = recur)]
fn resolve_piece_token(
    input: RecurSequenceInput<String>,
    output: RecurSequenceOutput<PromptTokenization>,
    vocab: List<TokenEntry>,
) -> RecurSequenceOutput<PromptTokenization> {
    let query = call!(begin_vocab_lookup, input);

    let hit = call_recur!(
        tile = scan_vocab_chunk,
        input = vocab,
        chunk = 4,
        state = VocabMatch {
            found: false,
            token_id: UNK_TOKEN_ID
        },
        args = (query,)
    );

    call!(append_token_id, output, hit)
}

/// Phase 1 entrypoint.
///
/// `tokenizer` and `initial_pieces` are the committed entry arguments (bound by
/// name in `input.json` / `input_manifest.json`, and by the root `Raster.toml`
/// when the stage runs as part of the chain). The return value is the
/// authorized output the next chain stage consumes.
#[sequence]
fn main(tokenizer: PromptTokenizer, initial_pieces: BpePieces) -> PromptTokenization {
    let merges = select!(List<BpeMerge>, tokenizer.clone().merges);
    let vocab = select!(List<TokenEntry>, tokenizer.vocab);
    let pieces = select!(List<String>, initial_pieces.pieces);

    let merged = call_recur_seq!(
        sequence = merge_prompt_piece,
        input = pieces,
        state = MergeCursor {
            pending: String::new(),
            has_pending: false
        },
        output = new!(MergedPieces),
        args = (merges,)
    );
    raster::println!("merge pass → {:?}", merged);

    let merged_pieces = select!(List<String>, merged.pieces);

    let tokenization = call_recur_seq!(
        sequence = resolve_piece_token,
        input = merged_pieces,
        output = new!(PromptTokenization),
        args = (vocab,)
    );
    raster::println!("vocab pass → {:?}", tokenization);

    tokenization
}
