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
    merge_buckets: List<MergeBucket>,
    merge_bucket_count: u32,
) -> (
    RecurSequenceState<MergeCursor>,
    RecurSequenceOutput<MergedPieces>,
) {
    let step = call!(begin_merge_step, state, input);

    // Hash the pair, then reach the one bucket it can be in. The index is an
    // authorized value, so this is a lookup the prover cannot steer; the scan
    // that follows walks a handful of rules instead of the whole table.
    let bucket_idx = call!(merge_bucket_index, step.clone(), merge_bucket_count);
    let bucket = select!(MergeBucket, merge_buckets[bucket_idx]);
    let rules = select!(List<BpeMerge>, bucket.rules);

    let hit = call_recur!(
        tile = scan_merge_rules,
        input = rules,
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
    vocab_buckets: List<VocabBucket>,
    vocab_bucket_count: u32,
) -> RecurSequenceOutput<PromptTokenization> {
    let query = call!(begin_vocab_lookup, input);

    let bucket_idx = call!(vocab_bucket_index, query.clone(), vocab_bucket_count);
    let bucket = select!(VocabBucket, vocab_buckets[bucket_idx]);
    let entries = select!(List<TokenEntry>, bucket.entries);

    let hit = call_recur!(
        tile = scan_vocab_chunk,
        input = entries,
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
    let merge_bucket_count = select!(u32, tokenizer.clone().merge_bucket_count);
    let vocab_bucket_count = select!(u32, tokenizer.clone().vocab_bucket_count);
    let merge_buckets = select!(List<MergeBucket>, tokenizer.clone().merge_buckets);
    let vocab_buckets = select!(List<VocabBucket>, tokenizer.vocab_buckets);
    let pieces = select!(List<String>, initial_pieces.pieces);

    let merged = call_recur_seq!(
        sequence = merge_prompt_piece,
        input = pieces,
        state = MergeCursor {
            pending: String::new(),
            has_pending: false
        },
        output = new!(MergedPieces),
        args = (merge_buckets, merge_bucket_count)
    );
    raster::println!("merge pass → {:?}", merged);

    let merged_pieces = select!(List<String>, merged.pieces);

    let tokenization = call_recur_seq!(
        sequence = resolve_piece_token,
        input = merged_pieces,
        output = new!(PromptTokenization),
        args = (vocab_buckets, vocab_bucket_count)
    );
    raster::println!("vocab pass → {:?}", tokenization);

    tokenization
}
