//! Shared Rastered data types for the `prompt-prepare` stage.
//!
//! Everything that crosses a tile boundary or is reached by `select!` lives
//! here, in the `no_std` library, so host and RISC0 guest see the same
//! definitions.
//!
//! Two vocabulary rules shape every type below:
//!
//! - collections are `List<T>` (never `Vec<T>`): `Selectable`, referenced,
//!   iterated by `call_recur!` — never materialized whole;
//! - a type a tile takes or returns must be `Materializable`, which the
//!   `Selectable` derive grants only to structs with no `List` field. That is
//!   why the "work" types here ([`MergeStep`], [`MergeMatch`], [`VocabQuery`],
//!   [`VocabMatch`], [`MergeCursor`]) are small and scalar, while the
//!   collection-bearing types ([`PromptTokenizer`], [`BpePieces`],
//!   [`MergedPieces`], [`PromptTokenization`]) are only ever selected into,
//!   iterated, or grown through a draft.

extern crate alloc;

use alloc::string::String;
use raster::List;
use serde::{Deserialize, Serialize};

/// Token id reserved for pieces that are not in the vocabulary.
pub const UNK_TOKEN_ID: u32 = 0;

/// One vocabulary entry: a token string and the id it maps to.
///
/// All-scalar, so it is `Materializable`: a tile may take one whole, and a
/// `chunk = N` recur step may take a `Block` of them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TokenEntry {
    pub token: String,
    pub id: u32,
}

/// One BPE merge rule: `left + right -> merged`, ranked by `rank`
/// (lower rank wins when several rules match the same pair).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BpeMerge {
    pub rank: u32,
    pub left: String,
    pub right: String,
    pub merged: String,
}

/// Committed tokenizer fixture: the vocabulary and the merge table.
///
/// Both fields are `List`s, so this struct is `Selectable` but **not**
/// `Materializable` — `main` selects the two lists out of it and hands each to
/// the recur that iterates it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PromptTokenizer {
    pub vocab: List<TokenEntry>,
    pub merges: List<BpeMerge>,
}

/// The committed prompt, pre-split into initial pieces.
///
/// The list MUST end with a terminator piece that appears in no merge rule
/// (the end-of-word marker, see `README.md`). The merge pass is a single
/// left-to-right pass, so the terminator is what flushes the last pending
/// token; without it the final token would stay in the loop-carried cursor,
/// which the pass has no way to append after the last iteration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BpePieces {
    pub pieces: List<String>,
}

/// Result of the merge pass: the pieces after greedy left-to-right merging.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct MergedPieces {
    pub pieces: List<String>,
}

/// Loop-carried cursor of the merge pass: the token being accumulated.
///
/// Recur state is re-committed on **every** iteration, so it stays scalar —
/// what grows is the draft in `output`, never this.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct MergeCursor {
    pub pending: String,
    pub has_pending: bool,
}

/// One merge decision in flight: the cursor's pending token paired with the
/// incoming piece. This is the small scalar item the merge-table scan reads
/// on every iteration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct MergeStep {
    pub pending: String,
    pub has_pending: bool,
    pub piece: String,
}

/// Outcome of scanning the merge table for a [`MergeStep`]'s pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct MergeMatch {
    pub matched: bool,
    pub rank: u32,
    pub merged: String,
}

/// One vocabulary lookup: the merged piece whose token id we need.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct VocabQuery {
    pub piece: String,
}

/// Outcome of scanning the vocabulary for a [`VocabQuery`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct VocabMatch {
    pub found: bool,
    pub token_id: u32,
}

/// The stage's authorized output: the prompt as token ids.
///
/// The field layout MUST match the consuming stage's copy of this type — the
/// chain links stages by structural commitment, not by Rust type name.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PromptTokenization {
    pub token_ids: List<u32>,
}
