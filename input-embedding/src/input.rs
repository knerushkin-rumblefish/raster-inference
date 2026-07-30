//! Shared Rastered data types for the `input-embedding` stage.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use raster::List;
use serde::{Deserialize, Serialize};

/// The stage's committed first input, bound by the chain to `prompt_prepare`'s
/// authorized output.
///
/// The field layout MUST match `PromptTokenization` in `prompt-prepare`
/// field-for-field: the chain links the two stages by the structural
/// commitment of this value, not by Rust type name.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PromptTokenization {
    pub token_ids: List<u32>,
}

/// One embedding-table row: the token id it belongs to, and its values packed
/// into a single hex leaf.
///
/// The packing is what makes a row `Materializable`: a `List<i32>` field would
/// make the row a collection-bearing struct that can never cross a tile
/// boundary, and it would also put every scalar in its own index node — at
/// model scale that is millions of nodes for data that is only ever read one
/// whole row at a time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct EmbeddingRow {
    pub token_id: u32,
    pub values_hex: String,
}

/// The committed embedding table.
///
/// `rows` is the collection this stage scans; `hidden_size` pins the width
/// every row must have.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct EmbeddingTable {
    pub hidden_size: u32,
    pub rows: List<EmbeddingRow>,
}

/// One row lookup in flight: the prompt token id whose row is wanted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct RowQuery {
    pub token_id: u32,
}

/// Outcome of scanning the embedding table for a [`RowQuery`].
///
/// This carries the matched row, so it is the one place in the program where
/// recur state is not scalar-small: a scan has nowhere else to put its answer
/// while it keeps looking. See `README.md` — a dynamic-index selection
/// primitive would replace the scan (and this state) with one proof.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct RowMatch {
    pub found: bool,
    pub values_hex: String,
}

/// Loop-carried summary of failed rows: how many, and the first message.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ErrorSummary {
    pub count: u32,
    pub first: String,
}

/// One prompt position: the token that produced it and its activation row,
/// packed exactly like [`EmbeddingRow::values_hex`].
///
/// The id travels *with* the row because this is the only point where both are
/// in hand. A downstream stage that keys on the token — Gemma's per-layer
/// embeddings, for instance — would otherwise get two parallel lists and no
/// way to walk them together: a recur iterates one collection, so index-aligned
/// pairs have to be zipped where they are produced.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationRow {
    pub token_id: u32,
    pub values_hex: String,
}

/// The stage's authorized output, and the pipeline's activation type: every
/// stage that passes activations on defines this field-for-field, so a layer's
/// output can feed the next layer's input.
///
/// `errors` records rows the stage could not produce. The sequence folds it and
/// fails the program if it is non-empty, so a successful run always carries one
/// row per prompt token.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationSequence {
    pub rows: List<ActivationRow>,
    pub errors: List<String>,
}

/// Number of hex chars one value occupies in a packed row.
pub const HEX_CHARS_PER_VALUE: u32 = 8;

/// Packs one row's values into the hex leaf form: 8 lowercase hex chars per
/// value, concatenated. The fixture generator and any consumer must agree on
/// this encoding byte-for-byte.
pub fn pack_row_hex(values: &[i32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = Vec::with_capacity(values.len() * HEX_CHARS_PER_VALUE as usize);
    for value in values {
        let bits = *value as u32;
        for shift in (0..HEX_CHARS_PER_VALUE).rev() {
            out.push(HEX[((bits >> (shift * 4)) & 0xf) as usize]);
        }
    }
    String::from_utf8(out).expect("hex packing is always ascii")
}
