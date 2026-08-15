//! Shared Rastered data types for the `decode_select_token` stage.
//!
//! `PrefillLogits` and `LogitEntry` are duplicated field-for-field from
//! `prefill-finalize/src/input.rs`: the chain authorizes a stage input by the
//! structural commitment of its bytes, not by Rust type name, so a renamed or
//! reordered field here produces a value this stage will refuse.

extern crate alloc;

use alloc::string::String;
use raster::List;
use serde::{Deserialize, Serialize};

/// One scored token.
///
/// MUST match `LogitEntry` in `prefill-finalize`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct LogitEntry {
    pub token_id: u32,
    pub value: i32,
}

/// The committed input: `prefill_finalize`'s authorized output.
///
/// MUST match `PrefillLogits` in `prefill-finalize`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PrefillLogits {
    pub decode_position: u32,
    pub logits: List<LogitEntry>,
    pub errors: List<String>,
}

/// Loop-carried argmax.
///
/// Scalar-small, which is what recur `state` requires — it is re-committed on
/// every iteration. `has_value` distinguishes "no logit seen yet" from a real
/// score of zero, so the first candidate always wins outright rather than having
/// to beat a sentinel.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ArgmaxState {
    pub has_value: bool,
    pub token_id: u32,
    pub value: i32,
}

/// The stage's authorized output: the greedily selected next token.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct SelectedToken {
    pub decode_position: u32,
    pub token_id: u32,
    pub value: i32,
}
