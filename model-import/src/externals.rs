//! Mirrors of every stage's committed external type.
//!
//! Each of these must match the stage's own definition **field-for-field**:
//! the chain authorizes an external by the structural commitment of its bytes,
//! so a renamed field or a reordered struct here produces an artifact the stage
//! will not accept. They are duplicated rather than shared for the same reason
//! the stages duplicate their boundary types — a chain member is a standalone
//! program, and its guest cannot depend on a host-side crate.

use raster::List;
use serde::{Deserialize, Serialize};

/// Number of hex chars one value occupies in a packed vector.
pub const HEX_CHARS_PER_VALUE: usize = 8;

/// Packs Q16.16 values into the hex leaf form every stage reads.
pub fn pack_hex(values: &[i32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = Vec::with_capacity(values.len() * HEX_CHARS_PER_VALUE);
    for value in values {
        let bits = *value as u32;
        for shift in (0..HEX_CHARS_PER_VALUE as u32).rev() {
            out.push(HEX[((bits >> (shift * 4)) & 0xf) as usize]);
        }
    }
    String::from_utf8(out).expect("hex packing is always ascii")
}

// --- prompt-prepare ---------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TokenEntry {
    pub token: String,
    pub id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BpeMerge {
    pub rank: u32,
    pub left: String,
    pub right: String,
    pub merged: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PromptTokenizer {
    pub vocab: List<TokenEntry>,
    pub merges: List<BpeMerge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BpePieces {
    pub pieces: List<String>,
}

// --- input-embedding --------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct EmbeddingRow {
    pub token_id: u32,
    pub values_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct EmbeddingTable {
    pub hidden_size: u32,
    pub rows: List<EmbeddingRow>,
}

// --- prefill-prepare-aux ----------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleEmbeddingRow {
    pub token_id: u32,
    pub values_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleLayerParams {
    pub layer_idx: u32,
    pub hidden_size: u32,
    pub ple_width: u32,
    pub embedding_scale: i32,
    pub projection_scalar: i32,
    pub input_scale: i32,
    pub norm_eps: i64,
    pub projection_hex: String,
    pub norm_weights_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleLayer {
    pub params: PleLayerParams,
    pub embedding_rows: List<PleEmbeddingRow>,
}

// --- prefill-range ----------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct LayerParams {
    pub layer_idx: u32,
    pub hidden_size: u32,
    pub ffn_size: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub sliding_window: u32,
    pub attn_scale: i32,
    pub layer_scalar: i32,
    pub norm_eps: i64,
    pub norm_input_hex: String,
    pub norm_post_attn_hex: String,
    pub norm_pre_ffw_hex: String,
    pub norm_post_ffw_hex: String,
    pub q_norm_hex: String,
    pub k_norm_hex: String,
    pub w_q_hex: String,
    pub w_k_hex: String,
    pub w_v_hex: String,
    pub w_o_hex: String,
    pub w_gate_hex: String,
    pub w_up_hex: String,
    pub w_down_hex: String,
}

// --- prefill-finalize -------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct LogitRow {
    pub token_id: u32,
    pub values_hex: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalHeadParams {
    pub hidden_size: u32,
    pub norm_eps: i64,
    pub softcap: i32,
    pub norm_weights_hex: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalHead {
    pub params: FinalHeadParams,
    pub rows: List<LogitRow>,
}
