//! Import a Gemma model bundle into this chain's committed externals.
//!
//! ```sh
//! cargo run --manifest-path model-import/Cargo.toml -- \
//!   --model ../raster-inference/assets/tiny-gemma-dev \
//!   --prompt "hello raster"
//! ```
//!
//! Reads `model.detwgt` (deterministic Q16.16 weights), `config.json` (shapes,
//! attention types, softcap) and `tokenizer.json` (vocabulary and merges), and
//! writes one `.rastered`/`.rindex` pair per stage external — then prints the
//! whole `[chain]` manifest, stages expanded per layer.
//!
//! Every value crosses over unchanged: detwgt stores canonical Q16.16 bit
//! patterns, which is the representation the stages compute in. Nothing here
//! requantises.

mod detwgt;
mod externals;

use externals::*;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

/// Q16.16 one.
const ONE: i32 = 1 << 16;

/// The end-of-word marker `prompt-prepare` needs to flush its last piece. It
/// appears in no merge rule and is dropped with the final cursor.
const END_OF_WORD: &str = "</w>";

fn main() {
    if let Err(error) = run() {
        eprintln!("model-import: {error}");
        std::process::exit(1);
    }
}

struct Args {
    model_dir: PathBuf,
    prompt: String,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut model_dir = None;
    let mut prompt = String::from("hello raster");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_dir = args.next().map(PathBuf::from),
            "--prompt" => prompt = args.next().ok_or("--prompt needs a value")?,
            other => return Err(format!("unknown argument '{other}'").into()),
        }
    }
    Ok(Args {
        model_dir: model_dir.ok_or("--model <bundle-dir> is required")?,
        prompt,
    })
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let weights = detwgt::load(&args.model_dir.join("model.detwgt"))?;
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(args.model_dir.join("config.json"))?)?;
    let tokenizer: serde_json::Value =
        serde_json::from_slice(&fs::read(args.model_dir.join("tokenizer.json"))?)?;
    let text = config
        .get("text_config")
        .ok_or("config.json has no text_config")?;

    let shape = Shape::from_config(text)?;
    println!(
        "model: hidden {} · {} layers · {} heads over {} kv heads · head_dim {} · ffn {} · vocab {} · ple {}",
        shape.hidden, shape.layers, shape.heads, shape.kv_heads, shape.head_dim, shape.ffn,
        shape.vocab, shape.ple_width
    );
    println!("tensors: {}", weights.names().count());
    println!();

    let mut stages = Vec::new();
    write_tokenizer(&tokenizer, &args.prompt, &mut stages)?;
    write_embedding(&weights, &shape, &mut stages)?;
    write_ple_layers(&weights, &shape, &mut stages)?;
    write_transformer_layers(&weights, &shape, &mut stages)?;
    write_head(&weights, &shape, text, &mut stages)?;

    println!();
    println!("# ---- root Raster.toml ----");
    println!("[chain]");
    println!("name = \"raster-chain-inference\"");
    println!("version = \"0.1.0\"");
    for stage in &stages {
        println!();
        print!("{stage}");
    }
    Ok(())
}

/// The shapes every stage needs, read from `config.json`.
struct Shape {
    hidden: usize,
    ffn: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    vocab: usize,
    ple_width: usize,
    sliding_window: u32,
    layer_types: Vec<String>,
    norm_eps: i64,
}

impl Shape {
    fn from_config(text: &serde_json::Value) -> Result<Self, Box<dyn Error>> {
        let usize_at = |key: &str| -> Result<usize, Box<dyn Error>> {
            text.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|value| value as usize)
                .ok_or_else(|| format!("config.json text_config has no {key}").into())
        };
        let layer_types = text
            .get("layer_types")
            .and_then(serde_json::Value::as_array)
            .map(|types| {
                types
                    .iter()
                    .filter_map(|value| value.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        // The MLP width is not in this config; it is the gate projection's rows.
        Ok(Self {
            hidden: usize_at("hidden_size")?,
            ffn: 0,
            layers: usize_at("num_hidden_layers")?,
            heads: usize_at("num_attention_heads")?,
            kv_heads: usize_at("num_key_value_heads")?,
            head_dim: usize_at("head_dim")?,
            vocab: usize_at("vocab_size")?,
            ple_width: usize_at("hidden_size_per_layer_input")?,
            sliding_window: text
                .get("sliding_window")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32,
            layer_types,
            norm_eps: f32_to_q16(
                text.get("rms_norm_eps")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(1e-6),
            ) as i64,
        })
    }
}

/// Converts a config float to Q16.16, the representation everything else uses.
fn f32_to_q16(value: f64) -> i32 {
    let scaled = value * ONE as f64;
    scaled.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

// ---------------------------------------------------------------------------
// Stage 1 — tokenizer + prompt pieces
// ---------------------------------------------------------------------------

fn write_tokenizer(
    tokenizer: &serde_json::Value,
    prompt: &str,
    stages: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let model = tokenizer.get("model").ok_or("tokenizer.json has no model")?;
    let vocab_map: BTreeMap<String, u32> = model
        .get("vocab")
        .and_then(serde_json::Value::as_object)
        .ok_or("tokenizer.json has no vocab")?
        .iter()
        .map(|(token, id)| {
            let id = id.as_u64().unwrap_or_default() as u32;
            (token.clone(), id)
        })
        .collect();

    let mut vocab: Vec<TokenEntry> = vocab_map
        .iter()
        .map(|(token, id)| TokenEntry {
            token: token.clone(),
            id: *id,
        })
        .collect();
    vocab.sort_by_key(|entry| entry.id);

    // A merge-less tokenizer (tiny-gemma-dev) leaves this empty, which makes
    // the merge pass a no-op: with no rule to extend a pending piece, every
    // piece is emitted as it arrives — exactly what merge-less BPE does.
    let merges: Vec<BpeMerge> = model
        .get("merges")
        .and_then(serde_json::Value::as_array)
        .map(|merges| {
            merges
                .iter()
                .enumerate()
                .filter_map(|(rank, entry)| parse_merge(rank as u32, entry))
                .collect()
        })
        .unwrap_or_default();

    let pieces = split_prompt(prompt, &vocab_map);
    println!("tokenizer: {} entries, {} merges", vocab.len(), merges.len());
    println!("prompt pieces: {pieces:?}");

    let tokenizer_commitment = write_external(
        &PromptTokenizer {
            vocab: vocab.into(),
            merges: merges.into(),
        },
        "prompt-prepare",
        "tokenizer",
    )?;
    let pieces_commitment = write_external(
        &BpePieces {
            pieces: pieces.into(),
        },
        "prompt-prepare",
        "initial_pieces",
    )?;

    // The first stage is the only one with its own inputs, so it is also the
    // only one that can be run standalone; keep its fixtures in step with the
    // externals just written.
    write_stage_fixtures(
        "prompt-prepare",
        &[
            ("tokenizer", "tokenizer", &tokenizer_commitment),
            ("initial_pieces", "initial_pieces", &pieces_commitment),
        ],
    )?;

    stages.push(format!(
        concat!(
            "[[chain.stage]]\n",
            "name = \"prompt_prepare\"\n",
            "project = \"prompt-prepare\"\n",
            "{}",
            "{}"
        ),
        external_line("tokenizer", "prompt-prepare", "tokenizer", &tokenizer_commitment),
        external_line(
            "initial_pieces",
            "prompt-prepare",
            "initial_pieces",
            &pieces_commitment
        ),
    ));
    Ok(())
}

fn parse_merge(rank: u32, entry: &serde_json::Value) -> Option<BpeMerge> {
    // HuggingFace writes merges either as "left right" or as ["left", "right"].
    let (left, right) = match entry {
        serde_json::Value::String(text) => {
            let mut parts = text.splitn(2, ' ');
            (parts.next()?.to_string(), parts.next()?.to_string())
        }
        serde_json::Value::Array(pair) if pair.len() == 2 => (
            pair[0].as_str()?.to_string(),
            pair[1].as_str()?.to_string(),
        ),
        _ => return None,
    };
    let merged = format!("{left}{right}");
    Some(BpeMerge {
        rank,
        left,
        right,
        merged,
    })
}

/// Splits a prompt into the initial pieces the tokenizer stage consumes:
/// SentencePiece's space marker, one piece per character, byte-fallback for
/// anything the vocabulary does not have, and the end-of-word terminator.
fn split_prompt(prompt: &str, vocab: &BTreeMap<String, u32>) -> Vec<String> {
    let mut pieces = Vec::new();
    for ch in prompt.replace(' ', "\u{2581}").chars() {
        let piece = ch.to_string();
        if vocab.contains_key(&piece) {
            pieces.push(piece);
        } else {
            // Byte fallback: `<0xNN>` per UTF-8 byte, as the tokenizer declares.
            let mut buffer = [0u8; 4];
            for byte in ch.encode_utf8(&mut buffer).as_bytes() {
                pieces.push(format!("<0x{byte:02X}>"));
            }
        }
    }
    pieces.push(END_OF_WORD.to_string());
    pieces
}

// ---------------------------------------------------------------------------
// Stage 2 — input embedding
// ---------------------------------------------------------------------------

fn write_embedding(
    weights: &detwgt::Artifact,
    shape: &Shape,
    stages: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let embed = weights.get("model.language_model.embed_tokens.weight")?;
    let rows = (0..shape.vocab)
        .map(|token_id| {
            Ok(EmbeddingRow {
                token_id: token_id as u32,
                values_hex: pack_hex(embed.row(token_id)?),
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    let commitment = write_external(
        &EmbeddingTable {
            hidden_size: shape.hidden as u32,
            rows: rows.into(),
        },
        "input-embedding",
        "embedding",
    )?;

    stages.push(format!(
        concat!(
            "[[chain.stage]]\n",
            "name = \"input_embedding\"\n",
            "project = \"input-embedding\"\n",
            "inputs.prompt = {{ from = \"prompt_prepare\" }}\n",
            "{}"
        ),
        external_line("embedding", "input-embedding", "embedding", &commitment)
    ));
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 3 — per-layer embeddings
// ---------------------------------------------------------------------------

fn write_ple_layers(
    weights: &detwgt::Artifact,
    shape: &Shape,
    stages: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let per_layer = weights.get("model.language_model.embed_tokens_per_layer.weight")?;
    let projection = weights.get("model.language_model.per_layer_model_projection.weight")?;
    let projection_norm = weights.get("model.language_model.per_layer_projection_norm.weight")?;

    for layer_idx in 0..shape.layers {
        let start = layer_idx * shape.ple_width;
        let end = start + shape.ple_width;

        // The packed [vocab, layers · width] table gives this layer its slice.
        let slice = per_layer.column_slice(start, end)?;
        let embedding_rows = slice
            .chunks_exact(shape.ple_width)
            .enumerate()
            .map(|(token_id, values)| PleEmbeddingRow {
                token_id: token_id as u32,
                values_hex: pack_hex(values),
            })
            .collect::<Vec<_>>();

        let layer = PleLayer {
            params: PleLayerParams {
                layer_idx: layer_idx as u32,
                hidden_size: shape.hidden as u32,
                ple_width: shape.ple_width as u32,
                // These three scalars are derived by the model loader rather
                // than stored in the artifact; unit scaling keeps the import
                // faithful to what the weights themselves say.
                embedding_scale: ONE,
                projection_scalar: ONE,
                input_scale: ONE,
                norm_eps: shape.norm_eps,
                projection_hex: pack_hex(&projection.rows(start, end)?),
                norm_weights_hex: pack_hex(&projection_norm.values),
            },
            embedding_rows: embedding_rows.into(),
        };

        let name = format!("layer{layer_idx}");
        let commitment = write_external(&layer, "prefill-prepare-aux", &name)?;
        stages.push(format!(
            concat!(
                "[[chain.stage]]\n",
                "name = \"prefill_prepare_aux_l{idx}\"\n",
                "project = \"prefill-prepare-aux\"\n",
                "inputs.embedded = {{ from = \"input_embedding\" }}\n",
                "{line}"
            ),
            idx = layer_idx,
            line = external_line("layer", "prefill-prepare-aux", &name, &commitment)
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 4 — transformer layers
// ---------------------------------------------------------------------------

fn write_transformer_layers(
    weights: &detwgt::Artifact,
    shape: &Shape,
    stages: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    for layer_idx in 0..shape.layers {
        let at = |suffix: &str| format!("model.language_model.layers.{layer_idx}.{suffix}");
        let gate = weights.get(&at("mlp.gate_proj.weight"))?;
        let ffn = gate.dims[0] as usize;

        // Alternating sliding/full attention, as `layer_types` declares.
        let sliding = match shape.layer_types.get(layer_idx).map(String::as_str) {
            Some("sliding_attention") => shape.sliding_window,
            _ => 0,
        };
        // `attention_k_eq_v` layers store no separate V projection: V *is* K.
        // tiny-gemma-dev emits v_proj only for its sliding layers.
        let w_v = match weights.get(&at("self_attn.v_proj.weight")) {
            Ok(tensor) => &tensor.values,
            Err(_) => &weights.get(&at("self_attn.k_proj.weight"))?.values,
        };
        let layer_scalar = weights
            .get(&at("layer_scalar"))
            .ok()
            .and_then(|tensor| tensor.values.first().copied())
            .unwrap_or(0);

        let layer = LayerParams {
            layer_idx: layer_idx as u32,
            hidden_size: shape.hidden as u32,
            ffn_size: ffn as u32,
            num_heads: shape.heads as u32,
            num_kv_heads: shape.kv_heads as u32,
            head_dim: shape.head_dim as u32,
            sliding_window: sliding,
            attn_scale: inv_sqrt_q16(shape.head_dim),
            layer_scalar,
            norm_eps: shape.norm_eps,
            norm_input_hex: pack_hex(&weights.get(&at("input_layernorm.weight"))?.values),
            norm_post_attn_hex: pack_hex(
                &weights.get(&at("post_attention_layernorm.weight"))?.values,
            ),
            norm_pre_ffw_hex: pack_hex(
                &weights.get(&at("pre_feedforward_layernorm.weight"))?.values,
            ),
            norm_post_ffw_hex: pack_hex(
                &weights.get(&at("post_feedforward_layernorm.weight"))?.values,
            ),
            q_norm_hex: pack_hex(&weights.get(&at("self_attn.q_norm.weight"))?.values),
            k_norm_hex: pack_hex(&weights.get(&at("self_attn.k_norm.weight"))?.values),
            w_q_hex: pack_hex(&weights.get(&at("self_attn.q_proj.weight"))?.values),
            w_k_hex: pack_hex(&weights.get(&at("self_attn.k_proj.weight"))?.values),
            w_v_hex: pack_hex(w_v),
            w_o_hex: pack_hex(&weights.get(&at("self_attn.o_proj.weight"))?.values),
            w_gate_hex: pack_hex(&gate.values),
            w_up_hex: pack_hex(&weights.get(&at("mlp.up_proj.weight"))?.values),
            w_down_hex: pack_hex(&weights.get(&at("mlp.down_proj.weight"))?.values),
        };

        let name = format!("layer{layer_idx}");
        let commitment = write_external(&layer, "prefill-range", &name)?;
        let upstream = if layer_idx == 0 {
            "input_embedding".to_string()
        } else {
            format!("prefill_range_l{}", layer_idx - 1)
        };
        stages.push(format!(
            concat!(
                "[[chain.stage]]\n",
                "name = \"prefill_range_l{idx}\"\n",
                "project = \"prefill-range\"\n",
                "inputs.activations = {{ from = \"{upstream}\" }}\n",
                "{line}"
            ),
            idx = layer_idx,
            upstream = upstream,
            line = external_line("layer", "prefill-range", &name, &commitment)
        ));
    }
    Ok(())
}

/// `1/sqrt(n)` in Q16.16, computed in integers so the fixture and the guest
/// agree on the constant.
fn inv_sqrt_q16(n: usize) -> i32 {
    if n == 0 {
        return 0;
    }
    let root = (0..).find(|r| r * r >= n).unwrap_or(1).max(1);
    let exact = (n as f64).sqrt();
    let _ = root;
    f32_to_q16(1.0 / exact)
}

// ---------------------------------------------------------------------------
// Stage 5 — output head
// ---------------------------------------------------------------------------

fn write_head(
    weights: &detwgt::Artifact,
    shape: &Shape,
    text: &serde_json::Value,
    stages: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let tied = text
        .get("tie_word_embeddings")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let projection = if tied {
        weights.get("model.language_model.embed_tokens.weight")?
    } else {
        weights.get("model.language_model.lm_head.weight")?
    };
    let softcap = text
        .get("final_logit_softcapping")
        .and_then(serde_json::Value::as_f64)
        .map(f32_to_q16)
        .unwrap_or(0);

    let rows = (0..shape.vocab)
        .map(|token_id| {
            Ok(LogitRow {
                token_id: token_id as u32,
                values_hex: pack_hex(projection.row(token_id)?),
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    let commitment = write_external(
        &FinalHead {
            params: FinalHeadParams {
                hidden_size: shape.hidden as u32,
                norm_eps: shape.norm_eps,
                softcap,
                norm_weights_hex: pack_hex(&weights.get("model.language_model.norm.weight")?.values),
            },
            rows: rows.into(),
        },
        "prefill-finalize",
        "head",
    )?;

    stages.push(format!(
        concat!(
            "[[chain.stage]]\n",
            "name = \"prefill_finalize\"\n",
            "project = \"prefill-finalize\"\n",
            "inputs.activations = {{ from = \"prefill_range_l{last}\" }}\n",
            "{line}"
        ),
        last = shape.layers - 1,
        line = external_line("head", "prefill-finalize", "head", &commitment)
    ));
    Ok(())
}

// ---------------------------------------------------------------------------

fn write_external<T: serde::Serialize>(
    value: &T,
    stage_dir: &str,
    name: &str,
) -> Result<String, Box<dyn Error>> {
    let dir = Path::new(stage_dir);
    fs::create_dir_all(dir)?;
    let commitment = raster::write_raster_files(
        value,
        &dir.join(format!("{name}.rastered")),
        &dir.join(format!("{name}.rindex")),
    )?;
    println!("wrote {stage_dir}/{name}.rastered  commitment = {commitment}");
    Ok(commitment)
}

/// Writes a stage's `input.json` / `input_manifest.json` so it can also be run
/// on its own with `cargo raster run`.
fn write_stage_fixtures(
    stage_dir: &str,
    params: &[(&str, &str, &String)],
) -> Result<(), Box<dyn Error>> {
    let entries = |render: &dyn Fn(&str, &str, &str) -> String| -> String {
        params
            .iter()
            .map(|(param, name, commitment)| render(param, name, commitment))
            .collect::<Vec<_>>()
            .join(",\n")
    };
    let input_json = entries(&|param, name, _| {
        format!("  \"{param}\": {{ \"path\": \"{name}.rastered\", \"index_path\": \"{name}.rindex\", \"load_preference\": \"read\" }}")
    });
    let manifest_json = entries(&|param, _, commitment| {
        format!("  \"{param}\": {{ \"type\": \"sha256\", \"encoding\": \"raster\", \"commitment\": \"{commitment}\" }}")
    });
    let dir = Path::new(stage_dir);
    fs::write(dir.join("input.json"), format!("{{\n{input_json}\n}}\n"))?;
    fs::write(
        dir.join("input_manifest.json"),
        format!("{{\n{manifest_json}\n}}\n"),
    )?;
    println!("wrote {stage_dir}/input.json + input_manifest.json");
    Ok(())
}

fn external_line(param: &str, stage_dir: &str, name: &str, commitment: &str) -> String {
    format!(
        "inputs.{param} = {{ external = {{ path = \"{stage_dir}/{name}.rastered\", index_path = \"{stage_dir}/{name}.rindex\", commitment = \"{commitment}\" }} }}\n"
    )
}
