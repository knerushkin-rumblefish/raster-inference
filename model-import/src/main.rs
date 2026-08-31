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

/// Gemma's turn markers, as `chat_template.jinja` emits them.
///
/// An instruction-tuned model was trained to answer inside its own turn, so a
/// bare prompt is not a question to it — it is a fragment, and the continuation
/// it produces is a turn break, not an answer. There is no Jinja engine here,
/// so the one shape this chain needs — a single user message with the
/// generation prompt appended — is written out directly.
const BOS_TOKEN: &str = "<bos>";
const TURN_OPEN: &str = "<|turn>";
const TURN_CLOSE: &str = "<turn|>";
const NEWLINE_TOKEN: &str = "\n";

fn main() {
    if let Err(error) = run() {
        eprintln!("model-import: {error}");
        std::process::exit(1);
    }
}

struct Args {
    model_dir: PathBuf,
    prompt: String,
    /// Commit the prompt exactly as given, with no turn markers and no `<bos>`.
    /// Needed for a tokenizer that has no such tokens — `tiny-gemma-dev` — and
    /// for reproducing a pre-template fixture.
    raw_prompt: bool,
    /// Exact number of generated tokens. The decode loop is statically
    /// expanded to this many select+transition iterations.
    tokens: u32,
    /// Write the complete chain manifest here instead of printing it.
    manifest: Option<PathBuf>,
    /// Rewrite only the tokenizer externals, leaving the weight externals
    /// alone. The weights are ~9 GB and depend on nothing the tokenizer
    /// touches, so a prompt or vocabulary-layout change has no reason to
    /// re-emit them. Prints only the `prompt_prepare` stage.
    only_tokenizer: bool,
    /// Rewrite only the transformer-layer externals (`prefill-range`). Nothing
    /// else depends on `LayerParams`, so a per-layer shape correction has no
    /// reason to re-emit the ~11 GB of tokenizer, embedding, PLE and head
    /// artifacts. Prints only the `prefill_range_*` stages.
    only_layers: bool,
    /// Rewrite only the embedding external. Same reasoning as `--only-layers`:
    /// nothing else depends on `EmbeddingTable`.
    only_embedding: bool,
    /// Rewrite only the PLE layer externals (`prefill-prepare-aux`).
    only_ple: bool,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut model_dir = None;
    let mut prompt = String::from("hello raster");
    let mut raw_prompt = false;
    let mut tokens = 1u32;
    let mut manifest = None;
    let mut only_tokenizer = false;
    let mut only_layers = false;
    let mut only_embedding = false;
    let mut only_ple = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_dir = args.next().map(PathBuf::from),
            "--prompt" => prompt = args.next().ok_or("--prompt needs a value")?,
            "--raw-prompt" => raw_prompt = true,
            "--tokens" => {
                tokens = args
                    .next()
                    .ok_or("--tokens needs a value")?
                    .parse()
                    .map_err(|_| "--tokens must be a non-negative integer")?
            }
            "--manifest" => manifest = args.next().map(PathBuf::from),
            "--only-tokenizer" => only_tokenizer = true,
            "--only-layers" => only_layers = true,
            "--only-embedding" => only_embedding = true,
            "--only-ple" => only_ple = true,
            other => return Err(format!("unknown argument '{other}'").into()),
        }
    }
    Ok(Args {
        model_dir: model_dir.ok_or("--model <bundle-dir> is required")?,
        prompt,
        raw_prompt,
        tokens,
        manifest,
        only_tokenizer,
        only_layers,
        only_embedding,
        only_ple,
    })
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let tokenizer: serde_json::Value =
        serde_json::from_slice(&fs::read(args.model_dir.join("tokenizer.json"))?)?;

    let eos_ids = load_eos_ids(&args.model_dir);

    if args.only_tokenizer {
        let mut stages = Vec::new();
        let decoder_commitment = write_tokenizer(
            &tokenizer,
            &args.prompt,
            args.raw_prompt,
            &eos_ids,
            &mut stages,
        )?;
        println!();
        println!("# ---- replaces prompt and decoder externals in the root Raster.toml ----");
        for stage in &stages {
            println!();
            print!("{stage}");
        }
        println!();
        println!(
            "{}",
            external_line(
                "decoder",
                "output-finalize",
                "decoder",
                &decoder_commitment
            )
        );
        return Ok(());
    }

    let weights = detwgt::load(&args.model_dir.join("model.detwgt"))?;
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(args.model_dir.join("config.json"))?)?;
    let text = config
        .get("text_config")
        .ok_or("config.json has no text_config")?;

    if args.only_ple {
        let shape = Shape::from_config(text)?;
        let mut stages = Vec::new();
        write_ple_layers(&weights, &shape, &mut stages)?;
        println!();
        println!("# ---- replaces the prefill_prepare_aux_* stages in the root Raster.toml ----");
        for stage in &stages {
            println!();
            print!("{stage}");
        }
        return Ok(());
    }

    if args.only_embedding {
        let shape = Shape::from_config(text)?;
        let mut stages = Vec::new();
        write_embedding(&weights, &shape, &mut stages)?;
        println!();
        println!("# ---- replaces the input_embedding stage in the root Raster.toml ----");
        for stage in &stages {
            println!();
            print!("{stage}");
        }
        return Ok(());
    }

    if args.only_layers {
        let shape = Shape::from_config(text)?;
        let mut stages = Vec::new();
        write_transformer_layers(&weights, &shape, &mut stages)?;
        println!();
        println!("# ---- replaces the prefill_range_* stages in the root Raster.toml ----");
        for stage in &stages {
            println!();
            print!("{stage}");
        }
        return Ok(());
    }

    let shape = Shape::from_config(text)?;
    println!(
        "model: hidden {} · {} layers · {} heads over {} kv heads · head_dim {} · ffn {} · vocab {} · ple {}",
        shape.hidden, shape.layers, shape.heads, shape.kv_heads, shape.head_dim, shape.ffn,
        shape.vocab, shape.ple_width
    );
    println!("tensors: {}", weights.names().count());
    println!();

    let mut stages = Vec::new();
    let decoder_commitment = write_tokenizer(
        &tokenizer,
        &args.prompt,
        args.raw_prompt,
        &eos_ids,
        &mut stages,
    )?;
    write_embedding(&weights, &shape, &mut stages)?;
    write_ple_layers(&weights, &shape, &mut stages)?;
    write_transformer_layers(&weights, &shape, &mut stages)?;
    write_head(&weights, &shape, text, &mut stages)?;

    let mut manifest = String::from(
        "[chain]\nname = \"raster-chain-inference\"\nversion = \"0.1.0\"\n",
    );
    manifest.push_str(&indexed_model_inputs(&shape, &stages)?);
    for stage in &stages {
        manifest.push('\n');
        manifest.push_str(stage);
    }
    manifest.push_str(&generation_stages(
        &shape,
        &decoder_commitment,
        args.tokens,
        &stages,
    )?);
    if let Some(path) = args.manifest {
        fs::write(&path, manifest)?;
        println!("wrote {}", path.display());
    } else {
        println!();
        println!("# ---- root Raster.toml ----");
        print!("{manifest}");
    }
    Ok(())
}

fn stage_external_commitment(
    stages: &[String],
    stage_name: &str,
    parameter: &str,
) -> Result<String, Box<dyn Error>> {
    let name = format!("name = \"{stage_name}\"");
    let prefix = format!("inputs.{parameter} = ");
    let line = stages
        .iter()
        .find(|stage| stage.lines().any(|line| line == name))
        .and_then(|stage| stage.lines().find(|line| line.starts_with(&prefix)))
        .ok_or_else(|| {
            format!("generated prefill manifest has no {stage_name}.{parameter} binding")
        })?;
    let marker = "commitment = \"";
    let start = line
        .find(marker)
        .ok_or_else(|| format!("{stage_name}.{parameter} is not an external binding"))?
        + marker.len();
    let end = line[start..]
        .find('"')
        .ok_or_else(|| format!("{stage_name}.{parameter} has an unterminated commitment"))?
        + start;
    Ok(line[start..end].to_string())
}

/// Declares model weights once so every decode iteration can bind them by its
/// static layer index without copying commitments into the repeat block.
fn indexed_model_inputs(shape: &Shape, stages: &[String]) -> Result<String, Box<dyn Error>> {
    let aux = (0..shape.layers)
        .map(|layer| {
            stage_external_commitment(
                stages,
                &format!("prefill_prepare_aux_l{layer}"),
                "layer",
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let transformer = (0..shape.layers)
        .map(|layer| {
            stage_external_commitment(stages, &format!("prefill_range_l{layer}"), "layer")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let render = |name: &str, dir: &str, commitments: &[String]| {
        let entries = commitments
            .iter()
            .enumerate()
            .map(|(index, commitment)| format!("  \"{commitment}\", # l{index}"))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "\n[chain.input.{name}]\n\
             index = \"l\"\n\
             path = \"{dir}/layer{{l}}.rastered\"\n\
             index_path = \"{dir}/layer{{l}}.rindex\"\n\
             commitments = [\n{entries}\n]\n"
        )
    };
    Ok(format!(
        "{}{}",
        render("aux_layer", "prefill-prepare-aux", &aux),
        render("transformer_layer", "prefill-range", &transformer)
    ))
}

/// Emits exactly `tokens` select+transition iterations as one static repeat.
/// Expansion still produces one auditable chain stage per program invocation.
fn generation_stages(
    shape: &Shape,
    decoder_commitment: &str,
    tokens: u32,
    prefill_stages: &[String],
) -> Result<String, Box<dyn Error>> {
    let embedding = stage_external_commitment(prefill_stages, "input_embedding", "embedding")?;
    let head = stage_external_commitment(prefill_stages, "prefill_finalize", "head")?;
    let first_shared = shape.layers.saturating_sub(shape.num_kv_shared_layers);
    let mut donors = (0..shape.layers)
        .map(|layer| shape.kv_donor_layer(layer))
        .filter(|donor| *donor >= 0)
        .collect::<Vec<_>>();
    donors.sort_unstable();
    donors.dedup();
    if donors.len() > 2 {
        return Err(format!(
            "decode repeat supports at most two donor candidates, model declares {}",
            donors.len()
        )
        .into());
    }
    let donor_a = donors.first().copied();
    let donor_b = donors.get(1).copied();

    let mut out = String::from(
        "\n[[chain.stage]]\nname = \"decode_init\"\nproject = \"decode-init\"\n\
         \n[[chain.repeat]]\nname = \"decode\"\nindex = \"t\"\n",
    );
    out.push_str(&format!("count = {tokens}\n"));
    out.push_str(
        "\n  [chain.repeat.exports.edge]\n\
         stage = \"decode_select_t{t}\"\n\
         entry = \"decode_init\"\n\
         \n  [chain.repeat.exports.logits]\n\
         stage = \"decode_finalize_t{t}\"\n\
         entry = \"prefill_finalize\"\n\
         \n  [[chain.repeat.stage]]\n\
         name = \"decode_select_t{t}\"\n\
         project = \"decode-select-token\"\n\
         inputs.logits = { from = \"decode_finalize_t{t-1}\", first = \"prefill_finalize\" }\n\
         inputs.prior = { from = \"decode_select_t{t-1}\", first = \"decode_init\" }\n\
         \n  [[chain.repeat.stage]]\n\
         name = \"decode_embed_t{t}\"\n\
         project = \"decode-embed\"\n\
         inputs.selected = { from = \"decode_select_t{t}\" }\n",
    );
    out.push_str(&format!(
        "  inputs.embedding = {{ external = {{ path = \"input-embedding/embedding.rastered\", index_path = \"input-embedding/embedding.rindex\", commitment = \"{embedding}\" }} }}\n"
    ));
    out.push_str(&format!(
        "\n  [[chain.repeat.stage]]\n\
         name = \"decode_aux_t{{t}}_l{{l}}\"\n\
         index = \"l\"\n\
         count = {}\n\
         project = \"prefill-prepare-aux\"\n\
         inputs.embedded = {{ from = \"decode_embed_t{{t}}\" }}\n\
         inputs.layer = {{ input = \"aux_layer_{{l}}\" }}\n",
        shape.layers
    ));

    if first_shared > 0 {
        out.push_str(&format!(
            "\n  [[chain.repeat.stage]]\n\
             name = \"decode_range_t{{t}}_l{{l}}\"\n\
             index = \"l\"\n\
             start = 0\n\
             count = {first_shared}\n\
             project = \"prefill-range\"\n\
             inputs.activations = {{ from = \"decode_range_t{{t}}_l{{l-1}}\", first = \"decode_embed_t{{t}}\" }}\n\
             inputs.prior_kv = {{ from = \"decode_range_t{{t-1}}_l{{l}}\", first = \"prefill_range_l{{l}}\" }}\n\
             inputs.donor_a_kv = {{ from = \"input_embedding\" }}\n\
             inputs.donor_b_kv = {{ from = \"input_embedding\" }}\n\
             inputs.ple = {{ from = \"decode_aux_t{{t}}_l{{l}}\" }}\n\
             inputs.layer = {{ input = \"transformer_layer_{{l}}\" }}\n"
        ));
    }

    if shape.num_kv_shared_layers > 0 {
        let donor_a_from = donor_a
            .map(|layer| format!("decode_range_t{{t}}_l{layer}"))
            .unwrap_or_else(|| String::from("input_embedding"));
        let donor_b_from = donor_b
            .map(|layer| format!("decode_range_t{{t}}_l{layer}"))
            .unwrap_or_else(|| String::from("input_embedding"));
        out.push_str(&format!(
            "\n  [[chain.repeat.stage]]\n\
             name = \"decode_range_t{{t}}_l{{l}}\"\n\
             index = \"l\"\n\
             start = {first_shared}\n\
             count = {}\n\
             project = \"prefill-range\"\n\
             inputs.activations = {{ from = \"decode_range_t{{t}}_l{{l-1}}\", first = \"decode_range_t{{t}}_l{}\" }}\n\
             inputs.prior_kv = {{ from = \"decode_range_t{{t-1}}_l{{l}}\", first = \"prefill_range_l{{l}}\" }}\n\
             inputs.donor_a_kv = {{ from = \"{donor_a_from}\" }}\n\
             inputs.donor_b_kv = {{ from = \"{donor_b_from}\" }}\n\
             inputs.ple = {{ from = \"decode_aux_t{{t}}_l{{l}}\" }}\n\
             inputs.layer = {{ input = \"transformer_layer_{{l}}\" }}\n",
            shape.num_kv_shared_layers,
            first_shared.saturating_sub(1)
        ));
    }

    out.push_str(&format!(
        "\n  [[chain.repeat.stage]]\n\
         name = \"decode_finalize_t{{t}}\"\n\
         project = \"prefill-finalize\"\n\
         inputs.activations = {{ from = \"decode_range_t{{t}}_l{}\" }}\n\
         inputs.head = {{ external = {{ path = \"prefill-finalize/head.rastered\", index_path = \"prefill-finalize/head.rindex\", commitment = \"{head}\" }} }}\n\
         \n[[chain.stage]]\n\
         name = \"output_finalize\"\n\
         project = \"output-finalize\"\n\
         inputs.edge = {{ from = \"decode.edge\" }}\n\
         {}",
        shape.layers - 1,
        external_line(
            "decoder",
            "output-finalize",
            "decoder",
            decoder_commitment
        )
    ));
    Ok(out)
}

/// Q16.16 bits of an `f32`, through the reference's own conversion so the
/// committed scalar is the same value the deterministic path uses, not a
/// near one.
fn det_act_bits(value: f32) -> i32 {
    raster_inference::shared::numerics::det_num::f32_to_act(value).to_bits()
}

fn as_f64(value: &serde_json::Value) -> Option<f64> {
    value.as_f64()
}

/// One key out of `rope_parameters.<attention_type>`, which is where Gemma 3n
/// puts its per-family RoPE settings.
fn rope_param(text: &serde_json::Value, attention_type: &str, key: &str) -> Option<f64> {
    text.get("rope_parameters")?
        .get(attention_type)?
        .get(key)
        .and_then(as_f64)
}

/// The shapes every stage needs, read from `config.json`.
struct Shape {
    hidden: usize,
    ffn: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    /// Head dim of the *full-attention* ("global") layers, which is not the
    /// same as `head_dim` on Gemma 3n: E2B declares `head_dim: 256` for the
    /// sliding layers and `global_head_dim: 512` for the full ones, and the
    /// weights follow — a full layer's `q_proj` is [heads · 512, hidden] and
    /// its `q_norm` has 512 values. Defaults to `head_dim` for models that
    /// declare no such split.
    global_head_dim: usize,
    /// RoPE base per attention type, from `rope_parameters`. Gemma 3n gives the
    /// two families different bases — 10 000 local, 1 000 000 global — so a
    /// single `rope_theta` would silently detune half the layers.
    rope_base_sliding: f64,
    rope_base_full: f64,
    /// Fraction of a full-attention head that rotates (`0.25` here). Sliding
    /// layers rotate the whole head.
    full_partial_rotary_factor: f64,
    /// How many trailing layers borrow their K/V from an earlier donor.
    num_kv_shared_layers: usize,
    vocab: usize,
    ple_width: usize,
    sliding_window: u32,
    layer_types: Vec<String>,
    norm_eps: i64,
}

impl Shape {
    fn is_sliding(&self, idx: usize) -> bool {
        matches!(
            self.layer_types.get(idx).map(String::as_str),
            Some("sliding_attention")
        )
    }

    /// Head dim of layer `idx` — `global_head_dim` for a full-attention layer,
    /// `head_dim` for a sliding one.
    fn head_dim_at(&self, idx: usize) -> usize {
        if self.is_sliding(idx) {
            self.head_dim
        } else {
            self.global_head_dim
        }
    }

    /// The layer whose K/V cache layer `idx` borrows, or `-1` if it computes
    /// its own.
    ///
    /// Mirrors `casettek/.../shared/model/gemma/io.rs:171` exactly: sharing
    /// starts at `num_hidden_layers - num_kv_shared_layers`, and a sharing
    /// layer's donor is the **last layer before that boundary with the same
    /// attention type**. For this model that resolves to layer 13 for every
    /// sliding layer and layer 14 for every full one — 20 layers in all, which
    /// is the count the config declares.
    fn kv_donor_layer(&self, idx: usize) -> i32 {
        let first_shared = self.layers.saturating_sub(self.num_kv_shared_layers);
        if self.num_kv_shared_layers == 0 || idx < first_shared {
            return -1;
        }
        let Some(attention_type) = self.layer_types.get(idx) else {
            return -1;
        };
        self.layer_types[..first_shared]
            .iter()
            .rposition(|candidate| candidate == attention_type)
            .map(|donor| donor as i32)
            .unwrap_or(-1)
    }

    /// RoPE parameters for layer `idx`: `(base_bits, rotary_dim, freq_base_dim)`.
    ///
    /// `base_bits` is the base as `Acc` (Q32.32) bits. Both bases this model
    /// declares are exact integers, so the encoding is a plain shift — asserted
    /// against the reference's `f32_to_acc` in `det-num/tests/equivalence.rs`.
    ///
    /// `freq_base_dim` is the layer's `head_dim`, which is *not* `rotary_dim`
    /// once the rotation is partial: a full-attention head is 512 wide and
    /// rotates its first 128 lanes against a ladder built over 512.
    fn rope_at(&self, idx: usize) -> (i64, u32, u32) {
        let head_dim = self.head_dim_at(idx);
        let (base, rotary_dim) = if self.is_sliding(idx) {
            (self.rope_base_sliding, head_dim)
        } else {
            (
                self.rope_base_full,
                (head_dim as f64 * self.full_partial_rotary_factor) as usize,
            )
        };
        ((base as i64) << 32, rotary_dim as u32, head_dim as u32)
    }
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
            global_head_dim: text
                .get("global_head_dim")
                .and_then(serde_json::Value::as_u64)
                .map(|value| value as usize)
                .unwrap_or(usize_at("head_dim")?),
            rope_base_sliding: rope_param(text, "sliding_attention", "rope_theta")
                .or_else(|| text.get("rope_local_base_freq").and_then(as_f64))
                .unwrap_or(10_000.0),
            rope_base_full: rope_param(text, "full_attention", "rope_theta")
                .or_else(|| text.get("rope_theta").and_then(as_f64))
                .unwrap_or(1_000_000.0),
            full_partial_rotary_factor: rope_param(
                text,
                "full_attention",
                "partial_rotary_factor",
            )
            .unwrap_or(1.0),
            num_kv_shared_layers: text
                .get("num_kv_shared_layers")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as usize,
            vocab: usize_at("vocab_size")?,
            ple_width: usize_at("hidden_size_per_layer_input")?,
            sliding_window: text
                .get("sliding_window")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32,
            layer_types,
            // `Acc` (Q32.32) bits, not Q16.16: the canonical `rms_norm` takes its
            // epsilon in accumulator units. Encoded through the reference's own
            // `f32_to_acc` so it is the same value, not a near one — 1e-6 in
            // Q16.16 rounds to zero, which is how this was silently disabled.
            norm_eps: raster_inference::shared::numerics::det_num::f32_to_acc(
                text.get("rms_norm_eps")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(1e-6) as f32,
            )
            .to_bits(),
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

/// The model's `eos_token_id` set, from `config.json`.
///
/// Gemma declares more than one — `<eos>`, `<turn|>` and friends — so this is a
/// set, and it is read from the bundle rather than hardcoded. A bundle without
/// a `config.json` (or without the key) yields an empty set: nothing is marked
/// terminal, which is exactly the behaviour before this existed.
fn load_eos_ids(model_dir: &Path) -> std::collections::BTreeSet<u32> {
    let Ok(bytes) = fs::read(model_dir.join("config.json")) else {
        return Default::default();
    };
    let Ok(config) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return Default::default();
    };
    let declared = config
        .get("eos_token_id")
        .or_else(|| config.get("text_config").and_then(|t| t.get("eos_token_id")));
    match declared {
        Some(serde_json::Value::Number(id)) => {
            id.as_u64().map(|id| id as u32).into_iter().collect()
        }
        Some(serde_json::Value::Array(ids)) => ids
            .iter()
            .filter_map(serde_json::Value::as_u64)
            .map(|id| id as u32)
            .collect(),
        _ => Default::default(),
    }
}

fn write_tokenizer(
    tokenizer: &serde_json::Value,
    prompt: &str,
    raw_prompt: bool,
    eos_ids: &std::collections::BTreeSet<u32>,
    stages: &mut Vec<String>,
) -> Result<String, Box<dyn Error>> {
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

    let special_ids: std::collections::BTreeSet<u32> = tokenizer
        .get("added_tokens")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| {
            entry
                .get("special")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        })
        .filter_map(|entry| entry.get("id").and_then(serde_json::Value::as_u64))
        .map(|id| id as u32)
        .collect();

    // The same entries by text, longest first, so `split_prompt` can keep each
    // one whole and prefer the longer of two that share a prefix.
    let mut special_tokens: Vec<String> = tokenizer
        .get("added_tokens")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| {
            entry
                .get("special")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        })
        .filter_map(|entry| entry.get("content").and_then(serde_json::Value::as_str))
        .map(str::to_string)
        .collect();
    special_tokens.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));

    let max_token_id = vocab.iter().map(|entry| entry.id).max().unwrap_or(0);
    let mut decoder_tokens =
        vec![DecoderToken::default(); max_token_id.saturating_add(1) as usize];
    for entry in &vocab {
        decoder_tokens[entry.id as usize] = DecoderToken {
            token: entry.token.clone(),
            special: special_ids.contains(&entry.id),
            terminal: eos_ids.contains(&entry.id),
        };
    }
    let decoder_commitment = write_external(
        &DecoderTable {
            tokens: decoder_tokens.into(),
        },
        "output-finalize",
        "decoder",
    )?;

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

    // An instruction-tuned model answers inside a turn it was asked to open.
    // Templating is therefore the default, and dropping to the bare prompt is
    // an explicit choice — either the caller's, or forced by a tokenizer that
    // has no turn markers to template with.
    let templated = !raw_prompt && supports_gemma_turns(&vocab_map);
    if !raw_prompt && !templated {
        println!(
            "prompt: no {TURN_OPEN}/{TURN_CLOSE}/{BOS_TOKEN} in this vocabulary — \
             committing the bare prompt"
        );
    }
    let rendered = if templated {
        render_gemma_turns(prompt)
    } else {
        prompt.to_string()
    };
    let pieces = if templated {
        split_prompt(&rendered, &vocab_map, &special_tokens)
    } else {
        split_prompt(&rendered, &vocab_map, &[])
    };

    println!("tokenizer: {} entries, {} merges", vocab.len(), merges.len());
    println!(
        "prompt: {} · {} terminal id(s)",
        if templated {
            "gemma turn template"
        } else {
            "raw"
        },
        eos_ids.len()
    );
    println!("rendered prompt: {rendered:?}");
    println!("prompt pieces: {pieces:?}");

    let (vocab_bucket_count, vocab_buckets) = bucket_vocab(vocab);
    let (merge_bucket_count, merge_buckets) = bucket_merges(merges);

    let tokenizer_commitment = write_external(
        &PromptTokenizer {
            vocab_bucket_count,
            merge_bucket_count,
            vocab_buckets: vocab_buckets.into(),
            merge_buckets: merge_buckets.into(),
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
    Ok(decoder_commitment)
}

/// Entries per bucket to aim for.
///
/// This is the knob that trades index size against scan length, and both are
/// cheap here: each lookup costs one dynamic-index selection plus this many
/// recur iterations, so 4 keeps a lookup under ~10 replay units while adding
/// only `len / 4` bucket nodes to the index.
const TARGET_BUCKET_LOAD: usize = 4;

/// Bucket count for `len` keys: at least one, so the modulus is never zero and
/// a computed index always has a bucket to land in.
fn bucket_count_for(len: usize) -> u32 {
    (len / TARGET_BUCKET_LOAD).max(1) as u32
}

/// Reports how evenly a bucketing came out — the number the lookup cost is
/// actually proportional to is the *max*, not the mean.
fn report_buckets(label: &str, count: u32, sizes: impl Iterator<Item = usize>) {
    let (mut max, mut used, mut total) = (0usize, 0usize, 0usize);
    for size in sizes {
        max = max.max(size);
        total += size;
        if size > 0 {
            used += 1;
        }
    }
    println!(
        "{label}: {total} keys in {count} buckets · avg {:.1} · max {max} · {} empty",
        total as f64 / count as f64,
        count as usize - used,
    );
}

fn bucket_vocab(vocab: Vec<TokenEntry>) -> (u32, Vec<VocabBucket>) {
    let count = bucket_count_for(vocab.len());
    let mut buckets: Vec<Vec<TokenEntry>> = vec![Vec::new(); count as usize];
    for entry in vocab {
        // The same call the tile makes, from the same crate.
        buckets[vocab_bucket_of(&entry.token, count) as usize].push(entry);
    }
    report_buckets("vocab", count, buckets.iter().map(Vec::len));
    (
        count,
        buckets
            .into_iter()
            .map(|entries| VocabBucket {
                entries: entries.into(),
            })
            .collect(),
    )
}

fn bucket_merges(merges: Vec<BpeMerge>) -> (u32, Vec<MergeBucket>) {
    let count = bucket_count_for(merges.len());
    let mut buckets: Vec<Vec<BpeMerge>> = vec![Vec::new(); count as usize];
    for rule in merges {
        buckets[merge_bucket_of(&rule.left, &rule.right, count) as usize].push(rule);
    }
    // Keep each bucket rank-ordered so the scan's "lowest rank wins" tie-break
    // sees rules in the same order the flat table presented them.
    for bucket in &mut buckets {
        bucket.sort_by_key(|rule| rule.rank);
    }
    report_buckets("merges", count, buckets.iter().map(Vec::len));
    (
        count,
        buckets
            .into_iter()
            .map(|rules| MergeBucket {
                rules: rules.into(),
            })
            .collect(),
    )
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

/// Renders one user message plus the generation prompt, the way the model's
/// `chat_template.jinja` does for `add_generation_prompt: true`.
fn render_gemma_turns(prompt: &str) -> String {
    format!(
        "{BOS_TOKEN}{TURN_OPEN}user\n{}{TURN_CLOSE}\n{TURN_OPEN}model\n",
        prompt.trim()
    )
}

/// Whether this tokenizer has the tokens the turn format is made of.
///
/// `tiny-gemma-dev` does not: templating against it would emit pieces the
/// vocabulary pass resolves to `<unk>`, which is worse than the bare prompt.
fn supports_gemma_turns(vocab: &BTreeMap<String, u32>) -> bool {
    [BOS_TOKEN, TURN_OPEN, TURN_CLOSE, NEWLINE_TOKEN]
        .iter()
        .all(|token| vocab.contains_key(*token))
}

/// Splits a prompt into the initial pieces the tokenizer stage consumes:
/// whole special tokens, then SentencePiece's space marker, one piece per
/// character, byte-fallback for anything the vocabulary does not have, and the
/// end-of-word terminator.
///
/// `specials` is matched longest-first and never split. A special token is a
/// single vocabulary entry whose text is several characters, and no merge rule
/// mentions one — so the merge pass could never reassemble it from characters,
/// and every turn marker would tokenize as its own punctuation.
fn split_prompt(prompt: &str, vocab: &BTreeMap<String, u32>, specials: &[String]) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut rest = prompt;
    while !rest.is_empty() {
        if let Some(special) = specials.iter().find(|token| rest.starts_with(token.as_str())) {
            pieces.push(special.clone());
            rest = &rest[special.len()..];
            continue;
        }
        let ch = rest.chars().next().expect("rest is non-empty");
        rest = &rest[ch.len_utf8()..];
        let piece = if ch == ' ' {
            String::from('\u{2581}')
        } else {
            ch.to_string()
        };
        if vocab.contains_key(&piece) {
            pieces.push(piece);
        } else {
            // Byte fallback: `<0xNN>` per UTF-8 byte, as the tokenizer declares.
            // Over the *piece*, not the source character: a space that reached
            // here is `▁`, and its three bytes are what the vocabulary would
            // have to spell.
            for byte in piece.as_bytes() {
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
    let mut values = Vec::with_capacity(shape.vocab * shape.hidden);
    for token_id in 0..shape.vocab {
        values.extend_from_slice(embed.row(token_id)?);
    }

    let commitment = write_external(
        &EmbeddingTable {
            hidden_size: shape.hidden as u32,
            // Gemma scales token embeddings by sqrt(hidden) before layer 0. Converted
            // by the reference's own `f32_to_act` so the committed bits match the
            // deterministic path exactly rather than approximately.
            embedding_scale: raster_inference::shared::numerics::det_num::f32_to_act(
                (shape.hidden as f32).sqrt(),
            )
            .to_bits(),
            values: paged_i32s(&values).map_err(|error| error.to_string())?,
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

        let layer = PleLayer {
            params: PleLayerParams {
                layer_idx: layer_idx as u32,
                hidden_size: shape.hidden as u32,
                ple_width: shape.ple_width as u32,
                // Derived, not stored in the artifact — the comment this
                // replaces claimed unit scaling was "faithful", but the model
                // loader computes real values and hardcoding 1.0 silently
                // dropped three scalings from every per-layer embedding
                // (`casettek/.../gemma/io.rs:1176-1179`).
                embedding_scale: det_act_bits((shape.ple_width as f32).sqrt()),
                projection_scalar: det_act_bits((shape.hidden as f32).powf(-0.5)),
                input_scale: det_act_bits(2f32.powf(-0.5)),
                norm_eps: shape.norm_eps,
                norm_weights: page_of_i32s(&projection_norm.values),
            },
            embeddings: paged_i32s(&slice).map_err(|error| error.to_string())?,
            projection: paged_i32s(&projection.rows(start, end)?)
                .map_err(|error| error.to_string())?,
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
    // Report the two RoPE families once. These are the numbers that silently
    // decide every attention score, so they are worth seeing rather than
    // trusting — a wrong base or rotary width produces a plausible-looking run
    // and the wrong token.
    for (label, idx) in shape
        .layer_types
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .fold(Vec::new(), |mut seen: Vec<(&str, usize)>, (t, i)| {
            if !seen.iter().any(|(s, _)| *s == t) {
                seen.push((t, i));
            }
            seen
        })
    {
        let (base, rotary, freq) = shape.rope_at(idx);
        println!(
            "rope[{label}]: head_dim {} · rotary_dim {rotary} · freq_base_dim {freq} · base {} (Q32.32 bits {base})",
            shape.head_dim_at(idx),
            base >> 32,
        );
    }

    // And the KV-sharing map, for the same reason: a wrong donor is not a crash,
    // it is a layer attending over another layer's keys.
    let donors: Vec<(usize, i32)> = (0..shape.layers)
        .map(|idx| (idx, shape.kv_donor_layer(idx)))
        .filter(|(_, donor)| *donor >= 0)
        .collect();
    let mut by_donor: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (idx, donor) in &donors {
        by_donor.entry(*donor).or_default().push(*idx);
    }
    println!(
        "kv sharing: {} of {} layers borrow (config declares {})",
        donors.len(),
        shape.layers,
        shape.num_kv_shared_layers
    );
    for (donor, borrowers) in &by_donor {
        println!("  donor layer {donor} -> {borrowers:?}");
    }
    if by_donor.len() > 2 {
        return Err(format!(
            "this chain supports at most two KV donor candidates, model declares {}",
            by_donor.len()
        )
        .into());
    }
    let mut donor_candidates = by_donor.keys().copied();
    let donor_a_layer = donor_candidates.next().unwrap_or(-2);
    let donor_b_layer = donor_candidates.next().unwrap_or(-3);

    for layer_idx in 0..shape.layers {
        let at = |suffix: &str| format!("model.language_model.layers.{layer_idx}.{suffix}");
        let gate = weights.get(&at("mlp.gate_proj.weight"))?;
        let ffn = gate.dims[0] as usize;

        // Alternating sliding/full attention, as `layer_types` declares.
        let sliding = match shape.layer_types.get(layer_idx).map(String::as_str) {
            Some("sliding_attention") => shape.sliding_window,
            _ => 0,
        };
        // Head dim is per-layer here, not global (see `Shape::global_head_dim`).
        let head_dim = shape.head_dim_at(layer_idx);
        let (rope_base, rotary_dim, rope_freq_base_dim) = shape.rope_at(layer_idx);
        let kv_donor_layer = shape.kv_donor_layer(layer_idx);
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

        let layer = TransformerLayer {
            params: LayerParams {
                layer_idx: layer_idx as u32,
                hidden_size: shape.hidden as u32,
                ffn_size: ffn as u32,
                num_heads: shape.heads as u32,
                num_kv_heads: shape.kv_heads as u32,
                head_dim: head_dim as u32,
                sliding_window: sliding,
                attn_scale: inv_sqrt_q16(head_dim),
                layer_scalar,
                norm_eps: shape.norm_eps,
                rope_base,
                rotary_dim,
                rope_freq_base_dim,
                kv_donor_layer,
                donor_a_layer,
                donor_b_layer,
                norm_input: page_of_i32s(&weights.get(&at("input_layernorm.weight"))?.values),
                norm_post_attn: page_of_i32s(
                    &weights.get(&at("post_attention_layernorm.weight"))?.values,
                ),
                norm_pre_ffw: page_of_i32s(
                    &weights.get(&at("pre_feedforward_layernorm.weight"))?.values,
                ),
                norm_post_ffw: page_of_i32s(
                    &weights.get(&at("post_feedforward_layernorm.weight"))?.values,
                ),
                q_norm: page_of_i32s(&weights.get(&at("self_attn.q_norm.weight"))?.values),
                k_norm: page_of_i32s(&weights.get(&at("self_attn.k_norm.weight"))?.values),
                ple_width: shape.ple_width as u32,
                // Hidden-width, despite the name: it normalises *after*
                // `per_layer_projection` maps back up from `ple_width`.
                ple_post_norm: page_of_i32s(
                    &weights.get(&at("post_per_layer_input_norm.weight"))?.values,
                ),
            },
            w_q: paged_i32s(&weights.get(&at("self_attn.q_proj.weight"))?.values)
                .map_err(|error| error.to_string())?,
            w_k: paged_i32s(&weights.get(&at("self_attn.k_proj.weight"))?.values)
                .map_err(|error| error.to_string())?,
            w_v: paged_i32s(w_v).map_err(|error| error.to_string())?,
            w_o: paged_i32s(&weights.get(&at("self_attn.o_proj.weight"))?.values)
                .map_err(|error| error.to_string())?,
            w_gate: paged_i32s(&gate.values).map_err(|error| error.to_string())?,
            w_up: paged_i32s(&weights.get(&at("mlp.up_proj.weight"))?.values)
                .map_err(|error| error.to_string())?,
            ple_input_gate: paged_i32s(&weights.get(&at("per_layer_input_gate.weight"))?.values)
                .map_err(|error| error.to_string())?,
            ple_layer_projection: paged_i32s(
                &weights.get(&at("per_layer_projection.weight"))?.values,
            )
            .map_err(|error| error.to_string())?,
            w_down: paged_i32s(&weights.get(&at("mlp.down_proj.weight"))?.values)
                .map_err(|error| error.to_string())?,
        };

        let name = format!("layer{layer_idx}");
        let commitment = write_external(&layer, "prefill-range", &name)?;
        let upstream = if layer_idx == 0 {
            "input_embedding".to_string()
        } else {
            format!("prefill_range_l{}", layer_idx - 1)
        };
        // Before the sharing boundary both candidates must be the empty input:
        // binding the eventual donor stages here would be a forward reference.
        // Sharing layers receive both candidates and committed
        // `kv_donor_layer` selects the exact one inside the tile.
        let first_shared = shape.layers.saturating_sub(shape.num_kv_shared_layers);
        let (donor_a, donor_b) = if layer_idx < first_shared {
            (
                String::from("input_embedding"),
                String::from("input_embedding"),
            )
        } else {
            (
                format!("prefill_range_l{donor_a_layer}"),
                format!("prefill_range_l{donor_b_layer}"),
            )
        };
        stages.push(format!(
            concat!(
                "[[chain.stage]]\n",
                "name = \"prefill_range_l{idx}\"\n",
                "project = \"prefill-range\"\n",
                "inputs.activations = {{ from = \"{upstream}\" }}\n",
                // Prefill inherits an empty cache. Bound to `input_embedding`
                // for every layer, the same stage the non-sharing layers use
                // for `donor_kv` — a decode stage is where this points at the
                // previous step instead.
                "inputs.prior_kv = {{ from = \"input_embedding\" }}\n",
                "inputs.donor_a_kv = {{ from = \"{donor_a}\" }}\n",
                "inputs.donor_b_kv = {{ from = \"{donor_b}\" }}\n",
                "inputs.ple = {{ from = \"prefill_prepare_aux_l{idx}\" }}\n",
                "{line}"
            ),
            idx = layer_idx,
            upstream = upstream,
            donor_a = donor_a,
            donor_b = donor_b,
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

    let mut rows = Vec::with_capacity(shape.vocab * shape.hidden);
    for token_id in 0..shape.vocab {
        rows.extend_from_slice(projection.row(token_id)?);
    }

    let commitment = write_external(
        &FinalHead {
            params: FinalHeadParams {
                hidden_size: shape.hidden as u32,
                norm_eps: shape.norm_eps,
                softcap,
                norm_weights: page_of_i32s(
                    &weights.get("model.language_model.norm.weight")?.values,
                ),
            },
            projection: paged_i32s(&rows).map_err(|error| error.to_string())?,
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
