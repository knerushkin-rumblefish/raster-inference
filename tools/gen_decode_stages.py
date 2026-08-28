#!/usr/bin/env python3
"""Emit the chain stages for one decode step, reusing a prefill manifest's externals.

A decode step is 3 + 2*L stages: embed, L aux, L range, finalize, select. Every
external it binds is one the prefill segment already committed to — the weights
are a property of the model, not of the step — so this reads them back out of the
existing manifest rather than inventing new ones.

Usage: gen_decode_stages.py <Raster.toml> <step> [--prev <step>]
"""
import re, sys

def externals(text):
    """external line, keyed by (stage-name, input-name)."""
    out, stage = {}, None
    for line in text.splitlines():
        m = re.match(r'\s*name\s*=\s*"([\w{}]+)"', line)
        if m:
            stage = m.group(1)
        m = re.match(r'\s*(inputs\.(\w+)\s*=\s*\{ *(?:external|input) *=.*\})\s*$', line)
        if m and stage:
            out[(stage, m.group(2))] = m.group(1).strip()
    return out

def donors(text):
    """layer -> donor stage name, straight from the prefill wiring."""
    out, stage = {}, None
    for line in text.splitlines():
        m = re.match(r'name = "prefill_range_l(\d+)"', line)
        if m:
            stage = int(m.group(1))
        m = re.match(r'inputs\.donor_kv = \{ from = "(\w+)" \}', line)
        if m and stage is not None:
            out[stage] = m.group(1)
            stage = None
    return out

def aux_binding(ext, l):
    """`prefill_prepare_aux` may be spelled out per layer or as one repeat block
    over an indexed `[chain.input]`. Both resolve to the same external; decode
    reuses whichever prefill wrote."""
    key = ("prefill_prepare_aux_l%d" % l, "layer")
    if key in ext:
        return ext[key]
    return 'inputs.layer = { input = "aux_layer_%d" }' % l


def main():
    path, step = sys.argv[1], int(sys.argv[2])
    prev = int(sys.argv[sys.argv.index("--prev") + 1]) if "--prev" in sys.argv else step - 1
    text = open(path).read()
    ext, don = externals(text), donors(text)
    layers = len(don)

    # Step 0 inherits from the prefill segment; later steps from the step before.
    if step == 0:
        sel_from, prior = "decode_select_token", "prefill_range_l{l}"
    else:
        sel_from, prior = f"decode_select_t{prev}", f"decode_range_t{prev}_l{{l}}"

    out = [f"\n# ---- decode step {step} ----\n"]
    out.append(f'''[[chain.stage]]
name = "decode_embed_t{step}"
project = "decode-embed"
inputs.selected = {{ from = "{sel_from}" }}
{ext[("input_embedding", "embedding")]}
''')
    for l in range(layers):
        out.append(f'''
[[chain.stage]]
name = "decode_aux_t{step}_l{l}"
project = "prefill-prepare-aux"
inputs.embedded = {{ from = "decode_embed_t{step}" }}
{aux_binding(ext, l)}
''')
    for l in range(layers):
        acts = f"decode_embed_t{step}" if l == 0 else f"decode_range_t{step}_l{l-1}"
        # A sharing layer's donor must be *this* step's cache, already grown by
        # the donor layer earlier in the same step — not the prefill one.
        d = don[l]
        dm = re.match(r"prefill_range_l(\d+)$", d)
        donor = f"decode_range_t{step}_l{dm.group(1)}" if dm else d
        out.append(f'''
[[chain.stage]]
name = "decode_range_t{step}_l{l}"
project = "prefill-range"
inputs.activations = {{ from = "{acts}" }}
inputs.prior_kv = {{ from = "{prior.format(l=l)}" }}
inputs.donor_kv = {{ from = "{donor}" }}
inputs.ple = {{ from = "decode_aux_t{step}_l{l}" }}
{ext[(f"prefill_range_l{l}", "layer")]}
''')
    out.append(f'''
[[chain.stage]]
name = "decode_finalize_t{step}"
project = "prefill-finalize"
inputs.activations = {{ from = "decode_range_t{step}_l{layers-1}" }}
{ext[("prefill_finalize", "head")]}

[[chain.stage]]
name = "decode_select_t{step}"
project = "decode-select-token"
inputs.logits = {{ from = "decode_finalize_t{step}" }}
''')
    sys.stdout.write("".join(out))

main()
