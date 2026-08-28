# Issues — gaps without an owner

Index of `docs/issues/`. Last reviewed 2026-08-27.

Same convention as the `raster` repo's `docs/issues/README.md`: an **issue** names a gap and stops
before the design. Every claim is cited to a file and line in this repo. Issues here are about
*this program* — how it uses the model. A gap in the model itself belongs in `raster`'s
`docs/issues/`, and an issue here should say which upstream issue it hands off to.

## Open

| issue | opened | subject | upstream |
| --- | --- | --- | --- |
| [`append-shaped-accumulators`](./append-shaped-accumulators.md) | 2026-08-25 | `mac_weight_page` and `score_key` carry an accumulator in `RecurState` that they never read back — the writes are appends, so they belong in `RecurOutput` and pay only their increment. `score_key`'s carrier grows, making the amplification the iteration count rather than a constant. `accumulate_context` genuinely needs read-back and is stuck. | `raster` `docs/issues/recur-accumulator-slots.md` |
| [`donor-binding-not-templatable`](./donor-binding-not-templatable.md) | 2026-08-27 | The 20 KV-donor bindings resolve by attention type (`l ≡ 4 mod 5` takes layer 14, the rest take 13), which a `chain.repeat` template cannot express — `{ident-1}` is its only arithmetic. They stay hand-written, and multi-token decode multiplies them by `max_new_tokens`. | `raster` `docs/proposals/chain-repeat.md` |

## Closed

| issue | opened | closed | subject |
| --- | --- | --- | --- |
| [`two-recurs-one-draft`](./two-recurs-one-draft.md) | 2026-08-27 | 2026-08-28 | One draft admitted one recur, so every list in a stage output came out the same length — a decode stage could not publish 1 activation row beside a cache of `prompt_len + t + 1`. Closed by `raster`'s `recur-deferred-finalize`. |
