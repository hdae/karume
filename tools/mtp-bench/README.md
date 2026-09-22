# mtp-bench — what speculative decoding is actually worth

A CLI that measures the **end-to-end payoff of speculative decoding** (MTP, ADR 0096) for Gemma 4
E2B: it runs three configurations alternately **in one process, on one pipeline, with one prompt**,
and writes one JSON line.

| Mode     | `Gemma4SequenceOptions.speculative` | What it answers                                                                                                                            |
| -------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `plain`  | `false`                             | The baseline: one token per run                                                                                                            |
| `always` | `"always"`                          | The **ceiling** — speculation on every cycle, no gate: what the drafter is worth at best                                                   |
| `auto`   | `true` (the library default)        | The **shipping** path — the self-funding gate falls back to a decode-shaped step whenever speculation is losing: what a user actually gets |

Measuring the paths in separate processes would compare runs that differ in driver clock state, PLE
residency and shader compilation as well as in the thing under test, so the difference could not be
attributed. One process, alternating turns, is the whole point of this tool.

Kernels and the generation loop are **not modified**. Per-run wall clock is taken by wrapping
`Session.prototype.run` from the tool, and what each run was (`prefill` / `decode` / `draft` /
`verify`) is named by the pipeline's own observation seat (`onRunDiagnostics`), which fires in the
synchronous region right after the run returns.

## Usage

```
deno run -A tools/mtp-bench/main.ts --workload <name> --sampler <greedy|recommended> [options]
```

| Option                           | Default                                                | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| -------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--source <dir>`                 | `models/karume-gemma4`                                 | Distribution mirror that carries the drafter weights, read through `denoDirectory`                                                                                                                                                                                                                                                                                                                                                             |
| `--workload <name>`              | required                                               | `extract` / `summarize` / `dialogue` / `freeform` (see below)                                                                                                                                                                                                                                                                                                                                                                                  |
| `--sampler <name>`               | required                                               | `greedy` = `{ temperature: 0 }`; `recommended` = the mirror's declared sampler plus `--seed`                                                                                                                                                                                                                                                                                                                                                   |
| `--seed <int>`                   | `42`                                                   | Only used by `--sampler recommended`. Passing it together with `--sampler greedy` is rejected: a knob that cannot take effect is never accepted silently                                                                                                                                                                                                                                                                                       |
| `--k <int>`                      | library default: the step count baked into the drafter | Drafts per cycle. Omitting it passes `speculative: {}`, so the drafter graph's own step count decides. The range gate lives in the library                                                                                                                                                                                                                                                                                                     |
| `--new-tokens <int>`             | `200`                                                  | `maxNewTokens` per turn                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `--capacity <int>`               | `8192`                                                 | KV capacity of every turn's sequence. Above the mirror's default 4096 because the long-context prompt is ≈4.8K tokens                                                                                                                                                                                                                                                                                                                          |
| `--document-chars <int>`         | `20000`                                                | Character budget for the document workloads, cut at a paragraph boundary. Rejected for the other workloads                                                                                                                                                                                                                                                                                                                                     |
| `--rounds <int>`                 | `3`                                                    | Rotation repetitions after the warm-up. One rotation is 8 turns, so the default is 27 turns in total (3 warm-ups + 24)                                                                                                                                                                                                                                                                                                                         |
| `--max-resident-ple-bytes <int>` | whole PLE tables                                       | Host RAM budget for the PLE tables. The default is derived from the mirror's index so that **no block is ever re-read** during a turn                                                                                                                                                                                                                                                                                                          |
| `--gemv-rows-target <int>`       | runtime default: 16384                                 | Overrides the runtime's `linearGemvRowsThreadTarget` (default 16384, the RTX 3080 Ti saturation point). Lower it on GPUs with fewer cores so the M=4 verify uses taller row blocks (fewer weight re-reads). Static — the tool records it in `config`                                                                                                                                                                                           |
| `--gate-early-leave <number>`    | library default: **off**                               | Self-funding gate: the speculate-side early leave. When given, a single block whose ratio exceeds `leave + earlyLeave` leaves speculation without waiting for `confirm` consecutive blocks. Off by default: on a reused sequence it misfired on 3 of 7 turns of a task speculation was winning, and that harm recurs every turn, while the gain is one block per sequence lifetime. **Affects the `auto` mode only** — `always` builds no gate |
| `--gate-burst-abort <number>`    | library default: `0.15`                                | Self-funding gate: the plain-side "strong loss" margin. An exploration burst whose ratio exceeds `leave + burstAbort` at or after its `burstMin`-th cycle is abandoned instead of run to full length. `auto` only                                                                                                                                                                                                                              |
| `--gate-burst-min <int>`         | library default: `min(4, burst)` = `4`                 | Self-funding gate: the earliest cycle at which a losing exploration burst can be abandoned. Raise it to make bursts run closer to full length (a slower exit, a more precise return). `auto` only                                                                                                                                                                                                                                              |
| `--gate-explore-base <int>`      | library default: `16`                                  | Self-funding gate: the base exploration interval — in steady `speculate` the number of cycles between the plain steps that keep `W1` fresh, and in steady `plain` the initial backoff. `auto` only — must not exceed `exploreMax` (512 by default)                                                                                                                                                                                             |
| `--out <file.jsonl>`             | —                                                      | Append the JSON line to this file as well (stdout always gets it). The parent directory must exist — it is checked before the model is loaded, and never created                                                                                                                                                                                                                                                                               |
| `--gpu-timing` (switch)          | off                                                    | Collect per-op GPU time via `acquireGpu({ gpuTiming: true })`                                                                                                                                                                                                                                                                                                                                                                                  |
| `--warm` (switch)                | off                                                    | One sequence **per mode**, reused across that mode's turns: the conversation grows instead of restarting, so the gate is only cold in the mode's first turn (see below)                                                                                                                                                                                                                                                                        |

Unknown options fail loudly: a mistyped knob that silently fell back to a default would make the
recorded configuration disagree with what was actually measured.

The `--gate-*` knobs are partial: whatever is not passed keeps the library's own default, and only
what was passed is recorded in `config.gate` (`null` when none were). Gate v2 — the revision that
preceded the current defaults — is `--gate-burst-abort 10 --gate-burst-min 8 --gate-explore-base 8`,
and gate v3 — the revision whose speculate-side early leave is now off by default — is
`--gate-early-leave 0.15`: the A/B pair to run against the defaults on the same script.

Progress goes to stderr, one line per turn; **stdout carries only the final JSON line**.

```sh
# The headline number: greedy, long-context extraction
deno run -A tools/mtp-bench/main.ts --workload extract --sampler greedy \
  --out outputs/bench/karume-gemma4/2026-09-09_mtp-stage4/turns.jsonl

# Sampling as shipped (temperature > 0 must still produce the same token ids)
deno run -A tools/mtp-bench/main.ts --workload dialogue --sampler recommended --seed 42

# Per-op GPU breakdown (a separate process — see the warning below)
deno run -A tools/mtp-bench/main.ts --workload freeform --sampler greedy --gpu-timing

# Multi-turn: one conversation per mode, so the gate is cold only once
deno run -A tools/mtp-bench/main.ts --workload freeform --sampler greedy --warm
```

## Protocol

One process = one configuration (the convention of `tools/ram-peak/measure.ts`).

1. Acquire the GPU, wrap `Session.prototype.run`, open the pipeline with the drafter
   (`speculative: {}`, plus `k` when `--k` was given and `gate` when any `--gate-*` was) and the full
   PLE budget. The gate knobs reach the `auto` turns only — an `always` turn builds no gate, so they
   have nothing to act on there.
2. Build the workload's chat messages, encode them once with `gemma4ChatPrompt`. All three modes send
   the **same prompt token ids** from the same starting position; each turn gets a fresh sequence
   (with `--warm`, the first turn of each mode does, and the rest continue that conversation).
3. **Warm up** with one turn of each mode. They are recorded (`warmup: true`) but excluded from the
   summary: the first turns include shader translation and params construction.
4. Run `--rounds` repetitions of the **rotation** — odd rounds `plain`, `always`, `always`, `plain`,
   `plain`, `auto`, `auto`, `plain`; even rounds swap the two speculative pairs (`auto` first) — which
   cancels the order effect (later turns being systematically faster or slower: on the RTX 3080 Ti a
   3-minute process warms up by about +4% in `plain`). Each speculative mode is bracketed by the same
   number of `plain` turns (two on each side), the two of them are never adjacent, and alternating
   their seats across rounds keeps the monotone drift from landing on one of them.
5. Summarise with **medians**, never means: a single turn can spike (PLE shard re-reads, clock state
   changes), and a mean carries the spike into the ratio.

**Without `--warm`, the gate starts cold in every `auto` turn.** One turn is one sequence, and the
gate lives as long as the sequence, so it carries neither its moving averages nor its exploration
counter across turns: every `auto` turn pays the exploration cost again from scratch. A real
application (`Gemma4ChatSession`) reuses one sequence across the whole conversation, so
`speedupAuto` measured this way is the **pessimistic** side of what shipping code gets. `--warm` is
the other end of that range — see below.

### `--warm`: one conversation per mode

With the switch, each mode gets **one sequence, created on its first turn and disposed after the
last**, and every later turn of that mode appends **a different user message** as a new user turn —
one growing conversation, the same shape a multi-turn chat has. The gate therefore starts cold only
once per mode (in the warm-up turn), which is what a real conversation does; the cold figure and this
one bracket what shipping code sees.

- **Every turn asks something new.** The follow-ups are `warmFollowUps` in `workloads.ts`: twelve
  utterances for `freeform` (a different creative request each time — a poem, a letter, a fable) and
  twelve for `dialogue` (the performance conversation continued, one new question per turn). Turn
  `n` of a mode sends follow-up `n − 1`.
- **Why not the same message every turn?** That is what the switch did at first, and it measures the
  wrong thing: the model copies its own previous answer, the drafter predicts text it has already
  seen, and acceptance jumps from 1.63 to 3.9 tokens per cycle (measured on the RTX 3080 Ti,
  `docs/research` 2026-09-09 §6.5). `--warm` exists to warm the **gate** on a workload where
  speculation still _loses_; a repeated message turns the workload into a copying task instead, so
  the gate is no longer being asked the question it was built for.
- **Twelve follow-ups mean `--rounds 3` is the ceiling.** A mode can run `1 + follow-ups` turns at
  most (the first turn sends the whole conversation and uses no follow-up), and `plain` takes four
  turns per rotation, so the default gives it exactly 13. A higher `--rounds` is rejected before the
  first turn runs (after the model has loaded — the same point as the capacity check), naming the mode and the turn count it would need.
- Each follow-up is drawn with **`gemma4ChatTurn`**, the same helper `Gemma4ChatSession` uses for the
  same purpose — never a hand-written template string, because the chat spellings belong to that
  function. All of them are drawn at start-up, so no turn pays tokenisation inside its wall clock.
- A turn that ran into `--new-tokens` did **not** close its model turn, so its frontier is an
  ordinary content token rather than the end-of-turn one that `gemma4ChatTurn`'s delta assumes. The
  tool then prepends the end-of-turn id itself, producing exactly the ids the model would have
  emitted had it stopped there. (`Gemma4ChatSession` instead drops the KV and redraws the whole
  conversation; that is the right call for an app, but it would defeat the point of this switch.)
- A turn that stopped on **any other stop token** (a distribution `<eos>`, a requested
  `<|tool_response>`) is **rejected before the next turn is issued**, naming the token id and the
  mode. Prepending the end-of-turn id there would push `body <eos> <turn|> delta` into the KV — ids
  no redraw of the conversation would ever produce — and not prepending it breaks what
  `gemma4ChatTurn`'s delta assumes, so neither is measurable.
- **Capacity is checked after the model loads and before the first turn runs** (the check needs the
  tokenizer for the follow-up turns). Needed positions for a mode with `n` turns are
  `prompt + n × new-tokens + (n − 1) × delta − 1`, where `delta` is the **longest** follow-up plus
  the end-of-turn id — the pessimistic side, since the follow-ups differ in length. `plain` runs four
  turns per rotation against the two speculative modes' two, so `plain` is always the binding one (13
  turns at the default `--rounds 3`). `freeform` (33 prompt tokens) and `dialogue` (230) fit in the
  default 8192. `extract` / `summarize` have no follow-up list at all and are rejected by name: their
  prompt is a document (≈4.8K), so a growing conversation of them would not fit either. That is
  intended — `--warm` exists for the workloads where the gate _loses_, which are the short-prompt
  ones.
- **The summary only uses own-turn numbers that all three modes reached** (`ownIndex ≤` the smallest
  per-mode turn count, reported as `summary.ownTurnLimit`). Without that, the ratio would compare
  `plain` turns late in a long conversation against `always` turns early in a short one, since
  `plain` accumulates context twice as fast. Token-id identity is likewise compared per own-turn
  number.
- `identity.contextAligned` says whether the three modes' `contextTokens` agree at **every own-turn
  number the summary used**, and `identity.firstContextMismatchTurn` names the first one that did not
  — once a mode's token ids part, the turns after it no longer start from the same position. It is
  recorded rather than enforced (the same treatment as a diverged column); the closing stderr line
  prints `ctx aligned yes` or `ctx aligned NO@<n>`.
- Three sequences are alive at once, so **KV residency is 3 × `--capacity`** instead of one.
- Per-mode wall clocks grow over the turns because the context does. Compare across modes at the
  same own-turn number, not across turns.
- On stderr, each turn line gains `#<ownIndex> ctx <contextTokens>` and the closing line names the
  own-turn limit together with the turn count each mode contributed.

### Denominators

- `msPerToken` divides by **`tokens - 1`**. The first token comes out of the prefill run, so the
  decode/cycle work of a turn covers `tokens - 1` tokens. Dividing by `tokens` makes both modes look
  faster and distorts the ratio.
- `tokensPerCycle` is **`delivered / cycles`**, the tally's own count of what the cycles committed.
  The older `(accepted + cycles) / cycles` assumed every cycle commits `1 + a` tokens, which
  overstates the numerator by one for each cycle that was cut short by a stop token. In `auto`, the
  gate's decode-shaped steps are not cycles, so they are in neither the numerator nor the
  denominator.
- `hostMsPerToken` (**every mode**) is `(generationMs − decode/draft/verify wall) / (tokens − 1)`.
  That remainder is the time spent outside `Session.run` — sampling, detokenising, event delivery,
  PLE reads — the part speculation cannot remove. The `prefill` wall is **not** subtracted: it lands
  before the first token, so it is not inside `generationMs` to begin with.
- `hostMsPerCycle` and `cycleMs` exist for `always` only. In `auto` a turn mixes cycles with
  decode-shaped steps while `cycles` counts only the former, so a per-cycle wall would have a
  numerator and a denominator that describe different work. `hostMsPerToken` is the only host-time
  figure comparable across all three modes.

**The `plain` baseline is measured on a device that holds the drafter.** One process with alternating
turns is what makes the two paths comparable at all, and that process opened the pipeline with the
drafter, so the `plain` turns also run with the drafter's weights resident and with whatever pressure
the drafter puts on the buffer pool. A `plain` turn here is therefore not the same thing as a `plain`
turn of a drafter-less distribution: if this one is the slower of the two, `speedup` is overstated by
that difference. The MTP head is small next to the target model, so the effect is expected to be
small — but that expectation is not measured.

**`tokensPerCycle` and `msPerToken` do not count the same tokens.** `delivered` is accumulated when a
cycle's acceptance is decided, so a turn whose consumer stopped reading mid-cycle has
`delivered > tokens - 1`. `msPerToken` is per token the caller received, `tokensPerCycle` is per
token the cycles committed; the two denominators are not interchangeable, and multiplying one by the
other does not give `cycleMs`.

## Workloads

Prompt material comes only from **git-tracked files inside the repository** (`outputs/` is untracked,
so reading from there would mean the same script produced a different prompt), and the document is
cut at a **paragraph boundary** (the last blank line before the limit). Cutting on a character count
ends mid-sentence and makes generation degenerate into a repeated phrase, which changes the
acceptance rate for a reason that has nothing to do with the drafter. Both rules are the ones
`tools/export-recipes/gemma4/export_drafter.py` uses for the drafter's golden material.

| Name        | Shape                                                                                                             | Why it is here                                                          |
| ----------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `extract`   | One user turn: instruction + `tools/exporter/README.md` (cut to `--document-chars`) + "list every shell command…" | Long context, copied verbatim from the prompt — the high-acceptance end |
| `summarize` | Same instruction and same document + "summarize in 10 bullet points"                                              | Same context, own words — acceptance drops                              |
| `dialogue`  | user → a **fixed** assistant answer (a baked constant) → "expand on the second point"                             | A continuing conversation, the practical shape                          |
| `freeform`  | One short creative request (the same text as the drafter golden's short case)                                     | The thinnest context — the low-acceptance end                           |

No system message is added in any of them. The assistant turn of `dialogue` is a constant in
`workloads.ts`: if it were generated, the prompt would differ between runs and neither ABBA nor the
median could cancel that.

## Output

One JSON line with these top-level keys:

| Key       | Contents                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tool`    | `"mtp-bench"`                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `host`    | `os`, `arch`, `deno`, and the `adapter` (`vendor` / `architecture` / `device` / `description`)                                                                                                                                                                                                                                                                                                                                                                |
| `config`  | The effective value of every option. Knobs that are not in effect are `null` (`k` is `null` when the drafter's own step count was used; `gate` is `null` when no `--gate-*` knob was passed, and otherwise holds exactly the ones that were). `config.asset` identifies what was measured: `defaultModel`, `defaultQuant`, and `manifestSha256` — the SHA-256 of the manifest body, which pins the asset because a distribution carries no version of its own |
| `prompt`  | `tokens` (the encoded prompt length) and the `messages` that produced it                                                                                                                                                                                                                                                                                                                                                                                      |
| `turns`   | One record per turn, warm-ups included: `mode` (`plain` / `always` / `auto`), round, `warmup`, `ownIndex` / `contextTokens` (see below), `tokens`, `stopReason`, the token `ids` and their decoded `text`, `turnMs` / `firstTokenMs` / `generationMs`, per-kind run counts and wall totals, the `trace` buckets (see below), and `speculation` for `always` and `auto` turns                                                                                  |
| `summary` | See below                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `gpu`     | Present only with `--gpu-timing`: the per-op GPU breakdown, folded as `gpu[mode][kind]` (see below)                                                                                                                                                                                                                                                                                                                                                           |

`summary` (warm-ups excluded, every value a median over turns):

| Field                                 | Meaning                                                                                                                                                                                                                                                                                                           |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `plain` / `always` / `auto`           | `turns`, `msPerToken`, `generationMs`, `turnMs`, `firstTokenMs`, `hostMsPerToken`                                                                                                                                                                                                                                 |
| `<mode>.runs[kind]`                   | `countPerTurn` and `msPerRun` per run kind. A kind with no runs has no `msPerRun` field (writing `0` would read as "measured, and it was 0 ms")                                                                                                                                                                   |
| `<mode>.hostMsPerToken`               | `(generationMs − decode/draft/verify wall) / (tokens − 1)` — see Denominators                                                                                                                                                                                                                                     |
| `<mode>.trace`                        | Phase buckets — see below                                                                                                                                                                                                                                                                                         |
| `always` / `auto`.`tokensPerCycle`    | `delivered / cycles`                                                                                                                                                                                                                                                                                              |
| `always` / `auto`.`k`                 | The effective step count, `acceptedHistogram.length − 1` (the tally has one bucket per acceptance count, `0..k`). Turns that disagree are an error, not a median                                                                                                                                                  |
| `always` / `auto`.`acceptedHistogram` | Element-wise sum of the per-turn histograms (index = accepted drafts in a cycle)                                                                                                                                                                                                                                  |
| `always.cycleMs`                      | `generationMs / cycles` (`always` only)                                                                                                                                                                                                                                                                           |
| `always.hostMsPerCycle`               | `(generationMs − draft wall − verify wall) / cycles` (`always` only)                                                                                                                                                                                                                                              |
| `auto.plainSteps`                     | Median number of decode-shaped steps the gate took in a turn. `0` means the gate never fired; the field's absence would mean there was no gate                                                                                                                                                                    |
| `auto.switches`                       | Median number of speculate ↔ plain transitions in a turn (a one-off exploration probe is not a transition)                                                                                                                                                                                                        |
| `speedup`                             | `plain.msPerToken / always.msPerToken` — the ceiling                                                                                                                                                                                                                                                              |
| `speedupAuto`                         | `plain.msPerToken / auto.msPerToken` — **the acceptance figure: "does leaving it on ever cost anything?"** Below 1 means the gate failed to stop a loss                                                                                                                                                           |
| `identity`                            | `plainConsistent`, `alwaysConsistent`, `autoConsistent`, `identical` / `firstDivergence` (plain vs `always`), `identicalAuto` / `firstDivergenceAuto` (plain vs `auto`) — and with `--warm` also `contextAligned` / `firstContextMismatchTurn` (the per-own-turn context length agreement across the three modes) |
| `ownTurnLimit`                        | `--warm` only: the largest own-turn number the summary used (see `--warm` above). Absent without the switch                                                                                                                                                                                                       |

Every turn record carries two fields for `--warm`, and they are written in either mode so that the
record has one shape: `ownIndex` is the 1-based position of that turn **within its own mode**
(the warm-up is 1), and `contextTokens` is what the conversation occupied **before** the turn was
issued (`GenerationSequence.used`, so a pending frontier token counts). Without `--warm` every
`contextTokens` is 0, since each turn gets a fresh sequence.

With `--warm`, `identity` compares `plain` against `always` / `auto` **per own-turn number**, and
`firstDivergenceTurn` / `firstDivergenceAutoTurn` name the first number that parted (the existing
`firstDivergence*` fields still give the index within that turn's ids). The three `*Consistent`
fields are **absent** there: turns of one mode continue a conversation rather than repeat a prompt,
so "did the same prompt produce the same ids twice" is not a question that can be asked — and
`false` would read as a broken invariant.

`identity` is a correctness check, not a performance one: speculation is a speed-only knob, so
`always` must produce exactly the `plain` token ids. If `identical` is false, read that before the
ratio — the ratio is comparing two different generations.

`identicalAuto` is **not** the same kind of invariant. The gate switches on wall-clock measurements,
and the default seat (`stateAttentionReduce: "parallel"`) reduces ①QK in a different order for the
M=4 verify and the M=1 decode shape, so two near-equal logits can split the argmax
(`docs/limitations.md`). A false `identicalAuto` with a true `identical` is that effect, not a bug in
the drafter; `firstDivergenceAuto` says where the runs parted.

### Phase buckets (`trace`)

`runs[kind]` splits runs by their **shape** (prefill / decode / draft / verify), which is not enough
for an `auto` turn: the gate's plain steps are decode-shaped, so they land on top of a `plain` turn's
runs, and an exploration burst's verify looks exactly like steady speculation. `turns[].trace` (present
for every mode) and `summary.<mode>.trace` split the same runs by **situation**, from the wall clock and
gate state the generation seat reports for each run (`GenerationRunPhase`: `wallMs`, `delivered`, `gate`).

Every bucket carries `runs`, `ms` (the sum of `wallMs`) and `delivered` (tokens the run committed —
the cycle's confirmed count for verify, 1 for decode). Inside a speculative turn the wall runs from the
head of the cycle (before the draft run is issued) to just after the acceptance decision — the same
value the gate feeds its own decision, so a slow consumer cannot inflate it. A `plain` turn's decode
run is measured over the same span — from the head of the step (before the derived inputs) to just
after sampling — so the `plain` bucket is comparable across modes.

| Bucket                 | The runs in it                                                                                                                                                                                                  |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `speculate`            | verify, with no gate (`always`) or in steady `speculate` with no switch yet in this turn                                                                                                                        |
| `speculateAfterReturn` | verify, in steady `speculate` after at least one switch in this turn — speculation resumed after a plain stretch                                                                                                |
| `burst`                | verify, in steady `plain` — an exploration burst. The cycle whose observation _caused_ the exit is here too (the gate state is read after the observation), so expect one such run per switch                   |
| `w1Probe`              | decode, in steady `speculate` — the plain steps the gate takes to keep `W1` fresh (one per exploration interval)                                                                                                |
| `plain`                | decode, with no gate (a `plain` turn) or in steady `plain` (the gate's steady state)                                                                                                                            |
| `unmeasured`           | Runs the gate kept out of its own wall statistics (`gate.measured === false`): the first cycle of every turn and the budget-tail forced plain. The wall is still counted here — it just did not feed a decision |
| `cold` / `rest`        | A **second, independent** split of the same runs: the turn's first 8 generation runs and everything after them. Reads how far the cold miss reaches (PLE shards, the first PreparedPlan / bind group)           |

Within one turn (`turns[].trace`) the six situation buckets are mutually exclusive and sum to
`cold + rest`; the per-field medians in `summary.<mode>.trace` do not add up that way. `prefill` and
`draft` runs are in neither: prefill belongs to `firstTokenMs`, and a draft run sits _inside_ the cycle
wall.

`firstExitRun` is the 1-based index — in the same decode/verify sequence the buckets count — of the run
at which `gate.switches` first reached 1. It is absent when the gate never switched (`0` would read as
"it switched before the first run"). In `summary.<mode>.trace` every bucket field is a median over the
measured turns taken **per field** (`runs`, `ms` and `delivered` separately, so `ms / runs` is not any
one turn's per-run wall), and `firstExitRun` is the median over the turns that switched at all.

A run with no `wallMs` is an error, not a zero: the tool refuses to print a breakdown that has quietly
lost a run's time.

## `--gpu-timing`

With the switch, each run's `lastRunTiming` is accumulated (warm-ups excluded) into
`gpu[mode][kind]`: `runs`, `msPerRun`, `dispatchesPerRun`, `clampedNegativeSamples`, and the top 12
pipeline keys by GPU time with their own `msPerRun` and `dispatchesPerRun`. Both levels are ordered
`plain` / `always` / `auto` and `prefill` / `decode` / `draft` / `verify`.

The **mode** level is what makes the table readable. The gate's plain steps and its `W1` probes are
decode-shaped runs, so folding by `phase.kind` alone puts them in the same bucket as a `plain`
turn's decodes (the observation seat does not distinguish them — no branch was added to the public
`Gemma4RunPhase`). Split by mode, `gpu.auto.decode` is the gate's own M=1 runs and `gpu.plain.decode`
is the non-speculative baseline, so the same kernel can be compared key by key between the two. The
mode level does not separate the gate's plain steps from its `W1` probes — both are `auto` decodes;
`summary.auto.trace` (`plain` vs `w1Probe`) is where that split lives.

A `(mode, kind)` pair with no runs has **no field at all** — `always` has no `decode`, `plain` has
neither `draft` nor `verify` — for the same reason `summary.<mode>.runs[kind].msPerRun` is absent
there: `0` would read as "measured, and it was 0 ms". A mode with no runs at all is missing
entirely rather than present as `{}`.

The stderr tail prints one line per `(mode, kind)` (`mode/kind`, runs, ms per run, dispatches per
run). The per-key table is only in the JSON.

**Never compare a timing-on wall clock with a timing-off one.** A timing-enabled device opens one
pass per dispatch, so the wall clock (and therefore `speedup`) grows. Take the ratio from a run
without the switch and the breakdown from a run with it. `clampedNegativeSamples` above zero means
the driver's timestamps were non-monotonic and the breakdown itself is suspect.

On macOS (Metal) the switch prints a warning: timestamp collection loses the device there
(`docs/limitations.md`). The tool does not refuse — the constraint belongs to the OS/driver stack, so
no removable gate is added to karume.

## Notes

- Generated files belong under `outputs/bench/<model>/<YYYY-MM-DD>_<purpose>/`, which is untracked
  (`docs/assets-layout.md`). Numbers meant to last go into `docs/research/`, and adoption decisions
  into `docs/perf-ledger.md`.
- The PLE budget default is computed from the `ple_index` asset inside the mirror's `model`
  container the same way
  `packages/models/tests/helpers/ple-budget.ts` does it. Byte budgets, not shard counts: shard width
  changes with the asset generation, so "N shards" means a different amount of RAM per generation
  (ADR 0085).
- `--k` is passed through to the library, which owns the range gate. This tool does not re-implement
  a second gate for the same knob.
