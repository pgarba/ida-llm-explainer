<p align="center"><img src="logo.svg" alt="LLM Explainer logo" width="128"></p>

# LLM Explainer

An IDA Pro 9.3 plugin that asks a locally-running [llama.cpp](https://github.com/ggml-org/llama.cpp)
server (`llama-server`) to explain the function you're looking at — in either the Hex-Rays
pseudocode view or the plain disassembly view — and streams the answer into a small,
non-modal review dialog. Nothing is written to your database until you explicitly click Accept.

![LLM Explainer result dialog with proposed name, signature, variable renames, and called-function renames](example.png)

The dialog above shows a real run against a license-check function: the model explained what it
does, proposed a clearer name and signature, renamed every local variable, and — after fetching
the code of the functions it calls — proposed names for those too. Each suggestion is its own
checkbox, editable before anything is written.

## Features

- **Right-click, explain**: works from the pseudocode view, the disassembly view, or a
  configurable hotkey (default `Ctrl-Alt-E`).
- **Live streaming**: the answer streams in as it's generated. For "thinking"/reasoning models
  (e.g. Qwen3) the chain-of-thought is shown separately, in gray italics, from the real answer.
- **Human in the loop**: every result is Accept / Reason More / Cancel — the model never writes
  to the database on its own.
- **Call-graph awareness**: optionally follow called functions recursively (up to a configurable
  depth) so the model sees callee code up front, and/or let the model ask for a specific callee's
  code on demand mid-conversation (`REQUEST_CODE: <name-or-address>`), which the plugin fetches
  and feeds back automatically.
- **String/global context**: referenced string literals and named globals are included in the
  prompt, since they're often the strongest clue to a function's purpose.
- **Rename & retype suggestions**: the model can propose a better function name, a full C
  signature (return type + argument types/names, Hex-Rays only), local variable renames, and
  renames for *called* functions it actually examined and are still under a default name —
  each shown as an editable, opt-in checkbox before you accept.
- **Struct detection**: when the pseudocode accesses a pointer at multiple constant offsets in a
  way that suggests an undefined structure, the model can define a proper struct type and apply
  it — to a function argument via the signature, or to a local variable directly — again as an
  editable, opt-in suggestion.
- **Batch mode**: pick a set of functions (filterable checklist), process them, then review and
  apply the results in bulk — same human-in-the-loop guarantee as the single-function flow.
- **Recursive auto-accept mode**: `Explain function with LLM (recursively)` also explains the
  target function's direct callees (depth 1 only) and applies every result automatically, with
  no review step — the one exception to the human-in-the-loop rule above. Still uses the same
  conservative apply defaults as a manual Accept, is capped by its own "Max recursive callees"
  setting, and shows a live, cancellable progress dialog. Use with care since it writes to the
  database unattended.
- **Multi-server parallelism with priority + failover**: configure more than one `llama-server`
  instance (list order = priority) and batch/recursive explain will run up to one function per
  server concurrently — N servers gives roughly an Nx speedup over a single server. The
  interactive single-function explain uses the first listed server. Either way, if a server
  refuses/drops the connection or returns HTTP 502/503/504, that request automatically retries
  against the next server on the list before giving up, so one offline server doesn't fail your
  requests.
- **LLM-guided CFG recovery for obfuscated functions**: right-click in the disassembly view →
  **Trace/Recover CFG...**, give it a start address, and it walks the function's basic blocks
  forward — auto-continuing through anything with a single successor, and at each real branch
  or indirect-jump decision point first trying a fast, deterministic constant-propagation pass
  (x86/x64 only) before ever asking the model. That pass resolves classic opaque predicates (a
  dispatcher/state variable set to a known constant earlier in the trace — one side real, the
  other dead) and also recognizes genuine data-dependent branches — a comparison against an
  incoming argument, something read through one, or a called function's result — as ordinary,
  non-obfuscated conditional logic, marking every side real automatically without needing to
  know what the argument/call result actually contains; only when it can't confidently resolve
  something does it fall back to the LLM,
  so LLM calls stay proportional to actual obfuscation complexity rather than every branch in
  the function. As it walks it also corrects any instruction boundaries IDA's original analysis
  got wrong (undefine + recreate, never touching a byte — the same fixup you'd do by hand with
  U then C), since obfuscated code routinely has real jump targets landing mid-instruction
  relative to IDA's initial linear-sweep guess; this happens immediately, not deferred to
  Accept. A live, cancellable transcript shows progress, alongside an optional live graph view
  (a native IDA graph, colored the same green/red/amber as the eventual disassembly marking,
  filling in node by node as the trace runs — double-click a node to jump there); once the
  trace finishes (or a configurable block-count safety cap is hit), a review table lets you
  check/uncheck each decided block before Accept. By default Accept only colors the disassembly
  and adds a comment — no byte patching. An opt-in **"Also patch bytes"** checkbox on the review
  screen (unchecked by default, with a confirmation prompt) additionally NOPs out confirmed dead
  code and redirects confirmed opaque-predicate jumps straight to their real target, so Hex-Rays
  can decompile the recovered function cleanly — see [below](#patching-the-recovered-cfg) for
  exactly what it does and does not touch.

![Trace/Recover CFG live view: the recovered control-flow graph, colored real/dead, next to the live transcript explaining each block's verdict](tracer.png)

The screenshot above shows a real trace against an obfuscated function: the live graph fills in
as each block is decided (green = real path, red = decoy), while the transcript streams the
model's reasoning behind every REAL/DEAD verdict — here, opaque predicates built on flag tricks
(`xor dl, 0` clearing the overflow flag, `xchg ax, ax` fixing the parity flag) that fall outside
what the constant-propagation pass models on its own, so the LLM works them out directly from
the flag semantics instead.

## Requirements

- IDA Pro 9.3 or later (PySide6 is bundled with IDA — no extra Python packages to install).
- One or more running `llama-server` instances reachable over HTTP (default
  `http://127.0.0.1:8080`). Configuring more than one lets batch/recursive explain run in
  parallel across all of them.
- The Hex-Rays decompiler is optional. If it isn't available for the current architecture, the
  plugin automatically falls back to a plain disassembly listing.

## Installation

Copy `llm_explainer.py` into one of IDA's plugin directories:

- **Per-user** (recommended, no admin rights needed):
  `<IDA user dir>\plugins\llm_explainer.py`
  On Windows this is typically `%APPDATA%\Hex-Rays\IDA Pro\plugins\llm_explainer.py`.
- **Global** (all users of this IDA install):
  `<IDA install dir>\plugins\llm_explainer.py`

Restart IDA (or reload plugins) afterward.

Alternatively, install via [`hcli`](https://hcli.docs.hex-rays.com/), Hex-Rays' plugin manager,
using the packaged `ida-plugin.json` described below.

## Packaging (hcli / Hex-Rays plugin repository)

This repo follows the [official plugin packaging format](https://hcli.docs.hex-rays.com/reference/plugin-packaging-and-format/):
`ida-plugin.json` sits at the repo root next to the entry point (`llm_explainer.py`), which is
already the flat, single-file layout the format expects — no subfolder needed.

To build the distributable archive:

```sh
python package.py
```

This writes `dist/llm-explainer-<version>.zip` with `ida-plugin.json` and `llm_explainer.py` at
the archive root. Attach that zip as a GitHub release asset — don't rely on GitHub's
auto-generated "Source code (zip)" link, since it wraps everything in a nested
`<repo>-<tag>/` folder and would put the manifest one level too deep. Validate with:

```sh
hcli plugin lint dist/llm-explainer-<version>.zip
```

Note: `ida-plugin.json`'s `categories` currently lists `"ai"` as a best guess — the docs page
doesn't publish the full list of accepted category values, so double-check that against
`hcli plugin lint` (or hcli's own validation) before publishing, and adjust if it's rejected.

## Usage

### Explain a single function

1. Open a function in the pseudocode or disassembly view.
2. Right-click → **LLM Explainer → Explain function with LLM...** (or press the hotkey).
3. Watch the answer stream in. If the model needs to see a called function's code to answer
   accurately, it will fetch it automatically (up to the configured limit) — you'll see this
   noted in the transcript.
4. Review the proposed comment and any suggested rename / signature / variable renames (each has
   its own checkbox and is editable before you commit).
5. Click **Accept & Add Comment** to write everything to the database, **Reason More** to ask a
   follow-up question, or **Cancel** to discard.

### Batch-explain multiple functions

1. Right-click in the **Functions** window → **LLM Explainer → Batch Explain Functions...**
   (or Edit → Plugins → Batch Explain Functions...).
2. Filter and check the functions you want processed. Nothing is pre-selected.
3. The progress dialog processes up to one function per configured server concurrently (see
   "Server base URL(s)" below) and shows live status per function, including which server is
   handling it.
4. Once finished, check/uncheck rows (successful ones are checked by default) and click
   **Apply Selected** to write all of them in one batch. There is no follow-up chat in batch
   mode — reopen the single-function flow on a specific function if you want to refine it further.

### Recursive auto-accept

Right-click a function → **LLM Explainer → Explain function with LLM (recursively)...**. This
explains the target function plus its direct callees (depth 1 only — callees of callees are not
included), and applies every successful result immediately as it completes, with no per-function
review. A progress dialog still shows live status and can be cancelled mid-run; the callee count
is capped by the separate "Max recursive callees" setting (default `10`) precisely because this
mode writes to the database unattended.

### Trace/Recover CFG (obfuscated functions)

1. In the **disassembly view**, put the cursor on the function's entry point (or any address
   you want to start from) and right-click → **LLM Explainer → Trace/Recover CFG...**.
2. Confirm or edit the start address, optionally leave "Show live CFG graph while tracing"
   checked (a native IDA graph view opens alongside, filling in as blocks are decided — use
   **Show Graph** at any point later to reopen it if closed), then click **Start**.
3. Watch the live transcript as it walks basic blocks, auto-continuing through single-successor
   blocks and, at each real branch/indirect jump, first trying to resolve it automatically via
   constant propagation before ever pausing for an LLM round-trip. As it walks it also fixes up
   any instruction boundaries IDA's original analysis got wrong (undefine + recreate, no bytes
   changed) — this happens immediately, since it's just correcting IDA's own analysis, not an
   LLM opinion. **Cancel** at any time; any boundary fixes already made stay (they're harmless
   corrections either way), but no REAL/DEAD/UNRESOLVED coloring/comments are written until you
   explicitly accept in step 4.
4. Once the worklist empties (or the "Max CFG trace blocks" cap is hit, producing a partial
   result), review the table of every decided/flagged block — verdict REAL/DEAD/UNRESOLVED, with
   a reason (rows resolved automatically are prefixed `[symbolic]`) — check/uncheck rows, and
   click **Accept & Mark Disassembly**. This only colors the affected instructions and adds a
   one-line comment on each block's first instruction; it never patches bytes, unless you've
   checked "Also patch bytes" (see below).

### Patching the recovered CFG

The review screen also has an **"Also patch bytes"** checkbox (unchecked by default, with a
confirmation prompt showing exact counts before anything is touched). When checked, **Accept**
additionally:

- **NOPs out every checked DEAD instruction's bytes** (`0x90` per byte — safe for any
  instruction length, no multi-byte NOP encoding needed).
- **Redirects every fully-resolved opaque predicate to its real target** — only for a
  `conditional_branch` block whose two successors were decided with *opposite* verdicts (one
  real, one dead) *and* both corresponding rows are checked. A short `Jcc` becomes a short `JMP`
  to the same target (2 bytes either way, opcode swap only); a near `Jcc` becomes a near `JMP`
  (one byte shorter, padded with a trailing NOP) with a freshly-computed displacement — never
  reused/guessed from the original bytes. Genuine data-dependent branches (both sides real) are
  never touched, since there's nothing to redirect — both paths are legitimately reachable.
- **Ensures a function is actually defined at the trace's start address**, since IDA sometimes
  never recognized it as one in the first place — the whole reason obfuscated dispatcher entry
  points need this feature at all. Only creates a function where none exists; if the address
  turns out to already be inside a *different* function, it's left alone rather than forcing a
  new boundary into existing analysis.

Both byte-patching operations re-decode and verify the actual bytes at each address immediately before
patching (never trust stale/cached data), and refuse to patch — logging why, leaving the
original bytes untouched — for anything they don't fully recognize: an unusual `Jcc` encoding,
a claimed target that doesn't match what the instruction actually encodes, or a dead-code
address that (per IDA's "overlapping blocks" edge case — see the code comments) might also be
claimed by a real, live instruction. This changes actual code bytes, not just IDB
colors/comments — it's revertible via IDA's own **Edit → Patches** menu, but review the counts
in the confirmation prompt before proceeding.

Blocks the model never gave a verdict for, addresses it invented outside the actual candidates,
and conflicting REAL/DEAD verdicts for the same address reached from different paths are all
surfaced as UNRESOLVED rather than silently resolved one way or the other, so you can just
review those manually rather than trust a guess. The same applies to the constant-propagation
pass: it only ever resolves a branch when it's actually confident (a concrete value it computed,
or a value it can positively attribute to runtime/caller data) — anything it isn't sure about
still falls back to the LLM rather than guessing.

**Loops and re-entered dispatchers** are handled specially, since a flattening dispatcher is
typically revisited every loop iteration with a *different* state value — a snapshot that
confidently resolves one case as real and the rest as dead on the first pass can be wrong for
every other pass. The trace detects both directions of this: a candidate that loops back to an
earlier block in the same trace is flagged for the LLM instead of being auto-resolved by the
constant-propagation pass (a value that looks fixed on this one pass through a loop isn't
necessarily fixed on every pass), and if a dispatcher block already fully decided gets reached
again from a *different* real block later in the trace — the loop-revisit pattern — any of its
successors previously marked dead are automatically un-marked and moved to UNRESOLVED for
review, rather than staying permanently (and possibly wrongly) dead.

**Constant-propagation pass limitations** (falls back to the LLM for these, same as if the
feature were off): x86/x64 only; indexed/scaled memory addressing (`[rcx+rdx*4]`,
`[rax*8+table]`) is supported for reads but never round-trip-tracked through writes; a `call`'s
effect on registers/memory isn't tracked precisely — its results are treated as genuine
runtime data (same as an incoming argument), which resolves an ordinary
`if (helper_call(...))`-shaped branch without an LLM call, but means the pass can't reason
through a call to derive a fixed value; only
`mov`/`movzx`/`movsx(d)`/`lea`/`add`/`sub`/`and`/`or`/`xor`/`not`/`neg`/`inc`/`dec`/`shl`/`shr`/
`sar`/`cmp`/`test`/`push`/`pop`/`nop` are modeled — anything else resets tracking for that path.
Disable "Resolve CFG trace branches via constant propagation" in Settings to force every
decision point through the LLM as before.

## Configuration

Open **Edit → Plugins → LLM Explainer** to configure:

| Setting | Default | Notes |
|---|---|---|
| Server base URL(s) | `http://127.0.0.1:8080` | One `llama-server` endpoint per line, in priority order, with an optional `# name` comment (e.g. `http://127.0.0.1:8080  # Home GPU`) shown in status/log messages instead of the raw URL; batch/recursive explain distributes work across all of them concurrently, and any single request automatically falls back to the next server on connection failure |
| Model name | *(blank)* | Only needed if your server hosts multiple models |
| API key | *(blank)* | Optional bearer token |
| Temperature | `0.2` | |
| Max tokens | `16384` | Reasoning models can spend thousands of tokens thinking before answering — keep this generous |
| Request timeout (s) | `300` | Per-chunk socket timeout, not a total-generation cap |
| Max context chars | `12000` | Per-function truncation budget |
| Include called-function names | on | |
| Max callees listed | `20` | |
| Include referenced strings/globals | on | |
| Max data refs listed | `20` | |
| Max string length shown | `150` | |
| Follow calls depth | `0` | `0` = target function only; `N>0` eagerly includes N levels of callee code |
| Max total context chars | `40000` | Overall budget when following calls |
| Max on-demand code requests | `5` | Cap on automatic `REQUEST_CODE` round-trips per conversation |
| Max recursive callees | `10` | Cap on direct callees processed by the recursive auto-accept action |
| Explain hotkey | `Ctrl-Alt-E` | |
| System prompt | *(editable)* | Governs the whole protocol below |
| Max CFG trace blocks | `200` | Safety cap on blocks decoded per CFG trace run; hitting it stops with a partial result |
| Resolve CFG trace branches via constant propagation | on | Tries a fast, deterministic symbolic pass (x86/x64 only) before asking the LLM at each decision point; disable to always ask the LLM |
| CFG trace colors | green / red / amber | Disassembly background colors for REAL / DEAD / UNRESOLVED blocks (`#RRGGBB`) |
| CFG trace system prompt | *(editable)* | Governs the REAL_TARGET/DEAD_TARGET/UNRESOLVED_TARGET protocol used by CFG tracing |

A **Restore Defaults** button resets the form (not saved until you click OK). Settings persist
as JSON under your IDA user directory (`llm_explainer.cfg.json`).

## The prompt protocol

The system prompt teaches the model a small text protocol so the plugin can parse structured
suggestions out of an otherwise free-form answer:

- `REQUEST_CODE: <function name or address>` — ask for a called function's code before
  answering (handled automatically, looping up to the configured limit).
- `SUGGESTED_NAME: <name>` — a proposed function name.
- `SUGGESTED_SIGNATURE: <C declaration>` — a proposed return type + argument types/names
  (Hex-Rays pseudocode only).
- `SUGGESTED_VAR: <old> -> <new>` — a proposed local variable rename (Hex-Rays pseudocode only,
  one per line).
- `SUGGESTED_CALLEE_NAME: <name-or-address> -> <new name>` — a proposed rename for a *called*
  function, only accepted if its code was actually shown to the model this conversation and its
  current name still looks auto-generated (`sub_`/`loc_`/`j_...`).
- `SUGGESTED_STRUCT: <C struct declaration>` — a proposed structure type (Hex-Rays pseudocode
  only), registered into the local types library on Accept.
- `SUGGESTED_VAR_TYPE: <var> <type expression>` — applies a type (e.g. a newly suggested struct
  pointer) to a local variable; for function arguments, the type is folded directly into
  `SUGGESTED_SIGNATURE` instead.

The final answer itself is kept to one short sentence, since it's written verbatim into the
function's comment.

## License

MIT — see [LICENSE](LICENSE).

## Copyright

© 2026 Peter Garba
