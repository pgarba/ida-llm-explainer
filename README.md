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
- **Multi-server parallelism**: configure more than one `llama-server` instance and batch/recursive
  explain will run up to one function per server concurrently — N servers gives roughly an Nx
  speedup over a single server. The interactive single-function explain always uses the first
  configured server.

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

## Configuration

Open **Edit → Plugins → LLM Explainer** to configure:

| Setting | Default | Notes |
|---|---|---|
| Server base URL(s) | `http://127.0.0.1:8080` | One `llama-server` endpoint per line; batch/recursive explain distributes work across all of them, one function per server at a time |
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
