<p align="center"><img src="logo.svg" alt="LLM Explainer logo" width="128"></p>

# LLM Explainer

Explain and rename IDA functions with a **local** [llama.cpp](https://github.com/ggml-org/llama.cpp)
server — private, offline, nothing written to your database until you click **Accept**. Works in
the Hex-Rays or the disassembly view.

![LLM Explainer result dialog](example.png)

## Features

- **Right-click → explain** — pseudocode or disassembly view, or a hotkey (`Ctrl-Alt-E`). Answer streams in live; reasoning models show their chain-of-thought separately.
- **Human in the loop** — every suggestion is a separate, editable checkbox. Accept / Reason More / Cancel. The model never writes on its own.
- **Rename & retype** — proposes a function name, full C signature, local-variable renames, **called-function** renames, and **global/data-variable** renames (`byte_…`, `qword_…` → meaningful names).
- **Struct detection** — infers an undefined struct from pointer-offset access patterns and applies it.
- **Call-graph aware** — follows callees (configurable depth) and can fetch a specific callee's code on demand mid-answer.
- **Batch mode** — explain a checklist of functions, review (incl. each proposed new name), apply in bulk.
- **Recursive auto-accept** — explains a function + its *undiscovered* (`sub_…`) callees and applies automatically; can re-analyze an already-named callee the model flags as misnamed.
- **Multi-server** — list several `llama-server` endpoints for ~Nx parallel batch throughput, with priority order + automatic failover.
- **CFG recovery for obfuscated code** — walks basic blocks, resolves opaque predicates / dead code / flattening dispatchers with a fast deterministic pass (falls back to the LLM only when unsure), then optionally **patches** or **rebuilds** the real control flow. x86/x64 and AArch64.

## Install

Copy `llm_explainer.py` into `<IDA user dir>\plugins\` (Windows: `%APPDATA%\Hex-Rays\IDA Pro\plugins\`) and restart IDA. Requires IDA 9.3+ (PySide6 ships with IDA) and a reachable `llama-server` (default `http://127.0.0.1:8080`). Hex-Rays is optional — it falls back to disassembly. Or install the packaged `dist/*.zip` via [`hcli`](https://hcli.docs.hex-rays.com/).

## Quick start

- **One function** — right-click → *Explain function with LLM…*, review the streamed suggestions, **Accept & Add Comment**.
- **Batch** — Functions window → *Batch Explain Functions…*, check functions, **Apply Selected** when done. A **New Name** column shows the proposed rename per function as it finishes (marked `(kept: …)` when the existing non-default name would be preserved).
- **Recursive** — right-click → *Explain function with LLM (recursively)…*. Auto-applies; capped by *Max recursive callees*; writes unattended, so use with care.
- **CFG recovery** — disassembly view → *Trace/Recover CFG…*, pick a start address, watch the live transcript/graph, then review each block and pick an **On Accept** mode.

## CFG patching modes

Chosen on the review screen (all re-verify actual bytes before touching anything, and refuse rather than guess):

- **Mark only** *(default)* — colors + comments blocks. No bytes changed.
- **Patch in place** — NOPs confirmed-dead code and redirects fully-resolved opaque-predicate branches to their real target; ensures a function exists at the entry. Also collapses a single-target computed/indirect jump to a direct branch (incl. AArch64 `BR`/`B.cond`).
- **Rebuild linear** — writes just the real blocks as one straight-line sequence at the entry point, re-encoding every branch/call explicitly; touches only `[entry, entry+size)`.

Results are cached for the session (**Load Cached Result**), and any in-place/rebuild patch is revertible with **Undo Patches**. Opt-in *Enumerate ARM64 computed jump tables* (experimental) recovers `*(base + i*stride + field)` dispatch handlers.

## Configuration

**Edit → Plugins → LLM Explainer.** Persisted as `llm_explainer.cfg.json` in your IDA user dir. Key settings:

| Setting | Default | Notes |
|---|---|---|
| Server base URL(s) | `http://127.0.0.1:8080` | One endpoint per line, priority order, optional `# name`; batch runs across all, with failover |
| Model / API key | *(blank)* | Only if your server needs them |
| Temperature / Max tokens | `0.2` / `16384` | Keep tokens generous for reasoning models |
| Follow calls depth | `0` | `N>0` eagerly includes N levels of callee code |
| Max recursive callees | `10` | Cap for the recursive auto-accept action |
| System prompt(s) | *(editable)* | Explain + CFG-trace protocols |
| Resolve branches via constant propagation | on | Fast deterministic pass before the LLM (disable to always ask) |
| Enumerate ARM64 computed jump tables | off | Experimental; see above |
| CFG trace colors / Max blocks | green/red/amber, `200` | REAL / DEAD / UNRESOLVED |

Saved prompts auto-update to the current default when you haven't customized them, so plugin updates take effect without editing the config.

## Prompt protocol

The system prompt asks the model to emit structured lines the plugin parses out of its free-form answer:

| Marker | Purpose |
|---|---|
| `REQUEST_CODE: <fn>` | fetch a callee's code before answering (automatic) |
| `SUGGESTED_NAME: <name>` | function name |
| `SUGGESTED_SIGNATURE: <decl>` | prototype (Hex-Rays only) |
| `SUGGESTED_VAR: <old> -> <new>` | local rename (Hex-Rays only) |
| `SUGGESTED_CALLEE_NAME: <fn> -> <new>` | rename a callee whose code was shown |
| `SUGGESTED_GLOBAL_NAME: <g> -> <new>` | rename a referenced global/data variable |
| `SUGGESTED_REANALYZE: <fn> - <why>` | flag an already-named callee for re-analysis (recursive scan) |
| `SUGGESTED_STRUCT: <decl>` | define + register a struct type |
| `SUGGESTED_VAR_TYPE: <var> <type>` | apply a type to a local |

The prose answer itself is kept to one sentence — it becomes the function comment.

![Trace/Recover CFG live view](tracer.png)

## License

MIT — see [LICENSE](LICENSE). © 2026 Peter Garba
