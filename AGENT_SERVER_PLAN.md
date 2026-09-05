# Agent Server — plan (v1.11.0)

> **STATUS: IMPLEMENTED in v1.11.0.** 61 tools (more than the ~45 planned),
> fixed default port 8181, `runIdaScript` kept and gated off by default.
> Protocol-layer tests live in `test_autocompact.py` (section 10-12);
> the IDA-specific handlers need a real IDB to exercise end-to-end.

An HTTP server inside the plugin that lets an external agent (an LLM agent
loop, a script, another tool) drive IDA Pro: read code/data/types, search,
and *mutate* the database (renames, comments, types, functions) — the full
loop of iterative, agent-based reverse engineering.

## 1. Transport & protocol

- **Stdlib only**: `http.server.ThreadingHTTPServer` (no new dependencies;
  the plugin stays a single file). Server thread started in
  `plugin.init()`, shut down in `term()`.
- **Bind**: `127.0.0.1` by default (configurable host). Fixed default
  port **8181** (configurable; `0` = auto-assign). The exact URL and
  token are always logged at startup:
  `[LLM Explainer] Agent server: http://127.0.0.1:8181  token: a1b2c3d4...`
- **Auth**: `Authorization: Bearer <token>`. Token is auto-generated
  (secrets.token_hex) per IDA session unless the user sets one in settings.
- **Endpoints**
  | Endpoint | Purpose |
  |---|---|
  | `GET /health` | liveness + `pid`, `idb_name` (no auth) |
  | `GET /tools` | self-describing catalog: every tool's name, description, JSON-schema params — the agent discovers the API itself |
  | `POST /call` | `{"name": "...", "args": {...}}` → `{"ok": true, "result": ...}` or `{"ok": false, "error": "..."}` |
  | `POST /batch` | `{"calls": [{name, args}, ...]}` executed sequentially, all results returned (saves agent round-trips) |
- **Address convention**: every address/name arg accepts `"0x140001230"`,
  decimal, or a symbol name. One shared resolver (reuse existing
  `_resolve_name_or_address`).
- **Errors**: HTTP 400 bad args · 401 bad token · 404 unknown tool ·
  408 IDA main-thread timeout · 500 IDA error. Always JSON
  `{"ok": false, "error": ...}`.
- **Threading**: IDA APIs are not thread-safe. The server thread never
  touches IDA directly — every handler body runs through
  `ida_kernwin.execute_sync(fn, MFF_FAST)` with a timeout (the same pattern
  the LLM workers already use). Requests serialize on the IDA main thread.
- **Output caps**: per-call `limit` params are clamped (≤ 500); text
  results truncated to `agent_max_output_chars` (default 20 000) with a
  `"truncated": true` flag — agents have context budgets too.

## 2. Tool catalog (~45 tools)

### Session / meta
| Tool | Args | Returns |
|---|---|---|
| `getMetadata` | – | idb name/path, processor, bitness, image base, entry points, IDA version |
| `getAnalysisStatus` | – | autoanalysis done? |
| `waitAnalysis` | `timeout` (s, ≤ 300) | blocks until autoanalysis finishes |
| `getCurrentCursor` | – | what the human is looking at (EA + function) — human-in-the-loop context |
| `showInIda` | `address` | jumps the IDA UI to an address |

### Discovery
| Tool | Args | Returns |
|---|---|---|
| `getFunctions` | `filter?`, `min_size?`, `offset?`, `limit?` | `[{address, name, size, is_thunk, is_library}]` |
| `getFunctionAt` | `address` | the function containing the EA |
| `getFunctionInfo` | `address\|name` | start/end/size/name/prototype/flags |
| `getStrings` | `filter?`, `min_length?`, `offset?`, `limit?` | `[{address, value, type}]` |
| `getImports` | `module?` | `[{address, name, module, ordinal}]` |
| `getExports` | `filter?` | `[{address, name, ordinal}]` |
| `getSegments` | – | `[{name, start, end, permissions, class}]` |
| `getEntryPoints` | – | `[{address, name}]` |
| `searchBytes` | `pattern` (`"48 8B ?? FF"`), `start?`, `end?`, `limit?` | matching EAs |
| `searchText` | `text`, `start?`, `end?`, `limit?` | EAs where disassembly text matches |

### Code reading
| Tool | Args | Returns |
|---|---|---|
| `getDisassembly` | `address`, `count?` (instructions) | disasm text from an EA |
| `getFunctionDisassembly` | `address\|name` | full function disasm |
| `getDecompilation` | `address\|name` | Hex-Rays pseudocode (error if no decompiler) |
| `getFunctionPrototype` | `address` | C prototype string |
| `getFunctionArguments` | `address` | `[{name, type, location}]` |
| `getFunctionLocals` | `address` | `[{name, type, location}]` |
| `getFunctionCalls` | `address` | callees `[{address, name}]` |
| `getCallers` | `address\|name` | `[{address, name, call_sites: [ea]}]` |
| `getBasicBlocks` | `address` | `[{start, end, successors, predecessors}]` |
| `getCallGraph` | `address\|name`, `depth?` (≤ 5) | nested call tree |
| `getCrossReferencesTo` | `address\|name`, `limit?` | `[{from, is_code, type}]` |
| `getCrossReferencesFrom` | `address`, `limit?` | `[{to, is_code, type}]` |
| `getFunctionSummary` | `address\|name` | one-shot bundle: info + prototype + decompilation + callees + callers (agent context saver) |

### Data & types
| Tool | Args | Returns |
|---|---|---|
| `readBytes` | `address`, `size` (≤ 64K) | hex + ASCII sidebar |
| `readMemory` | `address`, `size` | raw processor memory (works outside IDB) |
| `getDataAt` | `address` | item kind, size, rendered value, type, name |
| `getGlobalVariables` | `filter?`, `offset?`, `limit?` | `[{address, name, size, type}]` |
| `getVariableAt` | `address` | the global/label at an EA |
| `listStructs` | `filter?` | `[{name, size, n_fields}]` |
| `getStruct` | `name\|id` | fields `[{offset, size, type, name}]` |
| `getStructAt` | `address` | struct applied at an EA (if any) |
| `listEnums` / `getEnum` | `filter?` / `name` | enum definitions |
| `getTypeAt` | `address` | type string at an EA |
| `listLocalTypes` | `filter?` | user/local type names |

### Mutation (the iterative-RE core)
| Tool | Args | Notes |
|---|---|---|
| `renameFunction` | `address`, `name` | |
| `renameGlobal` | `address`, `name` | |
| `renameData` | `address`, `name` | |
| `createLabel` | `address`, `name` | user label at any EA |
| `renameVariable` | `function`, `variable`, `name` | local or arg — reuses `_set_lvar_name` |
| `setVariableType` | `function`, `variable`, `type` | reuses `_set_lvar_type` |
| `setFunctionPrototype` | `address`, `prototype` | reuses `_apply_function_signature` |
| `setFunctionType` | `address`, `type` | |
| `setComment` | `address`, `comment`, `repeatable?` | |
| `setFunctionComment` | `address`, `comment` | |
| `makeFunction` | `address`, `end?` | |
| `deleteFunction` | `address` | |
| `defineCode` | `address`, `size?` | |
| `undefine` | `address`, `size` | |
| `createData` | `address`, `type` (`"int"`, `"char[16]"`, `"struct X"`, …) | |
| `createStruct` | `name`, `fields: [{name, type}]` | reuses `_create_struct_type` |
| `deleteStruct` | `name` | |
| `setBookmark` / `getBookmarks` / `deleteBookmark` | `address[, text]` | agent scratch notes |
| `patchBytes` | `address`, `hex` | **gated** by `agent_allow_patches` (default off) |
| `runIdaScript` | `code` | **gated** escape hatch, **default off** — full Python with IDA imports for cases the catalog doesn't cover |

## 3. Config additions (Settings → "Agent server" group)

| Key | Default | Meaning |
|---|---|---|
| `agent_server_enabled` | `False` | opt-in |
| `agent_server_host` | `"127.0.0.1"` | bind address |
| `agent_server_port` | `8181` | fixed default; 0 = auto-assign |
| `agent_server_token` | `""` | empty = random per session |
| `agent_allow_patches` | `False` | enables `patchBytes` |
| `agent_allow_scripts` | `False` | enables `runIdaScript` |
| `agent_max_output_chars` | `20000` | per-result truncation |

## 4. Implementation sketch (single file)

- New section before the action handlers:
  - module-level handler fns, each `def agent_getFunctions(args): ...`
    returning JSON-serializable dicts;
  - `AGENT_TOOLS = {name: {"description": ..., "params": {...json schema...},
    "fn": agent_..., "gated": "patches"|"scripts"|None}}` — the registry
    doubles as the `/tools` schema source (single source of truth);
  - `class AgentServer(threading.Thread)` — request loop, auth check,
    dispatch, `execute_sync` wrapper (`_run_on_ida(fn, timeout=60)`),
    error mapping, truncation.
- Reuse existing battle-tested helpers: `_resolve_name_or_address`,
  `_rename_user_labels`, `_set_lvar_name`, `_set_lvar_type`,
  `_apply_function_signature`, `_create_struct_type`, `_image_base`.
- New guarded import: `ida_hexrays` (decompilation; degrade gracefully
  when the decompiler is absent — the plugin currently never imports it).
- `plugin.init()`: start server if enabled, log URL+token.
  `plugin.term()`: shutdown.
- Settings dialog: group box with the 7 fields + "Regenerate token" button;
  live "current URL" label when enabled.

## 5. Testing

`test_autocompact.py`-style harness (IDA/Qt stubs already exist) covers the
protocol layer, which is the part that can be unit-tested without a real IDB:
- server start/stop, `/health` without auth, `/tools` schema completeness
  (every tool has description + param schema; every `fn` callable);
- auth: 401 without/bad token, 200 with token;
- dispatch: unknown tool → 404, bad args → 400, gated tool refused when
  disabled (403), `/batch` sequencing;
- address resolver: hex/dec/name; output truncation + `truncated` flag.
IDA-specific handlers are exercised manually in real IDA (they need a real
IDB).

## 6. Risks / decisions

- **Main-thread blocking**: decompiling a huge function can take seconds;
  `execute_sync` timeout (60 s) returns 408 rather than hanging the agent.
- **Security**: localhost + bearer token; `patchBytes`/`runIdaScript`
  off by default; everything else is normal IDA editing (the user's own
  database, same trust level as doing it by hand — undo works).
- **Concurrent requests**: serialize naturally on the IDA main thread;
  `ThreadingHTTPServer` keeps connections responsive otherwise.
- **No decompiler**: `getDecompilation`/`getFunctionSummary` return a clear
  error; all other tools work.

## 7. Multi-instance discovery (v1.13.0)

> **STATUS: IMPLEMENTED in v1.13.0.**

Opening several IDA sessions at once, each analyzing a different binary,
used to fail: `agent_server_port` is a fixed default (8181) stored in one
config file shared by every IDA process on the machine, so the second
session's `ThreadingHTTPServer(...)` bind raised `OSError` and its agent
server never started.

- **Auto-fallback**: if the configured port is taken, `AgentServer.run()`
  retries once with port `0` (OS-assigned) instead of giving up. Multiple
  sessions now start automatically; only the *first* one gets the
  configured/default port.
- **Self-advertisement**: on successful bind, each instance writes
  `agent_instances/instance_<pid>.json` under the IDA user directory
  (`_write_agent_registry`) with its `host`, `port`, `token`, `idb_name`,
  `idb_path`, `input_file`, `processor`, `bits`, `ida_version`,
  `plugin_version`, `started_at`. The file is removed when the server
  stops (plugin unload, settings-triggered restart, or IDA exit).
- **Discovery**: `list_ida_agents.py` (repo root, no IDA dependency) reads
  that registry, probes each entry's `GET /health` to confirm it's still
  alive (pruning stale files left behind by crashed sessions), and prints
  the live sessions as JSON. An agent runs this first when it isn't told a
  fixed port up front, matches the session it wants by `idb_name`/
  `input_file`, and uses that entry's `host`/`port`/`token` for everything
  else.
- **`/health` enhancement**: now includes `pid` and `idb_name` (still
  unauthenticated — no more revealing than a local process listing), so a
  caller with only network reach can confirm which binary is behind a port.

## 8. Opening and closing files (v1.13.0)

> **STATUS: IMPLEMENTED in v1.13.0.**

An agent needed a way to bring up another binary and, when finished, end its
own session. IDA's architecture constrains what's possible here: a running
GUI process holds exactly one open database, and there is no supported way
to swap the input file of a live process in place — per the IDA 9.0 release
notes, opening a file while one is already open always spawns a **new IDA
instance** (a new process). Three tools embrace that instead of fighting it:

- **`openFile(path)`**: spawns a new IDA process on `path` via
  `subprocess.Popen([exe, path])`, where `exe` is resolved by asking the OS
  for *this process's own* executable path (`GetModuleFileNameW(None, ...)`
  on Windows, `/proc/self/exe` on Linux, `sys.executable` as a last resort)
  rather than guessing an install path/binary name — avoids the kind of
  API-drift bug this plugin has been bitten by before. Current session is
  untouched. Ungated (no more sensitive than launching any other program).
- **`openFileInThisWindow(path)`**: same spawn, then `idc.save_database("",
  0)` and a delayed `idc.qexit(0)` (run from a background thread ~0.5s
  later, so the HTTP response for the call is sent before the process
  exits). Net effect: the old window disappears, a new one appears for the
  new file — the closest equivalent to "reuse this window" IDA allows.
- **`closeDatabase(save=true)`**: same save + delayed-`qexit` pattern,
  without opening anything new. This is what an agent calls when it's done
  with a one-off analysis session.

Both of the latter two are gated by a new `agent_allow_close` setting (off
by default, alongside `agent_allow_patches`/`scripts`/`debug`) since they end
the session out from under whoever is looking at that window; `openFile`
needs no gate. `agents.md` instructs the agent to call `closeDatabase`
when it finishes a session it opened itself, but to treat a `403` as "leave
it open" rather than an error, and to expect the connection to drop without
a response once either tool actually runs (the process is exiting by then —
intended, not a bug).
