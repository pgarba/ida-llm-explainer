"""LLM Explainer - IDA Pro 9.3 plugin.

Asks a locally-running llama.cpp server (llama-server, OpenAI-compatible API)
to explain the function currently under the cursor, in either the Hex-Rays
pseudocode view or the plain disassembly view. The model's streamed answer
is shown in a small, non-modal dialog where you can Accept it (written into
the function's comment, visible in both views), ask to Reason More (send a
follow-up question and get a refined answer), or Cancel (discard, no
database changes).

Two settings help the model reason about code that calls into other
functions: "Follow calls depth" eagerly includes the code of called
functions (up to N levels) in the initial prompt, and independently of
that, the model can ask for a specific called function's code on demand
mid-conversation (it replies with a `REQUEST_CODE: <name-or-address>`
line, the plugin fetches that function and feeds the code back
automatically, up to "Max on-demand code requests" times).

Install by copying this single file into one of:
  - Per-user (recommended, no admin rights needed):
        <IDA user dir>\\plugins\\llm_explainer.py
    On Windows this is typically:
        %APPDATA%\\Hex-Rays\\IDA Pro\\plugins\\llm_explainer.py
  - Global (all users of this IDA install):
        <IDA install dir>\\plugins\\llm_explainer.py

Requires: IDA Pro 9.3+ (PySide6 is bundled with IDA, no extra install
needed), and a running llama.cpp `llama-server` reachable at the configured
base URL (default http://127.0.0.1:8080). The Hex-Rays decompiler is
optional - if it is not available for the current architecture the plugin
falls back to a plain disassembly listing automatically.

Configure the server URL, model, and other options via
Edit > Plugins > LLM Explainer.

Note: if llama-server is run with a single inference slot, opening several
"explain" dialogs at once will simply queue their requests on the server -
that is a server/deployment concern, not a bug in this plugin.

To process many functions at once, right-click in the Functions window (or
use Edit > Plugins > Batch Explain Functions...) to pick a set of functions,
process them sequentially, and review/apply the results in one batch -
still nothing is written to the database until you explicitly apply.

"Explain function with LLM (recursively)" (same right-click menu as the
regular explain action) explains the target function plus its direct
callees only (depth 1, not deeper), and - unlike every other action in
this plugin - auto-accepts every result with no review step. It still
applies the same conservative defaults as a manual Accept (e.g. only
renaming a function whose name looks auto-generated), the callee count is
capped by its own "Max recursive callees" setting, and a progress dialog
stays open so you can watch what happens and cancel mid-run.
"""

import functools
import json
import os
import re
import threading
import urllib.error
import urllib.request
from collections import namedtuple

import idaapi
import idautils
import idc
import ida_kernwin
import ida_funcs
import ida_lines
import ida_name
import ida_typeinf

try:
    import ida_hexrays
except ImportError:
    ida_hexrays = None

try:
    import ida_ida
except ImportError:
    ida_ida = None

from PySide6 import QtCore, QtGui, QtWidgets


PLUGIN_NAME = "LLM Explainer"
PLUGIN_VERSION = "1.0.0"
PLUGIN_COPYRIGHT = "© 2026 Peter Garba"
ACTION_ID_EXPLAIN = "llm_explainer:explain_function"
ACTION_ID_BATCH = "llm_explainer:batch_explain"
ACTION_ID_EXPLAIN_RECURSIVE = "llm_explainer:explain_recursive"
CONFIG_FILENAME = "llm_explainer.cfg.json"


def _add_copyright_footer(layout):
    label = QtWidgets.QLabel(PLUGIN_COPYRIGHT)
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
    font = label.font()
    font.setPointSize(max(7, font.pointSize() - 2))
    label.setFont(font)
    palette = label.palette()
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("gray"))
    label.setPalette(palette)
    layout.addWidget(label)

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert reverse engineer assisting inside IDA Pro. You will "
    "be given the decompiled pseudocode or disassembly of a target function, "
    "along with its name, address, target architecture, the names of "
    "functions it calls, notable string literals and named global data it "
    "references (when available), and - depending on settings - the code "
    "of some called functions already included below the target "
    "function.\n\n"
    "Before doing anything else, go through the \"Calls:\" list one "
    "function at a time. For each one whose name still looks "
    "auto-generated (sub_, loc_, j_ followed by a hex address) and whose "
    "code was not already included above, ask yourself honestly: am I "
    "about to guess its behavior from superficial clues alone - its "
    "argument count/types, a \"common pattern\" it resembles (e.g. "
    "assuming a two-argument call like foo(ptr, 16) is \"probably a free\" "
    "or \"probably a refcount decrement\"), or how similar calls usually "
    "look in other code - rather than from having actually seen what it "
    "does? A plausible-sounding guess is still a guess. If so, you MUST "
    "request that function's code before answering; do not settle for a "
    "guess just because you already have enough for A plausible-sounding "
    "answer. This applies to every such call in the function, not just "
    "the first or most obviously important one - it is common and "
    "expected to request several functions' code in the same reply when a "
    "function calls several unclear helpers. To request code, reply with "
    "one or more lines of the exact form\n"
    "REQUEST_CODE: <function name or address>\n"
    "and nothing else in that reply. You will then be given each "
    "function's code in a follow-up message and can continue reasoning. "
    "Only skip requesting a call's code when you are already certain what "
    "it does (e.g. you saw its code earlier this conversation, it is a "
    "well-known library/CRT function, or its exact behavior genuinely "
    "would not change your answer, such as a bare logging call) - not "
    "merely because you have a plausible guess. Never request the same "
    "function twice.\n\n"
    "Once you have enough information, work through ALL FOUR of the "
    "following steps before writing your final answer. These are a "
    "required part of a thorough analysis, not optional extras to drop "
    "once you feel you already have enough to say - a lazy answer that "
    "skips available renames or a clear struct is an incomplete answer:\n"
    "1. Propose a better name for the target function, unless it is "
    "already named descriptively, as one line of the exact form\n"
    "SUGGESTED_NAME: <name>\n"
    "The name must be a valid C identifier: letters, digits and "
    "underscores only, not starting with a digit, no spaces. Prefer "
    "short, conventional reverse-engineering style names (e.g. "
    "parse_http_header, aes_decrypt_block).\n"
    "2. If, and only if, the target function's own code was given to you "
    "as Hex-Rays pseudocode (not plain disassembly): propose a more "
    "accurate prototype, if you can determine one, as one line of the "
    "exact form\n"
    "SUGGESTED_SIGNATURE: <full C declaration>\n"
    "e.g. SUGGESTED_SIGNATURE: int __cdecl parse_header(char *buf, int len)\n"
    "and propose a rename for EVERY local variable (not arguments) whose "
    "default compiler-generated name (e.g. v1, v2, a1) could be more "
    "descriptive - go through each local variable in the code one by one "
    "and check it, not just the first or most obvious one - as one line "
    "per variable of the exact form\n"
    "SUGGESTED_VAR: <current_name> -> <new_name>\n"
    "3. If you actually examined the code of a CALLED function this "
    "conversation (because it was included up front, or you requested it "
    "with REQUEST_CODE) and its current name still looks auto-generated "
    "(e.g. sub_1402346D0, sub_401000, loc_403010), propose a rename for "
    "it - only that function, and only if you are confident about what it "
    "does from the code you actually saw, never from its name alone or a "
    "guess - as one line per function of the exact form\n"
    "SUGGESTED_CALLEE_NAME: <its current name or address> -> <new name>\n"
    "using the same identifier rules as SUGGESTED_NAME. Never for the "
    "target function itself, and never for a function whose code you "
    "never saw: this is enforced automatically, and any "
    "SUGGESTED_CALLEE_NAME for a function whose code was not actually "
    "shown to you earlier in this same conversation will be silently "
    "discarded, so do not bother emitting it in that case - request the "
    "function's code with REQUEST_CODE first if you want to propose a "
    "rename for it.\n"
    "4. If the pseudocode accesses memory through a pointer at multiple "
    "constant offsets in a way that suggests an undefined or "
    "generic-looking structure - e.g. *(a1 + 8), *(_DWORD *)(a1 + 0x10), a "
    "fixed-size header followed by a variable-length trailing buffer, or "
    "Hex-Rays already showing an anonymous/placeholder struct - define a "
    "proper structure type and apply it. Do not skip this just because "
    "the rest of the offsets are irregular or the layout takes a moment "
    "to work out; a partial struct covering the part you are confident "
    "about is still valuable. Add one line of the exact form\n"
    "SUGGESTED_STRUCT: <full struct declaration, all on this one line>\n"
    "e.g. SUGGESTED_STRUCT: struct tagPOINT { int x; int y; };\n"
    "(field offsets follow from declaration order and each member's size "
    "- add explicit padding members, e.g. char gap_4[4];, for gaps you "
    "are confident about), then apply it to whichever variable actually "
    "holds the pointer: if that variable is one of the target function's "
    "own arguments, reference the new struct type directly in "
    "SUGGESTED_SIGNATURE instead (e.g. \"tagPOINT *p\"); otherwise add one "
    "line per local variable of the exact form\n"
    "SUGGESTED_VAR_TYPE: <current_variable_name> <type expression>\n"
    "e.g. SUGGESTED_VAR_TYPE: v3 tagPOINT *\n\n"
    "For steps 2-4, only propose something you are reasonably confident "
    "about from the code itself, and skip these three steps entirely when "
    "you were given plain disassembly instead of pseudocode; step 1 "
    "(SUGGESTED_NAME) still applies to disassembly. Otherwise, do not "
    "leave a step out merely to save effort or keep the response short.\n\n"
    "Finally, give your final answer as exactly ONE short sentence (no "
    "more than ~20 words) stating precisely what the target function does "
    "- its core purpose only, not a step-by-step walkthrough. Do not "
    "restate the code line by line, and do not use markdown code fences "
    "or bullet points. This sentence will be written verbatim into an IDA "
    "function comment, so keep it self-contained and free of REQUEST_CODE "
    "lines. Keeping this sentence short does NOT mean doing less of steps "
    "1-4 above - list every SUGGESTED_* line that applies below your "
    "one-sentence answer; only the prose explanation itself needs to be "
    "brief. If asked for more detail in a follow-up, you may then answer "
    "at greater length."
)

DEFAULT_CONFIG = {
    "base_url": "http://127.0.0.1:8080",
    "model": "",
    "api_key": "",
    "temperature": 0.2,
    "max_tokens": 16384,
    "request_timeout": 300,
    "max_context_chars": 12000,
    "include_callees": True,
    "max_callees": 20,
    "follow_calls_depth": 0,
    "max_total_context_chars": 40000,
    "max_auto_fetch": 5,
    "include_data_refs": True,
    "max_data_refs": 20,
    "max_string_len": 150,
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "explain_hotkey": "Ctrl-Alt-E",
    "max_recursive_callees": 10,
}

ContextBundle = namedtuple("ContextBundle", ["kind", "text"])

# Safety valve for eager recursive call-following: never eagerly decompile
# more than this many functions for one initial request, regardless of the
# configured depth/char budget (deep or wide call graphs could otherwise
# make the initial request very slow).
_MAX_EAGER_FUNCTIONS = 40

_REQUEST_CODE_RE = re.compile(r"(?im)^\s*REQUEST_CODE:\s*(.+?)\s*$")
_SUGGESTED_NAME_RE = re.compile(r"(?im)^\s*SUGGESTED_NAME:\s*(.+?)\s*$")
_SUGGESTED_SIGNATURE_RE = re.compile(r"(?im)^\s*SUGGESTED_SIGNATURE:\s*(.+?)\s*$")
_SUGGESTED_VAR_RE = re.compile(r"(?im)^\s*SUGGESTED_VAR:\s*([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)\s*$")
_SUGGESTED_CALLEE_NAME_RE = re.compile(
    r"(?im)^\s*SUGGESTED_CALLEE_NAME:\s*(.+?)\s*->\s*([A-Za-z_]\w*)\s*$"
)
_SUGGESTED_STRUCT_RE = re.compile(r"(?im)^\s*SUGGESTED_STRUCT:\s*(.+?)\s*$")
_SUGGESTED_VAR_TYPE_RE = re.compile(r"(?im)^\s*SUGGESTED_VAR_TYPE:\s*([A-Za-z_]\w*)\s+(.+?)\s*$")
_VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_AUTO_NAME_RE = re.compile(r"^(sub|loc|nullsub|j_sub|j_nullsub)_[0-9A-Fa-f]+$")


def sanitize_suggested_name(name):
    if not name:
        return None
    name = name.strip().strip("`'\"").rstrip(".")
    if _VALID_NAME_RE.match(name):
        return name
    return None


def is_auto_generated_name(name):
    return bool(name) and bool(_AUTO_NAME_RE.match(name))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_markdown_fences(text):
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z0-9_+\-]*\s*\n", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def get_procname():
    if ida_ida is not None:
        try:
            return ida_ida.inf_get_procname()
        except Exception:
            pass
    try:
        return idc.get_inf_attr(idc.INF_PROCNAME)
    except Exception:
        return "unknown"


def gather_function_context(func):
    """Prefer Hex-Rays pseudocode; fall back to a plain disassembly listing."""
    cfunc = None
    if ida_hexrays is not None:
        try:
            if ida_hexrays.init_hexrays_plugin():
                cfunc = ida_hexrays.decompile(func)
        except Exception:
            cfunc = None
    if cfunc is not None:
        try:
            lines = [ida_lines.tag_remove(sl.line) for sl in cfunc.get_pseudocode()]
            return ContextBundle(kind="pseudocode", text="\n".join(lines))
        except Exception:
            pass

    lines = []
    for ea in idautils.FuncItems(func.start_ea):
        try:
            line = idc.generate_disasm_line(ea, idc.GENDSM_REMOVE_TAGS)
        except Exception:
            line = idc.GetDisasm(ea)
        if line:
            line = ida_lines.tag_remove(line)
        lines.append("%#010x  %s" % (ea, line or ""))
    return ContextBundle(kind="disassembly", text="\n".join(lines))


def gather_callee_funcs(func, max_callees):
    """Direct callees of func, as func_t objects, in first-seen order."""
    if max_callees <= 0:
        return []
    result = []
    seen = set()
    for ea in idautils.FuncItems(func.start_ea):
        for ref in idautils.CodeRefsFrom(ea, 0):
            callee = ida_funcs.get_func(ref)
            if callee and callee.start_ea == ref and ref != func.start_ea and ref not in seen:
                seen.add(ref)
                result.append(callee)
        if len(result) >= max_callees:
            break
    return result[:max_callees]


DataRef = namedtuple("DataRef", ["ea", "kind", "text"])  # kind: "string" | "global"


def gather_data_refs(func, max_refs, max_string_len):
    """String literals + named globals referenced by func's instructions
    (root function only, mirrors how "Calls:" is root-only). Skips refs
    that are themselves function entry points (already covered by
    gather_callee_funcs/"Calls:"). Dedupes by address, first-seen order,
    capped at max_refs. Defensive: one bad ref must never abort the rest.
    """
    if max_refs <= 0:
        return []
    result = []
    seen = set()
    for ea in idautils.FuncItems(func.start_ea):
        try:
            refs = list(idautils.DataRefsFrom(ea))
        except Exception:
            continue
        for ref in refs:
            if ref in seen:
                continue
            seen.add(ref)
            try:
                callee = ida_funcs.get_func(ref)
                if callee is not None and callee.start_ea == ref:
                    continue
            except Exception:
                pass
            try:
                strtype = idc.get_str_type(ref)
            except Exception:
                strtype = None
            if strtype is not None:
                try:
                    raw = idc.get_strlit_contents(ref, -1, strtype)
                    text = (raw or b"").decode("utf-8", "replace")
                except Exception:
                    continue
                if not text:
                    continue
                text = text.replace("\r", "\\r").replace("\n", "\\n")
                if len(text) > max_string_len:
                    text = text[:max_string_len] + "...[truncated]"
                result.append(DataRef(ea=ref, kind="string", text=text))
            else:
                try:
                    name = idc.get_name(ref)
                except Exception:
                    name = None
                if name:
                    result.append(DataRef(ea=ref, kind="global", text=name))
            if len(result) >= max_refs:
                return result
    return result


def format_data_refs_section(data_refs):
    strings = [r.text for r in data_refs if r.kind == "string"]
    names = [r.text for r in data_refs if r.kind == "global"]
    lines = []
    if strings:
        lines.append("Strings referenced: " + "; ".join('"%s"' % s for s in strings))
    if names:
        lines.append("Globals referenced: " + ", ".join(names))
    return "\n".join(lines)


def format_function_block(label, func_ea, ctx, config):
    name = ida_funcs.get_func_name(func_ea) or ("sub_%X" % func_ea)
    body = ctx.text
    if len(body) > config.max_context_chars:
        body = body[: config.max_context_chars] + "\n...[truncated]..."
    kind_label = "Pseudocode (Hex-Rays)" if ctx.kind == "pseudocode" else "Disassembly"
    return "--- %s: %s @ %#010x (%s) ---\n%s" % (label, name, func_ea, kind_label, body)


def gather_recursive_context(root_func, config):
    """Breadth-first walk of the call graph starting at root_func, up to
    config.follow_calls_depth levels. Returns a list of
    (depth, func_ea, ContextBundle) tuples, root first (depth 0). Bounded by
    config.max_total_context_chars and _MAX_EAGER_FUNCTIONS to keep the
    initial request fast even for large/recursive call graphs.
    """
    visited = {root_func.start_ea}
    root_ctx = gather_function_context(root_func)
    blocks = [(0, root_func.start_ea, root_ctx)]
    total_chars = len(root_ctx.text)
    frontier = [root_func]
    depth = 0
    while (
        depth < config.follow_calls_depth
        and frontier
        and total_chars < config.max_total_context_chars
        and len(visited) < _MAX_EAGER_FUNCTIONS
    ):
        next_frontier = []
        for func in frontier:
            for callee in gather_callee_funcs(func, config.max_callees):
                if callee.start_ea in visited:
                    continue
                if total_chars >= config.max_total_context_chars or len(visited) >= _MAX_EAGER_FUNCTIONS:
                    break
                visited.add(callee.start_ea)
                try:
                    ctx = gather_function_context(callee)
                except Exception:
                    continue
                blocks.append((depth + 1, callee.start_ea, ctx))
                total_chars += len(ctx.text)
                next_frontier.append(callee)
        frontier = next_frontier
        depth += 1
    return blocks


def resolve_function_query(query):
    """Resolve a model-supplied identifier (name or address) to a func_t."""
    query = (query or "").strip().strip("`'\"")
    if not query:
        return None
    ea = idc.get_name_ea_simple(query)
    if ea == idaapi.BADADDR:
        for base in (0, 16):
            try:
                ea = int(query, base)
                break
            except ValueError:
                ea = idaapi.BADADDR
    if ea == idaapi.BADADDR:
        return None
    return ida_funcs.get_func(ea)


def build_user_message(config, func, blocks, callee_names, data_refs=None):
    name = ida_funcs.get_func_name(func.start_ea) or ("sub_%X" % func.start_ea)
    header = (
        "Function: %s\n"
        "Address: %#010x\n"
        "Architecture: %s\n" % (name, func.start_ea, get_procname())
    )
    if callee_names:
        header += "Calls: %s\n" % ", ".join(callee_names)
    if data_refs:
        section = format_data_refs_section(data_refs)
        if section:
            header += section + "\n"
    if config.follow_calls_depth > 0:
        header += (
            "Code for up to %d level(s) of called functions is included "
            "below where available.\n" % config.follow_calls_depth
        )
    parts = [header]
    for depth, ea, ctx in blocks:
        label = "Target function" if depth == 0 else ("Called function (depth %d)" % depth)
        parts.append(format_function_block(label, ea, ctx, config))
    return "\n".join(parts)


def _resolve_func(ctx):
    ea = getattr(ctx, "cur_ea", None)
    if ea is None or ea == idaapi.BADADDR:
        ea = ida_kernwin.get_screen_ea()
    return ida_funcs.get_func(ea)


def _rename_lvars(func_ea, var_renames):
    """Rename local variables BEFORE any signature/type change is applied
    to the same function: retyping a function (e.g. via idc.SetType) can
    change how Hex-Rays decomposes its local variables, which would make
    the variable names the model actually saw (and is renaming) stale by
    the time we get to them - this was the main suspected cause of renames
    being silently skipped. Also validates the old name actually exists in
    the current decompilation (with a case-insensitive fallback) instead of
    just trying the exact name blind, and logs a clearer reason on failure.
    """
    if not var_renames or ida_hexrays is None:
        return
    try:
        hexrays_ready = ida_hexrays.init_hexrays_plugin()
    except Exception:
        hexrays_ready = False
    if not hexrays_ready:
        return

    lvar_names = None
    try:
        cfunc = ida_hexrays.decompile(func_ea)
        if cfunc is not None:
            lvar_names = {lv.name for lv in cfunc.get_lvars() if lv.name}
    except Exception:
        lvar_names = None

    for old_name, new_name_var in var_renames:
        actual_old_name = old_name
        if lvar_names is not None and old_name not in lvar_names:
            ci_match = next((n for n in lvar_names if n.lower() == old_name.lower()), None)
            if ci_match:
                actual_old_name = ci_match
            else:
                ida_kernwin.msg(
                    "[%s] Skipping variable rename '%s' -> '%s': no such "
                    "variable in the current decompilation.\n"
                    % (PLUGIN_NAME, old_name, new_name_var)
                )
                continue
        try:
            ok = ida_hexrays.rename_lvar(func_ea, actual_old_name, new_name_var)
        except Exception as exc:
            ok = False
            ida_kernwin.msg("[%s] Failed to rename variable '%s': %s\n" % (PLUGIN_NAME, actual_old_name, exc))
        if not ok:
            ida_kernwin.msg(
                "[%s] Failed to rename variable '%s' -> '%s'.\n" % (PLUGIN_NAME, actual_old_name, new_name_var)
            )


def _create_struct_type(decl_text):
    """Register a struct (or other UDT) declaration into the local types
    library via IDA's C declaration parser, e.g.
    "struct tagPOINT { int x; int y; };". Returns True on success. Must run
    before anything that might reference the new type (signature, variable
    types), since those are resolved against the local types library too.
    """
    try:
        decl = decl_text.strip()
        if not decl.endswith(";"):
            decl += ";"
        errors = idc.parse_decls(decl, 0)
        if errors:
            ida_kernwin.msg(
                "[%s] Failed to parse struct declaration (%d error(s)): %s\n"
                % (PLUGIN_NAME, errors, decl_text)
            )
            return False
        return True
    except Exception as exc:
        ida_kernwin.msg("[%s] Failed to parse struct declaration: %s\n" % (PLUGIN_NAME, exc))
        return False


def _set_lvar_type(func_ea, var_name, type_str):
    """Apply a C type expression (e.g. "tagPOINT *") to a Hex-Rays local
    variable. Parses the type via a throwaway variable declaration so
    arbitrary expressions (pointers, struct names, etc.) are accepted, then
    applies it through the lvar_saved_info_t / MLI_TYPE mechanism (the
    same mechanism the Hex-Rays UI itself uses for "Set lvar type").
    """
    decl = type_str.strip()
    if not decl:
        return False
    if not decl.endswith(";"):
        decl = decl + " __llm_dummy__;"
    tif = ida_typeinf.tinfo_t()
    parsed_name = ida_typeinf.parse_decl(tif, None, decl, ida_typeinf.PT_SIL | ida_typeinf.PT_VAR)
    if parsed_name is None or tif.empty():
        return False
    locator = ida_hexrays.lvar_locator_t()
    if not ida_hexrays.locate_lvar(locator, func_ea, var_name):
        return False
    info = ida_hexrays.lvar_saved_info_t()
    info.ll = locator
    info.type = tif
    return ida_hexrays.modify_user_lvar_info(func_ea, ida_hexrays.MLI_TYPE, info)


def _apply_var_types(func_ea, var_types):
    if not var_types or ida_hexrays is None:
        return
    try:
        hexrays_ready = ida_hexrays.init_hexrays_plugin()
    except Exception:
        hexrays_ready = False
    if not hexrays_ready:
        return
    for var_name, type_str in var_types:
        try:
            ok = _set_lvar_type(func_ea, var_name, type_str)
        except Exception as exc:
            ok = False
            ida_kernwin.msg("[%s] Failed to set type of variable '%s': %s\n" % (PLUGIN_NAME, var_name, exc))
        if not ok:
            ida_kernwin.msg(
                "[%s] Failed to apply type '%s' to variable '%s'.\n" % (PLUGIN_NAME, type_str, var_name)
            )


def _apply_suggestions_and_refresh(
    func_ea, comment, new_name=None, signature=None, var_renames=None, callee_renames=None,
    struct_decl=None, var_types=None,
):
    # Struct creation must happen first: the signature and/or variable
    # types below may reference the new type by name, and need it to
    # already exist in the local types library to resolve correctly.
    if struct_decl:
        _create_struct_type(struct_decl)

    # Variable types/renames operate on the CURRENT decompilation, before
    # the signature changes it (retyping a function can change how
    # Hex-Rays decomposes its locals and make old names/lookups stale).
    _apply_var_types(func_ea, var_types)
    _rename_lvars(func_ea, var_renames)

    try:
        idc.set_func_cmt(func_ea, comment, 0)
    except Exception as exc:
        ida_kernwin.msg("[%s] Failed to set comment: %s\n" % (PLUGIN_NAME, exc))

    if signature:
        try:
            decl = signature.strip()
            if not decl.endswith(";"):
                decl += ";"
            if not idc.SetType(func_ea, decl):
                ida_kernwin.msg("[%s] Failed to apply signature: %s\n" % (PLUGIN_NAME, signature))
        except Exception as exc:
            ida_kernwin.msg("[%s] Failed to apply signature '%s': %s\n" % (PLUGIN_NAME, signature, exc))

    if new_name:
        try:
            ok = ida_name.set_name(func_ea, new_name, ida_name.SN_NOWARN | ida_name.SN_FORCE)
            if not ok:
                ida_kernwin.msg("[%s] Failed to rename function to '%s'.\n" % (PLUGIN_NAME, new_name))
        except Exception as exc:
            ida_kernwin.msg("[%s] Failed to rename function: %s\n" % (PLUGIN_NAME, exc))

    if callee_renames:
        for callee_ea, callee_new_name in callee_renames:
            try:
                ok = ida_name.set_name(callee_ea, callee_new_name, ida_name.SN_NOWARN | ida_name.SN_FORCE)
                if not ok:
                    ida_kernwin.msg(
                        "[%s] Failed to rename called function to '%s'.\n" % (PLUGIN_NAME, callee_new_name)
                    )
                elif ida_hexrays is not None and ida_hexrays.init_hexrays_plugin():
                    try:
                        ida_hexrays.mark_cfunc_dirty(callee_ea)
                    except Exception:
                        pass
            except Exception as exc:
                ida_kernwin.msg("[%s] Failed to rename called function: %s\n" % (PLUGIN_NAME, exc))

    try:
        if ida_hexrays is not None and ida_hexrays.init_hexrays_plugin():
            ida_hexrays.mark_cfunc_dirty(func_ea)
    except Exception:
        pass
    try:
        ida_kernwin.request_refresh(ida_kernwin.IWID_DISASM | ida_kernwin.IWID_PSEUDOCODE)
    except Exception:
        try:
            ida_kernwin.refresh_idaview_anyway()
        except Exception:
            pass
    return 1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class PluginConfig(object):
    _FIELDS = list(DEFAULT_CONFIG.keys())

    def __init__(self, **kwargs):
        data = dict(DEFAULT_CONFIG)
        data.update({k: v for k, v in kwargs.items() if k in data})
        for key, value in data.items():
            setattr(self, key, value)
        self._validate()

    @staticmethod
    def _config_path():
        return os.path.join(idaapi.get_user_idadir(), CONFIG_FILENAME)

    @classmethod
    def load(cls):
        data = {}
        try:
            with open(cls._config_path(), "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                data = loaded
        except FileNotFoundError:
            pass
        except Exception as exc:
            ida_kernwin.msg("[%s] Failed to read config (%s); using defaults.\n" % (PLUGIN_NAME, exc))
        return cls(**data)

    def save(self):
        self._validate()
        path = self._config_path()
        tmp_path = path + ".tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=2)
            os.replace(tmp_path, path)
        except Exception as exc:
            ida_kernwin.msg("[%s] Failed to save config: %s\n" % (PLUGIN_NAME, exc))

    def to_dict(self):
        return {key: getattr(self, key) for key in self._FIELDS}

    def clone(self):
        return PluginConfig(**self.to_dict())

    def _validate(self):
        self.base_url = (self.base_url or DEFAULT_CONFIG["base_url"]).strip().rstrip("/")
        if not (self.base_url.startswith("http://") or self.base_url.startswith("https://")):
            self.base_url = DEFAULT_CONFIG["base_url"]
        try:
            self.temperature = max(0.0, min(2.0, float(self.temperature)))
        except (TypeError, ValueError):
            self.temperature = DEFAULT_CONFIG["temperature"]
        try:
            self.max_tokens = max(1, int(self.max_tokens))
        except (TypeError, ValueError):
            self.max_tokens = DEFAULT_CONFIG["max_tokens"]
        try:
            self.request_timeout = max(1, int(self.request_timeout))
        except (TypeError, ValueError):
            self.request_timeout = DEFAULT_CONFIG["request_timeout"]
        try:
            self.max_context_chars = max(500, int(self.max_context_chars))
        except (TypeError, ValueError):
            self.max_context_chars = DEFAULT_CONFIG["max_context_chars"]
        try:
            self.max_callees = max(0, int(self.max_callees))
        except (TypeError, ValueError):
            self.max_callees = DEFAULT_CONFIG["max_callees"]
        try:
            self.follow_calls_depth = max(0, min(5, int(self.follow_calls_depth)))
        except (TypeError, ValueError):
            self.follow_calls_depth = DEFAULT_CONFIG["follow_calls_depth"]
        try:
            self.max_total_context_chars = max(self.max_context_chars, int(self.max_total_context_chars))
        except (TypeError, ValueError):
            self.max_total_context_chars = DEFAULT_CONFIG["max_total_context_chars"]
        try:
            self.max_auto_fetch = max(0, int(self.max_auto_fetch))
        except (TypeError, ValueError):
            self.max_auto_fetch = DEFAULT_CONFIG["max_auto_fetch"]
        self.include_data_refs = bool(self.include_data_refs)
        try:
            self.max_data_refs = max(0, int(self.max_data_refs))
        except (TypeError, ValueError):
            self.max_data_refs = DEFAULT_CONFIG["max_data_refs"]
        try:
            self.max_string_len = max(20, min(2000, int(self.max_string_len)))
        except (TypeError, ValueError):
            self.max_string_len = DEFAULT_CONFIG["max_string_len"]
        try:
            self.max_recursive_callees = max(1, min(100, int(self.max_recursive_callees)))
        except (TypeError, ValueError):
            self.max_recursive_callees = DEFAULT_CONFIG["max_recursive_callees"]
        self.model = (self.model or "").strip()
        self.api_key = (self.api_key or "").strip()
        self.system_prompt = self.system_prompt or DEFAULT_SYSTEM_PROMPT
        self.explain_hotkey = (self.explain_hotkey or "").strip()
        self.include_callees = bool(self.include_callees)


# ---------------------------------------------------------------------------
# Networking (background thread, SSE streaming over urllib)
# ---------------------------------------------------------------------------

class LlamaStreamWorker(threading.Thread):
    """Runs one chat-completion request against llama-server, streaming the
    answer via SSE. All callbacks are marshalled onto IDA's main thread with
    execute_sync; this thread never touches Qt widgets or the IDA database
    directly.
    """

    def __init__(self, config, messages, on_delta, on_reasoning_delta, on_done, on_error):
        super().__init__(daemon=True)
        self._config = config
        self._messages = messages
        self._on_delta = on_delta
        self._on_reasoning_delta = on_reasoning_delta
        self._on_done = on_done
        self._on_error = on_error
        self._cancel_event = threading.Event()
        self._response = None

    def cancel(self):
        self._cancel_event.set()
        resp = self._response
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    def run(self):
        try:
            self._stream()
        except Exception as exc:
            if not self._cancel_event.is_set():
                ida_kernwin.execute_sync(
                    functools.partial(self._on_error, str(exc)), ida_kernwin.MFF_FAST
                )

    def _stream(self):
        url = self._config.base_url + "/v1/chat/completions"
        payload = {
            "messages": self._messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "stream": True,
        }
        if self._config.model:
            payload["model"] = self._config.model
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self._config.api_key:
            headers["Authorization"] = "Bearer " + self._config.api_key

        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            self._response = urllib.request.urlopen(req, timeout=self._config.request_timeout)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:2000]
            except Exception:
                pass
            raise RuntimeError("HTTP %s: %s" % (exc.code, body)) from None
        except urllib.error.URLError as exc:
            raise RuntimeError("Cannot connect to %s (%s)" % (url, exc.reason)) from None

        parts = []
        reasoning_parts = []
        finish_reason = [None]
        with self._response as resp:
            content_type = resp.headers.get("Content-Type", "")
            if not content_type.startswith("text/event-stream"):
                self._handle_non_stream_body(resp.read(), parts, reasoning_parts, finish_reason)
            else:
                for raw_line in resp:
                    if self._cancel_event.is_set():
                        return
                    line = raw_line.decode("utf-8", "replace").strip("\r\n")
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_str)
                    except ValueError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    if choices[0].get("finish_reason"):
                        finish_reason[0] = choices[0].get("finish_reason")
                    # Reasoning/"thinking" models (e.g. Qwen3) stream their
                    # chain-of-thought separately from the real answer.
                    reasoning_piece = delta.get("reasoning_content")
                    if reasoning_piece:
                        reasoning_parts.append(reasoning_piece)
                        ida_kernwin.execute_sync(
                            functools.partial(self._on_reasoning_delta, reasoning_piece), ida_kernwin.MFF_FAST
                        )
                    piece = delta.get("content")
                    if piece:
                        parts.append(piece)
                        ida_kernwin.execute_sync(
                            functools.partial(self._on_delta, piece), ida_kernwin.MFF_FAST
                        )

        if not self._cancel_event.is_set():
            ida_kernwin.execute_sync(
                functools.partial(
                    self._on_done, "".join(parts), "".join(reasoning_parts), finish_reason[0]
                ),
                ida_kernwin.MFF_FAST,
            )

    def _handle_non_stream_body(self, body, parts, reasoning_parts, finish_reason):
        try:
            obj = json.loads(body.decode("utf-8", "replace"))
        except Exception as exc:
            raise RuntimeError("Unexpected response from server: %s" % exc) from None
        choices = obj.get("choices") or []
        if not choices:
            err = obj.get("error")
            raise RuntimeError(str(err) if err else "Empty response from server.")
        finish_reason[0] = choices[0].get("finish_reason")
        message = choices[0].get("message") or {}
        reasoning = message.get("reasoning_content", "")
        if reasoning:
            reasoning_parts.append(reasoning)
            ida_kernwin.execute_sync(
                functools.partial(self._on_reasoning_delta, reasoning), ida_kernwin.MFF_FAST
            )
        content = message.get("content", "")
        if content:
            parts.append(content)
            ida_kernwin.execute_sync(
                functools.partial(self._on_delta, content), ida_kernwin.MFF_FAST
            )


# ---------------------------------------------------------------------------
# Conversation driver (no Qt/DB writes - safe for interactive and batch use)
# ---------------------------------------------------------------------------

ConversationResult = namedtuple("ConversationResult", [
    "text", "reasoning_text", "suggested_name", "suggested_signature",
    "suggested_vars", "suggested_callee_renames", "suggested_struct",
    "suggested_var_types", "root_is_pseudocode", "error",
])


class ConversationRunner(object):
    """Owns ONE function's explain conversation end-to-end: builds the
    initial prompt, drives LlamaStreamWorker request/response cycles
    (including the REQUEST_CODE auto-fetch loop), and extracts the
    SUGGESTED_* markers from the final answer. Only network I/O (via
    LlamaStreamWorker) and read-only IDA API calls happen here - no Qt
    widgets and no database writes - so the same instance/logic can drive
    either the interactive ExplainResultDialog (live streaming) or a
    headless batch controller (call start(), wait for on_result).
    """

    def __init__(self, config, func, on_delta=None, on_reasoning_delta=None, on_status=None):
        self.config = config
        self.func = func
        self.func_ea = func.start_ea
        self.messages = None
        self._on_delta = on_delta or (lambda piece: None)
        self._on_reasoning_delta = on_reasoning_delta or (lambda piece: None)
        self._on_status = on_status or (lambda text: None)
        self._fetched_eas = set()
        self._auto_fetch_rounds = 0
        self._forced_final = False
        self._root_is_pseudocode = False
        self.worker = None
        self._closed = False
        self._on_result_cb = None
        self._on_error_cb = None

    def build_initial_messages(self):
        blocks = gather_recursive_context(self.func, self.config)
        callee_funcs = (
            gather_callee_funcs(self.func, self.config.max_callees)
            if self.config.include_callees else []
        )
        callee_names = [
            ida_funcs.get_func_name(f.start_ea) or ("sub_%X" % f.start_ea) for f in callee_funcs
        ]
        data_refs = (
            gather_data_refs(self.func, self.config.max_data_refs, self.config.max_string_len)
            if self.config.include_data_refs else []
        )
        user_msg = build_user_message(self.config, self.func, blocks, callee_names, data_refs)
        self._fetched_eas = {ea for _, ea, _ in blocks}
        self._root_is_pseudocode = bool(blocks) and blocks[0][2].kind == "pseudocode"
        self.messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        return self.messages

    def start(self, on_result, on_error):
        self._on_result_cb = on_result
        self._on_error_cb = on_error
        try:
            self.build_initial_messages()
        except Exception as exc:
            self._on_error_cb("Failed to gather function context: %s" % exc)
            return
        self._issue_request()

    def send_followup(self, user_text, on_result, on_error):
        """Interactive-only: used by ExplainResultDialog's Reason More."""
        self._on_result_cb = on_result
        self._on_error_cb = on_error
        self.messages.append({"role": "user", "content": user_text})
        self._forced_final = False
        self._issue_request()

    def cancel(self):
        self._closed = True
        if self.worker is not None:
            self.worker.cancel()

    def _issue_request(self):
        self.worker = LlamaStreamWorker(
            self.config, list(self.messages),
            self._on_delta, self._on_reasoning_delta,
            self._on_worker_done, self._on_worker_error,
        )
        self.worker.start()

    def _on_worker_error(self, message):
        self.worker = None
        if not self._closed:
            self._on_error_cb(message)
        return 0

    def _on_worker_done(self, full_text, reasoning_text="", finish_reason=None):
        self.worker = None
        if self._closed:
            return 0
        text = full_text

        if not text.strip():
            if reasoning_text.strip():
                msg = (
                    "Model spent its whole token budget on reasoning and gave no "
                    "answer (finish_reason=%s)." % (finish_reason or "unknown")
                )
            else:
                msg = "Model returned an empty response (finish_reason=%s)." % (finish_reason or "unknown")
            self._on_result_cb(ConversationResult(
                text="", reasoning_text=reasoning_text, suggested_name=None,
                suggested_signature=None, suggested_vars=[], suggested_callee_renames=[],
                suggested_struct=None, suggested_var_types=[],
                root_is_pseudocode=self._root_is_pseudocode, error=msg,
            ))
            return 0

        self.messages.append({"role": "assistant", "content": text})

        requests = _REQUEST_CODE_RE.findall(text)
        if requests and not self._forced_final:
            if self._auto_fetch_rounds < self.config.max_auto_fetch:
                self._auto_fetch_rounds += 1
                self._on_status("Fetching requested code (%s)..." % ", ".join(requests))
                self._handle_code_requests(requests)
                return 0
            self._forced_final = True
            self.messages.append({
                "role": "user",
                "content": (
                    "You have reached the maximum number of code requests (%d). "
                    "Please give your best explanation now based on the "
                    "information already gathered, without requesting further "
                    "code." % self.config.max_auto_fetch
                ),
            })
            self._on_status("Auto-fetch limit reached; asking for a final answer...")
            self._issue_request()
            return 0

        suggested_name = None
        name_matches = _SUGGESTED_NAME_RE.findall(text)
        if name_matches:
            text = _SUGGESTED_NAME_RE.sub("", text).strip()
            suggested_name = sanitize_suggested_name(name_matches[-1])

        suggested_signature = None
        sig_matches = _SUGGESTED_SIGNATURE_RE.findall(text)
        if sig_matches:
            text = _SUGGESTED_SIGNATURE_RE.sub("", text).strip()
            candidate = sig_matches[-1].strip()
            if candidate and self._root_is_pseudocode:
                suggested_signature = candidate

        suggested_vars = []
        var_matches = _SUGGESTED_VAR_RE.findall(text)
        if var_matches:
            text = _SUGGESTED_VAR_RE.sub("", text).strip()
            if self._root_is_pseudocode:
                seen_old = set()
                for old_name, new_name_var in var_matches:
                    if old_name != new_name_var and old_name not in seen_old:
                        seen_old.add(old_name)
                        suggested_vars.append((old_name, new_name_var))

        suggested_callee_renames = []
        callee_matches = _SUGGESTED_CALLEE_NAME_RE.findall(text)
        if callee_matches:
            text = _SUGGESTED_CALLEE_NAME_RE.sub("", text).strip()
            seen_callee_eas = set()
            for query, callee_new_name_raw in callee_matches:
                callee_new_name = sanitize_suggested_name(callee_new_name_raw)
                if not callee_new_name:
                    ida_kernwin.msg(
                        "[%s] Ignored SUGGESTED_CALLEE_NAME for '%s': '%s' "
                        "is not a valid identifier.\n"
                        % (PLUGIN_NAME, query, callee_new_name_raw)
                    )
                    continue
                callee_func = resolve_function_query(query.strip())
                if callee_func is None:
                    ida_kernwin.msg(
                        "[%s] Ignored SUGGESTED_CALLEE_NAME: could not "
                        "resolve '%s' to a function.\n" % (PLUGIN_NAME, query)
                    )
                    continue
                callee_ea = callee_func.start_ea
                if callee_ea == self.func_ea or callee_ea in seen_callee_eas:
                    continue
                # Only allow renaming functions whose code the model actually
                # saw (eagerly included or fetched via REQUEST_CODE this
                # conversation), and only when still under a default name -
                # never overwrite a name a human already gave it. Both
                # rejections are logged (rather than silently dropped) since
                # from the outside "the LLM proposed a name but nothing
                # happened" looks exactly like a bug otherwise.
                if callee_ea not in self._fetched_eas:
                    ida_kernwin.msg(
                        "[%s] Ignored SUGGESTED_CALLEE_NAME for '%s' -> "
                        "'%s': the model never actually requested/received "
                        "this function's code in this conversation, so the "
                        "rename was not applied. Ask a follow-up (Reason "
                        "More) telling it to REQUEST_CODE that function "
                        "first if you want this considered.\n"
                        % (PLUGIN_NAME, query, callee_new_name)
                    )
                    continue
                current_callee_name = ida_funcs.get_func_name(callee_ea) or ("sub_%X" % callee_ea)
                if not is_auto_generated_name(current_callee_name):
                    ida_kernwin.msg(
                        "[%s] Ignored SUGGESTED_CALLEE_NAME for '%s': it "
                        "already has a non-default name ('%s'); not "
                        "overwriting it automatically.\n"
                        % (PLUGIN_NAME, query, current_callee_name)
                    )
                    continue
                seen_callee_eas.add(callee_ea)
                suggested_callee_renames.append((callee_ea, callee_new_name))

        suggested_struct = None
        struct_matches = _SUGGESTED_STRUCT_RE.findall(text)
        if struct_matches:
            text = _SUGGESTED_STRUCT_RE.sub("", text).strip()
            if self._root_is_pseudocode:
                candidate = struct_matches[-1].strip()
                if candidate:
                    suggested_struct = candidate

        suggested_var_types = []
        vartype_matches = _SUGGESTED_VAR_TYPE_RE.findall(text)
        if vartype_matches:
            text = _SUGGESTED_VAR_TYPE_RE.sub("", text).strip()
            if self._root_is_pseudocode:
                seen_vartype_names = set()
                for var_name, type_expr in vartype_matches:
                    type_expr = type_expr.strip()
                    if var_name and type_expr and var_name not in seen_vartype_names:
                        seen_vartype_names.add(var_name)
                        suggested_var_types.append((var_name, type_expr))

        text = strip_markdown_fences(_REQUEST_CODE_RE.sub("", text).strip())

        self._on_result_cb(ConversationResult(
            text=text, reasoning_text=reasoning_text,
            suggested_name=suggested_name, suggested_signature=suggested_signature,
            suggested_vars=suggested_vars, suggested_callee_renames=suggested_callee_renames,
            suggested_struct=suggested_struct, suggested_var_types=suggested_var_types,
            root_is_pseudocode=self._root_is_pseudocode,
            error=None,
        ))
        return 0

    def _handle_code_requests(self, requests):
        reply_parts = []
        queried = []
        seen_this_round = set()
        for query in requests:
            query = query.strip()
            if not query or query in seen_this_round:
                continue
            seen_this_round.add(query)
            queried.append(query)
            func = resolve_function_query(query)
            if func is None:
                reply_parts.append("No function found matching '%s'." % query)
                continue
            if func.start_ea in self._fetched_eas:
                name = ida_funcs.get_func_name(func.start_ea) or ("sub_%X" % func.start_ea)
                reply_parts.append("You already have the code for %s (see above)." % name)
                continue
            try:
                ctx = gather_function_context(func)
            except Exception as exc:
                reply_parts.append("Failed to retrieve code for '%s': %s" % (query, exc))
                continue
            self._fetched_eas.add(func.start_ea)
            reply_parts.append(format_function_block("Requested function", func.start_ea, ctx, self.config))

        reply_text = "\n\n".join(reply_parts) if reply_parts else "No additional code available."
        self.messages.append({"role": "user", "content": reply_text})
        self._issue_request()


# ---------------------------------------------------------------------------
# UI: result dialog
# ---------------------------------------------------------------------------

class ExplainResultDialog(QtWidgets.QDialog):
    def __init__(self, config, func, parent=None):
        super().__init__(parent)
        self.config = config
        self.func_ea = func.start_ea
        self.func_name = ida_funcs.get_func_name(func.start_ea) or ("sub_%X" % func.start_ea)
        self._buffer = []
        self._reasoning_buffer = []
        self._reasoning_shown = False
        self._last_answer_text = ""
        self._closed = False
        self._suggested_vars = []
        self._suggested_callee_renames = []
        self._suggested_var_types = []

        self.setWindowTitle("%s - %s @ %#x" % (PLUGIN_NAME, self.func_name, func.start_ea))
        self.resize(560, 620)

        layout = QtWidgets.QVBoxLayout(self)

        self.status_label = QtWidgets.QLabel("Contacting model...")
        layout.addWidget(self.status_label)

        self.stream_edit = QtWidgets.QPlainTextEdit()
        self.stream_edit.setReadOnly(True)
        self.stream_edit.setFont(QtGui.QFont("Consolas", 9))
        layout.addWidget(self.stream_edit, 1)

        followup_layout = QtWidgets.QHBoxLayout()
        self.followup_input = QtWidgets.QLineEdit()
        self.followup_input.setPlaceholderText(
            "Ask a follow-up question or request more detail, then click Reason More..."
        )
        self.followup_input.returnPressed.connect(self.on_reason_more)
        followup_layout.addWidget(self.followup_input, 1)
        self.reason_button = QtWidgets.QPushButton("Reason More")
        self.reason_button.clicked.connect(self.on_reason_more)
        self.reason_button.setEnabled(False)
        followup_layout.addWidget(self.reason_button)
        layout.addLayout(followup_layout)

        rename_layout = QtWidgets.QHBoxLayout()
        self.rename_check = QtWidgets.QCheckBox("Rename function to:")
        self.rename_check.setEnabled(False)
        rename_layout.addWidget(self.rename_check)
        self.rename_edit = QtWidgets.QLineEdit()
        self.rename_edit.setEnabled(False)
        rename_layout.addWidget(self.rename_edit, 1)
        layout.addLayout(rename_layout)

        signature_layout = QtWidgets.QHBoxLayout()
        self.signature_check = QtWidgets.QCheckBox("Apply suggested signature:")
        self.signature_check.setEnabled(False)
        signature_layout.addWidget(self.signature_check)
        self.signature_edit = QtWidgets.QLineEdit()
        self.signature_edit.setEnabled(False)
        signature_layout.addWidget(self.signature_edit, 1)
        layout.addLayout(signature_layout)

        varrename_layout = QtWidgets.QHBoxLayout()
        self.varrename_check = QtWidgets.QCheckBox("Apply suggested variable renames:")
        self.varrename_check.setEnabled(False)
        varrename_layout.addWidget(self.varrename_check)
        self.varrename_label = QtWidgets.QLineEdit()
        self.varrename_label.setReadOnly(True)
        self.varrename_label.setEnabled(False)
        varrename_layout.addWidget(self.varrename_label, 1)
        layout.addLayout(varrename_layout)

        calleerename_layout = QtWidgets.QHBoxLayout()
        self.calleerename_check = QtWidgets.QCheckBox("Rename called function(s) with default names:")
        self.calleerename_check.setEnabled(False)
        calleerename_layout.addWidget(self.calleerename_check)
        self.calleerename_label = QtWidgets.QLineEdit()
        self.calleerename_label.setReadOnly(True)
        self.calleerename_label.setEnabled(False)
        calleerename_layout.addWidget(self.calleerename_label, 1)
        layout.addLayout(calleerename_layout)

        struct_layout = QtWidgets.QHBoxLayout()
        self.struct_check = QtWidgets.QCheckBox("Create struct type:")
        self.struct_check.setEnabled(False)
        struct_layout.addWidget(self.struct_check)
        self.struct_edit = QtWidgets.QLineEdit()
        self.struct_edit.setEnabled(False)
        struct_layout.addWidget(self.struct_edit, 1)
        layout.addLayout(struct_layout)

        vartype_layout = QtWidgets.QHBoxLayout()
        self.vartype_check = QtWidgets.QCheckBox("Apply suggested variable types:")
        self.vartype_check.setEnabled(False)
        vartype_layout.addWidget(self.vartype_check)
        self.vartype_label = QtWidgets.QLineEdit()
        self.vartype_label.setReadOnly(True)
        self.vartype_label.setEnabled(False)
        vartype_layout.addWidget(self.vartype_label, 1)
        layout.addLayout(vartype_layout)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)
        self.accept_button = QtWidgets.QPushButton("Accept && Add Comment")
        self.accept_button.clicked.connect(self.on_accept)
        self.accept_button.setEnabled(False)
        button_layout.addWidget(self.accept_button)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        _add_copyright_footer(layout)

        self.runner = ConversationRunner(
            config, func,
            on_delta=self._on_delta,
            on_reasoning_delta=self._on_reasoning_delta,
            on_status=self._on_runner_status,
        )
        self._begin_request("Querying model...")
        self.runner.start(self._on_conversation_result, self._on_conversation_error)

    # -- request lifecycle --------------------------------------------------

    def _begin_request(self, status_text="Querying model..."):
        self._buffer = []
        self._reasoning_buffer = []
        self._reasoning_shown = False
        self.status_label.setText(status_text)
        self.reason_button.setEnabled(False)
        self.followup_input.setEnabled(False)

    def _on_runner_status(self, text):
        self._begin_request(text)
        try:
            self.stream_edit.appendPlainText("\n\n--- %s ---\n" % text)
        except RuntimeError:
            pass

    # -- worker callbacks (run on IDA's main thread via execute_sync) -------

    def _insert_styled(self, text, italic=False, color=None):
        try:
            cursor = self.stream_edit.textCursor()
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
            fmt = QtGui.QTextCharFormat()
            fmt.setFontItalic(italic)
            if color is not None:
                fmt.setForeground(color)
            cursor.insertText(text, fmt)
            self.stream_edit.setTextCursor(cursor)
            self.stream_edit.ensureCursorVisible()
        except RuntimeError:
            pass

    def _on_reasoning_delta(self, piece):
        if self._closed:
            return 0
        self._reasoning_buffer.append(piece)
        if not self._reasoning_shown:
            self._reasoning_shown = True
            self._insert_styled("[thinking] ", italic=True, color=QtGui.QColor("gray"))
        self._insert_styled(piece, italic=True, color=QtGui.QColor("gray"))
        return 0

    def _on_delta(self, piece):
        if self._closed:
            return 0
        if self._reasoning_shown:
            self._reasoning_shown = False
            self._insert_styled("\n\n")
        self._buffer.append(piece)
        self._insert_styled(piece)
        return 0

    # -- conversation-runner callbacks ---------------------------------------

    def _on_conversation_result(self, result):
        if self._closed:
            return
        if result.error:
            self.status_label.setText(result.error)
            self.reason_button.setEnabled(True)
            self.followup_input.setEnabled(True)
            partial = "".join(self._buffer).strip()
            self.accept_button.setEnabled(bool(partial))
            return

        if result.suggested_name:
            self.rename_edit.setText(result.suggested_name)
            self.rename_check.setEnabled(True)
            self.rename_edit.setEnabled(True)
            self.rename_check.setChecked(is_auto_generated_name(self.func_name))

        if result.suggested_signature:
            self.signature_edit.setText(result.suggested_signature)
            self.signature_check.setEnabled(True)
            self.signature_edit.setEnabled(True)
            self.signature_check.setChecked(True)

        if result.suggested_vars:
            self._suggested_vars = result.suggested_vars
            self.varrename_label.setText(", ".join("%s -> %s" % p for p in result.suggested_vars))
            self.varrename_check.setEnabled(True)
            self.varrename_label.setEnabled(True)
            self.varrename_check.setChecked(True)

        if result.suggested_callee_renames:
            self._suggested_callee_renames = result.suggested_callee_renames
            labels = []
            for callee_ea, callee_new_name in result.suggested_callee_renames:
                old_callee_name = ida_funcs.get_func_name(callee_ea) or ("sub_%X" % callee_ea)
                labels.append("%s -> %s" % (old_callee_name, callee_new_name))
            self.calleerename_label.setText(", ".join(labels))
            self.calleerename_check.setEnabled(True)
            self.calleerename_label.setEnabled(True)
            self.calleerename_check.setChecked(True)

        if result.suggested_struct:
            self.struct_edit.setText(result.suggested_struct)
            self.struct_check.setEnabled(True)
            self.struct_edit.setEnabled(True)
            self.struct_check.setChecked(True)

        if result.suggested_var_types:
            self._suggested_var_types = result.suggested_var_types
            self.vartype_label.setText(", ".join("%s -> %s" % p for p in result.suggested_var_types))
            self.vartype_check.setEnabled(True)
            self.vartype_label.setEnabled(True)
            self.vartype_check.setChecked(True)

        self._last_answer_text = result.text
        self.status_label.setText("Done.")
        self.reason_button.setEnabled(True)
        self.followup_input.setEnabled(True)
        self.accept_button.setEnabled(bool(result.text.strip()))

    def _on_conversation_error(self, message):
        if self._closed:
            return
        self.status_label.setText("Error: %s" % message)
        self.reason_button.setEnabled(True)
        self.followup_input.setEnabled(True)
        partial = "".join(self._buffer).strip()
        self.accept_button.setEnabled(bool(partial or self._last_answer_text.strip()))

    # -- button handlers ------------------------------------------------

    def on_reason_more(self):
        if self.runner.worker is not None:
            return
        followup = self.followup_input.text().strip()
        if not followup:
            followup = "Please explain your reasoning in more detail."
        self.followup_input.clear()
        self._begin_request("Querying model...")
        try:
            self.stream_edit.appendPlainText("\n\n--- Follow-up: %s ---\n" % followup)
        except RuntimeError:
            pass
        self.runner.send_followup(followup, self._on_conversation_result, self._on_conversation_error)

    def on_accept(self):
        text = (self._last_answer_text or "".join(self._buffer)).strip()
        if not text:
            ida_kernwin.warning("Nothing to accept yet.")
            return
        # Defensive: still strip markers/fences here even though
        # ConversationRunner already does this for a normal successful
        # result, because the "".join(self._buffer) fallback above (used
        # when accepting a partial answer after a transport error) never
        # passes through that cleanup.
        comment = strip_markdown_fences(text)
        comment = _REQUEST_CODE_RE.sub("", comment)
        comment = _SUGGESTED_NAME_RE.sub("", comment)
        comment = _SUGGESTED_SIGNATURE_RE.sub("", comment)
        comment = _SUGGESTED_VAR_RE.sub("", comment)
        comment = _SUGGESTED_CALLEE_NAME_RE.sub("", comment)
        comment = _SUGGESTED_STRUCT_RE.sub("", comment)
        comment = _SUGGESTED_VAR_TYPE_RE.sub("", comment).strip()

        new_name = None
        if self.rename_check.isChecked():
            new_name = sanitize_suggested_name(self.rename_edit.text())
            if not new_name:
                ida_kernwin.warning(
                    "'%s' is not a valid function name; skipping rename." % self.rename_edit.text()
                )

        signature = None
        if self.signature_check.isChecked() and self.signature_edit.text().strip():
            signature = self.signature_edit.text().strip()

        var_renames = list(self._suggested_vars) if (self.varrename_check.isChecked() and self._suggested_vars) else None
        callee_renames = (
            list(self._suggested_callee_renames)
            if (self.calleerename_check.isChecked() and self._suggested_callee_renames) else None
        )

        struct_decl = None
        if self.struct_check.isChecked() and self.struct_edit.text().strip():
            struct_decl = self.struct_edit.text().strip()

        var_types = (
            list(self._suggested_var_types)
            if (self.vartype_check.isChecked() and self._suggested_var_types) else None
        )

        ida_kernwin.execute_sync(
            functools.partial(
                _apply_suggestions_and_refresh, self.func_ea, comment, new_name, signature,
                var_renames, callee_renames, struct_decl, var_types,
            ),
            ida_kernwin.MFF_WRITE,
        )
        self.close()

    def closeEvent(self, event):
        self._closed = True
        self.runner.cancel()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# UI: settings dialog
# ---------------------------------------------------------------------------

class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("%s Settings" % PLUGIN_NAME)
        self.resize(480, 520)
        self.result_config = None
        self._base_config = config

        form = QtWidgets.QFormLayout()

        self.base_url_edit = QtWidgets.QLineEdit()
        form.addRow("Server base URL:", self.base_url_edit)

        self.model_edit = QtWidgets.QLineEdit()
        self.model_edit.setPlaceholderText("(optional - leave blank to use server default)")
        form.addRow("Model name:", self.model_edit)

        self.api_key_edit = QtWidgets.QLineEdit()
        self.api_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("(optional bearer token)")
        form.addRow("API key:", self.api_key_edit)

        self.temperature_spin = QtWidgets.QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        form.addRow("Temperature:", self.temperature_spin)

        self.max_tokens_spin = QtWidgets.QSpinBox()
        self.max_tokens_spin.setRange(1, 262144)
        self.max_tokens_spin.setToolTip(
            "Reasoning/thinking models can spend thousands of tokens on "
            "chain-of-thought before producing a real answer - keep this "
            "generous, well below the server's context size."
        )
        form.addRow("Max tokens:", self.max_tokens_spin)

        self.timeout_spin = QtWidgets.QSpinBox()
        self.timeout_spin.setRange(1, 3600)
        self.timeout_spin.setToolTip(
            "This is a per-chunk socket timeout, not a total-generation cap - "
            "it mainly needs headroom for slow prompt processing or a stalled "
            "gap between streamed tokens on local/CPU inference."
        )
        form.addRow("Request timeout (s):", self.timeout_spin)

        self.max_context_spin = QtWidgets.QSpinBox()
        self.max_context_spin.setRange(500, 200000)
        self.max_context_spin.setSingleStep(500)
        form.addRow("Max context chars:", self.max_context_spin)

        self.include_callees_check = QtWidgets.QCheckBox("Include called-function names in prompt")
        form.addRow(self.include_callees_check)

        self.max_callees_spin = QtWidgets.QSpinBox()
        self.max_callees_spin.setRange(0, 200)
        form.addRow("Max callees listed:", self.max_callees_spin)

        self.include_data_refs_check = QtWidgets.QCheckBox("Include referenced strings/globals in prompt")
        form.addRow(self.include_data_refs_check)

        self.max_data_refs_spin = QtWidgets.QSpinBox()
        self.max_data_refs_spin.setRange(0, 200)
        form.addRow("Max data refs listed:", self.max_data_refs_spin)

        self.max_string_len_spin = QtWidgets.QSpinBox()
        self.max_string_len_spin.setRange(20, 2000)
        self.max_string_len_spin.setSingleStep(10)
        form.addRow("Max string length shown:", self.max_string_len_spin)

        self.follow_calls_spin = QtWidgets.QSpinBox()
        self.follow_calls_spin.setRange(0, 5)
        self.follow_calls_spin.setToolTip(
            "0 = only the target function. N>0 eagerly includes the code of "
            "called functions up to N levels deep in the initial prompt."
        )
        form.addRow("Follow calls depth:", self.follow_calls_spin)

        self.max_total_context_spin = QtWidgets.QSpinBox()
        self.max_total_context_spin.setRange(1000, 1000000)
        self.max_total_context_spin.setSingleStep(1000)
        self.max_total_context_spin.setToolTip(
            "Overall char budget across the target function plus all "
            "eagerly-followed called functions combined."
        )
        form.addRow("Max total context chars:", self.max_total_context_spin)

        self.max_auto_fetch_spin = QtWidgets.QSpinBox()
        self.max_auto_fetch_spin.setRange(0, 50)
        self.max_auto_fetch_spin.setToolTip(
            "Max number of automatic REQUEST_CODE round-trips the model may "
            "make per conversation before being asked for a final answer."
        )
        form.addRow("Max on-demand code requests:", self.max_auto_fetch_spin)

        self.max_recursive_callees_spin = QtWidgets.QSpinBox()
        self.max_recursive_callees_spin.setRange(1, 100)
        self.max_recursive_callees_spin.setToolTip(
            "Cap on how many direct callees 'Explain function with LLM "
            "(recursively)' will also explain and auto-accept, in addition "
            "to the target function itself. Kept separate from and smaller "
            "than 'Max callees listed' since this auto-writes to the "
            "database without a review step."
        )
        form.addRow("Max recursive callees:", self.max_recursive_callees_spin)

        self.hotkey_edit = QtWidgets.QLineEdit()
        self.hotkey_edit.setPlaceholderText("e.g. Ctrl-Alt-E (leave blank for none)")
        form.addRow("Explain hotkey:", self.hotkey_edit)

        self.system_prompt_edit = QtWidgets.QPlainTextEdit()
        self.system_prompt_edit.setMinimumHeight(140)
        form.addRow("System prompt:", self.system_prompt_edit)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self._on_restore_defaults
        )

        outer = QtWidgets.QVBoxLayout(self)
        outer.addLayout(form)
        outer.addWidget(buttons)
        _add_copyright_footer(outer)

        self._populate(config)

    def _populate(self, config):
        self.base_url_edit.setText(config.base_url)
        self.model_edit.setText(config.model)
        self.api_key_edit.setText(config.api_key)
        self.temperature_spin.setValue(config.temperature)
        self.max_tokens_spin.setValue(config.max_tokens)
        self.timeout_spin.setValue(config.request_timeout)
        self.max_context_spin.setValue(config.max_context_chars)
        self.include_callees_check.setChecked(config.include_callees)
        self.max_callees_spin.setValue(config.max_callees)
        self.include_data_refs_check.setChecked(config.include_data_refs)
        self.max_data_refs_spin.setValue(config.max_data_refs)
        self.max_string_len_spin.setValue(config.max_string_len)
        self.follow_calls_spin.setValue(config.follow_calls_depth)
        self.max_total_context_spin.setValue(config.max_total_context_chars)
        self.max_auto_fetch_spin.setValue(config.max_auto_fetch)
        self.max_recursive_callees_spin.setValue(config.max_recursive_callees)
        self.hotkey_edit.setText(config.explain_hotkey)
        self.system_prompt_edit.setPlainText(config.system_prompt)

    def _on_restore_defaults(self):
        answer = QtWidgets.QMessageBox.question(
            self,
            "Restore Defaults",
            "Reset all settings on this screen to their default values?\n"
            "Nothing is saved until you click OK.",
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self._populate(PluginConfig())

    def _on_ok(self):
        cfg = self._base_config.clone()
        cfg.base_url = self.base_url_edit.text()
        cfg.model = self.model_edit.text()
        cfg.api_key = self.api_key_edit.text()
        cfg.temperature = self.temperature_spin.value()
        cfg.max_tokens = self.max_tokens_spin.value()
        cfg.request_timeout = self.timeout_spin.value()
        cfg.max_context_chars = self.max_context_spin.value()
        cfg.include_callees = self.include_callees_check.isChecked()
        cfg.max_callees = self.max_callees_spin.value()
        cfg.include_data_refs = self.include_data_refs_check.isChecked()
        cfg.max_data_refs = self.max_data_refs_spin.value()
        cfg.max_string_len = self.max_string_len_spin.value()
        cfg.follow_calls_depth = self.follow_calls_spin.value()
        cfg.max_total_context_chars = self.max_total_context_spin.value()
        cfg.max_auto_fetch = self.max_auto_fetch_spin.value()
        cfg.max_recursive_callees = self.max_recursive_callees_spin.value()
        cfg.explain_hotkey = self.hotkey_edit.text().strip()
        cfg.system_prompt = self.system_prompt_edit.toPlainText()
        cfg._validate()
        self.result_config = cfg
        self.accept()


# ---------------------------------------------------------------------------
# UI: batch explain
# ---------------------------------------------------------------------------

class BatchPickerDialog(QtWidgets.QDialog):
    """Checkable list of every function in the database, with a live text
    filter (hides/shows rows, never rebuilds the list - checked state
    survives filtering). Nothing is pre-checked; the user picks explicitly.

    Deliberately does NOT try to read the native Functions-window's current
    multi-selection (ida_kernwin.get_chooser_selection) to pre-populate
    this - that API's exact behavior, and whether its row order always
    matches idautils.Functions() address order (it may not if the user
    re-sorted a column), were not confidently verified for this plugin.
    A fully self-contained picker avoids that risk entirely.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("%s - Batch Explain: Select Functions" % PLUGIN_NAME)
        self.resize(520, 640)

        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Filter by name...")
        self.filter_edit.textChanged.connect(self._apply_filter)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setUniformItemSizes(True)
        self._populate()

        select_all_btn = QtWidgets.QPushButton("Select All Filtered")
        select_all_btn.clicked.connect(lambda: self._set_checked_filtered(True))
        deselect_all_btn = QtWidgets.QPushButton("Deselect All Filtered")
        deselect_all_btn.clicked.connect(lambda: self._set_checked_filtered(False))

        self.count_label = QtWidgets.QLabel("0 selected")
        self.list_widget.itemChanged.connect(self._update_count)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.filter_edit)
        layout.addWidget(self.list_widget, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(select_all_btn)
        row.addWidget(deselect_all_btn)
        row.addStretch(1)
        row.addWidget(self.count_label)
        layout.addLayout(row)
        layout.addWidget(buttons)
        _add_copyright_footer(layout)

    def _populate(self):
        self.list_widget.setUpdatesEnabled(False)
        for ea in idautils.Functions():
            name = ida_funcs.get_func_name(ea) or ("sub_%X" % ea)
            item = QtWidgets.QListWidgetItem("%s @ %#010x" % (name, ea))
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, ea)
            self.list_widget.addItem(item)
        self.list_widget.setUpdatesEnabled(True)

    def _apply_filter(self, text):
        needle = text.strip().lower()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def _set_checked_filtered(self, checked):
        state = QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not item.isHidden():
                item.setCheckState(state)

    def _update_count(self, _item=None):
        n = sum(
            1 for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState() == QtCore.Qt.CheckState.Checked
        )
        self.count_label.setText("%d selected" % n)

    def get_selected_functions(self):
        result = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                ea = item.data(QtCore.Qt.ItemDataRole.UserRole)
                func = ida_funcs.get_func(ea)
                if func is not None:
                    result.append(func)
        return result


BatchItemResult = namedtuple("BatchItemResult", [
    "func_ea", "orig_name", "ok", "message", "comment",
    "suggested_name", "suggested_signature", "suggested_vars",
    "suggested_callee_renames", "suggested_struct", "suggested_var_types",
    "root_is_pseudocode",
])


def _compute_apply_args(item):
    """Given a BatchItemResult, compute the positional args tuple for
    _apply_suggestions_and_refresh, using the same conservative defaults
    used throughout the plugin (rename only when the current name looks
    auto-generated; signature/variable/struct suggestions only when the
    root context was Hex-Rays pseudocode). Returns None if there's nothing
    to apply (missing or failed result).
    """
    if item is None or not item.ok:
        return None
    new_name = item.suggested_name if (item.suggested_name and is_auto_generated_name(item.orig_name)) else None
    signature = item.suggested_signature if item.root_is_pseudocode else None
    var_renames = item.suggested_vars if (item.root_is_pseudocode and item.suggested_vars) else None
    callee_renames = item.suggested_callee_renames or None
    struct_decl = item.suggested_struct if item.root_is_pseudocode else None
    var_types = item.suggested_var_types if (item.root_is_pseudocode and item.suggested_var_types) else None
    return (item.func_ea, item.comment, new_name, signature, var_renames, callee_renames, struct_decl, var_types)


class BatchController(object):
    """Drives a sequence of ConversationRunners, one function at a time, on
    IDA's main thread. No separate background thread is needed:
    ConversationRunner.start()/send_followup() already return immediately
    (spawning a LlamaStreamWorker thread), and that worker's completion
    callbacks already arrive on the main thread via execute_sync. So the
    batch driver is just a plain object whose on_result/on_error callback,
    once invoked, starts the next function.
    """

    def __init__(self, config, funcs, on_row_update, on_finished, on_item_result=None):
        self.config = config
        self.funcs = funcs
        self._on_row_update = on_row_update
        self._on_finished = on_finished
        self._on_item_result = on_item_result or (lambda item: None)
        self._index = 0
        self._cancelled = False
        self._runner = None
        self.results = {}

    def start(self):
        self._process_next()

    def cancel(self):
        """Cancelling an in-flight LlamaStreamWorker suppresses BOTH its
        on_done and on_error callbacks (both gated by the worker's own
        cancel_event check), so no completion callback will ever arrive
        for the in-flight function. This must therefore mark the current
        and all remaining rows as Cancelled synchronously, rather than
        waiting for a callback that will never come.
        """
        if self._cancelled:
            return
        self._cancelled = True
        if self._runner is not None:
            self._runner.cancel()
            self._on_row_update(self._index, "Cancelled", "")
            self._runner = None
            start_remaining = self._index + 1
        else:
            start_remaining = self._index
        for i in range(start_remaining, len(self.funcs)):
            self._on_row_update(i, "Cancelled", "")
        self._on_finished()

    def _process_next(self):
        if self._cancelled or self._index >= len(self.funcs):
            self._on_finished()
            return
        func = self.funcs[self._index]
        self._on_row_update(self._index, "Running", "")
        self._runner = ConversationRunner(
            self.config, func,
            on_status=functools.partial(self._on_status, self._index),
        )
        self._runner.start(
            on_result=functools.partial(self._on_result, self._index, func),
            on_error=functools.partial(self._on_error, self._index, func),
        )

    def _on_status(self, index, text):
        self._on_row_update(index, "Running", text)

    def _record_and_advance(self, item):
        self.results[item.func_ea] = item
        self._on_item_result(item)
        self._runner = None
        self._index += 1
        self._process_next()

    def _on_result(self, index, func, result):
        orig_name = ida_funcs.get_func_name(func.start_ea) or ("sub_%X" % func.start_ea)
        if result.error:
            item = BatchItemResult(func.start_ea, orig_name, False, result.error,
                                    None, None, None, [], [], None, [], result.root_is_pseudocode)
            self._on_row_update(index, "Error", result.error)
        else:
            item = BatchItemResult(func.start_ea, orig_name, True, None, result.text,
                                    result.suggested_name, result.suggested_signature,
                                    result.suggested_vars, result.suggested_callee_renames,
                                    result.suggested_struct, result.suggested_var_types,
                                    result.root_is_pseudocode)
            self._on_row_update(index, "Done", result.text[:120])
        self._record_and_advance(item)

    def _on_error(self, index, func, message):
        orig_name = ida_funcs.get_func_name(func.start_ea) or ("sub_%X" % func.start_ea)
        item = BatchItemResult(func.start_ea, orig_name, False, message,
                                None, None, None, [], [], None, [], False)
        self._on_row_update(index, "Error", message)
        self._record_and_advance(item)


def _apply_batch_and_refresh(items):
    """items: list of (func_ea, comment, new_name, signature, var_renames,
    callee_renames, struct_decl, var_types). One execute_sync/MFF_WRITE
    round-trip for the whole batch instead of N.
    """
    for func_ea, comment, new_name, signature, var_renames, callee_renames, struct_decl, var_types in items:
        _apply_suggestions_and_refresh(
            func_ea, comment, new_name, signature, var_renames, callee_renames, struct_decl, var_types
        )
    return 1


class BatchProgressDialog(QtWidgets.QDialog):
    """Table: [Apply checkbox | Function | Status | Comment/Error preview].

    Two modes:
    - auto_apply=False (default, used by "Batch Explain Functions..."):
      never auto-writes - Cancel-while-running, then an Apply-Selected-
      after-review step, matching the plugin's human-in-the-loop
      philosophy used everywhere else.
    - auto_apply=True (used by "Explain function with LLM (recursively)"):
      each function's suggestions are applied automatically the moment
      that function finishes, with the same conservative defaults the
      manual Apply Selected step uses (see _compute_apply_args) - no
      per-item review, but this dialog still shows live progress and can
      be cancelled mid-run, and the Apply column becomes a read-only
      "Applied" indicator rather than a checkbox.

    No "Reason More" in either mode - close this dialog and use the
    interactive single-function dialog to refine any one function further.
    """

    COL_APPLY, COL_FUNC, COL_STATUS, COL_PREVIEW = range(4)

    def __init__(self, config, funcs, parent=None, auto_apply=False):
        super().__init__(parent)
        self.auto_apply = auto_apply
        title = "Recursive Explain (auto-accept)" if auto_apply else "Batch Explain Progress"
        self.setWindowTitle("%s - %s" % (PLUGIN_NAME, title))
        self.resize(720, 480)
        self.funcs = funcs
        self._row_by_ea = {f.start_ea: i for i, f in enumerate(funcs)}

        self.table = QtWidgets.QTableWidget(len(funcs), 4)
        apply_header = "Applied" if auto_apply else "Apply"
        self.table.setHorizontalHeaderLabels([apply_header, "Function", "Status", "Comment / Error"])
        self.table.horizontalHeader().setStretchLastSection(True)
        for i, func in enumerate(funcs):
            name = ida_funcs.get_func_name(func.start_ea) or ("sub_%X" % func.start_ea)
            check_item = QtWidgets.QTableWidgetItem()
            if not auto_apply:
                check_item.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            check_item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.table.setItem(i, self.COL_APPLY, check_item)
            self.table.setItem(i, self.COL_FUNC, QtWidgets.QTableWidgetItem("%s @ %#x" % (name, func.start_ea)))
            self.table.setItem(i, self.COL_STATUS, QtWidgets.QTableWidgetItem("Pending"))
            self.table.setItem(i, self.COL_PREVIEW, QtWidgets.QTableWidgetItem(""))

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel)
        self.apply_button = QtWidgets.QPushButton("Apply Selected")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._apply_selected)
        if auto_apply:
            self.apply_button.setVisible(False)
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.close)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.apply_button)
        button_row.addWidget(self.close_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.table, 1)
        layout.addLayout(button_row)
        _add_copyright_footer(layout)

        self.controller = BatchController(
            config, funcs, self._on_row_update, self._on_batch_finished,
            on_item_result=self._on_item_auto_apply if auto_apply else None,
        )
        self.controller.start()

    def _on_row_update(self, index, status, preview):
        self.table.item(index, self.COL_STATUS).setText(status)
        if preview:
            self.table.item(index, self.COL_PREVIEW).setText(preview)

    def _on_item_auto_apply(self, item):
        args = _compute_apply_args(item)
        if args is not None:
            ida_kernwin.execute_sync(
                functools.partial(_apply_suggestions_and_refresh, *args), ida_kernwin.MFF_WRITE
            )
        row = self._row_by_ea.get(item.func_ea)
        if row is not None:
            check_item = self.table.item(row, self.COL_APPLY)
            check_item.setCheckState(
                QtCore.Qt.CheckState.Checked if args is not None else QtCore.Qt.CheckState.Unchecked
            )

    def _on_batch_finished(self):
        self.cancel_button.setEnabled(False)
        if self.auto_apply:
            return
        for i, func in enumerate(self.funcs):
            item = self.controller.results.get(func.start_ea)
            check_item = self.table.item(i, self.COL_APPLY)
            if item is not None and item.ok:
                check_item.setFlags(check_item.flags() | QtCore.Qt.ItemFlag.ItemIsEnabled)
                check_item.setCheckState(QtCore.Qt.CheckState.Checked)
            else:
                check_item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.apply_button.setEnabled(True)

    def _on_cancel(self):
        self.controller.cancel()

    def _apply_selected(self):
        items_to_apply = []
        for i, func in enumerate(self.funcs):
            if self.table.item(i, self.COL_APPLY).checkState() != QtCore.Qt.CheckState.Checked:
                continue
            args = _compute_apply_args(self.controller.results.get(func.start_ea))
            if args is not None:
                items_to_apply.append(args)
        if not items_to_apply:
            return
        ida_kernwin.execute_sync(
            functools.partial(_apply_batch_and_refresh, items_to_apply), ida_kernwin.MFF_WRITE
        )
        self.apply_button.setEnabled(False)

    def closeEvent(self, event):
        self.controller.cancel()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# IDA action / popup / plugin glue
# ---------------------------------------------------------------------------

class ExplainActionHandler(ida_kernwin.action_handler_t):
    def __init__(self):
        super().__init__()

    def activate(self, ctx):
        func = _resolve_func(ctx)
        if not func:
            ida_kernwin.warning("Place the cursor inside a function first.")
            return 0
        plugin = LLMExplainerPlugin.instance
        if plugin is None:
            return 0
        plugin.open_explain_dialog(func)
        return 1

    def update(self, ctx):
        widget = getattr(ctx, "widget", None)
        wtype = ida_kernwin.get_widget_type(widget) if widget else -1
        if wtype not in (ida_kernwin.BWN_PSEUDOCODE, ida_kernwin.BWN_DISASM):
            return ida_kernwin.AST_DISABLE_FOR_WIDGET
        return ida_kernwin.AST_ENABLE_FOR_WIDGET if _resolve_func(ctx) else ida_kernwin.AST_DISABLE_FOR_WIDGET


class ExplainRecursiveActionHandler(ida_kernwin.action_handler_t):
    """Explains the target function plus its direct (depth-1 only)
    callees, and auto-accepts every result - no per-function review step,
    unlike every other action in this plugin. See LLMExplainerPlugin.
    open_recursive_explain for the safety gating this still applies.
    """

    def __init__(self):
        super().__init__()

    def activate(self, ctx):
        func = _resolve_func(ctx)
        if not func:
            ida_kernwin.warning("Place the cursor inside a function first.")
            return 0
        plugin = LLMExplainerPlugin.instance
        if plugin is None:
            return 0
        plugin.open_recursive_explain(func)
        return 1

    def update(self, ctx):
        widget = getattr(ctx, "widget", None)
        wtype = ida_kernwin.get_widget_type(widget) if widget else -1
        if wtype not in (ida_kernwin.BWN_PSEUDOCODE, ida_kernwin.BWN_DISASM):
            return ida_kernwin.AST_DISABLE_FOR_WIDGET
        return ida_kernwin.AST_ENABLE_FOR_WIDGET if _resolve_func(ctx) else ida_kernwin.AST_DISABLE_FOR_WIDGET


class BatchActionHandler(ida_kernwin.action_handler_t):
    def __init__(self):
        super().__init__()

    def activate(self, ctx):
        plugin = LLMExplainerPlugin.instance
        if plugin is None:
            return 0
        picker = BatchPickerDialog()
        if picker.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return 0
        funcs = picker.get_selected_functions()
        if not funcs:
            ida_kernwin.warning("No functions selected.")
            return 0
        plugin.open_batch_dialog(funcs)
        return 1

    def update(self, ctx):
        return ida_kernwin.AST_ENABLE_ALWAYS


class PopupHooks(ida_kernwin.UI_Hooks):
    def finish_populating_widget_popup(self, widget, popup, ctx=None):
        wtype = ida_kernwin.get_widget_type(widget)
        if wtype in (ida_kernwin.BWN_PSEUDOCODE, ida_kernwin.BWN_DISASM):
            ida_kernwin.attach_action_to_popup(widget, popup, ACTION_ID_EXPLAIN, "LLM Explainer/")
            ida_kernwin.attach_action_to_popup(widget, popup, ACTION_ID_EXPLAIN_RECURSIVE, "LLM Explainer/")
        elif wtype == ida_kernwin.BWN_FUNCS:
            ida_kernwin.attach_action_to_popup(widget, popup, ACTION_ID_BATCH, "LLM Explainer/")


class LLMExplainerPlugin(idaapi.plugin_t):
    flags = idaapi.PLUGIN_KEEP
    comment = "Ask a local llama.cpp model to explain the current function"
    help = (
        "Right-click a function in the disassembly or pseudocode view (or "
        "use the configured hotkey) to ask the configured llama.cpp server "
        "to explain it. The same menu also has 'Explain function with LLM "
        "(recursively)', which additionally explains the function's direct "
        "callees and automatically applies every result without a review "
        "step - use with care. Right-click in the Functions window to "
        "batch-explain multiple functions. Edit > Plugins > LLM Explainer "
        "opens settings."
    )
    wanted_name = PLUGIN_NAME
    wanted_hotkey = ""

    instance = None

    def init(self):
        LLMExplainerPlugin.instance = self
        self.config = PluginConfig.load()
        self._open_dialogs = []
        self._action_handler = ExplainActionHandler()
        self._recursive_action_handler = ExplainRecursiveActionHandler()
        self._batch_action_handler = BatchActionHandler()

        action = ida_kernwin.action_desc_t(
            ACTION_ID_EXPLAIN,
            "Explain function with LLM...",
            self._action_handler,
            self.config.explain_hotkey or None,
            "Ask the local LLM (llama.cpp server) to explain this function",
            -1,
        )
        if not ida_kernwin.register_action(action):
            ida_kernwin.msg("[%s] Failed to register action.\n" % PLUGIN_NAME)

        recursive_action = ida_kernwin.action_desc_t(
            ACTION_ID_EXPLAIN_RECURSIVE,
            "Explain function with LLM (recursively)...",
            self._recursive_action_handler,
            None,
            "Also explain this function's direct callees (depth 1) and "
            "auto-accept every result without a review step",
            -1,
        )
        if not ida_kernwin.register_action(recursive_action):
            ida_kernwin.msg("[%s] Failed to register recursive action.\n" % PLUGIN_NAME)

        batch_action = ida_kernwin.action_desc_t(
            ACTION_ID_BATCH,
            "Batch Explain Functions...",
            self._batch_action_handler,
            None,
            "Ask the local LLM to explain multiple selected functions",
            -1,
        )
        if not ida_kernwin.register_action(batch_action):
            ida_kernwin.msg("[%s] Failed to register batch action.\n" % PLUGIN_NAME)
        try:
            ida_kernwin.attach_action_to_menu(
                "Edit/Plugins/", ACTION_ID_BATCH, ida_kernwin.SETMENU_APP
            )
        except Exception:
            pass

        self._popup_hooks = PopupHooks()
        self._popup_hooks.hook()

        ida_kernwin.msg("[%s] v%s loaded. Server: %s\n" % (PLUGIN_NAME, PLUGIN_VERSION, self.config.base_url))
        return idaapi.PLUGIN_KEEP

    def run(self, arg):
        dlg = SettingsDialog(self.config)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted and dlg.result_config is not None:
            old_hotkey = self.config.explain_hotkey
            self.config = dlg.result_config
            self.config.save()
            if self.config.explain_hotkey != old_hotkey:
                try:
                    ida_kernwin.update_action_shortcut(ACTION_ID_EXPLAIN, self.config.explain_hotkey or None)
                except Exception:
                    try:
                        ida_kernwin.unregister_action(ACTION_ID_EXPLAIN)
                    except Exception:
                        pass
                    ida_kernwin.register_action(
                        ida_kernwin.action_desc_t(
                            ACTION_ID_EXPLAIN,
                            "Explain function with LLM...",
                            self._action_handler,
                            self.config.explain_hotkey or None,
                            "Ask the local LLM (llama.cpp server) to explain this function",
                            -1,
                        )
                    )

    def term(self):
        for dlg in list(self._open_dialogs):
            try:
                dlg.close()
            except Exception:
                pass
        self._open_dialogs = []
        try:
            self._popup_hooks.unhook()
        except Exception:
            pass
        try:
            ida_kernwin.unregister_action(ACTION_ID_EXPLAIN)
        except Exception:
            pass
        try:
            ida_kernwin.unregister_action(ACTION_ID_EXPLAIN_RECURSIVE)
        except Exception:
            pass
        try:
            ida_kernwin.unregister_action(ACTION_ID_BATCH)
        except Exception:
            pass
        LLMExplainerPlugin.instance = None

    def _track_dialog(self, dlg):
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self._open_dialogs.append(dlg)

        def _cleanup(_result=None, dialog=dlg):
            if dialog in self._open_dialogs:
                self._open_dialogs.remove(dialog)

        dlg.finished.connect(_cleanup)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        return dlg

    def open_explain_dialog(self, func):
        dlg = ExplainResultDialog(self.config, func)
        return self._track_dialog(dlg)

    def open_batch_dialog(self, funcs):
        dlg = BatchProgressDialog(self.config, funcs)
        return self._track_dialog(dlg)

    def open_recursive_explain(self, func):
        """Target function plus its direct callees only (depth 1 - callees
        of callees are not included), each explained and auto-applied.
        Capped by config.max_recursive_callees, deliberately a smaller,
        separate setting from max_callees since this writes to the
        database automatically."""
        callees = gather_callee_funcs(func, self.config.max_recursive_callees)
        funcs = [func]
        seen_eas = {func.start_ea}
        for callee in callees:
            if callee.start_ea not in seen_eas:
                seen_eas.add(callee.start_ea)
                funcs.append(callee)
        dlg = BatchProgressDialog(self.config, funcs, auto_apply=True)
        return self._track_dialog(dlg)


def PLUGIN_ENTRY():
    return LLMExplainerPlugin()
