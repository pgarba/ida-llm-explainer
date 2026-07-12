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
needed), and one or more running llama.cpp `llama-server` instances
reachable at the configured base URL(s) (default http://127.0.0.1:8080).
The Hex-Rays decompiler is optional - if it is not available for the
current architecture the plugin falls back to a plain disassembly listing
automatically.

Configure the server URL(s), model, and other options via
Edit > Plugins > LLM Explainer. Server list order is priority order: the
interactive single-function explain talks to the first configured server,
and if a request to a server is refused, drops mid-stream, or the server
answers with HTTP 502/503/504, the conversation automatically retries
against the next configured server before giving up - so a server that's
offline or overloaded is skipped rather than failing the whole request. If
llama-server is run with a single inference slot, opening several
"explain" dialogs at once will simply queue their requests on that server -
that is a server/deployment concern, not a bug in this plugin.

To process many functions at once, right-click in the Functions window (or
use Edit > Plugins > Batch Explain Functions...) to pick a set of functions,
process them, and review/apply the results in one batch - still nothing is
written to the database until you explicitly apply. If more than one
server is configured, batch explain (and the recursive auto-accept action
below) runs up to one function per server concurrently, so listing N
llama-server instances gives roughly an Nx speedup over a single server.

"Explain function with LLM (recursively)" (same right-click menu as the
regular explain action) explains the target function plus its direct
callees only (depth 1, not deeper), and - unlike every other action in
this plugin - auto-accepts every result with no review step. It still
applies the same conservative defaults as a manual Accept (e.g. only
renaming a function whose name looks auto-generated), the callee count is
capped by its own "Max recursive callees" setting, and a progress dialog
stays open so you can watch what happens and cancel mid-run.

"Trace/Recover CFG..." (disassembly view only, since the whole point is a
blob that may not even be recognized as a function) walks an obfuscated
function's basic blocks forward from a given start address - the kind
produced by control-flow flattening around a dispatcher, opaque
predicates, or decoy blocks. It auto-continues through anything with a
single successor, and at each genuine decision point (conditional
branches, indirect jumps/switch dispatchers) first tries a lightweight,
deterministic symbolic-execution pass (x86/x64 only: constant
register/stack-slot tracking through mov/lea/arithmetic/cmp/test) to
resolve it WITHOUT an LLM call - this covers both classic opaque
predicates (a dispatcher/state variable set to a known constant earlier
in the trace) and genuine data-dependent branches (a comparison against
caller-supplied/runtime data, which is ordinary conditional logic, not
obfuscation - every side gets marked real automatically). Only when this
cannot confidently resolve something does it fall back to asking the
LLM, showing a live, cancellable transcript throughout, until the
worklist is empty or the configured block cap is hit. Several structural
defenses (not just prompt wording) further guard against known LLM
failure modes for whatever still reaches it: a candidate the model never
addresses is flagged for manual review rather than silently assumed dead,
an address outside the block's actual candidates is discarded, and
conflicting REAL/DEAD verdicts for the same address from different paths
keep the first verdict and surface the conflict. A dispatcher re-entered
from a different real block later in the trace - the classic loop/
flattening-dispatcher revisit pattern, where a single-snapshot verdict
from the first pass would otherwise leave every other pass's real blocks
wrongly stuck as dead - is detected and automatically un-marks those
successors back to unresolved for review, and a candidate that loops
back to an earlier block in the trace is deferred to the LLM rather than
auto-resolved, since a value that looks fixed on one pass through a loop
is not necessarily fixed on every pass. As it walks, it also
corrects any instruction boundaries IDA's original analysis got wrong -
undefining and recreating instructions one at a time (never touching a
byte), the same "U then C" fixup a human would do manually - since
obfuscated code routinely has real jump targets landing mid-instruction
relative to IDA's initial linear-sweep guess; this happens immediately
as part of walking, not deferred to Accept, since it is just correcting
IDA's own analysis rather than expressing an LLM opinion. A separate
overlap check guards this same fixup against a different hazard: a
target discovered LATER in the trace landing strictly inside an
instruction a DIFFERENT block already claimed and fixed up earlier in
this same run (the same bytes decoding differently depending on the
start offset, another classic obfuscation trick) - naively re-walking
that address would call IDA's own del_items, which deletes the whole
item covering any address passed to it, silently destroying the earlier
block's already-established boundary even if that block was already
confirmed real. Every instruction address claimed so far in the trace is
tracked, and a strict-overlap target is refused and flagged for manual
review instead, never touching those bytes at all. An optional
live graph view (a native IDA graph, colored to match the eventual
disassembly marking) fills in as the trace runs, alongside the transcript.
Once the trace finishes, a review table lets you check/uncheck each
decided block before Accept, which by default only colors the disassembly
and adds a comment for the REAL/DEAD/UNRESOLVED verdicts themselves - no
byte patching (the "Mark only" mode). "Patch in place" additionally NOPs
confirmed dead code and redirects confirmed opaque-predicate jumps
(Jcc -> JMP toward the real target, or NOP if the real path is the
fallthrough) straight to their real target, so the recovered function can
be handed to Hex-Rays. "Rebuild linear" takes a different approach:
leaves every original byte untouched except the function's own entry
point, and writes a freshly concatenated straight-line sequence of just
the real blocks there instead, with every jump/call re-encoded
explicitly (never relying on layout adjacency) - a block using RIP-
relative addressing or a live multi-case dispatch can't be safely moved
and is left at its original address, refusing the whole rebuild rather
than guessing if that address (or a relocated block's reference to it)
would fall inside the newly rebuilt range. Every patch, in either mode,
is computed from freshly re-decoded, verified bytes right before writing
and refuses (leaving the original bytes untouched) rather than guessing
at anything it doesn't fully recognize.
"""

import functools
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque, namedtuple, OrderedDict

import idaapi
import idautils
import idc
import ida_kernwin
import ida_auto
import ida_bytes
import ida_funcs
import ida_graph
import ida_idp
import ida_lines
import ida_name
import ida_nalt
import ida_segment
import ida_typeinf
import ida_ua
import ida_xref

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
PLUGIN_VERSION = "1.6.0"
PLUGIN_COPYRIGHT = "© 2026 Peter Garba"
ACTION_ID_EXPLAIN = "llm_explainer:explain_function"
ACTION_ID_BATCH = "llm_explainer:batch_explain"
ACTION_ID_EXPLAIN_RECURSIVE = "llm_explainer:explain_recursive"
ACTION_ID_TRACE_CFG = "llm_explainer:trace_cfg"
CONFIG_FILENAME = "llm_explainer.cfg.json"


def _add_copyright_footer(layout):
    label = QtWidgets.QLabel("%s v%s - %s" % (PLUGIN_NAME, PLUGIN_VERSION, PLUGIN_COPYRIGHT))
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
    "Once you have enough information, work through ALL FIVE of the "
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
    "with REQUEST_CODE), propose a rename for it - only that function, and "
    "only if you are confident about what it does from the code you "
    "actually saw, never from its name alone or a guess - as one line per "
    "function of the exact form\n"
    "SUGGESTED_CALLEE_NAME: <its current name or address> -> <new name>\n"
    "using the same identifier rules as SUGGESTED_NAME. This applies "
    "whether the callee currently has a default auto-generated name (e.g. "
    "sub_1402346D0, loc_403010) or an existing non-default name (including "
    "one from an earlier round of this same analysis) - if, having now "
    "seen its code, you can give it a materially more accurate or specific "
    "name than what it currently has, propose the rename; only skip it "
    "when the current name is already just as good. Never for the target "
    "function itself, and never for a function whose code you never saw: "
    "this is enforced automatically, and any "
    "SUGGESTED_CALLEE_NAME for a function whose code was not actually "
    "shown to you earlier in this same conversation will be silently "
    "discarded, so do not bother emitting it in that case - request the "
    "function's code with REQUEST_CODE first if you want to propose a "
    "rename for it.\n"
    "Separately, if a called function ALREADY has a non-default name but, "
    "from how the target function uses it, you believe that existing name "
    "is likely wrong, stale, or misleading and it deserves a fresh look, "
    "you may flag it (do NOT rename it yourself from usage alone) with one "
    "line of the exact form\n"
    "SUGGESTED_REANALYZE: <its current name or address> - <brief why>\n"
    "e.g. SUGGESTED_REANALYZE: validate_user - it actually parses a TLV "
    "header, not credential validation. Use this sparingly and only when "
    "you are genuinely unsure the current name is correct; it asks the "
    "tool to re-analyze that function on its own code (only the recursive "
    "scan acts on it).\n"
    "4. Propose better names for GLOBAL/data variables the target "
    "function references whose current name is still a default auto-"
    "generated one (e.g. byte_18002C6C3, dword_18002C178, qword_180011A20, "
    "off_BA6E0, unk_140030120) and whose purpose you can determine with "
    "confidence from how the function uses it - a one-shot init/'already "
    "hooked' flag, a saved original function pointer for a hook, a "
    "configuration or state word, a global handle/count, etc. Give one "
    "line per global of the exact form\n"
    "SUGGESTED_GLOBAL_NAME: <its current name or address> -> <new name>\n"
    "using the same identifier rules as SUGGESTED_NAME (e.g. "
    "SUGGESTED_GLOBAL_NAME: byte_18002C6C3 -> g_crypt_hook_installed, or "
    "SUGGESTED_GLOBAL_NAME: qword_18002C170 -> orig_CryptAcquireContextA). "
    "This applies whether the target function was given as pseudocode or "
    "plain disassembly. Only rename a global you are genuinely confident "
    "about from its usage, never from a guess; skip any whose role is "
    "unclear, and never propose a name for a global that already has a "
    "meaningful non-default name.\n"
    "5. If the pseudocode accesses memory through a pointer at multiple "
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
    "Only propose something you are reasonably confident about from the "
    "code itself, never a guess. Steps 2 and 5 (SUGGESTED_SIGNATURE, "
    "SUGGESTED_VAR, SUGGESTED_STRUCT, SUGGESTED_VAR_TYPE) require the "
    "target function's own code to have been given as Hex-Rays "
    "pseudocode - skip them entirely when you were given plain "
    "disassembly instead. Steps 1, 3 and 4 (SUGGESTED_NAME, "
    "SUGGESTED_CALLEE_NAME, SUGGESTED_GLOBAL_NAME) apply to disassembly "
    "too. Otherwise, do not leave a step out merely to save effort or "
    "keep the response short.\n\n"
    "Finally, give your final answer as exactly ONE short sentence (no "
    "more than ~20 words) stating precisely what the target function does "
    "- its core purpose only, not a step-by-step walkthrough. Do not "
    "restate the code line by line, and do not use markdown code fences "
    "or bullet points. This sentence will be written verbatim into an IDA "
    "function comment, so keep it self-contained and free of REQUEST_CODE "
    "lines. Keeping this sentence short does NOT mean doing less of steps "
    "1-5 above - list every SUGGESTED_* line that applies below your "
    "one-sentence answer; only the prose explanation itself needs to be "
    "brief. If asked for more detail in a follow-up, you may then answer "
    "at greater length."
)

CFG_TRACE_SYSTEM_PROMPT = (
    "You are an expert reverse engineer helping recover the original control "
    "flow of a function that has been deliberately obfuscated (e.g. control-"
    "flow flattening around a dispatcher/state variable, opaque predicates "
    "where only one side of a conditional branch is ever really reachable, "
    "or decoy/junk basic blocks that are never actually executed).\n\n"
    "You will be shown ONE basic block at a time - its disassembly, and the "
    "path taken to reach it so far (the earlier blocks and which choice was "
    "made at each branch) - followed by a numbered list of every candidate "
    "successor address IDA found for its final instruction. Your job is to "
    "decide, for THIS block only, which of its candidates are the real "
    "continuation of the original program and which are fake/dead.\n\n"
    "Follow this checklist for every block:\n"
    "1. Address every single candidate listed - do not silently ignore one "
    "just because another looks obviously correct. A block with two "
    "candidates needs two verdicts, not one.\n"
    "2. Reason from concrete state established earlier in the trace (e.g. "
    "the actual value assigned to a dispatcher/state variable a few blocks "
    "back, or a comparison result you can actually work out) - not from "
    "surface-level mnemonic pattern-guessing or assuming a branch is real "
    "just because it 'looks like normal code'.\n"
    "3. Before deciding, ask yourself WHY this condition's outcome is or "
    "isn't fixed. There are two fundamentally different situations, and "
    "you must tell them apart:\n"
    "   a) OPAQUE PREDICATE - the condition's outcome is fully determined "
    "by state the obfuscator itself established earlier in this exact "
    "trace (a dispatcher/state variable set to a known constant a few "
    "blocks back, a comparison against a value you can concretely work "
    "out). Exactly one side is ever really reachable - mark that "
    "candidate REAL and the other(s) DEAD.\n"
    "   b) GENUINE DATA-DEPENDENT BRANCH - the condition depends on "
    "something that is legitimately variable at runtime and was never "
    "fixed by this trace: a function argument, a field read from a "
    "pointer passed in by the caller, a global whose value this "
    "obfuscation does not control, external input, etc. This is NOT "
    "obfuscation - it is ordinary conditional logic from the original "
    "program (e.g. a bounds check, a dispatch on caller-supplied input), "
    "and the original control flow genuinely takes EITHER path depending "
    "on the input. In this situation mark EVERY candidate REAL. The fact "
    "that you personally cannot compute which side is taken from static "
    "analysis alone is not evidence that one side is dead - it is "
    "evidence that both sides are real and input-dependent, which is the "
    "normal, expected shape of most real conditionals. Do not fall back "
    "to UNRESOLVED_TARGET just because the value isn't statically known.\n"
    "4. For an indirect jump through a dispatcher table, work out the "
    "actual state value reaching this point if you can, rather than "
    "assuming any one case is reachable by default. If the dispatch value "
    "itself is genuinely data-dependent (case (b) above) rather than a "
    "fixed obfuscator-controlled state, mark every enumerated case REAL.\n"
    "5. Never invent an address that is not one of the listed candidates. "
    "The only exception is an indirect jump explicitly marked as having no "
    "enumerable candidates - only then may you name a specific address you "
    "can justify from the trace.\n"
    "6. LOOP BACK EDGES: a candidate marked '*** LOOP BACK EDGE ***' below is "
    "an address you have already visited earlier in this exact trace - "
    "looping back to it is completely normal control flow (a for/while "
    "loop, or a flattening dispatcher re-entered every iteration with a "
    "different state value), not evidence that anything is fake. Because "
    "such a dispatcher is revisited many times with DIFFERENT state each "
    "time, a comparison that looks like a fixed, one-sided opaque "
    "predicate on THIS ONE pass through the loop may not actually be "
    "fixed across every iteration - it can take the other path on a "
    "different pass through the very same code. If you cannot be "
    "confident the outcome is the same on every iteration, not just this "
    "one, prefer UNRESOLVED_TARGET over DEAD_TARGET for the affected "
    "candidate(s); only declare a loop-adjacent branch a genuine opaque "
    "predicate when you can see the deciding state is set once before "
    "the loop and never touched again anywhere reachable from it.\n"
    "7. REUSED FLAGS ACROSS MULTIPLE CONDITION CODES: a very common trick "
    "is several Jcc instructions in a row (sometimes separated by "
    "flag-preserving junk like nop, mov reg,reg with the SAME register, "
    "or other instructions that do not touch flags) that all actually "
    "test the SAME earlier comparison/arithmetic result, just with "
    "DIFFERENT condition codes - not independent tests. Before treating a "
    "Jcc as its own fresh test, check whether anything between it and the "
    "previous flag-setting instruction actually modifies flags at all "
    "(most instructions other than cmp/test/add/sub/and/or/xor/shifts/"
    "inc/dec/neg do not). If not, work out that earlier result's actual "
    "sign/zero/value and evaluate the current condition code against THAT "
    "- do not assume each subsequent branch is an independent, equally "
    "'genuinely data-dependent' test just because an earlier one in the "
    "same chain was. It is entirely possible for the first branch in such "
    "a chain to be genuinely data-dependent (e.g. testing one bit of a "
    "caller-supplied value) while a LATER branch in the very same chain is "
    "a fully determined opaque predicate given which side of the first "
    "branch was taken - e.g. if the first branch only proceeds past it "
    "when a value ANDed with a single-bit mask was nonzero, that value on "
    "this path is now KNOWN to equal exactly that mask, not merely "
    "'some unknown nonzero value' - and a later signed/unsigned comparison "
    "against that same (now effectively fixed) result can be a genuine "
    "opaque predicate even though the original bit test was not.\n\n"
    "Reply using exactly one line per candidate, in one of these three "
    "forms:\n"
    "REAL_TARGET: <address> - <short reason>\n"
    "DEAD_TARGET: <address> - <short reason>\n"
    "UNRESOLVED_TARGET: <address or \"indirect\"> - <short reason>\n"
    "Reserve UNRESOLVED_TARGET for genuine ambiguity you cannot resolve "
    "either way - e.g. an indirect jump with no enumerable candidates at "
    "all, or evidence that is actually contradictory - not for an "
    "ordinary data-dependent branch (see 3b above, which is REAL on every "
    "side, not UNRESOLVED) and not merely because a value isn't a fixed "
    "compile-time constant. Use plain addresses (e.g. 0x1400012A0), "
    "matching one of the candidates shown to you. Do not include any "
    "other text in your reply.\n\n"
    "Worked example 1 (opaque predicate) - a block ending in a conditional "
    "jump with candidates 0x140001050 (jump taken) and 0x140001058 "
    "(fallthrough), where the trace shows a state variable was just set to "
    "3 and this block compares it against 3:\n"
    "REAL_TARGET: 0x140001050 - state variable equals 3, so this comparison "
    "is always true and the jump is always taken\n"
    "DEAD_TARGET: 0x140001058 - fallthrough is never reached since the "
    "comparison above always succeeds\n\n"
    "Worked example 2 (genuine data-dependent branch, NOT obfuscation) - a "
    "block computes eax from dword ptr [rcx+18h] (a field of the struct "
    "pointer passed into this function as an argument), then does\n"
    "cmp eax, 4Dh\n"
    "ja short loc_140A7D045\n"
    "with candidates loc_140A7D045 (jump taken) and the fallthrough block. "
    "Nothing earlier in the trace fixes [rcx+18h] to a known constant - it "
    "is caller-supplied input, so this is an ordinary bounds/range check "
    "from the original program, not an opaque predicate:\n"
    "REAL_TARGET: <fallthrough address> - depends on the caller-supplied "
    "value at [rcx+18h]; this is a genuine range check, not a fixed "
    "obfuscator predicate, so both outcomes are real\n"
    "REAL_TARGET: <loc_140A7D045 address> - same reasoning; taken when the "
    "caller-supplied value is out of range\n\n"
    "Worked example 3 (loop back edge, NOT a one-sided opaque predicate) - a "
    "dispatcher block's indirect jump target depends on a state variable, and "
    "one enumerated case is marked '*** LOOP BACK EDGE: this address is an "
    "earlier block in this same trace ***'. The state variable currently "
    "selects only one case, but this dispatcher is reached again on every "
    "loop iteration with a different state value set by whichever block ran "
    "before it - so the matching case is real for THIS pass, but the other "
    "cases are not necessarily dead, since they are the real targets on "
    "OTHER passes through this same dispatcher:\n"
    "REAL_TARGET: <matching case address> - state variable currently selects "
    "this case\n"
    "UNRESOLVED_TARGET: <another case address> - this dispatcher is a loop "
    "back-edge target, revisited every iteration with different state; "
    "cannot confirm this case is unreachable on every pass, only on this one\n\n"
    "WRONG example (do not do this) - replying with only\n"
    "REAL_TARGET: 0x140001050 - looks like the normal path\n"
    "and leaving 0x140001058 unaddressed. Every candidate must get its own "
    "verdict line, even when you are confident about the other one. It is "
    "equally wrong to mark a genuine data-dependent branch (worked example "
    "2) as UNRESOLVED just because you cannot compute the runtime value - "
    "that pattern (a comparison against something read from an argument, "
    "with no earlier fixed state controlling it) should be recognized as "
    "case 3b and both sides marked REAL."
)

DEFAULT_CONFIG = {
    "server_urls": ["http://127.0.0.1:8080"],
    "server_names": {},
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
    "max_trace_blocks": 200,
    "cfg_trace_color_real": 0x99FF99,
    "cfg_trace_color_dead": 0x9999FF,
    "cfg_trace_color_unresolved": 0x99E0FF,
    "cfg_trace_system_prompt": CFG_TRACE_SYSTEM_PROMPT,
    "cfg_trace_use_symbolic": True,
    # Opt-in, experimental: statically enumerate the targets of a computed
    # ARM64 table dispatch (BR/BLR through *(base + index*stride + field))
    # by walking the table in the loaded image. Off by default because the
    # index range can't be bounded statically, so it walks until the first
    # entry that doesn't point at loaded executable code - which may under-
    # or over-shoot. Every enumerated target is still shown for review
    # before anything is written/patched. See _try_enumerate_arm64_table_dispatch.
    "cfg_trace_enumerate_computed_jumps": False,
    # sha256 of DEFAULT_SYSTEM_PROMPT / CFG_TRACE_SYSTEM_PROMPT as they were
    # when this config was last saved (set in save()). Lets a later load
    # tell "the stored prompt was that version's UNMODIFIED default" (hash
    # matches -> silently adopt the new default) apart from "the user
    # customized it" (hash differs -> keep). Empty for configs written
    # before this mechanism existed - those fall back to the frozen
    # historical registry below. See PluginConfig._upgrade_stale_default_prompts.
    "system_prompt_default_hash": "",
    "cfg_trace_prompt_default_hash": "",
}


def _prompt_hash(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


# sha256 of EVERY default prompt this plugin has ever shipped (extracted
# from git history). Used only to recognize a stored prompt in a config
# saved BEFORE the save-time hash above existed as a still-unmodified
# default of its era, so it can be auto-upgraded to the current default
# without clobbering a genuine user customization (whose hash is in
# neither this set nor the config's own stored default hash). This set is
# FROZEN: every config written from now on carries its own default hash,
# so it never needs another entry for future prompt revisions.
_KNOWN_DEFAULT_SYSTEM_PROMPT_HASHES = frozenset({
    "f551f29c65a516563273ac5bf40f3a44e3b22939cde5b5e85ff6ee5a0e67545b",
    "879f0df0bf6fe896ccf11daca50ffa1a61871bfe954cba0b7bf5f3431aa78b69",
    "c30b07204c8fbf1a8de9546d4a9a240e6971e409997c2f928733a9e0539ef4f0",
    "d3ee19b88ed3c98716c96f6d195761650603cb11ce9300e3e932596646da137e",
    "ca1e66af4cc2d8b18cf078ddddf865a3da2d950e6f9865d5ebe3b56d2592ca32",
    "d6bd189f212df5a6f262267c53897378fec8020626eab33f98b876eb463009ac",
    "b5a977a800269783751a13857451ec21e64089f04a986e03f1bc147f1fb3c363",
    "53621569a435f8a55340526a7efd1781591637dfeb23023bc6a64f773574fa40",
})
_KNOWN_CFG_TRACE_PROMPT_HASHES = frozenset({
    "f17ac4a347ed00271730bf66af960f656399c52b5366dd41f6eb09c4c4611f74",
    "647d49a3f20832c369f164617e20e906852c172ef7b7662419fb52a494576f05",
    "48c637e12b804769465a1e8f3f305e9144be722fcdbd502308806ffe2ceba94c",
})

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
_SUGGESTED_GLOBAL_NAME_RE = re.compile(
    r"(?im)^\s*SUGGESTED_GLOBAL_NAME:\s*(.+?)\s*->\s*([A-Za-z_]\w*)\s*$"
)
_SUGGESTED_STRUCT_RE = re.compile(r"(?im)^\s*SUGGESTED_STRUCT:\s*(.+?)\s*$")
_SUGGESTED_VAR_TYPE_RE = re.compile(r"(?im)^\s*SUGGESTED_VAR_TYPE:\s*([A-Za-z_]\w*)\s+(.+?)\s*$")
# SUGGESTED_REANALYZE: <callee name or address> [- <why>] - the model
# believes an already-named callee's current name is likely wrong/stale
# and wants it re-examined. Only acted on by the recursive scan (it queues
# that callee for a fresh analysis, allowing its name to be replaced);
# harmless/ignored elsewhere. Target is a single token; optional reason
# after a dash.
_SUGGESTED_REANALYZE_RE = re.compile(r"(?im)^\s*SUGGESTED_REANALYZE:\s*(\S+)(?:\s*-\s*(.*?))?\s*$")
_VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_AUTO_NAME_RE = re.compile(r"^(sub|loc|nullsub|j_sub|j_nullsub)_[0-9A-Fa-f]+$")

_REAL_TARGET_RE = re.compile(r"(?im)^\s*REAL_TARGET:\s*(\S+)(?:\s*-\s*(.*?))?\s*$")
_DEAD_TARGET_RE = re.compile(r"(?im)^\s*DEAD_TARGET:\s*(\S+)(?:\s*-\s*(.*?))?\s*$")
_UNRESOLVED_TARGET_RE = re.compile(r"(?im)^\s*UNRESOLVED_TARGET:\s*(\S+)(?:\s*-\s*(.*?))?\s*$")


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


def _is_arm64_target():
    """True when the current database's target architecture is AArch64 -
    gates every ARM64-specific encoding/pattern in this file (fixed-width
    branch/NOP instructions, the fixed-pointer-load indirect jump match
    in _resolve_indirect_jump_successors) so none of it is ever
    mistakenly applied to AArch32 (which shares the "ARM" procname but
    has different, and for Thumb variable-width, encodings this hasn't
    been taught)."""
    procname = (get_procname() or "").upper()
    if "ARM" not in procname:
        return False
    try:
        if ida_ida is not None:
            return bool(ida_ida.inf_is_64bit())
    except Exception:
        pass
    try:
        return bool(idc.get_inf_attr(idc.INF_LFLAGS) & idc.LFLG_64BIT)
    except Exception:
        return False


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


def gather_callee_funcs(func, max_callees, include=None):
    """Direct callees of func, as func_t objects, in first-seen order.
    If `include` is given, only callees for which include(callee) returns
    True are kept, and the max_callees cap applies to that filtered set
    (so e.g. the recursive walk can ask for "up to N still-undiscovered
    callees" rather than N callees of which some are skipped)."""
    if max_callees <= 0:
        return []
    result = []
    seen = set()
    for ea in idautils.FuncItems(func.start_ea):
        for ref in idautils.CodeRefsFrom(ea, 0):
            callee = ida_funcs.get_func(ref)
            if callee and callee.start_ea == ref and ref != func.start_ea and ref not in seen:
                seen.add(ref)
                if include is None or include(callee):
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


def _resolve_name_or_address(query):
    """Resolve a model-supplied identifier (a symbol name or a hex/decimal
    address) to a linear address, or idaapi.BADADDR if it resolves to
    neither. Shared by resolve_function_query and resolve_global_query."""
    query = (query or "").strip().strip("`'\"")
    if not query:
        return idaapi.BADADDR
    ea = idc.get_name_ea_simple(query)
    if ea == idaapi.BADADDR:
        for base in (0, 16):
            try:
                ea = int(query, base)
                break
            except ValueError:
                ea = idaapi.BADADDR
    return ea


def resolve_function_query(query):
    """Resolve a model-supplied identifier (name or address) to a func_t."""
    ea = _resolve_name_or_address(query)
    if ea == idaapi.BADADDR:
        return None
    return ida_funcs.get_func(ea)


def resolve_global_query(query):
    """Resolve a model-supplied identifier for a GLOBAL/data variable (e.g.
    byte_18002C6C3, dword_18002C178, or a raw address) to its linear
    address, or idaapi.BADADDR if it can't be resolved or the address
    turns out to be a function entry (functions are handled by
    SUGGESTED_CALLEE_NAME instead, never here). The address must be a
    valid, loaded location that actually has a name attached - i.e. a
    real data item the listing refers to - so a bare number that happens
    to parse but points at nothing is rejected rather than blindly
    renamed."""
    ea = _resolve_name_or_address(query)
    if ea == idaapi.BADADDR:
        return idaapi.BADADDR
    try:
        if not ida_bytes.is_loaded(ea):
            return idaapi.BADADDR
    except Exception:
        return idaapi.BADADDR
    func = ida_funcs.get_func(ea)
    if func is not None and func.start_ea == ea:
        return idaapi.BADADDR  # a function entry - not a data global
    return ea


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


def _dedupe_name(desired, taken):
    """Return `desired` if it isn't in the `taken` set, otherwise the first
    of desired_1, desired_2, ... that is free. Used so two locals the
    model wants to give the same descriptive name to (or a name that
    collides with an existing local it left alone) don't fail the second
    rename outright - IDA forbids two locals sharing a name."""
    if desired not in taken:
        return desired
    i = 1
    while ("%s_%d" % (desired, i)) in taken:
        i += 1
    return "%s_%d" % (desired, i)


def _rename_lvars(func_ea, var_renames):
    """Rename local variables BEFORE any signature/type change is applied
    to the same function: retyping a function (e.g. via idc.SetType) can
    change how Hex-Rays decomposes its local variables, which would make
    the variable names the model actually saw (and is renaming) stale by
    the time we get to them - this was the main suspected cause of renames
    being silently skipped. Also validates the old name actually exists in
    the current decompilation (with a case-insensitive fallback) instead of
    just trying the exact name blind, and logs a clearer reason on failure.

    Returns a dict mapping each ORIGINAL requested old name to the name
    actually applied (which may differ from the requested new name when a
    collision forced a numeric suffix - see _dedupe_name), so the caller
    can translate any follow-on SUGGESTED_VAR_TYPE lookups through it.
    """
    applied = {}
    if not var_renames or ida_hexrays is None:
        return applied
    try:
        hexrays_ready = ida_hexrays.init_hexrays_plugin()
    except Exception:
        hexrays_ready = False
    if not hexrays_ready:
        return applied

    lvar_names = None
    try:
        cfunc = ida_hexrays.decompile(func_ea)
        if cfunc is not None:
            lvar_names = {lv.name for lv in cfunc.get_lvars() if lv.name}
    except Exception:
        lvar_names = None

    # Live set of names currently in use, kept in sync as we rename, so
    # collisions against both untouched locals AND already-applied renames
    # this pass are avoided.
    taken = set(lvar_names) if lvar_names is not None else set()

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

        # De-collide against every name currently in use except this
        # variable's own (renaming a var to its current name is a no-op,
        # not a collision).
        desired = new_name_var
        if desired != actual_old_name:
            desired = _dedupe_name(desired, taken - {actual_old_name})

        try:
            ok = ida_hexrays.rename_lvar(func_ea, actual_old_name, desired)
        except Exception as exc:
            ok = False
            ida_kernwin.msg("[%s] Failed to rename variable '%s': %s\n" % (PLUGIN_NAME, actual_old_name, exc))
        if ok:
            if desired != new_name_var:
                ida_kernwin.msg(
                    "[%s] Renamed variable '%s' -> '%s' (requested '%s' was "
                    "already taken by another local).\n"
                    % (PLUGIN_NAME, actual_old_name, desired, new_name_var)
                )
            taken.discard(actual_old_name)
            taken.add(desired)
            applied[old_name] = desired
        else:
            ida_kernwin.msg(
                "[%s] Failed to rename variable '%s' -> '%s'.\n" % (PLUGIN_NAME, actual_old_name, desired)
            )
    return applied


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
    struct_decl=None, var_types=None, global_renames=None,
):
    # Struct creation must happen first: the signature and/or variable
    # types below may reference the new type by name, and need it to
    # already exist in the local types library to resolve correctly.
    if struct_decl:
        _create_struct_type(struct_decl)

    # Variable renames run BEFORE variable retyping, and both run before
    # the signature change below: renaming a variable is a pure name-swap
    # that doesn't touch how Hex-Rays decomposes locals, but assigning a
    # variable a new TYPE (e.g. a struct pointer from a SUGGESTED_STRUCT)
    # can - same as a signature change can - and that would silently make
    # any old v-name the model was still referring to (including the ones
    # in var_types itself, which are keyed by the ORIGINAL name the model
    # saw) stale for everything applied afterward. Doing renames first,
    # while every original name is still guaranteed to resolve, and only
    # then retyping, avoids that: var_types is translated through
    # var_renames below so it looks up by whatever name is now current.
    applied_renames = _rename_lvars(func_ea, var_renames)
    translated_var_types = var_types
    if var_types and applied_renames:
        # Translate through the names ACTUALLY applied (which may carry a
        # collision-avoidance suffix), not the requested ones, so a type
        # still lands on the right - possibly renamed - variable.
        translated_var_types = [
            (applied_renames.get(var_name, var_name), type_str) for var_name, type_str in var_types
        ]
    _apply_var_types(func_ea, translated_var_types)

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

    if global_renames:
        for global_ea, global_new_name in global_renames:
            try:
                ok = ida_name.set_name(global_ea, global_new_name, ida_name.SN_NOWARN | ida_name.SN_FORCE)
                if not ok:
                    ida_kernwin.msg(
                        "[%s] Failed to rename global variable to '%s'.\n" % (PLUGIN_NAME, global_new_name)
                    )
            except Exception as exc:
                ida_kernwin.msg("[%s] Failed to rename global variable: %s\n" % (PLUGIN_NAME, exc))

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
# CFG trace: basic-block extraction (disassembly-level, no Hex-Rays needed)
# ---------------------------------------------------------------------------

# role: "jump_target" (explicit direct branch target) | "fallthrough" |
# "case" (jump-table case, case_values set) | "resolved" (indirect jump
# with no switch table but exactly one plain code xref IDA's own static
# analysis already resolved) | "unresolved" (indirect jump whose targets
# IDA could not enumerate at all - ea is None in that case).
BlockSuccessor = namedtuple("BlockSuccessor", ["ea", "role", "case_values", "note"])

# kind: "return" | "unconditional_jump" | "conditional_branch" |
# "indirect_jump" | "undecodable" | "truncated" (hit the internal
# max_instrs safety valve without reaching a block-ending instruction) |
# "overlap" (walked into the middle of an instruction some OTHER block in
# this trace already claimed - see _find_overlapping_claim).
BlockInfo = namedtuple(
    "BlockInfo",
    ["start_ea", "end_ea", "text", "kind", "successors", "last_insn_ea", "insn_eas", "sym_insns"],
)

# Internal safety valve against a decode desync (e.g. walking into data)
# turning into a runaway loop. Not user-configurable - hitting it always
# means "something is wrong here", not "this is a big block".
_MAX_BLOCK_INSTRS = 300

# Architectural max x86/x64 instruction length - used by
# _find_overlapping_claim below to bound how far back it needs to look.
_MAX_X86_INSN_LEN = 15


def _find_overlapping_claim(ea, claimed_insns):
    """Returns (claim_start, claim_size) if ea falls STRICTLY inside an
    instruction some OTHER block in this same trace already claimed and
    fixed up in the database (claim_start != ea - an exact match at the
    same start address is a normal, safe merge point, not an overlap).
    x86/x64 instructions are at most 15 bytes, so checking that many
    preceding addresses is sufficient to catch any possible overlap.
    """
    for back in range(1, _MAX_X86_INSN_LEN):
        candidate = ea - back
        size = claimed_insns.get(candidate)
        if size is not None and candidate + size > ea:
            return candidate, size
    return None, None


def _fixup_instruction_boundary(ea, expected_size):
    """Ensures IDA has a correctly-bounded, freshly-analyzed instruction
    item at ea, undefining and recreating it if IDA's existing analysis
    disagrees with what decode_insn (a pure "interpret these raw bytes"
    call that never touches the database) just found. Never changes a
    single byte - only item boundaries/metadata, exactly like manually
    pressing U then C.

    This is not just cosmetic: obfuscated code routinely has real jump
    targets that land mid-instruction relative to IDA's original (wrong)
    linear-sweep disassembly. Besides the disassembly text being
    misleading there, idautils.CodeRefsFrom (used below to resolve branch
    targets) reads cross-references stored in the database, which are
    only populated for a properly analyzed instruction - a bare
    decode_insn() call does not create them.
    """
    try:
        head = ida_bytes.get_item_head(ea)
        flags = ida_bytes.get_full_flags(ea)
        if head == ea and ida_bytes.is_code(flags) and ida_bytes.get_item_size(ea) == expected_size:
            return True
    except Exception:
        pass
    try:
        ida_bytes.del_items(ea, ida_bytes.DELIT_SIMPLE, expected_size)
        return ida_ua.create_insn(ea) > 0
    except Exception:
        return False


def _arm64_insn_writes_reg(insn, reg):
    """Conservative "does this instruction clobber register `reg`" check
    for the backward scan in _try_resolve_arm64_fixed_pointer_jump: true
    if operand 0 or 1 is exactly this register (covers the destination
    of virtually every data-processing/load instruction, plus the second
    destination of a register pair like LDP). May return a false
    positive for a handful of instructions where operand 0 is actually a
    SOURCE (e.g. STR's first operand is the value being stored, not a
    destination) - deliberately left uncorrected, since a false positive
    here only means the scan gives up one instruction too early (safe:
    falls back to the LLM), whereas a false NEGATIVE could let a genuinely
    clobbered register look untouched and produce a wrong resolved
    address - so this only ever errs toward refusing.
    """
    for i in (0, 1):
        try:
            op = insn.ops[i]
        except Exception:
            break
        if op.type == ida_ua.o_void:
            break
        if op.type == ida_ua.o_reg and op.reg == reg:
            return True
    return False


def _try_resolve_arm64_fixed_pointer_jump(term_ea, insn_eas):
    """ARM64-specific fallback for a "BR/BLR Xn" whose value was loaded
    from a FIXED memory address - a computed-address stub/veneer/tail-
    call (common in obfuscated and even ordinary compiler-generated ARM64
    code), not a genuine jump table. Walks this block's OWN instructions
    (never past its start) backward from the branch for:
        <address materialized into Xbase - ADRP+ADD, ADRL, ADR, ...>
        ...(any instructions that don't touch Xbase)...
        LDR       Xn, [Xbase, #imm]
        ...(any instructions that don't touch Xn)...
        BR/BLR    Xn
    Rather than reconstruct the base address from the materializing
    instruction(s) itself - fragile, because IDA displays the common
    ADRP+ADD pair as a single merged "ADRL" line while it is really two
    separate 4-byte instructions, and other sequences exist - this reads
    the FIXED data address IDA's own value tracking already resolved for
    the LDR and recorded as that instruction's single data cross-
    reference (the same resolution that makes IDA display the pointer
    slot's name and the pointed-to symbol next to the LDR/BR), then reads
    the pointer stored there straight from the loaded binary image. The
    load address is a build-time constant, so the stored pointer is a
    single fixed target - exactly what a human reads off the listing.
    Returns a resolved target ea, or None if the pattern isn't matched
    with full confidence (no such LDR, or IDA has no single resolved data
    address for it) - never a guess. Fully wrapped so it can NEVER raise
    into gather_block (which is not itself guarded) - any unexpected error
    is logged and treated as "couldn't resolve", never a crash that would
    abort the whole trace.
    """
    if not _is_arm64_target():
        return None
    try:
        return _try_resolve_arm64_fixed_pointer_jump_impl(term_ea, insn_eas)
    except Exception as exc:
        try:
            ida_kernwin.msg(
                "[%s] ARM64 indirect-jump resolver hit an error at %#010x (%s); "
                "leaving it unresolved.\n" % (PLUGIN_NAME, term_ea, exc)
            )
        except Exception:
            pass
        return None


def _canon_mnem(insn):
    """insn.get_canon_mnem() can return None for some instructions - guard
    it the same way the rest of this file does, so a bare .lower() never
    raises AttributeError (which, uncaught inside gather_block, would abort
    the whole trace)."""
    try:
        return (insn.get_canon_mnem() or "").lower()
    except Exception:
        return ""


def _try_resolve_arm64_fixed_pointer_jump_impl(term_ea, insn_eas):
    term_insn = ida_ua.insn_t()
    if ida_ua.decode_insn(term_insn, term_ea) <= 0:
        return None
    if _canon_mnem(term_insn) not in ("br", "blr"):
        return None
    if term_insn.ops[0].type != ida_ua.o_reg:
        return None
    target_reg = term_insn.ops[0].reg

    try:
        term_idx = list(insn_eas).index(term_ea)
    except ValueError:
        return None

    # Walk back to the LDR that loads the branch register from memory,
    # refusing if anything else writes that register first (which would
    # mean the value going into the branch isn't this load's result).
    ldr_ea = None
    for idx in range(term_idx - 1, -1, -1):
        cand_ea = insn_eas[idx]
        insn = ida_ua.insn_t()
        if ida_ua.decode_insn(insn, cand_ea) <= 0:
            return None
        mnem = _canon_mnem(insn)
        if (
            mnem == "ldr" and insn.ops[0].type == ida_ua.o_reg
            and insn.ops[0].reg == target_reg
        ):
            if insn.ops[1].type not in (ida_ua.o_displ, ida_ua.o_mem, ida_ua.o_phrase):
                return None  # not a memory load - refuse
            ldr_ea = cand_ea
            break
        if _arm64_insn_writes_reg(insn, target_reg):
            return None  # something else set the branch register first
    if ldr_ea is None:
        return None

    # The fixed pointer slot(s) this LDR reads from, taken from IDA's own
    # resolved data cross-references rather than recomputed (which would
    # mean interpreting the ADRP+ADD/ADRL that materialized the base).
    # For each, read the stored pointer and keep it if it lands on a
    # loaded address. In the normal fixed-pointer veneer there is exactly
    # one such slot yielding exactly one target; if the reads disagree
    # (2+ distinct targets) it's ambiguous and we refuse rather than
    # guess. Logged either way so "why didn't my BR get patched" is
    # answerable from the Output window instead of looking like a bug.
    try:
        data_slots = [r for r in idautils.DataRefsFrom(ldr_ea) if r != idaapi.BADADDR]
    except Exception:
        data_slots = []
    targets = set()
    for slot in data_slots:
        try:
            if not ida_bytes.is_loaded(slot):
                continue
            val = ida_bytes.get_qword(slot)
        except Exception:
            continue
        if val and val != idaapi.BADADDR and ida_bytes.is_loaded(val):
            targets.add(val)

    if len(targets) == 1:
        target = targets.pop()
        ida_kernwin.msg(
            "[%s] Resolved indirect BR/BLR at %#010x to a single fixed target "
            "%#010x (via the pointer loaded at %#010x).\n"
            % (PLUGIN_NAME, term_ea, target, ldr_ea)
        )
        return target

    ida_kernwin.msg(
        "[%s] Could not resolve indirect BR/BLR at %#010x: its feeding LDR at "
        "%#010x has %d data reference(s) yielding %d distinct loaded target(s) "
        "- leaving it for the model to decide.\n"
        % (PLUGIN_NAME, term_ea, ldr_ea, len(data_slots), len(targets))
    )
    return None


# Absolute ceiling on how many entries a computed-jump-table walk will
# ever read, independent of the "stop at first non-code entry" rule -
# purely a runaway guard against a mis-extracted base/stride wandering
# through a huge mapped region.
_ARM64_TABLE_ENUM_HARD_CAP = 512


def _arm64_addr_is_code(ea):
    """True if ea is a loaded, 4-aligned address inside an executable
    segment - the acceptance test for a byte pattern to be treated as a
    real AArch64 branch destination (every enumerated table entry must
    pass this, which is what keeps a mis-extracted table from producing
    junk targets: garbage almost never satisfies all three)."""
    try:
        if ea is None or ea == idaapi.BADADDR or (ea & 3) != 0:
            return False
        if not ida_bytes.is_loaded(ea):
            return False
        seg = ida_segment.getseg(ea)
        return seg is not None and bool(seg.perm & ida_segment.SEGPERM_EXEC)
    except Exception:
        return False


def _arm64_trace_reg_back(insn_eas, before_idx, reg, want):
    """Walk insn_eas backward from before_idx for the instruction that
    writes `reg`, and if it matches `want` (a predicate over the decoded
    insn) return (idx, insn); otherwise return (None, None) as soon as
    `reg` is written by something that does NOT match (so we never trace
    THROUGH an unrelated clobber). Pure read-only decode."""
    for idx in range(before_idx, -1, -1):
        insn = ida_ua.insn_t()
        if ida_ua.decode_insn(insn, insn_eas[idx]) <= 0:
            return None, None
        if insn.ops[0].type == ida_ua.o_reg and insn.ops[0].reg == reg:
            if want(insn):
                return idx, insn
            return None, None  # reg written by something unexpected - stop
    return None, None


def _arm64_reg_immediate(insn_eas, before_idx, reg):
    """The immediate a MOV/MOVZ set into `reg` just before before_idx, via
    IDA's resolved operand value (handles MOV #imm and MOVZ/MOVN forms),
    or None if the most recent writer of reg isn't a simple immediate
    move."""
    idx, insn = _arm64_trace_reg_back(
        insn_eas, before_idx, reg,
        lambda ins: _canon_mnem(ins) in ("mov", "movz", "movn") and ins.ops[1].type == ida_ua.o_imm,
    )
    if insn is None:
        return None
    try:
        val = idc.get_operand_value(insn_eas[idx], 1)
    except Exception:
        return None
    return val if val is not None and val >= 0 else None


def _arm64_reg_fixed_address(insn_eas, before_idx, reg):
    """The fixed address materialized into `reg` (ADR/ADRL, or the
    ADRP+ADD pair IDA shows merged as ADRL), taken from IDA's own resolved
    data cross-reference on the materializing instruction rather than
    recomputed - same robustness argument as _try_resolve_arm64_fixed_
    pointer_jump. Scans back over the (up to two) instructions that build
    the address; returns the single data reference found, or None."""
    for idx in range(before_idx, -1, -1):
        insn = ida_ua.insn_t()
        if ida_ua.decode_insn(insn, insn_eas[idx]) <= 0:
            return None
        if insn.ops[0].type != ida_ua.o_reg or insn.ops[0].reg != reg:
            continue
        mnem = _canon_mnem(insn)
        try:
            refs = [r for r in idautils.DataRefsFrom(insn_eas[idx]) if r != idaapi.BADADDR]
        except Exception:
            refs = []
        if len(refs) == 1:
            return refs[0]
        if mnem in ("adr", "adrl"):
            try:
                val = idc.get_operand_value(insn_eas[idx], 1)
            except Exception:
                val = idaapi.BADADDR
            return val if val != idaapi.BADADDR else None
        # ADRP, or the ADD low-half of an ADRP+ADD pair: no single data
        # ref on THIS line - keep scanning back to the other half.
        if mnem in ("adrp", "add"):
            continue
        return None  # written by something that isn't address materialization
    return None


def _try_enumerate_arm64_table_dispatch(term_ea, insn_eas):
    """Opt-in (see config.cfg_trace_enumerate_computed_jumps) enumeration
    of an AArch64 computed table dispatch:
        <materialize table base into Xbase>
        MOV   Xstride, #<stride>
        MADD  Xt, Xindex, Xstride, Xbase      ; Xt = index*stride + base
        [ADD  Xt, Xt, #<field>]               ; optional field offset
        LDR   Xd, [Xt{, #<ldr_disp>}]
        BR/BLR Xd
    Reads the pointer at base + i*stride + field for i = 0,1,2,... from
    the loaded image, keeping each that lands on executable code, and
    stopping at the first that doesn't (bounded by _ARM64_TABLE_ENUM_HARD_
    CAP). Returns a sorted list of distinct target eas (possibly empty),
    or None if the block isn't this pattern at all. Never guesses: a mis-
    extracted base/stride simply yields entries that fail _arm64_addr_is_
    code and the walk ends immediately.
    """
    if not _is_arm64_target():
        return None
    term_insn = ida_ua.insn_t()
    if ida_ua.decode_insn(term_insn, term_ea) <= 0:
        return None
    if _canon_mnem(term_insn) not in ("br", "blr") or term_insn.ops[0].type != ida_ua.o_reg:
        return None
    branch_reg = term_insn.ops[0].reg
    try:
        term_idx = list(insn_eas).index(term_ea)
    except ValueError:
        return None

    # LDR Xd, [Xt {, #disp}] feeding the branch register.
    ldr_idx, ldr_insn = _arm64_trace_reg_back(
        insn_eas, term_idx - 1, branch_reg,
        lambda ins: _canon_mnem(ins) == "ldr" and ins.ops[1].type in (ida_ua.o_displ, ida_ua.o_phrase),
    )
    if ldr_insn is None or ldr_insn.ops[1].type == ida_ua.o_void:
        return None
    addr_reg = ldr_insn.ops[1].reg
    field = ldr_insn.ops[1].addr if ldr_insn.ops[1].type == ida_ua.o_displ else 0

    # Follow addr_reg back through any immediate ADDs (each contributing to
    # the field offset) until the MADD that formed index*stride + base.
    cur = addr_reg
    cur_idx = ldr_idx - 1
    madd_insn = None
    madd_idx = None
    for _ in range(8):  # small bound: real dispatch prologues are short
        idx, insn = _arm64_trace_reg_back(
            insn_eas, cur_idx, cur,
            lambda ins: _canon_mnem(ins) in ("madd", "add"),
        )
        if insn is None:
            return None
        if _canon_mnem(insn) == "madd":
            madd_insn, madd_idx = insn, idx
            break
        # ADD Xcur, Xsrc, #imm - fold the immediate into the field offset
        # and keep chasing Xsrc; anything else (register add) isn't this
        # pattern.
        if insn.ops[1].type == ida_ua.o_reg and insn.ops[2].type == ida_ua.o_imm:
            field += idc.get_operand_value(insn_eas[idx], 2)
            cur = insn.ops[1].reg
            cur_idx = idx - 1
            continue
        return None
    if madd_insn is None:
        return None

    # MADD Xd, Xn, Xm, Xa -> Xd = Xn*Xm + Xa. One of Xn/Xm is the constant
    # stride, the other the runtime index; Xa is the table base register.
    if (
        madd_insn.ops[1].type != ida_ua.o_reg or madd_insn.ops[2].type != ida_ua.o_reg
        or madd_insn.ops[3].type != ida_ua.o_reg
    ):
        return None
    reg_n, reg_m, reg_a = madd_insn.ops[1].reg, madd_insn.ops[2].reg, madd_insn.ops[3].reg

    stride = _arm64_reg_immediate(insn_eas, madd_idx - 1, reg_m)
    if stride is None:
        stride = _arm64_reg_immediate(insn_eas, madd_idx - 1, reg_n)
    if not stride or stride <= 0:
        return None

    table_base = _arm64_reg_fixed_address(insn_eas, madd_idx - 1, reg_a)
    if table_base is None or table_base == idaapi.BADADDR:
        return None

    targets = []
    seen = set()
    for i in range(_ARM64_TABLE_ENUM_HARD_CAP):
        slot = table_base + i * stride + field
        try:
            if not ida_bytes.is_loaded(slot):
                break
            ptr = ida_bytes.get_qword(slot)
        except Exception:
            break
        if not _arm64_addr_is_code(ptr):
            break  # first non-code entry ends the table
        if ptr not in seen:
            seen.add(ptr)
            targets.append(ptr)

    targets.sort()
    ida_kernwin.msg(
        "[%s] Computed jump table at %#010x: base=%#010x stride=%#x field=%#x "
        "-> %d target(s) enumerated%s.\n"
        % (PLUGIN_NAME, term_ea, table_base, stride, field, len(targets),
           " (hit hard cap)" if len(targets) == _ARM64_TABLE_ENUM_HARD_CAP else "")
    )
    return targets


def _resolve_indirect_jump_successors(ea, insn_eas=()):
    """Best-effort enumeration of an indirect jump/call's targets: prefers
    IDA's switch/jump-table analysis (a genuine multi-case dispatch);
    when that finds nothing, tries the ARM64 fixed-pointer-load pattern
    match (_try_resolve_arm64_fixed_pointer_jump - computes the answer
    directly, independent of IDA's own xref/comment analysis); and if
    that doesn't apply either (wrong architecture, or the pattern doesn't
    match), falls back to IDA's plain code cross-references IF there's
    exactly one - e.g. an x86 computed jump through a fixed pointer load,
    which is not a switch table but which IDA's own analysis frequently
    still resolves and xrefs on its own. Two or more code xrefs with no
    switch info is left unresolved rather than guessed at - that shape
    means something this hasn't specifically modeled (e.g. a real multi-
    target dispatch IDA didn't recognize as a formal switch), not a
    single safe answer. Returns a list of BlockSuccessor. If nothing
    could be resolved, returns a single role="unresolved" successor
    (ea=None) - the block itself is still real, only its final jump's
    destination is ambiguous.
    """
    try:
        si = ida_nalt.get_switch_info(ea)
    except Exception:
        si = None
    if si is None:
        arm_target = _try_resolve_arm64_fixed_pointer_jump(ea, list(insn_eas))
        if arm_target is not None:
            return [BlockSuccessor(
                ea=arm_target, role="resolved", case_values=None,
                note="Indirect jump; resolved by reading the pointer stored "
                     "at a fixed address computed from an ADR/ADRL + LDR "
                     "sequence feeding this branch (not a switch table).",
            )]
        try:
            xref_targets = sorted(set(idautils.CodeRefsFrom(ea, 0)))
        except Exception:
            xref_targets = []
        if len(xref_targets) == 1:
            return [BlockSuccessor(
                ea=xref_targets[0], role="resolved", case_values=None,
                note="Indirect jump; IDA statically resolved a single "
                     "target via its own analysis (not a switch table).",
            )]
        return [BlockSuccessor(
            ea=None, role="unresolved", case_values=None,
            note="Indirect jump; IDA did not resolve a jump table here.",
        )]
    try:
        results = ida_xref.calc_switch_cases(ea, si)
    except Exception:
        results = None
    if results is None or not len(results.targets):
        return [BlockSuccessor(
            ea=None, role="unresolved", case_values=None,
            note="Indirect jump; switch info present but no cases resolved.",
        )]
    successors = []
    for idx in range(len(results.targets)):
        target = results.targets[idx]
        case_values = list(results.cases[idx]) if idx < len(results.cases) else []
        successors.append(BlockSuccessor(
            ea=target, role="case", case_values=case_values,
            note="case %s" % ", ".join(str(v) for v in case_values) if case_values else None,
        ))
    return successors


# ---------------------------------------------------------------------------
# CFG trace: lightweight symbolic/constant-propagation engine
#
# Resolves the common "opaque predicate" and "flattening dispatcher" cases
# WITHOUT an LLM round-trip (this is what makes tracing fast), and also
# recognizes genuine data-dependent branches (an ordinary conditional on
# caller-supplied input, not obfuscation) so those get marked real on
# every side automatically too. Falls back to the LLM (CfgTraceRunner's
# existing path, unchanged) whenever it cannot confidently resolve
# something - this engine's failure mode is always "don't know", never a
# silently wrong verdict. See the module-level docstring in the class
# below for the full design reasoning.
# ---------------------------------------------------------------------------

class _ExternalTaint(object):
    """Sentinel: this value is known to derive from genuine runtime/
    caller-controlled data (e.g. a memory read from a writable segment,
    or through a pointer whose own value is untracked) - NOT the same as
    plain unknown (None), which means "hit something we don't model" and
    must always fall back to the LLM. Conflating the two would risk
    silently mismarking a real opaque predicate the engine just failed
    to model as if it were an ordinary data-dependent branch.
    """
    def __repr__(self):
        return "EXTERNAL"


EXTERNAL = _ExternalTaint()

_SYM_WIDTH_MASK = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF, 8: 0xFFFFFFFFFFFFFFFF}

# kind: "reg" | "imm" | "mem_direct" | "mem_simple" | "unsupported"
#   reg:         family=GPR family key (see _family_of), value=None, width=bytes
#   imm:         family=None, value=int,                            width=bytes
#   mem_direct:  family=None, value=absolute address,                width=bytes
#   mem_simple:  family=base GPR family, value=raw displacement,     width=bytes
#     (round-trip write/read-back tracked in SymState.mem, keyed by
#     (family, disp) - see SymState docstring)
#   mem_indexed: family=None, value=raw displacement, terms=((family,scale), ...)
#     (1 or 2 (family, scale) pairs - scale in {1,2,4,8}; base+index*scale
#     addressing, e.g. [rcx+rdx*4] or [rax*8+table]. Read-only: the
#     effective address is recomputed fresh from current register values
#     every time, never round-trip-tracked in SymState.mem, so there is
#     no staleness/invalidation concern for this kind at all)
#   unsupported: family=None, value=None, width=None, terms=None
SymOperand = namedtuple("SymOperand", ["kind", "family", "value", "width", "terms"], defaults=(None,))
SymInsn = namedtuple("SymInsn", ["mnem", "operands"])
_UNSUPPORTED_OPERAND = SymOperand(kind="unsupported", family=None, value=None, width=None)


def _sym_to_signed(val, width):
    bits = width * 8
    if val >= (1 << (bits - 1)):
        return val - (1 << bits)
    return val


def _sym_parity_even(val):
    """PF reflects the parity of only the LOW BYTE of the result,
    regardless of the operand's actual width (x86 architectural rule -
    unlike ZF/SF/etc., which use the full result). Always derivable from
    a concrete value alone, the same way ZF/SF already are - no flag
    guarantee needed, see _sym_zero_based's jp/jpe/jnp/jpo entries.
    """
    return bin(val & 0xFF).count("1") % 2 == 0


class SymState(object):
    """A snapshot of what the trace concretely knows about registers and
    a bounded set of base+displacement memory slots at one point in the
    walk. Every write to a register purges memory slots keyed by that
    register as a base - this is what makes frame/stack-relative slot
    tracking safe without ever computing an absolute stack address: two
    operands with the same (base, disp) key are only trusted to be the
    same slot as long as the base hasn't been reassigned in between.
    """
    __slots__ = ("regs", "mem", "last_cmp", "last_flags", "last_branch")

    def __init__(self):
        self.regs = {}          # family -> int | EXTERNAL
        self.mem = {}           # (base_family, raw_disp) -> (int|EXTERNAL, width)
        self.last_cmp = None    # (left, right, width) from cmp/sub - left/right may be EXTERNAL
        self.last_branch = None  # (condition_class, taken_bool, last_cmp_snapshot,
                                  # last_flags_snapshot) - set ONLY by
                                  # _compute_edge_refinements, for the specific edge
                                  # of a just-resolved conditional branch. Records
                                  # that THIS branch's own condition tautologically
                                  # evaluated to taken_bool to get here, alongside a
                                  # snapshot of last_cmp/last_flags AT THAT POINT -
                                  # sym_evaluate_branch_condition compares the LIVE
                                  # last_cmp/last_flags against this snapshot before
                                  # trusting it, so the tag is silently ignored (not
                                  # proactively cleared) the moment anything changes
                                  # the flags. See _compute_edge_refinements and
                                  # _sym_retest_from_last_branch for why: testing the
                                  # SAME flag twice with opposite senses through a
                                  # no-op in between (e.g. `js target; nop; jns
                                  # target`) is a common obfuscation trick that is
                                  # tautologically always-taken one way or the other,
                                  # yet each half looks independently
                                  # "data-dependent"/"unknown" without this context.
        self.last_flags = None  # (value, width, cf_zero, of_zero, refine_info) from test/
                                 # and/or/xor/other arith result. cf_zero/of_zero are True
                                 # only when THAT SPECIFIC flag is architecturally
                                 # guaranteed zero right now - and/or/xor/test guarantee
                                 # BOTH (Intel SDM: "the OF and CF flags are cleared"),
                                 # letting _sym_cmp_based's zero_fallback resolve ja/jae/jb/
                                 # jbe (need cf_zero) and jg/jge/jl/jle (need of_zero) from a
                                 # single result value the same way _sym_zero_based already
                                 # does for jz/jnz/js/jns. add/sub/neg/shifts get neither
                                 # (their CF/OF genuinely depend on the operands). inc/dec
                                 # are the one case that's split: real x86 does NOT touch CF
                                 # for inc/dec at all (a well-known quirk - only SF/ZF/OF/AF/
                                 # PF change), so their handler explicitly CARRIES FORWARD
                                 # whatever cf_zero already was rather than clearing it, while
                                 # still setting of_zero=False (inc/dec's OF genuinely can be
                                 # set, at the signed range boundary).
                                 # refine_info is (family, mask) only when this came from
                                 # `and reg, single_bit_mask` on a genuinely EXTERNAL reg -
                                 # see _compute_and_mask_refinement: even though the value
                                 # itself is unknown, AND-with-one-bit can only ever produce
                                 # exactly 0 or exactly that mask, so once a zero/nonzero
                                 # test resolves which side of a branch we're on, THAT
                                 # register is provably exactly one of those two concrete
                                 # values on that specific edge - letting anything further
                                 # down that one path resolve precisely instead of forever
                                 # treating the register as merely "some unknown value".

    def copy(self):
        s = SymState()
        s.regs = dict(self.regs)
        s.mem = dict(self.mem)
        s.last_cmp = self.last_cmp
        s.last_flags = self.last_flags
        s.last_branch = self.last_branch
        return s

    def reset(self):
        # last_branch is cleared here too, not just last_cmp/last_flags:
        # if it weren't, a coincidental (None, None) snapshot match after
        # a reset could make a now-meaningless last_branch tag look
        # "still valid" to _sym_retest_from_last_branch's comparison.
        self.last_branch = None
        self.regs = {}
        self.mem = {}
        self.last_cmp = None
        self.last_flags = None

    def write_reg(self, family, value):
        if family is None:
            return
        self.mem = {k: v for k, v in self.mem.items() if k[0] != family}
        if value is None:
            self.regs.pop(family, None)
        elif value is EXTERNAL:
            self.regs[family] = EXTERNAL
        else:
            self.regs[family] = value & _SYM_WIDTH_MASK[8]

    def read_reg(self, family, width):
        if family is None:
            return None
        v = self.regs.get(family)
        if v is None or v is EXTERNAL:
            return v
        return v & _SYM_WIDTH_MASK[width]


def _sym_read_static_memory(addr, width):
    """Reads a concrete value straight from the loaded binary image, but
    ONLY when the containing segment is read-only - a mutable global's
    static initializer is not a trustworthy stand-in for its value at an
    arbitrary point during execution. A writable/unknown segment yields
    EXTERNAL (genuinely runtime-determined), not a wrong guess.
    """
    try:
        seg = ida_segment.getseg(addr)
    except Exception:
        seg = None
    if seg is None or not seg.perm:
        return None
    writable = bool(seg.perm & ida_segment.SEGPERM_WRITE)
    try:
        if width == 1:
            value = ida_bytes.get_byte(addr)
        elif width == 2:
            value = ida_bytes.get_word(addr)
        elif width == 4:
            value = ida_bytes.get_wide_dword(addr)
        elif width == 8:
            value = ida_bytes.get_qword(addr)
        else:
            return None
    except Exception:
        return None
    return EXTERNAL if writable else value


def _sym_resolve_indexed_address(state, op):
    """Computes the effective address of a mem_indexed operand from
    CURRENT register values - never cached/round-trip-tracked, so there
    is nothing to invalidate on a later register write (unlike
    mem_simple's SymState.mem slots): this just recomputes fresh every
    time it's asked. Returns an int, EXTERNAL, or None (unsupported),
    with None taking precedence over EXTERNAL - see _sym_combine.
    """
    addr = op.value
    saw_external = False
    for family, scale in op.terms:
        v = state.regs.get(family)
        if v is None:
            return None
        if v is EXTERNAL:
            saw_external = True
        else:
            addr += v * scale
    if saw_external:
        return EXTERNAL
    return addr & _SYM_WIDTH_MASK[8]


def _sym_resolve_operand(state, op):
    if op.kind == "imm":
        return op.value & _SYM_WIDTH_MASK[op.width]
    if op.kind == "reg":
        return state.read_reg(op.family, op.width)
    if op.kind == "mem_direct":
        return _sym_read_static_memory(op.value, op.width)
    if op.kind == "mem_simple":
        slot = state.mem.get((op.family, op.value))
        if slot is not None and slot[1] == op.width:
            return slot[0]
        base_val = state.regs.get(op.family)
        if base_val is None:
            return None
        if base_val is EXTERNAL:
            return EXTERNAL
        addr = (base_val + op.value) & _SYM_WIDTH_MASK[8]
        return _sym_read_static_memory(addr, op.width)
    if op.kind == "mem_indexed":
        addr = _sym_resolve_indexed_address(state, op)
        if addr is None or addr is EXTERNAL:
            return addr
        return _sym_read_static_memory(addr, op.width)
    return None


def _sym_combine(a, b):
    if a is None or b is None:
        return "unsupported"
    if a is EXTERNAL or b is EXTERNAL:
        return "external"
    return "concrete"


def _sym_write_dest(state, dst_op, value):
    if dst_op.kind == "mem_simple":
        key = (dst_op.family, dst_op.value)
        if value is None:
            state.mem.pop(key, None)
        else:
            state.mem[key] = (value, dst_op.width)
        return
    if dst_op.kind != "reg":
        return  # mem_direct / mem_indexed / unsupported destinations are not tracked
    if dst_op.width not in (4, 8):
        # 8/16-bit partial-register writes don't clear the rest of the
        # register on real hardware and we don't model that precisely -
        # conservatively taint the whole family rather than guess.
        state.write_reg(dst_op.family, None)
        return
    if value is None:
        state.write_reg(dst_op.family, None)
    elif value is EXTERNAL:
        state.write_reg(dst_op.family, EXTERNAL)
    else:
        # x86-64 rule: a 32-bit write zero-extends into the full 64-bit
        # register; a 64-bit write is already the whole register.
        state.write_reg(dst_op.family, value & _SYM_WIDTH_MASK[dst_op.width])


def _sym_h_mov(state, ops):
    dst, src = ops[0], ops[1]
    _sym_write_dest(state, dst, _sym_resolve_operand(state, src))


def _sym_h_movzx(state, ops):
    dst, src = ops[0], ops[1]
    _sym_write_dest(state, dst, _sym_resolve_operand(state, src))


def _sym_h_movsx(state, ops):
    dst, src = ops[0], ops[1]
    val = _sym_resolve_operand(state, src)
    if isinstance(val, int):
        val = _sym_to_signed(val, src.width) & _SYM_WIDTH_MASK[8]
    _sym_write_dest(state, dst, val)


def _sym_h_lea(state, ops):
    dst, src = ops[0], ops[1]
    if src.kind == "mem_direct":
        _sym_write_dest(state, dst, src.value & _SYM_WIDTH_MASK[dst.width])
        return
    if src.kind == "mem_simple":
        base_val = state.regs.get(src.family)
        if base_val is None:
            _sym_write_dest(state, dst, None)
        elif base_val is EXTERNAL:
            _sym_write_dest(state, dst, EXTERNAL)
        else:
            addr = (base_val + src.value) & _SYM_WIDTH_MASK[8]
            _sym_write_dest(state, dst, addr & _SYM_WIDTH_MASK[dst.width])
        return
    if src.kind == "mem_indexed":
        addr = _sym_resolve_indexed_address(state, src)
        if isinstance(addr, int):
            addr &= _SYM_WIDTH_MASK[dst.width]
        _sym_write_dest(state, dst, addr)
        return
    _sym_write_dest(state, dst, None)


def _sym_is_single_bit_mask(value):
    return isinstance(value, int) and value != 0 and (value & (value - 1)) == 0


def _sym_make_alu2(op_name):
    def handler(state, ops):
        dst, src = ops[0], ops[1]
        left = _sym_resolve_operand(state, dst)
        right = _sym_resolve_operand(state, src)
        combo = _sym_combine(left, right)
        if combo == "unsupported":
            _sym_write_dest(state, dst, None)
            state.last_flags = None
            if op_name == "sub":
                state.last_cmp = None
            return
        if combo == "external":
            _sym_write_dest(state, dst, EXTERNAL)
            refine_info = None
            if (
                op_name == "and" and left is EXTERNAL and dst.kind == "reg"
                and src.kind == "imm" and _sym_is_single_bit_mask(src.value)
            ):
                # AND with exactly one bit set can only ever produce 0 or
                # that exact mask - even though the operand itself is
                # unknown, whichever side of a zero/nonzero test we end
                # up on afterward pins this register to one of those two
                # concrete values on that specific path (see
                # _compute_and_mask_refinement).
                refine_info = (dst.family, src.value)
            is_logical = op_name in ("and", "or", "xor")
            state.last_flags = (EXTERNAL, dst.width, is_logical, is_logical, refine_info)
            if op_name == "sub":
                state.last_cmp = (left, right, dst.width)
            return
        width = dst.width
        if op_name == "add":
            result = (left + right) & _SYM_WIDTH_MASK[width]
        elif op_name == "sub":
            result = (left - right) & _SYM_WIDTH_MASK[width]
            state.last_cmp = (left, right, width)
        elif op_name == "and":
            result = left & right
        elif op_name == "or":
            result = left | right
        elif op_name == "xor":
            result = left ^ right
        else:
            raise AssertionError(op_name)
        _sym_write_dest(state, dst, result)
        # and/or/xor architecturally guarantee OF=CF=0 (Intel SDM: "the OF
        # and CF flags are cleared") - add/sub's OF/CF genuinely depend on
        # the operands, so they don't get that guarantee here. No
        # refine_info here - the result is already concrete, nothing to
        # refine (the existing concrete-value handling already resolves
        # this precisely without needing per-edge narrowing).
        is_logical = op_name in ("and", "or", "xor")
        state.last_flags = (result, width, is_logical, is_logical, None)
    return handler


def _sym_h_cmp(state, ops):
    dst, src = ops[0], ops[1]
    left = _sym_resolve_operand(state, dst)
    right = _sym_resolve_operand(state, src)
    combo = _sym_combine(left, right)
    state.last_cmp = None if combo == "unsupported" else (left, right, dst.width)


def _sym_h_test(state, ops):
    dst, src = ops[0], ops[1]
    left = _sym_resolve_operand(state, dst)
    right = _sym_resolve_operand(state, src)
    combo = _sym_combine(left, right)
    if combo == "unsupported":
        state.last_flags = None
    elif combo == "external":
        state.last_flags = (EXTERNAL, dst.width, True, True, None)  # TEST also guarantees OF=CF=0
    else:
        state.last_flags = (left & right, dst.width, True, True, None)


def _sym_make_unary(op_name):
    def handler(state, ops):
        dst = ops[0]
        val = _sym_resolve_operand(state, dst)
        if val is None:
            _sym_write_dest(state, dst, None)
            if op_name != "not":
                state.last_flags = None
            return
        # inc/dec do NOT touch CF at all on real x86 (a well-known quirk -
        # only SF/ZF/OF/AF/PF change) - carry forward whatever cf_zero
        # already was instead of clearing it, so a CF-only condition
        # (jae/jb and aliases) reached through an inc/dec right after an
        # and/or/xor/test that established CF=0 still resolves, rather
        # than losing that fact the moment an inc/dec runs. neg's CF
        # genuinely depends on the operand (CF = the operand was
        # nonzero, not a fixed guarantee), so it does NOT get this.
        preserved_cf_zero = False
        if op_name in ("inc", "dec") and state.last_flags is not None:
            preserved_cf_zero = state.last_flags[2]
        if val is EXTERNAL:
            _sym_write_dest(state, dst, EXTERNAL)
            if op_name != "not":
                # OF is NOT guaranteed zero for any of these (neg's OF can
                # be set negating INT_MIN; inc/dec's OF can be set at the
                # signed range boundary) - never treat these as eligible
                # for the of_zero-gated zero_fallback and/or/xor/test get.
                state.last_flags = (EXTERNAL, dst.width, preserved_cf_zero, False, None)
            return
        width = dst.width
        if op_name == "not":
            _sym_write_dest(state, dst, (~val) & _SYM_WIDTH_MASK[width])
            return  # NOT does not affect flags on real x86
        if op_name == "neg":
            result = (-val) & _SYM_WIDTH_MASK[width]
        elif op_name == "inc":
            result = (val + 1) & _SYM_WIDTH_MASK[width]
        elif op_name == "dec":
            result = (val - 1) & _SYM_WIDTH_MASK[width]
        else:
            raise AssertionError(op_name)
        _sym_write_dest(state, dst, result)
        state.last_flags = (result, width, preserved_cf_zero, False, None)
    return handler


_SYM_SHIFT_COUNT_MASK = {1: 0x1F, 2: 0x1F, 4: 0x1F, 8: 0x3F}


def _sym_make_shift(op_name):
    def handler(state, ops):
        dst, src = ops[0], ops[1]
        if src.kind == "imm" and (src.value & _SYM_SHIFT_COUNT_MASK[dst.width]) == 0:
            # Real x86 masks the shift count BEFORE using it - to 5 bits
            # for an 8/16/32-bit operand, 6 bits only for a 64-bit one
            # (Intel SDM) - so e.g. `shl eax, 32` is masked to a count of
            # 0 and is a genuine no-op, not "shift by 32". Using a flat
            # 0x3F mask regardless of width (this function's own earlier,
            # narrower fix) would miss that for anything narrower than
            # 64 bits. A masked count of 0 leaves EVERY flag untouched
            # (Intel SDM: "If the count is 0, the flags are not
            # affected") and the value is trivially unchanged - a
            # complete no-op, checked BEFORE resolving the shifted
            # operand at all. This must come first: an obfuscator
            # inserting a "shift by 0" (or an equivalent masked-to-0
            # count) to break naive flag tracking (the exact pattern
            # this was found against: `or al, 0` then `sal al, 0` before
            # a jnb/jns pair) commonly does so right after a partial-
            # register write that already conservatively tainted the
            # whole operand to fully-unsupported/None (see
            # _sym_write_dest's width-not-in-(4,8) case) - checking this
            # after resolving val would see that taint and wipe the
            # PRECEDING instruction's real flags anyway, exactly the bug
            # this exists to prevent.
            return
        val = _sym_resolve_operand(state, dst)
        if src.kind != "imm" or val is None:
            _sym_write_dest(state, dst, None)
            state.last_flags = None
            return
        count = src.value & _SYM_SHIFT_COUNT_MASK[dst.width]
        if val is EXTERNAL:
            _sym_write_dest(state, dst, EXTERNAL)
            # Shift flags (esp. OF/CF) have enough count/architecture-
            # specific edge cases that this never claims OF=CF=0. Unlike
            # inc/dec, a shift by a NONZERO count genuinely does modify
            # CF (count==0 is the true no-op, already handled above by
            # returning before this point) - no carry-forward here.
            state.last_flags = (EXTERNAL, dst.width, False, False, None)
            return
        width = dst.width
        if op_name == "shl":
            result = (val << count) & _SYM_WIDTH_MASK[width]
        elif op_name == "shr":
            result = (val & _SYM_WIDTH_MASK[width]) >> count
        elif op_name == "sar":
            result = (_sym_to_signed(val, width) >> count) & _SYM_WIDTH_MASK[width]
        else:
            raise AssertionError(op_name)
        _sym_write_dest(state, dst, result)
        state.last_flags = (result, width, False, False, None)
    return handler


def _sym_h_push(state, ops, rsp_family):
    state.write_reg(rsp_family, None)


def _sym_h_pop(state, ops, rsp_family):
    dst = ops[0]
    if dst.kind == "reg":
        state.write_reg(dst.family, None)
    state.write_reg(rsp_family, None)


_SYM_INSN_HANDLERS = {
    "nop": lambda state, ops: None,
    "mov": _sym_h_mov,
    "movzx": _sym_h_movzx,
    "movsx": _sym_h_movsx,
    "movsxd": _sym_h_movsx,
    "lea": _sym_h_lea,
    "add": _sym_make_alu2("add"),
    "sub": _sym_make_alu2("sub"),
    "and": _sym_make_alu2("and"),
    "or": _sym_make_alu2("or"),
    "xor": _sym_make_alu2("xor"),
    "cmp": _sym_h_cmp,
    "test": _sym_h_test,
    "not": _sym_make_unary("not"),
    "neg": _sym_make_unary("neg"),
    "inc": _sym_make_unary("inc"),
    "dec": _sym_make_unary("dec"),
    "shl": _sym_make_shift("shl"),
    "sal": _sym_make_shift("shl"),
    "shr": _sym_make_shift("shr"),
    "sar": _sym_make_shift("sar"),
    # push/pop take rsp_family as an extra argument - dispatched specially
    # in sym_run_block below rather than through this plain handler dict.
}

_sym_control_flow_mnemonics_cache = None


def _sym_control_flow_mnemonics():
    """Mnemonics that end a basic block (every Jcc variant, jcxz/ecxz/
    rcxz, and jmp) have no DATA effect of their own to model - gather_block
    always includes the block's own terminating instruction as the LAST
    entry of sym_insns (so sym_resolve_indirect_jump can read a trailing
    jmp's target operand), and sym_run_block must treat these as a pure
    no-op rather than "an unmodeled instruction" (which would otherwise
    reset the state - including last_cmp/last_flags - on literally the
    last instruction of every conditional-branch block, immediately
    before CfgTraceRunner reads them to resolve the branch; this was a
    real bug, not just an unmodeled-instruction fallback).
    """
    global _sym_control_flow_mnemonics_cache
    if _sym_control_flow_mnemonics_cache is None:
        _sym_control_flow_mnemonics_cache = set(_SYM_CONDITION_EVALUATORS) | {"jcxz", "jecxz", "jrcxz", "jmp"}
    return _sym_control_flow_mnemonics_cache


def sym_run_block(state, sym_insns, rsp_family):
    """Executes a linear sequence of SymInsn against a COPY of state,
    returning the resulting state.

    "call" is special-cased: registers are reset to EXTERNAL (not fully
    unsupported/None) and memory slots are cleared, since a real
    function's return value/output-parameters are, in the overwhelming
    majority of real code, genuine runtime computation - the same
    "we don't control this, but it's not a modeling failure" case
    EXTERNAL already represents for argument-derived data. This lets an
    ordinary "if (helper_call(...)) ..." resolve as a data-dependent
    branch (both sides real) without an LLM round-trip. (The counter-risk
    - an obfuscator hiding a genuinely fixed opaque-predicate constant
    behind a call specifically to defeat this - is judged rare enough to
    accept, especially since nothing is ever written to the database
    without the user reviewing the result afterward.)

    Any OTHER unmodeled instruction (an unusual bit-trick/SIMD op inline
    in the block, for example) resets to fully unsupported/None instead -
    deliberately kept strict, since an obfuscator choosing an exotic
    inline instruction specifically to compute a real dispatcher constant
    is a much more plausible adversarial pattern than hiding one behind
    an entire function call.
    """
    state = state.copy()
    for insn in sym_insns:
        mnem = insn.mnem
        try:
            if mnem == "push":
                _sym_h_push(state, insn.operands, rsp_family)
                continue
            if mnem == "pop":
                _sym_h_pop(state, insn.operands, rsp_family)
                continue
            if mnem == "call":
                state.mem = {}
                state.regs = dict(_sym_all_external_regs())
                state.last_cmp = None
                state.last_flags = None
                continue
            if mnem in _sym_control_flow_mnemonics():
                # The block's own terminating branch/jump - no data effect
                # to model, must not disturb last_cmp/last_flags/regs.
                continue
            handler = _SYM_INSN_HANDLERS.get(mnem)
            if handler is None:
                state.reset()
                continue
            handler(state, insn.operands)
        except Exception:
            state.reset()
    return state


# -- condition evaluation ---------------------------------------------------
# Each evaluator returns ("taken", True/False) | ("data_dependent", None)
# | ("unknown", None) - "data_dependent" means the comparison genuinely
# involves EXTERNAL (runtime/caller-controlled) data, so this is ordinary
# conditional logic from the original program, not an opaque predicate -
# every candidate should be marked REAL, not resolved one way or picked
# arbitrarily.

def _sym_cmp_based(fn, zero_fallback=None, needs=None, value_independent=False):
    """fn(left, right, width) evaluates from an explicit two-operand
    comparison (state.last_cmp, from cmp/sub) - pass None to never
    attempt this path at all (used for jo/jno: computing signed overflow
    from two arbitrary operands correctly is its own, riskier problem
    this deliberately does not take on; only the of_zero guarantee below
    is trusted). zero_fallback(value, width), when given, additionally
    lets this resolve from state.last_flags - needs picks which guarantee
    is required there: "cf" for ja/jae/jb/jbe (and aliases), "of" for
    jg/jge/jl/jle and jo/jno. and/or/xor/test guarantee BOTH cf_zero and
    of_zero simultaneously (Intel SDM: "the OF and CF flags are
    cleared"); inc/dec carry forward cf_zero alone (real x86 never
    touches CF for inc/dec, but DOES change OF) - see the
    SymState.last_flags docstring. Without this fallback (fn/
    zero_fallback omitted for anything not confidently reasoned through),
    these conditions were previously unresolvable after a plain and/or/
    xor/test with no explicit cmp - a real gap that let a genuine opaque
    predicate built on flag reuse fall through to the LLM instead of
    resolving deterministically.

    value_independent=True is for jae/jnb/jnc, jb/jc/jnae, and jo/jno
    specifically - their condition is PURELY "CF=0"/"CF=1"/"OF=0"/"OF=1",
    nothing else, so once the relevant flag is guaranteed zero the
    outcome is fixed regardless of what the actual result value even is
    (EXTERNAL or concrete) - unlike ja/jnbe/jbe/jna/jg/jle (and aliases),
    which ALSO depend on ZF and so genuinely do need a known value to
    resolve. Without this, an EXTERNAL value forced "data_dependent" even
    for these value-independent conditions - a real gap that left e.g.
    `jnb` unresolved after `or reg, imm` on an unknown register, when
    CF=0 was already fully guaranteed either way.
    """
    def evaluator(state):
        if fn is not None and state.last_cmp is not None:
            left, right, width = state.last_cmp
            if left is EXTERNAL or right is EXTERNAL:
                return "data_dependent", None
            return "taken", fn(left, right, width)
        if zero_fallback is not None and state.last_flags is not None:
            value, width, cf_zero, of_zero, _refine_info = state.last_flags
            flag_zero = cf_zero if needs == "cf" else of_zero
            if flag_zero:
                if value_independent:
                    return "taken", zero_fallback(value, width)
                if value is EXTERNAL:
                    return "data_dependent", None
                return "taken", zero_fallback(value, width)
        return "unknown", None
    return evaluator


def _sym_zero_based(fn):
    def evaluator(state):
        if state.last_cmp is not None:
            left, right, width = state.last_cmp
            if left is EXTERNAL or right is EXTERNAL:
                return "data_dependent", None
            value = (left - right) & _SYM_WIDTH_MASK[width]
            return "taken", fn(value, width)
        if state.last_flags is not None:
            # ZF/SF are always derivable from the result value alone,
            # regardless of which operation produced it - unlike
            # _sym_cmp_based's zero_fallback, this doesn't need (or check)
            # of_cf_zero at all.
            value, width, _cf_zero, _of_zero, _refine_info = state.last_flags
            if value is EXTERNAL:
                return "data_dependent", None
            return "taken", fn(value, width)
        return "unknown", None
    return evaluator


_SYM_CONDITION_EVALUATORS = {
    "jz": _sym_zero_based(lambda v, w: v == 0),
    "je": _sym_zero_based(lambda v, w: v == 0),
    "jnz": _sym_zero_based(lambda v, w: v != 0),
    "jne": _sym_zero_based(lambda v, w: v != 0),
    "js": _sym_zero_based(lambda v, w: _sym_to_signed(v, w) < 0),
    "jns": _sym_zero_based(lambda v, w: _sym_to_signed(v, w) >= 0),
    # PF is always derivable from a concrete result value alone (only its
    # low byte matters, regardless of operand width) - no flag guarantee
    # needed, same family as jz/js above. Previously missing entirely -
    # jp/jnp always fell straight to the LLM, even after a concrete
    # result the engine could have computed parity from directly.
    "jp": _sym_zero_based(lambda v, w: _sym_parity_even(v)),
    "jpe": _sym_zero_based(lambda v, w: _sym_parity_even(v)),
    "jnp": _sym_zero_based(lambda v, w: not _sym_parity_even(v)),
    "jpo": _sym_zero_based(lambda v, w: not _sym_parity_even(v)),

    # zero_fallback below applies only when cf_zero is True (see
    # SymState.last_flags) - and/or/xor/test guarantee it directly;
    # inc/dec carry it forward from whatever it already was, since real
    # x86 never touches CF for inc/dec. With CF guaranteed 0: ja/jnbe
    # (CF=0,ZF=0) reduces to "value != 0", jae/jnb/jnc (CF=0) is
    # unconditionally true, jb/jc/jnae (CF=1) is unconditionally false,
    # jbe/jna (CF=1 or ZF=1) reduces to "value == 0".
    "ja": _sym_cmp_based(lambda l, r, w: l > r, zero_fallback=lambda v, w: v != 0, needs="cf"),
    "jnbe": _sym_cmp_based(lambda l, r, w: l > r, zero_fallback=lambda v, w: v != 0, needs="cf"),
    "jae": _sym_cmp_based(lambda l, r, w: l >= r, zero_fallback=lambda v, w: True, needs="cf", value_independent=True),
    "jnb": _sym_cmp_based(lambda l, r, w: l >= r, zero_fallback=lambda v, w: True, needs="cf", value_independent=True),
    "jnc": _sym_cmp_based(lambda l, r, w: l >= r, zero_fallback=lambda v, w: True, needs="cf", value_independent=True),
    "jb": _sym_cmp_based(lambda l, r, w: l < r, zero_fallback=lambda v, w: False, needs="cf", value_independent=True),
    "jc": _sym_cmp_based(lambda l, r, w: l < r, zero_fallback=lambda v, w: False, needs="cf", value_independent=True),
    "jnae": _sym_cmp_based(lambda l, r, w: l < r, zero_fallback=lambda v, w: False, needs="cf", value_independent=True),
    "jbe": _sym_cmp_based(lambda l, r, w: l <= r, zero_fallback=lambda v, w: v == 0, needs="cf"),
    "jna": _sym_cmp_based(lambda l, r, w: l <= r, zero_fallback=lambda v, w: v == 0, needs="cf"),

    # of_zero-gated (only and/or/xor/test guarantee this - inc/dec do NOT
    # carry it forward, since their OF genuinely can be set at the signed
    # range boundary). With OF guaranteed 0: jg/jnle (ZF=0,SF=OF) reduces
    # to signed value > 0, jge/jnl (SF=OF) to signed value >= 0, jl/jnge
    # (SF!=OF) to signed value < 0, jle/jng (ZF=1 or SF!=OF) to signed
    # value <= 0 - the exact gap that let the flag-reuse opaque-predicate
    # cascade described in the bug report fall through to the LLM instead
    # of resolving here.
    "jg": _sym_cmp_based(
        lambda l, r, w: _sym_to_signed(l, w) > _sym_to_signed(r, w),
        zero_fallback=lambda v, w: _sym_to_signed(v, w) > 0, needs="of",
    ),
    "jnle": _sym_cmp_based(
        lambda l, r, w: _sym_to_signed(l, w) > _sym_to_signed(r, w),
        zero_fallback=lambda v, w: _sym_to_signed(v, w) > 0, needs="of",
    ),
    "jge": _sym_cmp_based(
        lambda l, r, w: _sym_to_signed(l, w) >= _sym_to_signed(r, w),
        zero_fallback=lambda v, w: _sym_to_signed(v, w) >= 0, needs="of",
    ),
    "jnl": _sym_cmp_based(
        lambda l, r, w: _sym_to_signed(l, w) >= _sym_to_signed(r, w),
        zero_fallback=lambda v, w: _sym_to_signed(v, w) >= 0, needs="of",
    ),
    "jl": _sym_cmp_based(
        lambda l, r, w: _sym_to_signed(l, w) < _sym_to_signed(r, w),
        zero_fallback=lambda v, w: _sym_to_signed(v, w) < 0, needs="of",
    ),
    "jnge": _sym_cmp_based(
        lambda l, r, w: _sym_to_signed(l, w) < _sym_to_signed(r, w),
        zero_fallback=lambda v, w: _sym_to_signed(v, w) < 0, needs="of",
    ),
    "jle": _sym_cmp_based(
        lambda l, r, w: _sym_to_signed(l, w) <= _sym_to_signed(r, w),
        zero_fallback=lambda v, w: _sym_to_signed(v, w) <= 0, needs="of",
    ),
    "jng": _sym_cmp_based(
        lambda l, r, w: _sym_to_signed(l, w) <= _sym_to_signed(r, w),
        zero_fallback=lambda v, w: _sym_to_signed(v, w) <= 0, needs="of",
    ),

    # jo/jno test OF directly - purely "OF=1"/"OF=0", nothing else, so
    # once of_zero guarantees OF=0 the outcome is fixed regardless of the
    # actual value (same value_independent shape as jae/jb for CF).
    # fn=None deliberately: computing signed overflow from two arbitrary
    # cmp/sub operands correctly is a separate, riskier problem this does
    # not take on here - only the of_zero guarantee is trusted. Previously
    # missing entirely - jo/jno always fell straight to the LLM, even
    # after and/or/xor/test made OF=0 fully certain either way (e.g. the
    # classic obfuscator trick "xor reg, 0" specifically to make jno an
    # opaque predicate).
    "jo": _sym_cmp_based(None, zero_fallback=lambda v, w: False, needs="of", value_independent=True),
    "jno": _sym_cmp_based(None, zero_fallback=lambda v, w: True, needs="of", value_independent=True),
}


def sym_evaluate_branch_condition(state, mnem, rcx_family):
    mnem = mnem.lower()
    if mnem in ("jcxz", "jecxz", "jrcxz"):
        width = {"jcxz": 2, "jecxz": 4, "jrcxz": 8}[mnem]
        val = state.read_reg(rcx_family, width)
        if val is None:
            return "unknown", None
        if val is EXTERNAL:
            return "data_dependent", None
        return "taken", (val == 0)
    evaluator = _SYM_CONDITION_EVALUATORS.get(mnem)
    if evaluator is None:
        return "unknown", None
    kind, taken = evaluator(state)
    if kind == "taken":
        return kind, taken
    # The context-free evaluator couldn't resolve this (or could only
    # say "data_dependent") on the flags alone - see if a just-resolved
    # prior branch on THIS same edge tautologically determines it anyway
    # (see _sym_retest_from_last_branch). Checked as a fallback, not
    # first: this should never disagree with a "taken" the plain
    # evaluator already gave, so there's nothing to gain by checking it
    # earlier, only extra work.
    retest = _sym_retest_from_last_branch(state, mnem)
    if retest is not None:
        return "taken", retest
    return kind, taken


def sym_resolve_conditional_branch(state, mnem, successors, rcx_family):
    """successors: exactly [jump_target, fallthrough] (gather_block's
    fixed ordering for a 2-way conditional branch). Returns:
      ("opaque", {ea: True/False})       - definitive REAL/DEAD verdicts
      ("data_dependent", {ea: True, ...}) - every successor is REAL
      ("unknown", None)                   - caller must fall back to the LLM
    """
    if len(successors) != 2:
        return "unknown", None
    kind, taken = sym_evaluate_branch_condition(state, mnem, rcx_family)
    if kind == "unknown":
        return "unknown", None
    if kind == "data_dependent":
        return "data_dependent", {s.ea: True for s in successors}
    jump_target, fallthrough = successors[0], successors[1]
    if taken:
        return "opaque", {jump_target.ea: True, fallthrough.ea: False}
    return "opaque", {jump_target.ea: False, fallthrough.ea: True}


def _compute_and_mask_refinement(state, mnem, successors):
    """If state.last_flags carries an AND-single-bit-mask refinement
    opportunity (see SymState.last_flags / _sym_make_alu2's "and" case)
    and mnem is a plain zero/nonzero test (jz/je/jnz/jne - the only
    conditions where "!=0" is EXACTLY equivalent to "==mask", since AND
    with a single-bit mask can only ever produce 0 or that exact mask),
    returns {successor_ea: refined_state} - the register the AND wrote
    is narrowed from EXTERNAL to the concrete value now provably true on
    THAT SPECIFIC edge. Returns {} for anything else: a different
    condition code, no pending refinement, or fewer/more than two
    successors - deliberately narrow in scope (one bit, one specific
    instruction pattern) rather than a general value-range/constraint
    model. successors are assumed in gather_block's fixed
    [jump_target, fallthrough] order, matching sym_resolve_conditional_
    branch above.
    """
    if len(successors) != 2 or state.last_flags is None:
        return {}
    _value, width, cf_zero, of_zero, refine_info = state.last_flags
    if refine_info is None:
        return {}
    if mnem not in ("jz", "je", "jnz", "jne"):
        return {}
    family, mask = refine_info
    jump_target, fallthrough = successors[0], successors[1]
    if mnem in ("jz", "je"):
        zero_succ, nonzero_succ = jump_target, fallthrough
    else:
        nonzero_succ, zero_succ = jump_target, fallthrough
    refined = {}
    if zero_succ.ea is not None:
        s0 = state.copy()
        s0.write_reg(family, 0)
        # Refining regs[family] alone would not help a SUBSEQUENT
        # flags-based Jcc (jle/jg/etc.) reached with no intervening
        # flag-setting instruction - those consult last_flags directly,
        # never re-derive it from a register. Refine both, so the very
        # next such condition resolves precisely instead of still
        # seeing EXTERNAL there. refine_info is cleared (already
        # "consumed" - this concrete value needs no further narrowing).
        s0.last_flags = (0, width, cf_zero, of_zero, None)
        refined[zero_succ.ea] = s0
    if nonzero_succ.ea is not None:
        s1 = state.copy()
        s1.write_reg(family, mask)
        s1.last_flags = (mask, width, cf_zero, of_zero, None)
        refined[nonzero_succ.ea] = s1
    return refined


# Groups every Jcc mnemonic (and its aliases) by the EFLAGS condition it
# actually tests, for _compute_edge_refinements/_sym_retest_from_last_branch
# below. jcxz/jecxz/jrcxz deliberately have no entry - they test a
# register (RCX), not a flag, so they never participate in this
# mechanism either as a source or a target.
_JCC_CONDITION_CLASS = {
    "jz": "z", "je": "z",
    "jnz": "nz", "jne": "nz",
    "js": "s", "jns": "ns",
    "jp": "p", "jpe": "p",
    "jnp": "np", "jpo": "np",
    "ja": "a", "jnbe": "a",
    "jbe": "be", "jna": "be",
    "jae": "ae", "jnb": "ae", "jnc": "ae",
    "jb": "b", "jc": "b", "jnae": "b",
    "jg": "g", "jnle": "g",
    "jle": "le", "jng": "le",
    "jge": "ge", "jnl": "ge",
    "jl": "l", "jnge": "l",
    "jo": "o", "jno": "no",
}

# The exact logical negation of each condition class above (z<->nz,
# s<->ns, a<->be, etc.) - two Jccs whose classes are this pair test the
# SAME underlying flag(s) with opposite senses, so if one's truth value
# is known the other's is too, with nothing else needed.
_JCC_CLASS_NEGATION = {
    "z": "nz", "nz": "z",
    "s": "ns", "ns": "s",
    "p": "np", "np": "p",
    "a": "be", "be": "a",
    "ae": "b", "b": "ae",
    "g": "le", "le": "g",
    "ge": "l", "l": "ge",
    "o": "no", "no": "o",
}


def _sym_retest_from_last_branch(state, mnem):
    """If state.last_branch records a prior conditional branch's own
    edge-specific truth value, and NOTHING has touched last_cmp/
    last_flags since (verified by comparing the LIVE values against the
    stored snapshot - a plain equality check, not proactive
    invalidation, so this needs no changes to any instruction handler),
    and mnem tests the SAME or the exact logical NEGATION of that prior
    branch's condition, returns the tautologically-determined outcome
    for mnem here. Returns None otherwise (caller falls back to the
    normal, context-free per-mnem evaluator).

    This is what resolves the classic "test one flag twice with opposite
    senses through a no-op in between" obfuscation trick (e.g. `js
    target; <no-op>; jns target`) - each half looks independently
    unresolvable/data-dependent on its own (the flag's actual value may
    be genuinely unknown), yet the PAIR is tautologically always taken
    one way or the other, regardless of what determines the flag.
    """
    if state.last_branch is None:
        return None
    prior_class, prior_taken, cmp_snapshot, flags_snapshot = state.last_branch
    if state.last_cmp != cmp_snapshot or state.last_flags != flags_snapshot:
        return None  # something changed the flags since - the tag is stale
    this_class = _JCC_CONDITION_CLASS.get(mnem)
    if this_class is None:
        return None
    if this_class == prior_class:
        return prior_taken
    if _JCC_CLASS_NEGATION.get(prior_class) == this_class:
        return not prior_taken
    return None


def _compute_edge_refinements(state, mnem, successors):
    """Combines every per-edge state-narrowing refinement this trace
    knows about into one {successor_ea: refined_state} map:

    - AND-single-bit-mask + jz/je/jnz/jne (_compute_and_mask_refinement).
    - Same-flag retest (_sym_retest_from_last_branch): ANY conditional
      branch tautologically reveals its OWN condition's truth value on
      each of its two edges - jump-taken means the condition was true,
      fallthrough means it was false - regardless of what data
      determined it or how its overall REAL/DEAD verdict was decided
      (symbolic engine or LLM). Tagging this unconditionally (not just
      when this block's own verdict came from the engine) is what lets
      it help even when an LLM had to resolve THIS branch but a LATER
      one testing the same/negated condition can still be resolved for
      free. See SymState.last_branch and sym_evaluate_branch_condition.

    successors are assumed in gather_block's fixed [jump_target,
    fallthrough] order, matching sym_resolve_conditional_branch and
    _compute_and_mask_refinement above.
    """
    if len(successors) != 2:
        return {}
    jump_target, fallthrough = successors[0], successors[1]
    refined = _compute_and_mask_refinement(state, mnem, successors)
    mnem_class = _JCC_CONDITION_CLASS.get(mnem.lower())
    if mnem_class is not None:
        snapshot = (state.last_cmp, state.last_flags)
        for succ, taken in ((jump_target, True), (fallthrough, False)):
            if succ.ea is None:
                continue
            # Build on top of the AND-mask refinement's result if it
            # already narrowed this same edge, rather than discarding it.
            base = refined.get(succ.ea, state)
            merged = base.copy()
            merged.last_branch = (mnem_class, taken, snapshot[0], snapshot[1])
            refined[succ.ea] = merged
    return refined


def sym_resolve_indirect_jump(state, target_operand, successors):
    if len(successors) == 1 and successors[0].ea is None:
        return "unknown", None
    resolved = _sym_resolve_operand(state, target_operand)
    if resolved is None:
        return "unknown", None
    if resolved is EXTERNAL:
        return "data_dependent", {s.ea: True for s in successors}
    matches = [s for s in successors if s.ea == resolved]
    if len(matches) == 1:
        return "opaque", {s.ea: (s.ea == resolved) for s in successors}
    return "unknown", None


# -- ida_ua.insn_t / op_t adapter -------------------------------------------
# The only part of this engine that touches IDA APIs - translates real
# decoded instructions into the plain SymOperand/SymInsn descriptors above.

_GPR_64BIT_NAMES = (
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
)
_gpr_family_cache = {}
_gpr_family_set_cache = None


def _family_of(name):
    """Register-family index for a canonical 64-bit GPR name, resolved
    via ida_idp.str2reg (the documented name->index lookup) rather than
    assumed from a hardcoded numbering scheme - lazily cached, since
    str2reg needs a processor/database already loaded and this file's
    module-level code may run before one exists.
    """
    if name not in _gpr_family_cache:
        try:
            idx = ida_idp.str2reg(name)
        except Exception:
            idx = -1
        _gpr_family_cache[name] = idx if idx is not None and idx >= 0 else None
    return _gpr_family_cache[name]


def _gpr_family_set():
    global _gpr_family_set_cache
    if _gpr_family_set_cache is None:
        _gpr_family_set_cache = {f for f in (_family_of(n) for n in _GPR_64BIT_NAMES) if f is not None}
    return _gpr_family_set_cache


_all_external_regs_cache = None


def _sym_all_external_regs():
    global _all_external_regs_cache
    if _all_external_regs_cache is None:
        _all_external_regs_cache = {f: EXTERNAL for f in _gpr_family_set()}
    return _all_external_regs_cache


def sym_initial_state():
    """A SymState for the very start of a trace: every GPR is seeded
    EXTERNAL rather than left absent/unsupported, since at function
    entry every register genuinely holds SOME value the caller put
    there - reading through an untouched register (e.g. an incoming
    argument in rcx) is exactly the "genuine runtime/caller-controlled
    data" case EXTERNAL exists for, not a modeling gap. Without this,
    the extremely common "if (arg->field == N)"-shaped code never
    resolves automatically even though it isn't obfuscation at all.
    """
    state = SymState()
    state.regs = dict(_sym_all_external_regs())
    return state


_SYM_DTYPE_WIDTH = None


def _dtype_width(dtype):
    global _SYM_DTYPE_WIDTH
    if _SYM_DTYPE_WIDTH is None:
        _SYM_DTYPE_WIDTH = {
            ida_ua.dt_byte: 1, ida_ua.dt_word: 2, ida_ua.dt_dword: 4, ida_ua.dt_qword: 8,
        }
    return _SYM_DTYPE_WIDTH.get(dtype)


_NUMERIC_TERM_RE = re.compile(r"^[+-]?(?:0[xX][0-9a-fA-F]+|[0-9A-Fa-f]+[hH]?|\d+)$")
_PTR_PREFIX_RE = re.compile(r"^(?:byte|word|dword|qword|xmmword|ymmword|zmmword)\s+ptr\s+(.*)$", re.IGNORECASE)


_SCALE_RE = re.compile(r"^([A-Za-z0-9]+)\*([0-9]+)$")


def _parse_memory_operand_text(ea, opnum):
    """Parses a memory operand's base/index/scale registers directly from
    IDA's own RENDERED text (idc.print_operand), rather than the raw
    op_t SIB fields - those aren't safely decodable from documented
    Python fields without the C++ SDK headers (not shipped with this
    install), so guessing at their bit layout was judged too risky.
    Register NAMES and small integer SCALE factors are low-risk,
    well-defined tokens to pull out of text; the numeric displacement
    itself always comes from the structured op.addr field instead (see
    _extract_sym_operand), never from here.

    Returns (base_family_or_None, index_family_or_None, scale) - scale
    is only meaningful when index is not None - or None if the text
    can't be confidently parsed this way (caller must treat that as
    unsupported and fall back to the LLM, never guess). Assumes IDA's
    default Intel-syntax bracket rendering; a database configured for
    AT&T syntax will simply fail to parse here and fall back the same
    safe way.
    """
    try:
        text = idc.print_operand(ea, opnum) or ""
    except Exception:
        return None
    text = text.strip()
    m = _PTR_PREFIX_RE.match(text)
    if m:
        text = m.group(1).strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    inner = text[1:-1]

    reg_terms = []  # (family, scale_or_None)
    for raw_term in re.findall(r"[+-]?[^+-]+", inner):
        term = raw_term.strip()
        if not term:
            continue
        if term[0] in "+-":
            sign, body = term[0], term[1:].strip()
        else:
            sign, body = "+", term
        if _NUMERIC_TERM_RE.match(body):
            continue  # displacement - comes from op.addr instead, not here
        if sign == "-":
            return None  # a negated register term isn't an addressing form we handle
        scale_match = _SCALE_RE.match(body)
        if scale_match:
            reg_name, scale_str = scale_match.group(1), scale_match.group(2)
            scale_val = int(scale_str)
            if scale_val not in (1, 2, 4, 8):
                return None
        else:
            reg_name, scale_val = body, None
        fam = _family_of(reg_name.lower())
        if fam is None or fam not in _gpr_family_set():
            return None  # unrecognized register (segment override, rip, xmm, ...) -> unsupported
        reg_terms.append((fam, scale_val))

    if len(reg_terms) == 0:
        return (None, None, 1)
    if len(reg_terms) == 1:
        fam, sc = reg_terms[0]
        return (fam, None, 1) if sc is None else (None, fam, sc)
    if len(reg_terms) == 2:
        scaled = [t for t in reg_terms if t[1] is not None]
        unscaled = [t for t in reg_terms if t[1] is None]
        if len(scaled) == 1 and len(unscaled) == 1:
            return (unscaled[0][0], scaled[0][0], scaled[0][1])
        if len(scaled) == 0:
            # Neither term has an explicit *scale (e.g. "[rax+rbx]"), so
            # which one is "base" vs "index" is genuinely ambiguous from
            # text alone - but it doesn't matter: with scale 1 on both,
            # the effective-address sum is identical either way, and
            # mem_indexed operands are never round-trip-tracked (unlike
            # mem_simple), so there is no "wrong base for invalidation
            # purposes" concern here either.
            return (reg_terms[0][0], reg_terms[1][0], 1)
        return None  # both terms explicitly scaled - not a real addressing form, bail
    return None  # 3+ register terms - unexpected, bail out rather than guess


def _extract_sym_operand(ea, opnum, op):
    width = _dtype_width(op.dtype)
    if op.type == ida_ua.o_imm:
        if width is None:
            return _UNSUPPORTED_OPERAND
        return SymOperand(kind="imm", family=None, value=op.value, width=width)
    if op.type == ida_ua.o_reg:
        if width is None or op.reg not in _gpr_family_set():
            return _UNSUPPORTED_OPERAND
        return SymOperand(kind="reg", family=op.reg, value=None, width=width)
    if op.type == ida_ua.o_mem:
        if width is None:
            return _UNSUPPORTED_OPERAND
        return SymOperand(kind="mem_direct", family=None, value=op.addr, width=width)
    if op.type in (ida_ua.o_displ, ida_ua.o_phrase):
        if width is None:
            return _UNSUPPORTED_OPERAND
        parsed = _parse_memory_operand_text(ea, opnum)
        if parsed is None:
            return _UNSUPPORTED_OPERAND
        base, index, scale = parsed
        disp = op.addr if op.type == ida_ua.o_displ else 0
        if index is None:
            if base is None:
                return _UNSUPPORTED_OPERAND
            return SymOperand(kind="mem_simple", family=base, value=disp, width=width)
        terms = ((index, scale),) if base is None else ((base, 1), (index, scale))
        return SymOperand(kind="mem_indexed", family=None, value=disp, width=width, terms=terms)
    return _UNSUPPORTED_OPERAND


_sym_modeled_mnemonics_cache = None


def _sym_modeled_mnemonics():
    global _sym_modeled_mnemonics_cache
    if _sym_modeled_mnemonics_cache is None:
        # "jmp" is the one control-flow mnemonic whose OPERAND is actually
        # read afterward (sym_resolve_indirect_jump needs a trailing
        # indirect jmp's target operand) - every Jcc/jcxz variant only
        # ever has its .mnem looked at (by sym_resolve_conditional_branch),
        # never its operands, so they correctly stay out of this set.
        _sym_modeled_mnemonics_cache = set(_SYM_INSN_HANDLERS) | {"push", "pop", "jmp"}
    return _sym_modeled_mnemonics_cache


def _extract_sym_insn(ea, insn):
    mnem = (insn.get_canon_mnem() or "").lower()
    if mnem not in _sym_modeled_mnemonics():
        # sym_run_block resets the whole state for any unmodeled mnemonic
        # regardless of its operands, so extracting them (including the
        # idc.print_operand-based indexed-addressing check below, which
        # is the most expensive part of this) would be pure waste - most
        # of a typical function's instructions are NOT in the modeled
        # whitelist, so skipping this is a meaningful chunk of the
        # per-instruction cost of walking a large function.
        return SymInsn(mnem=mnem, operands=())
    operands = []
    for i in range(8):
        op = insn.ops[i]
        if op.type == ida_ua.o_void:
            break
        operands.append(_extract_sym_operand(ea, i, op))
    return SymInsn(mnem=mnem, operands=tuple(operands))


def render_block_text(insn_eas):
    """Renders disassembly text for a block on demand. NOT called during
    the main gather_block() walk - text is only ever needed for an LLM
    prompt (rare: only when a decision point can't be resolved
    symbolically) or a graph-view tooltip (only when the user actually
    hovers a node) - generating it unconditionally for every instruction
    of every block, most of which are never looked at again, was a
    meaningful chunk of wall-clock time on any function of real size.
    """
    lines = []
    for ea in insn_eas:
        try:
            line = idc.generate_disasm_line(ea, idc.GENDSM_REMOVE_TAGS)
        except Exception:
            line = idc.GetDisasm(ea)
        lines.append("%#010x  %s" % (ea, ida_lines.tag_remove(line) if line else ""))
    return "\n".join(lines)


def gather_block(start_ea, max_instrs=_MAX_BLOCK_INSTRS, claimed_insns=None):
    """Walks linearly forward from start_ea, one basic block's worth of
    instructions, classifying how it ends and enumerating its successors.
    Does not recurse into successors - callers drive the worklist.

    Fixes up each instruction's item boundary (undefine + recreate, never
    touching bytes) as it walks, before relying on it for anything -
    obfuscated code frequently has real code starting mid-way through
    what IDA's original analysis mis-disassembled as one bigger
    instruction, and CodeRefsFrom's branch-target resolution below is
    only correct once that's fixed. This is the one place this feature
    writes to the database outside of the user's explicit Accept step;
    see CfgTraceRunner for how these calls are kept off
    LlamaStreamWorker's MFF_FAST callback thread context.

    claimed_insns, if given, is a dict shared across every gather_block()
    call within the same trace (ea -> instruction size), used to detect
    a target landing STRICTLY inside an instruction some OTHER block in
    this trace already claimed and fixed up - a classic overlapping-
    instruction obfuscation trick (the same bytes decode differently
    depending on which address you start from). Re-walking into the
    middle of an already-established item would otherwise silently
    DESTROY that earlier block's boundary - del_items() deletes the
    WHOLE item covering any address passed to it, not just the requested
    range - corrupting what that earlier (possibly already-marked-real)
    block's own insn_eas point to, even though its actual REAL/DEAD
    verdict logic (decode_insn-based, independent of IDA's item state)
    remains correct. When detected, the walk stops immediately WITHOUT
    touching those bytes at all; the caller treats this the same as
    "undecodable"/"truncated" - flagged for manual review rather than
    guessing which interpretation of the overlapping bytes is real.

    BlockInfo.text is intentionally left empty here - call
    render_block_text(block.insn_eas) if you actually need it (see that
    function's docstring for why this is lazy).
    """
    insn_eas = []
    sym_insns = []
    ea = start_ea
    for _ in range(max_instrs):
        if claimed_insns is not None:
            overlap_start, overlap_size = _find_overlapping_claim(ea, claimed_insns)
            if overlap_start is not None:
                return BlockInfo(
                    start_ea=start_ea, end_ea=ea, text="", kind="overlap",
                    successors=[], last_insn_ea=ea, insn_eas=insn_eas, sym_insns=sym_insns,
                )
        insn = ida_ua.insn_t()
        size = ida_ua.decode_insn(insn, ea)
        if size <= 0:
            return BlockInfo(
                start_ea=start_ea, end_ea=ea, text="", kind="undecodable",
                successors=[], last_insn_ea=ea, insn_eas=insn_eas, sym_insns=sym_insns,
            )
        _fixup_instruction_boundary(ea, size)
        if claimed_insns is not None:
            claimed_insns[ea] = size
        insn_eas.append(ea)

        feature = insn.get_canon_feature()
        is_call = bool(feature & ida_idp.CF_CALL)
        is_stop = bool(feature & ida_idp.CF_STOP)
        is_indirect = bool(feature & ida_idp.CF_JUMP)
        next_ea = ea + size

        if is_call:
            # Assume the call returns (v1 does not reason about
            # non-returning calls) - the block simply continues past it.
            # The symbolic engine treats "call" as an unmodeled mnemonic
            # (full state reset), same conservative v1 simplification.
            sym_insns.append(SymInsn(mnem="call", operands=()))
            ea = next_ea
            continue

        sym_insns.append(_extract_sym_insn(ea, insn))

        if not is_stop and not is_indirect:
            # Plain non-control-transfer instruction, or a conditional
            # branch feature check below handles that case separately.
            explicit_targets = []
            try:
                explicit_targets = list(idautils.CodeRefsFrom(ea, 0))
            except Exception:
                pass
            if not explicit_targets:
                ea = next_ea
                continue
            # A control-transfer that CF_STOP/CF_JUMP didn't flag as such
            # (e.g. some conditional-branch encodings) but that does have
            # explicit code targets - treat it as ending the block.
            successors = [
                BlockSuccessor(ea=t, role="jump_target", case_values=None, note=None)
                for t in explicit_targets
            ]
            successors.append(BlockSuccessor(ea=next_ea, role="fallthrough", case_values=None, note=None))
            kind = "conditional_branch" if len(successors) > 1 else "unconditional_jump"
            return BlockInfo(
                start_ea=start_ea, end_ea=next_ea, text="", kind=kind,
                successors=successors, last_insn_ea=ea, insn_eas=insn_eas, sym_insns=sym_insns,
            )

        if is_indirect:
            try:
                successors = _resolve_indirect_jump_successors(ea, insn_eas)
            except Exception:
                # Never let successor enumeration abort the whole trace -
                # fall back to "unresolved" (the block itself is still
                # real, only its final jump's destination is unknown).
                successors = [BlockSuccessor(
                    ea=None, role="unresolved", case_values=None,
                    note="Indirect jump; successor enumeration raised an error.",
                )]
            return BlockInfo(
                start_ea=start_ea, end_ea=next_ea, text="", kind="indirect_jump",
                successors=successors, last_insn_ea=ea, insn_eas=insn_eas, sym_insns=sym_insns,
            )

        # CF_STOP: doesn't pass execution to the next instruction.
        explicit_targets = []
        try:
            explicit_targets = list(idautils.CodeRefsFrom(ea, 0))
        except Exception:
            pass
        if not explicit_targets:
            return BlockInfo(
                start_ea=start_ea, end_ea=next_ea, text="", kind="return",
                successors=[], last_insn_ea=ea, insn_eas=insn_eas, sym_insns=sym_insns,
            )
        if len(explicit_targets) == 1:
            successors = [BlockSuccessor(ea=explicit_targets[0], role="jump_target", case_values=None, note=None)]
            return BlockInfo(
                start_ea=start_ea, end_ea=next_ea, text="", kind="unconditional_jump",
                successors=successors, last_insn_ea=ea, insn_eas=insn_eas, sym_insns=sym_insns,
            )
        successors = [
            BlockSuccessor(ea=t, role="jump_target", case_values=None, note=None) for t in explicit_targets
        ]
        return BlockInfo(
            start_ea=start_ea, end_ea=next_ea, text="", kind="conditional_branch",
            successors=successors, last_insn_ea=ea, insn_eas=insn_eas, sym_insns=sym_insns,
        )

    return BlockInfo(
        start_ea=start_ea, end_ea=ea, text="", kind="truncated",
        successors=[], last_insn_ea=ea, insn_eas=insn_eas, sym_insns=sym_insns,
    )


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
        # Migrate old single-server configs (pre multi-server support) that
        # still have a "base_url" string but no "server_urls" list.
        if "server_urls" not in data and isinstance(data.get("base_url"), str) and data["base_url"].strip():
            data["server_urls"] = [data["base_url"].strip()]
        return cls(**data)

    def save(self):
        self._validate()
        # Stamp the CURRENT defaults' hashes so a future load can tell an
        # unmodified default (adopt the newer default then) from a genuine
        # customization (keep it). See _upgrade_stale_default_prompts.
        self.system_prompt_default_hash = _prompt_hash(DEFAULT_SYSTEM_PROMPT)
        self.cfg_trace_prompt_default_hash = _prompt_hash(CFG_TRACE_SYSTEM_PROMPT)
        data = self.to_dict()
        # Never persist a prompt that is just the current default - store it
        # empty so a non-customizer always picks up the latest default on
        # load (load restores the default from empty) instead of freezing a
        # copy that would shadow every future prompt improvement. A genuinely
        # customized prompt (differs from the default) is stored verbatim.
        if data.get("system_prompt") == DEFAULT_SYSTEM_PROMPT:
            data["system_prompt"] = ""
        if data.get("cfg_trace_system_prompt") == CFG_TRACE_SYSTEM_PROMPT:
            data["cfg_trace_system_prompt"] = ""
        path = self._config_path()
        tmp_path = path + ".tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp_path, path)
        except Exception as exc:
            ida_kernwin.msg("[%s] Failed to save config: %s\n" % (PLUGIN_NAME, exc))

    def to_dict(self):
        return {key: getattr(self, key) for key in self._FIELDS}

    def clone(self):
        return PluginConfig(**self.to_dict())

    def server_label(self, url):
        """Short display label for a server: its configured name/comment if
        any, else the raw URL. For compact UI use (status lines, table
        previews)."""
        name = (self.server_names or {}).get(url)
        return name if name else url

    def server_label_full(self, url):
        """Longer display label that always includes the URL, even when a
        name is set, for logs/tooltips where disambiguation matters more
        than brevity."""
        name = (self.server_names or {}).get(url)
        return "%s (%s)" % (name, url) if name else url

    def _validate(self):
        cleaned_urls = []
        seen_urls = set()
        url_normalize_map = {}
        for url in (self.server_urls or []):
            if not isinstance(url, str):
                continue
            original = url
            url = url.strip().rstrip("/")
            if not url or not (url.startswith("http://") or url.startswith("https://")):
                continue
            url_normalize_map[original] = url
            if url not in seen_urls:
                seen_urls.add(url)
                cleaned_urls.append(url)
        self.server_urls = cleaned_urls or list(DEFAULT_CONFIG["server_urls"])

        cleaned_names = {}
        if isinstance(self.server_names, dict):
            for raw_url, name in self.server_names.items():
                if not isinstance(raw_url, str) or not isinstance(name, str):
                    continue
                normalized_url = url_normalize_map.get(raw_url, raw_url.strip().rstrip("/"))
                name = name.strip()
                if name and normalized_url in seen_urls:
                    cleaned_names[normalized_url] = name
        self.server_names = cleaned_names
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
        self._upgrade_stale_default_prompts()
        self.system_prompt = self.system_prompt or DEFAULT_SYSTEM_PROMPT
        self.explain_hotkey = (self.explain_hotkey or "").strip()
        self.include_callees = bool(self.include_callees)
        try:
            self.max_trace_blocks = max(1, min(5000, int(self.max_trace_blocks)))
        except (TypeError, ValueError):
            self.max_trace_blocks = DEFAULT_CONFIG["max_trace_blocks"]
        for _color_key in ("cfg_trace_color_real", "cfg_trace_color_dead", "cfg_trace_color_unresolved"):
            try:
                setattr(self, _color_key, max(0, min(0xFFFFFF, int(getattr(self, _color_key)))))
            except (TypeError, ValueError):
                setattr(self, _color_key, DEFAULT_CONFIG[_color_key])
        self.cfg_trace_system_prompt = self.cfg_trace_system_prompt or CFG_TRACE_SYSTEM_PROMPT
        self.cfg_trace_use_symbolic = bool(self.cfg_trace_use_symbolic)
        self.cfg_trace_enumerate_computed_jumps = bool(self.cfg_trace_enumerate_computed_jumps)

    def _upgrade_stale_default_prompts(self):
        """Auto-adopt the current default for any stored prompt that is an
        UNMODIFIED default carried over from an older plugin version - the
        common "I updated the plugin but my old config still pins the old
        prompt, so none of the new SUGGESTED_* markers ever reach the
        model" trap. A stored prompt is treated as a stale default (and
        replaced) when its hash matches either the default hash recorded
        when THIS config was saved (save-time stamp, exact) or any default
        this plugin has ever shipped (frozen historical registry, for
        configs predating the stamp). A genuinely customized prompt
        matches neither and is left untouched."""
        self.system_prompt = self._maybe_adopt_default(
            self.system_prompt, DEFAULT_SYSTEM_PROMPT,
            getattr(self, "system_prompt_default_hash", ""),
            _KNOWN_DEFAULT_SYSTEM_PROMPT_HASHES, "system prompt",
        )
        self.cfg_trace_system_prompt = self._maybe_adopt_default(
            self.cfg_trace_system_prompt, CFG_TRACE_SYSTEM_PROMPT,
            getattr(self, "cfg_trace_prompt_default_hash", ""),
            _KNOWN_CFG_TRACE_PROMPT_HASHES, "CFG trace system prompt",
        )

    @staticmethod
    def _maybe_adopt_default(stored, current_default, saved_default_hash, known_hashes, label):
        if not stored or stored == current_default:
            return stored  # empty (default applied elsewhere) or already current
        h = _prompt_hash(stored)
        if (saved_default_hash and h == saved_default_hash) or h in known_hashes:
            ida_kernwin.msg(
                "[%s] Your saved %s was an older version's default; updating it to "
                "this version's default so new capabilities take effect.\n"
                % (PLUGIN_NAME, label)
            )
            return current_default
        return stored


# ---------------------------------------------------------------------------
# Networking (background thread, SSE streaming over urllib)
# ---------------------------------------------------------------------------

class _ServerConnectionError(RuntimeError):
    """Raised specifically for connection-level failures (refused, DNS,
    timeout, dropped mid-stream) and for HTTP 502/503/504, so callers can
    distinguish "this server is unreachable/overloaded" from other errors
    (bad request, model error) and fail over to another configured server
    instead of just reporting failure.
    """


class LlamaStreamWorker(threading.Thread):
    """Runs one chat-completion request against llama-server, streaming the
    answer via SSE. All callbacks are marshalled onto IDA's main thread with
    execute_sync; this thread never touches Qt widgets or the IDA database
    directly.
    """

    def __init__(self, config, server_url, messages, on_delta, on_reasoning_delta, on_done, on_error):
        super().__init__(daemon=True)
        self._config = config
        self._server_url = server_url
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
                is_connection_error = isinstance(exc, _ServerConnectionError)
                ida_kernwin.execute_sync(
                    functools.partial(self._on_error, str(exc), is_connection_error), ida_kernwin.MFF_FAST
                )

    def _stream(self):
        url = self._server_url + "/v1/chat/completions"
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
            if exc.code in (502, 503, 504):
                # Reachable (e.g. a reverse proxy answered) but the model
                # server itself is overloaded/unavailable - worth failing
                # over to another configured server, same as a dropped
                # connection.
                raise _ServerConnectionError("HTTP %s from %s: %s" % (exc.code, url, body)) from None
            raise RuntimeError("HTTP %s: %s" % (exc.code, body)) from None
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise _ServerConnectionError("Cannot connect to %s (%s)" % (url, reason)) from None

        parts = []
        reasoning_parts = []
        finish_reason = [None]
        try:
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
        except (urllib.error.URLError, OSError) as exc:
            if self._cancel_event.is_set():
                return
            raise _ServerConnectionError("Connection to %s lost while streaming (%s)" % (url, exc)) from None

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
    "suggested_var_types", "suggested_global_renames", "suggested_reanalyze",
    "root_is_pseudocode", "error",
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

    def __init__(self, config, func, server_url=None, on_delta=None, on_reasoning_delta=None, on_status=None):
        self.config = config
        self.func = func
        self.func_ea = func.start_ea
        # Priority order = config.server_urls list order (first = most
        # preferred). Snapshotted here so a mid-conversation config change
        # elsewhere can't shift priorities under an in-flight conversation.
        self._server_candidates = list(config.server_urls) or list(DEFAULT_CONFIG["server_urls"])
        self.server_url = server_url or self._server_candidates[0]
        self._failed_servers = set()
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
            self.config, self.server_url, list(self.messages),
            self._on_delta, self._on_reasoning_delta,
            self._on_worker_done, self._on_worker_error,
        )
        self.worker.start()

    def _next_fallback_server(self):
        """Highest-priority (list-order) configured server not yet tried
        this conversation, or None if every configured server has failed.
        """
        for candidate in self._server_candidates:
            if candidate not in self._failed_servers:
                return candidate
        return None

    def _on_worker_error(self, message, is_connection_error=False):
        self.worker = None
        if self._closed:
            return 0
        if is_connection_error and len(self._server_candidates) > 1:
            self._failed_servers.add(self.server_url)
            next_server = self._next_fallback_server()
            if next_server is not None:
                failed_from = self.server_url
                self.server_url = next_server
                self._on_status(
                    "%s is unreachable (%s) - falling back to %s..."
                    % (self.config.server_label(failed_from), message, self.config.server_label(next_server))
                )
                self._issue_request()
                return 0
            self._on_error_cb(
                "None of the %d configured llama-server(s) are reachable. Last error: %s"
                % (len(self._server_candidates), message)
            )
            return 0
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
                suggested_struct=None, suggested_var_types=[], suggested_global_renames=[],
                suggested_reanalyze=[],
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
                # conversation) - rejection is logged (rather than silently
                # dropped) since from the outside "the LLM proposed a name
                # but nothing happened" looks exactly like a bug otherwise.
                # Deliberately NOT gated on the callee's current name looking
                # auto-generated: the model may re-propose a better name for
                # a callee it (or a prior round) already renamed once it
                # understands the function better, and it's in a better
                # position than a fixed naming-pattern regex to judge
                # whether a fresh name is actually an improvement - trust
                # its judgment rather than hard-blocking every non-default
                # name.
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
                seen_callee_eas.add(callee_ea)
                suggested_callee_renames.append((callee_ea, callee_new_name))

        suggested_global_renames = []
        global_matches = _SUGGESTED_GLOBAL_NAME_RE.findall(text)
        if global_matches:
            text = _SUGGESTED_GLOBAL_NAME_RE.sub("", text).strip()
            seen_global_eas = set()
            for query, global_new_name_raw in global_matches:
                global_new_name = sanitize_suggested_name(global_new_name_raw)
                if not global_new_name:
                    ida_kernwin.msg(
                        "[%s] Ignored SUGGESTED_GLOBAL_NAME for '%s': '%s' "
                        "is not a valid identifier.\n"
                        % (PLUGIN_NAME, query, global_new_name_raw)
                    )
                    continue
                global_ea = resolve_global_query(query.strip())
                if global_ea == idaapi.BADADDR:
                    ida_kernwin.msg(
                        "[%s] Ignored SUGGESTED_GLOBAL_NAME: could not "
                        "resolve '%s' to a data/global variable.\n" % (PLUGIN_NAME, query)
                    )
                    continue
                if global_ea == self.func_ea or global_ea in seen_global_eas:
                    continue
                seen_global_eas.add(global_ea)
                suggested_global_renames.append((global_ea, global_new_name))

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

        suggested_reanalyze = []
        reanalyze_matches = _SUGGESTED_REANALYZE_RE.findall(text)
        if reanalyze_matches:
            text = _SUGGESTED_REANALYZE_RE.sub("", text).strip()
            seen_reanalyze_eas = set()
            for query, reason in reanalyze_matches:
                target = resolve_function_query(query.strip())
                if target is None:
                    ida_kernwin.msg(
                        "[%s] Ignored SUGGESTED_REANALYZE: could not resolve "
                        "'%s' to a function.\n" % (PLUGIN_NAME, query)
                    )
                    continue
                target_ea = target.start_ea
                if target_ea == self.func_ea or target_ea in seen_reanalyze_eas:
                    continue
                seen_reanalyze_eas.add(target_ea)
                suggested_reanalyze.append((target_ea, (reason or "").strip()))

        text = strip_markdown_fences(_REQUEST_CODE_RE.sub("", text).strip())

        self._on_result_cb(ConversationResult(
            text=text, reasoning_text=reasoning_text,
            suggested_name=suggested_name, suggested_signature=suggested_signature,
            suggested_vars=suggested_vars, suggested_callee_renames=suggested_callee_renames,
            suggested_struct=suggested_struct, suggested_var_types=suggested_var_types,
            suggested_global_renames=suggested_global_renames,
            suggested_reanalyze=suggested_reanalyze,
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
        self._suggested_global_renames = []

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

        globalrename_layout = QtWidgets.QHBoxLayout()
        self.globalrename_check = QtWidgets.QCheckBox("Rename global variable(s):")
        self.globalrename_check.setEnabled(False)
        globalrename_layout.addWidget(self.globalrename_check)
        self.globalrename_label = QtWidgets.QLineEdit()
        self.globalrename_label.setReadOnly(True)
        self.globalrename_label.setEnabled(False)
        globalrename_layout.addWidget(self.globalrename_label, 1)
        layout.addLayout(globalrename_layout)

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

        if result.suggested_global_renames:
            self._suggested_global_renames = result.suggested_global_renames
            global_labels = []
            for global_ea, global_new_name in result.suggested_global_renames:
                old_global_name = idc.get_name(global_ea) or ("%#x" % global_ea)
                global_labels.append("%s -> %s" % (old_global_name, global_new_name))
            self.globalrename_label.setText(", ".join(global_labels))
            self.globalrename_check.setEnabled(True)
            self.globalrename_label.setEnabled(True)
            self.globalrename_check.setChecked(True)

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
        comment = _SUGGESTED_GLOBAL_NAME_RE.sub("", comment)
        comment = _SUGGESTED_REANALYZE_RE.sub("", comment)
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
        global_renames = (
            list(self._suggested_global_renames)
            if (self.globalrename_check.isChecked() and self._suggested_global_renames) else None
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
                var_renames, callee_renames, struct_decl, var_types, global_renames,
            ),
            ida_kernwin.MFF_WRITE,
        )
        self.close()

    def closeEvent(self, event):
        self._closed = True
        self.runner.cancel()
        super().closeEvent(event)


def _rgb_hex_to_bgr_int(text, fallback):
    """Settings shows CFG-trace colors as plain '#RRGGBB' hex strings (the
    reading order everyone expects); idc.set_color's native representation
    is a 0xBBGGRR-packed int (BGR, not RGB) - convert on the way in/out so
    that trap stays contained to these two helpers.
    """
    text = (text or "").strip().lstrip("#")
    try:
        value = int(text, 16)
    except ValueError:
        return fallback
    if not (0 <= value <= 0xFFFFFF):
        return fallback
    r, g, b = (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF
    return (b << 16) | (g << 8) | r


def _bgr_int_to_rgb_hex(value):
    value = max(0, min(0xFFFFFF, int(value)))
    b, g, r = (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF
    return "#%02X%02X%02X" % (r, g, b)


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

        self.server_urls_edit = QtWidgets.QPlainTextEdit()
        self.server_urls_edit.setPlaceholderText(
            "One llama-server base URL per line, with an optional # name, e.g.\n"
            "http://127.0.0.1:8080  # Home GPU\n"
            "http://127.0.0.1:8081  # Office server"
        )
        self.server_urls_edit.setMaximumHeight(80)
        self.server_urls_edit.setToolTip(
            "List order is priority order, most preferred first. Add an "
            "optional '# name' after a URL to label it (shown instead of "
            "the raw URL in status messages and batch results). Batch "
            "Explain and the recursive auto-accept action distribute work "
            "across all of them at once (up to one function in flight per "
            "server); the interactive single-function explain uses the "
            "first one. If a server refuses/drops a connection or returns "
            "502/503/504, that conversation automatically retries against "
            "the next server on the list before giving up."
        )
        form.addRow("Server base URL(s):", self.server_urls_edit)

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

        self.max_trace_blocks_spin = QtWidgets.QSpinBox()
        self.max_trace_blocks_spin.setRange(1, 5000)
        self.max_trace_blocks_spin.setToolTip(
            "Safety cap on how many basic blocks 'Trace/Recover CFG' will "
            "decode in one run (every block counts, including auto-resolved "
            "ones). Hitting it stops with a partial result rather than "
            "running unbounded."
        )
        form.addRow("Max CFG trace blocks:", self.max_trace_blocks_spin)

        self.cfg_trace_use_symbolic_check = QtWidgets.QCheckBox(
            "Resolve CFG trace branches via constant propagation before asking the LLM"
        )
        self.cfg_trace_use_symbolic_check.setToolTip(
            "Tries a lightweight, deterministic symbolic-execution pass first "
            "(register/stack-slot constant tracking, x86/x64 only) at each "
            "branch or indirect jump - resolves classic opaque predicates "
            "and flattening dispatchers, and recognizes genuine "
            "data-dependent branches (marking every side real), without an "
            "LLM round-trip. Falls back to the LLM whenever it cannot "
            "confidently resolve something; never guesses. Disable to force "
            "every decision point through the LLM as before."
        )
        form.addRow(self.cfg_trace_use_symbolic_check)

        self.cfg_trace_enumerate_computed_jumps_check = QtWidgets.QCheckBox(
            "Enumerate ARM64 computed jump tables (experimental)"
        )
        self.cfg_trace_enumerate_computed_jumps_check.setToolTip(
            "When a trace hits an AArch64 'BR/BLR Xn' whose target is "
            "computed as *(table_base + index*stride + field) - a struct/"
            "vtable-style dispatch where the index is runtime data - walk "
            "the table in the loaded image and treat each stored pointer "
            "that lands on executable code as a real successor, so the "
            "dispatcher's handlers get explored. Experimental and OFF by "
            "default: the index range can't be bounded statically, so it "
            "walks entries until the first one that doesn't point at code, "
            "which may under- or over-shoot. Every recovered target is "
            "still shown for review before anything is written or patched."
        )
        form.addRow(self.cfg_trace_enumerate_computed_jumps_check)

        color_row = QtWidgets.QHBoxLayout()
        self.cfg_trace_color_real_edit = QtWidgets.QLineEdit()
        self.cfg_trace_color_real_edit.setPlaceholderText("#RRGGBB")
        self.cfg_trace_color_real_edit.setMaximumWidth(90)
        color_row.addWidget(QtWidgets.QLabel("Real:"))
        color_row.addWidget(self.cfg_trace_color_real_edit)
        self.cfg_trace_color_dead_edit = QtWidgets.QLineEdit()
        self.cfg_trace_color_dead_edit.setPlaceholderText("#RRGGBB")
        self.cfg_trace_color_dead_edit.setMaximumWidth(90)
        color_row.addWidget(QtWidgets.QLabel("Dead:"))
        color_row.addWidget(self.cfg_trace_color_dead_edit)
        self.cfg_trace_color_unresolved_edit = QtWidgets.QLineEdit()
        self.cfg_trace_color_unresolved_edit.setPlaceholderText("#RRGGBB")
        self.cfg_trace_color_unresolved_edit.setMaximumWidth(90)
        color_row.addWidget(QtWidgets.QLabel("Unresolved:"))
        color_row.addWidget(self.cfg_trace_color_unresolved_edit)
        color_row.addStretch(1)
        form.addRow("CFG trace colors:", color_row)

        self.cfg_trace_system_prompt_edit = QtWidgets.QPlainTextEdit()
        self.cfg_trace_system_prompt_edit.setMinimumHeight(140)
        form.addRow("CFG trace system prompt:", self.cfg_trace_system_prompt_edit)

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
        lines = []
        for url in config.server_urls:
            name = (config.server_names or {}).get(url)
            lines.append("%s  # %s" % (url, name) if name else url)
        self.server_urls_edit.setPlainText("\n".join(lines))
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
        self.max_trace_blocks_spin.setValue(config.max_trace_blocks)
        self.cfg_trace_use_symbolic_check.setChecked(config.cfg_trace_use_symbolic)
        self.cfg_trace_enumerate_computed_jumps_check.setChecked(config.cfg_trace_enumerate_computed_jumps)
        self.cfg_trace_color_real_edit.setText(_bgr_int_to_rgb_hex(config.cfg_trace_color_real))
        self.cfg_trace_color_dead_edit.setText(_bgr_int_to_rgb_hex(config.cfg_trace_color_dead))
        self.cfg_trace_color_unresolved_edit.setText(_bgr_int_to_rgb_hex(config.cfg_trace_color_unresolved))
        self.cfg_trace_system_prompt_edit.setPlainText(config.cfg_trace_system_prompt)

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
        server_urls = []
        server_names = {}
        for line in self.server_urls_edit.toPlainText().splitlines():
            line = line.strip()
            if not line:
                continue
            url_part, _, name_part = line.partition("#")
            url_part = url_part.strip()
            name_part = name_part.strip()
            if not url_part:
                continue
            server_urls.append(url_part)
            if name_part:
                server_names[url_part] = name_part
        cfg.server_urls = server_urls
        cfg.server_names = server_names
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
        cfg.max_trace_blocks = self.max_trace_blocks_spin.value()
        cfg.cfg_trace_use_symbolic = self.cfg_trace_use_symbolic_check.isChecked()
        cfg.cfg_trace_enumerate_computed_jumps = self.cfg_trace_enumerate_computed_jumps_check.isChecked()
        cfg.cfg_trace_color_real = _rgb_hex_to_bgr_int(
            self.cfg_trace_color_real_edit.text(), cfg.cfg_trace_color_real
        )
        cfg.cfg_trace_color_dead = _rgb_hex_to_bgr_int(
            self.cfg_trace_color_dead_edit.text(), cfg.cfg_trace_color_dead
        )
        cfg.cfg_trace_color_unresolved = _rgb_hex_to_bgr_int(
            self.cfg_trace_color_unresolved_edit.text(), cfg.cfg_trace_color_unresolved
        )
        cfg.cfg_trace_system_prompt = self.cfg_trace_system_prompt_edit.toPlainText()
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
    "suggested_global_renames", "suggested_reanalyze", "root_is_pseudocode",
])


def _compute_apply_args(item, allow_rename_named=False):
    """Given a BatchItemResult, compute the positional args tuple for
    _apply_suggestions_and_refresh, using the same conservative defaults
    used throughout the plugin (rename only when the current name looks
    auto-generated; signature/variable/struct suggestions only when the
    root context was Hex-Rays pseudocode). Returns None if there's nothing
    to apply (missing or failed result).

    allow_rename_named=True lifts the "only rename auto-generated names"
    guard for THIS function's own name - used for functions pulled in by a
    SUGGESTED_REANALYZE request, where the whole point is that the model
    thinks the existing (non-default) name is wrong and should be replaced.
    """
    if item is None or not item.ok:
        return None
    new_name = item.suggested_name if (
        item.suggested_name and (allow_rename_named or is_auto_generated_name(item.orig_name))
    ) else None
    signature = item.suggested_signature if item.root_is_pseudocode else None
    var_renames = item.suggested_vars if (item.root_is_pseudocode and item.suggested_vars) else None
    callee_renames = item.suggested_callee_renames or None
    struct_decl = item.suggested_struct if item.root_is_pseudocode else None
    var_types = item.suggested_var_types if (item.root_is_pseudocode and item.suggested_var_types) else None
    global_renames = item.suggested_global_renames or None
    return (item.func_ea, item.comment, new_name, signature, var_renames, callee_renames,
            struct_decl, var_types, global_renames)


class BatchController(object):
    """Drives functions through ConversationRunners on IDA's main thread,
    as a worker pool over config.server_urls: up to one function in flight
    per configured server, so N servers process up to N functions at once.
    No separate background thread is needed for the pool itself:
    ConversationRunner.start() already returns immediately (spawning a
    LlamaStreamWorker thread per request), and that worker's completion
    callbacks already arrive on the main thread via execute_sync. As soon
    as a server's function finishes, the next queued function (if any) is
    immediately dispatched to that now-free server.
    """

    def __init__(self, config, funcs, on_row_update, on_finished, on_item_result=None,
                 on_row_add=None, max_extra=0):
        self.config = config
        self.funcs = funcs
        self._on_row_update = on_row_update
        self._on_finished = on_finished
        self._on_item_result = on_item_result or (lambda item: None)
        # Called when a function is appended to the queue AFTER start (a
        # SUGGESTED_REANALYZE expansion) so the dialog can add a row for it.
        self._on_row_add = on_row_add or (lambda index, func: None)
        self._servers = list(config.server_urls) or list(DEFAULT_CONFIG["server_urls"])
        self._next_index = 0
        self._cancelled = False
        self._finished = False
        # server_url -> (index, ConversationRunner) for in-flight requests -
        # at most one per server, so this also caps concurrency at len(servers).
        self._active = {}
        self.results = {}
        # Every ea ever in the queue (initial + dynamically added), so a
        # reanalyze request never double-queues a function.
        self._known_eas = {f.start_ea for f in funcs}
        # Budget for dynamic SUGGESTED_REANALYZE additions, so a chain of
        # reanalyze requests can't make the run grow without bound.
        self._extra_budget = max(0, int(max_extra))

    def request_reanalysis(self, func):
        """Append a function discovered via SUGGESTED_REANALYZE to the
        queue (if not already present and budget remains). Safe to call
        from an on_item_result callback: it runs before this controller's
        _dispatch_next for the just-freed server, so the newly appended
        function is picked up on the next dispatch without any extra
        kicking. Returns True if it was actually queued."""
        if self._cancelled or self._finished or self._extra_budget <= 0:
            return False
        if func is None or func.start_ea in self._known_eas:
            return False
        self._extra_budget -= 1
        self._known_eas.add(func.start_ea)
        index = len(self.funcs)
        self.funcs.append(func)
        self._on_row_add(index, func)
        return True

    def start(self):
        for server_url in self._servers:
            self._dispatch_next(server_url)

    def cancel(self):
        """Cancelling an in-flight LlamaStreamWorker suppresses BOTH its
        on_done and on_error callbacks (both gated by the worker's own
        cancel_event check), so no completion callback will ever arrive
        for any in-flight function. This must therefore mark every active
        and remaining row as Cancelled synchronously, rather than waiting
        for callbacks that will never come.
        """
        if self._cancelled:
            return
        self._cancelled = True
        for index, runner in self._active.values():
            runner.cancel()
            self._on_row_update(index, "Cancelled", "")
        self._active = {}
        for i in range(self._next_index, len(self.funcs)):
            self._on_row_update(i, "Cancelled", "")
        self._maybe_finish()

    def _dispatch_next(self, server_url):
        if self._cancelled:
            return
        if self._next_index >= len(self.funcs):
            self._maybe_finish()
            return
        index = self._next_index
        self._next_index += 1
        func = self.funcs[index]
        self._on_row_update(index, "Running", "(%s)" % self.config.server_label(server_url))
        runner = ConversationRunner(
            self.config, func, server_url=server_url,
            on_status=functools.partial(self._on_status, index),
        )
        self._active[server_url] = (index, runner)
        runner.start(
            on_result=functools.partial(self._on_result, index, func, server_url),
            on_error=functools.partial(self._on_error, index, func, server_url),
        )

    def _on_status(self, index, text):
        self._on_row_update(index, "Running", text)

    def _maybe_finish(self):
        if self._finished:
            return
        if not self._active and (self._cancelled or self._next_index >= len(self.funcs)):
            self._finished = True
            self._on_finished()

    def _record_and_advance(self, item, server_url):
        self.results[item.func_ea] = item
        self._on_item_result(item)
        self._active.pop(server_url, None)
        self._dispatch_next(server_url)

    def _on_result(self, index, func, server_url, result):
        orig_name = ida_funcs.get_func_name(func.start_ea) or ("sub_%X" % func.start_ea)
        label = self.config.server_label(server_url)
        if result.error:
            item = BatchItemResult(func.start_ea, orig_name, False, result.error,
                                    None, None, None, [], [], None, [], [], [], result.root_is_pseudocode)
            self._on_row_update(index, "Error", "[%s] %s" % (label, result.error))
        else:
            item = BatchItemResult(func.start_ea, orig_name, True, None, result.text,
                                    result.suggested_name, result.suggested_signature,
                                    result.suggested_vars, result.suggested_callee_renames,
                                    result.suggested_struct, result.suggested_var_types,
                                    result.suggested_global_renames, result.suggested_reanalyze,
                                    result.root_is_pseudocode)
            self._on_row_update(index, "Done", "[%s] %s" % (label, result.text[:120]))
        self._record_and_advance(item, server_url)

    def _on_error(self, index, func, server_url, message):
        orig_name = ida_funcs.get_func_name(func.start_ea) or ("sub_%X" % func.start_ea)
        item = BatchItemResult(func.start_ea, orig_name, False, message,
                                None, None, None, [], [], None, [], [], [], False)
        self._on_row_update(index, "Error", "[%s] %s" % (self.config.server_label(server_url), message))
        self._record_and_advance(item, server_url)


def _apply_batch_and_refresh(items):
    """items: list of (func_ea, comment, new_name, signature, var_renames,
    callee_renames, struct_decl, var_types, global_renames). One
    execute_sync/MFF_WRITE round-trip for the whole batch instead of N.
    """
    for (func_ea, comment, new_name, signature, var_renames, callee_renames,
         struct_decl, var_types, global_renames) in items:
        _apply_suggestions_and_refresh(
            func_ea, comment, new_name, signature, var_renames, callee_renames,
            struct_decl, var_types, global_renames,
        )
    return 1


class BatchProgressDialog(QtWidgets.QDialog):
    """Table: [Apply checkbox | Function | New Name | Status | Comment/Error
    preview].

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

    COL_APPLY, COL_FUNC, COL_NEW_NAME, COL_STATUS, COL_PREVIEW = range(5)

    def __init__(self, config, funcs, parent=None, auto_apply=False):
        super().__init__(parent)
        self.auto_apply = auto_apply
        title = "Recursive Explain (auto-accept)" if auto_apply else "Batch Explain Progress"
        self.setWindowTitle("%s - %s" % (PLUGIN_NAME, title))
        self.resize(720, 480)
        self.funcs = funcs
        self._row_by_ea = {f.start_ea: i for i, f in enumerate(funcs)}
        # eas pulled in dynamically by a SUGGESTED_REANALYZE request - their
        # apply is allowed to replace an existing non-default name (that's
        # the whole point of re-analyzing them). See _on_item_auto_apply.
        self._reanalysis_eas = set()

        self.table = QtWidgets.QTableWidget(len(funcs), 5)
        apply_header = "Applied" if auto_apply else "Apply"
        self.table.setHorizontalHeaderLabels(
            [apply_header, "Function", "New Name", "Status", "Comment / Error"])
        self.table.horizontalHeader().setStretchLastSection(True)
        for i, func in enumerate(funcs):
            name = ida_funcs.get_func_name(func.start_ea) or ("sub_%X" % func.start_ea)
            check_item = QtWidgets.QTableWidgetItem()
            if not auto_apply:
                check_item.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            check_item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.table.setItem(i, self.COL_APPLY, check_item)
            self.table.setItem(i, self.COL_FUNC, QtWidgets.QTableWidgetItem("%s @ %#x" % (name, func.start_ea)))
            self.table.setItem(i, self.COL_NEW_NAME, QtWidgets.QTableWidgetItem(""))
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
            on_item_result=self._on_item_result,
            # Only the recursive auto-accept scan expands itself via
            # SUGGESTED_REANALYZE, bounded by the same recursive-callee cap.
            on_row_add=self._on_row_add_dynamic if auto_apply else None,
            max_extra=config.max_recursive_callees if auto_apply else 0,
        )
        self.controller.start()

    def _on_row_update(self, index, status, preview):
        self.table.item(index, self.COL_STATUS).setText(status)
        if preview:
            self.table.item(index, self.COL_PREVIEW).setText(preview)

    def _on_row_add_dynamic(self, index, func):
        """A SUGGESTED_REANALYZE request appended `func` to the controller's
        queue (recursive mode only) - add a matching table row so its
        progress shows, and remember it so its apply may replace an
        existing non-default name. `index` mirrors the controller's own
        row index (table rows stay 1:1 with controller.funcs)."""
        self._reanalysis_eas.add(func.start_ea)
        name = ida_funcs.get_func_name(func.start_ea) or ("sub_%X" % func.start_ea)
        self.table.insertRow(index)
        check_item = QtWidgets.QTableWidgetItem()
        check_item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.table.setItem(index, self.COL_APPLY, check_item)
        self.table.setItem(index, self.COL_FUNC,
                           QtWidgets.QTableWidgetItem("%s @ %#x  (reanalyze)" % (name, func.start_ea)))
        self.table.setItem(index, self.COL_NEW_NAME, QtWidgets.QTableWidgetItem(""))
        self.table.setItem(index, self.COL_STATUS, QtWidgets.QTableWidgetItem("Pending"))
        self.table.setItem(index, self.COL_PREVIEW, QtWidgets.QTableWidgetItem(""))
        self._row_by_ea[func.start_ea] = index

    def _on_item_result(self, item):
        """Runs for every finished function in BOTH modes: fill in the
        New Name column with the model's suggested name (annotated when
        the conservative rename guard would keep the existing name), then
        hand off to the auto-apply step in recursive mode."""
        row = self._row_by_ea.get(item.func_ea)
        if row is not None and item.ok and item.suggested_name:
            text = item.suggested_name
            allow_named = self.auto_apply and item.func_ea in self._reanalysis_eas
            if (item.suggested_name != item.orig_name
                    and not allow_named and not is_auto_generated_name(item.orig_name)):
                text += "  (kept: %s)" % item.orig_name
            self.table.item(row, self.COL_NEW_NAME).setText(text)
        if self.auto_apply:
            self._on_item_auto_apply(item)

    def _on_item_auto_apply(self, item):
        # A function pulled in by SUGGESTED_REANALYZE is allowed to have its
        # existing (non-default) name replaced - that's why it was re-queued.
        allow_named = item.func_ea in self._reanalysis_eas
        args = _compute_apply_args(item, allow_rename_named=allow_named)
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
        # Expand the recursive walk with any already-named callees the model
        # flagged for re-analysis (bounded by the controller's own budget;
        # duplicates/self are dropped there).
        for target_ea, reason in (item.suggested_reanalyze or []):
            func = ida_funcs.get_func(target_ea)
            if func is not None and self.controller.request_reanalysis(func):
                self._log_reanalyze(target_ea, reason)

    def _log_reanalyze(self, target_ea, reason):
        name = ida_funcs.get_func_name(target_ea) or ("sub_%X" % target_ea)
        detail = (" - %s" % reason) if reason else ""
        ida_kernwin.msg(
            "[%s] Re-analyzing %s at the model's request%s\n" % (PLUGIN_NAME, name, detail)
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
# CFG trace: LLM-guided worklist driver (no Qt/DB writes - mark/comment
# only happens later, in the review-step apply function)
# ---------------------------------------------------------------------------

def _parse_trace_address(token):
    """Parses a REAL_TARGET/DEAD_TARGET/UNRESOLVED_TARGET address token.
    Returns an int address, or None for a non-address token (e.g.
    "indirect") or anything that doesn't parse.
    """
    token = (token or "").strip().strip("`'\"")
    if not token or token.lower() in ("indirect", "unknown", "n/a", "none"):
        return None
    for base in (0, 16):
        try:
            return int(token, base)
        except ValueError:
            continue
    return None


# verdict: "real" | "dead" | "unresolved". insn_eas is the exact set of
# instruction addresses this record covers, used for per-instruction
# coloring/commenting on apply - no second decode pass needed.
CfgTraceBlockRecord = namedtuple("CfgTraceBlockRecord", ["start_ea", "end_ea", "insn_eas", "verdict", "reason"])

CfgTraceResult = namedtuple(
    "CfgTraceResult",
    ["blocks", "partial", "blocks_processed", "unexplored", "error", "symbolic_resolved_count", "jump_patches"],
)

# Cached in LLMExplainerPlugin._trace_cache, keyed by the trace's start
# address, so reopening Trace/Recover CFG on the same address can jump
# straight to the review page (transcript + table) without re-running
# anything - review-only or Accept still work identically off of it,
# since both just read runner/result, same as a freshly-finished trace.
# In-memory only (cleared on plugin term()/IDA close) - not persisted to
# the IDB, since pickling BlockInfo/SymInsn trees across code changes
# would be fragile; the cost of that is a cold cache after restarting IDA,
# which is a fair tradeoff against locking in a serialization format.
_CachedCfgTrace = namedtuple("_CachedCfgTrace", ["runner", "result", "timestamp"])

# Cached in LLMExplainerPlugin._applied_cfg_patches, keyed by the trace's
# start address, recording exactly which byte ranges "Patch in place" or
# "Rebuild linear" touched when last accepted there - enough to drive
# Undo Patches (see _undo_applied_cfg_patch) without needing to separately
# remember each byte's original value: IDA already tracks that itself for
# every patched byte (ida_bytes.revert_byte reads it back directly), this
# only needs to know WHICH addresses were touched by THIS plugin's own
# patch, so Undo never reverts something else entirely (e.g. a manual
# patch the user made elsewhere in the same range via IDA's own tools).
# touched_ranges: list of (start_ea, end_ea) byte ranges. In-memory only,
# same tradeoff as _CachedCfgTrace above.
_AppliedCfgPatch = namedtuple("_AppliedCfgPatch", ["mode", "touched_ranges", "timestamp"])


def _undo_applied_cfg_patch(applied):
    """Reverts every byte in applied.touched_ranges back to its original
    value via ida_bytes.revert_byte (IDA's own recorded pre-patch byte,
    not anything this plugin stored itself), then re-analyzes each range
    so the disassembly reflects the reverted bytes. Returns the number of
    bytes actually reverted (a byte with nothing to revert - e.g. already
    reverted by hand via Edit > Patches - is simply skipped, not an error).
    """
    reverted = 0
    for start, end in applied.touched_ranges:
        for ea in range(start, end):
            try:
                if ida_bytes.revert_byte(ea):
                    reverted += 1
            except Exception:
                pass
        try:
            ida_bytes.del_items(start, ida_bytes.DELIT_SIMPLE, end - start)
        except Exception:
            pass
        try:
            ida_auto.plan_and_wait(start, end)
        except Exception:
            pass
    return reverted

# A fully-resolved branch found during the trace, eligible for in-place
# byte patching. Two distinct cases feed this, both produced by
# CfgTraceRunner._compute_jump_patches:
#   - A conditional_branch block whose two successors have opposite
#     verdicts (one real, one dead) - a fully-resolved binary opaque
#     predicate. role_of_real: "jump_target" | "fallthrough" - which side
#     of the original Jcc/B.cond is the real one. dead_target_ea is the
#     other side's address (used for the "both sides checked" UI gate).
#   - An indirect_jump block (e.g. "BR X8" fed by a computed/table
#     pointer load) with exactly one confirmed-real successor and no
#     live alternative - the indirection always resolves to the same
#     fixed target, so it can be replaced outright with a direct branch.
#     role_of_real: "only". dead_target_ea is None (there is no opposite
#     side to require checked).
# NOT emitted for genuine data-dependent branches (both real), a live
# multi-case indirect dispatch (2+ real successors), or anything
# involving an unresolved successor - see CfgTraceRunner._build_result.
CfgTraceJumpPatch = namedtuple(
    "CfgTraceJumpPatch", ["jcc_ea", "real_target_ea", "dead_target_ea", "role_of_real"]
)


class CfgTraceRunner(object):
    """UI-free driver of one LLM-guided CFG-recovery trace, starting from a
    single address. Walks basic blocks via gather_block(), auto-continuing
    through anything with exactly one successor, and only round-trips to
    the LLM at genuine decision points (conditional branches / indirect
    jumps) - so LLM call count is proportional to real obfuscation
    complexity, not total block count. Reuses ConversationRunner's
    priority-order server-failover pattern. Nothing is written to the
    database here; callers get a CfgTraceResult to review before applying.
    """

    def __init__(self, config, start_ea, on_log=None, on_status=None):
        self.config = config
        self.start_ea = start_ea
        self._on_log = on_log or (lambda text: None)
        self._on_status = on_status or (lambda text: None)
        self._server_candidates = list(config.server_urls) or list(DEFAULT_CONFIG["server_urls"])
        self.server_url = self._server_candidates[0]
        self._failed_servers = set()

        self.worklist = deque([start_ea])
        self.queued = {start_ea}
        self.visited = {}          # ea -> BlockInfo, confidently decoded + reached via a REAL predecessor
        self.dead = set()          # eas confirmed dead
        self.dead_blocks = {}      # ea -> BlockInfo, marking-only extent for coloring
        self.dead_reason = {}      # ea -> reason text
        self.decision_summary = {} # ea -> one-line "why this was enqueued" text (keyed like predecessors)
        self.predecessors = {}     # ea -> predecessor ea, for breadcrumb reconstruction
        self.unresolved = []       # list of dicts, see _flag_unresolved/_apply_decision
        self._unresolved_eas = set()
        self.trace_log = []
        self.blocks_processed = 0
        self.symbolic_resolved_count = 0  # blocks resolved without an LLM call, for the summary banner
        # eas of decision-point blocks (conditional_branch/indirect_jump)
        # already corrected once for being re-entered from a different
        # real predecessor than the one that originally decided them -
        # see _enqueue's dispatcher-revisit handling.
        self._corrected_dispatchers = set()
        # ea -> instruction size, for every instruction any gather_block()
        # call has claimed and fixed up so far in this trace (real, dead,
        # or marking-only) - shared across every gather_block() call below
        # so a later target landing in the MIDDLE of an already-claimed
        # instruction (an overlapping-instruction obfuscation trick) is
        # detected and refused, rather than silently destroying the
        # earlier block's already-established boundary. See gather_block
        # and _find_overlapping_claim.
        self.claimed_insns = {}

        # ea -> SymState as of the end of that block's predecessor (i.e.
        # the state to run THIS block's own instructions against). Seeded
        # for the trace's start address (every register externally-tainted
        # - see sym_initial_state - since at any given start address every
        # register genuinely holds a value from outside this trace, most
        # commonly an incoming argument); every _enqueue() call propagates
        # one for its target, whether reached via auto-continue, a
        # symbolic-resolved verdict, or an LLM verdict - see _enqueue.
        self.sym_states = {start_ea: sym_initial_state()}

        self.worker = None
        self._closed = False
        self._on_finished_cb = None
        self._current_block = None
        self._current_block_sym_state = None

    # -- driving ------------------------------------------------------

    def start(self, on_finished):
        self._on_finished_cb = on_finished
        # gather_block() reads and now also writes the database (instruction
        # boundary fixups - see _fixup_instruction_boundary), so _pump()
        # must run under MFF_WRITE, never called directly from a network
        # callback context (see _on_decision_done/_on_decision_error below).
        ida_kernwin.execute_sync(self._pump, ida_kernwin.MFF_WRITE)

    def cancel(self):
        self._closed = True
        if self.worker is not None:
            self.worker.cancel()

    def _log(self, text):
        self.trace_log.append(text)
        self._on_log(text)

    def _pump(self):
        while True:
            if self._closed:
                return 0
            if not self.worklist:
                self._finish(partial=False)
                return 0
            if self.blocks_processed >= self.config.max_trace_blocks:
                self._log(
                    "Reached the max trace blocks cap (%d); stopping with a partial "
                    "result. %d address(es) remain unexplored." % (self.config.max_trace_blocks, len(self.worklist))
                )
                self._finish(partial=True)
                return 0
            ea = self.worklist.popleft()
            if ea in self.visited or ea in self.dead:
                continue
            # Note: deliberately NOT skipping ea in self._unresolved_eas here.
            # worklist only ever contains addresses enqueued via _enqueue()
            # (i.e. a REAL verdict) - a later conflicting DEAD/UNRESOLVED
            # verdict for the same address flags it unresolved for review
            # (first verdict wins) but must not stop it from actually being
            # traced as real, or the conflict-flagged address would be
            # silently dropped from the recovered path entirely.
            self.blocks_processed += 1
            block = gather_block(ea, claimed_insns=self.claimed_insns)
            self._log("Block %#010x (%s, %d instr)" % (ea, block.kind, len(block.insn_eas)))

            if block.kind in ("undecodable", "truncated", "overlap"):
                if block.kind == "undecodable":
                    note = "Address did not decode as a valid instruction."
                elif block.kind == "truncated":
                    note = "Hit the per-block instruction safety cap while decoding; needs manual review."
                else:
                    note = (
                        "This address lands in the middle of an instruction a different block in "
                        "this trace already claimed - likely an overlapping-instruction obfuscation "
                        "trick (the same bytes decode differently depending on the start offset). "
                        "Stopped here without touching those bytes; needs manual review to determine "
                        "which interpretation is real."
                    )
                self._flag_unresolved(ea, note, self.predecessors.get(ea), block=block)
                continue

            self.visited[ea] = block

            if block.kind == "return":
                continue
            if block.kind == "unconditional_jump":
                target = block.successors[0].ea
                if target is not None:
                    sym_state = sym_run_block(self.sym_states.get(ea, SymState()), block.sym_insns, _family_of("rsp"))
                    self._enqueue(target, ea, "auto-continue (only successor)", sym_state)
                continue
            if (
                block.kind == "indirect_jump"
                and len(block.successors) == 1
                and block.successors[0].ea is not None
            ):
                # A statically PRE-resolved single-target indirect jump -
                # e.g. an ARM64 fixed-pointer veneer resolved in
                # _resolve_indirect_jump_successors, or a one-case switch.
                # The destination is already known from the block's
                # successors, so treat it like an unconditional jump there
                # instead of handing it to the symbolic engine (which
                # can't model a load-from-memory and would return
                # "unknown") or spending an LLM round-trip re-deriving it.
                # This also makes it eligible for the "collapse to a
                # direct branch" byte patch - see _compute_jump_patches.
                target = block.successors[0].ea
                sym_state = sym_run_block(self.sym_states.get(ea, SymState()), block.sym_insns, _family_of("rsp"))
                self._enqueue(target, ea, "resolved single-target indirect jump", sym_state)
                continue

            # Genuine decision point: conditional_branch or indirect_jump.
            # Try to resolve it via constant propagation first - this is
            # what keeps LLM call count proportional to real obfuscation
            # complexity rather than total decision-point count; falls
            # back to the LLM whenever it cannot confidently resolve.
            self._current_block = block
            sym_state = sym_run_block(self.sym_states.get(ea, SymState()), block.sym_insns, _family_of("rsp"))
            self._current_block_sym_state = sym_state
            if self._try_symbolic_resolve(block, sym_state):
                continue
            if self._try_enumerate_computed_dispatch(block, sym_state):
                continue
            self._issue_decision_request(block)
            return 0

    def _try_enumerate_computed_dispatch(self, block, sym_state):
        """Opt-in (config.cfg_trace_enumerate_computed_jumps) handling for a
        computed ARM64 table dispatch that neither IDA nor the symbolic
        engine could resolve to fixed targets: statically walk the table
        (see _try_enumerate_arm64_table_dispatch) and, if it yields any
        destinations, mark them all real and continue - no LLM round-trip.
        Returns True if it handled the block. A genuinely runtime-indexed
        dispatch has many real successors, so it is deliberately NOT
        eligible for the 'collapse to one direct branch' byte patch (that
        only fires for a single real successor - see _compute_jump_patches);
        this just recovers the handlers for exploration/coloring.
        """
        if not self.config.cfg_trace_enumerate_computed_jumps:
            return False
        if block.kind != "indirect_jump" or not block.insn_eas:
            return False
        try:
            targets = _try_enumerate_arm64_table_dispatch(block.last_insn_ea, block.insn_eas)
        except Exception as exc:
            self._log("Computed-jump enumeration errored at %#010x (%s); deferring to the LLM." % (block.start_ea, exc))
            return False
        if not targets:
            return False
        self.symbolic_resolved_count += 1
        self._log(
            "Block %#010x: enumerated %d target(s) of a computed jump table "
            "statically - no LLM call needed." % (block.start_ea, len(targets))
        )
        reason = "[table] computed jump-table dispatch enumerated from the loaded image"
        for target in targets:
            # Each handler starts a fresh state - the dispatch index is
            # runtime data, so nothing from this block's state carries
            # meaningfully into a specific handler.
            self._apply_target_verdict(target, "real", reason, block.start_ea, sym_state_override=SymState())
        return True

    def _try_symbolic_resolve(self, block, sym_state):
        """Attempts to resolve a decision point without an LLM call, via
        the lightweight symbolic engine. Returns True (and has already
        applied verdicts + advanced the worklist) if resolved; False if
        the caller should fall back to the LLM as usual.
        """
        if not self.config.cfg_trace_use_symbolic:
            return False
        has_back_edge = any(
            s.ea is not None and self._is_ancestor(s.ea, block.start_ea) for s in block.successors
        )
        refined_states = {}
        if block.kind == "conditional_branch":
            if len(block.successors) != 2 or not block.sym_insns:
                return False
            last_mnem = block.sym_insns[-1].mnem
            kind, verdicts = sym_resolve_conditional_branch(
                sym_state, last_mnem, block.successors, _family_of("rcx")
            )
            # Per-edge state narrowing (see _compute_edge_refinements): a
            # genuinely correct no-op for anything neither of its two
            # refinements targets, so always safe to compute here
            # regardless of "kind".
            refined_states = _compute_edge_refinements(sym_state, last_mnem, block.successors)
        elif block.kind == "indirect_jump":
            if not block.sym_insns or not block.sym_insns[-1].operands:
                return False
            target_op = block.sym_insns[-1].operands[0]
            kind, verdicts = sym_resolve_indirect_jump(sym_state, target_op, block.successors)
        else:
            return False
        if kind == "unknown":
            return False
        if kind == "opaque" and has_back_edge:
            # The engine computed a single-snapshot "some real, some dead"
            # verdict, but one candidate is a loop back edge - a value
            # that looks fixed on THIS pass through the loop can still
            # differ on other iterations (the classic OLLVM dispatcher
            # pattern: re-entered every iteration with different state).
            # The engine has no notion of loop iterations, so this shape
            # of result can't be trusted here - defer to the LLM, which
            # will see the back edge explicitly annotated in the prompt.
            # (A "data_dependent" result, marking every side real, is
            # always safe regardless of back edges and is NOT blocked
            # here - only "opaque", which marks some candidates dead.)
            self._log(
                "Block %#010x: symbolic engine would mark some candidates dead, but "
                "one is a loop back edge - deferring to the LLM instead of trusting "
                "a single-iteration snapshot." % block.start_ea
            )
            return False
        self.symbolic_resolved_count += 1
        self._log(
            "Block %#010x resolved automatically via symbolic execution (%s) - no LLM call needed."
            % (block.start_ea, "genuine data-dependent branch" if kind == "data_dependent" else "opaque predicate")
        )
        reason = (
            "[symbolic] genuinely data-dependent branch - both paths are real"
            if kind == "data_dependent" else "[symbolic] resolved via constant propagation"
        )
        for succ in block.successors:
            if succ.ea is None or succ.ea not in verdicts:
                continue
            verdict = "real" if verdicts[succ.ea] else "dead"
            self._apply_target_verdict(
                succ.ea, verdict, reason, block.start_ea, sym_state_override=refined_states.get(succ.ea),
            )
        return True

    def _enqueue(self, ea, predecessor_ea, summary, sym_state=None):
        if ea in self.visited:
            self._note_dispatcher_revisit(ea, predecessor_ea)
            return
        if ea in self.queued or ea in self.dead:
            return
        self.queued.add(ea)
        self.predecessors[ea] = predecessor_ea
        self.decision_summary[ea] = summary
        self.sym_states[ea] = sym_state.copy() if sym_state is not None else SymState()
        self.worklist.append(ea)

    def _note_dispatcher_revisit(self, ea, predecessor_ea):
        """A REAL edge just targeted a block that's already fully decided -
        normal for an ordinary CFG merge point. But if that block is
        itself a decision point (conditional_branch/indirect_jump) and
        this edge comes from a DIFFERENT predecessor than the one that
        originally resolved it, this is the classic loop/dispatcher
        revisit pattern - e.g. an OLLVM flattening dispatcher re-entered
        on every loop iteration with a different state value. Whatever
        that block's own DEAD verdicts were (from resolving it against
        the FIRST entry's snapshot only) may not hold for this entry -
        retroactively re-flag them for manual review rather than
        continuing to silently trust a verdict that only ever considered
        one specific path through the dispatcher. This is exactly the
        failure mode where a genuinely reachable block ends up
        permanently mismarked dead.
        """
        if ea in self._corrected_dispatchers:
            return
        block = self.visited.get(ea)
        if block is None or block.kind not in ("conditional_branch", "indirect_jump"):
            return
        original_predecessor = self.predecessors.get(ea)
        if original_predecessor is None or original_predecessor == predecessor_ea:
            return  # same entry point (or no recorded predecessor) - not a revisit
        self._corrected_dispatchers.add(ea)
        self._log(
            "Block %#010x (a decision point) reached again from block %#010x - a "
            "DIFFERENT path than block %#010x, which originally decided it. This "
            "looks like a loop/dispatcher re-entered with different state each "
            "time; re-flagging its DEAD successors for manual review rather than "
            "trusting a verdict that may only hold for the first path through it."
            % (ea, predecessor_ea, original_predecessor)
        )
        for succ in block.successors:
            if succ.ea is None or succ.ea not in self.dead:
                continue
            reason = self.dead_reason.pop(succ.ea, "")
            old_block = self.dead_blocks.pop(succ.ea, None)
            self.dead.discard(succ.ea)
            self._flag_unresolved(
                succ.ea,
                "Re-flagged: block %#010x (which decided this) was reached again via "
                "a different path (%#010x) - the original DEAD verdict (%s) may only "
                "hold for one specific entry into that dispatcher, not every path "
                "through it." % (ea, predecessor_ea, reason or "no reason given"),
                predecessor_ea, block=old_block,
            )

    def _finish(self, partial):
        unexplored = list(self.worklist) if partial else []
        result = self._build_result(partial, unexplored)
        cb = self._on_finished_cb
        self._on_finished_cb = None
        if cb is not None:
            cb(result)

    def _build_result(self, partial, unexplored):
        # Documented v1 simplification (see gather_block): a later real
        # path jumping into the middle of an earlier block's range can
        # produce a second, overlapping BlockInfo record. Harmless for
        # coloring (idempotent), but would be a real hazard for NOPing
        # dead code - never let a "dead" (or unresolved) row claim an
        # address that a REAL block also claims.
        real_insn_eas = set()
        for block in self.visited.values():
            real_insn_eas.update(block.insn_eas)

        blocks = []
        for ea, block in self.visited.items():
            reason = self.decision_summary.get(ea) or ("Trace start" if ea == self.start_ea else "")
            blocks.append(CfgTraceBlockRecord(
                start_ea=block.start_ea, end_ea=block.end_ea, insn_eas=list(block.insn_eas),
                verdict="real", reason=reason,
            ))
        for ea, block in self.dead_blocks.items():
            safe_eas = [a for a in block.insn_eas if a not in real_insn_eas]
            if not safe_eas:
                continue
            blocks.append(CfgTraceBlockRecord(
                start_ea=block.start_ea, end_ea=block.end_ea, insn_eas=safe_eas,
                verdict="dead", reason=self.dead_reason.get(ea, ""),
            ))
        for item in self.unresolved:
            block = item.get("block")
            if block is not None:
                safe_eas = [a for a in block.insn_eas if a not in real_insn_eas]
                if not safe_eas:
                    continue
                blocks.append(CfgTraceBlockRecord(
                    start_ea=block.start_ea, end_ea=block.end_ea, insn_eas=safe_eas,
                    verdict="unresolved", reason=item.get("note", ""),
                ))
            else:
                anchor = item.get("anchor_ea")
                if anchor is not None and anchor not in real_insn_eas:
                    blocks.append(CfgTraceBlockRecord(
                        start_ea=anchor, end_ea=anchor, insn_eas=[anchor],
                        verdict="unresolved", reason=item.get("note", ""),
                    ))

        jump_patches = self._compute_jump_patches()

        return CfgTraceResult(
            blocks=blocks, partial=partial, blocks_processed=self.blocks_processed,
            unexplored=unexplored, error=None, symbolic_resolved_count=self.symbolic_resolved_count,
            jump_patches=jump_patches,
        )

    def _compute_jump_patches(self):
        """Finds every branch eligible for in-place byte patching - see
        CfgTraceJumpPatch above for the two cases (resolved conditional
        opaque predicate, and an indirect jump collapsed to one target).
        Deliberately excludes genuine data-dependent branches (both sides
        real - nothing to redirect, both are legitimately reachable), a
        live multi-case indirect dispatch, and anything touching an
        unresolved successor (never patch based on an undecided verdict).
        """
        patches = []
        for ea, block in self.visited.items():
            if block.kind == "conditional_branch" and len(block.successors) == 2:
                jump_succ = next((s for s in block.successors if s.role == "jump_target"), None)
                fall_succ = next((s for s in block.successors if s.role == "fallthrough"), None)
                if jump_succ is None or fall_succ is None or jump_succ.ea is None or fall_succ.ea is None:
                    continue
                jump_is_real = jump_succ.ea in self.visited or jump_succ.ea in self.queued
                jump_is_dead = jump_succ.ea in self.dead
                fall_is_real = fall_succ.ea in self.visited or fall_succ.ea in self.queued
                fall_is_dead = fall_succ.ea in self.dead
                if jump_is_real and fall_is_dead:
                    patches.append(CfgTraceJumpPatch(
                        jcc_ea=block.last_insn_ea, real_target_ea=jump_succ.ea,
                        dead_target_ea=fall_succ.ea, role_of_real="jump_target",
                    ))
                elif fall_is_real and jump_is_dead:
                    patches.append(CfgTraceJumpPatch(
                        jcc_ea=block.last_insn_ea, real_target_ea=fall_succ.ea,
                        dead_target_ea=jump_succ.ea, role_of_real="fallthrough",
                    ))
                # else: both real (data-dependent), both dead (shouldn't
                # happen - a live block always has a real predecessor
                # chain), or one/both unresolved - nothing eligible here.
            elif block.kind == "indirect_jump":
                # A computed/table-driven indirect jump (e.g. ARM64 "BR
                # X8" fed by a fixed pointer load, or an x86 "jmp [reg]")
                # that this trace found exactly one confirmed-real
                # destination for, with no other live candidate - i.e.
                # the indirection is not a genuine multi-case dispatch,
                # it always lands on the same address. Mirrors the
                # relocatability test _classify_block_for_rebuild already
                # uses for the same pattern in Rebuild Linear mode.
                real_succs = [
                    s for s in block.successors
                    if s.ea is not None and (s.ea in self.visited or s.ea in self.queued)
                ]
                if len(real_succs) == 1:
                    patches.append(CfgTraceJumpPatch(
                        jcc_ea=block.last_insn_ea, real_target_ea=real_succs[0].ea,
                        dead_target_ea=None, role_of_real="only",
                    ))
        return patches

    # -- LLM round-trip for one decision point -------------------------

    def _breadcrumb(self, ea):
        chain = [ea]
        cur = ea
        seen = {ea}
        while cur in self.predecessors:
            cur = self.predecessors[cur]
            if cur in seen:
                break  # defensive: predecessors should never cycle, but never hang if it does
            seen.add(cur)
            chain.append(cur)
        chain.reverse()
        return chain

    def _is_ancestor(self, candidate_ea, current_ea):
        """True if candidate_ea appears in current_ea's predecessor chain -
        i.e. this candidate is a genuine loop back edge (reaching it again
        would mean execution looped around), not just an unrelated
        already-visited merge point elsewhere in the graph. Same
        cycle-safe walk as _breadcrumb.
        """
        cur = current_ea
        seen = {current_ea}
        while cur in self.predecessors:
            cur = self.predecessors[cur]
            if cur == candidate_ea:
                return True
            if cur in seen:
                return False
            seen.add(cur)
        return False

    def _build_hop_prompt(self, block):
        lines = [
            "Trace stats: %d block(s) processed so far, %d confirmed real, %d confirmed "
            "dead, %d unresolved so far." % (
                self.blocks_processed, len(self.visited), len(self.dead), len(self.unresolved)
            ),
        ]
        breadcrumb = self._breadcrumb(block.start_ea)
        if len(breadcrumb) > 30:
            lines.append("Path so far (earlier path abbreviated - showing last 30 of %d blocks):" % len(breadcrumb))
            breadcrumb = breadcrumb[-30:]
        else:
            lines.append("Path so far (from trace start to this block):")
        for i, ea in enumerate(breadcrumb):
            marker = "  <- current block" if ea == block.start_ea else ""
            summary = self.decision_summary.get(ea, "")
            lines.append("  %d. %#010x%s%s" % (i + 1, ea, (" - " + summary) if summary else "", marker))
        if self.trace_log:
            lines.append("")
            lines.append("Recent trace log:")
            for entry in self.trace_log[-8:]:
                lines.append("  " + entry)
        lines.append("")
        lines.append("Current block (%#010x - %#010x), kind=%s:" % (block.start_ea, block.end_ea, block.kind))
        lines.append(render_block_text(block.insn_eas))
        lines.append("")
        lines.append("Candidates:")
        for idx, succ in enumerate(block.successors):
            if succ.ea is None:
                note = (" (%s)" % succ.note) if succ.note else ""
                lines.append("  %d. indirect target, not resolved by IDA%s" % (idx + 1, note))
            else:
                note = (" - %s" % succ.note) if succ.note else ""
                back_edge = (
                    "  *** LOOP BACK EDGE: this address is an earlier block in this "
                    "same trace - taking it means looping back, not a one-way opaque "
                    "predicate. A comparison that looks fixed on THIS pass through the "
                    "loop can still be genuinely reachable on other iterations - do not "
                    "mark it dead just because this snapshot doesn't take it. ***"
                    if self._is_ancestor(succ.ea, block.start_ea) else ""
                )
                lines.append("  %d. %#010x (%s)%s%s" % (idx + 1, succ.ea, succ.role, note, back_edge))
        return "\n".join(lines)

    def _issue_decision_request(self, block):
        prompt = self._build_hop_prompt(block)
        messages = [
            {"role": "system", "content": self.config.cfg_trace_system_prompt},
            {"role": "user", "content": prompt},
        ]
        self._on_status(
            "Asking %s about block %#010x (%d candidate(s))..."
            % (self.config.server_label(self.server_url), block.start_ea, len(block.successors))
        )
        self.worker = LlamaStreamWorker(
            self.config, self.server_url, messages,
            lambda piece: None, lambda piece: None,
            self._on_decision_done, self._on_decision_error,
        )
        self.worker.start()

    def _next_fallback_server(self):
        for candidate in self._server_candidates:
            if candidate not in self._failed_servers:
                return candidate
        return None

    def _on_decision_error(self, message, is_connection_error=False):
        # Called via LlamaStreamWorker's MFF_FAST callback (see its run()) -
        # per ida_kernwin's own docs, MFF_FAST must never touch the
        # database, so only pure-Python/Qt-log work happens directly here;
        # anything that calls gather_block/decode_insn is dispatched
        # through _handle_unrecoverable_error under MFF_WRITE instead.
        self.worker = None
        if self._closed:
            return 0
        if is_connection_error and len(self._server_candidates) > 1:
            self._failed_servers.add(self.server_url)
            next_server = self._next_fallback_server()
            if next_server is not None:
                self._log(
                    "%s is unreachable (%s); falling back to %s..."
                    % (self.config.server_label(self.server_url), message, self.config.server_label(next_server))
                )
                self.server_url = next_server
                self._issue_decision_request(self._current_block)  # starts a network thread only, no DB access
                return 0
        # Unrecoverable for this block: stop the whole trace here, keeping
        # everything already decided, rather than guessing.
        block = self._current_block
        self._log("LLM request failed for block %#010x: %s" % (block.start_ea, message))
        ida_kernwin.execute_sync(
            functools.partial(self._handle_unrecoverable_error, block, message), ida_kernwin.MFF_WRITE
        )
        return 0

    def _handle_unrecoverable_error(self, block, message):
        self._flag_unresolved(
            block.start_ea, "LLM request failed: %s" % message, self.predecessors.get(block.start_ea), block=block
        )
        self._finish(partial=True)
        return 1

    def _on_decision_done(self, full_text, reasoning_text="", finish_reason=None):
        # Same MFF_FAST constraint as _on_decision_error above: _apply_decision
        # (decode_insn probes) and _pump (gather_block, which now also
        # writes) must run under MFF_WRITE, not directly on this callback.
        self.worker = None
        if self._closed:
            return 0
        ida_kernwin.execute_sync(
            functools.partial(self._apply_decision_and_pump, self._current_block, full_text),
            ida_kernwin.MFF_WRITE,
        )
        return 0

    def _apply_decision_and_pump(self, block, text):
        self._apply_decision(block, text)
        self._pump()
        return 1

    # -- applying one block's LLM verdict, with structural defenses ---

    def _target_state(self, ea):
        if ea in self.visited or ea in self.queued:
            return "real"
        if ea in self.dead:
            return "dead"
        return None

    def _apply_decision(self, block, text):
        candidate_eas = {s.ea for s in block.successors if s.ea is not None}
        single_unresolved_candidate = (
            len(block.successors) == 1
            and block.successors[0].ea is None
            and block.successors[0].role == "unresolved"
        )
        # Same per-edge narrowing _try_symbolic_resolve applies when IT
        # resolves a conditional_branch - computed here too since a
        # decision reaching the LLM (this function) still tautologically
        # reveals its own condition's truth value per edge, regardless of
        # who/what determined the overall REAL/DEAD verdict. See
        # _compute_edge_refinements.
        refined_states = {}
        if (
            block.kind == "conditional_branch"
            and len(block.successors) == 2
            and block.sym_insns
            and self._current_block_sym_state is not None
        ):
            refined_states = _compute_edge_refinements(
                self._current_block_sym_state, block.sym_insns[-1].mnem, block.successors,
            )

        def resolve_and_validate(token):
            addr = _parse_trace_address(token)
            if addr is None:
                return None
            if addr in candidate_eas:
                return addr
            if single_unresolved_candidate:
                # The one case a novel address is legitimately allowed:
                # a fully-unresolved indirect jump with no enumerated
                # candidates at all - but only if it actually decodes.
                try:
                    probe = ida_ua.insn_t()
                    if ida_ua.decode_insn(probe, addr) > 0:
                        return addr
                except Exception:
                    pass
            return None

        addressed = set()
        addressed_generic_indirect = False
        verdicts = []  # (ea, verdict, reason)

        for m in _REAL_TARGET_RE.finditer(text):
            addr = resolve_and_validate(m.group(1))
            if addr is None:
                self._log("Block %#010x: ignored REAL_TARGET '%s' - not a valid candidate address." % (block.start_ea, m.group(1)))
                continue
            verdicts.append((addr, "real", (m.group(2) or "").strip()))
            addressed.add(addr)

        for m in _DEAD_TARGET_RE.finditer(text):
            addr = resolve_and_validate(m.group(1))
            if addr is None:
                self._log("Block %#010x: ignored DEAD_TARGET '%s' - not a valid candidate address." % (block.start_ea, m.group(1)))
                continue
            verdicts.append((addr, "dead", (m.group(2) or "").strip()))
            addressed.add(addr)

        for m in _UNRESOLVED_TARGET_RE.finditer(text):
            token = (m.group(1) or "").strip()
            if token.lower() in ("indirect", "unknown", "n/a", "none"):
                addressed_generic_indirect = True
                self._flag_unresolved(
                    None, (m.group(2) or "").strip() or "Model could not resolve this indirect target.",
                    block.start_ea, anchor_ea=block.last_insn_ea,
                )
                continue
            addr = resolve_and_validate(token)
            if addr is None:
                self._log("Block %#010x: ignored UNRESOLVED_TARGET '%s' - not a valid candidate address." % (block.start_ea, token))
                continue
            verdicts.append((addr, "unresolved", (m.group(2) or "").strip()))
            addressed.add(addr)

        for addr, verdict, reason in verdicts:
            self._apply_target_verdict(
                addr, verdict, reason, block.start_ea, sym_state_override=refined_states.get(addr),
            )

        # Omission defense: every candidate the block actually presented
        # must have been addressed - anything left over is flagged for
        # manual review, never silently assumed dead.
        for succ in block.successors:
            if succ.ea is None:
                if not addressed_generic_indirect and not addressed and single_unresolved_candidate:
                    self._log("Block %#010x: indirect jump target left unaddressed by the model." % block.start_ea)
                    self._flag_unresolved(
                        None, "Model gave no verdict for this indirect jump's target.",
                        block.start_ea, anchor_ea=block.last_insn_ea,
                    )
                continue
            if succ.ea not in addressed:
                self._log(
                    "Block %#010x: candidate %#010x was never addressed by the model - "
                    "flagging for manual review." % (block.start_ea, succ.ea)
                )
                self._flag_unresolved(succ.ea, "Model did not address this candidate.", block.start_ea)

    def _apply_target_verdict(self, addr, verdict, reason, predecessor_ea, sym_state_override=None):
        if addr in self._unresolved_eas and self._target_state(addr) is None:
            # A later path resolves what an earlier path could not - this
            # is an upgrade, not a conflict, so let it through.
            self._unresolved_eas.discard(addr)
            self.unresolved = [u for u in self.unresolved if u.get("ea") != addr]

        existing = self._target_state(addr)
        if verdict == "real":
            if existing == "dead":
                self._log(
                    "Conflict at %#010x: already marked DEAD from another path, but block "
                    "%#010x says REAL - keeping DEAD (first verdict wins)." % (addr, predecessor_ea)
                )
                self._flag_unresolved(
                    addr, "Conflicting REAL/DEAD verdicts across different paths.", predecessor_ea,
                    block=self.dead_blocks.get(addr), anchor_ea=addr,
                )
                return
            if existing == "real":
                return
            summary = "REAL from %#010x" % predecessor_ea
            if reason:
                summary += ": " + reason
            self.decision_summary[addr] = summary
            self._log("Block %#010x -> %#010x marked REAL: %s" % (predecessor_ea, addr, reason or "(no reason given)"))
            state_to_propagate = sym_state_override if sym_state_override is not None else self._current_block_sym_state
            self._enqueue(addr, predecessor_ea, summary, state_to_propagate)
        elif verdict == "dead":
            if existing == "real":
                self._log(
                    "Conflict at %#010x: already marked REAL from another path, but block "
                    "%#010x says DEAD - keeping REAL (first verdict wins)." % (addr, predecessor_ea)
                )
                self._flag_unresolved(
                    addr, "Conflicting REAL/DEAD verdicts across different paths.", predecessor_ea,
                    block=self.visited.get(addr), anchor_ea=addr,
                )
                return
            if existing == "dead":
                return
            self._log("Block %#010x -> %#010x marked DEAD: %s" % (predecessor_ea, addr, reason or "(no reason given)"))
            self.dead.add(addr)
            self.dead_reason[addr] = reason or ""
            if addr not in self.dead_blocks:
                try:
                    self.dead_blocks[addr] = gather_block(addr, claimed_insns=self.claimed_insns)
                except Exception:
                    pass
        else:  # "unresolved"
            if existing is not None:
                return  # already confidently resolved from another path - don't downgrade it
            self._flag_unresolved(addr, reason or "Model flagged this target as unresolved.", predecessor_ea)

    def _flag_unresolved(self, ea, note, predecessor_ea, block=None, anchor_ea=None):
        """Marks ea (or, when ea is None, anchor_ea - the jump instruction
        itself) as needing manual review. Does exactly one extra marking-
        only gather_block() call (never enqueued, never counted against
        max_trace_blocks) so the whole ambiguous region gets colored, not
        just its entry byte.
        """
        if ea is not None:
            if ea in self._unresolved_eas:
                return
            self._unresolved_eas.add(ea)
            if block is None and ea not in self.visited and ea not in self.dead:
                try:
                    block = gather_block(ea, claimed_insns=self.claimed_insns)
                except Exception:
                    block = None
        self.unresolved.append({
            "ea": ea, "anchor_ea": anchor_ea, "note": note,
            "predecessor": predecessor_ea, "block": block,
        })


_CfgTraceApplyItem = namedtuple("_CfgTraceApplyItem", ["insn_eas", "verdict", "comment"])

_PATCH_MASK64 = 0xFFFFFFFFFFFFFFFF

_ARM64_NOP_WORD = b"\x1f\x20\x03\xd5"  # little-endian encoding of 0xD503201F
_ARM64_BR_MASK = 0xFFFFFC1F   # BR Xn: bits [31:10] and [4:0] fixed, Rn in [9:5]
_ARM64_BR_FIXED = 0xD61F0000

# Recognized AArch64 conditional-branch encodings, as (mask, value) pairs
# on the 32-bit little-endian instruction word - used to VERIFY an
# instruction really is a conditional branch before rewriting it (never
# guess). B.cond: bits [31:24]=0x54 and bit[4]=0. CBZ/CBNZ: bits
# [30:25]=011010 (sf at [31] and op at [24] both free). TBZ/TBNZ: bits
# [30:25]=011011. All are 4 bytes and PC-relative, so all are safely
# replaceable in place with a same-size NOP or unconditional B.
_ARM64_COND_BRANCH_FORMS = (
    (0xFF000010, 0x54000000),  # B.cond
    (0x7E000000, 0x34000000),  # CBZ / CBNZ (both operand sizes)
    (0x7E000000, 0x36000000),  # TBZ / TBNZ (both operand sizes)
)


def _is_arm64_cond_branch_word(word):
    return any((word & mask) == value for mask, value in _ARM64_COND_BRANCH_FORMS)


def _compute_arm64_cond_branch_patch_bytes(jcc_ea, size, raw, real_target_ea, role_of_real):
    """AArch64 counterpart to _compute_jcc_patch_bytes for a resolved
    conditional opaque predicate. Verifies the instruction is one of the
    recognized 4-byte conditional branches (B.cond/CBZ/CBNZ/TBZ/TBNZ)
    before touching it, then:
      - role "fallthrough" (real path never branches) -> a 4-byte NOP, so
        execution always falls through;
      - role "jump_target" (real path always branches) -> a 4-byte
        unconditional B to real_target_ea (imm26, +/-128MB reach), which
        is the conditional branch's own taken destination.
    Returns the 4-byte replacement, or None if unrecognized/out of range.
    """
    if size != 4 or len(raw) != 4:
        return None
    word = int.from_bytes(raw, "little")
    if not _is_arm64_cond_branch_word(word):
        return None
    if role_of_real == "fallthrough":
        return _ARM64_NOP_WORD
    if role_of_real != "jump_target":
        return None
    delta = real_target_ea - jcc_ea
    if delta % 4 != 0:
        return None  # A64 branch targets are 4-byte aligned
    imm26 = delta // 4
    if not (-(1 << 25) <= imm26 < (1 << 25)):
        return None  # out of B's +/-128MB reach
    new_word = 0x14000000 | (imm26 & 0x3FFFFFF)
    return new_word.to_bytes(4, "little")


def _compute_jcc_patch_bytes(jcc_ea, size, raw, real_target_ea, role_of_real):
    """Pure byte-level computation (no IDA calls) - given a decoded
    conditional branch's address/size/raw bytes, the resolved real
    target, and which side is real, returns the exact `size`-byte
    replacement to write at jcc_ea, or None if this encoding isn't
    confidently recognized/supported (the caller must leave the original
    bytes untouched in that case - never guess at an encoding this hasn't
    verified). Handles x86/x64 Jcc and, on an AArch64 target, the 4-byte
    conditional branches via _compute_arm64_cond_branch_patch_bytes.
    """
    if len(raw) != size:
        return None
    if _is_arm64_target():
        return _compute_arm64_cond_branch_patch_bytes(jcc_ea, size, raw, real_target_ea, role_of_real)
    if role_of_real == "fallthrough":
        # Real path is "don't take the jump" - NOP the whole instruction
        # so execution always falls through, now unconditionally.
        return b"\x90" * size
    if role_of_real != "jump_target":
        return None
    if size == 2 and 0x70 <= raw[0] <= 0x7F:
        # Short Jcc (opcode + rel8) -> short JMP (0xEB + SAME rel8) -
        # identical length, identical displacement byte (the target is
        # unchanged - we're only making the existing jump unconditional),
        # opcode swap only.
        rel8 = raw[1] if raw[1] < 0x80 else raw[1] - 0x100
        if (jcc_ea + 2 + rel8) & _PATCH_MASK64 != real_target_ea & _PATCH_MASK64:
            return None  # inconsistent with the claimed target - refuse rather than guess
        return bytes([0xEB, raw[1]])
    if size == 6 and raw[0] == 0x0F and 0x80 <= raw[1] <= 0x8F:
        # Near Jcc (0F 8x + rel32, 6 bytes) -> near JMP (E9 + rel32, 5
        # bytes) - one byte SHORTER, so the displacement must be
        # recomputed from scratch (JMP's rel32 is relative to a
        # different reference point) and the spare byte NOPed to keep
        # every later instruction's address unchanged.
        new_rel32 = (real_target_ea - (jcc_ea + 5)) & 0xFFFFFFFF
        signed_new_rel32 = new_rel32 if new_rel32 < 0x80000000 else new_rel32 - 0x100000000
        if (jcc_ea + 5 + signed_new_rel32) & _PATCH_MASK64 != real_target_ea & _PATCH_MASK64:
            return None
        return bytes([0xE9]) + new_rel32.to_bytes(4, "little") + b"\x90"
    return None


def _compute_indirect_jump_patch_bytes(jcc_ea, size, raw, real_target_ea):
    """Pure byte-level computation for collapsing an indirect_jump (a
    computed/table-driven branch through a register - x86 "jmp reg/mem",
    ARM64 "BR Xn") that this trace found always lands on the SAME single
    target, into a direct unconditional branch to it. Same contract as
    _compute_jcc_patch_bytes: returns the exact `size`-byte replacement,
    or None if this encoding/architecture isn't confidently recognized
    (never guessed at) or the target is out of the encoding's reachable
    range.
    """
    if len(raw) != size:
        return None
    if _is_arm64_target():
        if size != 4:
            return None
        word = int.from_bytes(raw, "little")
        if (word & _ARM64_BR_MASK) != _ARM64_BR_FIXED:
            return None  # not a plain "BR Xn" - don't guess at anything else
        delta = real_target_ea - jcc_ea
        if delta % 4 != 0:
            return None  # A64 branch targets must be 4-byte aligned
        imm26 = delta // 4
        if not (-(1 << 25) <= imm26 < (1 << 25)):
            return None  # out of B's +/-128MB reach
        new_word = 0x14000000 | (imm26 & 0x3FFFFFF)
        return new_word.to_bytes(4, "little")
    # x86/x64: only handles indirect jumps at least as long as a JMP
    # rel32 (5 bytes), padding any remainder with NOP - same discipline
    # as the near-Jcc case above. Shorter forms (2-3 byte "jmp reg") are
    # refused rather than guessed at, since there's no room to encode a
    # 32-bit displacement.
    if size < 5:
        return None
    new_rel32 = (real_target_ea - (jcc_ea + 5)) & 0xFFFFFFFF
    signed_new_rel32 = new_rel32 if new_rel32 < 0x80000000 else new_rel32 - 0x100000000
    if (jcc_ea + 5 + signed_new_rel32) & _PATCH_MASK64 != real_target_ea & _PATCH_MASK64:
        return None
    return bytes([0xE9]) + new_rel32.to_bytes(4, "little") + b"\x90" * (size - 5)


def _patch_bytes_and_reanalyze(ea, new_bytes, make_code=False):
    for i, b in enumerate(new_bytes):
        ida_bytes.patch_byte(ea + i, b)
    end = ea + len(new_bytes)
    try:
        ida_bytes.del_items(ea, ida_bytes.DELIT_SIMPLE, len(new_bytes))
    except Exception:
        pass
    if make_code:
        # Force every patched instruction to be (re)created as code. When a
        # dead block is NOPed (or a branch rewritten), there is often no
        # live control flow left reaching these bytes, so IDA's auto-
        # analysis won't turn them back into instructions on its own - they
        # linger as undefined bytes or get coerced into a data item (e.g.
        # an ARM64 NOP word shown as "DCB 0x1F, 0x20, 3, 0xD5" with "---"
        # separators, instead of "NOP"). Creating each instruction
        # explicitly fixes the listing and keeps the region walkable.
        cur = ea
        step = 4 if _is_arm64_target() else 1  # A64 is fixed 4-byte; x86 byte-granular
        while cur < end:
            try:
                n = ida_ua.create_insn(cur)
            except Exception:
                n = 0
            cur += n if n > 0 else step
    try:
        ida_auto.plan_and_wait(ea, end)
    except Exception:
        pass


def _patch_nop_instruction(ea):
    """Decodes the instruction currently at ea (fresh, so this reflects
    whatever is really there right now) and overwrites every one of its
    bytes with a NOP encoding valid for the current architecture. x86 NOP
    (0x90) is a single byte so it tiles any instruction length; AArch64
    has no single-byte NOP, so its 4-byte NOP word is tiled instead
    (decode_insn always returns a multiple of 4 for A64, so this divides
    evenly). Returns the number of bytes touched, so the caller can
    record exactly what this patch covers (for Undo Patches - see
    _AppliedCfgPatch).
    """
    probe = ida_ua.insn_t()
    size = ida_ua.decode_insn(probe, ea)
    if size <= 0:
        size = 1  # could not decode - NOP at least the one byte we know about
    if _is_arm64_target():
        whole_words, remainder = divmod(size, 4)
        filler = _ARM64_NOP_WORD * whole_words + b"\x00" * remainder
    else:
        filler = b"\x90" * size
    _patch_bytes_and_reanalyze(ea, filler, make_code=True)
    return size


def _patch_jcc_to_real_target(jcc_ea, real_target_ea, role_of_real):
    """Returns (ok, reason_or_None, size_or_None) - size is the number of
    bytes touched (for Undo Patches - see _AppliedCfgPatch), only
    meaningful when ok is True. Re-decodes the instruction at jcc_ea
    fresh (never trusts stale data) and verifies its raw bytes match one
    of the recognized encodings before touching anything: conditional
    opaque predicates (role_of_real "jump_target"/"fallthrough") go
    through _compute_jcc_patch_bytes (x86/x64 Jcc, or AArch64 B.cond/
    CBZ/CBNZ/TBZ/TBNZ); a collapsed indirect jump (role_of_real "only")
    goes through _compute_indirect_jump_patch_bytes (x86/x64 or ARM64).
    Any other encoding (jcxz, unusual prefixes, anything neither helper
    has specifically verified) is refused, not guessed at.
    """
    insn = ida_ua.insn_t()
    size = ida_ua.decode_insn(insn, jcc_ea)
    if size <= 0:
        return False, "could not decode instruction at %#010x" % jcc_ea, None
    try:
        raw = ida_bytes.get_bytes(jcc_ea, size)
    except Exception as exc:
        return False, "could not read instruction bytes: %s" % exc, None
    if raw is None or len(raw) != size:
        return False, "could not read instruction bytes at %#010x" % jcc_ea, None
    if role_of_real == "only":
        patch = _compute_indirect_jump_patch_bytes(jcc_ea, size, raw, real_target_ea)
    else:
        patch = _compute_jcc_patch_bytes(jcc_ea, size, raw, real_target_ea, role_of_real)
    if patch is None:
        return False, "unrecognized or inconsistent branch encoding at %#010x (size=%d) - not patched" % (jcc_ea, size), None
    _patch_bytes_and_reanalyze(jcc_ea, patch, make_code=True)
    return True, None, size


# ---------------------------------------------------------------------------
# CFG trace: "Rebuild linear" - an alternative to in-place patching that
# leaves every original byte untouched EXCEPT the function's own entry
# point through the end of the newly rebuilt code. Every confirmed-real
# block (runner.visited - already the complete, closed set reachable from
# the trace's start address) is concatenated into one straight-line
# sequence with every control transfer re-encoded explicitly (never
# relying on layout adjacency), written starting exactly at the entry
# point. A block that cannot be safely relocated (RIP-relative addressing,
# or an indirect dispatch with more than one genuinely real case) is left
# at its original address entirely untouched, PROVIDED that address and
# every relocatable block's reference to it fall outside the rebuilt
# range - otherwise the whole rebuild is refused rather than guessed at.
# See _plan_linear_rebuild (pure, tested standalone) for the actual
# layout/encoding math, and _gather_linear_rebuild_plan for the glue that
# turns real IDA state into its inputs.
# ---------------------------------------------------------------------------

_RawPart = namedtuple("_RawPart", ["data"])
# A direct `call rel32` (always E8 + 4-byte rel32, 5 bytes total) embedded
# in a block's body - its absolute target doesn't move when the CALLER is
# relocated, so the displacement must be recomputed, unlike a plain
# RawPart which is copied verbatim.
_CallPart = namedtuple("_CallPart", ["target_ea"])

# Terminator kinds - always re-encoded explicitly, never left to rely on
# whatever happens to be adjacent after layout:
_VerbatimTerm = namedtuple("_VerbatimTerm", ["data"])  # ret, or an indirect jump/call with no PC-relative content - copied as-is
_SingleJump = namedtuple("_SingleJump", ["target_ea"])  # -> JMP rel32 (5 bytes) to the target's new address
_TwoWay = namedtuple("_TwoWay", ["cond_byte", "jump_target_ea", "fallthrough_ea"])  # -> Jcc rel32 (6B) + JMP rel32 (5B)

_RelocBlock = namedtuple("_RelocBlock", ["start_ea", "parts", "terminator"])

# What _gather_linear_rebuild_plan / _plan_linear_rebuild return.
LinearRebuildPlan = namedtuple(
    "LinearRebuildPlan",
    ["entry_ea", "new_addr_by_old_start", "code", "relocated_count", "anchored_count", "error"],
)


def _plan_linear_rebuild(entry_ea, layout_order, blocks_by_ea, anchored_ranges, anchored_targets, available_bytes):
    """Pure layout/encoding algorithm - no IDA calls, verified standalone
    before being wired in here (see test_linear_rebuild.py). Two-pass:
    sizes (fixed per block, independent of final position) determine
    every block's new address first; only then are the actual bytes
    generated, since displacements can be forward references. Returns
    (new_addr_by_old_start, code_bytes, None) on success, or
    (None, None, reason) on refusal - never raises for a bad/unsupported
    input shape it can detect, never guesses at an out-of-range
    displacement (verified via an explicit round-trip check, the same
    discipline _compute_jcc_patch_bytes already applies).
    """
    if not layout_order or layout_order[0] != entry_ea:
        return None, None, "layout must start with the function's entry address"
    if len(set(layout_order)) != len(layout_order):
        return None, None, "layout_order contains a duplicate address"
    for ea in layout_order:
        if ea not in blocks_by_ea:
            return None, None, "block %#010x in layout has no relocation info" % ea

    def _term_size(term):
        if isinstance(term, _VerbatimTerm):
            return len(term.data)
        if isinstance(term, _SingleJump):
            return 5
        if isinstance(term, _TwoWay):
            return 11
        raise ValueError("unknown terminator type")

    def _block_size(block):
        size = 0
        for part in block.parts:
            size += len(part.data) if isinstance(part, _RawPart) else 5
        return size + _term_size(block.terminator)

    new_addr_by_old_start = {}
    offset = 0
    for ea in layout_order:
        new_addr_by_old_start[ea] = entry_ea + offset
        offset += _block_size(blocks_by_ea[ea])
    total_size = offset

    if total_size > available_bytes:
        return None, None, (
            "rebuilt code needs %d byte(s), only %d available in the function's "
            "already-explored range" % (total_size, available_bytes)
        )

    canvas_start, canvas_end = entry_ea, entry_ea + total_size
    for (a_start, a_end) in anchored_ranges:
        if a_start < canvas_end and canvas_start < a_end:
            return None, None, (
                "an unrelocatable block at %#010x-%#010x overlaps the rebuilt code's range" % (a_start, a_end)
            )
    for t_ea in anchored_targets:
        if canvas_start <= t_ea < canvas_end:
            return None, None, (
                "an unrelocatable block's target %#010x falls inside the rebuilt code's range" % t_ea
            )

    def _rel32_or_none(from_ea_after_insn, target):
        rel32 = (target - from_ea_after_insn) & 0xFFFFFFFF
        signed = rel32 if rel32 < 0x80000000 else rel32 - 0x100000000
        if (from_ea_after_insn + signed) & _PATCH_MASK64 != target & _PATCH_MASK64:
            return None
        return rel32

    out = bytearray()
    for ea in layout_order:
        block = blocks_by_ea[ea]
        cursor = new_addr_by_old_start[ea]
        for part in block.parts:
            if isinstance(part, _RawPart):
                out += part.data
                cursor += len(part.data)
            else:
                rel32 = _rel32_or_none(cursor + 5, part.target_ea)
                if rel32 is None:
                    return None, None, "call target %#010x is out of rel32 range from the relocated call site" % part.target_ea
                out += bytes([0xE8]) + rel32.to_bytes(4, "little")
                cursor += 5
        term = block.terminator
        if isinstance(term, _VerbatimTerm):
            out += term.data
        elif isinstance(term, _SingleJump):
            target = new_addr_by_old_start.get(term.target_ea, term.target_ea)
            rel32 = _rel32_or_none(cursor + 5, target)
            if rel32 is None:
                return None, None, "jump target %#010x is out of rel32 range" % target
            out += bytes([0xE9]) + rel32.to_bytes(4, "little")
        elif isinstance(term, _TwoWay):
            jt = new_addr_by_old_start.get(term.jump_target_ea, term.jump_target_ea)
            ft = new_addr_by_old_start.get(term.fallthrough_ea, term.fallthrough_ea)
            jcc_rel32 = _rel32_or_none(cursor + 6, jt)
            if jcc_rel32 is None:
                return None, None, "conditional jump target %#010x is out of rel32 range" % jt
            out += bytes([0x0F, term.cond_byte]) + jcc_rel32.to_bytes(4, "little")
            jmp_ea = cursor + 6
            jmp_rel32 = _rel32_or_none(jmp_ea + 5, ft)
            if jmp_rel32 is None:
                return None, None, "fallthrough target %#010x is out of rel32 range" % ft
            out += bytes([0xE9]) + jmp_rel32.to_bytes(4, "little")
        else:
            raise ValueError("unknown terminator type")

    return new_addr_by_old_start, bytes(out), None


def _operand_is_rip_relative(ea, insn):
    """True if any operand of the instruction at ea renders with "rip" in
    its text (IDA's standard way of showing x64 RIP-relative addressing,
    e.g. "[rip+0x2000]"). Text-based, matching this file's established
    pattern for addressing details not safely reachable via documented
    raw op_t fields (see _parse_memory_operand_text) - a relocated copy
    of a RIP-relative instruction would compute the wrong address (RIP-
    relative displacements are relative to the NEXT instruction, which
    moves when relocated), so any block containing one is never a
    relocation candidate.
    """
    for i in range(8):
        try:
            op = insn.ops[i]
        except Exception:
            break
        if op.type == ida_ua.o_void:
            break
        try:
            text = idc.print_operand(ea, i) or ""
        except Exception:
            continue
        if "rip" in text.lower():
            return True
    return False


def _classify_block_for_rebuild(ea, block, visited):
    """Returns ("relocatable", parts, terminator) or ("anchored", None, None).
    parts/terminator are the _RawPart/_CallPart list and terminator value
    for _RelocBlock - built from FRESH decode_insn/get_bytes calls (never
    trusts block.insn_eas' vintage), refusing (anchoring) at the first
    sign of anything not confidently handled: RIP-relative addressing
    anywhere in the block, an unrecognized instruction, or (for the
    terminator specifically) more than one genuinely real successor on an
    indirect jump - see the module docstring above for why.
    """
    real_successors = [s for s in block.successors if s.ea is not None and s.ea in visited]
    body_eas = block.insn_eas[:-1] if block.insn_eas else []
    parts = []
    for insn_ea in body_eas:
        insn = ida_ua.insn_t()
        size = ida_ua.decode_insn(insn, insn_ea)
        if size <= 0:
            return "anchored", None, None
        if _operand_is_rip_relative(insn_ea, insn):
            return "anchored", None, None
        try:
            raw = ida_bytes.get_bytes(insn_ea, size)
        except Exception:
            raw = None
        if raw is None or len(raw) != size:
            return "anchored", None, None
        is_call = bool(insn.get_canon_feature() & ida_idp.CF_CALL)
        if is_call:
            try:
                direct_targets = list(idautils.CodeRefsFrom(insn_ea, 0))
            except Exception:
                direct_targets = []
            if len(direct_targets) == 1 and size == 5 and raw[0] == 0xE8:
                parts.append(_CallPart(target_ea=direct_targets[0]))
                continue
        parts.append(_RawPart(data=raw))

    if not block.insn_eas:
        return "anchored", None, None
    term_ea = block.insn_eas[-1]
    term_insn = ida_ua.insn_t()
    term_size = ida_ua.decode_insn(term_insn, term_ea)
    if term_size <= 0:
        return "anchored", None, None
    if _operand_is_rip_relative(term_ea, term_insn):
        return "anchored", None, None
    try:
        term_raw = ida_bytes.get_bytes(term_ea, term_size)
    except Exception:
        term_raw = None
    if term_raw is None or len(term_raw) != term_size:
        return "anchored", None, None

    if block.kind == "return":
        return "relocatable", parts, _VerbatimTerm(data=term_raw)
    if block.kind in ("unconditional_jump", "indirect_jump") and len(real_successors) == 1:
        return "relocatable", parts, _SingleJump(target_ea=real_successors[0].ea)
    if block.kind == "indirect_jump" and len(real_successors) == 0:
        # A raw register/memory-indirect jump IDA couldn't enumerate any
        # case for - safe to copy verbatim (no PC-relative content, just
        # confirmed above) regardless of where it ends up.
        return "relocatable", parts, _VerbatimTerm(data=term_raw)
    if block.kind == "conditional_branch" and len(real_successors) == 2:
        jump_succ = next((s for s in block.successors if s.role == "jump_target"), None)
        fall_succ = next((s for s in block.successors if s.role == "fallthrough"), None)
        if jump_succ is None or fall_succ is None:
            return "anchored", None, None
        if term_size < 2 or not (0x0F == term_raw[0] and 0x80 <= term_raw[1] <= 0x8F) and not (0x70 <= term_raw[0] <= 0x7F):
            return "anchored", None, None
        cond_byte = term_raw[1] if term_raw[0] == 0x0F else (term_raw[0] - 0x70 + 0x80)
        return "relocatable", parts, _TwoWay(
            cond_byte=cond_byte, jump_target_ea=jump_succ.ea, fallthrough_ea=fall_succ.ea,
        )
    if block.kind == "conditional_branch" and len(real_successors) == 1:
        return "relocatable", parts, _SingleJump(target_ea=real_successors[0].ea)
    # 2+ real successors on an indirect_jump (a live multi-case dispatch),
    # 0 real successors on anything but a return/no-candidate-indirect
    # (an undecided branch, e.g. from a cancelled/partial trace), or
    # anything else not explicitly recognized above - anchor rather than
    # guess.
    return "anchored", None, None


def _max_contiguous_claimed_run(entry_ea, claimed_insns):
    """How many bytes starting exactly at entry_ea are covered, with no
    gaps, by instructions this trace has already claimed (real or dead).
    Used only as the fast path inside _rebuild_canvas_available_bytes
    below - a real obfuscated function's real/dead blocks are rarely
    packed with zero gaps between them (unexplored padding, alignment,
    or simply bytes no edge this trace found ever pointed at), so
    stopping here entirely would refuse far more often than necessary;
    see that function for how gaps get cautiously crossed instead.
    """
    total = 0
    ea = entry_ea
    while True:
        size = claimed_insns.get(ea)
        if size is None:
            break
        total += size
        ea += size
    return total


def _rebuild_canvas_available_bytes(entry_ea, envelope_start, envelope_end, claimed_insns):
    """Bytes starting at entry_ea safe to use as the Rebuild Linear
    canvas, extending up to envelope_end (the furthest address any block
    this trace examined - real, dead, or unresolved - reaches).
    envelope_start is the earliest such address (can be before entry_ea -
    a back edge or shared/tail-merged block can legitimately sit there).
    Bytes already claimed by this trace (real or dead) are always safe by
    construction and skipped over at their full instruction size. A GAP
    byte/item (claimed by neither) is ALSO treated as safe UNLESS:
      - it belongs to a DIFFERENT existing IDA function than the one (if
        any) already recognized at entry_ea - obfuscated functions are
        very often ALREADY defined as one proc by IDA's own auto-
        analysis (this feature doesn't require them not to be; that's
        only ever true for the entry-point-not-recognized-at-all case),
        and a gap byte inside that SAME function is exactly the ordinary
        junk/decoy padding this is meant to cross, not something to stop
        at, or
      - it has an xref (code or data) from an address OUTSIDE
        [envelope_start, envelope_end) - i.e. genuinely outside every
        block this trace ever looked at, in any verdict. A flattening
        dispatcher's decoy/junk blocks routinely cross-reference EACH
        OTHER without any of them ever being walked as real or dead (the
        trace correctly never needed to visit unreachable junk) - an
        xref from another address the trace simply didn't individually
        classify is not evidence of anything external, only an xref from
        truly outside the whole span this trace ever examined is.
    This lets a genuinely scattered obfuscated function (real/dead
    blocks separated by unexplored padding/junk - the common case for
    OLLVM-style flattening) use its true extent instead of refusing at
    the first unexplored byte, while still refusing to eat into a
    genuinely different, already-recognized function or anything still
    referenced from truly outside this trace's whole span. Steps by
    IDA's own existing item size when one is defined in a gap (so an
    already-analyzed data item is checked - and skipped - as a whole,
    not split awkwardly), one byte at a time otherwise.

    Returns (available_bytes, stop_reason) - stop_reason is a short,
    human-readable explanation of exactly where and why the search
    stopped (hit the envelope cleanly, a different function, or an
    external xref), surfaced in the "Cannot rebuild" message so a refusal
    can actually be diagnosed instead of just reporting two numbers.
    """
    try:
        entry_func = ida_funcs.get_func(entry_ea)
    except Exception:
        entry_func = None
    entry_func_start = entry_func.start_ea if entry_func is not None else None

    ea = entry_ea
    stop_reason = "reached the furthest address this trace explored (%#010x)" % envelope_end
    while ea < envelope_end:
        size = claimed_insns.get(ea)
        if size is not None:
            ea += size
            continue
        try:
            func_here = ida_funcs.get_func(ea)
        except Exception:
            func_here = None
        if func_here is not None and func_here.start_ea != entry_func_start:
            try:
                other_name = idc.get_func_name(ea) or ("%#010x" % func_here.start_ea)
            except Exception:
                other_name = "%#010x" % func_here.start_ea
            stop_reason = (
                "hit a different function (%s, starting at %#010x) at %#010x"
                % (other_name, func_here.start_ea, ea)
            )
            break  # a genuinely DIFFERENT function's territory - stop here
        try:
            refs = list(idautils.XrefsTo(ea, 0))
        except Exception:
            refs = []
        external_frm = next(
            (
                getattr(xref, "frm", None) for xref in refs
                if not (envelope_start <= getattr(xref, "frm", -1) < envelope_end)
            ),
            None,
        )
        if external_frm is not None:
            stop_reason = (
                "%#010x is referenced from %#010x, an address outside everything this trace examined"
                % (ea, external_frm)
            )
            break  # referenced from truly outside this trace's whole span - stop here
        try:
            step = ida_bytes.get_item_size(ea) or 1
        except Exception:
            step = 1
        ea += max(step, 1)
    return min(ea, envelope_end) - entry_ea, stop_reason


def _gather_linear_rebuild_plan(runner):
    """Builds every input _plan_linear_rebuild needs from a finished
    CfgTraceRunner's real IDA state, then calls it. runner.visited is
    already the complete, closed set of confirmed-real blocks reachable
    from the trace's start address - not subject to the review table's
    checkboxes, since a partial linearization missing some of the real
    control flow would not be a safe/meaningful rebuild (checkboxes still
    control Mark-only coloring as before). Returns a LinearRebuildPlan;
    .error is None on success.
    """
    entry_ea = runner.start_ea
    layout_order = list(runner.visited.keys())
    if not layout_order or layout_order[0] != entry_ea:
        # dict order should always start with the trace's own seed, but
        # never trust that blindly - reorder defensively if not.
        layout_order = [entry_ea] + [ea for ea in layout_order if ea != entry_ea]
    if entry_ea not in runner.visited:
        return LinearRebuildPlan(entry_ea, None, None, 0, 0, "the function's own entry point was never confirmed real")

    blocks_by_ea = {}
    anchored_ranges = []
    anchored_targets = set()
    relocated_count = 0
    anchored_count = 0
    envelope_start = entry_ea
    envelope_end = entry_ea
    for ea, block in runner.visited.items():
        envelope_start = min(envelope_start, block.start_ea)
        envelope_end = max(envelope_end, block.end_ea)
        kind, parts, terminator = _classify_block_for_rebuild(ea, block, runner.visited)
        if kind == "relocatable":
            blocks_by_ea[ea] = _RelocBlock(start_ea=ea, parts=parts, terminator=terminator)
            relocated_count += 1
        else:
            anchored_ranges.append((block.start_ea, block.end_ea))
            for succ in block.successors:
                if succ.ea is not None:
                    anchored_targets.add(succ.ea)
            anchored_count += 1

    # dead_blocks are already fully covered by claimed_insns (safe to
    # overwrite, that's the whole point of removing dead code), but
    # unresolved ones are exactly "we don't know if this is real" - never
    # let the canvas eat into one of those, same protection as an
    # unrelocatable real block above. All three also extend the envelope:
    # the full span this trace looked at, in any verdict, bounds how far
    # the gap-crossing search below is even allowed to consider, and (via
    # envelope_start) which xrefs count as "from within this same blob"
    # rather than genuinely external - see _rebuild_canvas_available_bytes.
    for block in runner.dead_blocks.values():
        envelope_start = min(envelope_start, block.start_ea)
        envelope_end = max(envelope_end, block.end_ea)
    for item in runner.unresolved:
        block = item.get("block")
        if block is not None:
            anchored_ranges.append((block.start_ea, block.end_ea))
            envelope_start = min(envelope_start, block.start_ea)
            envelope_end = max(envelope_end, block.end_ea)
        else:
            anchor = item.get("anchor_ea")
            if anchor is not None:
                anchored_ranges.append((anchor, anchor + 1))
                envelope_start = min(envelope_start, anchor)
                envelope_end = max(envelope_end, anchor + 1)

    if entry_ea not in blocks_by_ea:
        return LinearRebuildPlan(
            entry_ea, None, None, relocated_count, anchored_count,
            "the entry block itself could not be safely relocated (RIP-relative addressing or "
            "an unresolved/multi-case indirect jump) - nothing valid could be placed at the "
            "entry point without either moving it (unsafe) or leaving it exactly as-is",
        )
    layout_order = [ea for ea in layout_order if ea in blocks_by_ea]

    available_bytes, stop_reason = _rebuild_canvas_available_bytes(
        entry_ea, envelope_start, envelope_end, runner.claimed_insns,
    )
    new_addr_by_old_start, code, err = _plan_linear_rebuild(
        entry_ea, layout_order, blocks_by_ea, anchored_ranges, anchored_targets, available_bytes,
    )
    if err is not None:
        # Surface exactly where/why the available-space search stopped for
        # the specific "doesn't fit" refusal - the only case that number
        # actually depends on the search above rather than an overlap
        # conflict (which already names its own address).
        if "available in the function" in err:
            err = "%s - the search %s" % (err, stop_reason)
        return LinearRebuildPlan(entry_ea, None, None, relocated_count, anchored_count, err)
    return LinearRebuildPlan(entry_ea, new_addr_by_old_start, code, relocated_count, anchored_count, None)


def _apply_linear_rebuild(plan):
    """Writes plan.code starting at plan.entry_ea - the ONLY bytes this
    touches anywhere in the binary. Reuses _patch_bytes_and_reanalyze
    exactly like every other patch in this file, with make_code so the
    freshly written linear instruction stream is decoded as code rather
    than left as undefined/data bytes.
    """
    _patch_bytes_and_reanalyze(plan.entry_ea, plan.code, make_code=True)


def _ensure_function_at(start_ea):
    """Best-effort: makes sure a proper function begins at start_ea. The
    whole premise of CFG tracing is that IDA's original analysis may
    never have recognized this address as a function at all - a common
    symptom of flattening/dispatcher-style obfuscation, where the real
    entry point is only reachable through control flow IDA couldn't
    follow - so once the recovered control flow has actually been
    patched in, this gives Hex-Rays something to decompile without the
    user needing to press P manually. Returns a short log message, or
    None if nothing needed doing. Never touches an address that's
    already inside a DIFFERENT function - that would mean start_ea
    turned out to belong to existing, unrelated code, and forcing a new
    function boundary there could break that function's own analysis.
    """
    try:
        existing = ida_funcs.get_func(start_ea)
    except Exception:
        existing = None
    if existing is not None:
        if existing.start_ea == start_ea:
            return None
        return (
            "%#010x is already inside another function (starting at %#010x) - "
            "not forcing a new function boundary there." % (start_ea, existing.start_ea)
        )
    try:
        ok = ida_funcs.add_func(start_ea)
    except Exception as exc:
        return "Failed to create a function at %#010x: %s" % (start_ea, exc)
    if not ok:
        return "IDA could not create a function at %#010x (add_func failed)." % start_ea
    return "Created a function at %#010x so it can be decompiled." % start_ea


def _apply_cfg_trace_and_refresh(config, items, jump_patches=None, start_ea=None, touched_ranges=None):
    """One execute_sync/MFF_WRITE round-trip for the whole accepted set.
    Always marks/comments (color + comment only, per the confirmed v1
    scope). If jump_patches is not None (only when the user explicitly
    opted in via "Also patch bytes" - it may still be an empty list),
    ALSO NOPs every checked dead instruction's bytes, redirects each
    listed conditional jump or collapsed indirect jump to its real
    target (see _patch_nop_instruction/_patch_jcc_to_real_target), and
    enforces that
    a proper function exists at start_ea (see _ensure_function_at) -
    only in patch mode, since forcing a function boundary is itself a
    database change beyond plain color/comment marking, consistent with
    everything else this option gates. If touched_ranges (a list) is
    given, every (start, end) byte range actually patched is appended to
    it, so the caller can record precisely what to revert later (Undo
    Patches - see _AppliedCfgPatch); coloring/comments are never
    recorded there, since those aren't bytes and Undo only reverts bytes.
    Colors are idc.set_color's native 0xBBGGRR (BGR, not RGB) packed ints;
    applied across every instruction in the block's range so the whole
    marked region is visually distinct, while the comment itself is only
    placed on the block's first instruction to avoid noisy per-line repeats.
    """
    color_by_verdict = {
        "real": config.cfg_trace_color_real,
        "dead": config.cfg_trace_color_dead,
        "unresolved": config.cfg_trace_color_unresolved,
    }
    for item in items:
        color = color_by_verdict.get(item.verdict)
        for idx, ea in enumerate(item.insn_eas):
            try:
                if color is not None:
                    idc.set_color(ea, idc.CIC_ITEM, color)
                if idx == 0 and item.comment:
                    idc.set_cmt(ea, item.comment, 0)
            except Exception:
                pass

    patch_log = []
    if jump_patches is not None:
        dead_insn_eas = set()
        for item in items:
            if item.verdict == "dead":
                dead_insn_eas.update(item.insn_eas)
        for ea in sorted(dead_insn_eas):
            try:
                size = _patch_nop_instruction(ea)
                if touched_ranges is not None:
                    touched_ranges.append((ea, ea + size))
            except Exception as exc:
                patch_log.append("Failed to NOP dead instruction at %#010x: %s" % (ea, exc))
        for jp in jump_patches:
            try:
                ok, reason, size = _patch_jcc_to_real_target(jp.jcc_ea, jp.real_target_ea, jp.role_of_real)
                if ok:
                    if touched_ranges is not None:
                        touched_ranges.append((jp.jcc_ea, jp.jcc_ea + size))
                else:
                    patch_log.append("Failed to patch jump at %#010x: %s" % (jp.jcc_ea, reason))
            except Exception as exc:
                patch_log.append("Failed to patch jump at %#010x: %s" % (jp.jcc_ea, exc))
        if start_ea is not None:
            try:
                msg = _ensure_function_at(start_ea)
                if msg:
                    patch_log.append(msg)
            except Exception as exc:
                patch_log.append("Failed to ensure a function at %#010x: %s" % (start_ea, exc))
    if patch_log:
        ida_kernwin.msg(
            "".join("[%s] %s\n" % (PLUGIN_NAME, line) for line in patch_log)
        )

    try:
        ida_kernwin.request_refresh(ida_kernwin.IWID_DISASM)
    except Exception:
        try:
            ida_kernwin.refresh_idaview_anyway()
        except Exception:
            pass
    return 1


class CfgTraceGraphView(ida_graph.GraphViewer):
    """Live, native IDA graph view of a CfgTraceRunner's progress -
    read-only observer over the runner's already-public state
    (visited/dead_blocks/unresolved), redrawn on demand via Refresh()
    from CfgTraceDialog's existing (already-throttled) log/status
    callbacks - no changes to CfgTraceRunner itself needed. Reuses the
    same cfg_trace_color_real/dead/unresolved config values as the final
    disassembly marking, so the graph and the marked disassembly agree
    visually. Double-click a node to jump the disassembly view there.
    """

    def __init__(self, config, runner, title):
        self.config = config
        self.runner = runner
        ida_graph.GraphViewer.__init__(self, title, close_open=True)

    def OnRefresh(self):
        self.Clear()
        node_by_ea = {}

        def add_node(ea, block, verdict, color):
            if ea in node_by_ea:
                return node_by_ea[ea]
            kind = block.kind if block is not None else "?"
            label = "%#010x\n%s (%s)" % (ea, verdict, kind)
            nid = self.AddNode((label, color, ea))
            node_by_ea[ea] = nid
            return nid

        for ea, block in self.runner.visited.items():
            add_node(ea, block, "REAL", self.config.cfg_trace_color_real)
        for ea, block in self.runner.dead_blocks.items():
            add_node(ea, block, "DEAD", self.config.cfg_trace_color_dead)
        for item in self.runner.unresolved:
            ea = item.get("ea")
            if ea is None:
                ea = item.get("anchor_ea")
            if ea is None:
                continue
            add_node(ea, item.get("block"), "UNRESOLVED", self.config.cfg_trace_color_unresolved)

        for ea, block in self.runner.visited.items():
            for succ in block.successors:
                if succ.ea is not None and succ.ea in node_by_ea:
                    self.AddEdge(node_by_ea[ea], node_by_ea[succ.ea])
        for ea, block in self.runner.dead_blocks.items():
            for succ in block.successors:
                if succ.ea is not None and succ.ea in node_by_ea:
                    self.AddEdge(node_by_ea[ea], node_by_ea[succ.ea])
        return True

    def OnGetText(self, node_id):
        label, color, _ea = self[node_id]
        return (label, color)

    def OnHint(self, node_id):
        _label, _color, ea = self[node_id]
        block = self.runner.visited.get(ea) or self.runner.dead_blocks.get(ea)
        if block is None:
            # Not real/dead - check unresolved too (the marking-only fetch
            # there often has a real BlockInfo available), so hovering an
            # UNRESOLVED node isn't silently less informative than the
            # other two verdicts.
            for item in self.runner.unresolved:
                if item.get("ea") == ea or item.get("anchor_ea") == ea:
                    block = item.get("block")
                    break
        if block is not None and block.insn_eas:
            try:
                return render_block_text(block.insn_eas)
            except Exception:
                pass
        return "%#010x" % ea

    def OnDblClick(self, node_id):
        _label, _color, ea = self[node_id]
        try:
            ida_kernwin.jumpto(ea)
        except Exception:
            pass
        return True

    def OnClose(self):
        # Without this, closing the graph tab by hand (the 'x' on the
        # widget, right-click Close, etc.) never told the plugin - it
        # kept the now-defunct CfgTraceGraphView in _open_graphs forever
        # (only pruned on plugin term()/IDA close), an unbounded leak
        # across a long session with many traces. Mirrors the .finished
        # signal cleanup _track_dialog already does for Qt dialogs.
        plugin = LLMExplainerPlugin.instance
        if plugin is not None:
            try:
                plugin._open_graphs.remove(self)
            except ValueError:
                pass


class CfgTraceDialog(QtWidgets.QDialog):
    """Three phases in one dialog, matching this plugin's human-in-the-loop
    philosophy (nothing is written to the database until Accept):
      1. Confirm - an editable, cursor-prefilled start address.
      2. Running - a live, cancellable transcript while CfgTraceRunner
         works, pausing for an LLM round-trip only at real decision points.
      3. Review - a checkbox table of every decided/flagged block; Accept
         applies the checked rows in one batch (color + comment only).
    """

    COL_APPLY, COL_RANGE, COL_VERDICT, COL_REASON = range(4)

    def __init__(self, config, start_ea, parent=None):
        super().__init__(parent)
        self.config = config
        self.start_ea = start_ea
        self.runner = None
        self.result = None
        self.graph_view = None
        self._log_lines_since_pump = 0
        self._last_graph_refresh_ts = 0.0
        self._refreshing_graph = False
        self._reasoning_log_dialog = None

        self.setWindowTitle("%s - Trace/Recover CFG" % PLUGIN_NAME)
        self.resize(760, 560)

        self.stack = QtWidgets.QStackedWidget()
        outer = QtWidgets.QVBoxLayout(self)
        outer.addWidget(self.stack, 1)
        _add_copyright_footer(outer)

        self._build_confirm_page()
        self._build_running_page()
        self._build_review_page()
        self.stack.setCurrentIndex(0)

    # -- phase 1: confirm start address --------------------------------

    def _build_confirm_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        info = QtWidgets.QLabel(
            "Traces basic blocks forward from the given address, asking the LLM at "
            "each branch or indirect-jump decision point which successor(s) are the "
            "real continuation of the original program and which are dead/decoy, "
            "until the control flow is fully recovered (or the block cap is hit). As "
            "it walks, it corrects any instruction boundaries IDA originally got "
            "wrong (undefine + recreate, matching pressing U then C - never changes "
            "a byte); this happens immediately, not deferred, since it's just fixing "
            "IDA's own analysis. The REAL/DEAD/UNRESOLVED coloring and comments "
            "themselves are NOT written until you review and accept the result on "
            "the next screen."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QtWidgets.QFormLayout()
        self.addr_edit = QtWidgets.QLineEdit("%#010x" % self.start_ea)
        form.addRow("Start address:", self.addr_edit)
        layout.addLayout(form)

        self.show_graph_check = QtWidgets.QCheckBox("Show live CFG graph while tracing")
        self.show_graph_check.setChecked(True)
        self.show_graph_check.setToolTip(
            "Opens a native IDA graph view, colored the same as the final disassembly "
            "marking (green/red/amber), that fills in live as the trace progresses. "
            "Double-click a node to jump the disassembly view there."
        )
        layout.addWidget(self.show_graph_check)

        self.cache_status_label = QtWidgets.QLabel("")
        self.cache_status_label.setWordWrap(True)
        layout.addWidget(self.cache_status_label)

        self.confirm_status_label = QtWidgets.QLabel("")
        palette = self.confirm_status_label.palette()
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#a33"))
        self.confirm_status_label.setPalette(palette)
        layout.addWidget(self.confirm_status_label)
        layout.addStretch(1)

        button_row = QtWidgets.QHBoxLayout()
        self.load_cached_button = QtWidgets.QPushButton("Load Cached Result")
        self.load_cached_button.setToolTip(
            "Skips straight to the review screen using the last trace result for this "
            "exact address, if one is still cached from earlier this IDA session - no "
            "re-tracing, no LLM calls. The cache reflects the database as it was at "
            "capture time; if you've since patched or modified this address, re-trace "
            "instead (accepting always re-verifies fresh bytes before writing regardless)."
        )
        self.load_cached_button.setEnabled(False)
        self.load_cached_button.clicked.connect(self._on_load_cached_clicked)
        button_row.addWidget(self.load_cached_button)
        self.undo_patch_button = QtWidgets.QPushButton("Undo Patches")
        self.undo_patch_button.setEnabled(False)
        self.undo_patch_button.clicked.connect(self._on_undo_patch_clicked)
        button_row.addWidget(self.undo_patch_button)
        button_row.addStretch(1)
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self._on_start_clicked)
        button_row.addWidget(self.start_button)
        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.close)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

        self.stack.addWidget(page)
        self.addr_edit.textChanged.connect(self._update_cache_status)
        self.addr_edit.textChanged.connect(self._update_undo_status)
        self._update_cache_status()
        self._update_undo_status()

    def _parse_addr_field(self):
        """Returns an int ea, or None if the address field's text doesn't
        parse - never raises, never touches the database (unlike
        _on_start_clicked's fuller validation, this just needs to answer
        "is there a cache entry for whatever's currently typed", re-run on
        every keystroke).
        """
        text = self.addr_edit.text().strip()
        for base in (0, 16):
            try:
                ea = int(text, base)
                return ea if ea != idaapi.BADADDR else None
            except ValueError:
                continue
        return None

    def _update_cache_status(self, *_args):
        plugin = LLMExplainerPlugin.instance
        ea = self._parse_addr_field()
        cached = plugin.get_cached_cfg_trace_result(ea) if plugin is not None and ea is not None else None
        if cached is None:
            self.cache_status_label.setText("")
            self.load_cached_button.setEnabled(False)
            return
        real_n = sum(1 for b in cached.result.blocks if b.verdict == "real")
        dead_n = sum(1 for b in cached.result.blocks if b.verdict == "dead")
        unresolved_n = sum(1 for b in cached.result.blocks if b.verdict == "unresolved")
        age_s = time.time() - cached.timestamp
        age_text = (
            "%d second(s) ago" % age_s if age_s < 60
            else "%d minute(s) ago" % (age_s // 60) if age_s < 3600
            else "%d hour(s) ago" % (age_s // 3600)
        )
        self.cache_status_label.setText(
            "Cached result available for this address (%s): %d real, %d dead, %d unresolved. "
            "Load it to review without re-tracing, or click Start to re-trace from scratch "
            "(replaces the cached result once finished)."
            % (age_text, real_n, dead_n, unresolved_n)
        )
        self.load_cached_button.setEnabled(True)

    def _on_load_cached_clicked(self):
        plugin = LLMExplainerPlugin.instance
        ea = self._parse_addr_field()
        cached = plugin.get_cached_cfg_trace_result(ea) if plugin is not None and ea is not None else None
        if cached is None:
            return
        self.start_ea = ea
        self.runner = cached.runner
        self.result = cached.result
        self.log_edit.clear()
        self.log_edit.appendPlainText("\n".join(cached.runner.trace_log))
        self._populate_review(cached.result)
        self.stack.setCurrentIndex(2)

    def _update_undo_status(self, *_args):
        plugin = LLMExplainerPlugin.instance
        ea = self._parse_addr_field()
        applied = plugin.get_applied_cfg_patch(ea) if plugin is not None and ea is not None else None
        if applied is None:
            self.undo_patch_button.setEnabled(False)
            self.undo_patch_button.setToolTip("")
            return
        total_bytes = sum(end - start for start, end in applied.touched_ranges)
        mode_label = "Patch in place" if applied.mode == "patch_in_place" else "Rebuild linear"
        self.undo_patch_button.setEnabled(True)
        self.undo_patch_button.setToolTip(
            "Reverts the %d byte(s) this plugin's \"%s\" patch touched here back to their "
            "original values, using IDA's own recorded pre-patch bytes (same effect as "
            "Edit > Patches > ..., but scoped to exactly this patch - nothing else this "
            "trace didn't itself write is touched). If you've since re-patched this address "
            "some other way, only the most recently accepted patch's own ranges are known "
            "here and would be reverted."
            % (total_bytes, mode_label)
        )

    def _on_undo_patch_clicked(self):
        plugin = LLMExplainerPlugin.instance
        ea = self._parse_addr_field()
        applied = plugin.get_applied_cfg_patch(ea) if plugin is not None and ea is not None else None
        if applied is None:
            return
        total_bytes = sum(end - start for start, end in applied.touched_ranges)
        mode_label = "Patch in place" if applied.mode == "patch_in_place" else "Rebuild linear"
        answer = QtWidgets.QMessageBox.question(
            self, "Undo Patches - %s" % PLUGIN_NAME,
            "This will revert %d byte(s) across %d range(s) back to their original values "
            "(from the \"%s\" patch accepted earlier at %#010x), using IDA's own recorded "
            "pre-patch bytes. This does not touch the color/comment marking, and does not "
            "remove any function definition that patch may have created. Continue?"
            % (total_bytes, len(applied.touched_ranges), mode_label, ea),
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        result_holder = {}

        def _do_undo():
            result_holder["count"] = _undo_applied_cfg_patch(applied)
            return 1

        ida_kernwin.execute_sync(_do_undo, ida_kernwin.MFF_WRITE)
        plugin.forget_applied_cfg_patch(ea)
        self._update_undo_status()
        QtWidgets.QMessageBox.information(
            self, "Undo Patches - %s" % PLUGIN_NAME,
            "Reverted %d byte(s)." % result_holder.get("count", 0),
        )

    def _on_start_clicked(self):
        text = self.addr_edit.text().strip()
        ea = idaapi.BADADDR
        for base in (0, 16):
            try:
                ea = int(text, base)
                break
            except ValueError:
                continue
        if ea == idaapi.BADADDR:
            self.confirm_status_label.setText("Could not parse that as an address.")
            return
        probe = ida_ua.insn_t()
        try:
            decoded = ida_ua.decode_insn(probe, ea) > 0
        except Exception:
            decoded = False
        if not decoded:
            self.confirm_status_label.setText("%#010x does not decode as a valid instruction." % ea)
            return
        self.start_ea = ea
        self._begin_trace()

    # -- phase 2: running -------------------------------------------------

    def _build_running_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.running_status_label = QtWidgets.QLabel("Starting...")
        layout.addWidget(self.running_status_label)
        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QtGui.QFont("Consolas", 9))
        layout.addWidget(self.log_edit, 1)
        button_row = QtWidgets.QHBoxLayout()
        self.show_graph_button = QtWidgets.QPushButton("Show Graph")
        self.show_graph_button.clicked.connect(self._ensure_graph_view)
        button_row.addWidget(self.show_graph_button)
        button_row.addStretch(1)
        self.running_cancel_button = QtWidgets.QPushButton("Cancel")
        self.running_cancel_button.clicked.connect(self._on_cancel_trace)
        button_row.addWidget(self.running_cancel_button)
        layout.addLayout(button_row)
        self.stack.addWidget(page)

    def _begin_trace(self):
        self.stack.setCurrentIndex(1)
        self.log_edit.clear()
        self._log_lines_since_pump = 0
        self.runner = CfgTraceRunner(
            self.config, self.start_ea, on_log=self._on_trace_log, on_status=self._on_trace_status,
        )
        if self.show_graph_check.isChecked():
            self._ensure_graph_view()
        self.runner.start(self._on_trace_finished)

    def _show_reasoning_log(self):
        if self.runner is None:
            return
        # Non-modal (so the review table stays usable alongside it) - kept
        # as an instance attribute rather than a bare local, since nothing
        # else would otherwise hold a Python reference to it and it could
        # get garbage-collected (and the window closed from under the
        # user) as soon as this method returns.
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Trace Reasoning - %s" % PLUGIN_NAME)
        dlg.resize(720, 520)
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        layout = QtWidgets.QVBoxLayout(dlg)
        text_edit = QtWidgets.QPlainTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QtGui.QFont("Consolas", 9))
        text_edit.setPlainText("\n".join(self.runner.trace_log))
        layout.addWidget(text_edit, 1)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(dlg.close)
        layout.addWidget(close_button)
        self._reasoning_log_dialog = dlg
        dlg.show()

    def _ensure_graph_view(self):
        if self.runner is None:
            return
        plugin = LLMExplainerPlugin.instance
        if plugin is None:
            return
        if self.graph_view is None:
            title = "CFG Trace - %#010x" % self.start_ea
            self.graph_view = plugin.open_cfg_trace_graph(self.runner, title)
        else:
            # Show() returns a bool - a manually-closed graph can come
            # back False without raising at all, so checking only for an
            # exception here missed that case and the button would
            # appear to do nothing.
            reshown = False
            try:
                reshown = bool(self.graph_view.Show())
            except Exception:
                reshown = False
            if not reshown:
                self.graph_view = plugin.open_cfg_trace_graph(self.runner, "CFG Trace - %#010x" % self.start_ea)
        self._refresh_graph_view(force=True)

    def _refresh_graph_view(self, force=False):
        if self.graph_view is None or self._refreshing_graph:
            return
        # A graph Refresh() means a full Clear() + rebuild + IDA's own
        # re-layout of the whole graph - much more expensive than a log
        # append, so this is throttled by WALL-CLOCK TIME rather than
        # block count. Must be called on every block (see _on_trace_log)
        # rather than gated behind a block-count threshold: a trace
        # short enough to finish before that threshold was ever reached
        # (very plausible, especially now that the symbolic engine
        # resolves most branches instantly) would otherwise never
        # refresh the graph at all until the very end.
        now = time.monotonic()
        if not force and (now - self._last_graph_refresh_ts) < 0.5:
            return
        self._last_graph_refresh_ts = now
        # processEvents() below can dispatch a queued click (e.g. "Show
        # Graph" clicked again while this call is still on the stack),
        # which would otherwise re-enter this same method - guard against
        # that rather than risk nested Clear()/rebuild calls stepping on
        # each other.
        self._refreshing_graph = True
        try:
            self._do_refresh_graph_view()
        finally:
            self._refreshing_graph = False

    def _do_refresh_graph_view(self):
        try:
            self.graph_view.Refresh()
        except Exception:
            pass
        # Refresh() only triggers OnRefresh() to rebuild the node/edge
        # data - it does not force an immediate repaint. During a long
        # auto-resolve run, _pump() can process many blocks synchronously
        # inside a single execute_sync(MFF_WRITE) call without ever
        # yielding back to IDA's own UI loop, so that rebuilt data was
        # visibly queued but never actually painted until the call
        # returned - refresh_viewer() is the lower-level "redraw this
        # widget now" call that Refresh() doesn't itself guarantee, and
        # processEvents() right after gives IDA's event loop an actual
        # chance to run that repaint before the tight loop continues.
        try:
            gv = self.graph_view.GetWidgetAsGraphViewer()
            if gv is not None:
                ida_graph.refresh_viewer(gv)
        except Exception:
            pass
        try:
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

    def _on_trace_log(self, text):
        try:
            self.log_edit.appendPlainText(text)
        except RuntimeError:
            return
        # Self-throttled internally (see above) - safe/cheap to call on
        # every block rather than gating it behind a block count.
        self._refresh_graph_view()
        # Even a long silent auto-resolve chain (zero LLM calls) should keep
        # the log visibly growing and Cancel responsive, not freeze the UI
        # until the next network round-trip - kept as a coarser fallback
        # pump independent of whether a graph view is even open.
        self._log_lines_since_pump += 1
        if self._log_lines_since_pump >= 20:
            self._log_lines_since_pump = 0
            QtWidgets.QApplication.processEvents()

    def _on_trace_status(self, text):
        try:
            self.running_status_label.setText(text)
        except RuntimeError:
            pass

    def _on_cancel_trace(self):
        if self.runner is not None:
            self.runner.cancel()
        self.close()

    def _on_trace_finished(self, result):
        self.result = result
        self._refresh_graph_view(force=True)
        self._populate_review(result)
        self.stack.setCurrentIndex(2)
        plugin = LLMExplainerPlugin.instance
        if plugin is not None and self.runner is not None:
            # Cached only on a natural stop (fully explored or block-cap
            # partial) - not on user Cancel, which can leave the runner
            # mid-decision-point in a less meaningful state to revisit.
            plugin.cache_cfg_trace_result(self.start_ea, self.runner, result)

    # -- phase 3: review ----------------------------------------------------

    def _build_review_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.summary_label = QtWidgets.QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.review_table = QtWidgets.QTableWidget(0, 4)
        self.review_table.setHorizontalHeaderLabels(["Apply", "Address range", "Verdict", "Reason"])
        self.review_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.review_table, 1)

        mode_box = QtWidgets.QGroupBox("On Accept:")
        mode_layout = QtWidgets.QVBoxLayout(mode_box)
        self.mode_group = QtWidgets.QButtonGroup(self)

        self.mode_mark_only = QtWidgets.QRadioButton("Mark only (color + comment, no bytes changed)")
        self.mode_mark_only.setChecked(True)
        self.mode_group.addButton(self.mode_mark_only)
        mode_layout.addWidget(self.mode_mark_only)

        self.mode_patch_in_place = QtWidgets.QRadioButton(
            "Patch in place (redirect confirmed opaque-predicate jumps to their real target, "
            "NOP dead code)"
        )
        self.mode_patch_in_place.setToolTip(
            "Unlike Mark only, this changes actual code bytes, not just IDB metadata (though "
            "it is revertible via Edit > Patches). Only redirects a conditional_branch block "
            "where BOTH successors were fully and oppositely decided (one real, one dead) AND "
            "both corresponding rows are checked above - never a genuine data-dependent branch "
            "(both sides real) or anything touching an unresolved verdict. You will be asked to "
            "confirm the exact count before anything is patched."
        )
        self.mode_group.addButton(self.mode_patch_in_place)
        mode_layout.addWidget(self.mode_patch_in_place)

        self.mode_rebuild_linear = QtWidgets.QRadioButton(
            "Rebuild linear (replace the entry point with a straight-line rebuild of the real "
            "code only)"
        )
        self.mode_rebuild_linear.setToolTip(
            "Different from the other two options: leaves every original byte in the binary "
            "untouched EXCEPT the function's own entry point through the end of the rebuilt "
            "code. Every confirmed-real block (not subject to the checkboxes above, which only "
            "control coloring - a partial rebuild missing some of the real control flow would "
            "not be safe) is concatenated into one straight-line sequence with every jump/call "
            "re-encoded explicitly, written starting exactly at the entry point, using space up "
            "to the furthest address any block this trace examined reached - crossing ordinary "
            "unexplored gaps too, unless a gap byte belongs to a different existing function or "
            "is referenced from outside this trace. A block using RIP-"
            "relative addressing or a live multi-case indirect dispatch cannot be safely moved "
            "and is left at its original address untouched, PROVIDED that doesn't overlap the "
            "rebuilt range - otherwise the whole rebuild is refused rather than guessed at, "
            "same for the entry block itself and if the rebuild simply doesn't fit."
        )
        self.mode_group.addButton(self.mode_rebuild_linear)
        mode_layout.addWidget(self.mode_rebuild_linear)

        self.mode_group.buttonToggled.connect(self._on_patch_mode_changed)
        layout.addWidget(mode_box)

        button_row = QtWidgets.QHBoxLayout()
        self.review_show_graph_button = QtWidgets.QPushButton("Show Graph")
        self.review_show_graph_button.clicked.connect(self._ensure_graph_view)
        button_row.addWidget(self.review_show_graph_button)
        self.review_show_log_button = QtWidgets.QPushButton("View Reasoning Log")
        self.review_show_log_button.setToolTip(
            "Opens the full trace transcript in a separate window - every decision, "
            "symbolic/LLM, in order. Works the same whether this result just finished "
            "or was loaded from the cache."
        )
        self.review_show_log_button.clicked.connect(self._show_reasoning_log)
        button_row.addWidget(self.review_show_log_button)
        button_row.addStretch(1)
        self.accept_button = QtWidgets.QPushButton("Accept && Mark Disassembly")
        self.accept_button.clicked.connect(self._on_accept)
        button_row.addWidget(self.accept_button)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

        self.stack.addWidget(page)

    def _on_patch_mode_changed(self, button, checked):
        if not checked:
            return
        if button is self.mode_rebuild_linear:
            self.accept_button.setText("Accept && Rebuild Linear")
        elif button is self.mode_patch_in_place:
            self.accept_button.setText("Accept && Mark/Patch Disassembly")
        else:
            self.accept_button.setText("Accept && Mark Disassembly")

    def _populate_review(self, result):
        status = "Partial (block cap reached)" if result.partial else "Complete"
        real_n = sum(1 for b in result.blocks if b.verdict == "real")
        dead_n = sum(1 for b in result.blocks if b.verdict == "dead")
        unresolved_n = sum(1 for b in result.blocks if b.verdict == "unresolved")
        extra = " %d address(es) left unexplored." % len(result.unexplored) if result.unexplored else ""
        sym_note = (
            " %d decision point(s) resolved automatically (no LLM call)." % result.symbolic_resolved_count
            if result.symbolic_resolved_count else ""
        )
        self.summary_label.setText(
            "%s - %d block(s) processed. %d real, %d dead, %d unresolved.%s%s"
            % (status, result.blocks_processed, real_n, dead_n, unresolved_n, sym_note, extra)
        )

        rows = sorted(result.blocks, key=lambda r: (r.start_ea, r.verdict))
        self.review_table.setRowCount(len(rows))
        self._review_rows = rows
        for i, rec in enumerate(rows):
            check_item = QtWidgets.QTableWidgetItem()
            check_item.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            default_checked = rec.verdict in ("real", "dead")
            check_item.setCheckState(
                QtCore.Qt.CheckState.Checked if default_checked else QtCore.Qt.CheckState.Unchecked
            )
            self.review_table.setItem(i, self.COL_APPLY, check_item)
            range_text = (
                "%#010x" % rec.start_ea if rec.start_ea == rec.end_ea
                else "%#010x - %#010x" % (rec.start_ea, rec.end_ea)
            )
            self.review_table.setItem(i, self.COL_RANGE, QtWidgets.QTableWidgetItem(range_text))
            self.review_table.setItem(i, self.COL_VERDICT, QtWidgets.QTableWidgetItem(rec.verdict.upper()))
            self.review_table.setItem(i, self.COL_REASON, QtWidgets.QTableWidgetItem(rec.reason))

    def _build_comment(self, rec):
        prefix = {
            "real": "[CFG trace: real path]",
            "dead": "[CFG trace: dead/decoy]",
            "unresolved": "[CFG trace: unresolved - needs manual review]",
        }[rec.verdict]
        return "%s %s" % (prefix, rec.reason) if rec.reason else prefix

    def _on_accept(self):
        items = []
        checked_by_addr_verdict = set()
        for i, rec in enumerate(self._review_rows):
            if self.review_table.item(i, self.COL_APPLY).checkState() != QtCore.Qt.CheckState.Checked:
                continue
            items.append(_CfgTraceApplyItem(
                insn_eas=rec.insn_eas, verdict=rec.verdict, comment=self._build_comment(rec),
            ))
            checked_by_addr_verdict.add((rec.start_ea, rec.verdict))
        if not items:
            self.close()
            return

        if self.mode_rebuild_linear.isChecked():
            self._on_accept_rebuild_linear(items)
            return

        jump_patches = None
        if self.mode_patch_in_place.isChecked():
            # A conditional-branch patch (role_of_real "jump_target"/
            # "fallthrough") only ever gets applied when BOTH of its
            # successors' rows are checked - if the user unchecked either
            # side, treat that jcc as "not confirmed enough to touch
            # bytes for", same as if patching were off entirely for that
            # one. A collapsed-indirect-jump patch (role_of_real "only")
            # has no dead side to require - it's gated on its single real
            # target alone.
            all_patches = self.result.jump_patches if self.result is not None else []
            jump_patches = [
                jp for jp in all_patches
                if (jp.real_target_ea, "real") in checked_by_addr_verdict
                and (
                    jp.role_of_real == "only"
                    or (jp.dead_target_ea, "dead") in checked_by_addr_verdict
                )
            ]
            dead_n = sum(1 for it in items if it.verdict == "dead")
            cond_n = sum(1 for jp in jump_patches if jp.role_of_real != "only")
            indirect_n = sum(1 for jp in jump_patches if jp.role_of_real == "only")
            answer = QtWidgets.QMessageBox.question(
                self, "Patch Bytes - %s" % PLUGIN_NAME,
                "This will modify the loaded binary image, not just IDB colors/comments:\n\n"
                "  - NOP out %d dead instruction address(es)\n"
                "  - Redirect %d conditional jump(s) to their confirmed real target\n"
                "  - Replace %d indirect jump(s) that always land on the same address "
                "with a direct jump to it\n"
                "  - Ensure a function is defined at %#010x (IDA sometimes never\n"
                "    recognized it as one), so it can be decompiled\n\n"
                "This is revertible via Edit > Patches > ... in IDA if needed. Continue?"
                % (dead_n, cond_n, indirect_n, self.start_ea),
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        touched_ranges = []
        ida_kernwin.execute_sync(
            functools.partial(
                _apply_cfg_trace_and_refresh, self.config, items, jump_patches, self.start_ea, touched_ranges,
            ),
            ida_kernwin.MFF_WRITE,
        )
        plugin = LLMExplainerPlugin.instance
        if plugin is not None:
            plugin.record_applied_cfg_patch(self.start_ea, "patch_in_place", touched_ranges)
        self.accept_button.setEnabled(False)
        self.close()

    def _on_accept_rebuild_linear(self, items):
        if self.runner is None:
            self.close()
            return
        plan_holder = {}

        def _compute():
            plan_holder["plan"] = _gather_linear_rebuild_plan(self.runner)
            return 1

        ida_kernwin.execute_sync(_compute, ida_kernwin.MFF_WRITE)
        plan = plan_holder.get("plan")
        if plan is None or plan.error is not None:
            QtWidgets.QMessageBox.warning(
                self, "Rebuild Linear - %s" % PLUGIN_NAME,
                "Cannot rebuild: %s" % (plan.error if plan is not None else "internal error"),
            )
            return

        anchored_note = (
            "%d block(s) could not be safely relocated (RIP-relative addressing or a live "
            "multi-case dispatch) and were left at their original address untouched.\n\n"
            % plan.anchored_count
            if plan.anchored_count else ""
        )
        answer = QtWidgets.QMessageBox.question(
            self, "Rebuild Linear - %s" % PLUGIN_NAME,
            "This will replace %d byte(s) starting at the function's entry point (%#010x) with "
            "a straight-line rebuild of %d confirmed-real block(s). %s"
            "Nothing else in the binary is modified - every other address (dead code, and any "
            "block left untouched above) keeps its exact original bytes.\n\n"
            "This is revertible via Edit > Patches > ... in IDA if needed. Continue?"
            % (len(plan.code), plan.entry_ea, plan.relocated_count, anchored_note),
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        ida_kernwin.execute_sync(
            functools.partial(_apply_cfg_trace_and_refresh, self.config, items, None, self.start_ea),
            ida_kernwin.MFF_WRITE,
        )
        ida_kernwin.execute_sync(functools.partial(_apply_linear_rebuild, plan), ida_kernwin.MFF_WRITE)
        plugin = LLMExplainerPlugin.instance
        if plugin is not None:
            plugin.record_applied_cfg_patch(
                self.start_ea, "rebuild_linear", [(plan.entry_ea, plan.entry_ea + len(plan.code))],
            )
        self.accept_button.setEnabled(False)
        self.close()

    def closeEvent(self, event):
        if self.runner is not None:
            self.runner.cancel()
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


class CfgTraceActionHandler(ida_kernwin.action_handler_t):
    """Disassembly-only (see update()): unlike ExplainActionHandler, this
    does NOT gate on _resolve_func succeeding - the whole premise is a
    blob that may not be recognized as a function at all, and every
    primitive this feature needs (decode_insn, CodeRefsFrom, switch_info_t)
    is instruction-address-level, not function-level.
    """

    def __init__(self):
        super().__init__()

    def activate(self, ctx):
        ea = ida_kernwin.get_screen_ea()
        if ea == idaapi.BADADDR:
            ida_kernwin.warning("Place the cursor on an address first.")
            return 0
        plugin = LLMExplainerPlugin.instance
        if plugin is None:
            return 0
        plugin.open_cfg_trace_dialog(ea)
        return 1

    def update(self, ctx):
        widget = getattr(ctx, "widget", None)
        wtype = ida_kernwin.get_widget_type(widget) if widget else -1
        if wtype != ida_kernwin.BWN_DISASM:
            return ida_kernwin.AST_DISABLE_FOR_WIDGET
        return ida_kernwin.AST_ENABLE_FOR_WIDGET


class PopupHooks(ida_kernwin.UI_Hooks):
    def finish_populating_widget_popup(self, widget, popup, ctx=None):
        wtype = ida_kernwin.get_widget_type(widget)
        if wtype == ida_kernwin.BWN_PSEUDOCODE:
            ida_kernwin.attach_action_to_popup(widget, popup, ACTION_ID_EXPLAIN, "LLM Explainer/")
            ida_kernwin.attach_action_to_popup(widget, popup, ACTION_ID_EXPLAIN_RECURSIVE, "LLM Explainer/")
        elif wtype == ida_kernwin.BWN_DISASM:
            ida_kernwin.attach_action_to_popup(widget, popup, ACTION_ID_EXPLAIN, "LLM Explainer/")
            ida_kernwin.attach_action_to_popup(widget, popup, ACTION_ID_EXPLAIN_RECURSIVE, "LLM Explainer/")
            ida_kernwin.attach_action_to_popup(widget, popup, ACTION_ID_TRACE_CFG, "LLM Explainer/")
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
        "batch-explain multiple functions. In the disassembly view, "
        "'Trace/Recover CFG...' asks the LLM to walk an obfuscated "
        "function's real control flow branch by branch and mark real/dead "
        "code (color + comment by default; 'Patch in place' can "
        "additionally NOP dead code and redirect confirmed opaque-"
        "predicate jumps, or 'Rebuild linear' replaces the entry point "
        "with a straight-line rebuild of just the real code). "
        "Edit > Plugins > LLM Explainer opens settings."
    )
    wanted_name = PLUGIN_NAME
    wanted_hotkey = ""

    instance = None

    def init(self):
        LLMExplainerPlugin.instance = self
        self.config = PluginConfig.load()
        self._open_dialogs = []
        self._open_graphs = []
        self._trace_cache = OrderedDict()  # start_ea -> _CachedCfgTrace, oldest-first
        self._applied_cfg_patches = OrderedDict()  # start_ea -> _AppliedCfgPatch, oldest-first
        self._action_handler = ExplainActionHandler()
        self._recursive_action_handler = ExplainRecursiveActionHandler()
        self._batch_action_handler = BatchActionHandler()
        self._trace_cfg_action_handler = CfgTraceActionHandler()

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

        trace_cfg_action = ida_kernwin.action_desc_t(
            ACTION_ID_TRACE_CFG,
            "Trace/Recover CFG...",
            self._trace_cfg_action_handler,
            None,
            "Ask the local LLM to trace this obfuscated function's real control "
            "flow, branch by branch, and mark real/dead code in the disassembly",
            -1,
        )
        if not ida_kernwin.register_action(trace_cfg_action):
            ida_kernwin.msg("[%s] Failed to register CFG trace action.\n" % PLUGIN_NAME)

        self._popup_hooks = PopupHooks()
        self._popup_hooks.hook()

        ida_kernwin.msg(
            "[%s] v%s loaded. Server(s): %s\n"
            % (
                PLUGIN_NAME,
                PLUGIN_VERSION,
                ", ".join(self.config.server_label_full(u) for u in self.config.server_urls),
            )
        )
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
        for graph in list(self._open_graphs):
            try:
                graph.Close()
            except Exception:
                pass
        self._open_graphs = []
        self._trace_cache = OrderedDict()
        self._applied_cfg_patches = OrderedDict()
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
        try:
            ida_kernwin.unregister_action(ACTION_ID_TRACE_CFG)
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

    def open_cfg_trace_dialog(self, start_ea):
        dlg = CfgTraceDialog(self.config, start_ea)
        return self._track_dialog(dlg)

    _TRACE_CACHE_MAX = 20

    def cache_cfg_trace_result(self, start_ea, runner, result):
        self._trace_cache[start_ea] = _CachedCfgTrace(runner=runner, result=result, timestamp=time.time())
        self._trace_cache.move_to_end(start_ea)
        while len(self._trace_cache) > self._TRACE_CACHE_MAX:
            self._trace_cache.popitem(last=False)  # evict oldest

    def get_cached_cfg_trace_result(self, start_ea):
        cached = self._trace_cache.get(start_ea)
        if cached is not None:
            self._trace_cache.move_to_end(start_ea)  # LRU-ish: recently viewed survives longer
        return cached

    def record_applied_cfg_patch(self, start_ea, mode, touched_ranges):
        """touched_ranges is only ever recorded when non-empty (Mark only
        never calls this at all - nothing was written). A second Accept
        at the same address REPLACES the previous record rather than
        merging with it - Undo then only reverts the most recent
        patch's own ranges, not anything an earlier one at this same
        address may have separately touched.
        """
        if not touched_ranges:
            return
        self._applied_cfg_patches[start_ea] = _AppliedCfgPatch(
            mode=mode, touched_ranges=list(touched_ranges), timestamp=time.time(),
        )
        self._applied_cfg_patches.move_to_end(start_ea)
        while len(self._applied_cfg_patches) > self._TRACE_CACHE_MAX:
            self._applied_cfg_patches.popitem(last=False)

    def get_applied_cfg_patch(self, start_ea):
        applied = self._applied_cfg_patches.get(start_ea)
        if applied is not None:
            self._applied_cfg_patches.move_to_end(start_ea)
        return applied

    def forget_applied_cfg_patch(self, start_ea):
        self._applied_cfg_patches.pop(start_ea, None)

    def open_cfg_trace_graph(self, runner, title):
        """Kept alive independently of whichever CfgTraceDialog created
        it (a plain Python reference must stay live for as long as the
        native graph widget is open), and closed on plugin term()."""
        graph = CfgTraceGraphView(self.config, runner, title)
        self._open_graphs.append(graph)
        graph.Show()
        return graph

    def open_recursive_explain(self, func):
        """Target function plus its direct callees only (depth 1 - callees
        of callees are not included), each explained and auto-applied.
        Capped by config.max_recursive_callees, deliberately a smaller,
        separate setting from max_callees since this writes to the
        database automatically.

        The target itself is always explained (the user invoked this ON
        it), but callees are limited to ones that are still UNDISCOVERED -
        i.e. carry a default auto-generated name (sub_/loc_/nullsub_/...).
        A callee that already has a meaningful name is left alone rather
        than re-analyzed (and, importantly, never at risk of having its
        existing name/comment auto-overwritten), since the point of the
        recursive pass is to fill in the unnamed ones. See
        is_auto_generated_name."""
        def _is_undiscovered(callee):
            name = ida_funcs.get_func_name(callee.start_ea) or ""
            return is_auto_generated_name(name)

        callees = gather_callee_funcs(
            func, self.config.max_recursive_callees, include=_is_undiscovered
        )
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
