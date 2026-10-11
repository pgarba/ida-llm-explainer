"""Standalone smoke test for the auto-compact logic.

Stubs the IDA/Qt modules so llm_explainer can be imported outside IDA, then
drives ConversationRunner's context-window machinery against a fake
llama-server (real HTTP on localhost).
"""
import json
import os
import sys
import threading
import types
import urllib.request

# ---------------------------------------------------------------------------
# Stub IDA + Qt modules
# ---------------------------------------------------------------------------
def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod

class _Any:
    """A universal no-op: callable, chainable, and coercible to the types
    the plugin code expects (str/int/float/bool/iterable)."""
    def __call__(self, *a, **k):
        return _Any()
    def __getattr__(self, item):
        return _Any()
    def __str__(self):
        return ""
    def __int__(self):
        return 0
    def __float__(self):
        return 0.0
    def __bool__(self):
        return False
    def __iter__(self):
        return iter(())
    def __or__(self, other):
        return _Any()
    def __and__(self, other):
        return _Any()
    def __add__(self, other):
        return _Any()
    def __radd__(self, other):
        return _Any()
    def __sub__(self, other):
        return _Any()
    def __rsub__(self, other):
        return _Any()
    def __mul__(self, other):
        return _Any()
    def __rmul__(self, other):
        return _Any()
    def __truediv__(self, other):
        return _Any()
    def __floordiv__(self, other):
        return _Any()
    def __mod__(self, other):
        return _Any()
    def __neg__(self):
        return _Any()
    def __len__(self):
        return 0
    def __eq__(self, other):
        return True
    def __ne__(self, other):
        return False
    def __lt__(self, other):
        return False
    def __gt__(self, other):
        return False
    def __le__(self, other):
        return False
    def __ge__(self, other):
        return False

# Qt accessors whose return type the plugin code depends on
_QT_STR_METHODS = {"text", "toPlainText", "toHtml", "currentText", "placeholderText"}
_QT_NUM_METHODS = {"value", "valueFromText", "count", "rowCount", "columnCount"}
_QT_BOOL_METHODS = {"isChecked", "isEnabled", "isVisible", "hasFocus"}

def _qt_method(name):
    if name in _QT_STR_METHODS:
        return (lambda *a, **k: "")
    if name in _QT_NUM_METHODS:
        return (lambda *a, **k: 0)
    if name in _QT_BOOL_METHODS:
        return (lambda *a, **k: False)
    # everything else (signals like `clicked`, setters, ...) is a chainable
    # no-op that also coerces to a sane value if called and used as one
    return _Any()

class _DummyMeta(type):
    """Metaclass so class-level Qt attribute access (e.g.
    QLineEdit.EchoMode.Password) resolves to a no-op instead of raising."""
    def __getattr__(cls, item):
        return _Any()

def _dummy_class(name):
    """A class that can be subclassed and instantiated with anything, and
    whose class-level attributes (Qt enums) also resolve to no-ops.
    The metaclass is called directly (not via type(..., metaclass=...))
    because Python 3.14 no longer consumes the metaclass keyword in type()."""
    return _DummyMeta(name, (), {
        "__init__": lambda self, *a, **k: None,
        "__getattr__": lambda self, item: _qt_method(item),
    })

class _ActionHandler:
    pass

class _UIHooks:
    def __init__(self, *a, **k):
        pass
    def hook(self, *a, **k):
        pass
    def unhook(self, *a, **k):
        pass

class _PluginT:
    def term(self): return 0
    def run(self, arg): return 0
    def init(self): return 0

_stub("idaapi", get_user_idadir=lambda: os.path.join(os.path.dirname(__file__), "test_idadir"),
      BADADDR=0xFFFFFFFFFFFFFFFF, action_handler_t=_ActionHandler, plugin_t=_PluginT,
      PLUGIN_KEEP=1, get_name=lambda ea: None, get_memstr=lambda ea, sz: None)
_stub("idautils", XrefsTo=lambda *a, **k: iter(()))
_stub("idc")
_stub("ida_kernwin", MFF_FAST=1, MFF_WRITE=2, msg=lambda s: print("[msg]", s),
      # real execute_sync returns non-zero on success; the stub runs the
      # callable inline and reports success
      execute_sync=lambda fn, *a: (fn(), 1), UI_Hooks=_UIHooks,
      action_handler_t=_ActionHandler, get_kernel_version=lambda: "stub",
      get_screen_ea=lambda: 0x1000)
for extra in ("ida_auto", "ida_bytes", "ida_idp", "ida_lines",
              "ida_name", "ida_segment", "ida_typeinf", "ida_ua",
              "ida_xref", "ida_x86", "ida_arm", "ida_entry", "ida_strlist"):
    _stub(extra)
_stub("ida_loader", get_path=lambda kind: "/tmp/stub.i64", PATH_TYPE_IDB=1)
# AgentServer reads the IDB name for /health and the instance registry (v1.13+)
_stub("ida_nalt", get_root_filename=lambda: "stub.bin", get_input_file_path=lambda: "/tmp/stub.bin")
_stub("ida_netnode", BADNODE=0xFFFFFFFF)
_stub("ida_funcs", get_func=lambda ea: None, get_func_name=lambda ea: "sub_%X" % ea)
_stub("ida_graph", GraphViewer=_dummy_class("GraphViewer"))

# Functional ida_dbg/ida_idd stubs so the agent's debugger tools can run
# against a fake "suspended" debuggee.
class _StubRegval:
    def __init__(self, ival):
        self.rvtype = 0  # RVT_INT
        self.ival = ival
    def pyval(self, dtype):
        return self.ival

class _StubRegvals:
    def __init__(self, vals):
        self._vals = vals
    def size(self):
        return len(self._vals)
    def __getitem__(self, i):
        return self._vals[i]

class _StubBpt:
    ea = 0x140001234
    type = 0
    pass_count = 0
    def enabled(self):
        return True
    def is_active(self):
        return True

class _StubFrame:
    def __init__(self, callea, funcea, funcok=True):
        self.callea = callea
        self.funcea = funcea
        self.funcok = funcok

class _StubCallStack:
    def __init__(self):
        self.frames = []
    def __iter__(self):
        return iter(self.frames)

_dbg_state = {"state": 0}  # 0 = DSTATE_SUSP

_stub("ida_dbg",
      DSTATE_SUSP=0, DSTATE_NOTASK=1, DSTATE_RUN=2, dbg_process_attach=1,
      is_debugger_on=lambda: True,
      get_process_state=lambda: _dbg_state["state"],
      dbg_can_query=lambda: True,
      get_thread_qty=lambda: 1,
      getn_thread=lambda n: 1,
      getn_thread_name=lambda n: "main",
      get_current_thread=lambda: 1,
      select_thread=lambda tid: True,
      get_ip_val=lambda: 0x140001000,
      get_sp_val=lambda: 0x200000000,
      get_reg_vals=lambda tid, clsmask=-1: _StubRegvals([_StubRegval(0x11223344), _StubRegval(0x55)]),
      get_bpt_qty=lambda: 1,
      bpt_t=_StubBpt,
      getn_bpt=lambda n, bpt: True,
      exist_bpt=lambda ea: False,
      add_bpt=lambda *a: True,
      del_bpt=lambda ea: True,
      set_reg_val=lambda name, val: True,
      suspend_process=lambda: (_dbg_state.update(state=0) or True),
      continue_process=lambda: (_dbg_state.update(state=2) or True),
      step_into=lambda: True,
      step_over=lambda: True,
      step_until_ret=lambda: True,
      run_to=lambda ea: True,
      detach_process=lambda: (_dbg_state.update(state=1) or True),
      exit_process=lambda: (_dbg_state.update(state=1) or True),
      attach_process=lambda pid, ev: (_dbg_state.update(state=0) or 1),
      start_process=lambda path, args="", sdir="": (_dbg_state.update(state=0) or 1),
      collect_stack_trace=lambda tid, trace: (trace.frames.append(_StubFrame(0x140001234, 0x140001000)) or True),
      get_module_info=lambda ea, mi: False,
      )
_stub("ida_idd",
      RVT_INT=0, RVT_FLOAT=1, RVT_UNAVAILABLE=2,
      dbg_get_name=lambda: "stubdbg",
      dbg_get_registers=lambda: [("rax", 0, 0, 5, None, 0), ("rbx", 0, 0, 5, None, 0)],
      dbg_read_memory=lambda ea, sz: b"hello-live",
      dbg_write_memory=lambda ea, buf: True,
      dbg_get_memory_info=lambda: [(0x140000000, 0x140001000, "stub.exe", "TEXT", 0x140000000, 2, "r-x")],
      call_stack_t=_StubCallStack,
      modinfo_t=_dummy_class("modinfo_t"),
      )

# PySide6 stubs: dummy classes (subclassable, instantiable with anything)
_QT_WIDGETS = ["QDialog", "QDialogButtonBox", "QWidget", "QFormLayout", "QPlainTextEdit", "QLineEdit",
               "QSpinBox", "QDoubleSpinBox", "QCheckBox", "QPushButton", "QHBoxLayout",
               "QVBoxLayout", "QGridLayout", "QScrollArea", "QLabel", "QComboBox",
               "QTabWidget", "QTableView", "QHeaderView", "QAbstractTableModel",
               "QMessageBox", "QFileDialog", "QInputDialog", "QProgressBar",
               "QSizePolicy", "QButtonGroup", "QFrame", "QTextEdit", "QListWidget",
               "QMenu", "QAction", "QSplitter", "QGroupBox"]
qtcore = _stub("PySide6.QtCore", Signal=lambda *a, **k: None, QObject=_dummy_class("QObject"),
               QTimer=_dummy_class("QTimer"), Qt=_dummy_class("Qt"),
               QEvent=_dummy_class("QEvent"), Slot=lambda *a, **k: (lambda s: None))
qtwidgets = _stub("PySide6.QtWidgets")
for _cn in _QT_WIDGETS:
    setattr(qtwidgets, _cn, _dummy_class(_cn))
qtgui = _stub("PySide6.QtGui", QAction=_dummy_class("QAction"), QColor=_dummy_class("QColor"), QPalette=_dummy_class("QPalette"),
              QKeySequence=_dummy_class("QKeySequence"), QFont=_dummy_class("QFont"),
              QPainter=_dummy_class("QPainter"), QPen=_dummy_class("QPen"),
              QBrush=_dummy_class("QBrush"), QDesktopServices=_dummy_class("QDesktopServices"),
              QIcon=_dummy_class("QIcon"), QTextCursor=_dummy_class("QTextCursor"))
_stub("shiboken6", isValid=lambda *a: True, wrapInstance=lambda *a: None)
pyside6 = types.ModuleType("PySide6")
pyside6.QtCore = qtcore
pyside6.QtWidgets = qtwidgets
pyside6.QtGui = qtgui
sys.modules["PySide6"] = pyside6

import llm_explainer as L

FAKE_FUNC = types.SimpleNamespace(start_ea=0x1000)

def _new_runner(cfg):
    r = L.ConversationRunner(cfg, FAKE_FUNC)
    r._on_result_cb = lambda *a: None
    r._on_error_cb = lambda *a: None
    return r

# ---------------------------------------------------------------------------
# Fake llama-server
# ---------------------------------------------------------------------------
from http.server import BaseHTTPRequestHandler, HTTPServer

CONTEXT_SIZE = 2000  # tokens

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            # the REAL llama.cpp shape (verified against a live server):
            # the window lives in meta.n_ctx, not context.size
            self._send({"data": [{"id": "test-model", "object": "model",
                                  "owned_by": "llamacpp",
                                  "meta": {"n_ctx": CONTEXT_SIZE,
                                            "n_ctx_train": CONTEXT_SIZE * 4,
                                            "n_embd": 5120}}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n).decode())
        if self.path != "/v1/chat/completions":
            self._send({"error": "not found"}, 404)
            return
        if req.get("stream"):
            # stream: single delta + usage chunk + [DONE]
            def sse(obj):
                return "data: %s\n\n" % json.dumps(obj)
            body = (sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]})
                    + sse({"choices": [{"delta": {}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 100, "completion_tokens": 5,
                                     "total_tokens": 105},
                           # the REAL llama.cpp final-chunk shape (timings)
                           "timings": {"predicted_n": 5, "predicted_ms": 119.0,
                                       "predicted_per_second": 42.0}})
                    + "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        # non-stream (the compact summarizer request)
        self._send({"choices": [{"message": {"role": "assistant",
                                            "content": "SUMMARY-DIGEST"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 10,
                              "total_tokens": 60}})

srv = HTTPServer(("127.0.0.1", 0), Handler)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = "http://127.0.0.1:%d" % port

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

# 1. context size detection
size = L.query_server_context_size(URL, "test-model")
check("query_server_context_size detects %d" % CONTEXT_SIZE, size == CONTEXT_SIZE)
check("negative cache for dead server", L.query_server_context_size("http://127.0.0.1:1") is None)

# 2. token estimate
msgs = [{"role": "system", "content": "s" * 3500}, {"role": "user", "content": "u" * 3500}]
est = L.estimate_prompt_tokens(msgs, 3.5)
check("estimate_prompt_tokens ~2000", abs(est - 2000.0) < 10)

# 3. config fields
cfg = L.PluginConfig(server_urls=[URL], model="test-model",
                     auto_compact_enabled=True, auto_compact_threshold=0.9,
                     server_context_size=0, chars_per_token=3.5, max_tokens=200)
check("config defaults round-trip", cfg.auto_compact_enabled and abs(cfg.auto_compact_threshold - 0.9) < 1e-9)
bad = L.PluginConfig(auto_compact_threshold=5.0)
check("threshold clamped to <=1.0", bad.auto_compact_threshold == 1.0)

# 4. runner: no compact when under threshold
runner = _new_runner(cfg)
runner._detect_context_size()
check("runner detected context size", runner._context_size == CONTEXT_SIZE)
runner.messages = [{"role": "system", "content": "s" * 1000},
                   {"role": "user", "content": "u" * 1000},
                   {"role": "assistant", "content": "a" * 1000},
                   {"role": "user", "content": "q" * 1000}]
check("no compact under threshold", runner._maybe_compact() is False)

# 4b. headroom must account for max_tokens, not just the fraction: a large
# reply budget relative to a small context window should force an earlier
# (tighter) limit than the flat 90% threshold would alone.
cfg_big_reply = L.PluginConfig(server_urls=[URL], model="test-model",
                               auto_compact_enabled=True, auto_compact_threshold=0.9,
                               server_context_size=0, chars_per_token=3.5, max_tokens=1900)
runner_headroom = _new_runner(cfg_big_reply)
runner_headroom._detect_context_size()
check("limit tightened by max_tokens",
      runner_headroom._compact_limit_tokens() == max(1000, CONTEXT_SIZE - 1900))
runner_headroom.messages = [{"role": "system", "content": "s" * 1000},
                            {"role": "user", "content": "u" * 1000},
                            {"role": "assistant", "content": "a" * 1000},
                            {"role": "user", "content": "q" * 1000}]  # ~1143 tokens > 1000-token limit
check("large max_tokens forces earlier compaction",
      runner_headroom._maybe_compact() is True)

# 4c. force_compact(): manual trigger, idle-only, does not re-issue a
# request when there is no pending user turn to answer.
runner_manual = _new_runner(cfg)
runner_manual._detect_context_size()
runner_manual.messages = [{"role": "system", "content": "sys"},
                          {"role": "user", "content": "u" * 500},
                          {"role": "assistant", "content": "reply"},
                          {"role": "user", "content": "another question"}]
check("force_compact starts when idle", runner_manual.force_compact() is True)
import time
for _ in range(100):
    if not runner_manual._compacting:
        break
    time.sleep(0.05)
check("force_compact does not spawn a worker", runner_manual.worker is None)
runner_short = _new_runner(cfg)
runner_short.messages = [{"role": "system", "content": "sys"},
                         {"role": "user", "content": "u"}]
check("force_compact no-ops with too little history", runner_short.force_compact() is False)

# 5. runner: over threshold -> compact path (fake server answers the
#    summarizer request synchronously; execute_sync stub runs inline)
runner2 = _new_runner(cfg)
runner2._detect_context_size()
# ~ 6000 chars / 3.5 = ~1714 tokens > 90% of 2000
runner2.messages = [{"role": "system", "content": "s" * 1000},
                    {"role": "user", "content": "u" * 3000},
                    {"role": "assistant", "content": "a" * 3000},
                    {"role": "user", "content": "pending question" }]
compacted = runner2._maybe_compact()
check("compact triggered over threshold", compacted is True)
# _on_compact_done re-enters _issue_request, which spawns a real streaming
# worker against the fake server; let it finish.
import time
for _ in range(100):
    if runner2.worker is None and not runner2._compacting:
        break
    time.sleep(0.05)
# after compaction the runner re-issued the request; the fake server's
# reply ("ok") was appended as an assistant turn, so assert on the head.
check("history replaced by summary",
      runner2.messages[0]["role"] == "system"
      and runner2.messages[1]["role"] == "user"
      and "SUMMARY-DIGEST" in runner2.messages[1]["content"]
      and "CONTEXT SUMMARY" in runner2.messages[1]["content"])
check("pending user turn preserved",
      any(m["content"] == "pending question" for m in runner2.messages))
check("old history gone", all("u" * 3000 not in m["content"] for m in runner2.messages))
check("cpt calibrated from usage", 1.0 <= runner2._cpt <= 10.0)

# 6. hard truncate: mechanical elision (summarizer-failure fallback)
runner3 = _new_runner(cfg)
runner3._detect_context_size()
runner3.messages = [{"role": "system", "content": "S" * 1000},
                    {"role": "user", "content": "u" * 4000},
                    {"role": "assistant", "content": "a" * 4000},
                    {"role": "user", "content": "q" * 100}]
before = sum(len(m["content"]) for m in runner3.messages)
runner3._hard_truncate(int(CONTEXT_SIZE * 0.9))
after = sum(len(m["content"]) for m in runner3.messages)
check("hard_truncate shrinks prompt (%d -> %d)" % (before, after), after < before * 0.8)
check("hard_truncate protects system prompt", runner3.messages[0]["content"] == "S" * 1000)
check("hard_truncate protects pending turn", runner3.messages[-1]["content"] == "q" * 100)

# 7. single-message overflow: initial code context alone too big
runner4 = _new_runner(cfg)
runner4._detect_context_size()
runner4.messages = [{"role": "system", "content": "s" * 400},
                    {"role": "user", "content": "c" * 10000}]
check("triggers for oversized initial context", runner4._maybe_compact() is True)
check("initial context truncated in place",
      len(runner4.messages[1]["content"]) < 10000 and "auto-compact" in runner4.messages[1]["content"])

# 8. context-usage gauge callback (the dialog's percentage counter source)
events = []
runner5 = _new_runner(cfg)
runner5._detect_context_size()
runner5._on_context_usage_cb = lambda est, size, tps: events.append((est, size, tps))
runner5.messages = [{"role": "system", "content": "s" * 1000},
                    {"role": "user", "content": "u" * 1000}]
runner5._notify_context_usage()
check("gauge callback reports (est, size)",
      len(events) == 1 and events[0][1] == CONTEXT_SIZE and 200 < events[0][0] < 700)
runner5._context_size = None
runner5._notify_context_usage()
check("gauge callback reports unknown as (0, 0, None)", events[-1] == (0, 0, None))

# 9b. t/s propagation: the compact test's re-issued reply carried timings
check("t/s captured from server timings", runner2._last_tps == 42.0)

# 9a. live gauge: refresh_context_gauge counts the in-flight reply
runner6 = _new_runner(cfg)
runner6._detect_context_size()
runner6.messages = [{"role": "system", "content": "s" * 1000},
                    {"role": "user", "content": "u" * 1000}]
base = []
runner6._on_context_usage_cb = lambda e, s, t: base.append(e)
runner6.refresh_context_gauge()
class _FakeWorker:
    _content_chars = 7000
runner6.worker = _FakeWorker()
runner6.refresh_context_gauge()
delta = base[1] - base[0]
check("live gauge counts in-flight reply (7000ch -> ~%d tok)" % delta,
      1500 < delta < 2500)
runner6.worker = None

# 9c. worker counts streamed content chars
seen = {}
def _on_done(text, reasoning="", finish=None):
    seen["text"] = text
w = L.LlamaStreamWorker(cfg, URL, [{"role": "user", "content": "hi"}],
                        lambda p: 0, lambda p: 0, _on_done,
                        lambda m: seen.setdefault("err", m))
w.start(); w.join(10)
check("worker tracks in-flight chars",
      w._content_chars == len(seen.get("text", "")) and w._content_chars > 0)

# 9. context.size fallback shape (non-llama.cpp OpenAI-compatible servers)  
import urllib.request as _ur
_real_urlopen = _ur.urlopen
def _fake_urlopen(req, timeout=None):
    class _R:
        def read(self):
            return json.dumps({"data": [{"id": "m", "context": {"size": 4096}}]}).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
    return _R()
_ur.urlopen = _fake_urlopen
try:
    check("context.size fallback shape",
          L.query_server_context_size("http://fallback-shape.test", "m") == 4096)
finally:
    _ur.urlopen = _real_urlopen

# 9d. SettingsDialog builds (tabbed layout) and _on_ok round-trips every
#     widget - this is the regression guard for "widget referenced in
#     _on_ok/_populate but never created in __init__" (the auto-compact
#     widgets had exactly that bug).
dlg = L.SettingsDialog(cfg)
dlg._on_ok()
rc = dlg.result_config
check("SettingsDialog builds + _on_ok round-trips",
      rc is not None and rc.server_context_size == 0
      and rc.auto_compact_threshold >= 0.05
      and rc.agent_server_enabled is False
      and rc.agent_server_port == 0)

# 10. agent server: protocol layer (real HTTP on localhost)
import urllib.error

agent_cfg = L.PluginConfig(server_urls=[URL], agent_server_enabled=True,
                           agent_server_port=0, agent_server_token="test-token-123")
asrv = L.AgentServer(agent_cfg)
asrv.start()
for _ in range(100):
    if asrv.bound_port is not None or asrv._bind_error:
        break
    time.sleep(0.05)
check("agent server binds (port %s)" % asrv.bound_port, asrv.bound_port is not None)

def _http(method, path, body=None, token="test-token-123"):
    req = urllib.request.Request(asrv.url + path, method=method)
    if token is not None:
        req.add_header("Authorization", "Bearer " + token)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

status, body = _http("GET", "/health")
check("/health open (no auth)", status == 200 and body.get("ok") is True)

status, body = _http("GET", "/tools", token=None)
check("/tools requires auth", status == 401)

status, body = _http("GET", "/tools")
tools = body.get("tools", {})
check("/tools lists %d tools" % len(tools), status == 200 and len(tools) >= 40)
check("every tool has description+params",
      all(isinstance(t.get("description"), str) and isinstance(t.get("params"), dict) for t in tools.values()))
check("registry fns all callable", all(callable(t["fn"]) for t in L.AGENT_TOOLS.values()))
check("gated tools flagged in catalog",
      tools.get("patchBytes", {}).get("gated") == "patches"
      and tools.get("runIdaScript", {}).get("gated") == "scripts"
      and tools.get("debugContinue", {}).get("gated") == "debug")

status, body = _http("POST", "/call", {"name": "getMetadata", "args": {}}, token="wrong")
check("bad token -> 401", status == 401)

status, body = _http("POST", "/call", {"name": "noSuchTool", "args": {}})
check("unknown tool -> 404", status == 404)

status, body = _http("POST", "/call", {"name": "getMetadata", "args": "nope"})
check("non-object args -> 400", status == 400)

status, body = _http("POST", "/call", {"name": "patchBytes", "args": {"address": "0x1000", "hex": "90"}})
check("gated patchBytes refused when disabled -> 403", status == 403)

status, body = _http("POST", "/call", {"name": "runIdaScript", "args": {"code": "result=1"}})
check("gated runIdaScript refused when disabled -> 403", status == 403)

# 10b. debugger tools (functional ida_dbg/ida_idd stubs, fake suspended process)
status, body = _http("POST", "/call", {"name": "getDebugState", "args": {}})
check("getDebugState reports suspended + IP",
      status == 200 and body["result"]["process_state"] == "suspended"
      and body["result"]["instruction_pointer"] == "0x140001000"
      and body["result"]["debugger_name"] == "stubdbg")

status, body = _http("POST", "/call", {"name": "getLiveRegisters", "args": {}})
check("getLiveRegisters dumps named registers",
      status == 200 and body["result"]["registers"].get("rax") == "0x11223344"
      and body["result"]["registers"].get("rbx") == "0x55")

status, body = _http("POST", "/call", {"name": "getLiveMemory", "args": {"address": 0x140000000, "size": 32}})
check("getLiveMemory reads live bytes",
      status == 200 and body["result"]["ascii"] == "hello-live")

status, body = _http("POST", "/call", {"name": "getLiveMemoryMap", "args": {}})
check("getLiveMemoryMap lists regions",
      status == 200 and body["result"]["regions"][0]["name"] == "stub.exe")

status, body = _http("POST", "/call", {"name": "getBreakpoints", "args": {}})
check("getBreakpoints lists bpts",
      status == 200 and body["result"]["count"] == 1
      and body["result"]["breakpoints"][0]["address"] == "0x140001234")

status, body = _http("POST", "/call", {"name": "getStackTrace", "args": {}})
check("getStackTrace returns frames",
      status == 200 and body["result"]["frames"][0]["return_address"] == "0x140001234")

status, body = _http("POST", "/call", {"name": "getThreads", "args": {}})
check("getThreads lists the current thread",
      status == 200 and body["result"]["threads"][0]["current"] is True)

for _tool in ("debugStart", "debugSuspend", "debugContinue", "debugStep",
              "debugRunTo", "debugDetach", "debugExit", "setBreakpoint",
              "removeBreakpoint", "setRegisterValue"):
    status, body = _http("POST", "/call", {"name": _tool, "args": {}})
    check("gated %s refused when disabled -> 403" % _tool, status == 403)

# enable the debug gate and drive the fake process through a step
asrv._config.agent_allow_debug = True
status, body = _http("POST", "/call", {"name": "setBreakpoint", "args": {"address": 0x140001000}})
check("setBreakpoint works when gated open", status == 200 and body["result"]["set"] is True)
status, body = _http("POST", "/call", {"name": "setRegisterValue", "args": {"name": "rax", "value": "0xDEAD"}})
check("setRegisterValue works when gated open", status == 200 and body["result"]["value"] == "0xdead")
status, body = _http("POST", "/call", {"name": "debugStep", "args": {"kind": "over"}})
check("debugStep completes (fake stays suspended)",
      status == 200 and body["result"]["process_state"] == "suspended")
_dbg_state["state"] = 2  # DSTATE_RUN
status, body = _http("POST", "/call", {"name": "getLiveRegisters", "args": {}})
check("getLiveRegisters refuses while running -> 400",
      status == 400 and "not suspended" in body.get("error", ""))
status, body = _http("POST", "/call", {"name": "debugSuspend", "args": {}})
check("debugSuspend transitions running -> suspended",
      status == 200 and body["result"]["process_state"] == "suspended")

status, body = _http("POST", "/call", {"name": "getMetadata", "args": {}})
check("IDA-backed call surfaces IDA error cleanly (stub env)",
      status == 500 and "IDA error" in body.get("error", ""))

status, body = _http("POST", "/batch",
                     {"calls": [{"name": "noSuchTool", "args": {}},
                                {"name": "patchBytes", "args": {}}]})
check("/batch returns per-call results",
      status == 200 and len(body.get("results", [])) == 2
      and body["results"][0].get("ok") is False and body["results"][1].get("ok") is False)

# 11. result truncation
obj, trunc = L._truncate_result({"a": "x" * 5000, "b": [1, {"c": "y" * 5000}]}, 100)
check("truncation caps strings + flags",
      trunc and len(obj["a"]) <= 115 and obj.get("_truncated") is True)

# 12. address resolver (int path; name path needs a real IDB)
check("_agent_ea int passthrough", L._agent_ea({"address": 0x1400}, "address") == 0x1400)
try:
    L._agent_ea({}, "address")
    check("_agent_ea missing -> _AgentError", False)
except L._AgentError:
    check("_agent_ea missing -> _AgentError", True)

asrv.shutdown()
asrv.join(5)

srv.shutdown()
print()
print("FAILURES: %s" % fails if fails else "ALL TESTS PASSED")
sys.exit(1 if fails else 0)
