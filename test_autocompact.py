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

def _dummy_class(name):
    """A class that can be subclassed and instantiated with anything."""
    return type(name, (), {
        "__init__": lambda self, *a, **k: None,
        "__getattr__": lambda self, item: (lambda *a, **k: None),
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
      PLUGIN_KEEP=1)
_stub("idautils", XrefsTo=lambda *a, **k: iter(()))
_stub("idc")
_stub("ida_kernwin", MFF_FAST=1, MFF_WRITE=2, msg=lambda s: print("[msg]", s),
      execute_sync=lambda fn, *a: (fn(), 0), UI_Hooks=_UIHooks,
      action_handler_t=_ActionHandler)
for extra in ("ida_auto", "ida_bytes", "ida_idp", "ida_lines", "ida_name",
              "ida_nalt", "ida_segment", "ida_typeinf", "ida_ua", "ida_xref",
              "ida_x86", "ida_arm"):
    _stub(extra)
_stub("ida_funcs", get_func=lambda ea: None, get_func_name=lambda ea: "sub_%X" % ea)
_stub("ida_graph", GraphViewer=_dummy_class("GraphViewer"))

# PySide6 stubs: dummy classes (subclassable, instantiable with anything)
_QT_WIDGETS = ["QDialog", "QWidget", "QFormLayout", "QPlainTextEdit", "QLineEdit",
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
qtgui = _stub("PySide6.QtGui", QAction=_dummy_class("QAction"), QColor=_dummy_class("QColor"),
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
                     server_context_size=0, chars_per_token=3.5)
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

srv.shutdown()
print()
print("FAILURES: %s" % fails if fails else "ALL TESTS PASSED")
sys.exit(1 if fails else 0)
