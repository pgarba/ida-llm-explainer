"""LLM Explainer - IDA Pro 9.2 plugin.

Asks a locally-running llama.cpp server (llama-server, OpenAI-compatible API)
to explain the function currently under the cursor, in either the Hex-Rays
pseudocode view or the plain disassembly view. The model's streamed answer
is shown in a small, non-modal dialog where you can Accept it (written into
the function's comment, visible in both views), ask to Reason More (send a
follow-up question and get a refined answer), or Cancel (discard, no
database changes).

Install by copying this single file into one of:
  - Per-user (recommended, no admin rights needed):
        <IDA user dir>\\plugins\\llm_explainer.py
    On Windows this is typically:
        %APPDATA%\\Hex-Rays\\IDA Pro\\plugins\\llm_explainer.py
  - Global (all users of this IDA install):
        <IDA install dir>\\plugins\\llm_explainer.py

Requires: IDA Pro 9.2+ (PySide6 is bundled with IDA, no extra install
needed), and a running llama.cpp `llama-server` reachable at the configured
base URL (default http://127.0.0.1:8080). The Hex-Rays decompiler is
optional - if it is not available for the current architecture the plugin
falls back to a plain disassembly listing automatically.

Configure the server URL, model, and other options via
Edit > Plugins > LLM Explainer.

Note: if llama-server is run with a single inference slot, opening several
"explain" dialogs at once will simply queue their requests on the server -
that is a server/deployment concern, not a bug in this plugin.
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
ACTION_ID_EXPLAIN = "llm_explainer:explain_function"
CONFIG_FILENAME = "llm_explainer.cfg.json"

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert reverse engineer assisting inside IDA Pro. You will "
    "be given the decompiled pseudocode or disassembly of a single function, "
    "along with its name, address, target architecture, and the names of "
    "functions it calls. Explain concisely and precisely what the function "
    "does: its purpose, inputs and outputs, notable algorithms or "
    "protocol/format handling, and any security-relevant behavior (e.g. "
    "unsafe string/memory operations, crypto, network I/O). Do not restate "
    "the code line by line. Respond in plain text without markdown code "
    "fences, since your answer will be written directly into an IDA "
    "function comment. Keep the initial answer under ~200 words unless "
    "asked for more detail in a follow-up."
)

DEFAULT_CONFIG = {
    "base_url": "http://127.0.0.1:8080",
    "model": "",
    "api_key": "",
    "temperature": 0.2,
    "max_tokens": 1024,
    "request_timeout": 30,
    "max_context_chars": 12000,
    "include_callees": True,
    "max_callees": 20,
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "explain_hotkey": "Ctrl-Alt-E",
}

ContextBundle = namedtuple("ContextBundle", ["kind", "text"])


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


def gather_callees(func, max_callees):
    if max_callees <= 0:
        return []
    names = []
    seen = set()
    for ea in idautils.FuncItems(func.start_ea):
        for ref in idautils.CodeRefsFrom(ea, 0):
            callee = ida_funcs.get_func(ref)
            if callee and callee.start_ea == ref and ref != func.start_ea:
                name = ida_funcs.get_func_name(ref)
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
        if len(names) >= max_callees:
            break
    return names[:max_callees]


def build_user_message(config, func, ctx, callees):
    name = ida_funcs.get_func_name(func.start_ea) or ("sub_%X" % func.start_ea)
    header = (
        "Function: %s\n"
        "Address: %#010x\n"
        "Architecture: %s\n" % (name, func.start_ea, get_procname())
    )
    if callees:
        header += "Calls: %s\n" % ", ".join(callees)
    body = ctx.text
    if len(body) > config.max_context_chars:
        body = body[: config.max_context_chars] + "\n...[truncated]..."
    kind_label = "Pseudocode (Hex-Rays)" if ctx.kind == "pseudocode" else "Disassembly"
    return "%s\n--- %s ---\n%s" % (header, kind_label, body)


def _resolve_func(ctx):
    ea = getattr(ctx, "cur_ea", None)
    if ea is None or ea == idaapi.BADADDR:
        ea = ida_kernwin.get_screen_ea()
    return ida_funcs.get_func(ea)


def _write_comment_and_refresh(func_ea, comment):
    try:
        idc.set_func_cmt(func_ea, comment, 0)
    except Exception as exc:
        ida_kernwin.msg("[%s] Failed to set comment: %s\n" % (PLUGIN_NAME, exc))
        return 0
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

    def __init__(self, config, messages, on_delta, on_done, on_error):
        super().__init__(daemon=True)
        self._config = config
        self._messages = messages
        self._on_delta = on_delta
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
        with self._response as resp:
            content_type = resp.headers.get("Content-Type", "")
            if not content_type.startswith("text/event-stream"):
                self._handle_non_stream_body(resp.read(), parts)
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
                    piece = (choices[0].get("delta") or {}).get("content")
                    if piece:
                        parts.append(piece)
                        ida_kernwin.execute_sync(
                            functools.partial(self._on_delta, piece), ida_kernwin.MFF_FAST
                        )

        if not self._cancel_event.is_set():
            ida_kernwin.execute_sync(
                functools.partial(self._on_done, "".join(parts)), ida_kernwin.MFF_FAST
            )

    def _handle_non_stream_body(self, body, parts):
        try:
            obj = json.loads(body.decode("utf-8", "replace"))
        except Exception as exc:
            raise RuntimeError("Unexpected response from server: %s" % exc) from None
        choices = obj.get("choices") or []
        if not choices:
            err = obj.get("error")
            raise RuntimeError(str(err) if err else "Empty response from server.")
        content = (choices[0].get("message") or {}).get("content", "")
        if content:
            parts.append(content)
            ida_kernwin.execute_sync(
                functools.partial(self._on_delta, content), ida_kernwin.MFF_FAST
            )


# ---------------------------------------------------------------------------
# UI: result dialog
# ---------------------------------------------------------------------------

class ExplainResultDialog(QtWidgets.QDialog):
    def __init__(self, config, func, parent=None):
        super().__init__(parent)
        self.config = config
        self.func_ea = func.start_ea
        self.func_name = ida_funcs.get_func_name(func.start_ea) or ("sub_%X" % func.start_ea)
        self.worker = None
        self.messages = None
        self._buffer = []
        self._last_answer_text = ""
        self._closed = False

        self.setWindowTitle("%s - %s @ %#x" % (PLUGIN_NAME, self.func_name, func.start_ea))
        self.resize(560, 420)

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

        self._start_initial_request(func)

    # -- request lifecycle --------------------------------------------------

    def _start_initial_request(self, func):
        try:
            ctx = gather_function_context(func)
            callees = gather_callees(func, self.config.max_callees) if self.config.include_callees else []
            user_msg = build_user_message(self.config, func, ctx, callees)
        except Exception as exc:
            self.status_label.setText("Failed to gather function context: %s" % exc)
            return
        self.messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        self.start_request()

    def start_request(self):
        self._buffer = []
        self.status_label.setText("Querying model...")
        self.reason_button.setEnabled(False)
        self.followup_input.setEnabled(False)
        self.worker = LlamaStreamWorker(
            self.config, list(self.messages), self._on_delta, self._on_done, self._on_error
        )
        self.worker.start()

    # -- worker callbacks (run on IDA's main thread via execute_sync) -------

    def _on_delta(self, piece):
        if self._closed:
            return 0
        self._buffer.append(piece)
        try:
            cursor = self.stream_edit.textCursor()
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
            cursor.insertText(piece)
            self.stream_edit.setTextCursor(cursor)
            self.stream_edit.ensureCursorVisible()
        except RuntimeError:
            pass
        return 0

    def _on_done(self, full_text):
        if self._closed:
            return 0
        text = full_text if full_text else "".join(self._buffer)
        self._last_answer_text = text
        if self.messages is not None:
            self.messages.append({"role": "assistant", "content": text})
        self.status_label.setText("Done.")
        self.reason_button.setEnabled(True)
        self.followup_input.setEnabled(True)
        self.accept_button.setEnabled(bool(text.strip()))
        self.worker = None
        return 0

    def _on_error(self, message):
        if self._closed:
            return 0
        self.status_label.setText("Error: %s" % message)
        self.reason_button.setEnabled(True)
        self.followup_input.setEnabled(True)
        partial = "".join(self._buffer).strip()
        self.accept_button.setEnabled(bool(partial or self._last_answer_text.strip()))
        self.worker = None
        return 0

    # -- button handlers ------------------------------------------------

    def on_reason_more(self):
        if self.worker is not None or self.messages is None:
            return
        followup = self.followup_input.text().strip()
        if not followup:
            followup = "Please explain your reasoning in more detail."
        self.messages.append({"role": "user", "content": followup})
        self.followup_input.clear()
        try:
            self.stream_edit.appendPlainText("\n\n--- Follow-up: %s ---\n" % followup)
        except RuntimeError:
            pass
        self.start_request()

    def on_accept(self):
        text = (self._last_answer_text or "".join(self._buffer)).strip()
        if not text:
            ida_kernwin.warning("Nothing to accept yet.")
            return
        comment = strip_markdown_fences(text)
        ida_kernwin.execute_sync(
            functools.partial(_write_comment_and_refresh, self.func_ea, comment),
            ida_kernwin.MFF_WRITE,
        )
        self.close()

    def closeEvent(self, event):
        self._closed = True
        if self.worker is not None:
            self.worker.cancel()
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

        self.base_url_edit = QtWidgets.QLineEdit(config.base_url)
        form.addRow("Server base URL:", self.base_url_edit)

        self.model_edit = QtWidgets.QLineEdit(config.model)
        self.model_edit.setPlaceholderText("(optional - leave blank to use server default)")
        form.addRow("Model name:", self.model_edit)

        self.api_key_edit = QtWidgets.QLineEdit(config.api_key)
        self.api_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("(optional bearer token)")
        form.addRow("API key:", self.api_key_edit)

        self.temperature_spin = QtWidgets.QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(config.temperature)
        form.addRow("Temperature:", self.temperature_spin)

        self.max_tokens_spin = QtWidgets.QSpinBox()
        self.max_tokens_spin.setRange(1, 32768)
        self.max_tokens_spin.setValue(config.max_tokens)
        form.addRow("Max tokens:", self.max_tokens_spin)

        self.timeout_spin = QtWidgets.QSpinBox()
        self.timeout_spin.setRange(1, 600)
        self.timeout_spin.setValue(config.request_timeout)
        form.addRow("Request timeout (s):", self.timeout_spin)

        self.max_context_spin = QtWidgets.QSpinBox()
        self.max_context_spin.setRange(500, 200000)
        self.max_context_spin.setSingleStep(500)
        self.max_context_spin.setValue(config.max_context_chars)
        form.addRow("Max context chars:", self.max_context_spin)

        self.include_callees_check = QtWidgets.QCheckBox("Include called-function names in prompt")
        self.include_callees_check.setChecked(config.include_callees)
        form.addRow(self.include_callees_check)

        self.max_callees_spin = QtWidgets.QSpinBox()
        self.max_callees_spin.setRange(0, 200)
        self.max_callees_spin.setValue(config.max_callees)
        form.addRow("Max callees listed:", self.max_callees_spin)

        self.hotkey_edit = QtWidgets.QLineEdit(config.explain_hotkey)
        self.hotkey_edit.setPlaceholderText("e.g. Ctrl-Alt-E (leave blank for none)")
        form.addRow("Explain hotkey:", self.hotkey_edit)

        self.system_prompt_edit = QtWidgets.QPlainTextEdit(config.system_prompt)
        self.system_prompt_edit.setMinimumHeight(140)
        form.addRow("System prompt:", self.system_prompt_edit)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)

        outer = QtWidgets.QVBoxLayout(self)
        outer.addLayout(form)
        outer.addWidget(buttons)

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
        cfg.explain_hotkey = self.hotkey_edit.text().strip()
        cfg.system_prompt = self.system_prompt_edit.toPlainText()
        cfg._validate()
        self.result_config = cfg
        self.accept()


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


class PopupHooks(ida_kernwin.UI_Hooks):
    def finish_populating_widget_popup(self, widget, popup, ctx=None):
        wtype = ida_kernwin.get_widget_type(widget)
        if wtype in (ida_kernwin.BWN_PSEUDOCODE, ida_kernwin.BWN_DISASM):
            ida_kernwin.attach_action_to_popup(widget, popup, ACTION_ID_EXPLAIN, "LLM Explainer/")


class LLMExplainerPlugin(idaapi.plugin_t):
    flags = idaapi.PLUGIN_KEEP
    comment = "Ask a local llama.cpp model to explain the current function"
    help = (
        "Right-click a function in the disassembly or pseudocode view (or "
        "use the configured hotkey) to ask the configured llama.cpp server "
        "to explain it. Edit > Plugins > LLM Explainer opens settings."
    )
    wanted_name = PLUGIN_NAME
    wanted_hotkey = ""

    instance = None

    def init(self):
        LLMExplainerPlugin.instance = self
        self.config = PluginConfig.load()
        self._open_dialogs = []
        self._action_handler = ExplainActionHandler()

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
        LLMExplainerPlugin.instance = None

    def open_explain_dialog(self, func):
        dlg = ExplainResultDialog(self.config, func)
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


def PLUGIN_ENTRY():
    return LLMExplainerPlugin()
