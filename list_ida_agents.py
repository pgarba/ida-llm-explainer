#!/usr/bin/env python3
"""List running IDA Pro agent-server sessions.

Each IDA session running the llm_explainer plugin with its agent server
enabled advertises itself in a small registry file under the IDA user
directory (see AgentServer / _write_agent_registry in llm_explainer.py).
This script has no dependency on IDA - it just reads that registry and
probes each entry's `GET /health` to confirm it's still alive, so an
external agent can find out which IDA sessions are open, what binary each
one is analyzing, and which host/port/token to use to talk to it.

Usage:
    python list_ida_agents.py [--timeout SECONDS] [--json]

Output is a JSON array (one object per live session), sorted by start
time, each with: pid, host, port, token, idb_name, idb_path, input_file,
processor, bits, ida_version, plugin_version, started_at, url.

Stale registry files (process gone, or nothing answering on the recorded
port) are removed as they're found, once they're old enough that a
just-starting session couldn't be mistaken for a dead one.
"""

import argparse
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request

AGENT_INSTANCES_DIRNAME = "agent_instances"
CONFIG_FILENAME = "llm_explainer.cfg.json"
STALE_GRACE_SECONDS = 10.0


def _ida_user_dirs():
    """Candidate IDA user directories, mirroring idaapi.get_user_idadir():
    IDAUSR (';' on Windows, ':' elsewhere) overrides the per-OS default."""
    idausr = os.environ.get("IDAUSR")
    if idausr:
        sep = ";" if os.name == "nt" else ":"
        dirs = [d for d in idausr.split(sep) if d]
        if dirs:
            return dirs
    if os.name == "nt":
        appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
        return [os.path.join(appdata, "Hex-Rays", "IDA Pro")]
    if sys.platform == "darwin":
        return [os.path.expanduser("~/Library/Application Support/Hex-Rays/IDA Pro")]
    return [os.path.expanduser("~/.idapro")]


def _registry_dirs():
    dirs = []
    for base in _ida_user_dirs():
        d = os.path.join(base, AGENT_INSTANCES_DIRNAME)
        if os.path.isdir(d):
            dirs.append(d)
    return dirs


def _probe_health(host, port, timeout):
    url = "http://%s:%d/health" % (host, port)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def list_sessions(timeout=1.0, cleanup=True):
    sessions = []
    for directory in _registry_dirs():
        for path in glob.glob(os.path.join(directory, "instance_*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = json.load(f)
            except (OSError, ValueError):
                continue

            host = entry.get("host") or "127.0.0.1"
            port = entry.get("port")
            if not isinstance(port, int):
                continue

            health = _probe_health(host, port, timeout)
            if health is None:
                age = time.time() - float(entry.get("started_at") or 0)
                if cleanup and age > STALE_GRACE_SECONDS:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                continue

            entry["url"] = "http://%s:%d" % (host, port)
            sessions.append(entry)

    sessions.sort(key=lambda e: e.get("started_at") or 0)
    return sessions


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timeout", type=float, default=1.0,
                        help="per-session /health probe timeout in seconds (default: 1.0)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="don't delete stale registry files for dead sessions")
    args = parser.parse_args()

    sessions = list_sessions(timeout=args.timeout, cleanup=not args.no_cleanup)
    print(json.dumps(sessions, indent=2))
    return 0 if sessions else 1


if __name__ == "__main__":
    sys.exit(main())
