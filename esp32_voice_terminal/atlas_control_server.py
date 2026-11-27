import argparse
import json
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class GamepadManager:
    def __init__(self):
        self.directory = Path(__file__).resolve().parent
        self.script = self.directory / "so101_lekiwi_gamepad.py"
        self.log_path = self.directory / "gamepad.log"
        self.process = None
        self.log_handle = None

    def status(self):
        running = self.process is not None and self.process.poll() is None
        return {
            "running": running,
            "pid": self.process.pid if running else None,
            "returncode": None if running or self.process is None else self.process.returncode,
        }

    def start(self):
        if self.status()["running"]:
            return {"ok": True, "status": "already_running", **self.status()}
        if not self.script.is_file():
            raise RuntimeError(f"Control script missing: {self.script}")
        self.log_handle = self.log_path.open("ab", buffering=0)
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(self.script),
                "--input-backend",
                "usb",
                "--arm-step",
                "0.7",
                "--rate",
                "30",
                "--record-fps",
                "10",
            ],
            cwd=self.directory,
            stdin=subprocess.DEVNULL,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        time.sleep(2.0)
        if self.process.poll() is not None:
            code = self.process.returncode
            self.log_handle.close()
            self.log_handle = None
            raise RuntimeError(f"Control process exited with {code}: {self.log_tail()}")
        return {"ok": True, "status": "started", **self.status()}

    def stop(self):
        if not self.status()["running"]:
            return {"ok": True, "status": "already_stopped", **self.status()}
        process = self.process
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        return {"ok": True, "status": "stopped", **self.status()}

    def log_tail(self):
        if not self.log_path.is_file():
            return ""
        return self.log_path.read_text(errors="replace")[-2000:]


class ControlServer(ThreadingHTTPServer):
    def __init__(self, address, handler, manager):
        super().__init__(address, handler)
        self.manager = manager


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.respond(200, {"ok": True})

    def do_GET(self):
        if self.path == "/api/health":
            self.respond(200, {"ok": True, "service": "atlas-control", **self.server.manager.status()})
        elif self.path == "/api/gamepad/status":
            self.respond(200, {"ok": True, **self.server.manager.status()})
        else:
            self.respond(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        try:
            if self.path == "/api/gamepad/start":
                self.respond(200, self.server.manager.start())
            elif self.path == "/api/gamepad/stop":
                self.respond(200, self.server.manager.stop())
            else:
                self.respond(404, {"ok": False, "error": "not found"})
        except Exception as exc:
            self.respond(500, {"ok": False, "error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description="Atlas local gamepad control API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    manager = GamepadManager()
    server = ControlServer((args.host, args.port), Handler, manager)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    print(f"Atlas control API listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        manager.stop()
        server.server_close()


if __name__ == "__main__":
    main()
