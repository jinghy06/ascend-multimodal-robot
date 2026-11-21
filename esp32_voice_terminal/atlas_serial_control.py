import argparse
import json
import signal
import time

import serial

from atlas_control_server import GamepadManager


def dispatch(manager, command):
    if command == "start":
        return manager.start()
    if command == "stop":
        return manager.stop()
    if command == "status":
        return {"ok": True, "status": "status", **manager.status()}
    return {"ok": False, "error": "unknown command"}


def serve(manager, port):
    with serial.Serial(port, 115200, timeout=0.2) as link:
        print(f"Listening on {port}", flush=True)
        while True:
            line = link.readline().decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(request, dict):
                continue
            try:
                response = dispatch(manager, request.get("cmd"))
            except Exception as exc:
                response = {"ok": False, "error": str(exc)}
            link.write(json.dumps(response, ensure_ascii=False).encode() + b"\n")
            link.flush()


def main():
    parser = argparse.ArgumentParser(description="Atlas USB serial gamepad control")
    parser.add_argument("--port", default="/dev/ttyACM1")
    args = parser.parse_args()
    manager = GamepadManager()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    try:
        while True:
            try:
                serve(manager, args.port)
            except serial.SerialException as exc:
                manager.stop()
                print(f"Serial unavailable: {exc}; retrying", flush=True)
                time.sleep(0.5)
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
