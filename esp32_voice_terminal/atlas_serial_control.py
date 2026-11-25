import argparse
import glob
import json
import signal
import socket
import struct
import threading
import time
from pathlib import Path

import serial

from atlas_control_server import GamepadManager
from atlas_demo_recorder import DemoCommandManager

USB_AUDIO_MAGIC = b"\xaa\x55"
USB_FRAME_MIC = 0x01
USB_FRAME_SPK = 0x02
USB_FRAME_SUBTITLE = 0x03
PCM_FRAME_BYTES = 640  # 20 ms @ 16 kHz, 16-bit mono
SUBTITLE_FRAME_BYTES = 560  # 160x28 1-bit bitmap (top 14px + bottom 14px)


def dispatch(manager, command, demo_manager=None, payload=None):
    payload = payload or {}
    if command == "start":
        return manager.start()
    if command == "stop":
        return manager.stop()
    if command == "status":
        return {"ok": True, "status": "status", **manager.status()}
    if command == "demo_start":
        if not demo_manager:
            return {"ok": False, "error": "demo unavailable"}
        return demo_manager.start(task=payload.get("task") or "pick up the orange")
    if command == "demo_stop":
        return demo_manager.stop() if demo_manager else {"ok": False, "error": "demo unavailable"}
    if command == "demo_status":
        return {"ok": True, **(demo_manager.status() if demo_manager else {"status": "unavailable"})}
    return {"ok": False, "error": "unknown command"}


def _speaker_udp_loop(link, write_lock, stop_event, speaker_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", speaker_port))
    sock.settimeout(0.2)
    header_prefix = struct.pack("<2sBBH", USB_AUDIO_MAGIC, USB_FRAME_SPK, 0, PCM_FRAME_BYTES)
    try:
        while not stop_event.is_set():
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) != PCM_FRAME_BYTES:
                continue
            try:
                with write_lock:
                    link.write(header_prefix + data)
            except serial.SerialException:
                break
    finally:
        sock.close()


def _handle_json_line(line, link, write_lock, manager, demo_manager):
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        return
    if not isinstance(request, dict):
        return
    try:
        response = dispatch(manager, request.get("cmd"), demo_manager, request)
    except Exception as exc:
        response = {"ok": False, "error": str(exc)}
    raw = json.dumps(response, ensure_ascii=False).encode() + b"\n"
    with write_lock:
        link.write(raw)
        link.flush()


def _tts_text_udp_loop(link, write_lock, stop_event, tts_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", tts_port))
    sock.settimeout(0.2)
    try:
        while not stop_event.is_set():
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            text = data.decode("utf-8", "replace").strip()
            if not text:
                continue
            if text.startswith("{"):
                raw = text.encode("utf-8") + b"\n"
            else:
                raw = json.dumps({"cmd": "tts", "text": text}, ensure_ascii=False).encode("utf-8") + b"\n"
            try:
                with write_lock:
                    link.write(raw)
                    link.flush()
            except serial.SerialException:
                break
    finally:
        sock.close()


def _subtitle_udp_loop(link, write_lock, stop_event, subtitle_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", subtitle_port))
    sock.settimeout(0.2)
    header_prefix = struct.pack("<2sBBH", USB_AUDIO_MAGIC, USB_FRAME_SUBTITLE, 0, SUBTITLE_FRAME_BYTES)
    try:
        while not stop_event.is_set():
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) != SUBTITLE_FRAME_BYTES:
                continue
            try:
                with write_lock:
                    link.write(header_prefix + data)
                    link.flush()
            except serial.SerialException:
                break
    finally:
        sock.close()


def serve(manager, port, demo_manager=None, mic_udp_port=5004, speaker_udp_port=5006, tts_udp_port=5007, subtitle_udp_port=5008, enable_audio=True):
    mic_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    write_lock = threading.Lock()
    stop_event = threading.Event()
    try:
        with serial.Serial(port, 115200, timeout=0.05) as link:
            print(f"Listening on {port} (mic->{mic_udp_port}, spk<-{speaker_udp_port}, tts<-{tts_udp_port}, sub<-{subtitle_udp_port})", flush=True)
            spk_thread = threading.Thread(
                target=_speaker_udp_loop,
                args=(link, write_lock, stop_event, speaker_udp_port),
                daemon=True,
            )
            spk_thread.start()
            tts_thread = threading.Thread(
                target=_tts_text_udp_loop,
                args=(link, write_lock, stop_event, tts_udp_port),
                daemon=True,
            )
            tts_thread.start()
            sub_thread = threading.Thread(
                target=_subtitle_udp_loop,
                args=(link, write_lock, stop_event, subtitle_udp_port),
                daemon=True,
            )
            sub_thread.start()
            last_audio_req = 0.0
            mic_frames = 0
            if enable_audio:
                with write_lock:
                    link.write(b'{"cmd":"audio_on"}\n')
                    link.flush()
                last_audio_req = time.monotonic()

            buf = bytearray()
            while True:
                if enable_audio and mic_frames == 0 and time.monotonic() - last_audio_req >= 1.5:
                    with write_lock:
                        link.write(b'{"cmd":"audio_on"}\n')
                        link.flush()
                    last_audio_req = time.monotonic()
                chunk = link.read(max(1, min(4096, link.in_waiting or 1)))
                if not chunk:
                    continue
                buf.extend(chunk)
                while buf:
                    b0 = buf[0]
                    if b0 == 0xAA:
                        if len(buf) < 6:
                            break
                        if buf[1] != 0x55:
                            del buf[0]
                            continue
                        frame_type = buf[2]
                        payload_len = buf[4] | (buf[5] << 8)
                        if payload_len > 2048:
                            del buf[:2]
                            continue
                        if len(buf) < 6 + payload_len:
                            break
                        payload = bytes(buf[6 : 6 + payload_len])
                        del buf[: 6 + payload_len]
                        if frame_type == USB_FRAME_MIC and payload_len == PCM_FRAME_BYTES:
                            if mic_frames == 0:
                                print("[AUDIO] Mic stream active", flush=True)
                            mic_frames += 1
                            try:
                                mic_sock.sendto(payload, ("127.0.0.1", mic_udp_port))
                            except OSError:
                                pass
                    elif b0 == ord("{"):
                        nl = buf.find(b"\n")
                        if nl < 0:
                            if len(buf) > 1024:
                                del buf[:]
                            break
                        line = bytes(buf[:nl]).decode("utf-8", "replace").strip()
                        del buf[: nl + 1]
                        if line:
                            threading.Thread(
                                target=_handle_json_line,
                                args=(line, link, write_lock, manager, demo_manager),
                                daemon=True,
                            ).start()
                    else:
                        next_aa = buf.find(b"\xaa")
                        next_lb = buf.find(b"{")
                        candidates = [i for i in (next_aa, next_lb) if i > 0]
                        if candidates:
                            del buf[: min(candidates)]
                        else:
                            del buf[:]
    finally:
        stop_event.set()
        mic_sock.close()


def resolve_port(pattern):
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else pattern


def _safe_cleanup(manager, demo_manager):
    try:
        demo_manager.stop()
    except Exception as exc:
        print(f"Cleanup demo stop error: {exc}", flush=True)
    try:
        manager.stop()
    except Exception as exc:
        print(f"Cleanup gamepad stop error: {exc}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Atlas USB serial control and audio multiplexer")
    parser.add_argument("--port", default="/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_*-if00")
    parser.add_argument("--mic-udp-port", type=int, default=5004)
    parser.add_argument("--speaker-udp-port", type=int, default=5006)
    parser.add_argument("--tts-udp-port", type=int, default=5007)
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    manager = GamepadManager()
    demo_manager = DemoCommandManager(manager, Path(__file__).resolve().parent)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    try:
        while True:
            try:
                serve(
                    manager,
                    resolve_port(args.port),
                    demo_manager,
                    mic_udp_port=args.mic_udp_port,
                    speaker_udp_port=args.speaker_udp_port,
                    tts_udp_port=args.tts_udp_port,
                    enable_audio=not args.no_audio,
                )
            except serial.SerialException as exc:
                # 串口瞬断重连时不强杀正在运行的手柄遥控，避免电机启动/USB抖动导致遥控中断
                print(f"Serial unavailable: {exc}; retrying (keeping gamepad alive)", flush=True)
                time.sleep(1.0)
    finally:
        _safe_cleanup(manager, demo_manager)


if __name__ == "__main__":
    main()
