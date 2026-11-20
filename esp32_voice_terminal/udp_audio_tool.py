import argparse
import socket
import time
import wave
from pathlib import Path

SAMPLE_RATE = 16000
FRAME_SAMPLES = 320
FRAME_BYTES = FRAME_SAMPLES * 2
MIC_PORT = 5004
SPEAKER_PORT = 5006


def record(args):
    output = Path(args.output)
    received = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, wave.open(str(output), "wb") as wav:
        sock.bind((args.bind, MIC_PORT))
        sock.settimeout(0.25)
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        deadline = time.monotonic() + args.seconds
        print(f"Listening on UDP {MIC_PORT}; target file: {output}")
        while time.monotonic() < deadline:
            try:
                data, address = sock.recvfrom(2048)
            except socket.timeout:
                continue
            if len(data) != FRAME_BYTES:
                print(f"Ignored {len(data)}-byte packet from {address}")
                continue
            wav.writeframes(data)
            received += 1
    print(f"Saved {received} frames ({received * FRAME_SAMPLES / SAMPLE_RATE:.2f}s) to {output}")


def play(args):
    with wave.open(str(args.input), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, SAMPLE_RATE):
            raise SystemExit("Input must be 16 kHz, mono, 16-bit PCM WAV")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            target = (args.target, SPEAKER_PORT)
            next_frame = time.monotonic()
            sent = 0
            while True:
                data = wav.readframes(FRAME_SAMPLES)
                if not data:
                    break
                if len(data) < FRAME_BYTES:
                    data = data.ljust(FRAME_BYTES, b"\x00")
                sock.sendto(data, target)
                sent += 1
                next_frame += FRAME_SAMPLES / SAMPLE_RATE
                time.sleep(max(0.0, next_frame - time.monotonic()))
    print(f"Sent {sent} frames to {args.target}:{SPEAKER_PORT}")


def main():
    parser = argparse.ArgumentParser(description="Test the ESP32 voice terminal with PCM WAV files.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    record_parser = subparsers.add_parser("record", help="Record microphone UDP packets to WAV")
    record_parser.add_argument("--bind", default="0.0.0.0")
    record_parser.add_argument("--seconds", type=float, default=5.0)
    record_parser.add_argument("--output", default="mic_capture.wav")
    record_parser.set_defaults(function=record)
    play_parser = subparsers.add_parser("play", help="Send a WAV file to the ESP32 speaker")
    play_parser.add_argument("input")
    play_parser.add_argument("--target", default="192.168.4.1")
    play_parser.set_defaults(function=play)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
