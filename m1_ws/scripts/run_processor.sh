#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-hardware}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WS_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(cd -- "$WS_DIR/.." && pwd)

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /root/resident-integration-20261021/ros/setup.bash
source "$WS_DIR/install/setup.bash"
export PYTHONUNBUFFERED=1

if [[ "$MODE" == "mock" ]]; then
  exec ros2 launch m1_vision processor_bringup.launch.py \
    use_mock:=true enable_asr:=false enable_semantic:=false \
    voice_root:="$PROJECT_DIR/voice_control"
elif [[ "$MODE" == "hardware" ]]; then
  exec ros2 launch m1_vision processor_bringup.launch.py \
    use_mock:=false enable_asr:="${ENABLE_ASR:-true}" \
    enable_tts:="${ENABLE_TTS:-false}" enable_semantic:="${ENABLE_SEMANTIC:-false}" \
    voice_root:="$PROJECT_DIR/voice_control"
else
  echo "usage: $0 [hardware|mock]" >&2
  exit 2
fi
