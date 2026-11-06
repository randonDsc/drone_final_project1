#!/usr/bin/env bash
# Run after installing .[rl,plot,test]. Never overwrites an existing result tree.
set -euo pipefail
cd "$(dirname "$0")/.."
drone_python="${DRONE_PYTHON:-.venv/bin/python}"
validation_root="${1:-mujoco_output/reproduced_validation}"
if [[ -e "$validation_root" ]]; then
  echo "Result directory already exists: $validation_root" >&2
  exit 1
fi
mkdir -p "$validation_root"
"$drone_python" -m pytest tests_mujoco -q > "$validation_root/tests.txt"
"$drone_python" -m mujoco_drone run --config configs/original_pid_replay.json \
  --output "$validation_root/original_pid"
"$drone_python" -m mujoco_drone suite --config configs/evaluation.json \
  --seeds 300000 300001 300002 --output "$validation_root/pid"
"$drone_python" -m mujoco_drone train --config configs/sac_train.json \
  --steps 20000 --seed 0 --checkpoint-interval 10000 --output "$validation_root/sac"
"$drone_python" -m mujoco_drone suite \
  --tasks hover straight waypoints circle figure_eight --winds calm gust turbulence \
  --seeds 400000 400001 400002 --model "$validation_root/sac/final_model.zip" \
  --output "$validation_root/paired"
"$drone_python" -m mujoco_drone suite --config configs/gust_stress.json \
  --tasks hover straight --winds gust --seeds 500000 500001 500002 \
  --model "$validation_root/sac/final_model.zip" --output "$validation_root/stress"
for suite in pid paired stress; do
  "$drone_python" -m mujoco_drone plot "$validation_root/$suite"
done
if [[ "${DRONE_RENDER_TEST:-0}" == "1" ]]; then
  MUJOCO_TEST_RENDER=1 "$drone_python" -m pytest tests_mujoco/test_recording.py::test_camera_rgb_depth_and_known_plane_distance -q
  "$drone_python" -m mujoco_drone run --config configs/camera_hover.json \
    --output "$validation_root/camera"
fi
echo "Measured validation outputs: $validation_root"
