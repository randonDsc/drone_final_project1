#!/usr/bin/env bash
# Headless source-protocol and navigation validation; refuses existing outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
drone_python="${DRONE_PYTHON:-.venv/bin/python}"
navigation_root="${1:-mujoco_output/navigation_validation}"
if [[ -e "$navigation_root" ]]; then
  echo "Result directory already exists: $navigation_root" >&2
  exit 1
fi
mkdir -p "$navigation_root"
"$drone_python" -m pytest tests_mujoco -q > "$navigation_root/tests.txt"
for mode in quintic_stops isaac_timed isaac_waypoints; do
  config_suffix="_$mode"
  if [[ "$mode" == quintic_stops ]]; then config_suffix=""; fi
  "$drone_python" -m mujoco_drone suite \
    --config "configs/warehouse_dijkstra${config_suffix}.json" --tasks dijkstra \
    --winds calm training_range unseen_gusts outside_range --seeds 300000 300001 300002 \
    --output "$navigation_root/warehouse_$mode"
  "$drone_python" -m mujoco_drone suite \
    --config "configs/blocks_dijkstra${config_suffix}.json" --tasks dijkstra \
    --winds calm gust turbulence --seeds 300000 --output "$navigation_root/blocks_$mode"
done
"$drone_python" -m mujoco_drone suite --config configs/isaac_path_evaluation.json \
  --seeds 300000 300001 300002 --output "$navigation_root/isaac_protocol_pid"
"$drone_python" -m mujoco_drone train --config configs/sac_train.json \
  --steps 20000 --seed 0 --checkpoint-interval 20000 --output "$navigation_root/sac_short"
"$drone_python" -m mujoco_drone suite --config configs/warehouse_dijkstra_isaac_timed.json \
  --tasks dijkstra --winds calm unseen_gusts outside_range --seeds 400000 400001 400002 \
  --model "$navigation_root/sac_short/final_model.zip" --output "$navigation_root/navigation_paired"
for mode in quintic_stops isaac_timed isaac_waypoints; do
  "$drone_python" -m mujoco_drone plot "$navigation_root/warehouse_$mode"
done
"$drone_python" -m mujoco_drone plot "$navigation_root/warehouse_quintic_stops/dijkstra_calm_300000_pid"
echo "Measured navigation outputs: $navigation_root"
