
"""Launch our trained ARL drone in Isaac Lab."""

import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_DIR / "checkpoints" / "drone_ppo.pt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--task",
        default="MyDrone-TrackPosition-v0",
    )
    args = parser.parse_args()

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Drone checkpoint not found: {CHECKPOINT}"
        )

    command = [
        "uv", "run", "--extra", "isaacsim",
        "isaaclab", "play",
        "--rl_library", "rsl_rl",
        "--task", args.task,
        "--num_envs", str(args.num_envs),
        "--viz", "kit",
        "--checkpoint", str(CHECKPOINT),
    ]

    print("Starting drone simulation...")
    print(f"Checkpoint: {CHECKPOINT}")

    subprocess.run(
        command,
        cwd=PROJECT_DIR,
        check=True,
    )

if __name__ == "__main__":
    main()
