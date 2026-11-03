"""Plot PD vs PPO Iris flight logs after training.

Run with regular activated VS Code .venv Python 3.13 (matplotlib installed).
Usage: python plot_ppo_comparison.py PATH_TO_RUN_FOLDER
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("run_folder", type=Path, help="Folder containing comparison.csv")
args = p.parse_args()
folder = args.run_folder.expanduser().resolve()
if not (folder / "comparison.csv").exists():
    raise FileNotFoundError(f"Cannot find comparison.csv in {folder}")

for wind in ("calm", "steady", "gust"):
    plt.figure(figsize=(8.5, 4.6))
    for controller in ("pd", "ppo"):
        path = folder / f"{controller}_{wind}.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        t = [float(r["time_s"]) for r in rows]
        y = [float(r["y"]) for r in rows]
        plt.plot(t, y, label=controller.upper())
    plt.axhline(0, linestyle="--", linewidth=1.0, label="Desired Y = 0")
    plt.axvline(5, linestyle=":", linewidth=1.0, label="Wind starts")
    plt.xlabel("Simulated time (s)")
    plt.ylabel("Sideways position Y (m)")
    plt.title(f"Iris: PD versus PPO residual control — {wind}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out = folder / f"comparison_{wind}.png"
    plt.savefig(out, dpi=170)
    plt.close()
    print(f"Saved {out}")
