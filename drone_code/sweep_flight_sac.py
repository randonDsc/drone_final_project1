"""Train and evaluate every reward-weight variant, one after another.

    /c/isaacsim/python.bat drone_code/sweep_flight_sac.py

Takes no arguments: variants are in flight_config.FLIGHT_SWEEP. Each is a separately
trained model in runs/flight_sac/<base_run_name>_<variant>/. A variant that already
has a final model is only evaluated again. To train them anew, set a new
base_run_name. Compare the summary.csv files under each evaluation folder.
"""
import os
import subprocess
import sys
from pathlib import Path

import flight_config


folder = Path(__file__).resolve().parent
project_dir = folder.parent
sweep = flight_config.FLIGHT_SWEEP

for variant in sweep["variants"]:
    run_dir = (
        project_dir
        / flight_config.RUN_DIR
        / f"{sweep['base_run_name']}_{variant}"
    )

    environment = {
        **os.environ,
        flight_config.SWEEP_VARIANT_VARIABLE: variant,
    }

    scripts = ["eval_flight_sac.py"]

    if not (run_dir / "final_model.zip").is_file():
        scripts.insert(0, "train_flight_sac.py")

    for script in scripts:
        print(f"Sweep {variant}: {script}", flush=True)

        result = subprocess.run(
            [sys.executable, str(folder / script)],
            env=environment,
        )

        if result.returncode != 0:
            raise SystemExit(f"Sweep stopped: {script} failed for {variant}.")

print("Sweep finished.", flush=True)
