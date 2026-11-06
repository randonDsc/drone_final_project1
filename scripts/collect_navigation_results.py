"""Retain small measured summaries and hashes of complete navigation inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("validation", "isaac", "paired", "training", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    suites = [("isaac_protocol", args.isaac, 54), ("navigation_paired", args.paired, 18)]
    for mode in ("quintic_stops", "isaac_timed", "isaac_waypoints"):
        suites += [("warehouse_"+mode, args.validation/("warehouse_"+mode), 12),
                   ("blocks_"+mode, args.validation/("blocks_"+mode), 3)]
    # Validate all inputs before creating the review bundle.
    flights = {}
    for name, directory, expected in suites:
        with (directory/"flights.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected or not (directory/"aggregate.json").is_file():
            raise ValueError(f"Incomplete {name}: expected {expected} measured flights, got {len(rows)}")
        flights[name] = rows
    training_result = json.loads((args.training/"training_result.json").read_text())
    if training_result["timesteps"] != 20000 or training_result["gradient_updates"] != 15000:
        raise ValueError("Expected the documented 20k/15k short training execution")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(artifacts={}, raw_flights={}, checkpoint={})

    def copy(source, relative):
        target = args.output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest["artifacts"][relative] = dict(source=str(source.resolve()), sha256=file_hash(source))

    for name, directory, _ in suites:
        for filename in ("flights.csv", "summary.csv", "aggregate.json", "suite.json", "summary.png",
                         "paired/paired.csv", "paired/comparison.json"):
            source = directory/filename
            if source.exists():
                copy(source, name+"/"+filename)
        for trial in sorted(directory.glob("*/metrics.json")):
            raw = trial.parent
            key = name+"/"+raw.name
            manifest["raw_flights"][key] = {filename: dict(path=str((raw/filename).resolve()), sha256=file_hash(raw/filename))
                         for filename in ("flight.csv", "metadata.json", "reference.npz", "metrics.json")}
        if name.startswith(("warehouse_", "blocks_")):
            representative = directory/"dijkstra_calm_300000_pid"
            for filename in ("metadata.json", "metrics.json", "navigation.png", "clearance.png", "flight.png",
                             "navigation/plan.json", "navigation/final_navigation.json", "navigation/waypoints_world.csv"):
                source = representative/filename
                if source.exists():
                    copy(source, name+"/calm_example/"+filename)
    for filename in ("run.json", "training_result.json", "episodes.csv.monitor.csv"):
        if (args.training/filename).exists():
            copy(args.training/filename, "sac_short/"+filename)
    for suffix in (".zip", ".contract.json", ".replay.pkl"):
        path = args.training/("final_model"+suffix)
        manifest["checkpoint"][path.name] = dict(path=str(path.resolve()), sha256=file_hash(path))
    (args.output/"ARTIFACTS.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Saved {sum(map(len, flights.values()))} measured flights to {args.output}")


if __name__ == "__main__":
    main()
