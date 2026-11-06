"""Copy small measured review artifacts, retaining hashes of their raw inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pid", "paired", "stress", "training", "original", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for name, directory, expected in (("pid", args.pid, 90), ("paired", args.paired, 90), ("stress", args.stress, 12)):
        with (directory/"flights.csv").open() as file:
            rows = list(csv.DictReader(file))
        if len(rows) != expected:
            raise ValueError(f"Incomplete {name} suite: expected {expected}, got {len(rows)}")
        files = [directory/"flights.csv", directory/"summary.csv", directory/"suite.json", directory/"aggregate.json"]
        if (directory/"summary.png").exists():
            files.append(directory/"summary.png")
        if name != "pid":
            files += [directory/"paired"/"paired.csv", directory/"paired"/"comparison.json"]
        for source in files:
            target = args.output/f"{name}_{source.name}"
            if source.parent == directory/"paired":
                target = args.output/f"{name}_{'deltas.csv' if source.name == 'paired.csv' else source.name}"
            shutil.copy2(source, target)
            manifest[target.name] = dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    for source, name in ((args.training/"training_result.json", "sac_training.json"),
                         (args.training/"run.json", "sac_run.json"),
                         (args.original/"metrics.json", "original_pid_metrics.json"),
                         (args.original/"metadata.json", "original_pid_metadata.json")):
        shutil.copy2(source, args.output/name)
        manifest[name] = dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (args.output/"ARTIFACTS.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
