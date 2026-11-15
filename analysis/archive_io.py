"""Read trusted formats from the supplied archive without executing model data."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import zipfile


SCENARIOS = ("hover", "straight", "turns", "altitude", "circle", "figure_eight", "dijkstra")
PROFILES = ("training_range", "unseen_gusts", "outside_range")
CONTROLLERS = ("pid_only", "pid_sac")
METRICS = ("position_rmse", "cross_track_rmse", "max_deviation", "cross_track_max", "saturation_rate")
OUTCOMES = ("completed", "failed", "collided")
NUMERIC = METRICS + ("residual_norm", "residual_change", "route_seconds")


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0"):
        return False
    raise ValueError(f"Invalid outcome value: {value!r}")


def parse_rows(raw):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    required = set(NUMERIC + OUTCOMES + ("scenario", "wind_profile", "controller", "seed", "wind_seed", "decisions", "failure_reason", "completion_time"))
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Missing evaluation columns: "+", ".join(sorted(required-set(reader.fieldnames or []))))
    result = []
    for source in reader:
        row = dict(source)
        for field in NUMERIC:
            row[field] = float(source[field])
            if not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError(f"Invalid {field}: {source[field]}")
        for field in OUTCOMES:
            row[field] = parse_boolean(source[field])
        # Python integers preserve the 64-bit wind seeds; never round via float.
        for field in ("seed", "wind_seed", "decisions"):
            row[field] = int(source[field])
        if row["decisions"] <= 0 or row["route_seconds"] <= 0:
            raise ValueError("Recorded decision count and route duration must be positive")
        if row["saturation_rate"] > 1 or row["residual_norm"] > 2+1e-8:
            raise ValueError("Invalid saturation fraction or normalized residual norm")
        row["wind_speed"] = float(source["wind_speed"]) if source.get("wind_speed") else None
        if row["wind_speed"] is not None and (not math.isfinite(row["wind_speed"]) or row["wind_speed"] < 0):
            raise ValueError("Invalid mean wind speed")
        row["completion_time"] = float(source["completion_time"]) if source["completion_time"] else None
        if row["completion_time"] is not None and (not math.isfinite(row["completion_time"]) or row["completion_time"] < 0):
            raise ValueError("Invalid completion time")
        if row["failed"] != bool(row["failure_reason"]):
            raise ValueError("Failure flag and failure reason disagree")
        row["clearance_violation_rate"] = float(source["clearance_violation_rate"]) if source.get("clearance_violation_rate") else None
        if row["clearance_violation_rate"] is not None and not 0 <= row["clearance_violation_rate"] <= 1:
            raise ValueError("Invalid clearance violation fraction")
        result.append(row)
    if not result:
        raise ValueError("Empty evaluation CSV")
    return result


def pair_rows(rows, *, expected_scenarios=None, expected_profiles=None):
    index = {}
    for row in rows:
        if row["controller"] not in CONTROLLERS:
            raise ValueError(f"Unknown controller: {row['controller']}")
        key = (row["scenario"], row["wind_profile"], row["seed"])
        group = index.setdefault(key, {})
        if row["controller"] in group:
            raise ValueError(f"Duplicate flight: {key}/{row['controller']}")
        group[row["controller"]] = row
    scenarios = set(r["scenario"] for r in rows)
    profiles = set(r["wind_profile"] for r in rows)
    seeds = set(r["seed"] for r in rows)
    if expected_scenarios is not None and scenarios != set(expected_scenarios):
        raise ValueError("Scenario set differs from the declared evaluation")
    if expected_profiles is not None and profiles != set(expected_profiles):
        raise ValueError("Wind profile set differs from the declared evaluation")
    expected = {(s, w, seed) for s in scenarios for w in profiles for seed in seeds}
    if set(index) != expected:
        raise ValueError("Incomplete scenario/profile/seed grid")
    pairs = []
    for key in sorted(index):
        members = index[key]
        if set(members) != set(CONTROLLERS):
            raise ValueError(f"Missing PID or SAC member: {key}")
        pid, sac = (members[c] for c in CONTROLLERS)
        if pid["wind_seed"] != sac["wind_seed"]:
            raise ValueError(f"Prescribed wind seed mismatch: {key}")
        for field in ("route_seconds", "wind_speed"):
            a, b = pid[field], sac[field]
            if (a is None) != (b is None) or (a is not None and not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(f"Paired {field} mismatch: {key}")
        if pid["controller"] == "pid_only" and (pid["residual_norm"] != 0 or pid["residual_change"] != 0):
            raise ValueError(f"Nonzero residual in PID-only flight: {key}")
        pairs.append(dict(scenario=key[0], wind_profile=key[1], seed=key[2], wind_seed=pid["wind_seed"],
                          wind_speed=pid["wind_speed"], route_seconds=pid["route_seconds"], pid=pid, sac=sac))
    return pairs


def validate_reference_files(rows, results, summary_csv):
    """Cross-check original tables, including the source's max-over-seeds fields."""
    def key(r):
        return r["scenario"], r["wind_profile"], int(r["seed"]), r["controller"]
    reference = {key(r): r for r in results["flights"]}
    if len(reference) != len(results["flights"]) or set(reference) != {key(r) for r in rows}:
        raise ValueError("CSV/JSON flight identifiers differ or repeat")
    for row in rows:
        original = reference[key(row)]
        for field in NUMERIC + OUTCOMES + ("seed", "wind_seed", "decisions", "completion_time", "wind_speed", "clearance_violation_rate", "failure_reason"):
            a, b = row.get(field), original.get(field)
            if field == "failure_reason":
                # CSV writes the JSON null reason as an empty cell.
                a, b = a or None, b or None
            if isinstance(a, float) and b is not None:
                if not math.isclose(a, float(b), rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f"CSV/JSON mismatch: {key(row)}/{field}")
            elif a != b:
                raise ValueError(f"CSV/JSON mismatch: {key(row)}/{field}")
    groups = {}
    for row in rows:
        groups.setdefault((row["scenario"], row["wind_profile"], row["controller"]), []).append(row)
    summaries = list(csv.DictReader(io.StringIO(summary_csv.decode("utf-8-sig"))))
    summary_keys = [(r["scenario"], r["wind_profile"], r["controller"]) for r in summaries]
    json_summary = {(r["scenario"], r["wind_profile"], r["controller"]): r for r in results["summary"]}
    if (len(set(summary_keys)) != len(summaries) or len(json_summary) != len(results["summary"])
            or set(summary_keys) != set(groups) or set(json_summary) != set(groups)):
        raise ValueError("Summary group identifiers differ or repeat")
    for summary in summaries:
        k = summary["scenario"], summary["wind_profile"], summary["controller"]
        group = groups[k]
        values = {"flights": len(group)}
        for field in ("position_rmse", "cross_track_rmse", "saturation_rate", "residual_norm", "residual_change"):
            values[field] = sum(r[field] for r in group)/len(group)
        for field in ("cross_track_max", "max_deviation"):
            values[field] = max(r[field] for r in group)
        for field, output in (("completed", "completion_rate"), ("failed", "failure_rate"), ("collided", "collision_rate")):
            values[output] = sum(r[field] for r in group)/len(group)
        times = [r["completion_time"] for r in group if r["completion_time"] is not None]
        values["completion_time"] = sum(times)/len(times) if times else None
        for field, expected in values.items():
            for actual in (summary.get(field), json_summary[k].get(field)):
                if expected is None:
                    if actual not in (None, ""):
                        raise ValueError(f"Summary mismatch: {k}/{field}")
                elif actual in (None, "") or not math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f"Summary mismatch: {k}/{field}")


def model_metadata(raw):
    """Read JSON and hash weight bytes, without pickle/cloudpickle/torch loading."""
    with zipfile.ZipFile(io.BytesIO(raw)) as model:
        data = json.loads(model.read("data"))
        return dict(model_sha256=hashlib.sha256(raw).hexdigest(),
                    policy_weights_sha256=hashlib.sha256(model.read("policy.pth")).hexdigest(),
                    timesteps=data["num_timesteps"], gradient_updates=data.get("_n_updates"),
                    training_seed=data.get("seed"), observation_shape=data["observation_space"]["_shape"],
                    action_shape=data["action_space"]["_shape"],
                    stable_baselines3=model.read("_stable_baselines3_version").decode().strip(),
                    system_info=model.read("system_info.txt").decode(),
                    deserialized_model=False)


def load_archive(path, run, source_output):
    path, source_output = Path(path), Path(source_output)
    if run not in ("flight1", "flight2"):
        raise ValueError("Select an audited flight1 or flight2 run")
    prefix = f"runs/flight_sac/{run}/"
    evaluation = prefix+"evaluation/final_model/"
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate ZIP member names")
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name:
                raise ValueError("Unsafe archive member path")
        if evaluation+"flights.csv" not in names:
            raise ValueError(f"No saved evaluation for {run}; no simulator is run as a fallback")
        raw_csv = archive.read(evaluation+"flights.csv")
        rows = parse_rows(raw_csv)
        pairs = pair_rows(rows, expected_scenarios=SCENARIOS, expected_profiles=PROFILES)
        raw_results = archive.read(evaluation+"results.json")
        raw_summary = archive.read(evaluation+"summary.csv")
        validate_reference_files(rows, json.loads(raw_results), raw_summary)
        model = model_metadata(archive.read(prefix+"final_model.zip"))
        contract = json.loads(archive.read(prefix+"flight_contract.json"))
        if model["observation_shape"] != [128] or model["action_shape"] != [4]:
            raise ValueError("Unexpected model observation/action dimensions")
        source_output.mkdir(parents=True, exist_ok=False)
        copied = {}
        selected = [(evaluation+f, f) for f in ("flights.csv", "summary.csv", "results.json", "evaluation_report.pdf")]
        selected += [(prefix+"flight_contract.json", "flight_contract.json")]
        selected += [(prefix+"eval_log.txt", "eval_log.txt")]
        selected += [(n, PurePosixPath(n).name) for n in names if n.startswith(prefix+"config_snapshot_") and n.endswith(".json")]
        # Preserve the small configuration evidence for the second run's lineage.
        selected += [(n, "flight2_"+PurePosixPath(n).name) for n in names
                     if n.startswith("runs/flight_sac/flight2/config_snapshot_") and n.endswith(".json")]
        selected += [("runs/flight_sac/flight2/flight_contract.json", "flight2_contract.json")]
        for source, target in selected:
            if source not in names:
                continue
            raw = archive.read(source)
            (source_output/target).write_bytes(raw)
            copied[target] = dict(archive_member=source, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        other_models = {}
        for other in ("flight1", "flight2"):
            name = f"runs/flight_sac/{other}/final_model.zip"
            if name in names:
                meta = model if other == run else model_metadata(archive.read(name))
                c = json.loads(archive.read(f"runs/flight_sac/{other}/flight_contract.json"))
                meta = dict(meta, curriculum=[dict(name=s["name"], until_step=s["until_step"], wind=s["wind"]) for s in c["training"]["curriculum"]],
                            evaluation_present=f"runs/flight_sac/{other}/evaluation/final_model/flights.csv" in names)
                other_models[other] = meta
        inventory = [dict(path=m.filename, bytes=m.file_size, compressed_bytes=m.compress_size,
                          modified_zip_local=list(m.date_time), crc32=f"{m.CRC:08x}") for m in members]
    audit = dict(flights=len(rows), pairs=len(pairs), conditions=len(SCENARIOS)*len(PROFILES),
                 evaluation_seeds=sorted({r["seed"] for r in rows}),
                 pairing_fields_verified=["scenario", "wind_profile", "seed", "wind_seed", "wind_speed", "route_seconds"],
                 duplicate_flights=0, missing_pairs=0, source_csv_json_summary_agree=True,
                 exact_initial_states_and_reference_hashes_available=False,
                 per_timestep_evaluation_traces_available=False,
                 different_recorded_lengths=sum(p["pid"]["decisions"] != p["sac"]["decisions"] for p in pairs),
                 completion_and_failure_overlap=sum(r["completed"] and r["failed"] for r in rows))
    metadata = dict(archive_path=str(path.resolve()), archive_bytes=path.stat().st_size,
                            archive_sha256=sha256_file(path), archive_inventory=inventory,
                            selected_run=run, selected_model=model, models=other_models,
                            contract=contract, source_files=copied, validation=audit)
    json_write(source_output/"dataset_metadata.json", metadata)
    return rows, pairs, metadata


def load_source_data(path, source_output):
    """Reproduce statistics using the small review bundle, without the ZIP."""
    path, source_output = Path(path), Path(source_output)
    metadata_raw = (path/"dataset_metadata.json").read_bytes()
    metadata = json.loads(metadata_raw)
    source_output.mkdir(parents=True, exist_ok=False)
    for name, entry in metadata["source_files"].items():
        if Path(name).name != name:
            raise ValueError("Source bundle filenames must be basenames")
        raw = (path/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Source file hash mismatch: "+name)
        (source_output/name).write_bytes(raw)
    (source_output/"dataset_metadata.json").write_bytes(metadata_raw)
    rows = parse_rows((source_output/"flights.csv").read_bytes())
    pairs = pair_rows(rows, expected_scenarios=SCENARIOS, expected_profiles=PROFILES)
    validate_reference_files(rows, json.loads((source_output/"results.json").read_text()),
                             (source_output/"summary.csv").read_bytes())
    return rows, pairs, metadata
