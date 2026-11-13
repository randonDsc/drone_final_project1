"""Helpers shared by Isaac runners: preflight checks, provenance records, path redaction. No Isaac imports at module level."""
import contextlib
import hashlib
import importlib.util
import io
import platform
import subprocess
import sys
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RESIDUAL_DIR.parent
ISAAC_ROOT_TOKEN = "<ISAAC_ROOT>"
EXPECTED_PROTOCOL_SHA = "18658b9ce0801677369f6ee16d5d49467965fe96888a091d6d7762fd7376d0dc"
SHARED = [
    "path_sac_env", "path_sac_policy", "path_sac_config", "path_reference",
    "hover_env", "hover_task", "hover_ppo_config", "sensor_config",
    "state_estimation", "model_contract", "controller", "sim_config",
    "drone_sim", "motor", "wind", "drone_diagnostics",
    "hover_sac_policy", "hover_sac_config",
]


def in_residual(path):
    return path is not None and str(Path(path).resolve()).lower().startswith(str(RESIDUAL_DIR.resolve()).lower())


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def preflight(say):
    """Manifest checks and import-path guard, before Isaac starts. Returns the checks line."""
    spec = importlib.util.spec_from_file_location("sync_shared", RESIDUAL_DIR / "tools" / "sync_shared.py")
    sync_shared = importlib.util.module_from_spec(spec)
    sys.modules["sync_shared"] = sync_shared
    spec.loader.exec_module(sync_shared)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = sync_shared.check() if sync_shared.team_repo_configured() else sync_shared.check_local()
    check_line = buf.getvalue().strip().splitlines()[-1]
    say(f"checks: {check_line} (rc={rc})")
    if rc != 0:
        print(buf.getvalue())
        raise SystemExit("shared files do not match the manifest; not running")
    bad = {}
    for name in SHARED:
        found = importlib.util.find_spec(name)
        origin = found.origin if found else None
        if not in_residual(origin):
            bad[name] = origin
    if bad:
        raise SystemExit(f"shared modules resolve outside residual_ppo: {bad}")
    say(f"import paths: all {len(SHARED)} shared modules resolve inside residual_ppo")
    return check_line


def loaded_outside_residual():
    loaded = {n: getattr(sys.modules.get(n), "__file__", None) for n in SHARED if n in sys.modules}
    return {n: f for n, f in loaded.items() if f and not in_residual(f)}


def git_info():
    def run(*args):
        return subprocess.run(["git", "-C", str(REPO_DIR), *args], capture_output=True, text=True).stdout.strip()
    return {"head": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "residual_ppo_status": run("status", "--short", "--", "residual_ppo").splitlines()}


def isaac_root():
    for d in Path(sys.executable).resolve().parents:
        if (d / "VERSION").is_file() and (d / "python.bat").is_file():
            return d
    return None


def isaac_version():
    root = isaac_root()
    return (root / "VERSION").read_text(encoding="utf-8").strip() if root else "unknown"


def redact_isaac(path):
    """Committed files carry no machine paths: the Isaac install prefix becomes <ISAAC_ROOT> (from M2.2 on)."""
    path = str(Path(path).resolve())
    root = isaac_root()
    if root is not None and path.lower().startswith(str(root).lower()):
        return ISAAC_ROOT_TOKEN + path[len(str(root)):]
    return path


def hardware():
    info = {"platform": platform.platform(), "cpu": platform.processor()}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        info["gpu"] = out
    except Exception as exc:  # informational only
        info["gpu"] = f"unavailable: {exc}"
    return info


def hesai_report(sim):
    """HESAI lidar mass contribution (M1.5 method): rigid-body mass under its root."""
    import numpy as np
    from pxr import Sdf, UsdPhysics
    import sensor_config

    root = str(Sdf.Path(sensor_config.LIDAR["path"]).GetParentPath())
    prims = rigid = mass_api = 0
    for prim in sim.stage.Traverse():
        path = str(prim.GetPath())
        if path == root or path.startswith(root + "/"):
            prims += 1
            rigid += int(prim.HasAPI(UsdPhysics.RigidBodyAPI))
            mass_api += int(prim.HasAPI(UsdPhysics.MassAPI))
    body_paths = list(sim.vehicle_bodies.prim_paths)
    masses = [float(m) for m in np.asarray(sim.vehicle_bodies.get_masses()).reshape(-1)]
    rb_mass = sum(m for p, m in zip(body_paths, masses) if p == root or p.startswith(root + "/"))
    return {"root": root, "prims": prims, "rigid_body_prims": rigid, "mass_api_prims": mass_api,
            "hesai_mass_contribution_kg": rb_mass}
