"""M1.5 Isaac environment smoke test: build PathSacEnv from residual_ppo/, reset, step, check.

NOT PID flight validation (M2) and trains nothing. Writes only:
    residual_ppo/results/m1_env_config_snapshot.json
    residual_ppo/results/m1_env_smoke_summary.txt
Never writes under runs/. Run with Isaac python.bat from the repo root:

    # Directory: <repo root> (drone_final_project1)
    & "C:\\isaacsim\\isaac-sim-standalone-6.2.0-windows-x86_64\\python.bat" residual_ppo\\tests\\isaac_env_smoke.py *> residual_ppo\\results\\m1_env_smoke.log

Lines that start with "[M1.5]" are this script's output; everything else is Isaac's log.
The script's verdict is the last "[M1.5] RESULT" line (written before app.close(), which may end the process).
"""
import contextlib
import importlib.util
import io
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RESIDUAL_DIR.parent
RESULTS_DIR = RESIDUAL_DIR / "results"
SNAPSHOT_PATH = RESULTS_DIR / "m1_env_config_snapshot.json"
SUMMARY_PATH = RESULTS_DIR / "m1_env_smoke_summary.txt"

SHARED = [
    "path_sac_env", "path_sac_policy", "path_sac_config", "path_reference",
    "hover_env", "hover_task", "hover_ppo_config", "sensor_config",
    "state_estimation", "model_contract", "controller", "sim_config",
    "drone_sim", "motor", "wind", "drone_diagnostics",
    "hover_sac_policy", "hover_sac_config",
]
CURRICULUM_STEPS = (0, 149_999, 150_000, 400_000, 800_000, 1_500_000)
EXPECTED_STAGES = ("short_slow", "short_slow", "turns", "climbs_loops", "full", "full")
ZERO_ACTION_DECISIONS = 30

sys.path.insert(0, str(RESIDUAL_DIR))

summary_lines = []
failures = []


def say(msg=""):
    line = f"[M1.5] {msg}"
    print(line, flush=True)
    summary_lines.append(line)


def expect(ok, what):
    if not ok:
        failures.append(what)
        say(f"  CHECK FAILED: {what}")
    return ok


def jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def rel(path):
    """Repo-relative path for logs and results (no machine-specific prefix)."""
    try:
        return Path(path).resolve().relative_to(REPO_DIR).as_posix()
    except ValueError:
        return str(path)


def in_residual(path):
    return path is not None and Path(path).resolve().parent == RESIDUAL_DIR


def imported_packages():
    """Version and location of the packages as imported in this process (after SimulationApp)."""
    import gymnasium
    import numpy
    import stable_baselines3
    import torch
    from stable_baselines3.common.utils import get_device

    out = {m.__name__: {"version": m.__version__, "file": m.__file__}
           for m in (numpy, gymnasium, stable_baselines3, torch)}
    out["torch"].update(cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda,
                        sb3_auto_device=str(get_device("auto")))
    return out


# ---------- 0. Before Isaac: shared files must be intact, imports must resolve here ----------
say(f"started {datetime.now(timezone.utc).isoformat(timespec='seconds')}  python {platform.python_version()}")
say(f"residual_ppo: {rel(RESIDUAL_DIR)}")

spec = importlib.util.spec_from_file_location("sync_shared", RESIDUAL_DIR / "tools" / "sync_shared.py")
sync_shared = importlib.util.module_from_spec(spec)
sys.modules["sync_shared"] = sync_shared  # dataclasses need the module registered before exec
spec.loader.exec_module(sync_shared)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    check_rc = sync_shared.check()
check_line = buf.getvalue().strip().splitlines()[-1]
say(f"sync_shared --check: {check_line} (rc={check_rc})")
if check_rc != 0:
    print(buf.getvalue())
    say("RESULT: FAIL (shared files do not match the manifest; not starting Isaac)")
    sys.exit(1)
manifest = json.loads((RESIDUAL_DIR / "SHARED_MANIFEST.json").read_text(encoding="utf-8"))

bad_specs = {n: (importlib.util.find_spec(n).origin if importlib.util.find_spec(n) else None) for n in SHARED}
bad_specs = {n: o for n, o in bad_specs.items() if not in_residual(o)}
expect(not bad_specs, f"find_spec outside residual_ppo: {bad_specs}")
drone_code_paths = [p for p in sys.path if p.rstrip("\\/").lower().endswith("drone_code")]
expect(not drone_code_paths, f"sys.path contains drone_code entries: {drone_code_paths}")
runs_before = (REPO_DIR / "runs").exists(), (RESIDUAL_DIR / "runs").exists()
if failures:
    say("RESULT: FAIL (import paths)")
    sys.exit(1)

# ---------- 1. Isaac ----------
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
env = None
snapshot = {}
try:
    import numpy as np
    from pxr import Sdf, UsdPhysics

    # ---------- 2. Imports after SimulationApp; every shared module must come from residual_ppo ----------
    import path_sac_env
    import path_sac_config
    import controller
    import hover_env
    import hover_ppo_config
    import sensor_config
    import sim_config

    for mod in (path_sac_env, path_sac_config, controller):
        say(f"{mod.__name__:<16} {rel(mod.__file__)}")
    scene_path = Path(hover_env.__file__).resolve().parents[1] / hover_ppo_config.SCENE
    say(f"scene: {rel(scene_path)}  exists={scene_path.is_file()}")
    expect(scene_path.is_file(), "scene file exists")

    # ---------- 3. Build ----------
    env = path_sac_env.make_path_sac_env(app)
    sim, core = env.sim, env.core

    loaded = {n: getattr(sys.modules.get(n), "__file__", None) for n in SHARED if n in sys.modules}
    outside = {n: f for n, f in loaded.items() if not in_residual(f)}
    not_loaded = [n for n in SHARED if n not in loaded]
    say(f"shared modules loaded after build: {len(loaded)}/{len(SHARED)}; not loaded: {not_loaded or 'none'}; "
        f"outside residual_ppo: {outside or 'none'}")
    expect(not outside, f"loaded modules outside residual_ppo: {outside}")

    say(f"action_space={env.action_space}  observation_space={env.observation_space}")
    say(f"decimation={core.decimation} (expect 12)  dt={sim.dt!r} (expect {1/120!r})")
    expect(core.decimation == 12, "decimation == 12")
    expect(abs(sim.dt - 1.0 / 120.0) < 1e-12, "dt == 1/120")
    say(f"residual limits (config)={path_sac_config.RESIDUAL_LIMITS}  array={core.limits.tolist()}")

    # ---------- 3b. Mass report ----------
    stage = sim.stage
    body_paths = list(sim.vehicle_bodies.prim_paths)
    body_masses = [float(m) for m in np.asarray(sim.vehicle_bodies.get_masses()).reshape(-1)]
    total_mass = float(sim.total_mass)
    say(f"vehicle rigid bodies under {sim_config.DRONE_ROOT}: {len(body_paths)}")
    for p, m in zip(body_paths, body_masses):
        say(f"    {p}: {m:.6f} kg")
    # drone_sim sums PhysX's float32 masses in float32; compare like for like, and show the float64 gap.
    sum32 = float(np.asarray(sim.vehicle_bodies.get_masses()).reshape(-1).sum())
    gap64 = abs(total_mass - sum(body_masses))
    say(f"total_mass_kg = {total_mass:.6f}  (float32 sum of bodies {sum32:.6f}; |float64 sum - total| = {gap64:.2e} kg)")
    expect(total_mass == sum32 and gap64 < 1e-5, "total_mass == sum of body masses (float32; float64 gap < 1e-5 kg)")

    hesai_root = str(Sdf.Path(sensor_config.LIDAR["path"]).GetParentPath())
    hesai_prim = stage.GetPrimAtPath(hesai_root)
    hesai = {"root": hesai_root, "exists": bool(hesai_prim and hesai_prim.IsValid()),
             "active": bool(hesai_prim and hesai_prim.IsValid() and hesai_prim.IsActive()),
             "prims": 0, "rigid_body_prims": [], "collision_prims": [], "mass_api_prims": []}
    if hesai["exists"]:
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not (path == hesai_root or path.startswith(hesai_root + "/")):
                continue
            hesai["prims"] += 1
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                hesai["rigid_body_prims"].append(path)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                hesai["collision_prims"].append({"path": path, "collisionEnabled": enabled})
            if prim.HasAPI(UsdPhysics.MassAPI):
                api = UsdPhysics.MassAPI(prim)
                hesai["mass_api_prims"].append({"path": path, "mass": api.GetMassAttr().Get(),
                                                "density": api.GetDensityAttr().Get()})
    hesai_rb_mass = sum(m for p, m in zip(body_paths, body_masses)
                        if p == hesai_root or p.startswith(hesai_root + "/"))
    hesai["rigid_body_mass_kg"] = hesai_rb_mass

    body_path = str(Sdf.Path(sim_config.DRONE_ROOT).AppendChild("body"))
    body_prim = stage.GetPrimAtPath(body_path)
    body_mass_api = None
    if body_prim and body_prim.IsValid() and body_prim.HasAPI(UsdPhysics.MassAPI):
        api = UsdPhysics.MassAPI(body_prim)
        body_mass_api = {"mass": api.GetMassAttr().Get(), "density": api.GetDensityAttr().Get()}
    body_has_authored_mass = bool(body_mass_api and body_mass_api.get("mass") not in (None, 0.0))
    enabled_colliders = [c for c in hesai["collision_prims"] if c["collisionEnabled"] is not False]
    if hesai["rigid_body_prims"] or hesai_rb_mass > 0:
        verdict = f"HESAI adds {hesai_rb_mass:.6f} kg as its own rigid body/bodies"
    elif enabled_colliders and not body_has_authored_mass:
        verdict = ("HESAI is not a rigid body, but has enabled colliders and body has no authored mass: "
                   "colliders may add density-based mass to body (FLAG)")
    elif enabled_colliders:
        verdict = "HESAI is not a rigid body; its colliders do not change body mass (body mass is authored)"
    else:
        verdict = "HESAI adds no rigid-body mass (not a rigid body, no enabled colliders)"
    hesai["verdict"] = verdict
    say(f"HESAI root {hesai_root}: exists={hesai['exists']} active={hesai['active']} prims={hesai['prims']}")
    say(f"    RigidBodyAPI prims: {hesai['rigid_body_prims'] or 'none'}")
    say(f"    CollisionAPI prims: {len(hesai['collision_prims'])} (enabled {len(enabled_colliders)})")
    for c in hesai["collision_prims"][:10]:
        say(f"        {c}")
    say(f"    MassAPI prims: {hesai['mass_api_prims'] or 'none'}")
    say(f"    hesai_rigid_body_mass_kg = {hesai_rb_mass:.6f}")
    say(f"{body_path} authored MassAPI: {body_mass_api}")
    say(f"HESAI verdict: {verdict}")

    snapshot = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "M1.5 Isaac environment smoke test (no training)",
        "source_commit": manifest["source_commit"],
        "manifest_check": check_line,
        "python": platform.python_version(),
        # From the modules actually imported, NOT importlib.metadata: after SimulationApp starts,
        # metadata.version("torch") reports a stale 2.11.0+cu128 while the imported torch is 2.14.1+cpu.
        "packages": imported_packages(),
        "scene": str(scene_path.relative_to(REPO_DIR).as_posix()),
        "action_space": str(env.action_space),
        "observation_space": str(env.observation_space),
        "decimation": core.decimation,
        "physics_dt": sim.dt,
        "rl_rate_hz": path_sac_config.RL_RATE_HZ,
        "residual_limits": path_sac_config.RESIDUAL_LIMITS,
        "total_mass_kg": total_mass,
        "vehicle_body_masses_kg": dict(zip(body_paths, body_masses)),
        "hover_thrust_n": float(core.hover_thrust),
        "hesai": hesai,
        "body_authored_mass_api": body_mass_api,
    }

    # ---------- 4. Reset ----------
    env.set_training_step(0)
    obs, info = env.reset(seed=123)
    say(f"reset(seed=123): obs shape={obs.shape} dtype={obs.dtype} finite={bool(np.all(np.isfinite(obs)))} "
        f"range=[{obs.min():.3f}, {obs.max():.3f}]")
    say(f"    info: {json.dumps(info, default=jsonable)}")
    expect(obs.shape == (128,), "obs shape (128,)")
    expect(obs.dtype == np.float32, "obs dtype float32")
    expect(bool(np.all(np.isfinite(obs))), "reset obs finite")

    # ---------- 5. Zero residual (PID only) ----------
    say(f"zero action for up to {ZERO_ACTION_DECISIONS} decisions:")
    zero = np.zeros(4, np.float32)
    sat_total = phys_total = 0
    steps_done = 0
    last = None
    for i in range(ZERO_ACTION_DECISIONS):
        obs, reward, terminated, truncated, info = env.step(zero)
        steps_done += 1
        sat_total += info["saturated_steps"]
        phys_total += info["physics_steps"]
        say(f"  {i:2d} reward={reward:+.5f} term={terminated} trunc={truncated} fail={info['failure_reason']} "
            f"pos_err={info['position_error']:.4f} sat={info['saturated_steps']}/{info['physics_steps']}")
        expect(bool(np.all(np.isfinite(obs))) and np.isfinite(reward), f"step {i} obs/reward finite")
        last = (terminated, truncated, info["failure_reason"])
        if terminated or truncated:
            break
    say(f"zero-action summary: decisions={steps_done} saturated physics steps={sat_total}/{phys_total} last={last}")
    snapshot["zero_action_smoke"] = {"decisions": steps_done, "saturated_steps": sat_total,
                                     "physics_steps": phys_total, "ended": last}

    # ---------- 6. Curriculum ----------
    stages = []
    for k, step in enumerate(CURRICULUM_STEPS):
        env.set_training_step(step)
        _, info = env.reset(seed=1000 + k)
        stages.append(info["stage"])
    say(f"curriculum stages {CURRICULUM_STEPS} -> {stages}")
    expect(tuple(stages) == EXPECTED_STAGES, f"stages == {EXPECTED_STAGES}")
    snapshot["curriculum_check"] = dict(zip(map(str, CURRICULUM_STEPS), stages))

    # ---------- 7. Determinism ----------
    env.set_training_step(0)
    obs_a, info_a = env.reset(seed=7)
    obs_b, info_b = env.reset(seed=7)
    same = {k: (info_a[k] == info_b[k]) for k in ("wind_seed", "route_type", "route_seconds")}
    max_diff = float(np.max(np.abs(obs_a - obs_b)))
    say(f"determinism reset(seed=7) x2: {same}  wind_mean equal={bool(np.array_equal(info_a['wind_mean'], info_b['wind_mean']))} "
        f"obs max|diff|={max_diff:.3e}")
    expect(all(same.values()), "seeded reset reproduces wind_seed/route_type/route_seconds")
    expect(max_diff == 0.0, "seeded reset reproduces initial obs exactly")
    snapshot["determinism_obs_max_abs_diff"] = max_diff

except Exception as exc:  # report, then still close cleanly
    import traceback
    traceback.print_exc()
    expect(False, f"exception: {type(exc).__name__}: {exc}")
finally:
    runs_after = (REPO_DIR / "runs").exists(), (RESIDUAL_DIR / "runs").exists()
    expect(runs_after == runs_before, f"runs/ folders unchanged (before {runs_before}, after {runs_after})")
    verdict_line = "RESULT: PASS" if not failures else f"RESULT: FAIL ({len(failures)}): {failures}"
    snapshot["smoke_result"] = verdict_line
    RESULTS_DIR.mkdir(exist_ok=True)
    if snapshot:
        SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, default=jsonable) + "\n", encoding="utf-8")
        say(f"snapshot written: {rel(SNAPSHOT_PATH)}")
    say(verdict_line)
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    if env is not None:
        env.close()
    app.close()
