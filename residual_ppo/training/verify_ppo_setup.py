"""M3 PPO setup verification (V1-V8, no gradient update) and the M3 timing probe. Run with Isaac's python.bat:

    # Directory: <repo root>
    & "<Isaac Sim>\\python.bat" residual_ppo\\training\\verify_ppo_setup.py --mode verify *> residual_ppo\\results\\m3\\ppo_setup_check.log
    & "<Isaac Sim>\\python.bat" residual_ppo\\training\\verify_ppo_setup.py --mode probe  *> residual_ppo\\results\\m3\\timing_probe.log

verify: V1 preflight + config checks; V2 torch import order and origins; V3 spaces; V4 parameter counts;
        V5 initial policy mean; V6 clip fraction of the 2,048 sampled actions; V7 one rollout with NO update;
        V8 rollout contents (finite, terminated/truncated, bootstrapping, curriculum stage, seed guard, speed).
        Writes residual_ppo/results/m3/ppo_setup_check.{json,txt}.
probe:  the M3 timing probe (id experiment_config.TIMING_PROBE_ID): 4 rollouts = 8,192 decisions WITH updates, to
        measure t_update. The model is discarded (never saved to a run folder) and never evaluated. Writes
        residual_ppo/results/m3/timing_probe.{json,txt} and timing_probe/progress.csv.
Success = the log ends with RESULT: DONE and the outputs exist (never the exit code).
"""
import argparse
import json
import math
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESIDUAL_DIR))
sys.path.insert(1, str(RESIDUAL_DIR / "evaluation"))
sys.path.insert(2, str(RESIDUAL_DIR / "training"))

import numpy as np  # noqa: E402

import run_common  # noqa: E402

M21_DECISIONS_PER_S = 49.1      # M2.1 speed.json, training_range, zero action (Confirmed)
EXPECTED_ACTOR = 99_848         # 33,024 + 65,792 + 1,028 + 4 (log_std)
EXPECTED_CRITIC = 99_073        # 33,024 + 65,792 + 257
MU_FLAG = 0.1

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("verify", "probe"), required=True)
args = parser.parse_args()
TAG = "M3.verify" if args.mode == "verify" else "M3.probe"
lines = []
checks = {}


def say(msg=""):
    line = f"[{TAG}] {msg}"
    print(line, flush=True)
    lines.append(line)


def verdict(key, ok, detail, flag=False):
    status = "PASS" if ok and not flag else ("FLAG" if ok else "FAIL")
    checks[key] = {"status": status, "detail": detail}
    say(f"{key}: {status} - {detail}")
    return ok


def rel(path):
    return Path(path).resolve().relative_to(RESIDUAL_DIR.parent.resolve()).as_posix()


# ---------- V1: preflight before Isaac ----------
check_line = run_common.preflight(say)
from config import experiment_config, ppo_config  # noqa: E402  (config asserts run here: gamma, n_steps % batch)

import controllers  # noqa: E402
import path_sac_config  # noqa: E402

v1 = {"checks": check_line}
cfg_files = {m.__name__: m.__file__ for m in (ppo_config, experiment_config)}
v1["config_inside_residual_ppo"] = all(run_common.in_residual(f) for f in cfg_files.values())
v1["gamma"] = [ppo_config.PPO["gamma"], path_sac_config.SAC["gamma"]]
v1["sweep_variant"] = path_sac_config.sweep_variant()
contract = ppo_config.contract()
roundtrip = json.loads(json.dumps(contract))
v1["contract_policy_equals_policy_contract"] = roundtrip["policy"] == json.loads(json.dumps(path_sac_config.policy_contract()))
v1["contract_file_name_matches_adapter"] = experiment_config.CONTRACT_FILE == controllers.PPO_CONTRACT_FILE
forbidden = experiment_config.forbidden_reset_seeds()
v1["forbidden_reset_seeds"] = len(forbidden)
v1["rollouts"] = ppo_config.rollouts()
v1["planned_decisions"] = ppo_config.planned_decisions()
v1["max_gradient_steps"] = ppo_config.max_gradient_steps()
ck = experiment_config.checkpoint_steps()
v1["checkpoints"] = [len(ck), ck[0], ck[-1]]
v1["sac_checkpoint_steps_included"] = all(s in ck for s in experiment_config.SAC_CHECKPOINT_STEPS)
v1["decisions_trained_on_150k"] = experiment_config.decisions_trained_on(150_000)
v1_ok = (v1["config_inside_residual_ppo"] and v1["gamma"][0] == v1["gamma"][1] and v1["sweep_variant"] is None
         and v1["contract_policy_equals_policy_contract"] and v1["contract_file_name_matches_adapter"]
         and v1["forbidden_reset_seeds"] == 40 and v1["rollouts"] == 733 and v1["planned_decisions"] == 1_501_184
         and v1["max_gradient_steps"] == 58_640 and v1["checkpoints"] == [30, 50_000, 1_500_000]
         and v1["sac_checkpoint_steps_included"])

# ---------- V2: torch + torch._dynamo BEFORE SimulationApp ----------
import ppo_setup  # noqa: E402

torch = ppo_setup.import_torch_first()
t_app0 = time.perf_counter()
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
t_app = time.perf_counter() - t_app0
venv = None
result = {}
try:
    verdict("V1 preflight", v1_ok, f"{check_line}; config files inside residual_ppo; gamma {v1['gamma'][0]} == SAC "
            f"{v1['gamma'][1]}; sweep variant None; contract policy == policy_contract(); contract file name == "
            f"adapter's; {v1['forbidden_reset_seeds']} forbidden reset seeds; {v1['rollouts']} rollouts = "
            f"{v1['planned_decisions']:,} decisions; <= {v1['max_gradient_steps']:,} gradient steps; "
            f"{v1['checkpoints'][0]} checkpoints incl. 150k/400k/800k/1.5M")

    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    t_env0 = time.perf_counter()
    env, guard, venv = ppo_setup.build_training_env(app, forbidden)
    t_env = time.perf_counter() - t_env0
    env.set_training_step(0)
    assert env.core.decimation == 12
    model = ppo_setup.build_model(venv)
    vers = ppo_setup.versions()
    origins = ppo_setup.torch_origin_report()
    say(f"startup: SimulationApp {t_app:.1f} s, env build {t_env:.1f} s; versions {vers}")
    optimizer = model.policy.optimizer
    verdict("V2 import order", type(optimizer).__name__ == "Adam" and origins["outside_package_dir"] == 0
            and origins["modules_from_ml_archive_pip_prebundle"] == 0
            and vers["torch"] == "2.14.1+cpu" and not vers["torch_cuda_available"] and str(model.device) == "cpu",
            f"PPO created after SimulationApp + env with no ImportError; optimizer {type(optimizer).__name__}; "
            f"torch {vers['torch']} from {origins['torch_package_dir']}; {origins['torch_modules_with_file']} torch "
            f"modules with a file, {origins['outside_package_dir']} outside it {origins['outside_examples']} "
            f"(placeholder __file__ skipped: {origins['synthetic_file_attributes']}); modules from the ml_archive "
            f"pip_prebundle: {origins['modules_from_ml_archive_pip_prebundle']} {origins['prebundle_examples']}; "
            f"any module path containing ml_archive: {origins['modules_from_ml_archive']} "
            f"{origins['ml_archive_examples']} (the Isaac extension package itself); model device {model.device}")

    # ---------- V3: spaces ----------
    import gymnasium as gym

    obs_box = gym.spaces.Box(-10.0, 10.0, (128,), dtype=np.float32)
    act_box = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
    v3_ok = (env.observation_space == obs_box and env.action_space == act_box
             and model.observation_space == env.observation_space and model.action_space == env.action_space
             and venv.observation_space == env.observation_space and venv.action_space == env.action_space)
    verdict("V3 spaces", v3_ok, f"env obs {env.observation_space}, act {env.action_space}; model and VecEnv equal")

    # ---------- V4: parameter counts and architecture ----------
    counts = ppo_setup.parameter_counts(model)
    pol = model.policy
    acts = sorted({type(m).__name__ for m in list(pol.mlp_extractor.policy_net) + list(pol.mlp_extractor.value_net)
                   if not isinstance(m, torch.nn.Linear)})
    log_std = pol.log_std.detach().cpu().numpy().tolist()
    v4_ok = (counts["actor"] == EXPECTED_ACTOR and counts["critic"] == EXPECTED_CRITIC
             and counts["features_extractor"] == 0 and counts["total"] == EXPECTED_ACTOR + EXPECTED_CRITIC
             and acts == ["ReLU"] and log_std == [-1.0] * 4)
    verdict("V4 parameters", v4_ok, f"actor {counts['actor']:,} (expected {EXPECTED_ACTOR:,}), critic "
            f"{counts['critic']:,} (expected {EXPECTED_CRITIC:,}), features extractor {counts['features_extractor']}, "
            f"total {counts['total']:,}; hidden activations {acts}; log_std {log_std}; "
            f"pi {pol.mlp_extractor.policy_net}; vf {pol.mlp_extractor.value_net}")

    if args.mode == "verify":
        # ---------- V7 setup: one rollout, no gradient update ----------
        grad = ppo_setup.count_gradient_steps(optimizer)
        hash0 = ppo_setup.state_dict_sha256(model)
        opt_state0 = len(optimizer.state)
        gamma = model.gamma

        class Recorder(BaseCallback):
            """Raw rewards BEFORE SB3's timeout bootstrap, plus done flags, stages and terminal observations."""

            def __init__(self):
                super().__init__()
                self.raw, self.trunc, self.term, self.stages = [], [], [], []
                self.terminal_obs, self.reasons, self.completed, self.episodes = {}, [], 0, []

            def _on_step(self):
                info = self.locals["infos"][0]
                done = bool(self.locals["dones"][0])
                self.raw.append(float(np.asarray(self.locals["rewards"])[0]))
                self.stages.append(info["stage"])
                tr = done and bool(info.get("TimeLimit.truncated", False))
                self.trunc.append(tr)
                self.term.append(done and not tr)
                if done:
                    self.reasons.append(info["failure_reason"])
                    self.completed += int(bool(info["completed"]))
                    if info.get("terminal_observation") is not None:
                        self.terminal_obs[len(self.raw) - 1] = np.array(info["terminal_observation"], copy=True)
                    if "episode" in info:
                        self.episodes.append({"r": float(info["episode"]["r"]), "l": int(info["episode"]["l"])})
                return True

        rec = Recorder()
        cb = CallbackList([ppo_setup.make_curriculum_callback(env), rec])
        _, cb = model._setup_learn(model.n_steps, cb, True, "m3_verify", False)
        # Fresh dicts: at module level locals() IS globals(), and SB3's update_locals() would write
        # collect_rollouts' locals (env, actions, ...) into this script's globals.
        cb.on_training_start({}, {})
        t0 = time.perf_counter()
        ok_collect = model.collect_rollouts(model.env, cb, model.rollout_buffer, n_rollout_steps=model.n_steps)
        t_collect = time.perf_counter() - t0
        hash1 = ppo_setup.state_dict_sha256(model)
        v7_ok = (ok_collect and hash1 == hash0 and grad["steps"] == 0 and len(optimizer.state) == 0 == opt_state0
                 and model._n_updates == 0 and model.num_timesteps == model.n_steps)
        verdict("V7 no update", v7_ok, f"collect_rollouts({model.n_steps}) returned {ok_collect}; num_timesteps "
                f"{model.num_timesteps}; state_dict sha256 before {hash0[:12]} after {hash1[:12]} "
                f"({'identical' if hash1 == hash0 else 'CHANGED'}); optimizer steps {grad['steps']}; Adam state "
                f"entries {len(optimizer.state)}; SB3 _n_updates {model._n_updates}")

        buf = model.rollout_buffer
        obs = buf.observations.reshape(model.n_steps, -1)
        actions = buf.actions.reshape(model.n_steps, -1)

        # ---------- V5: initial policy mean ----------
        with torch.no_grad():
            dist = model.policy.get_distribution(torch.as_tensor(obs))
            mu = dist.distribution.mean.cpu().numpy()
            std = dist.distribution.stddev.cpu().numpy()
        mu_abs_mean = np.abs(mu).mean(axis=0)
        mu_abs_max = float(np.abs(mu).max())
        verdict("V5 initial mean", True, f"mean |mu| per component {np.round(mu_abs_mean, 5).tolist()}, max |mu| "
                f"{mu_abs_max:.5f} (flag if > {MU_FLAG}); std {float(std.mean()):.4f} (exp(-1) = 0.3679); "
                f"obs |x| mean {float(np.abs(obs).mean()):.3f}, max {float(np.abs(obs).max()):.3f}",
                flag=mu_abs_max > MU_FLAG)

        # ---------- V6: clip fraction of the 2,048 sampled actions ----------
        exp_comp, exp_any = ppo_config.expected_initial_clip()
        outside = np.abs(actions) > 1.0
        comp_frac = float(outside.mean())
        per_dim = outside.mean(axis=0)
        any_frac = float(outside.any(axis=1).mean())
        verdict("V6 clip fraction", True, f"{comp_frac:.4%} of components (expected {exp_comp:.4%}), per component "
                f"{np.round(per_dim * 100, 3).tolist()} %, {any_frac:.3%} of actions (expected {exp_any:.3%}); "
                f"flag if > 2x expected; |a| max {float(np.abs(actions).max()):.3f}",
                flag=comp_frac > 2 * exp_comp or any_frac > 2 * exp_any)

        # ---------- V8: rollout contents ----------
        arrays = {k: getattr(buf, k) for k in ("rewards", "values", "returns", "advantages", "log_probs")}
        finite = {k: bool(np.isfinite(v).all()) for k, v in arrays.items()}
        raw = np.array(rec.raw, dtype=np.float32)
        diff = buf.rewards.reshape(-1) - raw
        trunc_idx = [i for i, t in enumerate(rec.trunc) if t]
        boot_err = []
        with torch.no_grad():
            for i in trunc_idx:
                v = model.policy.predict_values(model.policy.obs_to_tensor(rec.terminal_obs[i])[0])[0].item()
                boot_err.append(abs(float(diff[i]) - gamma * v))
        non_trunc_changed = int(sum(1 for i in range(len(diff)) if i not in set(trunc_idx) and diff[i] != 0.0))
        all_have_terminal_obs = all(i in rec.terminal_obs for i in trunc_idx)
        boot_ok = all_have_terminal_obs and all(e < 1e-4 for e in boot_err) and non_trunc_changed == 0
        stages = sorted(set(rec.stages))
        seeds_ok = guard.seeded_resets == [0] and not (set(guard.seeded_resets) & forbidden)
        dps = model.n_steps / t_collect
        reasons = {}
        for r in rec.reasons:
            reasons[str(r)] = reasons.get(str(r), 0) + 1
        v8_ok = all(finite.values()) and boot_ok and stages == ["short_slow"] and seeds_ok and len(rec.raw) == model.n_steps
        verdict("V8 rollout contents", v8_ok,
                f"finite {finite}; {len(rec.episodes)} episodes ended ({sum(rec.term)} terminated, {len(trunc_idx)} "
                f"truncated; end reasons {reasons}; completed {rec.completed}); truncations bootstrapped "
                f"{len(trunc_idx)}/{len(trunc_idx)} with |r_buf - r_raw - gamma*V(terminal)| max "
                f"{max(boot_err) if boot_err else 0:.2e}, non-truncated rewards changed {non_trunc_changed}; "
                f"stages {stages}; resets: seeded {guard.seeded_resets}, unseeded {guard.unseeded_resets} "
                f"(no forbidden seed, no options); {dps:.1f} decisions/s incl. policy forward (M2.1 zero action: "
                f"{M21_DECISIONS_PER_S})")
        ep_len = [e["l"] for e in rec.episodes]
        result_mode = {
            "rollout": {"decisions": model.n_steps, "seconds": t_collect, "decisions_per_s": dps,
                        "episodes_ended": len(rec.episodes), "terminated": int(sum(rec.term)),
                        "truncated": len(trunc_idx), "end_reasons": reasons, "completed": rec.completed,
                        "episode_length_mean": float(np.mean(ep_len)) if ep_len else None,
                        "episode_return_mean": float(np.mean([e["r"] for e in rec.episodes])) if ep_len else None,
                        "bootstrap_abs_err_max": max(boot_err) if boot_err else 0.0,
                        "non_truncated_rewards_changed": non_trunc_changed, "stages": stages,
                        "seeded_resets": guard.seeded_resets, "unseeded_resets": guard.unseeded_resets,
                        "reward_raw_mean": float(raw.mean()), "reward_raw_min": float(raw.min()),
                        "value_mean": float(buf.values.mean()), "return_mean": float(buf.returns.mean()),
                        "advantage_std": float(buf.advantages.std())},
            "initial_policy": {"mu_abs_mean": mu_abs_mean.tolist(), "mu_abs_max": mu_abs_max,
                               "std_mean": float(std.mean())},
            "clip": {"component_fraction": comp_frac, "per_component": per_dim.tolist(), "action_fraction": any_frac,
                     "expected_component": exp_comp, "expected_action": exp_any},
            "no_update": {"state_dict_sha256_before": hash0, "after": hash1, "optimizer_steps": grad["steps"],
                          "adam_state_entries": len(optimizer.state), "sb3_n_updates": model._n_updates},
        }
    else:
        # ---------- timing probe: 4 rollouts WITH updates; model discarded ----------
        from stable_baselines3.common.logger import configure

        probe_dir = RESIDUAL_DIR / experiment_config.TIMING_PROBE_RESULTS
        probe_dir.mkdir(parents=True, exist_ok=True)
        model.set_logger(configure(str(probe_dir), ["csv"]))
        grad = ppo_setup.count_gradient_steps(optimizer)

        class RolloutTimer(BaseCallback):
            """Collect time (rollout start -> end) and update time (rollout end -> next start / training end)."""

            def __init__(self):
                super().__init__()
                self.rows, self.t_start, self.t_end, self.grad0 = [], None, None, 0

            def _on_rollout_start(self):
                self._close_update()
                self.t_start = time.perf_counter()

            def _on_rollout_end(self):
                now = time.perf_counter()
                a = self.model.rollout_buffer.actions.reshape(self.model.n_steps, -1)
                out = np.abs(a) > 1.0
                self.rows.append({"rollout": len(self.rows) + 1, "collect_s": now - self.t_start,
                                  "action_clip_component": float(out.mean()),
                                  "action_clip_action": float(out.any(axis=1).mean()),
                                  "num_timesteps": int(self.model.num_timesteps)})
                self.t_end, self.grad0 = now, grad["steps"]

            def _on_training_end(self):
                self._close_update()

            def _close_update(self):
                if self.t_end is None:
                    return
                row = self.rows[-1]
                row["update_s"] = time.perf_counter() - self.t_end
                row["gradient_steps"] = grad["steps"] - self.grad0
                vals = dict(self.model.logger.name_to_value)
                for k in ("value_loss", "clip_fraction", "approx_kl", "policy_gradient_loss", "entropy_loss",
                          "explained_variance", "std", "n_updates", "loss"):
                    if f"train/{k}" in vals:
                        row[k] = float(vals[f"train/{k}"])
                self.t_end = None

            def _on_step(self):
                return True

        timer = RolloutTimer()
        t0 = time.perf_counter()
        model.learn(total_timesteps=experiment_config.TIMING_PROBE_DECISIONS,
                    callback=CallbackList([ppo_setup.make_curriculum_callback(env), timer]))
        t_learn = time.perf_counter() - t0
        with tempfile.TemporaryDirectory(dir=RESIDUAL_DIR / "scratch") as tmp:  # gitignored; deleted on exit
            ts = time.perf_counter()
            model.save(Path(tmp) / "probe_model")
            t_save = time.perf_counter() - ts
        rows = timer.rows
        for r in rows:
            say(f"rollout {r['rollout']}: collect {r['collect_s']:.1f} s ({2048 / r['collect_s']:.1f} decisions/s), "
                f"update {r.get('update_s', float('nan')):.2f} s, {r.get('gradient_steps')} gradient steps, "
                f"value_loss {r.get('value_loss', float('nan')):.4g}, PPO clip_fraction "
                f"{r.get('clip_fraction', float('nan')):.4f}, approx_kl {r.get('approx_kl', float('nan')):.4f}, "
                f"std {r.get('std', float('nan')):.4f}, explained_var {r.get('explained_variance', float('nan')):.3f}, "
                f"action clip {r['action_clip_component']:.3%} of components / {r['action_clip_action']:.2%} of actions")
        collect = float(np.mean([r["collect_s"] for r in rows]))
        upd = [r["update_s"] for r in rows]
        steps = [r["gradient_steps"] for r in rows]
        per_step = sum(upd) / sum(steps)
        t_update_full = per_step * ppo_config.minibatches_per_rollout()
        n_roll = ppo_config.rollouts()
        n_ck = len(experiment_config.checkpoint_steps())
        env_h = n_roll * collect / 3600
        upd_h = n_roll * t_update_full / 3600
        total_h = env_h + upd_h + (n_ck + 1) * t_save / 3600 + (t_app + t_env) / 3600
        m21_h = ppo_config.planned_decisions() / M21_DECISIONS_PER_S / 3600
        say(f"t_update: mean {np.mean(upd):.2f} s per rollout over {len(upd)} updates "
            f"({sum(steps)} gradient steps, {per_step * 1000:.1f} ms each); full 80-step update {t_update_full:.2f} s")
        say(f"ESTIMATE 1.5M ({n_roll} rollouts = {ppo_config.planned_decisions():,} decisions): collect "
            f"{env_h:.2f} h (probe rate {2048 / collect:.1f} decisions/s; at M2.1's {M21_DECISIONS_PER_S}/s: "
            f"{m21_h:.2f} h) + updates <= {upd_h:.2f} h + {n_ck + 1} saves ({t_save:.2f} s each) + startup "
            f"{t_app + t_env:.0f} s = {total_h:.2f} h")
        probe_ok = (model.num_timesteps == experiment_config.TIMING_PROBE_DECISIONS and len(rows) == 4
                    and all("update_s" in r for r in rows) and guard.seeded_resets == [0])
        verdict("probe", probe_ok, f"{model.num_timesteps} decisions in {len(rows)} rollouts with updates; "
                f"learn() {t_learn:.1f} s; seeded resets {guard.seeded_resets}; model discarded")
        result_mode = {
            "probe_id": experiment_config.TIMING_PROBE_ID, "model": "discarded (never saved to a run folder)",
            "decisions": int(model.num_timesteps), "learn_s": t_learn, "rollouts": rows,
            "collect_s_mean": collect, "update_s_mean": float(np.mean(upd)), "gradient_steps": steps,
            "gradient_step_s": per_step, "t_update_full_80_steps_s": t_update_full, "save_s": t_save,
            "sb3_n_updates_epochs": int(model._n_updates), "episodes": len(model.ep_info_buffer),
            "estimate_1p5m": {"rollouts": n_roll, "decisions": ppo_config.planned_decisions(),
                              "collect_h": env_h, "collect_h_at_m21_rate": m21_h, "updates_h_upper": upd_h,
                              "saves": n_ck + 1, "startup_s": t_app + t_env, "total_h": total_h,
                              "basis": "probe collect rate (short_slow stage, policy forward included) x 733; "
                                       "updates at 80 gradient steps each (no early stop)"},
            "progress_csv": rel(probe_dir / "progress.csv"),
        }

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": ("M3 PPO setup verification V1-V8 (one rollout, no gradient update)" if args.mode == "verify"
                    else f"M3 timing probe {experiment_config.TIMING_PROBE_ID} (model discarded, never evaluated)"),
        "checks": checks,
        "v1": v1,
        "ppo": ppo_config.PPO, "policy_kwargs": ppo_config.policy_kwargs_record(),
        "parameter_counts": counts,
        "versions": vers, "torch_origins": origins, "model_device": str(model.device),
        "startup_s": {"simulation_app": t_app, "env_build": t_env},
        "total_mass_kg": float(env.sim.total_mass), "hesai": run_common.hesai_report(env.sim),
        "isaac_version": run_common.isaac_version(), "hardware": run_common.hardware(), "git": run_common.git_info(),
        args.mode: result_mode,
    }
except Exception:
    import traceback

    traceback.print_exc()
    say("RESULT: FAIL (exception)")
    raise
finally:
    out = RESIDUAL_DIR / (experiment_config.VERIFY_RESULTS if args.mode == "verify"
                          else experiment_config.TIMING_PROBE_RESULTS)
    out.parent.mkdir(parents=True, exist_ok=True)
    if result:
        out.with_suffix(".json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        failed = [k for k, v in checks.items() if v["status"] == "FAIL"]
        flagged = [k for k, v in checks.items() if v["status"] == "FLAG"]
        say(f"summary: {len(checks) - len(failed) - len(flagged)} PASS, {len(flagged)} FLAG {flagged}, "
            f"{len(failed)} FAIL {failed}")
        say(f"written: {rel(out.with_suffix('.json'))}, {rel(out.with_suffix('.txt'))}")
        say("RESULT: DONE")
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if venv is not None:
        venv.close()
    app.close()
