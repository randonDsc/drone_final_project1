"""M8 unit tests for PPO direct control (no Isaac). Gate 4 is MappingGateTest + ZeroCommandHoldTest.

    # Directory: <repo root>
    & "<python>" -m unittest discover -s residual_ppo\\tests -v

The inner loop is the shared, unmodified PIDController, driven by target substitution (direct_control/direct_env.py).
These tests check, against the shared code itself:
  - the substituted target makes the controller REQUEST the commanded velocity (round trip <= 1e-14 m/s, computed
    with controller.py's own expressions) and receive the commanded yaw angle;
  - the command reaches the controller's OUTPUT (recovered from thrust and torques, below the caps);
  - a zero command is an exact hover equilibrium and brakes any velocity;
  - the integrate call leaves the xy/height integrators at 0 (and documents the wind-up it avoids);
  - DirectCore sends the inner loop's command to the mixer with no residual and observes the inner loop;
  - the direct configs differ from the residual arm only where Claire decided, and pid_ppo refuses direct models.
"""
import json
import math
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path

import numpy as np

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESIDUAL_DIR))
sys.path.insert(1, str(RESIDUAL_DIR / "evaluation"))
sys.path.insert(2, str(RESIDUAL_DIR / "training"))

import path_sac_config  # noqa: E402
import sim_config  # noqa: E402
from controller import PIDController  # noqa: E402
from path_reference import TimedReference  # noqa: E402
from path_sac_policy import CompletionTracker, wrap_angle  # noqa: E402

from direct_control import direct_env as de  # noqa: E402

MASS = 1.618614
GRAVITY_IN_CONTROLLER = 9.81                       # controller.py:127 literal
MAX_TOTAL_THRUST = 4 * 5.84e-6 * 1100.0**2        # controller.py:165-167 default (~28.27 N)
BASE_TARGET = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "mass": MASS, "max_xy_speed": sim_config.MAX_XY_SPEED,
               "max_z_speed": sim_config.MAX_Z_SPEED, "max_tilt_degrees": sim_config.MAX_TILT_DEGREES,
               "max_total_thrust": MAX_TOTAL_THRUST}
GAINS = sim_config.PID
ROUND_TRIP_BOUND = 1e-14                           # gate 4 (Claire): m/s
# Domain of the round-trip bound: |x|, |y| <= 100 m, 0 <= z <= 50 m. Training/eval routes start within +-3 m in x, y
# (EPISODE start_xy), stay in z 1.2-4.8 m, and a flight fails 3 m from the reference, so real positions are far inside.
XY_RANGE, Z_RANGE = 100.0, (0.0, 50.0)


def yaw_quaternion(yaw):
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def state(position, velocity=(0.0, 0.0, 0.0), yaw=0.0, angular_velocity=(0.0, 0.0, 0.0)):
    return {"position": np.asarray(position, dtype=float), "orientation": yaw_quaternion(yaw),
            "linear_velocity": np.asarray(velocity, dtype=float),
            "angular_velocity": np.asarray(angular_velocity, dtype=float)}


def controller_request(target, position):
    """The velocity the shared controller requests for `target`, with controller.py's own expressions BEFORE the
    speed clips: :146 + :152 (height) and :195-196 (horizontal)."""
    p = np.asarray(position, dtype=float).reshape(3)
    xy_error = np.asarray(target["xy"], dtype=float).reshape(2) - p[:2]
    vxy = GAINS["xy"]["kp"] / GAINS["xy"]["kd"] * xy_error
    height_error = float(target["height"]) - p[2]
    vz = GAINS["height"]["kp"] / GAINS["height"]["kd"] * height_error
    return np.array([vxy[0], vxy[1], vz])


def new_pid():
    return de.VelocityPID(BASE_TARGET, MASS)


def below_caps(v_cmd, vel, margin=0.98):
    """Level state, zero integrals: True if thrust is not clipped and the horizontal demand is below the cap
    min(1.5, a_vert tan(max_tilt)) (controller.py:154-171, 201-216), so outputs can be inverted exactly."""
    vertical_force = MASS * GRAVITY_IN_CONTROLLER + GAINS["height"]["kd"] * (v_cmd[2] - vel[2])
    if not 0.0 < vertical_force < MAX_TOTAL_THRUST:
        return False
    a_vert = max(vertical_force / MASS, 0.5 * GRAVITY_IN_CONTROLLER)
    cap = min(1.5, a_vert * math.tan(math.radians(BASE_TARGET["max_tilt_degrees"])))
    return float(np.linalg.norm(GAINS["xy"]["kd"] * (np.asarray(v_cmd[:2]) - vel[:2]))) < margin * cap


class ActionMappingTest(unittest.TestCase):
    def test_limits_and_clipping(self):
        v, yaw = de.action_to_command(np.array([1, -1, 1, -1], dtype=np.float32))
        np.testing.assert_array_equal(v, [3.0, -3.0, 1.5])
        self.assertEqual(yaw, -math.pi)
        v, yaw = de.action_to_command(np.array([5, -7, 0.5, 2], dtype=np.float32))
        np.testing.assert_array_equal(v, [3.0, -3.0, 0.75])
        self.assertEqual(yaw, math.pi)
        v, yaw = de.action_to_command(np.zeros(4, dtype=np.float32))
        np.testing.assert_array_equal(v, np.zeros(3))
        self.assertEqual(yaw, 0.0)

    def test_commanded_speeds_never_hit_the_shared_speed_clips(self):
        # |v_xy| <= 3 sqrt 2 = 4.24 < max_xy_speed and |vz| <= 1.5 < max_z_speed: the clips at controller.py:152-153,
        # 197-200 never alter the command.
        self.assertLess(math.hypot(3.0, 3.0), BASE_TARGET["max_xy_speed"])
        self.assertLess(1.5, BASE_TARGET["max_z_speed"])


class MappingGateTest(unittest.TestCase):
    """GATE 4a: the substitution delivers the commanded velocity to the controller, round trip <= 1e-14 m/s."""

    def test_round_trip_class_path(self):
        rng = np.random.default_rng(0)
        pid = new_pid()
        worst = 0.0
        for _ in range(20_000):
            p = np.array([*rng.uniform(-XY_RANGE, XY_RANGE, 2), rng.uniform(*Z_RANGE)])
            v, yaw = de.action_to_command(rng.uniform(-1.0, 1.0, 4).astype(np.float32))
            pid.set_command(v, yaw)
            target = pid._target(pid.substituted_target(p))
            worst = max(worst, float(np.max(np.abs(controller_request(target, p) - v))))
            self.assertEqual(target["yaw"], yaw)
        print(f"\n[gate 4a] class path, 20,000 cases: max |requested - commanded| = {worst:.3e} m/s")
        self.assertLessEqual(worst, ROUND_TRIP_BOUND)

    def test_round_trip_vectorized_million_and_edges(self):
        rng = np.random.default_rng(1)
        n = 1_000_000
        p = np.column_stack([rng.uniform(-XY_RANGE, XY_RANGE, (n, 2)), rng.uniform(*Z_RANGE, n)])
        a = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
        edges = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        a[:125] = np.array(np.meshgrid(edges, edges, edges)).reshape(3, -1).T
        v = a.astype(float) * de.VELOCITY_LIMITS
        lead = new_pid().velocity_lead
        target = p + lead * v                                   # same expression as substituted_target
        xy = GAINS["xy"]["kp"] / GAINS["xy"]["kd"] * (target[:, :2] - p[:, :2])
        z = GAINS["height"]["kp"] / GAINS["height"]["kd"] * (target[:, 2] - p[:, 2])
        err = np.abs(np.column_stack([xy, z]) - v)
        print(f"\n[gate 4a] vectorized, {n:,} cases: max error {err.max():.3e} m/s; exact in "
              f"{np.mean(err[:, :2] == 0):.1%} (xy) / {np.mean(err[:, 2] == 0):.1%} (z) of components")
        self.assertLessEqual(float(err.max()), ROUND_TRIP_BOUND)
        # the class path computes exactly the same numbers
        pid = new_pid()
        for i in rng.integers(0, n, 200):
            pid.set_command(v[i], 0.0)
            np.testing.assert_array_equal(pid.substituted_target(p[i]), target[i])

    def test_command_reaches_the_controller_output(self):
        """Black box: recover the requested velocity and yaw error from thrust and torques (below the caps)."""
        rng = np.random.default_rng(2)
        kd_xy, kd_z = GAINS["xy"]["kd"], GAINS["height"]["kd"]
        kp_att = np.asarray(GAINS["attitude"]["kp"], dtype=float)
        worst_v = worst_yaw = 0.0
        checked = 0
        while checked < 5_000:
            yaw = rng.uniform(-math.pi, math.pi)
            p = np.array([*rng.uniform(-20, 20, 2), rng.uniform(1, 5)])
            vel = rng.uniform(-1.0, 1.0, 3)
            a = rng.uniform(-1.0, 1.0, 4).astype(np.float32)
            v_cmd, yaw_cmd = de.action_to_command(a)
            if not below_caps(v_cmd, vel):
                continue
            pid = new_pid()
            pid.set_command(v_cmd, yaw_cmd)
            s = state(p, vel, yaw)
            thrust, roll_t, pitch_t, yaw_t = pid.read(s, None, False)
            vertical_force = thrust                                   # level: R22 = 1, thrust not clipped here
            vz_des = (vertical_force - MASS * GRAVITY_IN_CONTROLLER) / kd_z + vel[2]
            a_vert = max(vertical_force / MASS, 0.5 * GRAVITY_IN_CONTROLLER)
            target_pitch = pitch_t / kp_att[1]                        # level: pitch = roll = 0, body rate 0
            target_roll = roll_t / kp_att[0]
            ax_h = a_vert * math.tan(target_pitch)
            ay_h = -math.tan(target_roll) * math.hypot(ax_h, a_vert)
            c, s_ = math.cos(yaw_cmd), math.sin(yaw_cmd)
            a_xy = np.array([c * ax_h - s_ * ay_h, s_ * ax_h + c * ay_h])
            v_des_xy = a_xy / kd_xy + vel[:2]
            worst_v = max(worst_v, float(np.max(np.abs(np.array([*v_des_xy, vz_des]) - v_cmd))))
            yaw_error = yaw_t / kp_att[2]
            worst_yaw = max(worst_yaw, abs(float(wrap_angle(yaw_error - wrap_angle(yaw_cmd - yaw)))))
            checked += 1
        print(f"\n[gate 4a] black box, {checked:,} cases: velocity recovered from thrust/torques within "
              f"{worst_v:.3e} m/s; yaw error within {worst_yaw:.3e} rad")
        self.assertLess(worst_v, 1e-11)
        self.assertLess(worst_yaw, 1e-11)

    def test_yaw_angle_command_takes_the_short_way(self):
        pid = new_pid()
        pid.set_command(np.zeros(3), math.pi * 0.999)
        _, _, _, yaw_t = pid.read(state((0, 0, 2), yaw=-math.pi * 0.999), None, False)
        # raw error 1.998 pi wraps to -0.002 pi: a small negative torque, not a full spin
        self.assertAlmostEqual(yaw_t, GAINS["attitude"]["kp"][2] * (-0.002 * math.pi), places=12)


class ZeroCommandHoldTest(unittest.TestCase):
    """GATE 4b (NumPy half): a zero velocity command holds position. The Isaac half is tools/check_direct_env.py H."""

    def test_exact_hover_equilibrium(self):
        rng = np.random.default_rng(3)
        for _ in range(1_000):
            p = np.array([*rng.uniform(-50, 50, 2), rng.uniform(0.5, 20)])
            pid = new_pid()
            pid.set_command(np.zeros(3), 0.0)
            thrust, roll_t, pitch_t, yaw_t = pid.read(state(p), None, False)
            self.assertEqual(thrust, MASS * GRAVITY_IN_CONTROLLER)   # exactly weight: no vertical acceleration
            self.assertEqual((roll_t, pitch_t, yaw_t), (0.0, 0.0, 0.0))  # no tilt, no turn: no horizontal force

    def test_zero_command_brakes_any_velocity(self):
        rng = np.random.default_rng(4)
        kd_xy, kd_z = GAINS["xy"]["kd"], GAINS["height"]["kd"]
        for _ in range(1_000):
            vel = rng.uniform(-1.0, 1.0, 3)
            pid = new_pid()
            thrust, roll_t, pitch_t, _ = pid.read(state((0, 0, 2), vel), None, False)
            self.assertLess((thrust - MASS * GRAVITY_IN_CONTROLLER) * vel[2], 0.0)   # opposes vertical motion
            a_vert = max(thrust / MASS, 0.5 * GRAVITY_IN_CONTROLLER)
            ax = a_vert * math.tan(pitch_t / GAINS["attitude"]["kp"][1])
            ay = -math.tan(roll_t / GAINS["attitude"]["kp"][0]) * math.hypot(ax, a_vert)
            a_xy = np.array([ax, ay])
            self.assertLess(float(a_xy @ vel[:2]), 0.0)                          # opposes horizontal motion
            expected = -kd_xy * vel[:2]
            if below_caps(np.zeros(3), vel):
                np.testing.assert_allclose(a_xy, expected, atol=1e-11)
            np.testing.assert_allclose(thrust, MASS * GRAVITY_IN_CONTROLLER - kd_z * vel[2], atol=1e-11)


class IntegratorTest(unittest.TestCase):
    def test_integrate_keeps_xy_and_height_at_zero_and_integrates_yaw(self):
        pid = new_pid()
        pid.set_command(np.array([1.0, -0.5, 0.5]), 0.2)
        s = state((1.0, 2.0, 3.0), velocity=(0.3, 0.1, -0.2), yaw=0.0)
        for _ in range(120 * 60):   # 60 s, the episode time limit
            pid.read(s, None, False)
            pid.integrate(s, None, 1.0 / 120.0, False)
        self.assertEqual(pid.pid.height_integral, 0.0)
        np.testing.assert_array_equal(pid.pid.xy_integral, np.zeros(2))
        self.assertEqual(tuple(pid.pid.attitude_integral[:2]), (0.0, 0.0))   # roll/pitch ki = 0
        self.assertGreater(pid.pid.attitude_integral[2], 0.0)                 # yaw error 0.2 rad integrates
        self.assertLessEqual(pid.pid.attitude_integral[2], GAINS["attitude"]["integral_limit"][2])

    def test_documents_the_wind_up_that_integrating_at_the_substituted_target_would_cause(self):
        pid = PIDController()
        p = np.array([0.0, 0.0, 3.0])
        target = {**BASE_TARGET, "xy": p[:2] + np.array([2.0, 0.0]), "height": 3.0}   # 1 m/s command in x
        s = state(p, velocity=(1.0, 0.0, 0.0))
        for _ in range(120 * 4):
            pid.update(s["position"], s["orientation"], s["linear_velocity"], s["angular_velocity"], target,
                       dt=1.0 / 120.0)
        self.assertAlmostEqual(pid.xy_integral[0], GAINS["xy"]["integral_limit"], places=9)   # 1.2 after 4 s


class _FakeSim:
    """Only what PathCore.step needs: dt, the wind's mean, and step_command (records what reaches the mixer)."""

    class _Wind:
        mean_wind = np.zeros(3)

    def __init__(self):
        self.dt = 1.0 / 120.0
        self.wind = self._Wind()
        self.commands = []

    def step_command(self, state, thrust, roll, pitch, yaw, render=True):
        self.commands.append(np.array([thrust, roll, pitch, yaw]))
        return None, False, None, None


def make_core():
    """DirectCore as PathCore.__init__ would build it, without the sim geometry check (needs Isaac)."""
    core = de.DirectCore.__new__(de.DirectCore)
    core.sim = _FakeSim()
    core.decimation = path_sac_config.physics_steps_per_decision()
    core.hover_thrust = MASS * 9.81
    core.pid = new_pid()
    core.frames = deque(maxlen=path_sac_config.HISTORY_LENGTH)
    core.completion = CompletionTracker()
    core.reference = TimedReference([(0.0, 0.0, 2.0), (4.0, 0.0, 2.0)], 1.0, 0.5, 0.5, 0.5, start_hold=1.0,
                                    end_hold=1.0)
    core.clear()
    return core


class DirectCoreTest(unittest.TestCase):
    def test_apply_action_sets_command_and_no_residual(self):
        core = make_core()
        a = np.array([0.5, -0.25, 1.0, -0.5], dtype=np.float32)
        core.apply_action(a)
        np.testing.assert_array_equal(core.correction, np.zeros(4))
        np.testing.assert_array_equal(core.pid.velocity_command, [1.5, -0.75, 1.5])
        self.assertEqual(core.pid.yaw_command, -0.5 * math.pi)
        np.testing.assert_array_equal(core.previous_action, np.zeros(4))
        core.apply_action(np.zeros(4, dtype=np.float32))
        np.testing.assert_array_equal(core.previous_action, a)

    def test_mixer_receives_exactly_the_inner_loop_command(self):
        core = make_core()
        core.apply_action(np.array([0.3, 0.1, -0.2, 0.1], dtype=np.float32))
        s = state((0.05, -0.02, 2.0), velocity=(0.1, 0.0, 0.0))
        expected = np.array(core.pid.read(s, None, False))
        expected_desired = core.pid.desired_attitude(s, expected)   # before step(): it advances the yaw integral
        core.step(s)
        np.testing.assert_array_equal(core.sim.commands[-1], expected)   # final = command + 0
        np.testing.assert_array_equal(core.desired, expected_desired)
        self.assertAlmostEqual(core.desired[2], 0.1 * math.pi, places=6)  # desired yaw = the commanded yaw angle

    def test_observation_features_come_from_the_inner_loop(self):
        core = make_core()
        core.apply_action(np.array([0.4, 0.0, 0.0, 0.0], dtype=np.float32))
        s = state((0.0, 0.0, 2.0))
        obs = core.observe(s).reshape(-1, 32)[-1]
        command = np.array(core.pid.read(s, None, False))
        scales = path_sac_config.OBSERVATION_SCALES
        np.testing.assert_allclose(obs[24], (command[0] - core.hover_thrust) / scales["pid_thrust"], atol=1e-6)
        np.testing.assert_allclose(obs[25:28], command[1:] / scales["pid_torque"], atol=1e-6)
        self.assertGreater(abs(obs[26]), 0.0)    # a forward command tilts: the inner loop's pitch torque is visible

    def test_reset_clears_the_command(self):
        core = make_core()
        core.apply_action(np.ones(4, dtype=np.float32))
        core.pid.reset()
        np.testing.assert_array_equal(core.pid.velocity_command, np.zeros(3))
        self.assertEqual(core.pid.yaw_command, 0.0)


class DirectConfigTest(unittest.TestCase):
    def setUp(self):
        from config import direct_experiment_config as dec
        from config import direct_ppo_config as dpc
        from config import experiment_config as ec
        from config import ppo_config as pc
        self.dpc, self.pc, self.dec, self.ec = dpc, pc, dec, ec

    def test_only_net_arch_and_log_std_differ(self):
        dpc, pc = self.dpc, self.pc
        self.assertEqual(dpc.PPO, pc.PPO)
        self.assertEqual(dpc.PPO["target_kl"], 0.02)
        self.assertEqual(dpc.NET_ARCH, {"pi": [64, 128, 256, 64], "vf": [128, 256, 256, 128]})
        self.assertEqual(dpc.LOG_STD_INIT, math.log(0.4))
        self.assertAlmostEqual(dpc.initial_std(), 0.4, places=15)
        self.assertEqual((dpc.ACTIVATION_FN, dpc.ORTHO_INIT, dpc.TOTAL_DECISIONS, dpc.POLICY),
                         (pc.ACTIVATION_FN, pc.ORTHO_INIT, pc.TOTAL_DECISIONS, pc.POLICY))
        d, r = dpc.training_contract(), pc.training_contract()
        self.assertEqual(d.pop("arm"), "direct")
        dk, rk = d.pop("policy_kwargs"), r.pop("policy_kwargs")
        self.assertEqual(d, r)
        self.assertEqual({k for k in dk if dk[k] != rk[k]}, {"net_arch", "log_std_init"})

    def test_run_ids_and_rules(self):
        dec, ec = self.dec, self.ec
        main = dec.run_settings("ppo_direct_s0_1p5m")
        self.assertEqual((main["smoke"], main["total_decisions"], main["checkpoint_interval"]),
                         (False, 1_500_000, 50_000))
        smoke = dec.run_settings("smoke_direct_s0_3")
        self.assertEqual((smoke["smoke"], smoke["total_decisions"], smoke["checkpoint_interval"]),
                         (True, 10_240, 4_096))
        for bad in ("ppo_s0_1p5m", "smoke_ppo_s0_1", "smoke_direct_s0_", "ppo_direct_s1_1p5m"):
            with self.assertRaises(ValueError):
                dec.run_settings(bad)
        for bad in ("ppo_direct_s0_1p5m", "smoke_direct_s0_1"):
            with self.assertRaises(ValueError):
                ec.run_settings(bad)
        self.assertEqual(dec.run_dir("x").parent.name, "ppo_direct")
        self.assertEqual(dec.decisions_trained_on(150_000), ec.decisions_trained_on(150_000))
        self.assertEqual(dec.checkpoint_steps(), ec.checkpoint_steps())
        self.assertEqual(dec.forbidden_reset_seeds(), ec.forbidden_reset_seeds())

    def test_pid_ppo_adapter_refuses_a_direct_model(self):
        import controllers
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "final_model.zip"
            (Path(tmp) / "ppo_contract.json").write_text(json.dumps(self.dpc.contract()), encoding="utf-8")
            with self.assertRaises(controllers.ControllerRefused):
                controllers.check_ppo_contract(model)
            (Path(tmp) / "ppo_contract.json").write_text(json.dumps(self.pc.contract()), encoding="utf-8")
            controllers.check_ppo_contract(model)   # the residual contract is still accepted

    def test_arm_configs(self):
        import ppo_setup
        self.assertEqual(ppo_setup.arm_configs("direct"), (self.dec, self.dpc))
        self.assertEqual(ppo_setup.arm_configs("residual"), (self.ec, self.pc))
        with self.assertRaises(ValueError):
            ppo_setup.arm_configs("sac")


def _sb3_available():
    import importlib.util
    return importlib.util.find_spec("stable_baselines3") is not None and importlib.util.find_spec("torch") is not None


@unittest.skipUnless(_sb3_available(), "stable_baselines3 / torch not installed")
class DirectPolicyTest(unittest.TestCase):
    """Builds the SB3 policy on a dummy env with the real spaces (no Isaac, no SimulationApp)."""

    def _model(self, config):
        import gymnasium as gym

        import ppo_setup

        class Dummy(gym.Env):
            observation_space = gym.spaces.Box(-path_sac_config.OBSERVATION_CLIP, path_sac_config.OBSERVATION_CLIP,
                                               (128,), np.float32)
            action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)

            def reset(self, seed=None, options=None):
                return np.zeros(128, np.float32), {}

            def step(self, action):
                return np.zeros(128, np.float32), 0.0, False, False, {}

        from stable_baselines3.common.vec_env import DummyVecEnv
        return ppo_setup.build_model(DummyVecEnv([Dummy]), config), ppo_setup

    def test_parameter_counts_architecture_and_log_std(self):
        import torch
        from config import direct_ppo_config as dpc
        from config import ppo_config as pc
        model, ppo_setup = self._model(dpc)
        counts = ppo_setup.parameter_counts(model)
        self.assertEqual((counts["actor"], counts["critic"], counts["total"]), (66_312, 148_353, 214_665))
        pol = model.policy
        self.assertEqual([m.out_features for m in pol.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)],
                         [64, 128, 256, 64])
        self.assertEqual([m.out_features for m in pol.mlp_extractor.value_net if isinstance(m, torch.nn.Linear)],
                         [128, 256, 256, 128])
        self.assertTrue(all(isinstance(m, torch.nn.ReLU) for m in pol.mlp_extractor.policy_net
                            if not isinstance(m, torch.nn.Linear)))
        self.assertEqual(tuple(pol.log_std.shape), (4,))   # state-independent, learned, one per action dimension
        np.testing.assert_allclose(pol.log_std.detach().numpy(), np.full(4, math.log(0.4)), rtol=0, atol=1e-7)
        self.assertEqual(ppo_setup.parameter_counts(self._model(pc)[0])["total"], 198_921)   # residual arm


if __name__ == "__main__":
    unittest.main()
