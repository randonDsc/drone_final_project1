# Milestone 8 — PPO direct control: design and open decisions (DESIGN ONLY)

Labels: **Confirmed** (ran or read it), **Documented** (written in a paper, plan, README or code comment),
**Inference** (reasoned, not tested), **Proposed**.

Status: draft for Claire (2026-11-15; updated 2026-11-16).
- No environment, policy, training or evaluation code exists for direct control.
- Decided: what Claire fixed on 2026-11-15 (section 3) and her decisions of 2026-11-16 (section 3b: reward,
  target_kl, evaluation path, observation features, run ids).
- Prespec amendment A4 (2026-11-16) retired P1 and replaced it with P1′, because the direct arm sits upstream of the
  1.5 m/s² cap (section 2a; `docs/residual_vs_direct_prespec.md`, section 10).
- Still open (section 4 and prespec section 8): D-env-1 confirmation, D-env-3 integrators, D-env-4 yaw, D-env-5
  realization hash, the inner-loop route (section 2e), P1′ (c)'s numeric criterion, whether P2 stays. They go into a
  committed prespec addendum before anything is built.

## 1. Source paper (Documented: Liu, Hao, Wang & Wang, *Sensors* 2025, 25, 4895; `drone_team_main/papers/Trajectory Tracking Controller.pdf`, team commit 61d2722)

Page numbers are the journal's (p. x of 18). Figures are images and were not readable as text.

**Control architecture.**
- **The policy outputs velocity setpoints plus a yaw angle to a flight controller.** The paper uses PX4 in Gazebo (§2.1,
  p. 4; §4.1, p. 9). The inner loops that turn those setpoints into thrust and torques are PX4's, and the paper does not
  describe them or their limits.
- **Observation:** a 9-dim relative state S*_t = [Δp, ΔΘ, Δv] (position, Euler-angle and velocity error to the target;
  eq. 3, p. 4), stacked over the current and previous n frames (eq. 4). With n = 3 that gives 36 inputs (§4.1, p. 10;
  n chosen in §4.2, p. 11).

**Policy output (§2.1, p. 4).**
- u_t = [v_x, v_y, v_z, γ], where v_x, v_y, v_z are *inertial-frame* velocities and γ is the yaw angle.
- **No bounds or scaling are given for any component.**

**Networks (§4.1, p. 10).**
- Separate actor and critic, 4 hidden layers each, ReLU.
- Actor 64-128-256-64; critic 128-256-256-128.
- The critic is said to "accept system states and actions as input" while estimating V(S_t), which is inconsistent for
  a state-value function; not explained.

**Training parameters (Table 2, p. 10).**

| Parameter | Value |
|---|---|
| GAE λ | 0.95 |
| Actor learning rate | 2e-4 |
| Critic learning rate | 3e-4 |
| Policy update epochs | 9 |
| Initial action std | 0.4 |
| Action std decay rate | 2e-4 |
| Learning-rate decay coefficient | 1e-4 |
| Discount ζ | 0.98 |
| CBP ρ | 1e-5 |
| CBP η | 0.99 |
| CBP m | 50 |

**Reward (eq. 5, p. 5; weights §4.1, p. 10).**
- r_t = −(ω1‖Δp‖ + ω2‖ΔΘ‖ + ω3‖Δv‖ + ω4‖u_t − u_{t−1}‖) + r⁺_p + r⁻_crash.
- Weights: ω1 = ω2 = 0.12, ω3 = 0.05, ω4 = 0.1.
- r⁺_p = +9 when ‖Δp‖ < d = 0.05 m; r⁻_crash = −100 for a crash or leaving the 5 m × 5 m × 5 m boundary.
- There is **no action-magnitude cost**, only the smoothness term ω4.

**Continual Backpropagation (§3.3, pp. 7–8; Algorithm 1, p. 9; from Dohare et al., Nature 2024).**
- Each hidden unit i of layer l keeps a utility u ← η·u + (1−η)·|h_i|·Σ_k|w_{i,k}| (eq. 13) and an age.
- Units younger than the maturity threshold m are exempt.
- After each update, a fraction ρ of mature units per layer, those with the lowest utility, are reinitialised.
- The text says the outgoing weights are set to zero, so the network function is not perturbed (p. 7). Algorithm 1
  writes "w_{l,i,·} ∼ N(0, σ)".
- The overhead is O(parameters) per update.

**Training budget and curriculum.**
- The foundation PPO policy is trained **wind-free for 2,000 episodes** (§4.2, p. 11).
- Then **stepwise wind** (§4.3.1, p. 13): +7 m/s along both x and y every 200 episodes, up to a cumulative 35 m/s.
- Or **stochastic wind** (§4.3.2, pp. 14–15): N(4, 2.5) m/s per axis, resampled every 10 episodes ("≈1 min", so about
  6 s per episode, Inference), for 100 episodes.
- The trajectory is always one circle, radius 1 m at 1 m/s from (1, 0, 3) m. There is **no route curriculum**.
- The result is "≈15%" lower average error than PPO (p. 13). It comes from one run, with no seeds and no CIs reported.

**What the paper does NOT specify for our setup.**
- Action bounds and scaling; the ±3 / ±1.5 m/s and ±π values in section 3 are ours.
- How setpoints are executed: inner-loop gains, acceleration or tilt limits, rate.
- Decision rate and steps per episode; total environment steps.
- Rollout length (n_steps), minibatch size, clip ε, entropy coefficient (only "entropy regularization", p. 11),
  value-loss coefficient, target_kl (none mentioned), gradient clipping, observation normalization.
- Whether std is learned or a decayed constant. "Initial std 0.4, decay rate 2e-4" suggests a scheduled, not learned,
  std (Inference). The schedule's unit (per step or per episode) is not given.
- How separate actor and critic learning rates and the "learning-rate decay coefficient" are applied.
- What "target attitude" ΔΘ refers to when the policy commands velocities.
- For CBP: which networks get it (actor, critic or both); σ; how the fractional ρ·N is accumulated
  (ρ = 1e-5 × 64–256 units gives ⌊ρ·N⌋ = 0 at every update unless fractions accumulate, as in Dohare et al.); whether
  the optimizer state of a reset unit is cleared.
- Drone (2 kg, Table 1, p. 9) and wind model differ from ours (1.6186 kg, `wind.py` drag).
- The paper's discount 0.98 cannot be used here: our failure penalty is cost_cap / (1 − 0.99) (CLAUDE.md).

## 2. Code facts for the design (Confirmed, read-only; lines in `residual_ppo/`, shared modules = team 61d2722)

### 2a. The controller's entry point

- **There is no velocity → acceleration → tilt → torque entry point.** The only API is
  `PIDController.update(positions, orientations, velocities, angular_velocities, target, dt, actuators_saturated)`
  (`controller.py:52-61`). The layers are internal.
- **Inputs:** world position [m], unit quaternion [w, x, y, z], world linear velocity [m/s], world angular velocity
  [rad/s] (`controller.py:3-6, 70-75`).
- **Output:** body +Z thrust [N] and body roll/pitch/yaw torque [N·m] (`:298-303`).
- **The target dict** holds `height` and `xy` (world position targets), `yaw` [rad], `mass`, `max_xy_speed`,
  `max_z_speed`, `max_tilt_degrees` and `max_total_thrust` (`:121-126, 151, 165-167, 180-215`).
- **The velocity layer is internal:**
  - height: `desired_vz = clip(kp/kd · (height − z), ±max_z_speed)` (`:152-153`);
  - horizontal: `desired_v_xy = kp/kd · (xy − p_xy)`, capped in norm at `max_xy_speed` (`:195-200`).
- **The acceleration layer** is `a_xy = kd·(desired_v − v) + xy_integral` (`:201-204`). It is capped at
  **`min(1.5, a_vert·tan(max_tilt))`**, and **the 1.5 m/s² is a literal in shared code** (`:213-216`). The tilt limit
  comes from the target dict.
- **Tilt layer:** acceleration rotated into the desired-yaw axes, then converted to pitch and roll targets
  (`:225-243`).
- **Attitude layer:** torque = kp·error − kd·body rate + integral (`:261-276`).
- **What PathSacEnv passes.** `ReferencePID` (`path_sac_policy.py:112-233`) builds the target from
  `sim.base_target()` (`drone_sim.py:409-420`): roll 0, pitch 0, **yaw 0**, `mass = total_mass`,
  `max_xy_speed = 5`, `max_z_speed = 5` (`sim_config.py:66-67`), `max_tilt_degrees = 10` (`sim_config.py:43`),
  `max_total_thrust`. It adds a position target, and makes two calls per physics step:
  1. a **read** with `dt = 0` at the lead point p_ref + (kd/kp)·v_ref + a_ref·(1/kp or m/kp)
     (`path_sac_policy.py:140-149, 165-170, 186-188`). With gains xy 0.6/0.15/1.2 and height 10/3/5 this turns the
     PID's output into 0.6 e_p + 1.2 e_v + a_ref horizontally (`:115-121`);
  2. an **integrate** call at p_ref itself, whose output is discarded (`:190-192`). This keeps the integrators off the
     lead (`:124-128`).
- **Consequence: the inner loop can be commanded a velocity without editing it** (Inference from the algebra above).
  Pass `xy = p + (kd/kp)·v_cmd = p + 2·v_cmd`, `height = z + 0.5·vz_cmd`. Then desired_v = v_cmd exactly, as long as
  |v_cmd| ≤ `max_xy_speed` = 5 (≥ 3√2 = 4.24) and |vz_cmd| ≤ `max_z_speed` = 5.
  - The integrators would then integrate 2·v_cmd and 0.5·vz_cmd, and wind up in the direction of travel. This is the
    same hazard ReferencePID avoids (`:124-128`).
  - That makes integrator handling a design decision (D-env-3).
- **Authority consequence (Inference, important for the prespec).**
  - Commanded through this inner loop, direct control gets **at most 1.5 m/s² horizontal acceleration**: 1.5 × 1.6186
    = 2.43 N. It also gets 10° tilt, but at hover a_vert·tan 10° = 1.73 > 1.5, so the 1.5 cap binds first.
  - The residual arm adds thrust and torque *after* the PID, so it can exceed both limits.
  - Drag is 0.5 · 1.225 · (Cd·A = 0.03 m²) · v² = 0.018375·v² N (`sim_config.py:245-250`, Confirmed; README §2 rounds
    it to 0.018·v²). It equals 2.43 N at about **11.5 m/s**. So direct control through this loop may be unable to hold
    mean winds above about 11.5 m/s whatever it learns, which covers most of strong_range (10–14) and all of
    beyond_strong (14–16).
  - Prespec P1 predicted direct control's *largest advantage* there, on the premise that direct control is "not
    bounded by L". With this inner loop that premise does not hold. **P1 was retired by prespec amendment A4
    (2026-11-16) and replaced by P1′**, after the committed eval flights confirmed that the cap binds on 55–74% of
    PID's physics steps in outside_range, strong_range and beyond_strong, and the tilt cap never does.

### 2b. Yaw wrap

- **The controller wraps the attitude error, including yaw, to the short way around** (`controller.py:261-270`):
  `attitude_error = (attitude_error + π) % (2π) − π`. Yaw is measured as `arctan2(R[1,0], R[0,0])` ∈ (−π, π]
  (`:115`).
- **Worked example:** target π − ε, current −π + ε. The raw error is 2π − 2ε, which wraps to −2ε. A command near ±π
  therefore produces a small torque, not a full spin.
- The horizontal layer uses only cos/sin of the target yaw (`:226-232`), which is continuous across ±π.
- **Two caveats (Inference):**
  1. **Policy-side discontinuity.** With yaw = π·a₄, the actions +1 and −1 are the same heading. A policy holding a
     heading near ±π can flip a₄ between them. `action_rate` then charges 0.02·(Δa₄)² ≈ 0.08 per flip, about half of
     a typical whole-decision cost (2d), for no physical change.
  2. **Axis rotation lag.** During a large yaw slew the horizontal acceleration is rotated into the *commanded*
     heading's axes (`:226-232`) while the body still points elsewhere. That is a near-hover approximation.
- The reference has no yaw: `base_target` yaw = 0 (`drone_sim.py:414`) and the initial yaw is ±5°
  (`path_sac_config.py:250`). So tracking never needs a yaw away from about 0.

### 2c. The 32 features per frame (`path_sac_policy.py:11-23`, built at `:236-265`; scales `path_sac_config.py:153-164`; clipped ±10; 4 frames, oldest first)

| Index | Feature | Scale | Under direct control |
|---|---|---|---|
| 0–2 | position error p_ref − p, world | 1.0 m | meaningful |
| 3–5 | velocity error v_ref − v, world | 1.0 m/s | meaningful |
| 6–8 | reference velocity, world | 2.0 m/s | meaningful |
| 9–11 | reference acceleration, world | 1.0 m/s² | meaningful |
| 12–17 | first two columns of R (body → world) | — | meaningful |
| 18–20 | attitude error from the **PID's desired attitude** (rotation vector, body frame) | 0.35 rad | **PID-specific** |
| 21–23 | body angular velocity | 2.0 rad/s | meaningful |
| 24 | **PID thrust − hover thrust** | 5.0 N | **PID-specific** |
| 25–27 | **PID roll/pitch/yaw torque** | 0.1 N·m | **PID-specific** |
| 28–31 | previous normalized action | — | meaning changes: was residual, becomes previous velocity/yaw command |

How the PID-specific features are produced:
- `desired` is reconstructed from the PID's torque command (`desired_attitude`, `:194-210`).
- The PID command comes from `ReferencePID.read`, at `:509` during observe and `:563` during step.

Each PID-specific feature can be filled in three ways:
1. **From a "shadow" ReferencePID** that is computed but not flown. This feeds the PID's own plan to the policy, so
   the "no PID" arm still learns from PID, which is misleading for the ablation.
2. **From the inner loop under the policy's own command.** Then it means "how far the attitude lags the commanded
   tilt" and "what the inner loop is commanding", which is meaningful but different.
3. **Zeroed**, which keeps 128 inputs but wastes 7 of 32.

### 2d. Reward terms and their typical size

**Formula.** r = −0.1 · (min(Σ costs, 10) + failure penalty). The failure penalty is 10 / (1 − 0.99) = 1,000 on
termination (`path_sac_env.py:238-246`; `path_sac_policy.py:363-370`). Weights are in `path_sac_config.py:173-216`.
Physics terms are averaged over the 12 physics steps of a decision (`path_sac_env.py:163-195, 227-237`) and weighted in
`decision_costs` (`path_sac_policy.py:330-360`).

| Term | Weight | Computed from (code) | Mean per decision, ALL (5th–95th pct per rollout) | Share of cost |
|---|---|---|---|---|
| position | λ_v 1.0 | Σ_axes Huber((p_ref − p)/0.25, δ 1) (`:317`) | 0.0799 (0.029–0.140) | 51.0% |
| speed | λ_v 1.0 | Huber(\|‖v_ref‖ − ‖v‖\|/0.5) (`:318`) | 0.0182 | 11.6% |
| direction | λ_v 1.0 | Huber(w_dir·(1 − cos∠(v_ref, v))) (`:292-303, 319`) | 0.0027 | 1.7% |
| attitude | λ_a 1.0 | Σ Huber(rotvec(**PID desired** → actual)/0.2) (`:320-322`) | 0.0262 | 16.7% |
| rate | λ_a 1.0 | Σ Huber(ω_ref − ω_body), ω_ref = rate of the **PID's desired attitude**, clipped ±1 rad/s (`:323`; `:512-520`) | 0.0104 | 6.6% |
| unsafe_tilt | 1.0 | ((tilt − 20°)/20°)² above 20° (`:324`) | 0 | 0% |
| unsafe_rate | 0.5 | ((‖ω‖ − 3)/3)² above 3 rad/s (`:325`) | 0 | 0% |
| route_deviation | 1.0 | ((‖e_p‖ − 1)/1)² above 1 m (`:326`) | 0 | 0% |
| **action** | 0.005 | Σᵢ aᵢ² of the clipped normalized action (`:352`) | **0.0056** (0.0025–0.0083) | **3.5%** |
| **action_rate** | 0.02 | Σᵢ (aᵢ − a_prev,i)² (`:353`) | **0.0137** (0.0103–0.0219) | **8.8%** |
| **saturation** | 0.5 | share of the decision's physics steps in which the **mixer clipped a rotor** (`path_sac_env.py:208`; flag from `sim.step_command`, `path_sac_policy.py:578`) | **0** in all 733 rollouts | **0%** |
| persistent_saturation | 2.0 | ≥ 5 fully saturated decisions in a row (`path_sac_env.py:209-215`) | 0 | 0% |
| **total** | | sum of means | **0.157** (short_slow 0.196, turns 0.086, climbs_loops 0.142, full 0.181) | |

**Source of the magnitudes (Confirmed).** `runs/ppo_path/ppo_s0_1p5m/logs/progress_20261114_164903.csv`, which is the
run's full SB3 log. It is untracked; the committed `results/m6/m6_progress_condensed.csv` has **no `cost/*` columns**.
`train_path_ppo.py:329-330` records each value as `record_mean` over a 2,048-decision rollout, unscaled and before the
cap.

**What the three action-side terms penalize, now and under a ±3 m/s world-velocity action.**
The normalized action would be a = (v_x/3, v_y/3, v_z/1.5, γ/π).

- **action (0.005·Σa²).**
  - *Now:* the size of the correction to a PID that already flies, i.e. a pull back toward "trust the PID".
  - *Direct:* the commanded **speed squared plus the absolute commanded heading squared**. a = 0 means "hover in place
    at heading 0", so the term becomes a **bias against moving**, which the policy must pay to follow any moving
    reference.
  - *Size (Inference, arithmetic):* eval routes ("full" stage) run at 1–2 m/s horizontal and ≤ 1 m/s vertical
    (`path_sac_config.py:358-361`). That costs 0.005·(v/3)² = 0.0006–0.0022 horizontally and up to 0.0022 vertically.
    A steady 7 m/s wind adds roughly a 0.45 m/s command offset, about +0.0001. This totals about 0.003–0.005 per
    decision, the same order as today's 0.0056 (2–3% of the total).
  - *Yaw:* this term is the only thing that pulls the heading toward 0, which is what the PID holds.
- **action_rate (0.02·ΣΔa²).**
  - *Now:* jumps in thrust and torque correction.
  - *Direct:* jumps in commanded velocity, i.e. commanded acceleration. A real maneuver at the inner loop's 1.5 m/s²
    cap changes v by 0.15 m/s per decision, Δa = 0.05, costing **5e-5 per axis: negligible**.
  - *What it measures in practice (Inference):* today's value is what exploration noise alone would give. Independent
    Gaussian noise with std σ costs 0.02·4·2σ², which is 0.0144 at σ = 0.3, against 0.0137 logged.
  - So in both arms the term mainly taxes exploration noise and chattering. It also charges the spurious yaw-wrap
    flip (2b).
- **saturation (0.5·share of saturated physics steps).**
  - *Both arms:* computed from the mixer clipping a rotor, **not from the action**.
  - *Direct:* it would penalize velocity commands that drive the inner loop's thrust or torque demand into the rotor
    limits.
  - It was **0 in every training rollout and every eval flight** (M6, M7), so it has contributed nothing so far.
  - *Under direct control (Inference):* the 1.5 m/s² cap and 10° tilt keep the horizontal demand small. Vertically,
    vertical force = m g + 5·(vz_cmd − v_z) + integral (`controller.py:154-158`). Reversing from v_z = −1.5 to a
    +1.5 m/s command asks for about 15.9 + 15 + 3 ≈ 33.9 N, **above the 28.3 N total thrust limit**.
  - The controller then clips total thrust (`:169-171`). With every rotor at its limit, any torque on top is clipped
    by the mixer, so **saturation can fire under direct control**, briefly, on aggressive vertical commands.
  - Under direct control, this term would therefore act as a real penalty on aggressive climb and descent reversals,
    which it never did in the residual run.

**Terms that cannot stay identical in *meaning*.** attitude and rate (23% of the cost) are measured against the PID's
desired attitude. The same code (`physics_terms(truth, sample, core.desired, core.omega_ref, …)`) needs a decision on
what `core.desired` is when no PID position loop flies. That decision is the same choice as for features 18–27 (2c).
(Decided 2026-11-16, section 3b: the inner loop's desired attitude, in both arms.)

### 2e. Can the inner loop be driven from a velocity command without modifying `controller.py`?

**Short answer (Confirmed from the code, plus one scratch measurement): yes, through the public `update()` with a
substituted target. Reimplementation is not the only route. Subclassing and "calling internals" are not routes.**

- **Subclassing gives no hook.** `PIDController` has three methods: `__init__` (`controller.py:38-40`), `reset`
  (`:42-50`) and `update` (`:52-303`). The whole cascade is inline in `update`: the velocity request is computed from
  the target position at `:152-153` (height) and `:195-200` (horizontal), then acceleration, cap, tilt and torque
  follow in the same method body. There is no method that takes a velocity, so a subclass that accepts one would have
  to override `update` as a whole. That is a reimplementation placed in a subclass, not reuse.
- **There are no internals to call.** The only separate piece is the module function `integrate` (`:17-32`, the
  anti-windup step). The integrator states (`height_integral`, `xy_integral`, `attitude_integral`) are plain
  attributes.
- **Target substitution works on the unchanged class.** The horizontal request is `kp/kd · (xy − p)` and the vertical
  one `kp/kd · (height − z)`. Passing
  `xy = p + (kd/kp)·v_cmd = p + 2·v_cmd` and `height = z + (kd/kp)·vz_cmd = z + 0.5·vz_cmd`
  makes the shared code request v_cmd itself. Everything downstream (acceleration, the 1.5 cap, tilt, attitude, torque,
  thrust clip) is the shared code running unmodified.
  - **This is the shared env's own technique.** `ReferencePID` "flies a reference through the unchanged
    position-only PIDController" by commanding a lead point (`path_sac_policy.py:112-122, 165-170`). The residual arm
    already relies on it.
  - **No speed clip is hit:** |v_xy| ≤ 3√2 = 4.24 < `max_xy_speed` 5, and |vz| ≤ 1.5 < `max_z_speed` 5.
  - **Integrators:** the substituted target would make them integrate 2·v_cmd and 0.5·vz_cmd. The same read
    (`dt = 0`) / integrate split that `ReferencePID` uses (`path_sac_policy.py:186-192`) separates them. What the
    integrate call targets is D-env-3.
  - **It is not bit-exact in v_cmd (Confirmed, scratch test `subst_rounding.py`, 1,000,000 random samples, NumPy, venv).**
    The gain ratios are exact in floating point (kd/kp = 2.0 and 0.5; kp/kd = 0.5 and 2.0). But `(p + 2v) − p` is not
    always `2v`. The recovered velocity equals v_cmd exactly in 64.5% of horizontal and 35.8% of vertical samples; the
    largest error was 1.8e-15 m/s, for |p| ≤ 30 m and z ≤ 10 m. So the cascade is fed a velocity that differs from
    the command by about 1e-15 m/s at most. Physically irrelevant, but it rules out a claim of bit-exactness in v_cmd.
- **What each route lets Claire claim:**
  - **Substitution:** "the inner loop is the shared `PIDController.update`, unmodified; the policy's velocity enters
    through the same target interface the residual env uses, exact to about 1e-15 m/s". No cascade-equivalence test
    is needed, because the cascade is the shared code. A mapping test (below) records the rounding.
  - **Reimplementation** (our own class taking v directly): bit-exact in v, but it is new code, so "unchanged" must be
    proven by the equivalence test below.
  - Both keep the 1.5 m/s² cap, as A4 assumes.

**Equivalence test for a reimplementation (Proposed; NumPy only, no Isaac).**
- **Claim tested:** for every input, our class fed `v_in` = the velocity the shared PID itself computes returns
  **thrust and all three torques, and the three integrator states after the call, bit-for-bit equal** to
  `PIDController.update` (`numpy.array_equal` on float64 and equal `.tobytes()`). `v_in` is the **pre-clip** request,
  computed in the test with the shared code's own expressions: `kp/kd · (xy − p)` (`:196`) and
  `kp_height/kd_height · (height − z)` (`:152`). Our class then applies the same speed clips.
- **One input case:** state (p, wxyz quaternion, v, ω); target dict (`xy`, `height`, `yaw`, `mass` = 1.618614,
  `max_xy_speed`, `max_z_speed`, `max_tilt_degrees`, `max_total_thrust`); the three integrator states before the call;
  `dt` ∈ {0, 1/120}; `actuators_saturated` ∈ {False, True}. Both controllers get identical copies.
- **Part A, independent cases: 200,000, generated with `numpy.random.default_rng(0)`, stratified so every branch is
  hit on both sides.** At least 10,000 cases on each side of each branch, with the hit counts reported; the test fails
  if a branch falls short:
  - horizontal cap active vs inactive, with 10% of cases within ±1% of `max_acceleration`;
  - cap set by 1.5 vs by tilt (low vertical acceleration, the 0.5 g floor at `:208-211`);
  - horizontal speed clip and vertical speed clip active vs inactive;
  - thrust clipped at 0, at `max_total_thrust`, or not;
  - R₂₂ below vs above the 0.5 floor (`:161`);
  - yaw error near ±π (the wrap at `:268-270`);
  - integrators at ±limit, errors inside vs outside `integral_zone`, blocked × growing (`integrate`, `:25-32`).
- **Part B, sequences: 200 sequences × 1,842 physics steps = 368,400 steps.** 1,842 is the median physics steps of a
  completed eval route (Confirmed, step 1). Both controllers start from `reset()` and are stepped in lockstep, carrying
  their own integrators. Each step uses the two-call read (`dt = 0`) / integrate (`dt = 1/120`) pattern. States come
  from a seeded smooth random walk (seed 1): Ornstein–Uhlenbeck velocity and body rate, integrated position and
  quaternion, tilt up to 30°. Outputs and integrators must be bit-equal at every step, which tests integrator carry-over.
- **Part C, real states (optional; needs one Isaac run by Claire).** Log the (state, target, saturated) the shared PID
  actually saw in the canonical PID flights of validation seed 400000 (36 cells, up to about 66,000 physics steps),
  through a read-only `step_command` wrapper like `Diagnostics`, then replay them offline. This covers the real state
  distribution, including the capped steps in strong wind.
- **Negative tests (each must be detected by Part A):**
  - cap 1.5 replaced by `numpy.nextafter(1.5, 2)`;
  - the three-term vertical-force sum re-associated, `m g + (kd·Δv + integral)` (floating-point addition is not
    associative);
  - the yaw wrap removed;
  - a float32 intermediate.
- **Where it runs:** both interpreters (venv Python 3.13 and Isaac `python.bat` 3.12), as in M1.4. Bit-identity is
  required within each interpreter, with both sides on the same NumPy.
- **Report:** case counts, seeds, branch hit counts, max |diff| (must be 0), negative tests detected.

**Mapping test for the substitution route (Proposed).** On the Part A and B inputs, compare
`update(target with xy = p + 2·v, height = z + 0.5·vz)`, where v and vz are computed from the original target, against
`update(original target)`. Report the fraction of bit-identical outputs and the max |diff| per output. The expectation
(Inference) is not all bit-identical, with differences of order 1e-15 relative.

## 3. Design fixed by Claire (2026-11-15; recorded, not re-opened)

- **Environment.** A new env class in our namespace (proposed path `residual_ppo/direct/…`; name pending, section 6)
  that **composes** the shared physics (`drone_sim` via `hover_env.make_hover_sim`), wind, reference, failure rules,
  completion tracker and `PIDController` as the inner loop. **No shared module is edited**, and
  `tools/sync_shared.py --check` must stay 21/21.
- **Action.** Box(−1, 1, (4,)) mapped to world-frame **v_x, v_y ∈ ±3 m/s, v_z ∈ ±1.5 m/s, yaw angle ∈ ±π**.
- **Policy.**

  | Setting | Value |
  |---|---|
  | Library | SB3 PPO |
  | `net_arch` | `dict(pi=[64, 128, 256, 64], vf=[128, 256, 256, 128])` |
  | Activation | ReLU (paper and residual run) |
  | log_std | one state-independent, learned, `log_std_init = ln 0.4 = −0.9163` |

  Parameter count (Inference, arithmetic for 128 inputs): actor 66,312 (including 4 log_std), critic 148,353, total
  **214,665**. The residual PPO has 198,921.
- **Training.** Same curriculum (`path_sac_config.CURRICULUM`), the same 1,500,000-decision budget (733 rollouts =
  1,501,184 decisions with n_steps 2,048) and seed 0. Every other hyperparameter is as the residual run unless section 5
  changes it: n_steps 2,048, batch 256, 10 epochs, lr 3e-4, γ 0.99, λ 0.95, clip 0.2, ent 0.001, no VecNormalize,
  device cpu.
- **Differences from the residual arm that the addendum must record with their reason** (prespec section 8.3 requires
  it): net_arch (paper's, not 256×256); log_std_init (−0.9163, not −1.0); and the paper's learned-vs-decayed std
  ambiguity (section 1).
  - These are **additional differences between the arms beyond the control interface** (Inference): any residual-vs-
    direct difference is then attributable to "architecture + interface", not interface alone.
  - Claire may accept this, or run a 256×256 / −1.0 variant (not proposed here).

## 3b. Decisions made by Claire (2026-11-16; recorded, not re-opened)

- **D1 — reward: byte-identical** (option A). Same reward code and constants in both arms.
  - Same code, different behaviour between arms: `cost/saturation` was **exactly 0 in every residual training rollout**
    (all 733) and every eval flight. In the direct arm it **can fire**, because a vz command can demand about
    **33.9 N** against the **28.3 N** total rotor limit (vz reversal −1.5 → +1.5 m/s: 15.9 + 5 × 3 + integral; 2d).
  - The action terms change meaning too (2d), and the reward numbers are not comparable across arms. No endpoint uses
    reward.
- **D2 — target_kl stays 0.02**, for parity with the residual run.
  - **Pre-declared undertraining check:** report the **gradient steps of both arms** (optimizer-hook count; residual
    10,305) and **whether validation performance was still improving from the 800k checkpoint to the final model**,
    on the validation seeds 400000–400009, as for the residual arm's learning curve.
  - The exact rule for "still improving" (which metrics, which profiles, what counts as improving) belongs in the
    prespec addendum (Proposed: the same quantities as `learning_curve.py`).
- **D3 — evaluation through a separate, unfrozen runner** (option 2). It imports the frozen `run_protocol.fly` and
  the realization guard unchanged. **No protocol v2, nothing re-flown**; v1 stays the single protocol and every
  committed v1 result stays the citation. The runner is frozen by its own blob ID in the addendum before any direct
  flight, with a negative guard test as in M2.5.
- **D-env-2 — observation features 18–20 and 24–27 are defined against the inner loop's desired attitude and the
  inner loop's outputs, in BOTH arms.** In the residual arm the inner loop is the PID, so nothing changes there. In
  the direct arm it is `PIDController` under the policy's velocity command. The same definition feeds the attitude
  and rate reward terms (`core.desired`, `core.omega_ref`). **No shadow PID anywhere.**
  - Consequence (Inference): an integrate call targeting the reference path p_ref (as `ReferencePID.integrate` does)
    would give the direct arm's inner loop an integral pull toward the reference, i.e. part of a PID position loop.
    That conflicts with "no shadow PID" and is excluded from D-env-3.
- **Run ids: `ppo_direct_s0_1p5m`** (main run) **and `smoke_direct_s0_<n>`** (smoke runs). They may be renamed later
  to match the team convention (section 6).

## 4. Environment design items that follow from section 2 (to settle in the addendum; not resolved)

- **D-env-1 — inner-loop authority (affected prespec P1; see A4).** Commanding `PIDController` caps direct control at
  1.5 m/s² horizontal (a literal in `controller.py:214`). **A4's P1′ is written for option (a)**, the shared inner
  loop with its cap (Inference: (b) and (c) would change the mechanism P1′ predicts). Claire to confirm (a) in the
  addendum. Options:
  - (a) keep the shared inner loop as is. Same PID inner loop as both other arms; P1's premise is then probably false
    by construction (section 2a), and the result should be read that way.
  - (b) raise only the tilt limit through the target dict (allowed, no code edit). **Ineffective**: the 1.5 cap binds
    first.
  - (c) our own inner-loop class, a reimplementation of the velocity → attitude layers with a configurable cap. This is
    new code, not a shared edit. It is no longer "the same PID", and it would need its own validation, such as
    bit-identical output to `PIDController` at cap 1.5.
- **D-env-2 — DECIDED 2026-11-16 (section 3b):** the inner loop's desired attitude and outputs, in both arms, for
  both observation and reward. No shadow PID.
- **D-env-3, D-env-4, D-env-5: resolved by code reading on 2026-11-16, see section 8** (the text below is the
  2026-11-15 record).
- **D-env-3 — inner-loop integrators.** Using the velocity trick (2a, 2e) the integrators wind up on the command.
  Options (corrected 2026-11-16):
  - a second `update(dt)` call with target = current position. Then xy and height errors are 0, so those two
    integrals do not move (`integrate`, `controller.py:25`: step = ki · 0 · dt). The **yaw** attitude integral still
    advances on the true yaw error: the yaw target comes from the target dict, and attitude ki is (0, 0, 0.02), so
    roll and pitch never integrate (`sim_config.py:208`). The 2026-11-15 draft wrongly said "the integrators
    never move";
  - `dt = 0` only, with no integrate call: all integrals stay 0, yaw included;
  - a `PIDController(gains=…)` copy with ki = 0 (no shared edit);
  - integrate velocity error instead (needs a reimplementation, 2e);
  - integrating on p_ref (as the residual env does) is **excluded** by "no shadow PID" (section 3b).

  In every option the policy must cancel steady wind itself; it observes velocity error.
- **D-env-4 — yaw.** Keep ±π as specified (accepting the spurious wrap flip in action_rate, 2b). Or wrap the policy
  output (e.g. γ = π·a₄ with a₄ clipped), or use a smaller range. The reference never needs yaw.
- **D-env-5 — the realization interface**, so the evaluation can prove identical conditions (section 5, D3). Expose
  `reset()` info `wind_seed`, `wind_mean`, `route_seconds`, `max_decisions` exactly as PathSacEnv does. Also compute the
  **residual-style observation at reset** purely to hash it as `initial_obs_sha256` (Proposed). That proves the same
  initial drone state as the canonical PID run, even though the policy sees a different observation.

## 5. Decisions D1–D3 (DECIDED 2026-11-16, section 3b; the evidence and options are kept as the record)

### D1 — Reward: byte-identical, or adjust the action-cost terms (DECIDED: A, byte-identical)

**Evidence (2d, Confirmed unless marked):**
- action + action_rate are 12.3% of the residual run's cost, and saturation is 0%.
- Position (51%), attitude (16.7%) and speed (11.6%) dominate.
- Under the direct mapping the action term stays about 0.003–0.005 per decision (Inference), but it changes meaning
  from "deviation from PID" to "speed and heading".
- action_rate on physically realizable commands is negligible (5e-5 per axis), and otherwise taxes exploration noise,
  as it does now (Inference).

**Options:**
- **(A) Byte-identical reward** (same code and constants; with D-env-2, `core.desired` comes from the inner loop).
  - *Costs the ablation:* the action term becomes a small hover/heading prior (2–3% of cost), and attitude/rate measure
    inner-loop lag instead of deviation from the PID's plan. The reward *numbers* are not comparable across arms, but
    no endpoint uses reward: E5 uses RMSE, failures and decisions. The reward *code* is identical.
- **(B) Adjust the action-cost terms**, e.g. charge 0.005·‖a − a_ff‖² with a feed-forward a_ff = (v_ref/3, v_ref/3,
  v_z,ref/1.5, 0). That penalizes deviation from the reference velocity, the analogue of "deviation from PID".
  - *Costs the ablation:* a second manipulated variable. Any direct-vs-residual difference is then partly a reward
    difference. It needs its own justification, and it is "intent parity" bought at the price of "code parity".
- **(C) Drop the action term** for direct control.
  - *Costs:* the same as (B), plus the policy loses the only pull toward heading 0.

**Recommendation (Proposed): (A).** The terms are small, (B) and (C) add a confound that the prespec would have to
carry, and action_rate behaves the same way in both arms. Exception to consider with D-env-4: if yaw stays ±π, the
wrap flip can cost 0.08 per occurrence. That is better handled in the action mapping than in the reward.

### D2 — target_kl 0.02 (strict parity) vs the risk of undertraining (DECIDED: a, 0.02 + undertraining check)

**Evidence (Confirmed):**
- SB3 stops a rollout's epochs once `approx_kl > 1.5 · target_kl` = 0.03 (SB3 2.7.0 `ppo.py:267`).
- In `ppo_s0_1p5m` that happened in **709 of 733 rollouts**: short_slow 49/73, every later rollout 100%.
- PPO made 10,305 of a possible 58,640 gradient steps (17.6%).

**Why direct control may need more (Inference):**
- The residual starts from a working PID: a = 0 already flies.
- A direct policy starts from a = 0 = "hover in place". It falls behind every moving reference from the first
  decision, so it has a much larger function to learn under the same KL throttle.
- With the same std (0.4 vs 0.37), each update's KL budget is about the same, so it would get a similar number of
  gradient steps, about 10k.

**Options:**
- **(a) 0.02, strict parity.** Clean, but "direct is worse" becomes confounded with "direct is undertrained at this
  throttle".
- **(b) Another value or `None`** (SB3 default; the paper mentions none). Breaks parity in one optimizer setting; E7b
  would show the difference in gradient steps.
- **(c) 0.02 primary plus one pre-declared secondary run with `None`.** About 2 × 8.5 h of training (Inference) and
  answers both questions, but the prespec must fix which is primary before either exists.

**Pre-declarable undertraining check, whichever option (Proposed):**
- the validation learning curve at 150k / 400k / 800k / final, as for the residual run;
- the training-rollout failure rate and position error per stage.

### D3 — How direct control is evaluated (DECIDED: option 2, separate runner)

The frozen `fly` builds only `path_sac_env` and the `pid` / `pid_ppo` / `pid_sac` controllers (`run_protocol.py:289-335`;
`controllers.py:195`).

**Option 1 — protocol v2 with an additive adapter.**
- *Steps:*
  1. Add a `ppo_direct` controller to `controllers.py` and an `--env direct` choice to `run_protocol.py fly`. Leave all
     v1 behaviour paths unchanged.
  2. Bump the version (new protocol doc `eval_protocol_v2.md`, new blob IDs, a new `EXPECTED_PROTOCOL_SHA` only if
     `eval_protocol_v1.py` changes).
  3. Re-fly **every** controller (PID about 38 min, pid_ppo about 55, pid_sac about 56) into `results/eval/eval-v2/`.
  4. Prove v2 reproduces v1 with `compare` mode, bit-for-bit, for all three.
  5. Fly direct with `--check0`, then the reports.
- *Risks:*
  - It touches the 5 frozen files, so any slip changes v1 behaviour. Mitigation: bit-for-bit `compare` against v1.
  - About 2.5 h of extra flights.
  - `Diagnostics` (`run_protocol.py:102-173`) reads `core.correction`, `core.desired` and `core.pid.acceleration_capped`,
    which are PID internals. A direct env must provide equivalents or diagnostics must be off, so check0's meaning
    changes for it.
  - The v1 results stay valid; v2 becomes the citation for any table with direct control.

**Option 2 — a separate, unfrozen runner** (e.g. `evaluation/run_direct.py`, frozen by its own blob ID in the addendum
before any direct flight).
- *Steps:*
  1. Import, unchanged, `eval_protocol_v1` (seeds, profiles, scenarios, E3 rules), `run_protocol.fly` and
     `run_protocol.make_realization_guard`. Both are generic in `env`/`controller` (`:178-229`, `:260-279`; Inference:
     `run_protocol` imports only `run_common` at module level, so it can be imported). Use `protocol_stats` for the
     statistics.
  2. Build the direct env, and load the model before SimulationApp (M2.5 rule).
  3. Guard every reset against the canonical PID `realizations.json`, using all 5 key fields. This requires D-env-5's
     residual-style `initial_obs_sha256`; otherwise drop that field and record why.
  4. Write `results/eval/eval-v1/ppo_direct/` in the same format.
  5. Run the direct-vs-residual comparison script (prespec section 4) and, separately, direct vs PID through
     `controller_report`.
- *Risks:*
  - The runner is new code outside the frozen set, so it must be frozen and tested (a negative guard test as in M2.5)
    before use.
  - `fly`'s attribute contract (`env.trace`, `env.core.reference.path`, `env.core.completion`, `env.max_decisions`,
    `step_info["failure_reason"]`) must be matched exactly by the new env.
  - Without D-env-5 the initial-state identity is weaker.
  - The frozen `report` mode cannot be used for direct vs PID: it requires `results.json` from `fly` and the canonical
    PID run. It *would* work if the output format is identical (Inference; to be tested).
  - No re-flying of existing controllers, and v1 stays the single protocol.

**Comparison (Inference):**
- Option 2 is less work, and it leaves every committed v1 result as the citation.
- Option 1 gives one runner for all four arms (including continual PPO later) at the cost of a v2 re-fly.

## 6. Naming convention

**Not recorded anywhere in the repo** (Confirmed): no Discord mention or naming rule was found in `drone_final_project1`
or `drone_team_main`.

Observed patterns (Documented):
- team: `runs/<workflow>/<run>/final_model.zip` with the contract json next to it; workflow names `flight_sac` and
  `sac_hover`; `path_sac_config.py`; HF repo `pcldsc/Iris_drone_residual_correction_SAC`;
- ours: `runs/ppo_path/<run_id>/`, `ppo_contract.json`, `pid_ppo` as the controller name.

**Decided 2026-11-16 (section 3b):** run ids `ppo_direct_s0_1p5m` and `smoke_direct_s0_<n>`, which may be renamed
later to match the team convention. Workflow and controller name `ppo_direct` remain placeholders until Claire
confirms the Discord convention.

## 7. Next step (as of 2026-11-15; superseded by section 8)

The items listed here were settled on 2026-11-16: D-env-1 (a), P1′ (c), P2 and the "still improving" rule by prespec
A5 (b684b81); the inner-loop route (target substitution, no fork) by Claire; D-env-3/4/5 in section 8.

## 8. Remaining env items resolved by code reading, and what was built (2026-11-16)

### 8.1 D-env-3 — integrators (Confirmed, code + scratch measurement `windup.py` with the shared `controller.py`)

**The integral code** (`controller.py:17-32`, called at `:245-253` xy, `:278-286` height, `:288-296` attitude):

```python
def integrate(integral, error, ki, limit, zone, dt, blocked):
    step = np.asarray(ki, dtype=float) * error * dt
    step = np.where(np.abs(error) <= zone, step, 0.0)          # holds while |error| > zone
    growing = np.abs(integral + step) > np.abs(integral)
    step = np.where(blocked & growing, 0.0, step)              # anti-windup while saturated / capped
    return np.clip(integral + step, -np.asarray(limit), np.asarray(limit))   # clamped
```

- **Clamped, not leaky, not unbounded.** Each axis is clamped to ±limit and holds while its error is outside
  `integral_zone`. Growth also stops while the output is blocked (rotor saturation, or for xy the 1.5 m/s² cap).
- **Gains (Confirmed, `sim_config.py:183-201`):**

  | Integrator | ki | limit | zone |
  |---|---|---|---|
  | height | **3.0 N/(m·s)** | 3.0 N | 0.5 m |
  | xy (horizontal) | **0.15 1/s³** | 1.2 m/s² | 2.0 m |

  Correction to the brief: ki = 3.0 is the **height** loop's; the horizontal loop's ki is 0.15.
- **What the substituted target would feed them if it were also used for integration.** The position error is
  2·v_cmd (xy) and 0.5·vz_cmd (height). Results, holding the drone at the commanded velocity and stepping the real
  shared controller at 120 Hz:

  | Command | Error seen | Integral after 60 s and after 184 s | Time to the limit |
  |---|---|---|---|
  | vx 0.5 m/s | 1.0 m (inside the 2 m zone) | **1.2 m/s²** (the limit) | 8.0 s |
  | vx 1.0 m/s | 2.0 m (at the zone edge) | **1.2 m/s²** | 4.0 s |
  | vx 2.0 m/s | 4.0 m (outside the zone) | 0 (held) | never |
  | vx 3.0 m/s | 6.0 m | 0 (held) | never |
  | vz 0.5 m/s | 0.25 m (inside the 0.5 m zone) | **3.0 N** (the limit) | 4.0 s |
  | vz 1.0 m/s | 0.5 m (at the zone edge) | **3.0 N** | 2.0 s |
  | vz 1.5 m/s | 0.75 m | 0 (held) | never |

  - A sustained **3 m/s** command therefore does **nothing** to the xy integral, because it sits outside the zone.
    The damage comes from **slow** commands, which are exactly the route speeds (1–2 m/s, `full` stage): ≤ 1 m/s
    saturates the integral within 4–8 s.
  - Route length does not matter beyond that, because the integral is clamped. The longest eval route is 39.7 s and
    the episode limit is 60 s (`EPISODE["max_seconds"]`); no route is 184 s long.
  - A saturated xy integral is **1.2 m/s² = 80% of the 1.5 m/s² cap**. It adds a steady overspeed of
    1.2 / kd = **1.0 m/s** in the direction the command once pointed. A saturated height integral adds
    3.0 / 5 = **0.6 m/s**. This is hidden, history-dependent state that the policy does not observe.
- **Minimal handling that does not alter the inner loop: none exists in the strict sense, stated plainly.**
  - In the residual arm the xy and height integrators integrate the true tracking error p_ref − p. A velocity-commanded
    inner loop has no position error of its own.
  - The only position error available is the reference's, and feeding it in would be a shadow PID position loop,
    which section 3b excludes.
  - So in the direct arm these two integrators cannot do what they do in the residual arm. Every option changes
    their *behaviour*; none needs a code change.

  | Option | What happens | Cost |
  |---|---|---|
  | **A — integrate call at the current position (BUILT)** | Same two-call pattern as `ReferencePID` (read with `dt = 0` at the substituted target, integrate with `dt` at p itself). xy and height errors are exactly 0, so those integrators stay at 0 from every reset. The yaw integrator still integrates the true yaw error, as in the residual arm (roll/pitch ki = 0). | No inner-loop integral action on xy and height. The policy must cancel steady wind and thrust bias itself, by biasing its command: holding 0.9 m/s² of drag needs a 0.75 m/s command offset. It observes velocity error, so it can. |
  | B — `dt = 0` only (no integrate call) | All integrators stay 0, yaw included. | Steady yaw disturbance torque is no longer trimmed. |
  | C — one integrating call at the substituted target | Wind-up as in the table above. | Hidden bias up to 80% of the cap, depending on command history. |
  | D — integrate on p_ref (as the residual env does) | The inner loop gets an integral pull toward the reference. | A shadow PID position loop; **excluded** (section 3b). |
  | E — integrate velocity error | Needs a reimplementation of `update`. | Excluded (no fork, A5). |

  **Built: A** (Proposed; Claire to confirm in the addendum). Verified:
  - unit test, 60 s of steps: xy and height integrals exactly 0, yaw integral > 0 and ≤ its limit;
  - the Isaac check H4 asserts it on real flights.

### 8.2 D-env-4 — yaw (Confirmed, code)

**What we pass:** `target["yaw"] = yaw_cmd = π·a₃` [rad], the yaw ANGLE itself, in the controller's convention (world
yaw, `arctan2(R[1,0], R[0,0])`, `controller.py:115`). It goes into both the read and the integrate call
(`VelocityPID._target`). No conversion is needed:
- the attitude error is `target_yaw − yaw`, wrapped to (−π, π] (`:264-270`). A command anywhere in ±π reaches the
  controller as an angle and is approached the short way;
- the horizontal acceleration is rotated into the commanded heading's axes with cos/sin of the same angle
  (`:226-232`), which is continuous across ±π.

Tested: the yaw angle arrives exactly (`target["yaw"] == π·a₃`), and the yaw error recovered from the yaw torque
matches wrap(yaw_cmd − yaw) within 1.8e-15 rad. A command of 0.999π at a heading of −0.999π gives the small torque of
the −0.002π error, not a spin.

Caveats (Inference, unchanged from 2b):
- the a₃ = ±1 flip costs 0.08 in action_rate;
- a large yaw error asks for up to 0.1 · π = 0.31 N·m of yaw torque, which may saturate the mixer (it would show in
  `saturation_rate`).

### 8.3 D-env-5 — realizations (Confirmed by design; to be proven in Isaac by `tools/check_direct_env.py`)

- **Same wind realization, route and initial state: yes, by construction.** `DirectEnv` inherits `PathSacEnv.reset`
  unchanged (`path_sac_env.py:79-141`): wind, then route, then initial state, from the same `np_random` stream in the
  same order. The four non-observation key fields (`wind_seed`, `wind_speed`, `route_seconds`, `max_decisions`)
  therefore match the PID reference.
- **Same initial-observation hash: NO, and it cannot.** The initial observation's features 18–20 and 24–27 come from
  the inner loop under the zero command (D-env-2). The PID reference hashed the PID's lead-point command. By Claire's
  own decision those 7 features differ, so `initial_obs_sha256` cannot match. That is a consequence of D-env-2, not
  a bug.
- **How it is proven (Isaac, before any training; Claire runs it):** one sim, the shared `PathSacEnv` and `DirectEnv`
  side by side. For all 360 validation cells (seeds 400000–400009, reset only, no flights):
  - R1: `PathSacEnv` reproduces all 5 key fields of `pid_validation/realizations.json` through the frozen
    `run_protocol.make_realization_guard`, so the check itself is sound;
  - R2: `DirectEnv` reproduces the 4 non-observation fields through the same guard;
  - R3: `DirectEnv`'s drone state after reset (every value `read_drone_state()` returns) and the mean wind are
    bit-identical to `PathSacEnv`'s;
  - R4: every controller-independent observation feature (0–17, 21–23, 28–31, all 4 frames) is bit-identical;
  - R5: a reset with seed + 1 is refused (negative test).
- **For the eval runner (not built yet; decision for Claire):**
  - (i) guard on the 4 fields per flight, plus R1–R4 once. This is weaker per flight, but uses no PID at all;
  - (ii) additionally reconstruct the residual-style initial observation once per reset with a hash-only shared
    `PathCore`. Its output never reaches the policy, reward or dynamics, so the frozen 5-field guard can be used
    unchanged. It is the strongest proof, but it is a PID computation at reset, which "no shadow PID anywhere" may
    forbid.

  Recommendation (Proposed): (i) + R1–R4, which keeps the rule literal.
- **First Isaac run of the check (2026-11-16 01:04): 7/10, RESULT: FAIL.**
  - **R2 0/360 was a bug in the check script, not in the env.** The 4-field guard was handed the full stored
    reference rows (5 fields). The frozen `row_differences` (`evaluation/protocol_stats.py:345-355`) counts a field
    missing on one side as a difference, so every cell was refused for lacking `initial_obs_sha256`.
  - All 360 refusals named only that field. `wind_seed`, `wind_speed` and `route_seconds` come before it in the row's
    field order, so they matched in all 360 cells.
  - **Fix:** the reference rows are cut down to the 4 compared fields before they reach the guard. Offline, the fixed
    construction accepts 360/360 NumPy-rebuilt realizations; the first-run construction accepts 0/360.
  - **Why `initial_obs_sha256` is excluded:** features 18–20 and 24–27 are controller-derived, and no shadow PID is
    allowed, so the hash cannot match between arms. Physical identity is proven instead by R3 (simulator state and
    mean wind bit-identical, 360/360 in the first run) and R4 (controller-independent features bit-identical,
    360/360).
  - **R5's first-run PASS was vacuous:** the broken guard refused every reset, so refusing seed + 1 proved nothing.
    R5 now passes only if R2 also passes.
  - **H2/H3 (hold thresholds) failed:** max drift 0.2118 m, max final speed 0.1387 m/s. The thresholds are unchanged
    pending `tools/hold_diagnostic.py`, whose decision rule Claire fixed before its run.

### 8.4 What was built (2026-11-16; NOT committed, by instruction)

| File | What |
|---|---|
| `direct_control/direct_env.py` (new) | `action_to_command`; `VelocityPID(ReferencePID)` (target substitution, yaw angle, integrate at p); `DirectCore(PathCore)` (only `apply_action` differs: correction 0, sets the command); `DirectEnv(PathSacEnv)` (only the core differs; `step` adds read-only integrator magnitudes to info); `make_direct_env`. No fork of `controller.py`; no shared module edited. |
| `config/direct_ppo_config.py` (new) | `PPO` = the residual arm's dict (asserted equal); net_arch pi [64, 128, 256, 64], vf [128, 256, 256, 128]; `log_std_init = ln 0.4`; SB3's state-independent learned log_std (one value per action dimension, as the residual arm). Policy contract = `policy_contract()` + arm + action mapping, so `pid_ppo` refuses direct models. |
| `config/direct_experiment_config.py` (new) | `runs/ppo_direct/<id>/`; ids `ppo_direct_s0_1p5m`, `smoke_direct_s0_<n>`; every other rule imported from `experiment_config`. |
| `training/train_path_ppo.py`, `training/ppo_setup.py` (edited) | `--arm residual\|direct` (default residual: residual behaviour unchanged); `arm_configs`, `make_arm_env`; direct-arm `inner/*` integrator columns in progress.csv. |
| `tools/audit_ppo_run.py`, `tools/launch_ppo_run.ps1` (edited) | `--arm` / `-Arm`. The direct audit requires `pid_ppo` to REFUSE every direct model. |
| `tools/check_direct_env.py` (new) | Isaac: R1–R5 (above) and the zero-command hold H1–H4. |
| `tests/test_direct_control.py` (new, 19 tests) | Gate 4 (mapping, hold), integrators, DirectCore, configs, SB3 policy (214,665 parameters; residual 198,921). |

**Single-log_std reading (Inference, flagged):** "single state-independent log_std" is implemented as SB3's standard
state-independent log_std **vector** (4 learned values), exactly as in the residual arm. A single scalar shared
across the 4 dimensions would need a custom policy class, i.e. a second difference from the residual arm. Say if a
scalar is what was meant.

### 8.5 Gate, hold diagnostics and the limit cycle (2026-11-16; full record in prespec A6–A10)

- **Gate `tools/check_direct_env.py`:**
  - run 1: 7/10 (R2 failed on a script bug, now fixed);
  - run 2: 8/10 (R1–R5 PASS; the old H2/H3 settling thresholds failed).
- **Hold diagnostic `tools/hold_diagnostic.py`:**
  - v1: 30 s;
  - `--suite m8b`: T1–T4. It found a bounded limit cycle in the shared inner loop, in both arms.
    - Direct: period 2.61 s, 0.068–0.075 m/s RMS.
    - Residual: period 2.82–2.88 s, 0.087–0.092 m/s RMS.
    - Still air, 120 s, 10/10 seeds hold.
  - It also found a steady bias of about 0.02 m/s (source unidentified).
- **Claire's decision (rule 2, launch authorized):**
  - H2/H3 replaced by limit-cycle regression gates: amplitude band [0.0539, 0.0874] m/s and ratio band
    [0.8265, 1.0977];
  - both bands derived from T1 (validation seeds) plus stated margins, as recorded in the script docstring.
  - Expected re-run: 10/10.
