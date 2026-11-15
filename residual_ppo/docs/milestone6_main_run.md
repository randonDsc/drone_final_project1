# Milestone 6: PPO main training run `ppo_s0_1p5m`

Labels: **Confirmed** (ran or read it), **Documented** (written in a plan or README), **Inference**, **Proposed**.
Evidence: `results/m6/` (condensed copies, model hashes and the audit). Full run folder:
`runs/ppo_path/ppo_s0_1p5m/` (untracked; zips and console logs are gitignored).

## 1. Outcome (Confirmed)

| Item | Value |
|---|---|
| Result | `RESULT: DONE`; `status.json` state `done`, finished 2026-11-15 09:19:52 UTC |
| Code | commit 6c6e549 (training script and config unchanged since M5b) |
| Launches | 1 (fresh, no resume); seeded resets [0]; torch origin clean at start and end |
| Wall clock | 30,649 s = 8 h 30 m 49 s (projection 8 h 30 m) |
| Decisions | collected 1,501,184 = trained on 1,501,184 + discarded 0 + pending 0 (733 rollouts × 2,048) |
| Gradient steps | **10,305** (of a 58,640 maximum = 733 × 80; 17.6%) |
| SB3 epochs (`_n_updates`) | 1,681 |
| target_kl early stops | 709 of 733 rollouts (96.7%) |
| Failures / invalid steps | 0 failures in 10,465 episodes; 0 invalid steps; no non-finite value in any rollout row |
| Models | `final_model.zip` + 30 checkpoints (50k … 1.5M), each with `ppo_contract.json` next to it; sha256 in `results/m6/m6_model_hashes.json` |
| Audit | `tools/audit_ppo_run.py --run-id ppo_s0_1p5m`: **46/46 PASS** (`results/m6/m6_audit_ppo_s0_1p5m.txt`) |
| Hardware | CPU torch 2.14.1+cpu, total mass 1.618614 kg, HESAI 0 kg |

**E7b (training cost, decided 2026-11-14):** PPO 1,501,184 decisions, 10,305 gradient steps, 1,681 SB3 epochs,
8.51 h on this PC (CPU torch). Team SAC (from its checkpoint): 1,500,000 decisions, 1,494,999 gradient updates,
~30 h on an RTX 4070 SUPER (Documented, team README §7). So PPO took 0.69% of SAC's gradient updates.
Wall-clock time is context, not a head-to-head score.

### Per stage (`training_summary.txt`; training flights with exploration, not evaluation results)

| Stage | Decisions | Episodes | Failure rate | Position error (m) | std | Rollouts | Early stops | Action clip % | Gradient steps |
|---|---|---|---|---|---|---|---|---|---|
| short_slow | 150,041 | 1,787 | 0.000 | 0.080 | 0.3043 | 73 | 49 | 0.72 | 3,431 |
| turns | 250,091 | 2,129 | 0.000 | 0.057 | 0.2444 | 122 | 122 | 0.98 | 2,327 |
| climbs_loops | 399,979 | 2,561 | 0.000 | 0.081 | 0.2226 | 195 | 195 | 8.17 | 2,283 |
| full | 701,073 | 3,988 | 0.000 | 0.092 | 0.2117 | 343 | 343 | 24.79 | 2,264 |

Within each stage the team's `flight/position_error` falls (mean of the first 10 → last 10 rollouts, from
`results/m6/m6_progress_condensed.csv`): short_slow 0.217 → 0.044, turns 0.061 → 0.051, climbs_loops 0.076 → 0.070,
full 0.104 → 0.085 m. Across stages the numbers are not comparable, because each stage has harder routes and stronger
wind (Inference). Policy std_mean: 0.3665 (rollout 1) → 0.2082 (rollout 733).

### Learning observations for discussion (descriptive; NOT protocol changes)
1. **Action clipping grows with the curriculum:** about 25% (25–28% in the last rollouts) of sampled action
   components fall outside [-1, 1] in `full`. The policy mean presses against the residual limits in strong wind
   (Inference). This relates to the clipped-Gaussian bias (M3 P6: SB3 stores the unclipped action) and to the
   authority limits (CLAUDE.md analysis note (a)). Action limits stay unchanged, as decided.
2. **target_kl 0.02 (provisional) early-stopped 97% of rollouts**, usually within the first epoch. That is why PPO
   used 10,305 of 58,640 possible gradient steps. Relevant to E7b and to any discussion of PPO sample efficiency.

## 2. Launch history

- **Launch 1 (2026-11-14 15:03 local) died after 5 minutes** at 12,288 decisions, before the first checkpoint
  (Confirmed: process gone, no `RESULT` line, no traceback, no crash dump; `results/m6/m6_launch1_killed_status.json`).
  It had been started through Claude Code's PowerShell tool. The leading explanation, **not proven**, is that it ran
  inside the tool host's Windows job object and was terminated with it (Confirmed: a probe started the same way was
  in a job object). The dead attempt was moved, not deleted, to `runs/ppo_path/_attempts/ppo_s0_1p5m_launch1_killed/`.
- **Rule since then (Claire):** every training launch starts from a standalone PowerShell window opened from the
  Start menu; Claude only monitors, read-only.
- **Launch 2 = the run above (2026-11-14 16:49 local)**, started by Claire that way; not in a job object (Confirmed);
  it passed launch 1's death point, the 50k checkpoint and every curriculum boundary, and finished.

## 3. Monitoring lessons (Confirmed)

- **Reading `status.json` of a live run can crash it on Windows.** The trainer replaces status / meta / summary
  files with `os.replace`, which fails with `PermissionError` (WinError 5) while another process has the target open:
  19% of replaces failed under a PowerShell `Get-Content` loop, 65% under Python `open().read()`, 78% even with
  `FileShare ReadWrite+Delete` (scratch test). Safe-read rule while a run is alive: read only `logs/rollouts_*.csv`,
  directory listings and process info.
- The console log written by Windows PowerShell 5.1 `*>` is UTF-16; read it with `Get-Content`, not grep.
- A CSV's NTFS LastWriteTime lags 65–91 s while the trainer holds it open, so a heartbeat must count new rows.

## 4. Open items (Proposed)

- **Code fix + commit:** retry `os.replace` on `PermissionError` for a few hundred ms in `run_files.atomic_*`.
  Not done during the run, because a modified tracked file would have blocked a resume (D12).
- **M7:** final model as `pid_ppo` and the team SAC as `pid_sac` on eval seeds 300000–300029 with the frozen protocol
  v1 (`report` mode), plus the secondary learning curve (150k / 400k / 800k / final on validation seeds 400000–400009,
  E6, D10, D11). The checkpoints labelled 150k / 400k / 800k were trained on 149,504 / 399,360 / 798,720 decisions.
