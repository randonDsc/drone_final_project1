# Measured validation artifacts

Read [the results report](../../docs/MUJOCO_RESULTS.md) for interpretation and
limitations. `pid_*` covers the 90-flight environmental matrix; `paired_*` covers
45 PID/PID+SAC pairs; `stress_*` covers six stronger-gust pairs. `*_deltas.csv`
is right (PID+SAC) minus left (PID). Failures are included in the schema.

`sac_training.json` records actual decisions/gradient updates. `sac_run.json`
records software, settings and source hashes. `original_pid_*` records the
separate historical-controller diagnostic. `ARTIFACTS.json` connects copied
review files to their original measured output paths and hashes.

The full flight CSVs, individual metadata, reference samples, ideal sensor
arrays and checkpoint/replay files remain under `mujoco_output/`, ignored by
Git. The files here are small review copies; no Isaac results were fabricated.
