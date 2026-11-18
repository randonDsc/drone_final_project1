# M9 summary - descriptive demo, not eval-v1

no controller perceives obstacles; crosswind toward obstacles (worst side); indoor wind = artificial uniform disturbance (no sheltering). One flight per cell (n = 1): no CIs, no tests.

| Scene | Track | Wind | RMSE PID / PID+PPO / direct (m) | Max dev PID / PID+PPO / direct (m) | Completed P/R/D | Failures | Contacts | Min clearance P/R/D (m) |
|---|---|---|---|---|---|---|---|---|
| indoor | I1_aisle | calm (0 m/s) | 0.100 / 0.045 / 0.098 | 0.19 / 0.10 / 0.16 | Y/Y/Y | none | none | 3.60 / 3.62 / 3.58 |
| indoor | I1_aisle | 3.5 m/s | 0.285 / 0.071 / 0.130 | 0.51 / 0.13 / 0.19 | n/Y/Y | none | none | 3.37 / 3.58 / 3.53 |
| indoor | I1_aisle | 7 m/s | 0.995 / 0.164 / 0.260 | 1.35 / 0.23 / 0.34 | n/Y/n | none | none | 2.38 / 3.44 / 3.32 |
| indoor | I2_uturn | calm (0 m/s) | 0.180 / 0.055 / 0.119 | 0.40 / 0.13 / 0.25 | Y/Y/Y | none | none | 3.46 / 3.58 / 3.55 |
| indoor | I2_uturn | 3.5 m/s | 0.326 / 0.079 / 0.163 | 0.79 / 0.20 / 0.31 | n/Y/Y | none | none | 3.45 / 3.54 / 3.49 |
| indoor | I2_uturn | 7 m/s | 0.804 / 0.133 / 0.295 | 1.59 / 0.29 / 0.60 | Y/Y/Y | none | none | 2.99 / 3.48 / 3.49 |
| indoor | I3_climb | calm (0 m/s) | 0.239 / 0.062 / 0.121 | 0.62 / 0.19 / 0.27 | Y/Y/Y | none | none | 2.06 / 2.15 / 2.16 |
| indoor | I3_climb | 3.5 m/s | 0.237 / 0.062 / 0.103 | 0.39 / 0.14 / 0.26 | Y/Y/Y | none | none | 1.99 / 2.11 / 2.10 |
| indoor | I3_climb | 7 m/s | 0.663 / 0.136 / 0.178 | 1.03 / 0.29 / 0.32 | Y/Y/Y | none | none | 1.38 / 2.06 / 1.95 |
| forest | F1_north_edge | calm (0 m/s) | 0.118 / 0.046 / 0.125 | 0.29 / 0.15 / 0.20 | Y/Y/Y | none | none | 3.33 / 3.30 / 3.36 |
| forest | F1_north_edge | 3.5 m/s | 0.163 / 0.051 / 0.131 | 0.33 / 0.15 / 0.22 | Y/Y/Y | none | none | 3.09 / 3.31 / 3.33 |
| forest | F1_north_edge | 7 m/s | 0.957 / 0.139 / 0.220 | 1.23 / 0.20 / 0.31 | n/Y/n | none | none | 2.55 / 3.17 / 3.16 |
| forest | F2_sw_corner | calm (0 m/s) | 0.173 / 0.051 / 0.101 | 0.38 / 0.11 / 0.17 | Y/Y/Y | none | none | 2.60 / 2.63 / 2.64 |
| forest | F2_sw_corner | 3.5 m/s | 0.225 / 0.065 / 0.098 | 0.52 / 0.14 / 0.22 | Y/Y/Y | none | none | 2.63 / 2.64 / 2.64 |
| forest | F2_sw_corner | 7 m/s | 0.729 / 0.129 / 0.162 | 1.11 / 0.28 / 0.34 | n/Y/Y | none | none | 2.52 / 2.63 / 2.62 |
| forest | F3_weave | calm (0 m/s) | 0.243 / 0.055 / 0.135 | 0.57 / 0.11 / 0.25 | n/Y/Y | none | none | 1.13 / 1.18 / 1.09 |
| forest | F3_weave | 3.5 m/s | 0.254 / 0.073 / 0.136 | 0.44 / 0.18 / 0.25 | n/Y/Y | none | none | 1.12 / 1.20 / 1.10 |
| forest | F3_weave | 7 m/s | 0.704 / 0.155 / 0.269 | 1.16 / 0.25 / 0.54 | n/Y/Y | none | none | 0.90 / 1.17 / 1.08 |

Mean RMSE over the 3 tracks of a scene (m):

| Scene | Wind | PID | PID + PPO residual | PPO direct |
|---|---|---|---|---|
| indoor | calm (0 m/s) | 0.173 | 0.054 | 0.113 |
| indoor | 3.5 m/s | 0.283 | 0.071 | 0.132 |
| indoor | 7 m/s | 0.821 | 0.144 | 0.245 |
| forest | calm (0 m/s) | 0.178 | 0.051 | 0.120 |
| forest | 3.5 m/s | 0.214 | 0.063 | 0.122 |
| forest | 7 m/s | 0.796 | 0.141 | 0.217 |
