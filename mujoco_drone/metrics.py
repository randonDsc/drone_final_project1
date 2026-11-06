"""Time-indexed tracking, geometric cross track, and sustained gust recovery."""
import numpy as np

from drone_code.path_reference import distances_to_path
from drone_code import path_sac_config


METRIC_VERSION = 1


def recovery_time(times, errors, velocity_errors, gust_end, dwell=1.0, radius=.15, speed=.2):
    """Delay from gust end to start of the first confirmed sustained recovery.

    None means insufficient post-gust horizon or no such recovery. A flight
    already inside both bands at gust end has zero delay once dwell is proven.
    """
    start = None
    for time, error, velocity in zip(times, errors, velocity_errors):
        if time < gust_end-1e-8:
            continue
        if error <= radius and velocity <= speed:
            if start is None:
                start = max(float(time), gust_end)
            if time-start >= dwell-1e-8:
                return round(max(0.0, start-gust_end), 10)
        else:
            start = None
    return None


class FlightMetrics:
    def __init__(self):
        self.times = []
        self.positions = []
        self.reference_positions = []
        self.velocity_errors = []
        self.saturated_steps = 0
        self.saturated_motors = 0
        self.max_tilt = 0.0

    def record(self, env, state, sample):
        self.times.append(float(env.sim.data.time))
        self.positions.append(state["position"].copy())
        self.reference_positions.append(sample.position.copy())
        self.velocity_errors.append(float(np.linalg.norm(state["linear_velocity"]-sample.velocity)))
        self.saturated_steps += int(env.sim.last["saturated"])
        self.saturated_motors += int(np.count_nonzero(np.abs(env.sim.last["requested"]-env.sim.last["controls"]) > 1e-10))
        R = env.sim.data.xmat[env.sim.body_id].reshape(3, 3)
        self.max_tilt = max(self.max_tilt, float(np.rad2deg(np.arccos(np.clip(R[2, 2], -1, 1)))))

    def summarize(self, env):
        if not self.times:
            raise ValueError("No valid physics samples; no flight metrics can be computed")
        positions = np.asarray(self.positions)
        delta = positions-np.asarray(self.reference_positions)
        errors = np.linalg.norm(delta, axis=1)
        cross = distances_to_path(positions, env.core.reference.path)
        failure = env.failure
        completed = (env.core.completion.completed and failure is None
                     and env.core.completion.settled >= path_sac_config.COMPLETION["dwell"]
                     and env.core.time >= env.core.reference.motion_end)
        wind = env.config.wind
        recovery = None
        if wind.scenario == "gust":
            recovery = recovery_time(self.times, errors, self.velocity_errors,
                                     wind.gust_start+wind.gust_duration)
        return dict(metric_version=METRIC_VERSION, seconds=self.times[-1], samples=len(self.times),
                    position_rmse=float(np.sqrt(np.mean(errors**2))),
                    cross_track_rmse=float(np.sqrt(np.mean(cross**2))),
                    cross_track_max=float(np.max(cross)), max_deviation=float(np.max(errors)),
                    altitude_rmse=float(np.sqrt(np.mean(delta[:, 2]**2))),
                    altitude_max_error=float(np.max(np.abs(delta[:, 2]))),
                    final_position_error=float(errors[-1]),
                    completion_time=env.core.completion.completion_time,
                    completed=bool(completed), failed=failure is not None,
                    failure_reason=failure, crashed=failure == "collision",
                    saturation_rate=self.saturated_steps/len(self.times),
                    motor_saturation_rate=self.saturated_motors/(4*len(self.times)),
                    maximum_tilt_degrees=self.max_tilt, gust_recovery_seconds=recovery,
                    gust_recovery_observed=recovery is not None if wind.scenario == "gust" else None,
                    episode_return=env.total_reward)


def aggregate(rows):
    """Include failures; never silently select only successful flights."""
    if not rows:
        raise ValueError("Cannot aggregate an empty experiment")
    result = dict(flights=len(rows), completed=sum(bool(r["completed"]) for r in rows),
                  completion_rate=float(np.mean([r["completed"] for r in rows])),
                  failure_rate=float(np.mean([r["failed"] for r in rows])),
                  crash_rate=float(np.mean([r["crashed"] for r in rows])))
    for key in ("position_rmse", "cross_track_rmse", "max_deviation", "altitude_rmse", "saturation_rate", "motor_saturation_rate"):
        values = np.array([r[key] for r in rows], dtype=float)
        result[key+"_mean"] = float(np.mean(values))
        result[key+"_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    recovered = [r["gust_recovery_seconds"] for r in rows if r["gust_recovery_seconds"] is not None]
    applicable = [r for r in rows if r["gust_recovery_observed"] is not None]
    result["gust_recovery_rate"] = len(recovered)/len(applicable) if applicable else None
    result["gust_recovery_seconds_mean"] = float(np.mean(recovered)) if recovered else None
    result["gust_recovery_unobserved_flights"] = len(applicable)-len(recovered)
    return result
