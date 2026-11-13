"""Deterministic mean-wind steps and event metrics; no simulator imports."""
from dataclasses import dataclass
import math
import numpy as np

# Horizontal airflow vectors are velocities toward a direction, not meteorological bearings.
SCENARIOS = {
    "control": (4.0, 4.0, 0.0),
    "turn90": (4.0, 4.0, 90.0),
    "turn180": (4.0, 4.0, 180.0),
    "speed_up": (2.0, 6.0, 0.0),
    "speed_down": (6.0, 2.0, 0.0),
}

@dataclass(frozen=True)
class WindEvent:
    scenario: str
    heading: float
    event_s: float = 5.0

    def __post_init__(self):
        if self.scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {self.scenario}")
        if not math.isfinite(self.heading) or not math.isfinite(self.event_s) or self.event_s <= 0:
            raise ValueError("Heading must be finite and event time positive")

    def at(self, time):
        before, after, turn = SCENARIOS[self.scenario]
        changed = time + 1e-9 >= self.event_s
        angle = self.heading + (math.radians(turn) if changed else 0.0)
        speed = after if changed else before
        return np.array([speed * math.cos(angle), speed * math.sin(angle), 0.0])


def event_metrics(times, errors, event_s, failed=False, threshold=0.1, dwell_s=1.0):
    """Decision-rate estimates. Recovery starts after the largest post-event error.

    Null recovery means censored/not recovered, never a zero-second recovery.
    Failed flights cannot count as recovered even if they crossed the threshold.
    """
    t, e = np.asarray(times, float), np.asarray(errors, float)
    if t.ndim != 1 or t.shape != e.shape or len(t) == 0 or not np.all(np.isfinite(t)):
        raise ValueError("Expected nonempty matching time/error vectors")
    if np.any(np.diff(t) <= 0) or threshold <= 0 or dwell_s <= 0:
        raise ValueError("Times must increase and threshold/dwell must be positive")
    finite = np.isfinite(e)
    failed = bool(failed or not finite.all())
    post = np.flatnonzero(t > event_s + 1e-9)
    usable = post[finite[post]]
    peak = int(usable[np.argmax(e[usable])]) if len(usable) else None
    pre = e[t <= event_s + 1e-9]
    recovery = None
    start = None
    if peak is not None and not failed:
        for i in range(peak, len(t)):
            if e[i] < threshold:
                start = i if start is None else start
                if t[i] - t[start] >= dwell_s - 1e-9:
                    recovery = float(t[start] - event_s)
                    break
            else:
                start = None
    return {
        "failed": failed,
        "invalid_samples": int((~finite).sum()),
        "event_reached": bool(len(post)),
        "pre_event_rmse_m": float(np.sqrt(np.mean(pre**2))) if len(pre) and np.isfinite(pre).all() else None,
        "position_rmse_m": float(np.sqrt(np.mean(e**2))) if finite.all() else None,
        "post_event_peak_m": float(e[peak]) if peak is not None else None,
        "peak_time_after_event_s": float(t[peak] - event_s) if peak is not None else None,
        "recovered": recovery is not None,
        "recovery_time_s": recovery,
        "observed_post_event_s": max(0.0, float(t[-1] - event_s)),
        "recovery_threshold_m": threshold,
        "recovery_dwell_s": dwell_s,
    }


def trajectory_event_time(reference, duration, requested=None):
    """Choose a time during reference motion, leaving >=2 seconds of follow-up.

    Automatic timing is 35% into the available motion after the 0.5 s launch hold.
    This evaluates a fixed-duration tracking segment, not full-route completion.
    """
    end = min(float(reference.motion_end), duration - 2.0)
    event = 0.5 + 0.35 * (end - 0.5) if requested is None else float(requested)
    if not math.isfinite(event) or event <= 0 or event > duration - 2:
        raise ValueError("Event must be positive with at least 2 seconds of follow-up")
    if np.linalg.norm(reference.sample(event).velocity) <= 0.05:
        raise ValueError("Wind event would occur while the reference is stationary; choose another event time")
    return event
