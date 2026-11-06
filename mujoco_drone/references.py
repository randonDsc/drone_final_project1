"""Differentiable controlled tasks; the legacy Isaac references stay available."""
import numpy as np

from drone_code.path_reference import ReferenceSample


def smooth_phase(u):
    """Quintic progress and its first two derivatives on the unit interval."""
    u = np.clip(u, 0.0, 1.0)
    return (10*u**3 - 15*u**4 + 6*u**5,
            30*u**2 - 60*u**3 + 30*u**4,
            60*u - 180*u**2 + 120*u**3)


class TaskReference:
    def __init__(self, config):
        self.config = config
        self.start = np.array(config.start, dtype=float)
        self.kind = config.kind
        self.hold = config.start_hold
        if self.kind == "hover":
            self.motion_end = 0.0
            self.duration = config.hover_seconds
            self.path = self.start[None, :]
            self.final_position = self.start.copy()
            return
        if self.kind in ("straight", "waypoints"):
            points = np.array([config.start, config.goal] if self.kind == "straight" else config.waypoints, dtype=float)
            keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
            self.points = points[keep]
            self.start = self.points[0]
            lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
            # Quintic max first/second derivatives: 1.875 and ~5.774.
            self.segment_durations = np.maximum(1.875*lengths/config.speed,
                                               np.sqrt(5.774*lengths/config.acceleration))
            self.times = self.hold + np.r_[0.0, np.cumsum(self.segment_durations)]
            self.motion_end = float(self.times[-1])
            self.path = self.points.copy()
            self.final_position = self.points[-1].copy()
        else:
            # Conservative derivative bounds for both analytic loop shapes.
            length_bound = 2*np.pi*config.radius*(1.5 if self.kind == "figure_eight" else 1)
            self.motion_seconds = max(1.875*length_bound/config.speed,
                                      np.sqrt((5.774*length_bound + 2*(1.875*length_bound)**2/config.radius)/config.acceleration))
            self.motion_end = self.hold + self.motion_seconds
            self.final_position = self.start.copy()
            self.path = np.array([self._loop(theta)[0] for theta in np.linspace(0, 2*np.pi, 501)])
        self.duration = self.motion_end + config.end_hold

    def _loop(self, theta):
        r = self.config.radius
        if self.kind == "circle":
            return (self.start + r*np.array([np.cos(theta)-1, np.sin(theta), 0]),
                    r*np.array([-np.sin(theta), np.cos(theta), 0]),
                    r*np.array([-np.cos(theta), -np.sin(theta), 0]))
        return (self.start + r*np.array([np.sin(theta), 0.5*np.sin(2*theta), 0]),
                r*np.array([np.cos(theta), np.cos(2*theta), 0]),
                r*np.array([-np.sin(theta), -2*np.sin(2*theta), 0]))

    def sample(self, time):
        if self.kind == "hover" or time <= self.hold or time >= self.motion_end:
            point = self.final_position if time >= self.motion_end else self.start
            return ReferenceSample(point.copy(), np.zeros(3), np.zeros(3), time >= self.motion_end)
        if self.kind in ("straight", "waypoints"):
            i = int(np.searchsorted(self.times, time, side="right") - 1)
            duration = self.segment_durations[i]
            p, v, a = smooth_phase((time-self.times[i])/duration)
            delta = self.points[i+1]-self.points[i]
            return ReferenceSample(self.points[i]+p*delta, v*delta/duration,
                                   a*delta/duration**2, False)
        p, v, a = smooth_phase((time-self.hold)/self.motion_seconds)
        theta = 2*np.pi*p
        omega = 2*np.pi*v/self.motion_seconds
        alpha = 2*np.pi*a/self.motion_seconds**2
        position, derivative, second = self._loop(theta)
        return ReferenceSample(position, derivative*omega, second*omega**2+derivative*alpha, False)


def make_reference(config):
    return TaskReference(config)
