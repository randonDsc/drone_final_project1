"""MuJoCo specialization of the existing PID + held SAC residual flight loop."""
from collections import deque

import numpy as np

from drone_code import hover_task, path_sac_config
from drone_code.path_sac_policy import PathCore, CompletionTracker
from .controllers import make_controller


class FlightCore(PathCore):
    def __init__(self, sim):
        self.sim = sim
        self.decimation = int(round(0.1 / sim.dt))
        self.limits = hover_task.residual_limits_array(path_sac_config.RESIDUAL_LIMITS)
        # This matrix is ALREADY about the COM. No additional origin shift.
        original_bound = np.abs(np.linalg.inv(sim.allocation_matrix)) @ self.limits
        self.residual_budget = (path_sac_config.ROTOR_CORRECTION_FRACTION
                                * path_sac_config.ROTOR_LIMIT_MASS_FACTOR
                                * sim.total_mass * hover_task.GRAVITY)
        self.residual_scale = min(1.0, self.residual_budget / original_bound.max())
        self.limits *= self.residual_scale
        self.rotor_residual_bound = np.abs(np.linalg.inv(sim.allocation_matrix)) @ self.limits
        self.hover_thrust = sim.total_mass*hover_task.GRAVITY
        self.pid = make_controller(sim)
        self.frames = deque(maxlen=path_sac_config.HISTORY_LENGTH)
        self.completion = CompletionTracker()
        self.base_gust_timescale = sim.wind.gust_timescale
        self.base_local_gust_timescale = sim.wind.local_gust_timescale
        self.reference = None
        self.clear()
