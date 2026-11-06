"""Wind speeds in m/s; reuse the teammate's distributed OU/drag algorithm."""
import numpy as np

from drone_code.wind import GustingWind


class ScenarioWind(GustingWind):
    def __init__(self, config, rotor_positions, seed):
        self.config = config
        self.scenario = config.scenario
        self.requested_mean = np.array(config.mean_velocity, dtype=float)
        if self.scenario == "randomized":
            rng = np.random.default_rng(seed)
            speed = rng.uniform(*config.speed_range)
            direction = rng.uniform(-np.pi, np.pi)
            self.requested_mean = speed * np.array([np.cos(direction), np.sin(direction), 0])
        noisy = self.scenario in ("turbulence", "randomized", "curriculum", "training_range", "unseen_gusts", "outside_range")
        points = [("hull", (0, 0, 0.04), 0.4), ("lower_hull", (0, 0, 0.0), 0.1)]
        points += [(f"rotor{i+1}", tuple(p), 0.125) for i, p in enumerate(rotor_positions)]
        super().__init__(
            mean_wind=self.requested_mean,
            gust_std=np.array(config.gust_std) if noisy else np.zeros(3),
            local_gust_std=np.array(config.local_gust_std) if noisy else np.zeros(3),
            gust_timescale=config.gust_timescale,
            local_gust_timescale=config.local_gust_timescale,
            correlation_length=config.correlation_length,
            air_density=config.air_density,
            drag_coefficient=config.drag_coefficient,
            effective_area=config.effective_area,
            sample_points=points, seed=seed, enabled=config.enabled,
        )
        self.mean_wind = self.programmed_mean(0.0)
        self.point_winds = self._compose()

    def programmed_mean(self, time):
        cfg = self.config
        if self.scenario == "curriculum":
            return self.mean_wind.copy()
        if self.scenario == "calm":
            return np.zeros(3)
        if self.scenario == "gust":
            active = cfg.gust_start <= time < cfg.gust_start + cfg.gust_duration
            return np.array(cfg.gust_velocity, dtype=float) if active else np.zeros(3)
        if self.scenario == "periodic":
            return np.array(cfg.gust_velocity) * np.sin(2 * np.pi * cfg.frequency_hz * time)
        angle = cfg.direction_rate * time
        c, s = np.cos(angle), np.sin(angle)
        x, y, z = self.requested_mean
        return np.array([c*x - s*y, s*x + c*y, z])

    def step(self, dt):
        self.mean_wind = self.programmed_mean(self.time)
        result = super().step(dt)
        if not self.enabled:
            self.point_winds = np.zeros_like(result)
        return self.point_winds

    def reset_episode(self, seed, mean_wind, gust_scale=1.0):
        self.scenario = "curriculum"
        super().reset_episode(seed, mean_wind, gust_scale)
