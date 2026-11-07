import numpy as np

class WeatherEnvironment:
    """
    Generates external disturbance forces for a MuJoCo drone.

    All forces are in world coordinates, measured in newtons.
    """

    def __init__(self, scenario="calm", seed=42):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)

        # Stateful turbulence updated once per physics step.
        self.turbulence = np.zeros(3)

    def get_force(self, t, dt=0.01):

        if self.scenario == "calm":
            return np.zeros(3)

        elif self.scenario == "constant_wind":
            return np.array([0.0, 0.03, 0.0])

        elif self.scenario == "gust":
            if 2.0 <= t < 4.0:
                return np.array([0.0, 0.10, 0.0])
            return np.zeros(3)

        elif self.scenario == "periodic_wind":
            return np.array([
                0.0,
                0.05 * np.sin(2 * np.pi * 0.5 * t),
                0.0
            ])

        elif self.scenario == "turbulence":
            # Correlated random force, rather than unrelated
            # random values at every timestep.
            tau = 0.5
            sigma = 0.05

            decay = np.exp(-dt / tau)
            noise_scale = sigma * np.sqrt(1 - decay**2)

            self.turbulence = (
                decay * self.turbulence
                + noise_scale * self.rng.normal(size=3)
            )

            return self.turbulence.copy()

        else:
            raise ValueError(
                f"Unknown weather scenario: {self.scenario}"
            )



