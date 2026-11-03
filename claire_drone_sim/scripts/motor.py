import numpy as np


class MotorModel:
    def __init__(self, thrust_coefficient, torque_coefficient,
                 max_speed, reaction_signs):
        self.kf = thrust_coefficient
        self.km = torque_coefficient
        self.max_speed = max_speed  # rad/s
        self.reaction_signs = np.asarray(reaction_signs, dtype=float)

    def calculate(self, speeds):
        """Four rotor speeds in rad/s → thrusts in N and torques in N·m."""
        speeds = np.asarray(speeds, dtype=float)

        if speeds.shape != (4,):
            raise ValueError("Provide exactly four rotor speeds.")

        speeds = np.clip(speeds, 0.0, self.max_speed)

        thrusts = self.kf * speeds**2
        reaction_torques = self.reaction_signs * self.km * speeds**2

        return thrusts, reaction_torques