"""Static motor model: angular speed -> thrust and reaction torque.

This preserves the uploaded model; it does not simulate motor lag, propeller
flow, battery voltage, or aerodynamic drag. Physics integration lives in Sim.
"""
import numpy as np


class MotorModel:
    def __init__(self, thrust_coefficient, torque_coefficient,
                 max_speed, reaction_signs):
        # kf [N/(rad/s)^2] scales angular speed squared into upward thrust.
        # km [N*m/(rad/s)^2] scales it into rotor drag/reaction torque.
        self.kf = float(thrust_coefficient)
        self.km = float(torque_coefficient)
        self.max_speed = float(max_speed)  # rad/s, not RPM
        # A rotor's reaction sign determines its contribution to body yaw.
        # Ordering must match rotor_positions in motor_physics.py.
        self.reaction_signs = np.asarray(reaction_signs, dtype=float)
        if (self.kf <= 0 or self.km < 0 or self.max_speed <= 0 or
                self.reaction_signs.shape != (4,) or
                not np.all(np.isin(self.reaction_signs, [-1, 1]))):
            raise ValueError('Invalid motor coefficients, speed limit, or reaction signs.')

    def calculate(self, speeds):
        """Four rotor speeds in rad/s -> thrusts in N and torques in N*m."""
        speeds = np.asarray(speeds, dtype=float)
        if speeds.shape != (4,) or not np.all(np.isfinite(speeds)):
            raise ValueError('Provide exactly four finite rotor speeds.')
        speeds = np.clip(speeds, 0.0, self.max_speed)
        # The speed magnitude is nonnegative; the separate reaction_signs array
        # handles CW/CCW. A negative requested speed is stopped, not reversed.
        thrusts = self.kf * speeds**2
        reaction_torques = self.reaction_signs * self.km * speeds**2
        return thrusts, reaction_torques
