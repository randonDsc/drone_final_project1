
"""Simple altitude PD controller for the ARL drone."""

class HoverController:
    def __init__(
        self,
        mass=1.24,
        gravity=9.81,
        kp=3.0,
        kd=2.0,
        min_thrust=0.0,
        max_thrust=6.0,
    ):
        self.mass = mass
        self.gravity = gravity
        self.kp = kp
        self.kd = kd

        self.min_thrust = min_thrust
        self.max_thrust = max_thrust

        # Theoretical hover force per motor (N)
        self.hover_thrust = mass * gravity / 4.0

    def compute(self, current_z, target_z, velocity_z):
        """Return four equal motor thrust commands in Newtons."""

        # Distance from desired altitude
        altitude_error = target_z - current_z

        # Correct altitude and vertical movement
        correction = (
            self.kp * altitude_error
            - self.kd * velocity_z
        )

        # Distribute collective thrust over four motors
        motor_thrust = self.hover_thrust + correction / 4.0

        # Respect motor command limits
        motor_thrust = max(
            self.min_thrust,
            min(self.max_thrust, motor_thrust),
        )

        return [motor_thrust] * 4
