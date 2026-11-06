"""Hover-oriented cascaded PID; world Z-up, quaternion wxyz, body torque in N m.

Position P -> velocity PID -> Euler attitude P -> body-rate PI.
Euler attitude control is intended for modest tilts, not inverted recovery.
No simulator imports: the caller supplies ground-truth or estimated state.
"""

import math
from dataclasses import asdict, dataclass

import torch


@dataclass
class CascadedPIDCfg:
    position_p: float = 1.2
    velocity_p: float = 2.5
    velocity_i: float = 0.3
    velocity_d: float = 0.1
    attitude_p: float = 5.0
    rate_p: tuple = (0.08, 0.08, 0.05)
    rate_i: tuple = (0.01, 0.01, 0.005)
    velocity_limit: float = 2.0
    rate_limit: float = 2.0
    tilt_limit: float = math.pi / 6
    integral_limit: float = 1.0
    derivative_filter_s: float = 0.05

    def to_dict(self):
        return asdict(self)


class CascadedPID:
    def __init__(self, mass, gravity, dt, cfg=None):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.mass, self.gravity, self.dt = mass, gravity, dt
        self.cfg = cfg or CascadedPIDCfg()
        self.reset()

    def reset(self):
        self.velocity_integral = torch.zeros_like(self.gravity)
        self.rate_integral = torch.zeros_like(self.gravity)
        self.acceleration = torch.zeros_like(self.gravity)
        self.previous_velocity = None
        self.saturated = torch.zeros_like(self.mass, dtype=torch.bool)

    def allocation_feedback(self, raw, commanded):
        """Freeze integrators on the next step if allocation clipped any motor."""
        self.saturated = (raw - commanded).abs().amax(dim=-1) > 1e-6

    def compute(self, target, position, velocity, quaternion, angular_velocity):
        c = self.cfg
        w, x, y, z = quaternion.unbind(-1)
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        velocity_target = (c.position_p * (target[:, :3] - position)).clamp(-c.velocity_limit, c.velocity_limit)
        error = velocity_target - velocity
        if self.previous_velocity is not None:
            alpha = self.dt / (c.derivative_filter_s + self.dt)
            self.acceleration.lerp_((velocity - self.previous_velocity) / self.dt, alpha)
        self.previous_velocity = velocity.clone()
        integrate = (~self.saturated).unsqueeze(-1)
        self.velocity_integral = (self.velocity_integral + integrate * error * self.dt).clamp(
            -c.integral_limit, c.integral_limit
        )
        acceleration = c.velocity_p * error + c.velocity_i * self.velocity_integral - c.velocity_d * self.acceleration
        force = self.mass[:, None] * (acceleration - self.gravity)
        # Constrain the desired thrust direction to the configured tilt cone.
        force[:, 2].clamp_(min=0.1)
        xy_norm = force[:, :2].norm(dim=-1).clamp_min(1e-6)
        scale = (force[:, 2] * math.tan(c.tilt_limit) / xy_norm).clamp(max=1)
        force[:, :2] *= scale[:, None]
        heading = target[:, 3]
        magnitude = force.norm(dim=-1).clamp_min(1e-6)
        desired_roll = torch.asin(
            ((force[:, 0] * heading.sin() - force[:, 1] * heading.cos()) / magnitude).clamp(-1, 1)
        )
        desired_pitch = torch.atan2(force[:, 0] * heading.cos() + force[:, 1] * heading.sin(), force[:, 2])
        angle_error = torch.stack((desired_roll - roll, desired_pitch - pitch, heading - yaw), dim=-1)
        angle_error = torch.atan2(angle_error.sin(), angle_error.cos())
        euler_rates = c.attitude_p * angle_error
        # Transform Euler rates to body rates for the current attitude.
        rates = torch.stack(
            (
                euler_rates[:, 0] - pitch.sin() * euler_rates[:, 2],
                roll.cos() * euler_rates[:, 1] + roll.sin() * pitch.cos() * euler_rates[:, 2],
                -roll.sin() * euler_rates[:, 1] + roll.cos() * pitch.cos() * euler_rates[:, 2],
            ),
            dim=-1,
        ).clamp(-c.rate_limit, c.rate_limit)
        rate_error = rates - angular_velocity
        self.rate_integral = (self.rate_integral + integrate * rate_error * self.dt).clamp(
            -c.integral_limit, c.integral_limit
        )
        torque = rate_error * force.new_tensor(c.rate_p) + self.rate_integral * force.new_tensor(c.rate_i)
        body_z = torch.stack((2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)), dim=-1)
        wrench = force.new_zeros((force.shape[0], 6))
        wrench[:, 2] = (force * body_z).sum(-1).clamp_min(0)
        wrench[:, 3:] = torque
        return wrench
