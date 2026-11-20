"""Path flight failure limits: outcome-based, plus sustained loss of authority."""
import unittest

import numpy as np

import path_sac_config
from path_sac_policy import PathCore


def state(tilt_degrees=0.0, speed=0.0):
    half = np.deg2rad(tilt_degrees) / 2
    return {
        "position": np.array([0.0, 0.0, 2.0]),
        "orientation": np.array([np.cos(half), np.sin(half), 0.0, 0.0]),
        "linear_velocity": np.array([speed, 0.0, 0.0]),
        "angular_velocity": np.zeros(3),
    }


def core(lost_seconds=0.0):
    c = PathCore.__new__(PathCore)
    c.authority_lost_seconds = lost_seconds
    return c


TARGET = np.array([0.0, 0.0, 2.0])
LIMITS = path_sac_config.FAILURE


class FailureLimitsTest(unittest.TestCase):
    def test_tilt_up_to_the_controller_cap_is_not_a_failure(self):
        self.assertIsNone(core().failure_reason(state(45.0), TARGET, LIMITS))
        self.assertIsNone(core().failure_reason(state(58.0), TARGET, LIMITS))
        self.assertEqual(
            core().failure_reason(state(62.0), TARGET, LIMITS), "excessive_tilt"
        )

    def test_no_fixed_speed_limit(self):
        self.assertNotIn("max_speed", LIMITS)
        self.assertIsNone(core().failure_reason(state(speed=12.0), TARGET, LIMITS))

    def test_authority_loss_must_be_sustained(self):
        limit = LIMITS["max_authority_loss_seconds"]
        self.assertIsNone(core(limit - 0.01).failure_reason(state(), TARGET, LIMITS))
        self.assertEqual(
            core(limit + 0.01).failure_reason(state(), TARGET, LIMITS),
            "lost_authority",
        )

    def test_soft_tilt_above_holding_tilt_in_strong_wind(self):
        self.assertGreaterEqual(path_sac_config.REWARD["soft_tilt_degrees"], 30.0)


if __name__ == "__main__":
    unittest.main()
