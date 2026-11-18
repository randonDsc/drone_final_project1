"""M9 visual demo (descriptive, not eval-v1): offline checks. NumPy only; no Isaac, no SB3.

Run from residual_ppo:  python -m unittest discover -s tests -p "test_*.py"
"""
import sys
import unittest
from pathlib import Path

import numpy as np

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESIDUAL_DIR))

from demo import demo_config, tracks  # noqa: E402
from demo.geometry import Obstacles  # noqa: E402

GEOMETRY = {s: RESIDUAL_DIR / "results" / "m9_demo" / f"geometry_{s}.json" for s in ("indoor", "forest")}
RESIDUAL_MODEL = RESIDUAL_DIR / "runs" / "ppo_path" / "ppo_s0_1p5m" / "final_model.zip"
DIRECT_MODEL = RESIDUAL_DIR / "runs" / "ppo_direct" / "ppo_direct_s0_1p5m" / "final_model.zip"


class SeedsAndWind(unittest.TestCase):
    def test_seed_guard(self):
        for ok in (600000, 600017, 600900):
            self.assertEqual(demo_config.assert_demo_seed(ok), ok)
        for bad in (0, 300000, 300029, 400000, 400009, 599999):
            with self.assertRaises(ValueError):
                demo_config.assert_demo_seed(bad)

    def test_wind_levels(self):
        calm = demo_config.demo_wind("calm", (0.0, 1.0))
        self.assertEqual(calm["gust_scale"], 0.0)                       # truly calm, as still_air
        np.testing.assert_array_equal(calm["mean_wind"], np.zeros(3))
        w7 = demo_config.demo_wind("w7", (3.0, 4.0))
        np.testing.assert_allclose(w7["mean_wind"], [4.2, 5.6, 0.0])
        self.assertEqual((w7["gust_scale"], w7["direction_rate"]), (0.75, 0.0))
        self.assertEqual(len({demo_config.WIND_LEVELS[k]["seed"] for k in demo_config.WIND_LEVELS}), 3)

    def test_profiles_registered_in_memory_only(self):
        import path_sac_config

        names = demo_config.register_demo_profiles()
        self.assertEqual(names, ["m9_calm", "m9_w3p5", "m9_w7"])
        self.assertEqual(path_sac_config.EVAL_WIND_PROFILES["m9_w7"]["mean_speed_range"], (7.0, 7.0))


class Geometry(unittest.TestCase):
    def test_exact_distances_and_contacts(self):
        r = np.sqrt(0.5)
        shapes = [
            {"path": "box", "kind": "box", "center": [0, 0, 1], "axes": [[r, r, 0], [-r, r, 0], [0, 0, 1]],
             "half": [1, 0.5, 1], "aabb": [[-1.1, -1.1, 0], [1.1, 1.1, 2]], "rigid_body": False},
            {"path": "cyl", "kind": "cylinder", "center": [10, 0, 2], "axis": [0, 0, 1], "half_height": 2,
             "radius": 0.2, "aabb": [[9.8, -0.2, 0], [10.2, 0.2, 4]], "rigid_body": False},
            {"path": "sph", "kind": "sphere", "center": [0, 10, 5], "radius": 2, "aabb": [[-2, 8, 3], [2, 12, 7]],
             "rigid_body": False},
        ]
        o = Obstacles(shapes)
        d, who = o.clearance([[0, 0, 3.5], [10, 1.2, 2], [10, 0, 5], [0, 10, 8], [0, 10, 5]])
        np.testing.assert_allclose(d, [1.5, 1.0, 1.0, 1.0, 0.0], atol=1e-12)
        hits = o.contacts([[10, 0.44, 2], [10, 0.46, 2], [0, 7.76, 5], [0, 7.74, 5]], demo_config.COLLISION_HALF_SIZE)
        self.assertEqual(hits.tolist(), [1, -1, 2, -1])

    @unittest.skipUnless(all(p.is_file() for p in GEOMETRY.values()), "geometry files not exported yet")
    def test_tracks_meet_requirements(self):
        for scene, path in GEOMETRY.items():
            checked = tracks.check_tracks(scene, Obstacles.load(path))
            self.assertEqual(len(checked), 3)
            for name, t in checked.items():
                self.assertTrue(t["clearance_ok"], f"{name}: {t['min_clearance_m']}")
                self.assertEqual(t["in_distribution_flags"], [], name)
                self.assertLessEqual(t["max_ref_speed_mps"], demo_config.TRACK_LIMITS["speed"] + 1e-9)


@unittest.skipUnless(RESIDUAL_MODEL.is_file() and DIRECT_MODEL.is_file(), "run folders not present (untracked)")
class Contracts(unittest.TestCase):
    """Both directions, contract functions only (they run before any SB3 import)."""

    def test_direct_adapter_contract(self):
        from demo.direct_controller import ControllerRefused, check_direct_contract

        self.assertTrue(check_direct_contract(DIRECT_MODEL).is_file())
        with self.assertRaises(ControllerRefused):
            check_direct_contract(RESIDUAL_MODEL)

    def test_pid_ppo_adapter_contract(self):
        from evaluation import controllers as frozen

        self.assertTrue(frozen.check_ppo_contract(RESIDUAL_MODEL).is_file())
        with self.assertRaises(frozen.ControllerRefused):
            frozen.check_ppo_contract(DIRECT_MODEL)


if __name__ == "__main__":
    unittest.main()
