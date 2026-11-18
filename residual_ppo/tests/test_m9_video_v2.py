"""M9 Step 8 (video v2, descriptive, not eval-v1): offline checks of the camera / framing / inset geometry.
NumPy only; no Isaac, no OpenCV. The render-side checks (composed layer == .npz, projected vs rendered drone pixel
offsets) run inside render_replay_v2.py and are recorded in results/m9_demo/video_v2/<case>_stills*.json
(*_refhidden.json = checked with the white reference curve hidden in the check renders only).

Run from residual_ppo:  python -m unittest discover -s tests -p "test_*.py"
"""
import sys
import unittest
from pathlib import Path

import numpy as np

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESIDUAL_DIR))

from demo import video_v2_layout as L  # noqa: E402


def _cam(pos=(0.0, 0.0, 0.0), fwd=(1.0, 0.0, 0.0), s=1000.0):
    r, u, f = L.look_basis(np.asarray(fwd, dtype=float))
    return {"pos": np.asarray(pos, dtype=float), "r": r, "u": u, "f": f, "s": s}


class Colours(unittest.TestCase):
    def test_srgb_to_linear(self):
        np.testing.assert_allclose(L.srgb_to_linear([0.0, 1.0]), [0.0, 1.0])
        np.testing.assert_allclose(L.srgb_to_linear(0.5), 0.21404, atol=1e-5)
        np.testing.assert_allclose(L.srgb_to_linear(0.04045), 0.04045 / 12.92)

    def test_legend_colours_are_v1_colours(self):
        from demo.make_replay import CONTROLLERS

        for c in L.ORDER:
            self.assertEqual(L.COLORS_SRGB[c], CONTROLLERS[c][1])
        self.assertEqual(L.LABELS, {"pid": "PID", "pid_ppo": "PID+PPO", "ppo_direct": "PPO only"})


class Projection(unittest.TestCase):
    def test_axis_and_offset(self):
        cam = _cam(s=800.0)
        px, depth = L.project([[5.0, 0.0, 0.0], [5.0, -1.0, 0.5]], cam)
        np.testing.assert_allclose(px[0], [L.W / 2, L.H / 2])
        # camera looks along +x: right = -y, up = +z
        np.testing.assert_allclose(px[1], [L.W / 2 + 800.0 * 1.0 / 5.0, L.H / 2 - 800.0 * 0.5 / 5.0])
        np.testing.assert_allclose(depth, [5.0, 5.0])

    def test_scale_is_focal_over_aperture(self):
        cam = _cam(s=12.0 / L.H_APERTURE * L.W)
        self.assertAlmostEqual(L.focal_mm(cam), 12.0)


class Framing(unittest.TestCase):
    def test_panels_touch_the_margin(self):
        """Precondition for max_scale: every panel reaches the frame margin, so the allowed region is star-shaped
        around the image centre, which itself is allowed."""
        for name, (x0, y0, x1, y1) in L.PANELS.items():
            touches = (x0 <= L.EDGE or y0 <= L.EDGE or x1 >= L.W - L.EDGE or y1 >= L.H - L.EDGE)
            self.assertTrue(touches, name)
        self.assertGreater(L.pixel_margin(np.array([[L.W / 2, L.H / 2]]))[0], 0)
        # inset and caption bar are joined by the panel gap (the corner region stays connected to the edge)
        self.assertGreaterEqual(L.INSET[3] + L.PANEL_GAP, L.PANELS["caption"][1] - L.PANEL_GAP)

    def test_max_scale_is_exact_per_point(self):
        rng = np.random.default_rng(1)
        n = rng.normal(size=(400, 2)) * 0.6
        s = L.max_scale(n[:, 0][:, None], n[:, 1][:, None])           # one point per row
        for (nx, ny), sm in zip(n, s):
            inside = np.array([[L.W / 2 + 0.999 * sm * nx, L.H / 2 - 0.999 * sm * ny]])
            outside = np.array([[L.W / 2 + 1.001 * sm * nx, L.H / 2 - 1.001 * sm * ny]])
            self.assertGreater(L.pixel_margin(inside)[0], -1e-6)
            self.assertLessEqual(L.pixel_margin(outside)[0], 1e-6)
            for frac in np.linspace(0.0, 0.999, 15):                    # star-shaped: the whole segment is allowed
                q = np.array([[L.W / 2 + frac * sm * nx, L.H / 2 - frac * sm * ny]])
                self.assertGreater(L.pixel_margin(q)[0], -1e-6)

    def test_global_scale_is_min_over_points(self):
        rng = np.random.default_rng(2)
        n = rng.normal(size=(50, 2)) * 0.5
        self.assertAlmostEqual(float(L.max_scale(n[:, 0], n[:, 1])),
                               float(np.min(L.max_scale(n[:, 0][:, None], n[:, 1][:, None]))))


class CasePlan(unittest.TestCase):
    """The v1 framing bug case: every point must now be framed (hard assert inside plan_case)."""

    @classmethod
    def setUpClass(cls):
        cls.plan, cls.data, cls.cam = L.plan_case("indoor_I1_aisle_w7", verbose=False)

    def test_framing(self):
        self.assertTrue(self.plan["framing_ok"])
        for name, g in self.plan["framing"].items():
            self.assertGreater(g["min_margin_px"], 0, name)
            self.assertGreater(g["min_depth_m"], 0.3, name)
        start_px, _ = L.project(self.data["start"][None], self.cam)
        self.assertGreaterEqual(L.pixel_margin(start_px)[0], L.START_MARGIN_PX)

    def test_camera_inside_warehouse(self):
        lo, hi = L.SCENE_BOUNDS["indoor"]
        self.assertTrue(np.all(self.cam["pos"] >= lo) and np.all(self.cam["pos"] <= hi))
        self.assertLess(self.cam["pos"][2], 9.0)
        self.assertGreaterEqual(self.plan["camera"]["obstacle_clearance_m"], L.CAMERA_CLEARANCE)
        self.assertGreaterEqual(self.plan["camera"]["focal_mm"], L.FOCAL_MIN_MM)
        self.assertLessEqual(self.plan["camera"]["focal_mm"], L.FOCAL_MAX_MM + 1e-9)

    def test_same_start_and_wind_arrow(self):
        self.assertTrue(self.plan["same_start_all_controllers"])
        tail, tip = np.array(self.plan["wind_arrow"]["tail"]), np.array(self.plan["wind_arrow"]["tip"])
        d = (tip - tail)[:2] / np.linalg.norm((tip - tail)[:2])
        np.testing.assert_allclose(d, self.data["wind_dir"], atol=1e-12)
        self.assertAlmostEqual(tail[2], tip[2])                          # horizontal
        self.assertGreaterEqual(self.plan["wind_arrow"]["obstacle_clearance_m"], 0.25)

    def test_inset_north_up(self):
        tf = self.plan["inset"]["transform"]
        c = np.array(tf["world_center"])
        p = L.to_inset(np.stack([c, c + [0.0, 1.0], c + [1.0, 0.0]]), tf)
        np.testing.assert_allclose(p[0], tf["px_center"])
        self.assertLess(p[1, 1], p[0, 1])                                # north = up on screen
        self.assertGreater(p[2, 0], p[0, 0])                             # east = right
        pts = L.to_inset(L.scene_points(self.data, self.data["arrow"])[:, :2], tf)
        x0, y0, x1, y1 = L.INSET
        self.assertTrue(np.all((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0 + L.INSET_TITLE)
                               & (pts[:, 1] < y1)))


if __name__ == "__main__":
    unittest.main()
