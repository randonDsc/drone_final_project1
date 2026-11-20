"""trace_view without Isaac Sim: a stand-in records what would be drawn."""
import unittest

import numpy as np

import trace_view
from trace_view import TraceView, error_color


class FakeDraw:
    def __init__(self):
        self.lines = []
        self.points = []

    def clear_lines(self):
        self.lines = []

    def clear_points(self):
        self.points = []

    def draw_lines(self, starts, ends, colors, widths):
        assert len(starts) == len(ends) == len(colors) == len(widths)
        self.lines += list(zip(starts, ends, colors, widths))

    def draw_points(self, points, colors, sizes):
        assert len(points) == len(colors) == len(sizes)
        self.points += list(zip(points, colors, sizes))


def make_view():
    view = TraceView.__new__(TraceView)
    view.draw = FakeDraw()
    view.markers = []
    view.last = None
    return view


class TraceViewTest(unittest.TestCase):
    def test_colour_goes_from_blue_to_red(self):
        self.assertEqual(error_color(0.0), tuple(trace_view.ON_TRACK_COLOR))
        np.testing.assert_allclose(error_color(trace_view.FULL_RED_ERROR), trace_view.OFF_TRACK_COLOR)
        np.testing.assert_allclose(error_color(5.0), trace_view.OFF_TRACK_COLOR)
        middle = error_color(trace_view.FULL_RED_ERROR / 2)
        self.assertAlmostEqual(middle[0], 0.55)

    def test_planned_route_and_markers(self):
        view = make_view()
        path = np.column_stack([np.linspace(0, 10, 11), np.zeros(11), np.ones(11)])
        view.begin(path)

        self.assertEqual(len(view.draw.lines), 10)
        self.assertTrue(all(line[2] == trace_view.PLANNED_COLOR for line in view.draw.lines))
        self.assertEqual([p[0] for p in view.draw.points], [(0.0, 0.0, 1.0), (10.0, 0.0, 1.0)])

    def test_long_route_is_thinned_but_keeps_its_ends(self):
        view = make_view()
        n = 3 * trace_view.MAX_PLANNED_SEGMENTS
        path = np.column_stack([np.arange(n, dtype=float), np.zeros(n), np.zeros(n)])
        view.begin(path)

        self.assertLessEqual(len(view.draw.lines), trace_view.MAX_PLANNED_SEGMENTS)
        self.assertEqual(view.draw.lines[0][0], (0.0, 0.0, 0.0))
        self.assertEqual(view.draw.lines[-1][1], (n - 1.0, 0.0, 0.0))

    def test_flown_trace_grows_and_skips_tiny_moves(self):
        view = make_view()
        view.begin(np.array([[0.0, 0, 1], [1.0, 0, 1]]))
        planned = len(view.draw.lines)

        view.record([0, 0, 1], [0, 0, 1])
        view.record([0.001, 0, 1], [0, 0, 1])   # too small: no segment
        view.record([0.1, 0, 1], [0.1, 0, 1])   # on track: blue
        view.record([0.2, 0.3, 1], [0.2, 0, 1])  # 0.3 m off: red

        flown = view.draw.lines[planned:]
        self.assertEqual(len(flown), 2)
        self.assertEqual(flown[0][2], tuple(trace_view.ON_TRACK_COLOR))
        np.testing.assert_allclose(flown[1][2], trace_view.OFF_TRACK_COLOR)
        self.assertEqual(flown[1][0], (0.1, 0.0, 1.0))

        # Start, goal, and the moving reference marker.
        self.assertEqual(len(view.draw.points), 3)
        self.assertEqual(view.draw.points[-1][0], (0.2, 0.0, 1.0))

    def test_failure_marker_stays_and_nan_is_ignored(self):
        view = make_view()
        view.begin(np.array([[0.0, 0, 1], [1.0, 0, 1]]))
        lines = len(view.draw.lines)

        view.record([np.nan, 0, 1], [0, 0, 1])
        self.assertEqual(len(view.draw.lines), lines)

        view.mark_failure([0.5, 0, 0.2])
        view.record([0.5, 0, 0.2], [0.6, 0, 1])
        self.assertIn((0.5, 0.0, 0.2), [p[0] for p in view.draw.points])

        view.begin(np.array([[0.0, 0, 1], [1.0, 0, 1]]))
        self.assertNotIn((0.5, 0.0, 0.2), [p[0] for p in view.draw.points])


if __name__ == "__main__":
    unittest.main()
