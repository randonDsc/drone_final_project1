"""Behavior tests: run `python -m unittest discover -s tests -v` here.

NumPy-only tests always run. USD scene tests run when pxr is available, e.g.
with Isaac Sim python.bat. No Isaac Sim GUI is needed to exercise USD reading.
"""
import heapq
import itertools
import math
import unittest
import numpy as np
import sim_config
from grid_map import Box, Plane, GridMap, rasterize, segment_hits_box
from mapping import SceneGeometry, build_occupancy_map, find_launch_entry, validate_launch_departure
from navigation import (dijkstra, plan_flight, shorten_path, path_length,
                        line_of_sight, WaypointFollower)
from controller import controller
from motor import MotorModel

try:
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False

S2, S3 = math.sqrt(2), math.sqrt(3)


def empty_map(shape=(8, 8, 8)):
    return GridMap(np.zeros(shape, np.uint8), np.zeros(3), 1.0, [], [])


def touched_cells(a, b):
    """Independent footprint: every cell in the bounding box of two neighbors."""
    ranges = [range(min(p, q), max(p, q) + 1) for p, q in zip(a, b)]
    return list(itertools.product(*ranges))


def reference_dijkstra(grid, start, goal):
    """Plain dict-based 26-connected Dijkstra with the same no-corner-cut rule."""
    best = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        cost, node = heapq.heappop(heap)
        if node == goal:
            return cost
        if cost > best[node]:
            continue
        for d in itertools.product((-1, 0, 1), repeat=3):
            if d == (0, 0, 0):
                continue
            q = tuple(n + o for n, o in zip(node, d))
            if not all(0 <= v < s for v, s in zip(q, grid.shape)):
                continue
            if any(grid[c] for c in touched_cells(node, q)):
                continue
            c2 = cost + math.sqrt(sum(o * o for o in d))
            if c2 < best.get(q, math.inf):
                best[q] = c2
                heapq.heappush(heap, (c2, q))
    return None


def assert_no_corner_cut(test, grid, cells):
    for a, b in zip(cells[:-1], cells[1:]):
        test.assertLessEqual(np.abs(b - a).max(), 1)
        for c in touched_cells(tuple(a), tuple(b)):
            test.assertEqual(grid[c], 0, f'edge {a}->{b} touches blocked {c}')


def floor_geometry(half=(.3, .3, .2), bottom=-.1):
    return SceneGeometry([], [Plane(np.array([0., 0., 1.]), 0., 'floor')],
                         np.array(half), bottom, [])


def segments_free(m, points):
    return all(m.segment_free(a, b) for a, b in zip(points[:-1], points[1:]))


class GridTests(unittest.TestCase):
    def test_coordinate_round_trip_and_bounds(self):
        m = empty_map()
        m.origin = np.array([-3., -4., -5.])
        for cell in [(0, 0, 0), (7, 3, 2)]:
            self.assertEqual(m.index(m.center(cell)), cell)
        with self.assertRaises(ValueError): m.index(m.upper)
        with self.assertRaises(ValueError): m.index(m.origin - .01)

    def test_thin_wall_blocks_even_voxel_face(self):
        b = Box(np.array([2., 0., 0.]), np.array([2., 4., 4.]), 'wall')
        m = rasterize([0, 0, 0], [4, 4, 4], 1., [b], [], 1000)
        self.assertTrue(np.all(m.grid[1:3] == 1))
        self.assertFalse(m.segment_free([.5, .5, .5], [3.5, .5, .5]))

    def test_infinite_plane_blocks_solid_halfspace(self):
        m = rasterize([-2, -2, -2], [2, 2, 2], .5, [],
                      [Plane(np.array([0., 0., 1.]), 0., 'floor')], 1000)
        self.assertEqual(m.grid[2, 2, 4], 1)
        self.assertEqual(m.grid[2, 2, 5], 0)
        self.assertFalse(m.segment_free([0, 0, 1], [0, 0, -1]))

    def test_memory_budget(self):
        with self.assertRaises(ValueError):
            rasterize([0, 0, 0], [1000, 1000, 1000], .1, [], [], 1000)

    def test_outside_obstacle_does_not_wrap(self):
        b = Box(np.array([-5., -5., -5.]), np.array([-2., -2., -2.]), 'outside')
        m = rasterize([0, 0, 0], [3, 3, 3], 1., [b], [], 1000)
        self.assertEqual(m.grid.sum(), 0)

    def test_segment_parallel_touch_and_cross(self):
        self.assertFalse(segment_hits_box([0, 0, 0], [0, 0, 4], [1, 1, 1], [2, 2, 2]))
        self.assertTrue(segment_hits_box([1, 1, 0], [1, 1, 4], [1, 1, 1], [2, 2, 2]))
        self.assertTrue(segment_hits_box([0, 1.5, 1.5], [4, 1.5, 1.5], [1, 1, 1], [2, 2, 2]))

    def test_segment_cells_free_counts_corner_contact(self):
        m = empty_map((3, 3, 1))
        m.grid[1, 0, 0] = 1
        # Passes exactly through the corner point (1, 1) shared with the blocked cell.
        self.assertFalse(m.segment_cells_free([.5, .5, .5], [1.5, 1.5, .5]))
        # A line clear of the blocked cell's closed volume is allowed.
        self.assertTrue(m.segment_cells_free([.5, .5, .5], [.5, 2.5, .5]))
        self.assertTrue(m.segment_cells_free([.5, 1.5, .5], [2.5, 2.5, .5]))
        with self.assertRaises(ValueError): m.segment_cells_free([.5, .5, .5], [5, 5, .5])


class SearchTests(unittest.TestCase):
    def test_empty_space_diagonal_costs(self):
        m = empty_map()
        for goal, expected in [
            ((1, 0, 0), 1.),            # face move
            ((1, 1, 0), S2),            # edge diagonal
            ((1, 1, 1), S3),            # corner diagonal
            ((5, 4, 3), 3 * S3 + S2 + 1),  # 3 corner + 1 edge + 1 face
            ((7, 7, 7), 7 * S3),
        ]:
            cells, cost, _ = dijkstra(m, [.5] * 3, m.center(goal))
            self.assertAlmostEqual(cost, expected, places=9, msg=goal)
            steps = np.linalg.norm(np.diff(cells, axis=0), axis=1)
            self.assertAlmostEqual(steps.sum(), expected, places=9)

    def test_cost_scales_with_cell_size(self):
        m = GridMap(np.zeros((4, 4, 4), np.uint8), np.zeros(3), .25, [], [])
        _, cost, _ = dijkstra(m, m.center((0, 0, 0)), m.center((3, 3, 3)))
        self.assertAlmostEqual(cost, 3 * S3 * .25, places=9)

    def test_edge_diagonal_cannot_cut_blocked_corner(self):
        m = empty_map((3, 3, 1))
        m.grid[1, 0, 0] = 1  # (0,0)->(1,1) would graze this cell's corner.
        cells, cost, _ = dijkstra(m, m.center((0, 0, 0)), m.center((1, 1, 0)))
        self.assertAlmostEqual(cost, 2.)  # forced around: two face moves
        assert_no_corner_cut(self, m.grid, cells)

    def test_corner_diagonal_cannot_cut_any_of_eight_cells(self):
        for blocked in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)]:
            m = empty_map((2, 2, 2))
            m.grid[blocked] = 1
            cells, cost, _ = dijkstra(m, m.center((0, 0, 0)), m.center((1, 1, 1)))
            self.assertAlmostEqual(cost, 1 + S2, msg=blocked)
            assert_no_corner_cut(self, m.grid, cells)

    def test_diagonal_only_gap_is_unreachable(self):
        # Start and goal regions touch only along a diagonal edge.
        m = empty_map((2, 2, 1))
        m.grid[1, 0, 0] = m.grid[0, 1, 0] = 1
        with self.assertRaises(RuntimeError):
            dijkstra(m, m.center((0, 0, 0)), m.center((1, 1, 0)))

    def test_must_change_altitude_to_cross_wall(self):
        m = empty_map()
        m.grid[3, :, :5] = 1
        cells, _, _ = dijkstra(m, [1.5, 2.5, 1.5], [5.5, 2.5, 1.5])
        self.assertGreaterEqual(cells[:, 2].max(), 5)
        assert_no_corner_cut(self, m.grid, cells)

    def test_unreachable_and_blocked_goal(self):
        m = empty_map(); m.grid[3, :, :] = 1
        with self.assertRaises(RuntimeError): dijkstra(m, [.5]*3, [6.5]*3)
        m.grid[6, 6, 6] = 1
        with self.assertRaises(ValueError): dijkstra(m, [.5]*3, [6.5]*3)

    def test_budget_progress_and_same_cell(self):
        m = empty_map()
        with self.assertRaises(RuntimeError): dijkstra(m, [.5]*3, [6.5]*3, max_expanded=2)
        cells, cost, _ = dijkstra(m, [.1]*3, [.9]*3)
        self.assertEqual(cost, 0.)
        self.assertEqual(len(cells), 1)
        # The progress callback still fires on long searches.
        calls = []
        big = empty_map((40, 40, 40)); big.grid[20, :, :39] = 1
        import navigation
        original = navigation.time.monotonic
        ticks = itertools.count()
        navigation.time.monotonic = lambda: next(ticks)  # every check looks "late"
        try:
            dijkstra(big, [.5]*3, [39.5, .5, .5], progress=calls.append)
        finally:
            navigation.time.monotonic = original
        self.assertTrue(calls and all(c % 2048 == 0 for c in calls))

    def test_matches_independent_reference_on_random_maps(self):
        rng = np.random.default_rng(13)
        for _ in range(15):
            m = empty_map((5, 5, 5)); m.grid[:] = rng.random(m.grid.shape) < .3
            start, end = (0, 0, 0), (4, 4, 4)
            m.grid[start] = m.grid[end] = 0
            expected = reference_dijkstra(m.grid, start, end)
            if expected is None:
                with self.assertRaises(RuntimeError): dijkstra(m, m.center(start), m.center(end))
            else:
                cells, actual, _ = dijkstra(m, m.center(start), m.center(end))
                self.assertAlmostEqual(actual, expected, places=9)
                assert_no_corner_cut(self, m.grid, cells)


class ShorteningTests(unittest.TestCase):
    def test_empty_space_collapses_to_straight_line(self):
        m = empty_map()
        route = [m.center(c) for c in [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 1, 0), (4, 2, 1)]]
        out = shorten_path(m, route)
        self.assertEqual(len(out), 2)
        np.testing.assert_array_equal(out[0], route[0])
        np.testing.assert_array_equal(out[-1], route[-1])

    def test_shortcut_avoids_obstacle_and_uses_farthest_visible(self):
        box = Box(np.array([1.2, 1.2, 0.]), np.array([2.8, 2.8, 4.]), 'pillar')
        m = rasterize([0, 0, 0], [4, 4, 4], .5, [box], [], 10_000)
        z = 2.25
        # Valid L-shaped route around the pillar.
        route = [np.array(p) for p in [
            (.25, .25, z), (.75, .25, z), (3.25, .25, z), (3.75, .25, z),
            (3.75, .75, z), (3.75, 3.25, z), (3.75, 3.75, z)]]
        self.assertTrue(segments_free(m, route))
        out = shorten_path(m, route)
        self.assertTrue(all(line_of_sight(m, a, b) for a, b in zip(out[:-1], out[1:])))
        self.assertLess(path_length(out), path_length(route) - 1e-9)
        self.assertFalse(line_of_sight(m, route[0], route[-1]))
        # Greedy farthest rule: from the start, nothing beyond the kept point is visible.
        i = next(k for k, p in enumerate(route) if np.array_equal(p, out[1]))
        self.assertTrue(all(not line_of_sight(m, route[0], route[k]) for k in range(i + 1, len(route))))

    def test_shortcut_respects_grid_only_blocked_cells(self):
        m = empty_map((3, 3, 1))
        m.grid[1, 1, 0] = 1  # no box exists; only the grid knows
        route = [m.center(c) for c in [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0), (2, 2, 0)]]
        out = shorten_path(m, route)
        for a, b in zip(out[:-1], out[1:]):
            self.assertTrue(m.segment_cells_free(a, b))
        self.assertGreater(len(out), 2)

    def test_short_inputs_and_bounds(self):
        m = empty_map()
        self.assertEqual(len(shorten_path(m, [[.5]*3])), 1)
        self.assertEqual(len(shorten_path(m, [[.5]*3, [1.5]*3])), 2)
        with self.assertRaises(ValueError):
            line_of_sight(m, [.5]*3, [20., .5, .5])


class FlightPlanTests(unittest.TestCase):
    def test_free_start_exact_goal_and_continuous_route(self):
        g = floor_geometry()
        start, goal = np.array([0., 0., 1.]), np.array([3.11, 2.14, 1.37])
        m = build_occupancy_map(g, start, goal)
        plan = plan_flight(m, g, start, goal)
        np.testing.assert_array_equal(plan.waypoints[0], start)
        np.testing.assert_array_equal(plan.waypoints[-1], goal)
        self.assertTrue(segments_free(m, plan.waypoints))
        # Open space: the whole route becomes a single straight line.
        self.assertEqual(len(plan.waypoints), 2)
        self.assertAlmostEqual(plan.distance, np.linalg.norm(goal - start))

    def test_thin_obstacle_between_free_endpoints(self):
        g = floor_geometry(half=(.1, .1, .05), bottom=-.05)
        g.boxes.append(Box(np.array([2., -1., 0.]), np.array([2.02, 1., 1.2]), 'thin_wall'))
        start, goal = np.array([0., 0., .6]), np.array([4., 0., .6])
        m = build_occupancy_map(g, start, goal)
        self.assertFalse(m.segment_free(start, goal))  # direct line hits the wall
        plan = plan_flight(m, g, start, goal)
        np.testing.assert_array_equal(plan.waypoints[0], start)
        np.testing.assert_array_equal(plan.waypoints[-1], goal)
        for a, b in zip(plan.waypoints[:-1], plan.waypoints[1:]):
            self.assertTrue(line_of_sight(m, a, b))
        assert_no_corner_cut(self, m.grid, plan.cells)
        cell_route = [m.center(c) for c in plan.cells]
        self.assertLessEqual(plan.distance, path_length([start, *cell_route, goal]) + 1e-9)
        self.assertLessEqual(len(plan.waypoints), 5)

    def platform_scene(self):
        g = floor_geometry()
        g.boxes.append(Box(np.array([-.5, -.5, 0.]), np.array([.5, .5, .5]), '/World/Cube'))
        return g

    def test_platform_departure_preserved_and_exact_goal(self):
        self.assertIn('/World/Cube', sim_config.LAUNCH_SUPPORT_ROOTS)
        g = self.platform_scene()
        start = np.array([0.03, -0.02, .6])  # resting on the cube, inside its clearance
        goal = np.array([3.3, 2.1, 1.4])
        m = build_occupancy_map(g, start, goal)
        self.assertEqual(m.grid[m.index(start)], 1)  # platform remains an obstacle
        blocked_before = m.grid.copy()
        plan = plan_flight(m, g, start, goal)
        np.testing.assert_array_equal(m.grid, blocked_before)  # no cells cleared
        entry = find_launch_entry(m, g, start)
        wp = plan.waypoints
        np.testing.assert_array_equal(wp[0], start)
        np.testing.assert_array_equal(wp[1], entry)
        np.testing.assert_array_equal(wp[1][:2], start[:2])  # straight up
        self.assertGreater(wp[1][2], start[2])
        np.testing.assert_array_equal(wp[-1], goal)
        # The direct launch->goal shortcut would cross the platform clearance.
        self.assertFalse(m.segment_free(start, goal))
        # Every segment after the departure is an ordinary checked segment.
        for a, b in zip(wp[1:-1], wp[2:]):
            self.assertTrue(line_of_sight(m, a, b))
        self.assertLess(len(wp), 6)

    def test_unreachable_destination(self):
        g = SceneGeometry([Box(np.array([1.9, -10., -10.]), np.array([2.1, 10., 10.]), 'wall')],
                          [], np.array([.3, .3, .2]), -.1, [])
        start, goal = np.array([0., 1., 1.5]), np.array([3.6, 1., 1.5])
        m = build_occupancy_map(g, start, goal, bounds=([-1., -1., 0.], [4.5, 3., 3.]))
        with self.assertRaisesRegex(RuntimeError, 'No path'):
            plan_flight(m, g, start, goal)

    def test_obstructed_departure_rejected(self):
        g = self.platform_scene()
        g.boxes.append(Box(np.array([-.2, -.2, 1.6]), np.array([.2, .2, 1.8]), 'ceiling'))
        with self.assertRaisesRegex(ValueError, 'ceiling'):
            validate_launch_departure(g, [0, 0, .6], [0, 0, 2.5])

    def test_large_floor_does_not_expand_map_but_remains_blocked(self):
        g = floor_geometry(); g.planes = []
        g.boxes.append(Box(np.array([-500.,-500.,-.1]), np.array([500.,500.,0.]), 'floor'))
        m = build_occupancy_map(g, [0,0,.1], [3,3,1])
        self.assertLess(m.grid.size, 100_000)
        self.assertEqual(m.grid[m.index([0,0,0])], 1)


def simulate(follower, start, seconds, mass=1.5):
    """Approximate rigid-body flight using the REAL controller (not Isaac Sim).

    Inertia is the PX4 Iris value; no drag, rotor dynamics or COM offset.
    Returns positions over time.
    """
    inertia = np.diag([0.029125, 0.029125, 0.055225])
    dt = 1. / 120.
    p, v, w = np.array(start, float), np.zeros(3), np.zeros(3)
    q = np.array([1., 0., 0., 0.])
    trace = []
    for _ in range(int(seconds / dt)):
        cmd = follower.update(p, v, dt)
        target = {'height': cmd[2], 'xy': cmd[:2], 'mass': mass,
                  'max_xy_speed': sim_config.MAX_XY_SPEED, 'max_z_speed': sim_config.MAX_Z_SPEED,
                  'max_tilt_degrees': sim_config.MAX_TILT_DEGREES}
        thrust, *torque = controller(p, q, v, w, target)
        qw, qx, qy, qz = q
        R = np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                      [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
                      [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]])
        wb = R.T @ w
        wb = wb + np.linalg.solve(inertia, np.array(torque) - np.cross(wb, inertia @ wb)) * dt
        w = R @ wb
        v = v + (R[:, 2] * thrust / mass - [0, 0, 9.81]) * dt
        p = p + v * dt
        dq = 0.5 * np.array([-qx*wb[0]-qy*wb[1]-qz*wb[2], qw*wb[0]+qy*wb[2]-qz*wb[1],
                             qw*wb[1]+qz*wb[0]-qx*wb[2], qw*wb[2]+qx*wb[1]-qy*wb[0]])
        q = q + dq * dt; q /= np.linalg.norm(q)
        trace.append(p.copy())
        if follower.done:
            break
    return np.array(trace)


def max_route_deviation(trace, waypoints):
    def dist(p, a, b):
        d = b - a; t = np.clip((p - a) @ d / max(d @ d, 1e-12), 0, 1)
        return np.linalg.norm(p - (a + t * d))
    return max(min(dist(p, a, b) for a, b in zip(waypoints[:-1], waypoints[1:])) for p in trace)


class FollowerTests(unittest.TestCase):
    def test_follower_requires_low_speed_dwell_then_holds_goal(self):
        f = WaypointFollower([[0,0,1], [1,0,1]])
        for _ in range(10): f.update(np.array([0,0,1]), np.array([1,0,0]), .1)
        self.assertEqual(f.index, 0)
        for _ in range(3): f.update(np.array([0,0,1]), np.zeros(3), .1)
        self.assertEqual(f.index, 1)
        self.assertFalse(f.done)
        for _ in range(3): f.update(np.array([1,0,1]), np.zeros(3), .1)
        self.assertTrue(f.done)
        np.testing.assert_allclose(f.update(np.array([1,0,1]), np.zeros(3), .1), [1,0,1])

    def test_axis_aligned_and_departure_segments_stay_point_to_point(self):
        f = WaypointFollower([[0, 0, .6], [0, 0, 1.2], [3, 0, 1.2]])
        f.index = 1
        np.testing.assert_array_equal(f.command(np.array([0, 0, .7])), [0, 0, 1.2])
        f.index = 2
        np.testing.assert_array_equal(f.command(np.array([.5, 0, 1.2])), [3, 0, 1.2])

    def test_sloped_segment_tracked_close_to_line(self):
        waypoints = np.array([[0., 0., 1.], [4., 3., 3.5], [6., 0., 1.5]])
        for guided in (True, False):
            f = WaypointFollower(waypoints, guided=guided)
            trace = simulate(f, waypoints[0], 150.)
            dev = max_route_deviation(trace, waypoints)
            if guided:
                self.assertTrue(f.done)
                self.assertLess(np.linalg.norm(trace[-1] - waypoints[-1]), sim_config.WAYPOINT_RADIUS)
                self.assertLess(dev, sim_config.SAFETY_MARGIN)
                guided_dev = dev
            else:
                # Point-to-point flight leaves the checked line by far more.
                self.assertGreater(dev, 3 * guided_dev)

    def test_lead_shrinks_gradually_off_the_line(self):
        f = WaypointFollower([[0., 0., 1.], [10., 0., 3.]])
        f.index = 1
        length = np.hypot(10., 2.)
        tol, stop = sim_config.TRACK_TOLERANCE, sim_config.TRACK_STOP_DISTANCE

        def lead_fraction(offset):
            # Drone 20% along the line, displaced sideways (perpendicular).
            cmd = f.command(np.array([2., offset, 1.4]))
            return (cmd[0] / 10. - 0.2) * length / sim_config.TRACK_XY_LEAD

        self.assertAlmostEqual(lead_fraction(0.), 1.)
        self.assertAlmostEqual(lead_fraction(tol), 1.)
        # Just past the tolerance the drone keeps nearly full lead (no halt).
        self.assertGreater(lead_fraction(tol + 0.01), 0.9)
        self.assertAlmostEqual(lead_fraction(0.5 * (tol + stop)), 0.5)
        self.assertAlmostEqual(lead_fraction(stop + 0.05), 0.)

    def test_controller_hover_and_direction(self):
        target = {'height':1., 'xy':[0,0], 'mass':1.5}
        result = controller([0,0,1], [1,0,0,0], [0,0,0], [0,0,0], target)
        np.testing.assert_allclose(result, [1.5*9.81,0,0,0])
        target['xy'] = [10,0]
        result = controller([0,0,1], [1,0,0,0], [0,0,0], [0,0,0], target)
        self.assertGreater(result[2], 0)
        self.assertEqual(result[1], 0)

    def test_motor_clamp_and_invalid_input(self):
        m = MotorModel(1., .1, 10., [-1,-1,1,1])
        thrust, torque = m.calculate([-1,2,3,20])
        np.testing.assert_allclose(thrust, [0,4,9,100])
        np.testing.assert_allclose(torque, [0,-.4,.9,10])
        with self.assertRaises(ValueError): m.calculate([0,0,0,np.nan])


@unittest.skipUnless(USD_AVAILABLE, 'pxr is not installed')
class USDTests(unittest.TestCase):
    def scene(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(stage, '/World')
        UsdGeom.Xform.Define(stage, '/World/iris')
        body = UsdGeom.Cube.Define(stage, '/World/iris/body')
        body.CreateSizeAttr(.2)
        body.AddTranslateOp().Set(Gf.Vec3d(0,0,.1))
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        wall = UsdGeom.Cube.Define(stage, '/World/wall')
        wall.CreateSizeAttr(1.)
        wall.AddTranslateOp().Set(Gf.Vec3d(3,0,.5))
        UsdPhysics.CollisionAPI.Apply(wall.GetPrim())
        return stage

    def test_transformed_bounds_disabled_collision_drone_exclusion(self):
        from mapping import read_scene_geometry
        stage = self.scene()
        disabled = UsdGeom.Cube.Define(stage, '/World/disabled')
        UsdPhysics.CollisionAPI.Apply(disabled.GetPrim()).CreateCollisionEnabledAttr(False)
        geometry = read_scene_geometry(stage, [0,0,.1])
        self.assertEqual(len(geometry.boxes), 1)
        self.assertEqual(geometry.boxes[0].path, '/World/wall')
        np.testing.assert_allclose(geometry.boxes[0].lower, [2.5,-.5,0])

    def test_infinite_plane_not_finite_box(self):
        if not hasattr(UsdGeom, 'Plane'): self.skipTest('USD has no Plane schema')
        from mapping import read_scene_geometry
        stage = self.scene()
        floor = UsdGeom.Plane.Define(stage, '/World/floor')
        floor.CreateAxisAttr('Z')
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        geometry = read_scene_geometry(stage, [0,0,.1])
        self.assertEqual(len(geometry.planes), 1)
        np.testing.assert_allclose(geometry.planes[0].normal, [0,0,1])

    def test_units_rejected(self):
        from mapping import read_scene_geometry
        stage = self.scene(); UsdGeom.SetStageMetersPerUnit(stage, .01)
        with self.assertRaisesRegex(ValueError, 'meter-based'):
            read_scene_geometry(stage, [0,0,.1])


if __name__ == '__main__':
    unittest.main()
