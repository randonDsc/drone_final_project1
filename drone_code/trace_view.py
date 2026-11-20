"""Draw the planned route and the flown trace in the Isaac Sim viewport.

GUI runs only. Lines are drawn with isaacsim.util.debug_draw: they sit at their
world positions, so they show from any camera and in a screen or Movie Capture
recording, but they are not USD prims and are not saved with the stage.

    planned route   green, drawn once when the flight begins
    flown trace     grows as the drone flies, coloured by how far the drone is
                    from where the reference says it should be: blue on track,
                    red at FULL_RED_ERROR or more
    markers         start (green), goal (white), reference point now (yellow),
                    failure or collision (red)

Import only after SimulationApp has started.
"""
import numpy as np


PLANNED_COLOR = (0.15, 0.85, 0.25, 1.0)
PLANNED_WIDTH = 3.0

ON_TRACK_COLOR = np.array([0.1, 0.55, 1.0, 1.0])
OFF_TRACK_COLOR = np.array([1.0, 0.1, 0.1, 1.0])
FULL_RED_ERROR = 0.3      # [m] tracking error drawn fully red
FLOWN_WIDTH = 5.0

# A new flown segment only once the drone has moved this far [m], so hovering
# does not pile up thousands of zero-length lines.
MIN_SEGMENT = 0.02

# Long planned routes are thinned to at most this many segments.
MAX_PLANNED_SEGMENTS = 4000

START_COLOR = (0.15, 0.85, 0.25, 1.0)
GOAL_COLOR = (1.0, 1.0, 1.0, 1.0)
REFERENCE_COLOR = (1.0, 0.85, 0.1, 1.0)
FAILURE_COLOR = (1.0, 0.1, 0.1, 1.0)
MARKER_SIZE = 14.0
REFERENCE_SIZE = 10.0

EXTENSION = "isaacsim.util.debug_draw"


def _points(array):
    return [tuple(float(v) for v in point) for point in np.asarray(array, dtype=float)]


def error_color(error):
    """Blue at zero tracking error, red at FULL_RED_ERROR and beyond."""
    share = float(np.clip(error / FULL_RED_ERROR, 0.0, 1.0))
    return tuple(float(v) for v in ON_TRACK_COLOR + share * (OFF_TRACK_COLOR - ON_TRACK_COLOR))


class TraceView:
    """Planned and flown traces for one flight at a time."""

    def __init__(self):
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()

        if not manager.is_extension_enabled(EXTENSION):
            manager.set_extension_enabled_immediate(EXTENSION, True)

        from isaacsim.util.debug_draw import _debug_draw

        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.markers = []
        self.last = None

    def begin(self, path):
        """Clear the previous flight and draw this one's planned route."""
        path = np.asarray(path, dtype=float).reshape(-1, 3)

        self.draw.clear_lines()
        self.draw.clear_points()
        self.last = None

        if len(path) > MAX_PLANNED_SEGMENTS + 1:
            keep = np.linspace(0, len(path) - 1, MAX_PLANNED_SEGMENTS + 1)
            path = path[np.unique(np.round(keep).astype(int))]

        if len(path) >= 2:
            count = len(path) - 1
            self.draw.draw_lines(
                _points(path[:-1]),
                _points(path[1:]),
                [PLANNED_COLOR] * count,
                [PLANNED_WIDTH] * count,
            )

        self.markers = []

        if len(path):
            self.markers = [
                (tuple(path[0]), START_COLOR, MARKER_SIZE),
                (tuple(path[-1]), GOAL_COLOR, MARKER_SIZE),
            ]

        self._draw_markers(None)

    def record(self, position, reference_position):
        """Extend the flown trace to position, coloured by the tracking error."""
        position = np.asarray(position, dtype=float).reshape(3)
        reference_position = np.asarray(reference_position, dtype=float).reshape(3)

        if not np.all(np.isfinite(position)):
            return

        if self.last is None:
            self.last = position
        elif np.linalg.norm(position - self.last) >= MIN_SEGMENT:
            error = np.linalg.norm(reference_position - position)
            self.draw.draw_lines(
                _points([self.last]),
                _points([position]),
                [error_color(error)],
                [FLOWN_WIDTH],
            )
            self.last = position

        self._draw_markers(reference_position)

    def mark_failure(self, position):
        """A red marker where the flight failed; it stays until the next begin."""
        position = np.asarray(position, dtype=float).reshape(3)

        if np.all(np.isfinite(position)):
            self.markers.append((tuple(position), FAILURE_COLOR, MARKER_SIZE * 1.5))
            self._draw_markers(None)

    def _draw_markers(self, reference_position):
        # Points cannot be moved, so the few markers are redrawn each time.
        self.draw.clear_points()

        markers = list(self.markers)

        if reference_position is not None and np.all(np.isfinite(reference_position)):
            markers.append((tuple(reference_position), REFERENCE_COLOR, REFERENCE_SIZE))

        if markers:
            self.draw.draw_points(
                [tuple(float(v) for v in point) for point, _, _ in markers],
                [color for _, color, _ in markers],
                [size for _, _, size in markers],
            )
