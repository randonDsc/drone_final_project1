"""
Shared simulation, mapping, and navigation settings.

Distances: meters
Time: seconds
Angles: radians unless explicitly named degrees

Edit the final destination in motor_physics.py's target dictionary.
"""

# ---------- Occupancy grid ----------

# Each grid cell is a cube with this edge length.
# Smaller cells resolve smaller gaps but increase memory/search cost.
CELL_SIZE = 0.25

# Automatically chosen map bounds include:
# - the actual starting position
# - the final destination
# - finite obstacle bounds
#
# Padding extends those bounds in all directions.
# Headroom adds additional space above the scene.
MAP_PADDING = 1.0
MAP_HEADROOM = 2.0

# Very wide, flat floor slabs remain obstacles but do not determine
# the horizontal size of the planning area.
LARGE_FLOOR_WIDTH = 30.0

# Explicit memory and search limits.
MAX_GRID_CELLS = 2_000_000
MAX_EXPANDED_NODES = 1_000_000

# ---------- Drone clearance ----------

# The drone's dimensions are estimated from its USD geometry.
# This margin is added around that envelope to allow clearance
# and some position-tracking error.
SAFETY_MARGIN = 0.01

# Used by both the controller and the drone-envelope estimate.
MAX_TILT_DEGREES = 10.0

# Your confirmed launch platform.
#
# The platform remains an obstacle in the occupancy grid.
# Only the initial upward departure may start inside its
# expanded clearance region.
LAUNCH_SUPPORT_ROOTS = (
    "/World/Cube",
)

# ---------- Waypoint following ----------

# A waypoint is reached when both position error and speed
# remain below these thresholds for WAYPOINT_DWELL seconds.
WAYPOINT_RADIUS = 0.08
WAYPOINT_SPEED = 0.12
WAYPOINT_DWELL = 0.25

# Desired speed limits used by the controller.
# These limit requested velocities, not measured velocities.
MAX_XY_SPEED = 5
MAX_Z_SPEED = 5

# Stop with a diagnostic error if a waypoint never settles.
WAYPOINT_TIMEOUT = 180.0

# ---------- Guided tracking of sloped segments ----------

# The controller's height loop responds much faster than its horizontal
# loop. Commanding only the far end of a segment that both climbs and
# translates would make the drone fly an L-shaped curve instead of the
# straight, collision-checked line.
#
# For such segments the follower slides intermediate targets along the
# line. A segment is flown point-to-point, as before, when even the
# worst L-shaped deviation would stay within TRACK_TOLERANCE.
#
# TRACK_TOLERANCE must remain below SAFETY_MARGIN. When the measured
# distance from the segment exceeds it, the lead shrinks gradually, reaching
# zero (pull straight back onto the line) at TRACK_STOP_DISTANCE. Aiming at
# a point ahead on the line steers back on its own, so small drift only
# slows progress instead of stopping the drone.
TRACK_TOLERANCE = 0.12
TRACK_STOP_DISTANCE = 0.24

# How far ahead of the drone's progress along the segment the horizontal
# and height targets are placed, measured along the segment [m].
#
# TRACK_Z_LEAD = TRACK_XY_LEAD / 6 matches the controller's
# position-to-velocity ratios (horizontal 0.3/0.9, height 10/5).
# Both axes then ask for the same speed along the line, about
# TRACK_XY_LEAD / 3 m/s.
TRACK_XY_LEAD = 1.2
TRACK_Z_LEAD = 0.2

# ---------- USD paths ----------

DRONE_ROOT = "/World/iris"
BODY_PATH = "/World/iris/body"

# Exclude the drone, including its sensors, from obstacles.
# This does not exclude it from the drone-size calculation.
EXCLUDE_ROOTS = (
    DRONE_ROOT,
)

# Enabled collision geometry is mapped automatically.
#
# Add roots here only if you also want visual geometry without
# collision APIs to count as obstacles.
EXTRA_OBSTACLE_ROOTS = ()

# ---------- Propeller animation ----------

# Spin the propeller rigid bodies on their revolute joints so the motion
# is visible. Thrust, torque and yaw reaction are still applied to the
# body by the motor model, exactly as before; the spin is visual only.
PROPELLER_SPIN = True

# Revolute joints in motor order: joint i carries rotor i, matching
# rotor_positions and reaction_signs in motor_physics.py.
ROTOR_JOINT_PATHS = (
    "/World/iris/rotor0/joint0",
    "/World/iris/rotor1/joint1",
    "/World/iris/rotor2/joint2",
    "/World/iris/rotor3/joint3",
)

# Displayed speed = motor speed * scale.
#
# Real speeds (~820 rad/s at hover) turn more than once per 120 Hz frame,
# so they strobe on screen (look frozen or backwards). Spinning the rotor
# bodies that fast would also add gyroscopic torque the controller does
# not model. 0.05 gives about 40 rad/s at hover.
PROPELLER_SPIN_SCALE = 0.5

# Torque limit of each spin drive [N m]. Keeps the drive's reaction on the
# body small (controller torques are of order 0.05 N m) while still
# reaching the displayed speed in about 0.1 s. for show
PROPELLER_DRIVE_MAX_TORQUE = 0.1

# ---------- Wind ----------

# Gusting wind with distributed quadratic drag; see wind.py for the model.
# This is the only aerodynamic drag in the simulation: motor.py models none
# and the USD authors no PhysX linear damping on the drone bodies.
#
# Only mean_wind is the requested example value. Everything else is a
# tunable starting assumption, not a measured or validated Iris property.
WIND = {
    "enabled": True,

    # Mean world-frame wind velocity [m/s] (direction the air moves toward).
    "mean_wind": (8.0, 0.0, 0.0),

    # Shared gust felt by the whole drone: per-axis standard deviation
    # [m/s] and correlation time [s]. Larger timescale = longer gusts.
    "gust_std": (2.0, 2.0, 0.7),
    "gust_timescale": 2.0,

    # Smaller, faster variation across the airframe. Per point and axis the
    # total wind variance is gust_std**2 + local_gust_std**2.
    "local_gust_std": (0.6, 0.6, 0.3),
    "local_gust_timescale": 0.3,

    # Distance [m] over which local fluctuations stay correlated:
    # correlation = exp(-distance / correlation_length). The airframe spans
    # about 0.5 m. Smaller = more uneven wind = more rotational disturbance.
    "correlation_length": 0.4,

    # Drag: F = 0.5 * air_density * drag_coefficient * area * |v|^2.
    # drag_coefficient * effective_area = 0.03 m^2 gives about 1.8 N
    # at 10 m/s relative airspeed.
    "air_density": 1.225,
    "drag_coefficient": 1.0,
    "effective_area": 0.03,

    # Same seed -> same gust sequence on every run and every Stop -> Play.
    # None draws a new sequence each time.
    "seed": 7,

    # (name, body-frame offset from the body origin [m], area weight).
    # Offsets follow the USD geometry: hull mesh bbox x -0.14..0.16,
    # z -0.067..0.047; rotor hubs match rotor_positions in motor_physics.py.
    # Weights are normalized to sum to 1, so effective_area is split among
    # the points rather than multiplied by their count.
    "sample_points": (
        ("hull", (0.000, 0.000, -0.010), 0.24),
        ("landing_gear", (0.000, 0.000, -0.055), 0.10),
        ("arm0", (0.069, -0.103, 0.012), 0.04),
        ("arm1", (-0.063, 0.109, 0.012), 0.04),
        ("arm2", (0.069, 0.101, 0.012), 0.04),
        ("arm3", (-0.062, -0.111, 0.012), 0.04),
        ("rotor0", (0.138, -0.207, 0.023), 0.125),
        ("rotor1", (-0.125, 0.219, 0.023), 0.125),
        ("rotor2", (0.138, 0.203, 0.023), 0.125),
        ("rotor3", (-0.124, -0.222, 0.023), 0.125),
    ),
}

# ---------- Sensors ----------

# Sensors provide optional logging.
# Navigation currently uses the USD map and simulator state.
ENABLE_SENSORS = True