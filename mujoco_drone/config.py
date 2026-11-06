"""Validated, JSON-serializable experiment inputs. All physical values are SI."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math


def positive(name, value, allow_zero=False):
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


def vector(name, value, size=3):
    if len(value) != size or not all(math.isfinite(x) for x in value):
        raise ValueError(f"{name} must contain {size} finite values")


@dataclass
class PhysicsConfig:
    timestep: float = 1 / 120
    controller: str = "original_adapted"
    max_xy_speed: float = 2.0
    max_z_speed: float = 1.0
    max_tilt_degrees: float = 10.0

    def __post_init__(self):
        positive("timestep", self.timestep)
        if abs(0.1 / self.timestep - round(0.1 / self.timestep)) > 1e-8:
            raise ValueError("timestep must divide the SAC decision interval (0.1 s)")
        if self.controller not in ("original_adapted", "isaac_pid", "original_literal"):
            raise ValueError("Unknown PID family")
        for name in ("max_xy_speed", "max_z_speed", "max_tilt_degrees"):
            positive(name, getattr(self, name))
        if self.max_tilt_degrees > 30:
            raise ValueError("Near-hover controllers require max_tilt_degrees <= 30")


@dataclass
class WindConfig:
    scenario: str = "calm"
    mean_velocity: tuple = (0.0, 3.0, 0.0)
    gust_velocity: tuple = (0.0, 4.0, 0.0)
    gust_start: float = 5.0
    gust_duration: float = 2.0
    frequency_hz: float = 0.25
    gust_std: tuple = (0.7, 0.7, 0.25)
    local_gust_std: tuple = (0.2, 0.2, 0.1)
    gust_timescale: float = 2.0
    local_gust_timescale: float = 0.3
    correlation_length: float = 0.4
    speed_range: tuple = (0.0, 6.0)
    direction_rate: float = 0.0
    air_density: float = 1.225
    drag_coefficient: float = 1.0
    effective_area: float = 0.03
    enabled: bool = True

    def __post_init__(self):
        if self.scenario not in ("calm", "constant", "gust", "periodic", "turbulence", "randomized", "curriculum"):
            raise ValueError(f"Unknown wind scenario: {self.scenario}")
        for name in ("mean_velocity", "gust_velocity", "gust_std", "local_gust_std"):
            vector(name, getattr(self, name))
        if any(x < 0 for x in (*self.gust_std, *self.local_gust_std)):
            raise ValueError("Wind standard deviations cannot be negative")
        for name in ("gust_timescale", "local_gust_timescale", "correlation_length", "frequency_hz", "gust_duration"):
            positive(name, getattr(self, name))
        for name in ("gust_start", "air_density", "drag_coefficient", "effective_area"):
            positive(name, getattr(self, name), allow_zero=True)
        vector("speed_range", self.speed_range, 2)
        if not 0 <= self.speed_range[0] <= self.speed_range[1]:
            raise ValueError("speed_range must be ordered and nonnegative")
        if not math.isfinite(self.direction_rate):
            raise ValueError("direction_rate must be finite")


@dataclass
class TaskConfig:
    kind: str = "hover"
    start: tuple = (0.0, 0.0, 1.5)
    goal: tuple = (4.0, 0.0, 1.5)
    waypoints: tuple = ((0.0, 0.0, 1.5), (2.0, 0.0, 1.5), (2.0, 2.0, 2.0), (0.0, 2.0, 1.5))
    speed: float = 0.6
    acceleration: float = 0.4
    radius: float = 1.0
    start_hold: float = 3.0
    end_hold: float = 8.0
    hover_seconds: float = 60.0
    initial_position_range: tuple = (0.1, 0.1, 0.1)
    initial_velocity_range: tuple = (0.05, 0.05, 0.05)
    initial_angle_range_degrees: tuple = (2.0, 2.0, 3.0)
    initial_angular_velocity_range: tuple = (0.05, 0.05, 0.05)

    def __post_init__(self):
        if self.kind not in ("hover", "straight", "waypoints", "circle", "figure_eight"):
            raise ValueError(f"Unknown task: {self.kind}")
        for name in ("start", "goal", "initial_position_range", "initial_velocity_range", "initial_angle_range_degrees", "initial_angular_velocity_range"):
            vector(name, getattr(self, name))
        for name in ("speed", "acceleration", "radius", "hover_seconds"):
            positive(name, getattr(self, name))
        for name in ("start_hold", "end_hold"):
            positive(name, getattr(self, name), allow_zero=True)
        if not self.waypoints:
            raise ValueError("waypoints cannot be empty")
        for p in self.waypoints:
            vector("waypoint", p)
        for name in ("initial_position_range", "initial_velocity_range", "initial_angle_range_degrees", "initial_angular_velocity_range"):
            if any(x < 0 for x in getattr(self, name)):
                raise ValueError(f"{name} must be nonnegative")


@dataclass
class SensorConfig:
    capture_camera: bool = False
    camera: str = "drone_camera"
    width: int = 320
    height: int = 240
    camera_hz: float = 5.0

    def __post_init__(self):
        for name in ("width", "height"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        positive("camera_hz", self.camera_hz)


@dataclass
class ExperimentConfig:
    seed: int = 0
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    max_seconds: float = 120.0

    def __post_init__(self):
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        positive("max_seconds", self.max_seconds)
        if self.sensors.camera_hz > 1 / self.physics.timestep:
            raise ValueError("camera_hz cannot exceed the physics rate")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key, factory in (("physics", PhysicsConfig), ("wind", WindConfig), ("task", TaskConfig), ("sensors", SensorConfig)):
            if key in value:
                value[key] = factory(**value[key])
        return cls(**value)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
