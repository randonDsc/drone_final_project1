"""Read the indoor sensors: LiDAR scans and the downward rangefinder.

These readers produce measurements and nothing more. They are not connected to
the estimator, controller, or policy, and this is not odometry: turning scans
and ranges into a position needs the LiDAR-inertial odometry described in
INDOOR_SENSORS.md, which does not exist yet.

Only measurement channels are read. The raycast sensor's hit positions, normals,
and prim paths, and the LiDAR's object IDs, velocities, and hit normals, are
simulator knowledge a real sensor would not report.

The processing functions are NumPy only. The reader classes need SimulationApp
running; import of this module does not.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class LidarScan:
    """One complete sweep, in the sensor frame."""

    points: np.ndarray             # Nx3 [m]
    point_times: np.ndarray        # N, seconds since the first point in the sweep
    start_time_ns: int             # the sensor's timestamp of the first point
    intensity: np.ndarray          # N
    channel_elevation: np.ndarray  # N, nearest whole degree, in place of a ring id

    def __len__(self):
        return len(self.points)


def clean_scan(points, timestamps_ns, distances, intensity, elevation,
               min_range):
    """Drop invalid returns; recorded scans contain points at distance 0."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
    distances = np.asarray(distances, dtype=float)

    if len(points) == 0:
        return None

    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(distances)
        & (distances >= min_range)
    )

    if not valid.any():
        return None

    start = int(timestamps_ns[valid].min())

    return LidarScan(
        points=points[valid],
        point_times=(timestamps_ns[valid] - start) * 1e-9,
        start_time_ns=start,
        intensity=np.asarray(intensity, dtype=float)[valid],
        channel_elevation=np.rint(
            np.asarray(elevation, dtype=float)[valid]
        ).astype(int),
    )


def channel_counts(scan):
    """{elevation degrees: returns}, highest channel first."""
    channels, counts = np.unique(scan.channel_elevation, return_counts=True)

    return {
        int(channel): int(count)
        for channel, count in zip(channels[::-1], counts[::-1])
    }


class RangefinderModel:
    """Turn a raw ray depth into what the sensor would report, at its rate."""

    def __init__(self, settings, rng):
        self.min_range = float(settings["min_range"])
        self.max_range = float(settings["max_range"])
        self.period = 1.0 / float(settings["update_rate_hz"])
        self.noise_std = float(settings["noise_std"])
        self.noise_fraction = float(settings["noise_fraction"])
        self.resolution = float(settings["resolution"])
        self.dropout = float(settings["dropout_probability"])
        self.rng = rng
        self.next_time = 0.0

    def due(self, time_s):
        """True once per sample period; a late sample does not shift the rest."""
        if time_s + 1e-9 < self.next_time:
            return False

        self.next_time = max(self.next_time + self.period, time_s)

        return True

    def measure(self, depth):
        """Reported range [m], or None for no return.

        A ray that hits nothing reports about max_range, not a flag.
        """
        if (
            not np.isfinite(depth)
            or depth < self.min_range
            or depth >= 0.999 * self.max_range
            or self.rng.random() < self.dropout
        ):
            return None

        std = self.noise_std + self.noise_fraction * depth
        noisy = depth + std * self.rng.normal()

        if self.resolution > 0:
            noisy = self.resolution * round(noisy / self.resolution)

        return float(np.clip(noisy, self.min_range, self.max_range))


class RangefinderReader:
    """The IsaacRaycastSensor prim. Works without rendering."""

    def __init__(self, stage, settings, rng):
        import omni.kit.app

        path = settings["path"]
        prim = stage.GetPrimAtPath(path)

        if not prim.IsValid() or prim.GetTypeName() != "IsaacRaycastSensor":
            raise RuntimeError(f"Rangefinder prim not found: {path}")

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            "isaacsim.sensors.experimental.physics", True
        )

        from isaacsim.sensors.experimental.physics import RaycastSensor

        self.sensor = RaycastSensor(path)
        self.model = RangefinderModel(settings, rng)

    def read(self, time_s):
        """Latest range [m], or None if not due, not ready, or no return."""
        if not self.model.due(time_s):
            return None

        reading = self.sensor.get_sensor_reading()

        if not reading.is_valid or len(reading.depths) == 0:
            return None

        return self.model.measure(float(reading.depths[0]))


class LidarScanReader:
    """The RTX LiDAR. Only produces scans while the simulation is rendering."""

    def __init__(self, stage, settings):
        import omni.kit.app
        import omni.replicator.core as rep

        path = settings["path"]

        if not stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(f"LiDAR prim not found: {path}")

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            "isaacsim.sensors.rtx.nodes", True
        )

        self.min_range = float(settings["min_valid_range"])
        self.last_start = None

        self.render_product = rep.create.render_product(path, (1, 1))
        self.annotator = rep.AnnotatorRegistry.get_annotator(
            "IsaacCreateRTXLidarScanBuffer",
            init_params={
                "outputDistance": True,
                "outputElevation": True,
                "outputIntensity": True,
                "outputTimestamp": True,
            },
        )
        self.annotator.attach([self.render_product.path])

    def read(self):
        """The newest full sweep, or None if it has been returned already.

        The annotator repeats a sweep until the next one finishes, so reading
        faster than 10 Hz would otherwise feed odometry duplicates.
        """
        data = self.annotator.get_data()

        if not isinstance(data, dict) or len(data.get("data", ())) == 0:
            return None

        scan = clean_scan(
            data["data"],
            data["timestamp"],
            data["distance"],
            data["intensity"],
            data["elevation"],
            self.min_range,
        )

        if scan is None or scan.start_time_ns == self.last_start:
            return None

        self.last_start = scan.start_time_ns

        return scan

    def close(self):
        self.annotator.detach([self.render_product.path])
        self.render_product.destroy()
