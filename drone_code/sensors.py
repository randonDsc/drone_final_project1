"""Optional IMU/LiDAR connection, reading and cleanup from the original project.

These readings are logged for inspection. They do not feed the occupancy map
or controller yet; both use simulator ground truth in this static-map baseline.
"""
import numpy as np


class SensorSuite:
    def __init__(self, stage, enabled=True, log=print):
        self.log = log
        self.imu = None
        self.lidar = None
        self.render_product = None
        if not enabled:
            log('Sensor logging disabled in config.py.')
            return
        import omni.kit.app
        manager = omni.kit.app.get_app().get_extension_manager()
        imu_path = '/World/iris/body/imu'
        lidar_path = '/World/iris/body/HESAI_XT32_SD10/PandarXT_32_10hz'
        # Treat each sensor separately: a missing sensor is not an obstacle-map
        # failure. Report API errors explicitly instead of claiming it connected.
        if stage.GetPrimAtPath(imu_path).IsValid():
            try:
                manager.set_extension_enabled_immediate(
                    'isaacsim.sensors.experimental.physics', True)
                from isaacsim.sensors.experimental.physics import IMUSensor
                self.imu = IMUSensor(imu_path)
                log(f'IMU connected: {imu_path}')
            except Exception as exc:
                log(f'IMU logging unavailable: {exc}')
        else:
            log(f'IMU absent; continuing without sensor logging: {imu_path}')
        if stage.GetPrimAtPath(lidar_path).IsValid():
            try:
                manager.set_extension_enabled_immediate('isaacsim.sensors.rtx.nodes', True)
                import omni.replicator.core as rep
                self.render_product = rep.create.render_product(lidar_path, (1, 1))
                self.lidar = rep.AnnotatorRegistry.get_annotator(
                    'IsaacExtractRTXSensorPointCloud', device='cpu')
                self.lidar.attach([self.render_product.path])
                log(f'LiDAR connected: {lidar_path}')
            except Exception as exc:
                log(f'LiDAR logging unavailable: {exc}')
                self.close_lidar()
        else:
            log(f'LiDAR absent; continuing without sensor logging: {lidar_path}')

    def log_readings(self):
        if self.imu is not None:
            reading = self.imu.get_sensor_reading()
            if reading.is_valid:
                frame = self.imu.get_data()
                self.log(f"IMU time [s]: {frame['time']}\n"
                         f"IMU acceleration [m/s^2]: {frame['linear_acceleration']}\n"
                         f"IMU angular velocity [rad/s]: {frame['angular_velocity']}\n"
                         f"IMU orientation [w,x,y,z]: {frame['orientation']}")
            else:
                self.log('IMU: no valid reading yet.')
        if self.lidar is not None:
            data = self.lidar.get_data()
            points = data.get('data') if isinstance(data, dict) else data
            points = np.asarray(points) if points is not None else np.empty((0, 3))
            self.log(f'LiDAR point array shape: {points.shape}')
            if points.size:
                self.log(f'LiDAR first 3 points:\n{points[:3]}')

    def close_lidar(self):
        try:
            if self.lidar is not None and self.render_product is not None:
                self.lidar.detach([self.render_product.path])
        finally:
            self.lidar = None
            if self.render_product is not None:
                self.render_product.destroy()
                self.render_product = None

    def close(self):
        self.close_lidar()
