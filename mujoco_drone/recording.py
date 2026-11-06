"""Streaming flight logs, with explicit frames, units, and control intervals."""
import csv
from pathlib import Path

import numpy as np

from .sensors import read_imu


def columns(row, prefix, values, axes="xyz"):
    for name, value in zip(axes, np.asarray(values).ravel()):
        row[f"{prefix}_{name}"] = float(value)


class FlightRecorder:
    def __init__(self, path):
        self.file = Path(path).open("w", newline="")
        self.writer = None

    def record(self, env, state, sample):
        sim, core = env.sim, env.core
        last = sim.last
        row = dict(time_s=float(sim.data.time),
                   control_interval_start_s=last["interval_start"],
                   control_interval_end_s=last["interval_end"],
                   any_motor_saturated=int(last["saturated"]))
        for prefix, values, axes in (
            ("truth_origin_position_m", state["position"], "xyz"),
            ("truth_quaternion", state["orientation"], "wxyz"),
            ("truth_origin_velocity_m_s", state["linear_velocity"], "xyz"),
            ("truth_origin_acceleration_m_s2", state["origin_acceleration"], "xyz"),
            ("truth_com_position_m", state["com_position"], "xyz"),
            ("truth_com_velocity_m_s", state["com_velocity"], "xyz"),
            ("truth_com_acceleration_m_s2", state["com_acceleration"], "xyz"),
            ("truth_angular_velocity_world_rad_s", state["angular_velocity"], "xyz"),
            ("truth_angular_velocity_body_rad_s", state["body_angular_velocity"], "xyz"),
            ("reference_position_m", sample.position, "xyz"),
            ("reference_velocity_m_s", sample.velocity, "xyz"),
            ("reference_acceleration_m_s2", sample.acceleration, "xyz"),
            ("wind_mean_world_m_s", sim.wind.mean_wind, "xyz"),
            ("wind_shared_gust_m_s", sim.wind.shared_gust, "xyz"),
            ("wind_area_mean_world_m_s", sim.wind.area_weights @ last["wind"], "xyz"),
            ("drag_force_world_N", last["force"], "xyz"),
            ("drag_torque_world_about_com_Nm", last["torque"], "xyz"),
        ):
            columns(row, prefix, values, axes)
        for prefix, values in read_imu(sim).items():
            columns(row, "sensor_"+prefix, values)
        for i in range(4):
            row[f"motor_{i+1}_requested_N"] = float(last["requested"][i])
            row[f"motor_{i+1}_command_N"] = float(last["controls"][i])
            row[f"motor_{i+1}_actuator_force_N"] = float(sim.data.actuator_force[sim.rotor_ids[i]])
            row[f"motor_{i+1}_saturated"] = int(abs(last["requested"][i]-last["controls"][i]) > 1e-10)
            row[f"motor_{i+1}_at_limit"] = int(last["controls"][i] <= 1e-9 or last["controls"][i] >= sim.control_range[i, 1]-1e-9)
            row[f"action_{i}"] = float(core.action[i])
            row[f"pid_command_{i}"] = float(last["command"][i]-core.correction[i])
            row[f"residual_{i}"] = float(core.correction[i])
        for i, name in enumerate(sim.wind.point_names):
            columns(row, f"wind_{name}_world_m_s", last["wind"][i])
            columns(row, f"drag_{name}_world_N", last["point_forces"][i])
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)

    def close(self):
        self.file.close()
