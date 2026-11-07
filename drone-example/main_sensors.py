import time
import numpy as np
import json

from pathlib import Path
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt

import mujoco
import mujoco.viewer
import pickle

from simple_pid import PID

def save_data(filename, positions, velocities):
    data = {'positions': positions, 'velocities': velocities}
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def pid_to_thrust(input: np.array):
  """ Maps controller output to manipulated variable.

  Args:
      input (np.array): w € [3x1]

  Returns:
      np.array: [3x4]
  """
  c_to_F =np.array([
      [-0.25, 0.25, 0.25, -0.25],
      [0.25, 0.25, -0.25, -0.25],
      [-0.25, 0.25, -0.25, 0.25]
  ]).transpose()

  return np.dot((c_to_F*input),np.array([1,1,1]))

def outer_pid_to_thrust(input: np.array):
  """ Maps controller output to manipulated variable.

  Args:
      input (np.array): w € [3x1]

  Returns:
      np.array: [3x4]
  """
  c_to_F =np.array([
      [0.25, 0.25, -0.25, -0.25],
      [0.25, -0.25, -0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25]
  ]).transpose()

  return np.dot((c_to_F*input),np.array([1,1,1]))

class PDController:
  def __init__(self, kp, kd, setpoint):
    self.kp = kp
    self.kd = kd
    self.setpoint = setpoint
    self.prev_error = 0

  def compute(self, measured_value):
    error = self.setpoint - measured_value
    derivative = error - self.prev_error
    output = (self.kp * error) + (self.kd * derivative)
    self.prev_error = error
    return output

class PIDController:
  def __init__(self, kp, ki, kd, setpoint):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.setpoint = setpoint
    self.prev_error = 0
    self.integral = 0

  def compute(self, measured_value):
    error = self.setpoint - measured_value
    self.integral += error
    derivative = error - self.prev_error
    output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
    self.prev_error = error
    return output

class dummyPlanner:
  """Generate Path from 1 point directly to another"""

  def __init__(self, target, vel_limit = 2) -> None:
    # TODO: MPC
    self.target = target  
    self.vel_limit = vel_limit
    # setpoint target location, controller output: desired velocity.
    self.pid_x = PID(2, 0.15, 1.5, setpoint = self.target[0],
                output_limits = (-vel_limit, vel_limit),)
    self.pid_y = PID(2, 0.15, 1.5, setpoint = self.target[1],
                output_limits = (-vel_limit, vel_limit))
  
  def __call__(self, loc: np.array):
    """Calls planner at timestep to update cmd_vel"""
    velocites = np.array([0,0,0])
    velocites[0] = self.pid_x(loc[0])
    velocites[1] = self.pid_y(loc[1])
    return velocites

  def get_velocities(self,loc: np.array, target: np.array,
                     time_to_target: float = None,
                     flight_speed: float = 0.5) -> np.array:
    """Compute

    Args:
        loc (np.array): Current location in world coordinates.
        target (np.array): Desired location in world coordinates
        time_to_target (float): If set, adpats length of velocity vector.

    Returns:
        np.array: returns velocity vector in world coordinates.
    """

    direction = target - loc
    distance = np.linalg.norm(direction)
    # maps drone velocities to one.
    if distance > 1:
        velocities = flight_speed * direction / distance

    else:
        velocities =  direction * distance

    return velocities

  def get_alt_setpoint(self, loc: np.array) -> float:

    target = self.target
    distance = target[2] - loc[2]
    
    # maps drone velocities to one.
    if distance > 0.5:
        time_sample = 1/4
        time_to_target =  distance / self.vel_limit
        number_steps = int(time_to_target/time_sample)
        # compute distance for next update
        delta_alt = distance / number_steps

        # 2 times for smoothing
        alt_set = loc[2] + 2 * delta_alt
    
    else:
        alt_set = target[2]

    return alt_set

  def update_target(self, target):
    """Update targets"""
    self.target = target  
    # setpoint target location, controller output: desired velocity.
    self.pid_x.setpoint = self.target[0]
    self.pid_y.setpoint = self.target[1]

class dummySensor:
  """Dummy sensor data. So the control code remains intact."""
  def __init__(self, d):
    self.position = d.qpos
    self.velocity = d.qvel
    self.acceleration = d.qacc

  def get_position(self):
    return self.position
  
  def get_velocity(self):
    return self.velocity
  
  def get_acceleration(self):
    return self.acceleration

class drone:
  """Simple drone classe."""
  def __init__(self, target=np.array((0,0,0))):
    self.m = mujoco.MjModel.from_xml_path('mujoco_menagerie-main/skydio_x2/scene.xml')
    self.d = mujoco.MjData(self.m)

    self.planner = dummyPlanner(target=target)
    self.sensor = dummySensor(self.d)

    # instantiate controllers

    # inner control to stabalize inflight dynamics
    self.pid_alt = PID(5.50844,0.57871, 1.2,setpoint=0,) # PIDController(0.050844,0.000017871, 0, 0) # thrust
    self.pid_roll = PID(2.6785,0.56871, 1.2508, setpoint=0, output_limits = (-1,1) ) #PID(11.0791,2.5263, 0.10513,setpoint=0, output_limits = (-1,1) )
    self.pid_pitch = PID(2.6785,0.56871, 1.2508, setpoint=0, output_limits = (-1,1) )
    self.pid_yaw =  PID(0.54, 0, 5.358333, setpoint=1, output_limits = (-3,3) )# PID(0.11046, 0.0, 15.8333, setpoint=1, output_limits = (-2,2) )

    # outer control loops
    self.pid_v_x = PID(0.1, 0.003, 0.02, setpoint = 0,
                output_limits = (-0.1, 0.1))
    self.pid_v_y = PID(0.1, 0.003, 0.02, setpoint = 0,
                  output_limits = (-0.1, 0.1))

  def update_outer_conrol(self):
    """Updates outer control loop for trajectory planning"""
    v = self.sensor.get_velocity()
    location = self.sensor.get_position()[:3]

    # Compute velocites to target
    velocites = self.planner(loc=location)
    
    # In this example the altitude is directly controlled by a PID
    self.pid_alt.setpoint = self.planner.get_alt_setpoint(location)
    self.pid_v_x.setpoint = velocites[0]
    self.pid_v_y.setpoint = velocites[1]

    # Compute angles and set inner controllers accordingly
    angle_pitch = self.pid_v_x(v[0])
    angle_roll = - self.pid_v_y(v[1])

    self.pid_pitch.setpoint= angle_pitch
    self.pid_roll.setpoint = angle_roll

  def update_inner_control(self):
    """Upates inner control loop and sets actuators to stabilize flight
    dynamics"""
    alt = self.sensor.get_position()[2]
    angles = self.sensor.get_position()[3:] # roll, yaw, pitch
    
    # apply PID
    cmd_thrust = self.pid_alt(alt) + 3.2495
    cmd_roll = - self.pid_roll(angles[1])
    cmd_pitch = self.pid_pitch(angles[2])
    cmd_yaw = - self.pid_yaw(angles[0])

    #transfer to motor control
    out = self.compute_motor_control(cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw)
    self.d.ctrl[:4] = out

  #  as the drone is underactuated we set
  def compute_motor_control(self, thrust, roll, pitch, yaw):
    motor_control = [
      thrust + roll + pitch - yaw,
      thrust - roll + pitch + yaw,
      thrust - roll -  pitch - yaw,
      thrust + roll - pitch + yaw
    ]
    return motor_control

# ---------------- Initialization and recording ----------------

my_drone = drone(target=np.array((0, 0, 1)))

output_dir = Path("sensor_outputs")
output_dir.mkdir(exist_ok=True)

# Confirm the model has the required camera and IMU sensors.
assert my_drone.m.camera("drone_camera").id >= 0

for sensor_name in ("body_gyro", "body_linacc"):
    assert my_drone.m.sensor(sensor_name).id >= 0

# Record a camera frame every 20 physics steps.
# At dt=0.01 seconds, this corresponds to 5 FPS of simulation time.
capture_interval = 20
capture_fps = 1.0 / (
    capture_interval * my_drone.m.opt.timestep
)

imu_log = []
rgb_frames = []
depth_frames = []
frame_times = []

renderer = mujoco.Renderer(
    my_drone.m,
    height=240,
    width=320
)

try:
    with mujoco.viewer.launch_passive(
        my_drone.m, my_drone.d
    ) as viewer:

        time.sleep(5)
        start = time.time()
        step = 1

        while viewer.is_running() and time.time() - start < 30:
            step_start = time.time()
            elapsed = time.time() - start

            # Keep the original flight path.
            if elapsed > 18:
                my_drone.planner.update_target(
                    np.array((-1, -1, 0.5))
                )
            elif elapsed > 10:
                my_drone.planner.update_target(
                    np.array((-1, 1, 2))
                )
            elif elapsed > 2:
                my_drone.planner.update_target(
                    np.array((1, 1, 1))
                )

            # Original drone controllers.
            if step % 20 == 0:
                my_drone.update_outer_conrol()

            my_drone.update_inner_control()

            # Advance the physics.
            mujoco.mj_step(my_drone.m, my_drone.d)

            d = my_drone.d

            # Capture IMU data at every physics step.
            gyro = d.sensor("body_gyro").data.copy()
            accel = d.sensor("body_linacc").data.copy()

            imu_log.append({
                "time": d.time,
                "x": d.qpos[0],
                "y": d.qpos[1],
                "z": d.qpos[2],
                "gyro_x": gyro[0],
                "gyro_y": gyro[1],
                "gyro_z": gyro[2],
                "accel_x": accel[0],
                "accel_y": accel[1],
                "accel_z": accel[2],
            })

            # Capture RGB and depth from the same camera pose.
            if step % capture_interval == 0:
                renderer.disable_depth_rendering()
                renderer.update_scene(
                    d, camera="drone_camera"
                )
                rgb_frames.append(
                    renderer.render().copy()
                )

                renderer.enable_depth_rendering()
                renderer.update_scene(
                    d, camera="drone_camera"
                )
                depth_frames.append(
                    renderer.render().copy()
                )

                frame_times.append(d.time)

            # Update the interactive viewer.
            viewer.sync()
            step += 1

            # Approximate real-time playback.
            remaining = (
                my_drone.m.opt.timestep
                - (time.time() - step_start)
            )
            if remaining > 0:
                time.sleep(remaining)

finally:
    renderer.close()

    # Save collected data even if recording is interrupted.
    df = pd.DataFrame(imu_log)
    df.to_csv(output_dir / "imu_data.csv", index=False)

    if rgb_frames:
        imageio.mimsave(
            output_dir / "drone_rgb.mp4",
            rgb_frames,
            fps=capture_fps
        )

    if depth_frames:
        np.savez_compressed(
            output_dir / "depth_frames.npz",
            depth=np.stack(depth_frames),
            timestamps=np.array(frame_times)
        )

        # Visualize one depth frame.
        middle = len(depth_frames) // 2
        plt.imsave(
            output_dir / "depth_sample.png",
            depth_frames[middle],
            cmap="viridis"
        )

    print("Saved results to:", output_dir.resolve())
    print("IMU samples:", len(imu_log))
    print("RGB frames:", len(rgb_frames))
    print("Depth frames:", len(depth_frames))
    print("Camera FPS (simulation time):", capture_fps)