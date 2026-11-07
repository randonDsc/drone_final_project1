import time
import numpy as np
import json

import mujoco
import mujoco.viewer
import pickle

from simple_pid import PID
from scipy.spatial.transform import Rotation

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
    velocites[0] = self.pid_x(loc[0], dt=0.20)
    velocites[1] = self.pid_y(loc[1], dt=0.20)
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
    self.pid_yaw =  PID(0.54, 0, 5.358333, setpoint=0, output_limits = (-3,3) )# PID(0.11046, 0.0, 15.8333, setpoint=1, output_limits = (-2,2) )

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
    angle_pitch = self.pid_v_x(v[0], dt=0.20)
    angle_roll = -self.pid_v_y(v[1], dt=0.20)

    self.pid_pitch.setpoint= angle_pitch
    self.pid_roll.setpoint = angle_roll

  def update_inner_control(self):
    """Upates inner control loop and sets actuators to stabilize flight
    dynamics"""
    alt = self.sensor.get_position()[2]
    #angles = self.sensor.get_position()[3:] # roll, yaw, pitch
    
    # MuJoCo quaternion order: [w, x, y, z]
    quat = self.sensor.get_position()[3:7]
    
    # SciPy quaternion order: [x, y, z, w]
    scipy_quat = [quat[1], quat[2], quat[3], quat[0]]
    
    # Convert to roll, pitch, yaw in radians
    roll, pitch, yaw = Rotation.from_quat(
        scipy_quat
    ).as_euler("xyz", degrees=False)
    
    # apply PID
    cmd_thrust = self.pid_alt(alt, dt=0.01) + 3.2495
    cmd_roll = -self.pid_roll(roll, dt=0.01)
    cmd_pitch = self.pid_pitch(pitch, dt=0.01)
    cmd_yaw = -self.pid_yaw(yaw, dt=0.01)

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

# ============================================================
# POINT-TO-POINT DRONE FLIGHT WITH WIND DISTURBANCES
# ============================================================

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("wind_results/corrected")
OUTPUT_DIR.mkdir(exist_ok=True)

START = np.array([0.0, 0.0, 1.0])
GOAL = np.array([5.0, 0.0, 1.0])

MAX_TIME = 35.0
GUST_START = 5.0
GUST_END = 7.0

# Success criteria
GOAL_TOLERANCE = 0.30       # meters
SPEED_TOLERANCE = 0.30      # meters per second
SUCCESS_HOLD_TIME = 1.0    # seconds

# Stop if the drone falls near the ground after takeoff.
CRASH_ALTITUDE = 0.12


def get_wind_force(t, scenario):
    """
    Return the external disturbance force in world coordinates.

    This is a force in newtons, not wind velocity in m/s.
    """
    if scenario == "calm":
        return np.array([0.0, 0.0, 0.0])

    if scenario == "gust":
        if GUST_START <= t < GUST_END:
            return np.array([0.0, 0.05, 0.0])
        return np.array([0.0, 0.0, 0.0])

    if scenario == "strong_gust":
        if GUST_START <= t < GUST_END:
            return np.array([0.0, 0.10, 0.0])
        return np.array([0.0, 0.0, 0.0])

    raise ValueError(f"Unknown scenario: {scenario}")


def run_experiment(scenario):
    print(f"\nRunning scenario: {scenario}")

    # New drone and controller for every experiment.
    my_drone = drone(target=START.copy())

    model = my_drone.m
    data = my_drone.d

    dt = model.opt.timestep

    if not np.isclose(dt, 0.01):
        raise ValueError(
            "This script expects an XML timestep of 0.01 seconds."
        )

    # Start from the drone's existing hover keyframe.
    hover_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "hover"
    )

    if hover_id < 0:
        raise ValueError("Hover keyframe not found in drone XML.")

    mujoco.mj_resetDataKeyframe(model, data, hover_id)
    mujoco.mj_forward(model, data)

    # Find the drone body so we can apply external forces.
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "x2"
    )

    if body_id < 0:
        raise ValueError("Drone body 'x2' not found.")

    records = []
    step = 0
    hold_duration = 0.0
    success = False
    crashed = False

    # Give the drone 2 seconds to approach the starting hover.
    TAKEOFF_TIME = 2.0

    while data.time < MAX_TIME:

        t = float(data.time)

        # Fly toward START first, then toward GOAL.
        if t < TAKEOFF_TIME:
            target = START
        else:
            target = GOAL

        # Avoid unnecessarily resetting the planner target.
        if not np.array_equal(my_drone.planner.target, target):
            my_drone.planner.update_target(target.copy())

        # Existing PID flight controller.
        if step % 20 == 0:
            my_drone.update_outer_conrol()

        my_drone.update_inner_control()

        # Apply a reproducible external force.
        force = get_wind_force(t, scenario)

        data.xfrc_applied[body_id, :] = 0.0
        data.xfrc_applied[body_id, :3] = force

        # Advance MuJoCo physics by one step.
        mujoco.mj_step(model, data)

        position = data.qpos[:3].copy()
        velocity = data.qvel[:3].copy()

        gyro = data.sensor("body_gyro").data.copy()
        accel = data.sensor("body_linacc").data.copy()

        distance_to_goal = float(
            np.linalg.norm(position - GOAL)
        )
        speed = float(np.linalg.norm(velocity))

        # Record the actual drone state and controller commands.
        records.append({
            "time": float(data.time),
            "x": position[0],
            "y": position[1],
            "z": position[2],
            "vx": velocity[0],
            "vy": velocity[1],
            "vz": velocity[2],
            "goal_x": GOAL[0],
            "goal_y": GOAL[1],
            "goal_z": GOAL[2],
            "distance_to_goal": distance_to_goal,
            "speed": speed,
            "force_x": force[0],
            "force_y": force[1],
            "force_z": force[2],
            "gyro_x": gyro[0],
            "gyro_y": gyro[1],
            "gyro_z": gyro[2],
            "accel_x": accel[0],
            "accel_y": accel[1],
            "accel_z": accel[2],
            "motor_1": data.ctrl[0],
            "motor_2": data.ctrl[1],
            "motor_3": data.ctrl[2],
            "motor_4": data.ctrl[3],
        })

        # Basic crash detection.
        # Only check after the initial takeoff period.
        if data.time >= TAKEOFF_TIME:
            if position[2] < CRASH_ALTITUDE:
                crashed = True
                print(f"  Crash detected at t={data.time:.2f}s")
                break

        # Success requires arriving near B and staying slow.
        if data.time >= TAKEOFF_TIME:
            if (
                distance_to_goal < GOAL_TOLERANCE
                and speed < SPEED_TOLERANCE
            ):
                hold_duration += dt
            else:
                hold_duration = 0.0

            if hold_duration >= SUCCESS_HOLD_TIME:
                success = True
                print(f"  Goal reached at t={data.time:.2f}s")
                break

        step += 1

    df = pd.DataFrame(records)
    df.to_csv(
        OUTPUT_DIR / f"{scenario}_flight.csv",
        index=False
    )

    result = {
        "scenario": scenario,
        "success": success,
        "crashed": crashed,
        "flight_time": float(data.time),
        "final_goal_error": float(
            np.linalg.norm(data.qpos[:3] - GOAL)
        ),
        "maximum_lateral_deviation": float(
            df["y"].abs().max()
        ),
        "minimum_altitude": float(df["z"].min()),
    }

    print("  Success:", result["success"])
    print("  Final goal error:",
          round(result["final_goal_error"], 3), "m")
    print("  Maximum lateral deviation:",
          round(result["maximum_lateral_deviation"], 3), "m")

    return df, result


# ============================================================
# RUN CALM AND GUST EXPERIMENTS
# ============================================================

all_results = {}

for scenario in ["calm", "gust", "strong_gust"]:
    flight_df, summary = run_experiment(scenario)
    all_results[scenario] = summary

summary_df = pd.DataFrame(all_results.values())

summary_df.to_csv(
    OUTPUT_DIR / "experiment_summary.csv",
    index=False
)

print("\nEXPERIMENT SUMMARY")
print(summary_df.to_string(index=False))


# ============================================================
# PLOT TRAJECTORIES
# ============================================================

plt.figure(figsize=(10, 6))

for scenario in ["calm", "gust", "strong_gust"]:
    df = pd.read_csv(
        OUTPUT_DIR / f"{scenario}_flight.csv"
    )
    plt.plot(df["x"], df["y"], label=scenario)

plt.scatter(
    [START[0], GOAL[0]],
    [START[1], GOAL[1]],
    color="black",
    marker="x",
    s=100,
    label="Start / Goal"
)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Drone flight trajectories under disturbances")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "trajectory_comparison.png",
    dpi=150
)
plt.show()
