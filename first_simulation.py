import time
import mujoco
import mujoco.viewer

model_path = "mujoco_menagerie/unitree_go2/scene.xml"

# Load the robot and its simulation state
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Open the viewer and run physics for 20 seconds
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    while viewer.is_running() and time.time() - start < 20:
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(max(0, model.opt.timestep -
                       (time.time() - step_start)))

print("Simulation finished.")

