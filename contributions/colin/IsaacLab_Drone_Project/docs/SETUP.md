# Setup for this checkpoint

Tested on Windows, NVIDIA RTX 4060 Laptop GPU, Python 3.12, uv, locked project dependencies, and Isaac Lab revision `0c114a24a3b9e854d299d3527ada507139fc4e40`. The Isaac Lab checkout was clean when recorded. Isaac Sim is selected through the isaacsim extra in uv.lock; arbitrary Isaac Lab releases may use different APIs.

The pyproject uses editable paths to a sibling checkout:

```text
parent/
  IsaacLab/
  IsaacLab_Drone_Project/  # pyproject.toml, scripts, src
```

For the group contribution, parent is `contributions/colin`. For Colin's existing installation it is Desktop. Run commands inside IsaacLab_Drone_Project, not the group repository root.

From parent, if you do not already have the required dependency checkout:

```powershell
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab
git -C IsaacLab checkout 0c114a24a3b9e854d299d3527ada507139fc4e40
cd IsaacLab_Drone_Project
uv sync --extra isaacsim --locked
```

This recipe reflects the verified local layout; a clean-machine installation has not been tested. NVIDIA runtime/driver prerequisites and remote USD asset access are required. Isaac Sim, Isaac Lab, the virtual environment, and checkpoints are not bundled. Create a new environment with uv on each machine.

Confirm the revision using `git -C ../IsaacLab rev-parse HEAD`. Keep `--extra isaacsim` on simulation commands.

The older direct-PPO playback helper needs checkpoints/drone_ppo.pt, excluded from this submission and unnecessary for Lee/PID validation.
