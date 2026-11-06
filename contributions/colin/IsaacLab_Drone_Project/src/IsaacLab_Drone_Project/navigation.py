
"""Target position settings for the drone."""

# Keep the original target initially.
TARGET_POSITION = (2.0, 0.0, 2)


def configure_target(env_cfg):
    """Apply a fixed target to the drone environment."""
    x, y, z = TARGET_POSITION

    ranges = env_cfg.commands.target_pose.ranges

    ranges.pos_x = (x, x)
    ranges.pos_y = (y, y)
    ranges.pos_z = (z, z)

    return env_cfg
