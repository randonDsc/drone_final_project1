"""What a saved policy's observations and baseline controller mean.

Two policies can share an observation shape and still be incompatible. With
state source "imu" the same 23 slots hold noisy estimates; with "ground_truth"
they hold perfect state. A residual also only fits the controller it was trained
against. SB3 checks only the shape, so it would load the wrong model without
complaint, and the normalization statistics would be wrong too.

The training scripts write observation_contract.json in the run folder. Standard
library only, so scripts can check this before Isaac Sim starts.
"""
import json
from pathlib import Path


FILE_NAME = "observation_contract.json"

# Bump when any PPO observation's meaning, order, or scale, or the baseline
# controller, changes. Version 3 is the PID baseline.
VERSION = 3

# PPO models saved before contracts existed, such as hover1.
LEGACY = {"version": 1, "state_source": "ground_truth"}


def make_contract(state_source, imu_noise, estimator, observation_size):
    return {
        "version": VERSION,
        "state_source": state_source,
        "observation_size": observation_size,
        "imu_noise": dict(imu_noise),
        "estimator": dict(estimator),
    }


def run_folder(model_path):
    """Run folder of a final model or of a file in its checkpoints folder."""
    folder = Path(model_path).parent

    return folder.parent if folder.name == "checkpoints" else folder


def read_contract(model_path):
    path = run_folder(model_path) / FILE_NAME

    if not path.is_file():
        return dict(LEGACY)

    return json.loads(path.read_text(encoding="utf-8"))


def write_contract(run_dir, contract):
    (Path(run_dir) / FILE_NAME).write_text(
        json.dumps(contract, indent=2),
        encoding="utf-8",
    )


def incompatibility(model_path, state_source):
    """Why this PPO model must not be used, or None if it may be."""
    contract = read_contract(model_path)

    if (
        contract.get("algorithm", "ppo") != "ppo"
        or contract["state_source"] != state_source
        or contract["version"] != VERSION
    ):
        return (
            f"{model_path}\n"
            f"was trained on state source '{contract.get('state_source')}' "
            f"(observation version {contract['version']}), but this run uses "
            f"'{state_source}' (version {VERSION}, PID baseline).\n"
            "Its inputs have a different meaning, and its correction was "
            "learned for another baseline controller. Train a new model."
        )

    return None


def setting_differences(model_path, imu_noise, estimator):
    """Sensor settings that differ from those the model was trained with."""
    contract = read_contract(model_path)
    differences = []

    for group, current in (("imu_noise", imu_noise), ("estimator", estimator)):
        trained = contract.get(group, {})

        for name, value in current.items():
            if name in trained and trained[name] != value:
                differences.append(
                    f"{group}.{name}: trained {trained[name]}, now {value}"
                )

    return differences


def sac_incompatibility(model_path, expected):
    """Why this model cannot be used as this SAC residual, or None.

    expected: hover_sac_config.contract(). Every value must match exactly, because a
    SAC policy has no separate normalization to absorb a change.
    """
    path = run_folder(model_path) / FILE_NAME

    if not path.is_file():
        return (
            f"{model_path}\n"
            f"has no {FILE_NAME}, so it is not a SAC residual from "
            "train_hover_sac.py. PPO models cannot be loaded into SAC."
        )

    saved = json.loads(path.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(expected))

    different = sorted(
        name for name, value in expected.items() if saved.get(name) != value
    )

    if different:
        return (
            f"{model_path}\n"
            f"does not match the current settings: {', '.join(different)}.\n"
            "Train a new model, or restore the settings it was trained with."
        )

    return None
