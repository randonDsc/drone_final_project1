"""Commands for controlled flights and original Isaac-style experiments."""
import argparse
from dataclasses import replace
import json

from .config import ExperimentConfig, SceneConfig

DEFAULT_TASKS = ["hover", "straight", "waypoints", "circle", "figure_eight"]
TASKS = DEFAULT_TASKS + ["altitude", "dijkstra"]
DEFAULT_WINDS = ["calm", "constant", "gust", "periodic", "turbulence", "randomized"]
ISAAC_WINDS = ["training_range", "unseen_gusts", "outside_range"]
WINDS = DEFAULT_WINDS + ISAAC_WINDS
REFERENCE_MODES = ["quintic_stops", "isaac_timed", "isaac_waypoints"]


def _config(args):
    if getattr(args, "config", None):
        config = ExperimentConfig.load(args.config)
    elif args.command in ("navigate", "plan") or (getattr(args, "scene", None) in ("blocks", "warehouse") and
            (getattr(args, "task", None) == "dijkstra" or getattr(args, "tasks", None) == ["dijkstra"])):
        from .scenes import navigation_preset
        config = navigation_preset(args.scene or "blocks")
    else:
        config = ExperimentConfig()
    if getattr(args, "scene", None):
        config.scene = SceneConfig(name=args.scene)
    if getattr(args, "protocol", None):
        config.protocol = args.protocol
    if getattr(args, "pid_family", None):
        config.physics = replace(config.physics, controller=args.pid_family)
    if getattr(args, "reference_mode", None):
        config.navigation = replace(config.navigation, reference_mode=args.reference_mode)
    if getattr(args, "duration", None) is not None:
        config = replace(config, max_seconds=args.duration, task=replace(config.task, hover_seconds=args.duration))
    if args.command in ("navigate", "plan"):
        config.task = replace(config.task, kind="dijkstra")
    if getattr(args, "task", None):
        config.task = replace(config.task, kind=args.task)
    if getattr(args, "start", None):
        config.task = replace(config.task, start=tuple(args.start))
    if getattr(args, "goal", None):
        config.task = replace(config.task, goal=tuple(args.goal))
    if getattr(args, "wind", None):
        config.wind = replace(config.wind, scenario=args.wind)
    if getattr(args, "seed", None) is not None:
        config = replace(config, seed=args.seed)
    if getattr(args, "capture_camera", False):
        config.sensors = replace(config.sensors, capture_camera=True)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Skydio X2 environmental flight experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Compile and audit the physical model")
    inspect.add_argument("--config")
    inspect.add_argument("--scene", choices=["open", "blocks", "warehouse"])
    run = commands.add_parser("run", help="Record a fixed-horizon flight")
    navigate = commands.add_parser("navigate", help="Plan and fly a Dijkstra obstacle mission")
    plan = commands.add_parser("plan", help="Save a collision-checked route before flight")
    suite = commands.add_parser("suite", help="Run paired-seed environmental experiments")
    for command in (run, navigate, plan, suite):
        command.add_argument("--config")
        command.add_argument("--output", required=True)
        command.add_argument("--scene", choices=["open", "blocks", "warehouse"])
        command.add_argument("--reference-mode", choices=REFERENCE_MODES)
        command.add_argument("--protocol", choices=["controlled", "isaac_path"])
        command.add_argument("--duration", type=float, help="Maximum seconds; also sets hover duration")
        command.add_argument("--pid-family", choices=["original_adapted", "isaac_pid", "original_literal"])
        command.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "Z"))
        command.add_argument("--goal", nargs=3, type=float, metavar=("X", "Y", "Z"))
        if command is not plan:
            command.add_argument("--model", help="MuJoCo SAC .zip with a compatible contract")
    for command in (run, navigate, plan):
        command.add_argument("--wind", choices=WINDS)
        command.add_argument("--seed", type=int)
        if command is not plan:
            command.add_argument("--capture-camera", action="store_true")
            command.add_argument("--viewer", action="store_true", help="Real-time viewer; on macOS use mjpython")
    run.add_argument("--task", choices=TASKS)
    suite.add_argument("--tasks", nargs="+", choices=TASKS)
    suite.add_argument("--winds", nargs="+", choices=WINDS)
    suite.add_argument("--seeds", nargs="+", type=int, default=[300000, 300001, 300002])
    training = commands.add_parser("train", help="Train or resume residual SAC on MuJoCo")
    training.add_argument("--output", required=True)
    training.add_argument("--config")
    training.add_argument("--steps", type=int, default=2_000_000)
    training.add_argument("--seed", type=int)
    training.add_argument("--resume")
    training.add_argument("--checkpoint-interval", type=int, default=5000)
    training.add_argument("--learning-starts", type=int)
    training.add_argument("--buffer-size", type=int)
    plot = commands.add_parser("plot", help="Save a flight, planned route or suite figure")
    plot.add_argument("directory")
    plot.add_argument("--output")
    comparison = commands.add_parser("compare", help="Compare verified paired metric CSVs")
    comparison.add_argument("left")
    comparison.add_argument("right")
    comparison.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        config = _config(args)
        if args.command == "inspect":
            from .physics import DroneSimulation
            sim = DroneSimulation(config.physics, config.wind, config.seed, scene=config.scene)
            result = sim.audit()
            result["scene"] = dict(name=config.scene.name, geoms=sim.model.ngeom,
                manifest_sha256=sim.scene_manifest["sha256"] if sim.scene_manifest else None)
            print(json.dumps(result, indent=2))
            sim.close()
        elif args.command in ("run", "navigate", "plan", "suite"):
            from .experiments import run_experiment, run_suite, plan_experiment
            if args.command == "suite":
                if len(set(args.seeds)) != len(args.seeds):
                    raise ValueError("Suite seeds must be unique")
                tasks = args.tasks or (DEFAULT_TASKS+["altitude"] if config.protocol == "isaac_path" else DEFAULT_TASKS)
                winds = args.winds or (ISAAC_WINDS if config.protocol == "isaac_path" else DEFAULT_WINDS)
                run_suite(config, args.output, tasks, winds, args.seeds, args.model)
            elif args.command == "plan":
                print(json.dumps(plan_experiment(config, args.output), indent=2))
            else:
                print(json.dumps(run_experiment(config, args.output, args.model, args.viewer), indent=2))
        elif args.command == "train":
            from .training import train
            print(train(args.output, args.steps, config.seed, args.resume, args.checkpoint_interval,
                        args.learning_starts, args.buffer_size, config if args.config else None))
        elif args.command == "plot":
            from pathlib import Path
            from .plotting import plot_run, plot_suite, plot_navigation
            directory = Path(args.directory)
            if (directory/"summary.csv").exists():
                function = plot_suite
            elif (directory/"navigation"/"plan.json").exists() and not (directory/"flight.csv").exists():
                function = plot_navigation
            else:
                function = plot_run
            print(function(directory, args.output))
        else:
            from .comparison import compare
            print(json.dumps(compare(args.left, args.right, args.output), indent=2))
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
