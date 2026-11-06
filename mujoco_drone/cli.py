"""Command line entry points; run from any directory after installation."""
import argparse
from dataclasses import replace
import json

from .config import ExperimentConfig


TASKS = ["hover", "straight", "waypoints", "circle", "figure_eight"]
WINDS = ["calm", "constant", "gust", "periodic", "turbulence", "randomized"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Skydio X2 environmental flight experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Compile and audit the physical model")
    inspect.add_argument("--config")
    run = commands.add_parser("run", help="Record a fixed-horizon flight")
    suite = commands.add_parser("suite", help="Run paired-seed environmental experiments")
    for command in (run, suite):
        command.add_argument("--config")
        command.add_argument("--output", required=True)
        command.add_argument("--model", help="MuJoCo SAC .zip with its contract")
        command.add_argument("--duration", type=float, help="Maximum seconds; also sets hover duration")
        command.add_argument("--pid-family", choices=["original_adapted", "isaac_pid", "original_literal"])
    run.add_argument("--task", choices=TASKS)
    run.add_argument("--wind", choices=WINDS)
    run.add_argument("--seed", type=int)
    run.add_argument("--capture-camera", action="store_true")
    run.add_argument("--viewer", action="store_true", help="On macOS, launch using mjpython")
    suite.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    suite.add_argument("--winds", nargs="+", choices=WINDS, default=WINDS)
    suite.add_argument("--seeds", nargs="+", type=int, default=[300000, 300001, 300002])
    training = commands.add_parser("train", help="Train or resume residual SAC on MuJoCo")
    training.add_argument("--output", required=True)
    training.add_argument("--config")
    training.add_argument("--steps", type=int, default=2_000_000, help="Additional SAC decisions")
    training.add_argument("--seed", type=int)
    training.add_argument("--resume")
    training.add_argument("--checkpoint-interval", type=int, default=5000)
    training.add_argument("--learning-starts", type=int)
    training.add_argument("--buffer-size", type=int)
    plot = commands.add_parser("plot", help="Save a flight or suite figure")
    plot.add_argument("directory")
    plot.add_argument("--output")
    comparison = commands.add_parser("compare", help="Compare verified paired metric CSVs")
    comparison.add_argument("left")
    comparison.add_argument("right")
    comparison.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        config = ExperimentConfig.load(args.config) if getattr(args, "config", None) else ExperimentConfig()
        if args.command == "inspect":
            from .physics import DroneSimulation
            sim = DroneSimulation(config.physics, config.wind, config.seed)
            print(json.dumps(sim.audit(), indent=2))
            sim.close()
        elif args.command in ("run", "suite"):
            from .experiments import run_experiment, run_suite
            if args.pid_family:
                config.physics = replace(config.physics, controller=args.pid_family)
            if args.duration is not None:
                config = replace(config, max_seconds=args.duration,
                                 task=replace(config.task, hover_seconds=args.duration))
            if args.command == "run":
                if args.task:
                    config.task = replace(config.task, kind=args.task)
                if args.wind:
                    config.wind = replace(config.wind, scenario=args.wind)
                if args.seed is not None:
                    config = replace(config, seed=args.seed)
                if args.capture_camera:
                    config.sensors = replace(config.sensors, capture_camera=True)
                result = run_experiment(config, args.output, args.model, args.viewer)
                print(json.dumps(result, indent=2))
            else:
                if len(set(args.seeds)) != len(args.seeds):
                    raise ValueError("Suite seeds must be unique")
                run_suite(config, args.output, args.tasks, args.winds, args.seeds, args.model)
        elif args.command == "train":
            from .training import train
            seed = args.seed if args.seed is not None else config.seed
            print(train(args.output, args.steps, seed, args.resume, args.checkpoint_interval,
                        args.learning_starts, args.buffer_size, config if args.config else None))
        elif args.command == "plot":
            from pathlib import Path
            from .plotting import plot_run, plot_suite
            function = plot_suite if (Path(args.directory)/"summary.csv").exists() else plot_run
            print(function(args.directory, args.output))
        else:
            from .comparison import compare
            print(json.dumps(compare(args.left, args.right, args.output), indent=2))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
