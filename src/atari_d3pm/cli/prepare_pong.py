"""Download, convert, and audit the Minari Pong expert dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from atari_d3pm.data import convert_minari_dataset
from atari_d3pm.eda import run_eda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="atari/pong/expert-v0")
    parser.add_argument("--output", type=Path, default=Path("data/pong/v1"))
    parser.add_argument("--minari-root", type=Path, default=Path("data/minari"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--skip-environment-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.minari_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((args.report_dir / ".matplotlib").resolve()))
    os.environ["MINARI_DATASETS_PATH"] = str(args.minari_root.resolve())
    import minari

    dataset = minari.load_dataset(args.dataset_id, download=True)
    metadata = convert_minari_dataset(
        dataset,
        args.output,
        split_seed=args.split_seed,
        train_episodes=args.train_episodes,
        verify_environment=not args.skip_environment_check,
    )
    summary = run_eda(args.output, args.report_dir)
    print(
        f"Prepared {metadata['dataset_id']}: {summary['num_episodes']} episodes, "
        f"{summary['num_steps']} steps -> {args.output}"
    )


if __name__ == "__main__":
    main()
