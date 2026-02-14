from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained model.")
    p.add_argument("--checkpoint", type=Path, required=False)
    p.add_argument("--output", type=Path, default=Path("outputs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    (args.output / "figures").mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Implement inference/evaluation once a model format is chosen.")


if __name__ == "__main__":
    main()
