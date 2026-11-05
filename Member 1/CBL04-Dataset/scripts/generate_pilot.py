from __future__ import annotations

import argparse
from pathlib import Path

from generator import generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate matched fictional dark transaction screenshot pairs.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--count", type=int, default=20, help="Pilot count; default is 20 pairs.")
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")
    generate(args.output.resolve(), args.count, args.seed)
    print(f"Generated {args.count} matched pairs at {args.output.resolve()}")


if __name__ == "__main__":
    main()
