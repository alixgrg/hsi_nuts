"""Validate and materialize the frozen eight-track scientific protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import experiment_config as expcfg
from src.protocol_governance import (  # noqa: E402
    freeze_protocol,
    validate_protocol_contract,
    verify_frozen_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate tasks 01-02 and write their versioned protocol artifacts."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT.joinpath(
            *expcfg.PROTOCOL_ARTIFACT_RELATIVE_DIR
        ),
        help="Destination directory for the frozen protocol artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenerate the currently declared version. Scientific changes "
            "must use a new PROTOCOL_VERSION instead."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Run blocking protocol checks without writing files.",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="Verify the semantic hashes and checksums of the frozen bundle.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_only:
        checks = validate_protocol_contract(strict=True)
        print(checks.to_string(index=False))
        print(f"Protocol {expcfg.PROTOCOL_VERSION}: all checks passed.")
        return 0
    if args.verify:
        checks = verify_frozen_protocol(args.output_dir, strict=True)
        print(checks.to_string(index=False))
        print(
            f"Protocol {expcfg.PROTOCOL_VERSION}: frozen bundle verified."
        )
        return 0

    result = freeze_protocol(args.output_dir, overwrite=args.overwrite)
    print(f"Protocol {expcfg.PROTOCOL_VERSION} frozen in {args.output_dir}")
    print(f"configuration_sha256={result['configuration_sha256']}")
    print(f"inference_plan_sha256={result['inference_plan_sha256']}")
    print(f"planned_contrasts_sha256={result['planned_contrasts_sha256']}")
    print(f"lock_sha256={result['lock_sha256']}")
    for name, path in result["paths"].items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
