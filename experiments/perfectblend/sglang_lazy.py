"""Source-checkout entrypoint for deferred-prefill smoke or paired pilot."""

import argparse
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", choices=("smoke", "evaluate"), default="smoke")
    options, arguments = parser.parse_known_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    sys.argv = [sys.argv[0], *arguments]
    runpy.run_module(
        "sglang.srt.early_exit.lazy_" + options.stage,
        run_name="__main__",
    )
