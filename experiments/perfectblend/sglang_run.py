"""Source-checkout entrypoint for the real-model GPU routing smoke test."""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    source = Path(__file__).resolve().parents[2] / "python"
    sys.path.insert(0, str(source))
    runpy.run_module("sglang.srt.early_exit.smoke", run_name="__main__")
