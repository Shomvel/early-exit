"""Source-checkout entrypoint for the SGLang early-exit benchmark evaluator."""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    source = Path(__file__).resolve().parents[2] / "python"
    sys.path.insert(0, str(source))
    runpy.run_path(
        str(source / "sglang/srt/early_exit/evaluate.py"), run_name="__main__"
    )
