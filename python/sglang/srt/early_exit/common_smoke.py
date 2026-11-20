# SPDX-License-Identifier: Apache-2.0
"""Real-model parity and padding controls for one action per whole batch."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from sglang.srt.early_exit import batch_smoke as b


def check_routes(result, size, schedule):
    assert result["routing"] == "batch_min"
    assert all(len(regions) == 8 for regions in result["converted_regions"])
    for step, route in enumerate(result["routing_steps"]):
        proposed = route["proposed"]
        assert proposed[:size] == schedule[step % len(schedule)]
        assert all(value == -1 for value in proposed[size:])
        action = min(proposed[:size])
        assert route["applied"] == [action] * size + [-1] * (len(proposed) - size)
    assert b.padding_checks(result, size, result["steps"])["passed"]


def main(args):
    import torch
    from sglang.srt.early_exit.batch_evaluate import protocol

    out = Path(os.environ["RUN_DIR"])
    protocol().verify_model(args)
    b.save(out / "config.yaml", dict(model=str(args.model), classifier=str(args.classifier),
           routing="batch_min", modes=["full", "serial", "async"], batch_sizes=args.batch_sizes, steps=args.steps))
    for mode in ("full", "serial", "async"):
        env = {k: v for k, v in os.environ.items() if not k.startswith("SGLANG_EARLY_EXIT_") or k == "SGLANG_EARLY_EXIT_SOURCE_ROOT"}
        if mode != "full":
            env.update(SGLANG_EARLY_EXIT_CHECKPOINT=str(args.classifier),
                       SGLANG_EARLY_EXIT_CACHE=mode, SGLANG_EARLY_EXIT_ROUTING="batch_min")
        subprocess.run([sys.executable, "-u", "-m", __spec__.name, *sys.argv[1:], "--worker", mode], env=env, check=True)
    summaries = {mode: json.loads((out / f"batch-{mode}.json").read_text()) for mode in ("full", "serial", "async")}
    tensors = {mode: torch.load(out / f"batch-{mode}-tensors.pt", map_location="cpu", weights_only=True) for mode in summaries}
    cases = []
    for size in args.batch_sizes:
        key = str(size)
        for mode in ("serial", "async"):
            for name in ("reference_logits", "reference_state"):
                assert b.comparisons(tensors[mode][key][name], tensors["full"][key][name])["passed"]
            assert summaries[mode][key]["reference"]["generated"] == summaries["full"][key]["reference"]["generated"]
            check_routes(summaries[mode][key]["reference"], size, [[0] * size])
            check_routes(summaries[mode][key]["mixed"], size, b.action_schedule(size))
        for name in ("mixed_logits", "mixed_state"):
            assert b.comparisons(tensors["serial"][key][name], tensors["async"][key][name], exact=True)["passed"]
        cases.append(dict(batch_size=size, full_no_skip=True, serial_async_exact=True, zero_veto=True, padding_canary=True))
    result = dict(passed=True, routing="batch_min", cases=cases)
    b.save(out / "common-smoke-summary.json", result)
    print("PB_COMMON_SMOKE_PASS " + json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--weight-inventory", type=Path, default=Path(os.environ.get("SGLANG_EARLY_EXIT_SOURCE_ROOT", Path(__file__).resolve().parents[4])) / "experiments/perfectblend/references/model-weight-inventory.json")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 3, 4, 8])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--worker", choices=("full", "serial", "async"))
    options = parser.parse_args()
    (b.worker if options.worker else main)(options)
