"""Exercise the real overlay launcher concurrently without model dependencies."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class OverlayReplicaTest(unittest.TestCase):
    def run_replicas(self, count):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            launcher = source / "experiments/perfectblend/launch_overlay.py"
            launcher.parent.mkdir(parents=True)
            shutil.copyfile(
                Path(__file__).resolve().parents[1] / "launch_overlay.py", launcher
            )
            installed = root / "installed/sglang"
            installed.mkdir(parents=True)
            (installed / "__init__.py").write_text("")
            before, after = b'VALUE = "image"\n', b'VALUE = "overlay"\n'
            (installed / "value.py").write_bytes(before)
            (installed / "probe.py").write_text(
                "import json,os\nfrom pathlib import Path\nfrom .value import VALUE\n"
                'assert VALUE == "overlay"\n'
                'p=Path(os.environ["RUN_DIR"])/os.environ["KUBEBRAIN_REPLICA"]\n'
                'p.write_text(json.dumps({"value":VALUE,"package":__file__}))\n'
            )
            replacement = source / "python/sglang/value.py"
            replacement.parent.mkdir(parents=True)
            replacement.write_bytes(after)
            (source / "overlay-manifest.json").write_text(
                json.dumps(
                    dict(
                        base_sha256={"value.py": hashlib.sha256(before).hexdigest()},
                        overlay_sha256={"value.py": hashlib.sha256(after).hexdigest()},
                        fork_source={"scope": "concurrent launcher regression"},
                    )
                )
            )
            run = root / "run"
            run.mkdir()
            children = []
            for replica in range(count):
                environment = dict(
                    os.environ,
                    RUN_DIR=str(run),
                    PYTHONPATH=str(installed.parent),
                    KUBEBRAIN_REPLICA=str(replica),
                    KUBEBRAIN_REPLICA_TOTAL=str(count),
                )
                children.append(
                    subprocess.Popen(
                        [sys.executable, str(launcher), "--module", "sglang.probe"],
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            for child in children:
                stdout, stderr = child.communicate(timeout=20)
                self.assertEqual(child.returncode, 0, stdout + stderr)
            for replica in range(count):
                runtime = run if count == 1 else run / f"runtime-replica-{replica}"
                observed = json.loads((run / str(replica)).read_text())
                self.assertEqual(observed["value"], "overlay")
                self.assertEqual(
                    Path(observed["package"]).parent, runtime / "sglang-python/sglang"
                )
                receipt = json.loads(
                    (runtime / "runtime-overlay-receipt.json").read_text()
                )
                self.assertEqual(
                    receipt["overlay_sha256"]["value.py"],
                    hashlib.sha256(after).hexdigest(),
                )

    def test_four_replicas_materialize_and_execute_isolated_overlays(self):
        self.run_replicas(4)

    def test_single_replica_retains_original_runtime_path(self):
        self.run_replicas(1)


if __name__ == "__main__":
    unittest.main()
