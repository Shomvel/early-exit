# SPDX-License-Identifier: Apache-2.0
"""CUDA conditional nodes for marked PyTorch graph regions.

Every decision remains on device. The C++ graph rewrite happens once before
instantiation; replay launches one graph containing real conditional bodies.
"""

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch


class ConditionalRegions:
    def __init__(self):
        source = Path(__file__).with_suffix(".cu")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        build = Path(
            os.environ.get("SGLANG_EARLY_EXIT_BUILD_DIR", tempfile.gettempdir())
        ) / ("early-exit-" + digest)
        build.mkdir(parents=True, exist_ok=True)
        target = build / "conditional.so"
        if not target.is_file():
            pending = build / f"conditional-{os.getpid()}.so"
            cuda_root = Path(shutil.which("nvcc")).resolve().parents[1]
            subprocess.run(
                [
                    "nvcc",
                    "-std=c++17",
                    "-O2",
                    "-shared",
                    "-Xcompiler=-fPIC",
                    "-gencode=arch=compute_90,code=sm_90",
                    str(source),
                    "-L" + str(cuda_root / "lib64/stubs"),
                    "-lcuda",
                    "-o",
                    str(pending),
                ],
                check=True,
            )
            pending.replace(target)
        self.library = ctypes.CDLL(str(target))
        self.library.ee_error.restype = ctypes.c_char_p
        self.library.ee_dump.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
        self.library.ee_marker.argtypes = [ctypes.c_uint64, ctypes.c_int]
        self.library.ee_pdl_increment.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        self.library.ee_convert.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]

    def _check(self, status):
        if status:
            raise RuntimeError(self.library.ee_error().decode())

    def marker(self, code):
        self._check(
            self.library.ee_marker(torch.cuda.current_stream().cuda_stream, code)
        )

    def pdl_increment(self, counter):
        self._check(
            self.library.ee_pdl_increment(
                torch.cuda.current_stream().cuda_stream, counter.data_ptr()
            )
        )

    def convert(self, graph, action, widths):
        assert action.is_cuda and action.dtype == torch.int32 and action.numel() == 1
        limits = (ctypes.c_int * len(widths))(*widths)
        moved = (ctypes.c_int * len(widths))()
        self._check(
            self.library.ee_convert(
                graph.raw_cuda_graph(), action.data_ptr(), limits, len(widths), moved
            )
        )
        if path := os.environ.get("SGLANG_EARLY_EXIT_DEBUG_DOT"):
            self._check(self.library.ee_dump(graph.raw_cuda_graph(), os.fsencode(path)))
        return list(moved)
