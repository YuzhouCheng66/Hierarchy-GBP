"""Small stdlib-only utilities shared by benchmark entry points."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath

NUMERICAL_PREFIXES = (
    "GBP_", "HGBP_", "OMP_", "KMP_", "GOMP_", "MKL_", "OPENBLAS_",
    "BLIS_", "VECLIB_", "NUMEXPR_", "TBB_", "EIGEN_", "GOTO_", "BLAS_",
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value, *, newline=None):
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("w", encoding="utf-8", newline=newline) as stream:
        stream.write(encoded)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(x in ("", ".", "..") for x in value.split("/")) or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe relative path: {value}")
    return path


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def affinity_mask(value):
    try:
        mask = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer CPU mask, e.g. 0xffff") from exc
    if not 0 < mask < 2**64:
        raise argparse.ArgumentTypeError("affinity mask must be a nonzero 64-bit mask")
    return hex(mask)


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be finite positive seconds")
    return seconds


@contextmanager
def process_affinity(mask):
    """Children inherit this CPU allocation; restore the driver even on failure."""
    if mask is None:
        yield
        return
    bits = int(mask, 0) if isinstance(mask, str) else mask
    if isinstance(bits, bool) or not isinstance(bits, int) or not 0 < bits < 2**64:
        raise ValueError("affinity mask must be a nonzero 64-bit mask")
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        handle = kernel.GetCurrentProcess()
        previous, available = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(previous), ctypes.byref(available)):
            raise ctypes.WinError(ctypes.get_last_error())
        if bits & available.value != bits:
            raise ValueError("requested affinity is unavailable; select another mask or opt out")

        def set_mask(value):
            if not kernel.SetProcessAffinityMask(handle, value):
                raise ctypes.WinError(ctypes.get_last_error())

        set_mask(bits)
        restore = lambda: set_mask(previous.value)
    elif hasattr(os, "sched_setaffinity"):
        previous = os.sched_getaffinity(0)
        selected = {cpu for cpu in range(64) if bits & (1 << cpu)}
        if not selected <= previous:
            raise ValueError("requested affinity is unavailable; select another mask or opt out")
        os.sched_setaffinity(0, selected)
        restore = lambda: os.sched_setaffinity(0, previous)
    else:
        raise ValueError("affinity is unsupported on this OS; opt out explicitly")
    try:
        yield
    finally:
        restore()
