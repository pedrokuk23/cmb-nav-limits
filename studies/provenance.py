"""Code fingerprint for the study CSVs, so that a study never resumes from rows computed with different code.

Each study that resumes from its CSV keeps a sidecar file `<csv>.provenance.json` with a SHA-256 fingerprint of
the simulator package (cmbsim/*.py and cmbsim/data/*) and of the study scripts (studies/*.py). Before resuming,
the study compares the stored fingerprint with the current one; if they differ (or the sidecar is missing), it
stops and asks to delete the CSV, unless --resume-anyway is given, in which case the sidecar records the mix.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fingerprint() -> str:
    h = hashlib.sha256()
    files = sorted((ROOT / "cmbsim").glob("*.py")) + sorted((ROOT / "cmbsim" / "data").glob("*")) \
        + sorted((ROOT / "studies").glob("*.py"))
    for f in files:
        if f.is_file():
            h.update(str(f.relative_to(ROOT)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def _sidecar(csv_path) -> Path:
    return Path(str(csv_path) + ".provenance.json")


def check_resume(csv_path, resume_anyway=False) -> bool:
    """Returns True if the existing rows were computed with other code (only possible with resume_anyway)."""
    if not os.path.exists(csv_path):
        return False
    side = _sidecar(csv_path)
    stored = json.loads(side.read_text()).get("fingerprint") if side.exists() else None
    if stored == fingerprint():
        return False
    msg = (f"{csv_path} was written by different code (fingerprint {'missing' if stored is None else stored[:12]}, "
           f"current {fingerprint()[:12]}). Delete it to recompute from scratch, or pass --resume-anyway.")
    if not resume_anyway:
        sys.exit(msg)
    print("WARNING: " + msg, file=sys.stderr)
    return True


def stamp(csv_path, mixed=False):
    import numpy as np
    _sidecar(csv_path).write_text(json.dumps(dict(
        fingerprint=fingerprint(), mixed_code_versions=bool(mixed), date=time.strftime("%Y-%m-%d %H:%M"),
        python=platform.python_version(), numpy=np.__version__), indent=1) + "\n")
