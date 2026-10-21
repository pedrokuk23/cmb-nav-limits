"""Code fingerprint for the study CSVs, so that a study never resumes from rows computed with different code.

Each study that resumes from its CSV keeps a sidecar file `<csv>.provenance.json` with a SHA-256 fingerprint of
the code that produced the rows: the simulator package (cmbsim/*.py and cmbsim/data/*), the study script, and the
study modules it imports from studies/. Before resuming, the study compares the stored fingerprint with the
current one; if they differ (or the sidecar is missing), it stops and asks to delete the CSV, unless
--resume-anyway is given, in which case the sidecar records the mix.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDIES = ROOT / "studies"


def _study_files(study: Path) -> list[Path]:
    """The study script and, recursively, the modules of studies/ that it imports (provenance.py excluded)."""
    seen, todo = set(), [study.resolve()]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for node in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for n in names:
                p = STUDIES / f"{n.split('.')[0]}.py"
                if p.is_file() and p.name != "provenance.py":
                    todo.append(p.resolve())
    return sorted(seen)


def _main_script() -> Path:
    return Path(sys.modules["__main__"].__file__).resolve()


def fingerprint(study=None) -> str:
    study = Path(study) if study else _main_script()
    h = hashlib.sha256()
    files = sorted((ROOT / "cmbsim").glob("*.py")) + sorted((ROOT / "cmbsim" / "data").glob("*")) \
        + _study_files(study)
    for f in files:
        if f.is_file():
            h.update(str(f.resolve().relative_to(ROOT)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def _sidecar(csv_path) -> Path:
    return Path(str(csv_path) + ".provenance.json")


def check_resume(csv_path, resume_anyway=False, study=None) -> bool:
    """Returns True if the existing rows were computed with other code (only possible with resume_anyway)."""
    if not os.path.exists(csv_path):
        return False
    side = _sidecar(csv_path)
    stored = json.loads(side.read_text()).get("fingerprint") if side.exists() else None
    current = fingerprint(study)
    if stored == current:
        return False
    msg = (f"{csv_path} was written by different code (fingerprint {'missing' if stored is None else stored[:12]}, "
           f"current {current[:12]}). Delete it to recompute from scratch, or pass --resume-anyway.")
    if not resume_anyway:
        sys.exit(msg)
    print("WARNING: " + msg, file=sys.stderr)
    return True


def stamp(csv_path, mixed=False, study=None):
    import numpy as np
    _sidecar(csv_path).write_text(json.dumps(dict(
        fingerprint=fingerprint(study), mixed_code_versions=bool(mixed), date=time.strftime("%Y-%m-%d %H:%M"),
        python=platform.python_version(), numpy=np.__version__), indent=1) + "\n")
