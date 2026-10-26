from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def guarded_output_directory(target: str | Path):
    """Stage a result directory and atomically publish it without overwriting."""
    target = Path(target)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {target}")
    if target.exists():
        target.rmdir()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    committed = False
    try:
        yield staging
        os.replace(staging, target)
        committed = True
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)

