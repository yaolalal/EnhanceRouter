#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eargtc.router_v7_decision.final_audit import audit_final_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "runs/router_v7_decision_embeddings_20260827_v1",
    )
    parser.add_argument(
        "--hidden-root",
        type=Path,
        default=ROOT / "runs/router_v7_decision_llm_hidden_20260827_v1",
    )
    args = parser.parse_args()
    result = audit_final_run(args.run_root, args.hidden_root)
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    output = args.run_root.resolve() / "FINAL_AUDIT.json"
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(f"Final audit passed: {output}")


if __name__ == "__main__":
    main()
