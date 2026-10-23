from __future__ import annotations

import json
from pathlib import Path



def load_clean_replay_config(
    *, project_root: Path, manifest_path: Path, dataset: str
) -> dict[str, object]:
    """Resolve the exact clean-run prompt/class mapping for a shifted replay."""
    project_root = Path(project_root).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entry = manifest.get("datasets", {}).get(str(dataset))
    if not isinstance(entry, dict) or "source_dir" not in entry:
        raise ValueError(f"clean LLM manifest misses dataset={dataset}")
    source = Path(entry["source_dir"])
    if not source.is_absolute():
        source = project_root / source
    chunk_root = source / "chunks"
    if not chunk_root.is_dir():
        chunk_root = source / str(dataset) / "chunks"
    metadata_paths = sorted(chunk_root.glob("*.meta.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"clean replay metadata missing: {chunk_root}")
    metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    if str(metadata.get("dataset")) != str(dataset):
        raise ValueError("clean replay metadata dataset mismatch")
    if not bool(metadata.get("use_chat_template")):
        raise ValueError("Router-v7 replay requires the clean chat-template setting")
    for key in ("model_path", "shots", "seed", "max_input_length", "num_classes"):
        if key not in metadata:
            raise ValueError(f"clean replay metadata misses {key}")
    for key in ("prompt_template_path", "label_number_map_json"):
        value = metadata.get(key)
        if value is not None and not Path(value).is_file():
            raise FileNotFoundError(f"clean replay {key} missing: {value}")
    return {
        "source_metadata_path": str(metadata_paths[0].resolve()),
        "model_path": str(metadata["model_path"]),
        "shots": int(metadata["shots"]),
        "seed": int(metadata["seed"]),
        "max_input_length": int(metadata["max_input_length"]),
        "use_chat_template": True,
        "prompt_template_path": metadata.get("prompt_template_path"),
        "label_number_map_json": metadata.get("label_number_map_json"),
        "num_classes": int(metadata["num_classes"]),
        "number_to_label_ids": [int(value) for value in metadata.get("number_to_label_ids", range(int(metadata["num_classes"])))],
        "number_token_ids": [int(value) for value in metadata["number_token_ids"]],
        "dtype": "float16",
    }

DEFAULT_ENVIRONMENTS = (
    "text_truncate_050",
    "text_mask_mixed_030",
    "text_clean_replay",
    "text_span_delete_030",
)


def plan_text_jobs(
    *,
    input_root: Path,
    output_root: Path,
    datasets: list[str],
    environment_ids: list[str] | tuple[str, ...] = DEFAULT_ENVIRONMENTS,
) -> list[dict[str, object]]:
    """Plan one immutable numbered-token replay job per dataset/environment."""
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    jobs: list[dict[str, object]] = []
    for environment_id in environment_ids:
        ref_root = input_root / str(environment_id)
        if not (ref_root / "datasets").is_dir():
            raise FileNotFoundError(f"missing shifted dataset root: {ref_root / 'datasets'}")
        for dataset in datasets:
            cohort_path = input_root / "cohorts" / f"{dataset}.json"
            cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
            node_ids = cohort.get("node_ids")
            if not isinstance(node_ids, list) or not node_ids:
                raise ValueError(f"invalid or empty probe cohort: {cohort_path}")
            jobs.append(
                {
                    "dataset": str(dataset),
                    "environment_id": str(environment_id),
                    "start_index": 0,
                    "end_index": len(node_ids),
                    "ref_root": str(ref_root),
                    "run_root": str(output_root / str(environment_id)),
                    "node_ids_path": str(cohort_path),
                }
            )
    return jobs
