#!/usr/bin/env python3
"""Lightweight verifier for the curated public package."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "requirements-experiments.txt",
    "examples/vc_merge_tifxyz_summary_minimal.json",
    "scripts/benchmark_thaumato_blur.py",
    "scripts/run_thaumato_surface_detection_ablation.py",
    "scripts/plot_thaumato_patch_graph_growth_qa.py",
    "scripts/create_vc3d_tifxyz_smoke_volpkg.py",
    "scripts/summarize_vc3d_merge_summary_adapter.py",
    "scripts/summarize_vc3d_multi_chart_adapter_contract.py",
    "scripts/summarize_pherc1667_multi_chart_method_decision.py",
    "reports/README.md",
    "reports/2026-05-07_multi_chart_atlas_realdata_manifest.md",
    "reports/2026-05-07_multi_chart_atlas_generalization_manifest.md",
    "reports/thaumato_pherc1667_multi_chart_method_decision_2026-05-08.md",
]

EXPECTED_CASES = {
    "reports/thaumato_patch_graph_growth_qa_PHerc1667_13_4_3_recto_seed1_multi_chart_atlas_realdata90_2026-05-07.json": (
        0.230291,
        0.005552,
        6,
    ),
    "reports/thaumato_patch_graph_growth_qa_PHerc1667_13_4_4_verso_seed0_multi_chart_atlas_generalization75_2026-05-07.json": (
        0.506859,
        0.010780,
        2,
    ),
    "reports/thaumato_patch_graph_growth_qa_PHerc1667_14_4_4_verso_seed0_multi_chart_atlas_generalization75_2026-05-07.json": (
        0.300417,
        0.039103,
        3,
    ),
}

FORBIDDEN_SUFFIXES = {
    ".zarr",
    ".tif",
    ".tiff",
    ".nrrd",
    ".nii",
    ".obj",
    ".ply",
    ".stl",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
}

SKIP_DIRS = {".git", ".venv", "__pycache__"}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.relative_to(root).parts)
        if parts & SKIP_DIRS:
            continue
        yield path


def nested_get(payload: dict, keys: list[str]):
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            fail(f"missing key path {'.'.join(keys)}")
        current = current[key]
    return current


def assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-6) -> None:
    if abs(actual - expected) > tolerance:
        fail(f"{label}: expected {expected}, got {actual}")


def main() -> None:
    root = Path.cwd()
    missing = [path for path in REQUIRED_FILES if not (root / path).exists()]
    if missing:
        fail(f"missing required files: {missing}")

    for path in iter_files(root):
        relative = path.relative_to(root)
        if any(str(relative).endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            fail(f"forbidden raw/model artifact present: {relative}")

    report_count = len(list((root / "reports").glob("*")))
    if report_count > 40:
        fail(f"public reports directory is too noisy: {report_count} files")

    for path, (direct_p90, multi_p90, bridges) in EXPECTED_CASES.items():
        payload = json.loads((root / path).read_text(encoding="utf-8"))
        atlas = nested_get(payload, ["local_atlas_sequence", "multi_chart_atlas_reconciliation"])
        assert_close(float(atlas["direct_global_p90_edge_distortion"]), direct_p90, path)
        assert_close(
            float(atlas["multi_chart_atlas_bridge_aware_p90_edge_distortion"]),
            multi_p90,
            path,
        )
        if int(atlas["multi_chart_atlas_bridge_edge_count"]) != bridges:
            fail(f"{path}: expected {bridges} scored bridges")
        png = root / path.replace(".json", ".png")
        if not png.exists():
            fail(f"missing matching PNG for {path}")

    print(f"OK: curated public package verified with {report_count} report artifacts")


if __name__ == "__main__":
    main()
