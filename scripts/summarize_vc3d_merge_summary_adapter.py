#!/usr/bin/env python3
"""Summarize vc_merge_tifxyz summary JSON into scalar bridge-QA adapter fields."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY = Path("examples/vc_merge_tifxyz_summary_minimal.json")
DEFAULT_PUBLIC_CONTEXT = Path("reports/2026-05-08_public_vc3d_merge_context_refresh.md")
DEFAULT_THRESHOLDS = {
    "min_anchor_count": 3,
    "min_ransac_inlier_fraction": 0.50,
    "min_real_overlap": 200,
    "max_pair_scale_delta": 0.25,
}
NON_CLAIMS = [
    "no overlap metadata writing",
    "no surface names or paths in adapter output",
    "no coordinates or geometry payloads",
    "no OBJ, tifxyz, or raster output",
    "no predictions, ink, letters, or title",
    "no automatic VC3D code mutation",
]
UPSTREAM_SCHEMA_REFERENCE = {
    "repository": "ScrollPrize/villa",
    "commit": "c257b402fce35b5967fee752e415067a18d281c4",
    "path": "volume-cartographer/apps/src/vc_merge_tifxyz.cpp",
    "summary_writer_lines": "2184-2220",
    "edge_json_lines": "711-718, 1911-1918",
    "checked_utc": "2026-05-08T01:51:43Z",
}
UPSTREAM_FIELDS_VERIFIED = [
    "summary.merge_json",
    "summary.output",
    "summary.obj_out",
    "summary.ref_surface",
    "summary.strip_cols",
    "summary.strip_count",
    "summary.surfaces[].name",
    "summary.surfaces[].path",
    "summary.surfaces[].H",
    "summary.surfaces[].W",
    "summary.surfaces[].valid",
    "summary.edges[].best_threshold",
    "summary.edges[].best_score",
    "summary.edges[].per_threshold",
    "summary.edges[].anchor_count",
    "summary.edges[].anchor_bin_size",
    "summary.edges[].anchor_seed_side",
    "summary.edges[].ransac_inliers",
    "summary.edges[].ransac_total",
    "summary.edges[].ransac_thresh",
    "summary.edges[].ransac_sigma_in",
    "summary.edges[].pair_scale",
    "summary.edges[].real_overlap_A",
    "summary.edges[].real_overlap_B",
    "summary.joint_affine[]",
    "summary.surface_rbf[]",
    "summary.params",
]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def rounded(value: Any, digits: int = 6) -> float | None:
    return round(float(value), digits) if is_number(value) else None


def fmt(value: Any, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}" if is_number(value) else "n/a"


def safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_counts(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "median": None, "max": None, "total": 0}
    return {
        "min": min(values),
        "median": rounded(median([float(value) for value in values])),
        "max": max(values),
        "total": sum(values),
    }


def edge_failure_reasons(
    anchor_count: int | None,
    inlier_fraction: float | None,
    real_overlap_min: int | None,
    pair_scale_delta: float | None,
    thresholds: dict[str, float],
) -> list[str]:
    reasons: list[str] = []
    if anchor_count is None or anchor_count < thresholds["min_anchor_count"]:
        reasons.append("low_anchor_count")
    if inlier_fraction is None or inlier_fraction < thresholds["min_ransac_inlier_fraction"]:
        reasons.append("low_ransac_inlier_fraction")
    if real_overlap_min is None or real_overlap_min < thresholds["min_real_overlap"]:
        reasons.append("low_real_overlap")
    if pair_scale_delta is None or pair_scale_delta > thresholds["max_pair_scale_delta"]:
        reasons.append("pair_scale_outlier")
    return reasons


def edge_row(
    edge: dict[str, Any],
    edge_index: int,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    anchor_count = safe_int(edge.get("anchor_count"))
    ransac_total = safe_int(edge.get("ransac_total"))
    ransac_inliers = safe_int(edge.get("ransac_inliers"))
    inlier_fraction = (
        ransac_inliers / ransac_total
        if ransac_total and ransac_total > 0 and ransac_inliers is not None
        else None
    )
    pair_scale = rounded(edge.get("pair_scale"))
    pair_scale_delta = rounded(abs(pair_scale - 1.0)) if pair_scale is not None else None
    real_overlap_a = safe_int(edge.get("real_overlap_A"))
    real_overlap_b = safe_int(edge.get("real_overlap_B"))
    real_overlap_values = [
        value for value in (real_overlap_a, real_overlap_b) if value is not None
    ]
    real_overlap_min = min(real_overlap_values) if real_overlap_values else None
    reasons = edge_failure_reasons(
        anchor_count,
        inlier_fraction,
        real_overlap_min,
        pair_scale_delta,
        thresholds,
    )
    return {
        "edge_index": edge_index,
        "best_threshold": rounded(edge.get("best_threshold")),
        "best_score": rounded(edge.get("best_score")),
        "threshold_count": len(edge.get("per_threshold") or []),
        "anchor_count": anchor_count,
        "anchor_bin_size": safe_int(edge.get("anchor_bin_size")),
        "ransac_total": ransac_total,
        "ransac_inliers": ransac_inliers,
        "ransac_inlier_fraction": rounded(inlier_fraction),
        "ransac_threshold": rounded(edge.get("ransac_thresh")),
        "ransac_sigma_in": rounded(edge.get("ransac_sigma_in")),
        "pair_scale": pair_scale,
        "pair_scale_delta": pair_scale_delta,
        "real_overlap_min": real_overlap_min,
        "review_bucket": "ready_for_bridge_qa" if not reasons else "needs_upstream_merge_review",
        "failure_reasons": reasons,
        "overlap_metadata_action": "none_scalar_only",
    }


def build_payload(
    summary_path: Path,
    public_context_path: Path | None = DEFAULT_PUBLIC_CONTEXT,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    summary = load_json(summary_path)
    surfaces = summary.get("surfaces") if isinstance(summary.get("surfaces"), list) else []
    edges = summary.get("edges") if isinstance(summary.get("edges"), list) else []
    rows = [
        edge_row(edge, index, thresholds)
        for index, edge in enumerate(edges)
        if isinstance(edge, dict)
    ]
    ready_rows = [row for row in rows if row["review_bucket"] == "ready_for_bridge_qa"]
    review_rows = [row for row in rows if row["review_bucket"] != "ready_for_bridge_qa"]
    anchor_counts = [
        row["anchor_count"] for row in rows if isinstance(row.get("anchor_count"), int)
    ]
    inlier_fractions = [
        row["ransac_inlier_fraction"]
        for row in rows
        if is_number(row.get("ransac_inlier_fraction"))
    ]
    pair_scale_deltas = [
        row["pair_scale_delta"] for row in rows if is_number(row.get("pair_scale_delta"))
    ]
    real_overlap_mins = [
        row["real_overlap_min"] for row in rows if isinstance(row.get("real_overlap_min"), int)
    ]
    surface_valid_counts = [
        safe_int(surface.get("valid"))
        for surface in surfaces
        if isinstance(surface, dict) and safe_int(surface.get("valid")) is not None
    ]
    bucket_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in rows:
        bucket = str(row["review_bucket"])
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        for reason in row["failure_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "method": "vc3d_merge_summary_scalar_adapter",
        "source_tool": "vc_merge_tifxyz",
        "source_summary": str(summary_path),
        "source_public_context": str(public_context_path) if public_context_path else None,
        "upstream_schema_reference": UPSTREAM_SCHEMA_REFERENCE,
        "metadata_write_action": "none_scalar_only",
        "thresholds": thresholds,
        "surface_count": len(surfaces),
        "edge_count": len(rows),
        "strip_count": safe_int(summary.get("strip_count")),
        "ready_for_bridge_qa_count": len(ready_rows),
        "needs_upstream_merge_review_count": len(review_rows),
        "bucket_counts": bucket_counts,
        "failure_reason_counts": reason_counts,
        "anchor_count_summary": scalar_counts(anchor_counts),
        "surface_valid_count_summary": scalar_counts(
            [value for value in surface_valid_counts if value is not None]
        ),
        "worst_ransac_inlier_fraction": rounded(min(inlier_fractions))
        if inlier_fractions
        else None,
        "max_pair_scale_delta": rounded(max(pair_scale_deltas)) if pair_scale_deltas else None,
        "min_real_overlap": min(real_overlap_mins) if real_overlap_mins else None,
        "adapter_contract": {
            "input_source": "vc_merge_tifxyz summary.json",
            "upstream_fields_verified": UPSTREAM_FIELDS_VERIFIED,
            "input_fields_consumed": [
                "surfaces[].valid",
                "edges[].best_threshold",
                "edges[].best_score",
                "edges[].anchor_count",
                "edges[].anchor_bin_size",
                "edges[].ransac_inliers",
                "edges[].ransac_total",
                "edges[].ransac_thresh",
                "edges[].ransac_sigma_in",
                "edges[].pair_scale",
                "edges[].real_overlap_A",
                "edges[].real_overlap_B",
                "strip_count",
            ],
            "fields_not_copied": [
                "surface names",
                "surface paths",
                "merge_json path",
                "output directory",
                "obj_out path",
                "coordinates",
                "meshes",
                "tifxyz payloads",
            ],
            "output_fields": [
                "edge_index",
                "review_bucket",
                "failure_reasons",
                "anchor_count",
                "ransac_inlier_fraction",
                "pair_scale_delta",
                "real_overlap_min",
                "overlap_metadata_action",
            ],
            "non_claims": NON_CLAIMS,
        },
        "edge_rows": rows,
        "decision": (
            "adapter_schema_ready_for_current_main_summary"
            if rows and ready_rows
            else "extend_with_real_vc_merge_summary"
        ),
        "recommended_next_steps": [
            "Keep this adapter read-only and scalar-only inside the progress-prize package.",
            "When a real current-main vc_merge_tifxyz summary is available, run this adapter and compare its review buckets to the existing multi-chart bridge-QA rows.",
            "Only propose upstream code after a bounded current-main reproduction shows a concrete failure not covered by existing VC3D merge tooling.",
        ],
        "storage_policy": (
            "Read vc_merge_tifxyz summary scalars only; do not carry surface names, paths, "
            "coordinates, OBJ paths, tifxyz outputs, meshes, overlap metadata, predictions, "
            "ink, letters, or titles into the adapter output."
        ),
        "non_claims": NON_CLAIMS,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# VC3D Merge Summary Scalar Adapter",
        "",
        f"- timestamp UTC: {payload['generated_at_utc']}",
        f"- source tool: `{payload['source_tool']}`",
        f"- source summary: `{payload['source_summary']}`",
        f"- public context: `{payload['source_public_context']}`",
        f"- upstream schema: `{payload['upstream_schema_reference']['repository']}@{payload['upstream_schema_reference']['commit']}` `{payload['upstream_schema_reference']['path']}`",
        "- script: `scripts/summarize_vc3d_merge_summary_adapter.py`",
        "- machine: current `g2-standard-16` L4 VM",
        "- new data downloads: none",
        "- GPU runtime: none",
        "- cost impact: CPU-only scalar summary generation, negligible incremental cost",
        "- command: `python3 scripts/summarize_vc3d_merge_summary_adapter.py`",
        "- hypothesis: current `vc_merge_tifxyz` outputs can feed a read-only scalar bridge-QA adapter without copying paths, names, geometry, or metadata writes",
        "- visual checks: none; scalar-only JSON and Markdown table output",
        "- saved data policy: scalar aggregate JSON only; no raw chunks, surface paths, names, coordinates, OBJ, tifxyz outputs, meshes, predictions, letters, or titles",
        "",
        "## Summary",
        "",
        f"- surfaces: `{payload['surface_count']}`",
        f"- edges: `{payload['edge_count']}`",
        f"- strip count: `{payload['strip_count']}`",
        f"- ready for bridge QA: `{payload['ready_for_bridge_qa_count']}`",
        f"- needs upstream merge review: `{payload['needs_upstream_merge_review_count']}`",
        f"- metadata write action: `{payload['metadata_write_action']}`",
        f"- decision: `{payload['decision']}`",
        "",
        "## Adapter Contract",
        "",
        "- input source: `vc_merge_tifxyz summary.json`",
        "- adapter mode: read-only scalar bridge-candidate review",
        "- overlap metadata action: `none_scalar_only`",
        "- fields not copied: surface names, surface paths, output directories, OBJ paths, coordinates, meshes, tifxyz payloads",
        "",
        "## Edge Review Rows",
        "",
        "| edge | anchors | inlier frac | pair scale delta | real overlap min | bucket | reasons |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["edge_rows"]:
        reasons = ",".join(row["failure_reasons"]) if row["failure_reasons"] else "none"
        lines.append(
            f"| {row['edge_index']} | {row['anchor_count']} | "
            f"{fmt(row['ransac_inlier_fraction'])} | {fmt(row['pair_scale_delta'])} | "
            f"{row['real_overlap_min']} | {row['review_bucket']} | {reasons} |"
        )
    lines.extend(
        [
            "",
            "## Integration Boundary",
            "",
            "This adapter is read-only. It summarizes current `vc_merge_tifxyz` diagnostics for review and does not reimplement N-way merge, alignment, blending, rasterization, or overlap metadata writing.",
            "",
            "It is not an `overlapping.json` writer, does not mutate VC3D metadata, and does not claim to solve an active VC3D issue by itself.",
            "",
            "This is not an unwrap, text, letter, title, or ink claim.",
            "",
            "## Ranked Next Steps",
            "",
        ]
    )
    lines.extend(f"- {step}" for step in payload["recommended_next_steps"])
    lines.extend(["", "## Storage Policy", "", payload["storage_policy"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def threshold_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        "min_anchor_count": float(args.min_anchor_count),
        "min_ransac_inlier_fraction": float(args.min_ransac_inlier_fraction),
        "min_real_overlap": float(args.min_real_overlap),
        "max_pair_scale_delta": float(args.max_pair_scale_delta),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--public-context", type=Path, default=DEFAULT_PUBLIC_CONTEXT)
    parser.add_argument("--min-anchor-count", type=int, default=DEFAULT_THRESHOLDS["min_anchor_count"])
    parser.add_argument(
        "--min-ransac-inlier-fraction",
        type=float,
        default=DEFAULT_THRESHOLDS["min_ransac_inlier_fraction"],
    )
    parser.add_argument("--min-real-overlap", type=int, default=DEFAULT_THRESHOLDS["min_real_overlap"])
    parser.add_argument(
        "--max-pair-scale-delta",
        type=float,
        default=DEFAULT_THRESHOLDS["max_pair_scale_delta"],
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("reports/vc3d_merge_summary_adapter_contract_2026-05-08.json"),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("reports/vc3d_merge_summary_adapter_contract_2026-05-08.md"),
    )
    args = parser.parse_args()

    payload = build_payload(args.summary_json, args.public_context, threshold_args(args))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, args.md_out)
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
