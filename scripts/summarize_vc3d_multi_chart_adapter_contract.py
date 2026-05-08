#!/usr/bin/env python3
"""Summarize the VC3D read-only adapter contract for multi-chart atlas QA."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


DEFAULT_REPORTS = [
    Path(
        "reports/thaumato_patch_graph_growth_qa_PHerc1667_13_4_3_recto_seed1_multi_chart_atlas_realdata90_2026-05-07.json"
    ),
    Path(
        "reports/thaumato_patch_graph_growth_qa_PHerc1667_13_4_4_verso_seed0_multi_chart_atlas_generalization75_2026-05-07.json"
    ),
    Path(
        "reports/thaumato_patch_graph_growth_qa_PHerc1667_14_4_4_verso_seed0_multi_chart_atlas_generalization75_2026-05-07.json"
    ),
    Path(
        "reports/thaumato_patch_graph_growth_qa_PHercParis4_3_4_4_recto_seed0_multi_chart_atlas_control75_2026-05-08.json"
    ),
    Path(
        "reports/thaumato_patch_graph_growth_qa_PHercParis4_3_4_4_verso_seed0_multi_chart_atlas_control75_2026-05-08.json"
    ),
]
DEFAULT_PUBLIC_CONTEXT = Path("reports/2026-05-07_public_vc3d_overlap_context_refresh.md")

NON_CLAIMS = [
    "no overlap metadata writing",
    "no endpoint or component identifiers",
    "no geometry or coordinate payloads",
    "no meshes",
    "no predictions, ink, letters, or title",
    "no automatic tracer patch",
]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def rounded(value: Any, digits: int = 6) -> float | None:
    return round(float(value), digits) if is_number(value) else None


def fmt(value: Any, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}" if is_number(value) else "n/a"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_label(chunk: Any) -> str:
    if isinstance(chunk, list):
        return ",".join(str(part) for part in chunk)
    return str(chunk)


def role_for(sample: str, reached: bool, direct_cap: bool, multi_cap: bool) -> str:
    if sample == "PHerc1667" and reached and not direct_cap and multi_cap:
        return "direct_global_blocker_positive"
    if sample != "PHerc1667" or direct_cap:
        return "specificity_control"
    return "manual_review"


def action_for(role: str) -> tuple[str, str, str]:
    if role == "direct_global_blocker_positive":
        return (
            "pass_scalar_multi_chart_bridge_gate",
            "direct_global_blocker_resolved_by_multi_chart_atlas",
            "quality_gate_pass_review_candidate",
        )
    if role == "specificity_control":
        return (
            "do_not_promote_specificity_control",
            "control_not_a_direct_global_blocker",
            "no_positive_action_control",
        )
    return (
        "manual_review_required",
        "needs_more_validation",
        "manual_review_required",
    )


def report_row(path: Path) -> dict[str, Any]:
    report = load_json(path)
    source = report.get("source") or {}
    parameters = report.get("parameters") or {}
    sequence = report.get("local_atlas_sequence") or {}
    reconciliation = sequence.get("multi_chart_atlas_reconciliation") or {}

    sample = str(source.get("sample_id"))
    side = str(parameters.get("side"))
    seed = int(parameters.get("seed") or 0)
    chunk = chunk_label(source.get("chunk"))
    reached = sequence.get("reached") is True
    direct_cap = reconciliation.get("direct_global_cap_compliant") is True
    multi_cap = reconciliation.get("multi_chart_cap_compliant") is True
    role = role_for(sample, reached, direct_cap, multi_cap)
    bridge_action, review_bucket, integration_action = action_for(role)

    return {
        "case": f"{sample.lower()}_{chunk.replace(',', '_')}_{side}_seed{seed}",
        "role": role,
        "sample": sample,
        "chunk": chunk,
        "side": side,
        "seed": seed,
        "target": rounded(sequence.get("target", parameters.get("growth_target"))),
        "local_atlas_reached": reached,
        "local_atlas_final_fraction": rounded(sequence.get("final_largest_component_fraction")),
        "local_atlas_final_quality_p90": rounded(sequence.get("final_quality_p90")),
        "local_atlas_step_count": int(sequence.get("step_count") or 0),
        "local_atlas_stop_reason": sequence.get("stop_reason"),
        "reconciliation_decision": reconciliation.get("decision"),
        "direct_global_p90": rounded(reconciliation.get("direct_global_p90_edge_distortion")),
        "direct_global_cap_compliant": direct_cap,
        "multi_chart_bridge_aware_p90": rounded(
            reconciliation.get("multi_chart_atlas_bridge_aware_p90_edge_distortion")
        ),
        "multi_chart_bridge_p90": rounded(
            reconciliation.get("multi_chart_atlas_bridge_p90_edge_distortion")
        ),
        "multi_chart_internal_p90": rounded(
            reconciliation.get("multi_chart_atlas_internal_p90_edge_distortion")
        ),
        "multi_chart_cap_compliant": multi_cap,
        "bridge_count": int(reconciliation.get("multi_chart_atlas_bridge_edge_count") or 0),
        "placed_node_fraction": rounded(
            reconciliation.get("multi_chart_atlas_placed_node_fraction")
        ),
        "unplaced_bridge_count": int(
            reconciliation.get("multi_chart_atlas_unplaced_bridge_edge_count") or 0
        ),
        "singleton_count": int(
            reconciliation.get("multi_chart_atlas_singleton_endpoint_placement_count") or 0
        ),
        "tiny_component_count": int(
            reconciliation.get("multi_chart_atlas_too_small_component_placement_count") or 0
        ),
        "bridge_quality_action": bridge_action,
        "review_bucket": review_bucket,
        "integration_action": integration_action,
        "overlap_metadata_action": "none_scalar_only",
        "vc3d_candidate_source_relation": (
            "consume candidates from SurfacePatchIndex or the existing overlap writer; "
            "score only saved scalar bridge-quality reports"
        ),
        "source_report": str(path),
        "non_claims": NON_CLAIMS,
    }


def adapter_contract() -> dict[str, Any]:
    return {
        "candidate_source": "SurfacePatchIndex_or_existing_overlap_writer",
        "metadata_write_action": "none_scalar_only",
        "required_scalar_input_fields": [
            "sample",
            "chunk",
            "side",
            "seed",
            "local_atlas_reached",
            "local_atlas_final_fraction",
            "direct_global_p90_edge_distortion",
            "direct_global_cap_compliant",
            "multi_chart_atlas_bridge_aware_p90_edge_distortion",
            "multi_chart_cap_compliant",
            "multi_chart_atlas_bridge_edge_count",
            "multi_chart_atlas_placed_node_fraction",
            "multi_chart_atlas_unplaced_bridge_edge_count",
            "multi_chart_atlas_singleton_endpoint_placement_count",
            "multi_chart_atlas_too_small_component_placement_count",
        ],
        "output_fields": [
            "bridge_quality_action",
            "review_bucket",
            "overlap_metadata_action",
            "non_claims",
        ],
        "non_claims": NON_CLAIMS,
    }


def build_payload(
    report_paths: list[Path],
    public_context_path: Path | None = DEFAULT_PUBLIC_CONTEXT,
) -> dict[str, Any]:
    rows = [report_row(path) for path in report_paths]
    positive_rows = [row for row in rows if row["role"] == "direct_global_blocker_positive"]
    control_rows = [row for row in rows if row["role"] == "specificity_control"]
    metadata_action = "none_scalar_only"
    package_ready = (
        len(positive_rows) >= 3
        and len(control_rows) >= 1
        and all(row["overlap_metadata_action"] == metadata_action for row in rows)
    )
    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "method": "vc3d_surfacepatchindex_multi_chart_adapter_contract",
        "source_reports": [str(path) for path in report_paths],
        "source_public_context": str(public_context_path) if public_context_path else None,
        "case_count": len(rows),
        "positive_blocker_count": len(positive_rows),
        "specificity_control_count": len(control_rows),
        "metadata_write_action": metadata_action,
        "adapter_contract": adapter_contract(),
        "case_rows": rows,
        "current_vc3d_context": [
            "Current VC3D main uses SurfacePatchIndex-style spatial queries for candidate discovery.",
            "Current overlap writers own overlap metadata writes and update both source and target sides.",
            "This adapter is read-only and applies a scalar multi-chart bridge-quality gate after candidate discovery.",
        ],
        "decision": (
            "package_read_only_multi_chart_bridge_quality_adapter"
            if package_ready
            else "extend_multi_chart_adapter_validation"
        ),
        "recommended_next_steps": [
            "Package the read-only adapter contract with the Thaumato progress-prize candidate.",
            "If public repository credentials become available, submit this as a scalar QA artifact before proposing VC3D code changes.",
            "Only write a VC3D patch after a current-main reproduction shows a concrete candidate-discovery or metadata-writing defect.",
        ],
        "storage_policy": (
            "Read saved scalar JSON only; no raw chunks, point clouds, geometry payloads, "
            "patch identifiers, bridge endpoints, overlap metadata, meshes, predictions, letters, or titles."
        ),
        "non_claims": NON_CLAIMS,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# VC3D Multi-Chart Adapter Contract",
        "",
        f"- timestamp UTC: {payload['generated_at_utc']}",
        f"- public context: `{payload['source_public_context']}`",
        "- script: `scripts/summarize_vc3d_multi_chart_adapter_contract.py`",
        "- machine: current `g2-standard-16` L4 VM",
        "- new data downloads: none",
        "- GPU runtime: none",
        "- cost impact: CPU-only scalar summary generation, negligible incremental cost",
        "- command: `python3 scripts/summarize_vc3d_multi_chart_adapter_contract.py`",
        "- hypothesis: the multi-chart atlas result is useful to VC3D as a read-only bridge-quality adapter after SurfacePatchIndex candidate discovery, not as an overlap metadata writer",
        "- visual checks: none; scalar-only JSON and Markdown table output",
        "- saved data policy: scalar aggregate JSON only; no raw chunks, endpoints, geometry payloads, meshes, predictions, letters, or titles",
        "",
        "## Summary",
        "",
        f"- cases: `{payload['case_count']}`",
        f"- positive blockers: `{payload['positive_blocker_count']}`",
        f"- specificity controls: `{payload['specificity_control_count']}`",
        f"- metadata write action: `{payload['metadata_write_action']}`",
        f"- decision: `{payload['decision']}`",
        "",
        "## Adapter Contract",
        "",
        "- candidate source: `SurfacePatchIndex_or_existing_overlap_writer`",
        "- adapter mode: read-only scalar bridge-quality scoring",
        "- overlap metadata action: `none_scalar_only`",
        "- output fields: `bridge_quality_action`, `review_bucket`, `overlap_metadata_action`, `non_claims`",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["current_vc3d_context"])
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| case | role | target | reached | direct p90 | direct cap | multi bridge-aware p90 | multi cap | bridges | placed | unplaced | action |",
            "| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["case_rows"]:
        lines.append(
            f"| {row['case']} | {row['role']} | {fmt(row['target'])} | "
            f"{row['local_atlas_reached']} | {fmt(row['direct_global_p90'])} | "
            f"{row['direct_global_cap_compliant']} | {fmt(row['multi_chart_bridge_aware_p90'])} | "
            f"{row['multi_chart_cap_compliant']} | {row['bridge_count']} | "
            f"{fmt(row['placed_node_fraction'])} | {row['unplaced_bridge_count']} | "
            f"{row['integration_action']} |"
        )
    lines.extend(
        [
            "",
            "## Integration Boundary",
            "",
            "This summary is a read-only adapter contract. `SurfacePatchIndex` or the current overlap writer should remain responsible for candidate discovery; this package only scores saved scalar bridge-quality evidence after a candidate exists.",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", action="append", type=Path, dest="reports")
    parser.add_argument("--public-context", type=Path, default=DEFAULT_PUBLIC_CONTEXT)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("reports/thaumato_vc3d_multi_chart_adapter_contract_2026-05-08.json"),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("reports/thaumato_vc3d_multi_chart_adapter_contract_2026-05-08.md"),
    )
    args = parser.parse_args()

    report_paths = args.reports if args.reports is not None else DEFAULT_REPORTS
    payload = build_payload(report_paths, args.public_context)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, args.md_out)
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
