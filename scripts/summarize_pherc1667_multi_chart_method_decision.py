#!/usr/bin/env python3
"""Join multi-chart adapter and route-substitution evidence into a method decision."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


DEFAULT_ADAPTER_REPORT = Path(
    "reports/thaumato_vc3d_multi_chart_adapter_contract_2026-05-08.json"
)
DEFAULT_ROUTE_REPORT = Path(
    "reports/thaumato_pherc1667_route_substitution_audit_2026-05-08.json"
)

NON_CLAIMS = [
    "not an unwrap",
    "not text, ink, letters, or title",
    "not an overlapping.json writer",
    "not a route-substitution recovery claim",
    "not a VC3D metadata mutation",
]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def rounded(value: Any, digits: int = 6) -> float | None:
    return round(float(value), digits) if is_number(value) else None


def fmt(value: Any, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}" if is_number(value) else "n/a"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def adapter_positive_rows(adapter: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in adapter.get("case_rows") or []
        if isinstance(row, dict) and row.get("role") == "direct_global_blocker_positive"
    ]


def adapter_control_rows(adapter: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in adapter.get("case_rows") or []
        if isinstance(row, dict) and row.get("role") == "specificity_control"
    ]


def decision_ready(
    adapter: dict[str, Any],
    route: dict[str, Any],
    shared_rows: list[dict[str, Any]],
) -> bool:
    return (
        adapter.get("method") == "vc3d_surfacepatchindex_multi_chart_adapter_contract"
        and adapter.get("decision") == "package_read_only_multi_chart_bridge_quality_adapter"
        and adapter.get("metadata_write_action") == "none_scalar_only"
        and route.get("method") == "pherc1667_route_substitution_audit"
        and route.get("decision")
        == "defer_short_route_substitution_pivot_to_multi_chart_representation"
        and int(adapter.get("positive_blocker_count") or 0) >= 3
        and int(adapter.get("specificity_control_count") or 0) >= 1
        and int(route.get("case_count") or 0) >= 2
        and int(route.get("target_connecting_path_count") or 0) > 0
        and int(route.get("cap_compliant_target_path_count") or 0) == 0
        and len(shared_rows) >= 2
    )


def positive_case_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": row.get("case"),
        "direct_global_p90": rounded(row.get("direct_global_p90")),
        "direct_global_cap_compliant": row.get("direct_global_cap_compliant") is True,
        "multi_chart_bridge_aware_p90": rounded(row.get("multi_chart_bridge_aware_p90")),
        "multi_chart_cap_compliant": row.get("multi_chart_cap_compliant") is True,
        "multi_chart_bridge_count": int(row.get("bridge_count") or 0),
        "multi_chart_unplaced_bridge_count": int(row.get("unplaced_bridge_count") or 0),
        "multi_chart_integration_action": row.get("integration_action"),
    }


def route_case_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": row.get("case"),
        "target_connecting_path_count": int(row.get("target_connecting_path_count") or 0),
        "cap_compliant_target_path_count": int(
            row.get("cap_compliant_target_path_count") or 0
        ),
        "target_path_p90_min": rounded(row.get("target_path_p90_min")),
        "best_p90_cap_gap": rounded(row.get("best_p90_cap_gap")),
        "classification": row.get("classification"),
    }


def shared_case_row(
    adapter_row: dict[str, Any],
    route_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case": adapter_row.get("case"),
        "direct_global_p90": rounded(adapter_row.get("direct_global_p90")),
        "multi_chart_bridge_aware_p90": rounded(
            adapter_row.get("multi_chart_bridge_aware_p90")
        ),
        "multi_chart_cap_compliant": adapter_row.get("multi_chart_cap_compliant") is True,
        "route_target_connecting_path_count": int(
            route_row.get("target_connecting_path_count") or 0
        ),
        "route_cap_compliant_target_path_count": int(
            route_row.get("cap_compliant_target_path_count") or 0
        ),
        "route_best_target_path_p90": rounded(route_row.get("target_path_p90_min")),
        "route_best_p90_cap_gap": rounded(route_row.get("best_p90_cap_gap")),
        "method_read": (
            "multi_chart_bridge_representation_positive_route_substitution_negative"
        ),
    }


def build_payload(adapter_path: Path, route_path: Path) -> dict[str, Any]:
    adapter = load_json(adapter_path)
    route = load_json(route_path)
    positives = adapter_positive_rows(adapter)
    controls = adapter_control_rows(adapter)
    route_rows = [
        row for row in route.get("case_rows") or [] if isinstance(row, dict)
    ]
    adapter_by_case = {row.get("case"): row for row in positives}
    route_by_case = {row.get("case"): row for row in route_rows}
    shared = [
        shared_case_row(adapter_by_case[case], route_by_case[case])
        for case in sorted(set(adapter_by_case).intersection(route_by_case))
    ]
    ready = decision_ready(adapter, route, shared)
    decision = (
        "package_multi_chart_bridge_representation_defer_route_substitution"
        if ready
        else "extend_multi_chart_route_decision_evidence"
    )
    route_target_count = int(route.get("target_connecting_path_count") or 0)
    route_cap_count = int(route.get("cap_compliant_target_path_count") or 0)
    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "method": "pherc1667_multi_chart_method_decision",
        "source_reports": [str(adapter_path), str(route_path)],
        "decision": decision,
        "metadata_write_action": adapter.get("metadata_write_action"),
        "multi_chart_positive_blocker_count": int(
            adapter.get("positive_blocker_count") or len(positives)
        ),
        "multi_chart_specificity_control_count": int(
            adapter.get("specificity_control_count") or len(controls)
        ),
        "route_substitution_case_count": int(route.get("case_count") or len(route_rows)),
        "route_substitution_target_path_count": route_target_count,
        "route_substitution_cap_path_count": route_cap_count,
        "shared_positive_route_negative_count": len(shared),
        "multi_chart_positive_rows": [positive_case_row(row) for row in positives],
        "route_substitution_rows": [route_case_row(row) for row in route_rows],
        "shared_case_rows": shared,
        "evidence_read": (
            f"{len(positives)} multi-chart direct-global blocker positives, "
            f"{len(controls)} specificity control, {route_target_count} target-connecting "
            f"route-substitution paths, and {route_cap_count} cap-compliant route paths."
        ),
        "storage_policy": (
            "Read saved scalar JSON only; no raw chunks, endpoints, path signatures, "
            "patch identifiers, component identifiers, coordinates, meshes, predictions, "
            "ink, letters, or titles."
        ),
        "non_claims": NON_CLAIMS,
        "recommended_next_steps": [
            "Package the multi-chart bridge representation as the reviewer-facing method decision.",
            "Keep route substitution as bounded failure analysis unless a future representation lowers the strict p90 gap.",
            "If continuing technically, add one current-public-repo adapter dry-run or one additional specificity control before proposing code changes.",
        ],
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# PHerc1667 Multi-Chart Method Decision",
        "",
        f"- timestamp UTC: {payload['generated_at_utc']}",
        "- script: `scripts/summarize_pherc1667_multi_chart_method_decision.py`",
        "- machine: current `g2-standard-16` L4 VM",
        "- new data downloads: none",
        "- GPU runtime: none; scalar-only summary generation",
        "- cost impact: negligible CPU-only package bookkeeping",
        "- command: `python3 scripts/summarize_pherc1667_multi_chart_method_decision.py`",
        "- hypothesis: if short-route substitution is the right next method, saved blocker cases should expose at least one strict-cap-compliant target path; if not, package the multi-chart bridge representation instead",
        "- visual checks: none; scalar-only JSON and Markdown table output",
        "- saved data policy: scalar aggregate JSON only; no raw chunks, endpoints, path signatures, component identifiers, coordinates, meshes, predictions, letters, or titles",
        "",
        "## Summary",
        "",
        f"- decision: `{payload['decision']}`",
        f"- metadata write action: `{payload['metadata_write_action']}`",
        f"- multi-chart direct-global blocker positives: `{payload['multi_chart_positive_blocker_count']}`",
        f"- multi-chart specificity controls: `{payload['multi_chart_specificity_control_count']}`",
        f"- route-substitution cases: `{payload['route_substitution_case_count']}`",
        f"- route-substitution target paths: `{payload['route_substitution_target_path_count']}`",
        f"- route-substitution cap paths: `{payload['route_substitution_cap_path_count']}`",
        f"- shared positive/route-negative cases: `{payload['shared_positive_route_negative_count']}`",
        "",
        f"The package decision is to emphasize multi-chart bridge representation, while keeping route substitution as bounded failure analysis. The adapter contract has {payload['multi_chart_positive_blocker_count']} PHerc1667 direct-global blocker positives plus {payload['multi_chart_specificity_control_count']} specificity controls with `none_scalar_only` metadata action. The route audit finds `{payload['route_substitution_target_path_count']}` target-connecting short routes across two blocker cases and zero cap-compliant target paths under the strict `0.20` p90 cap.",
        "",
        "## Shared Cases",
        "",
        "| case | direct p90 | multi-chart p90 | multi-chart cap | route target paths | route cap paths | route best p90 | route p90 gap |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["shared_case_rows"]:
        lines.append(
            f"| {row['case']} | {fmt(row['direct_global_p90'])} | "
            f"{fmt(row['multi_chart_bridge_aware_p90'])} | "
            f"{row['multi_chart_cap_compliant']} | "
            f"{row['route_target_connecting_path_count']} | "
            f"{row['route_cap_compliant_target_path_count']} | "
            f"{fmt(row['route_best_target_path_p90'])} | "
            f"{fmt(row['route_best_p90_cap_gap'])} |"
        )
    lines.extend(
        [
            "",
            "## Multi-Chart Positive Cases",
            "",
            "| case | direct p90 | multi-chart p90 | bridges | unplaced | action |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["multi_chart_positive_rows"]:
        lines.append(
            f"| {row['case']} | {fmt(row['direct_global_p90'])} | "
            f"{fmt(row['multi_chart_bridge_aware_p90'])} | "
            f"{row['multi_chart_bridge_count']} | "
            f"{row['multi_chart_unplaced_bridge_count']} | "
            f"{row['multi_chart_integration_action']} |"
        )
    lines.extend(
        [
            "",
            "## Route-Substitution Audit",
            "",
            "| case | target paths | cap paths | best p90 | p90 gap | classification |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["route_substitution_rows"]:
        lines.append(
            f"| {row['case']} | {row['target_connecting_path_count']} | "
            f"{row['cap_compliant_target_path_count']} | "
            f"{fmt(row['target_path_p90_min'])} | {fmt(row['best_p90_cap_gap'])} | "
            f"{row['classification']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is the reviewer-facing method decision that ties the positive and negative evidence together. Multi-chart bridge representation resolves the saved direct-global blocker cases at the scalar QA level, while route substitution provides zero cap-compliant alternatives on the two overlapping blocker cases. The package should therefore present route substitution as a rejected branch, not as the main contribution.",
            "",
            "This is not an unwrap, text, ink, letter, title, `overlapping.json` writer, or VC3D metadata mutation claim.",
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
    parser.add_argument("--adapter-json", type=Path, default=DEFAULT_ADAPTER_REPORT)
    parser.add_argument("--route-json", type=Path, default=DEFAULT_ROUTE_REPORT)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(
            "reports/thaumato_pherc1667_multi_chart_method_decision_2026-05-08.json"
        ),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path(
            "reports/thaumato_pherc1667_multi_chart_method_decision_2026-05-08.md"
        ),
    )
    args = parser.parse_args()

    payload = build_payload(args.adapter_json, args.route_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, args.md_out)
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
