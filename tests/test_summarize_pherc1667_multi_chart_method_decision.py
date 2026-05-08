import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_pherc1667_multi_chart_method_decision.py"
)
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "summarize_pherc1667_multi_chart_method_decision", SCRIPT_PATH
)
summary = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(summary)


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def adapter_row(case, direct_p90, multi_p90, bridge_count):
    return {
        "case": case,
        "role": "direct_global_blocker_positive",
        "direct_global_p90": direct_p90,
        "direct_global_cap_compliant": False,
        "multi_chart_bridge_aware_p90": multi_p90,
        "multi_chart_cap_compliant": True,
        "bridge_count": bridge_count,
        "unplaced_bridge_count": 0,
        "integration_action": "quality_gate_pass_review_candidate",
    }


def route_row(case, target_paths, best_p90, gap):
    return {
        "case": case,
        "target_connecting_path_count": target_paths,
        "cap_compliant_target_path_count": 0,
        "target_path_p90_min": best_p90,
        "best_p90_cap_gap": gap,
        "classification": "route_substitution_negative",
    }


class SummarizePHerc1667MultiChartMethodDecisionTests(unittest.TestCase):
    def test_build_payload_joins_positive_adapter_and_negative_routes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            adapter = root / "adapter.json"
            route = root / "route.json"
            write_json(
                adapter,
                {
                    "method": "vc3d_surfacepatchindex_multi_chart_adapter_contract",
                    "decision": "package_read_only_multi_chart_bridge_quality_adapter",
                    "metadata_write_action": "none_scalar_only",
                    "positive_blocker_count": 3,
                    "specificity_control_count": 1,
                    "case_rows": [
                        adapter_row("pherc1667_13_4_3_recto_seed1", 0.230291, 0.005552, 6),
                        adapter_row("pherc1667_13_4_4_verso_seed0", 0.506859, 0.01078, 2),
                        adapter_row("pherc1667_14_4_4_verso_seed0", 0.300417, 0.039103, 3),
                        {
                            "case": "phercparis4_3_4_4_recto_seed0",
                            "role": "specificity_control",
                        },
                    ],
                },
            )
            write_json(
                route,
                {
                    "method": "pherc1667_route_substitution_audit",
                    "decision": "defer_short_route_substitution_pivot_to_multi_chart_representation",
                    "case_count": 2,
                    "target_connecting_path_count": 139,
                    "cap_compliant_target_path_count": 0,
                    "case_rows": [
                        route_row("pherc1667_13_4_3_recto_seed1", 91, 0.232185, 0.032185),
                        route_row("pherc1667_14_4_4_verso_seed0", 48, 0.301094, 0.101094),
                    ],
                },
            )

            payload = summary.build_payload(adapter, route)

        self.assertEqual(payload["method"], "pherc1667_multi_chart_method_decision")
        self.assertEqual(
            payload["decision"],
            "package_multi_chart_bridge_representation_defer_route_substitution",
        )
        self.assertEqual(payload["metadata_write_action"], "none_scalar_only")
        self.assertEqual(payload["multi_chart_positive_blocker_count"], 3)
        self.assertEqual(payload["multi_chart_specificity_control_count"], 1)
        self.assertEqual(payload["route_substitution_target_path_count"], 139)
        self.assertEqual(payload["route_substitution_cap_path_count"], 0)
        self.assertEqual(payload["shared_positive_route_negative_count"], 2)
        shared_by_case = {row["case"]: row for row in payload["shared_case_rows"]}
        first = shared_by_case["pherc1667_13_4_3_recto_seed1"]
        self.assertEqual(first["route_target_connecting_path_count"], 91)
        self.assertTrue(first["multi_chart_cap_compliant"])
        self.assertEqual(
            first["method_read"],
            "multi_chart_bridge_representation_positive_route_substitution_negative",
        )
        second = shared_by_case["pherc1667_14_4_4_verso_seed0"]
        self.assertEqual(second["route_cap_compliant_target_path_count"], 0)
        self.assertEqual(second["route_best_p90_cap_gap"], 0.101094)

    def test_write_markdown_states_decision_and_boundaries(self):
        payload = {
            "generated_at_utc": "2026-05-08T00:00:00+00:00",
            "decision": "package_multi_chart_bridge_representation_defer_route_substitution",
            "metadata_write_action": "none_scalar_only",
            "multi_chart_positive_blocker_count": 3,
            "multi_chart_specificity_control_count": 1,
            "route_substitution_case_count": 2,
            "route_substitution_target_path_count": 139,
            "route_substitution_cap_path_count": 0,
            "shared_positive_route_negative_count": 2,
            "shared_case_rows": [
                {
                    "case": "pherc1667_14_4_4_verso_seed0",
                    "direct_global_p90": 0.300417,
                    "multi_chart_bridge_aware_p90": 0.039103,
                    "multi_chart_cap_compliant": True,
                    "route_target_connecting_path_count": 48,
                    "route_cap_compliant_target_path_count": 0,
                    "route_best_target_path_p90": 0.301094,
                    "route_best_p90_cap_gap": 0.101094,
                }
            ],
            "multi_chart_positive_rows": [
                {
                    "case": "pherc1667_14_4_4_verso_seed0",
                    "direct_global_p90": 0.300417,
                    "multi_chart_bridge_aware_p90": 0.039103,
                    "multi_chart_bridge_count": 3,
                    "multi_chart_unplaced_bridge_count": 0,
                    "multi_chart_integration_action": "quality_gate_pass_review_candidate",
                }
            ],
            "route_substitution_rows": [
                {
                    "case": "pherc1667_14_4_4_verso_seed0",
                    "target_connecting_path_count": 48,
                    "cap_compliant_target_path_count": 0,
                    "target_path_p90_min": 0.301094,
                    "best_p90_cap_gap": 0.101094,
                    "classification": "route_substitution_negative",
                }
            ],
            "recommended_next_steps": ["Package the multi-chart bridge representation."],
            "storage_policy": "Scalar only.",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "summary.md"
            summary.write_markdown(payload, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("multi-chart bridge representation", text)
        self.assertIn("route substitution", text)
        self.assertIn("`139`", text)
        self.assertIn("zero cap-compliant target paths", text)
        self.assertIn("not an unwrap", text)
        self.assertIn("none_scalar_only", text)


if __name__ == "__main__":
    unittest.main()
