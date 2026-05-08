import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_vc3d_multi_chart_adapter_contract.py"
)
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "summarize_vc3d_multi_chart_adapter_contract", SCRIPT_PATH
)
summary = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(summary)


def write_report(
    path,
    sample,
    chunk,
    side,
    seed,
    target,
    reached,
    direct_p90,
    direct_cap,
    multi_p90,
    multi_cap,
    decision,
):
    path.write_text(
        json.dumps(
            {
                "source": {"sample_id": sample, "chunk": chunk},
                "parameters": {"side": side, "seed": seed, "growth_target": target},
                "local_atlas_sequence": {
                    "target": target,
                    "reached": reached,
                    "step_count": 2 if reached else 1,
                    "stop_reason": "target_reached" if reached else "no_eligible_bridge",
                    "final_largest_component_fraction": 0.76 if reached else 0.06,
                    "final_quality_p90": 0.05 if reached else 0.09,
                    "multi_chart_atlas_reconciliation": {
                        "decision": decision,
                        "direct_global_p90_edge_distortion": direct_p90,
                        "direct_global_cap_compliant": direct_cap,
                        "multi_chart_atlas_bridge_aware_p90_edge_distortion": multi_p90,
                        "multi_chart_atlas_bridge_p90_edge_distortion": multi_p90,
                        "multi_chart_atlas_internal_p90_edge_distortion": 0.001,
                        "multi_chart_cap_compliant": multi_cap,
                        "multi_chart_atlas_bridge_edge_count": 2,
                        "multi_chart_atlas_placed_node_fraction": 0.76 if reached else 0.06,
                        "multi_chart_atlas_unplaced_bridge_edge_count": 0,
                        "multi_chart_atlas_singleton_endpoint_placement_count": 1,
                        "multi_chart_atlas_too_small_component_placement_count": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


class SummarizeVC3DMultiChartAdapterContractTests(unittest.TestCase):
    def test_build_payload_maps_positive_and_specificity_control(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            positive = root / "positive.json"
            control = root / "control.json"
            public_context = root / "public.md"
            public_context.write_text("# Public context\n", encoding="utf-8")
            write_report(
                positive,
                "PHerc1667",
                [13, 4, 4],
                "verso",
                0,
                0.75,
                True,
                0.506859,
                False,
                0.01078,
                True,
                "multi_chart_atlas_resolves_direct_global_blocker",
            )
            write_report(
                control,
                "PHercParis4",
                [3, 4, 4],
                "recto",
                0,
                0.75,
                False,
                0.090298,
                True,
                0.265047,
                False,
                "direct_global_already_cap_compliant",
            )

            payload = summary.build_payload([positive, control], public_context)

        self.assertEqual(payload["method"], "vc3d_surfacepatchindex_multi_chart_adapter_contract")
        self.assertEqual(payload["positive_blocker_count"], 1)
        self.assertEqual(payload["specificity_control_count"], 1)
        self.assertEqual(payload["metadata_write_action"], "none_scalar_only")
        self.assertEqual(payload["decision"], "extend_multi_chart_adapter_validation")
        self.assertEqual(
            payload["adapter_contract"]["candidate_source"],
            "SurfacePatchIndex_or_existing_overlap_writer",
        )
        rows = {row["case"]: row for row in payload["case_rows"]}
        positive_row = rows["pherc1667_13_4_4_verso_seed0"]
        self.assertEqual(positive_row["role"], "direct_global_blocker_positive")
        self.assertEqual(
            positive_row["integration_action"],
            "quality_gate_pass_review_candidate",
        )
        self.assertEqual(positive_row["overlap_metadata_action"], "none_scalar_only")
        control_row = rows["phercparis4_3_4_4_recto_seed0"]
        self.assertEqual(control_row["role"], "specificity_control")
        self.assertEqual(control_row["integration_action"], "no_positive_action_control")
        self.assertIn("no overlap metadata writing", control_row["non_claims"])

    def test_write_markdown_states_read_only_boundary(self):
        payload = {
            "generated_at_utc": "2026-05-08T00:00:00+00:00",
            "source_public_context": "public.md",
            "case_count": 1,
            "positive_blocker_count": 1,
            "specificity_control_count": 0,
            "metadata_write_action": "none_scalar_only",
            "decision": "extend_multi_chart_adapter_validation",
            "current_vc3d_context": [
                "Current VC3D main uses SurfacePatchIndex-style spatial queries."
            ],
            "case_rows": [
                {
                    "case": "pherc1667_13_4_4_verso_seed0",
                    "role": "direct_global_blocker_positive",
                    "target": 0.75,
                    "local_atlas_reached": True,
                    "direct_global_p90": 0.506859,
                    "direct_global_cap_compliant": False,
                    "multi_chart_bridge_aware_p90": 0.01078,
                    "multi_chart_cap_compliant": True,
                    "bridge_count": 2,
                    "placed_node_fraction": 0.751244,
                    "unplaced_bridge_count": 0,
                    "integration_action": "quality_gate_pass_review_candidate",
                }
            ],
            "recommended_next_steps": ["Package the read-only adapter contract."],
            "storage_policy": "Scalar only.",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "summary.md"
            summary.write_markdown(payload, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("SurfacePatchIndex", text)
        self.assertIn("read-only", text)
        self.assertIn("none_scalar_only", text)
        self.assertIn("not an `overlapping.json` writer", text)
        self.assertIn("not an unwrap", text)


if __name__ == "__main__":
    unittest.main()
