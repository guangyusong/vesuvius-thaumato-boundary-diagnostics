import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_vc3d_merge_summary_adapter.py"
)
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "summarize_vc3d_merge_summary_adapter", SCRIPT_PATH
)
summary = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(summary)


def write_summary(path):
    path.write_text(
        json.dumps(
            {
                "merge_json": "/tmp/volpkg/merge.json",
                "output": "/tmp/volpkg/paths/surface_a_merged",
                "obj_out": "/tmp/volpkg/paths/surface_a_merged.obj",
                "ref_surface": "surface_a",
                "strip_count": 1,
                "surfaces": [
                    {"name": "surface_a", "path": "/tmp/a", "H": 256, "W": 256, "valid": 62000},
                    {"name": "surface_b", "path": "/tmp/b", "H": 256, "W": 256, "valid": 61000},
                    {"name": "surface_c", "path": "/tmp/c", "H": 256, "W": 256, "valid": 59000},
                ],
                "edges": [
                    {
                        "a": "surface_a",
                        "b": "surface_b",
                        "best_threshold": 5.0,
                        "best_score": 0.82,
                        "per_threshold": [{"threshold": 5.0}],
                        "anchor_count": 480,
                        "anchor_bin_size": 1,
                        "ransac_inliers": 420,
                        "ransac_total": 480,
                        "ransac_thresh": 6.0,
                        "ransac_sigma_in": 1.4,
                        "pair_scale": 1.02,
                        "real_overlap_A": 2100,
                        "real_overlap_B": 1980,
                    },
                    {
                        "a": "surface_b",
                        "b": "surface_c",
                        "best_threshold": 7.0,
                        "best_score": 0.12,
                        "per_threshold": [{"threshold": 7.0}],
                        "anchor_count": 2,
                        "anchor_bin_size": 1,
                        "ransac_inliers": 1,
                        "ransac_total": 2,
                        "ransac_thresh": 8.0,
                        "ransac_sigma_in": 4.0,
                        "pair_scale": 1.05,
                        "real_overlap_A": 60,
                        "real_overlap_B": 55,
                    },
                    {
                        "a": "surface_c",
                        "b": "surface_d",
                        "best_threshold": 6.0,
                        "best_score": 0.45,
                        "per_threshold": [{"threshold": 6.0}],
                        "anchor_count": 120,
                        "anchor_bin_size": 1,
                        "ransac_inliers": 20,
                        "ransac_total": 120,
                        "ransac_thresh": 8.0,
                        "ransac_sigma_in": 5.0,
                        "pair_scale": 1.42,
                        "real_overlap_A": 340,
                        "real_overlap_B": 360,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


class SummarizeVC3DMergeSummaryAdapterTests(unittest.TestCase):
    def test_build_payload_keeps_only_scalar_edge_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            summary_path = root / "summary.json"
            public_context = root / "public.md"
            public_context.write_text("# Public context\n", encoding="utf-8")
            write_summary(summary_path)

            payload = summary.build_payload(summary_path, public_context)

        self.assertEqual(payload["method"], "vc3d_merge_summary_scalar_adapter")
        self.assertEqual(payload["source_tool"], "vc_merge_tifxyz")
        self.assertEqual(
            payload["upstream_schema_reference"]["commit"],
            "c257b402fce35b5967fee752e415067a18d281c4",
        )
        self.assertIn(
            "summary.edges[].real_overlap_A",
            payload["adapter_contract"]["upstream_fields_verified"],
        )
        self.assertEqual(payload["metadata_write_action"], "none_scalar_only")
        self.assertEqual(payload["surface_count"], 3)
        self.assertEqual(payload["edge_count"], 3)
        self.assertEqual(payload["ready_for_bridge_qa_count"], 1)
        self.assertEqual(payload["needs_upstream_merge_review_count"], 2)
        self.assertEqual(payload["failure_reason_counts"]["low_anchor_count"], 1)
        self.assertEqual(payload["failure_reason_counts"]["pair_scale_outlier"], 1)
        self.assertEqual(payload["anchor_count_summary"]["total"], 602)
        self.assertEqual(payload["min_real_overlap"], 55)
        rows = payload["edge_rows"]
        self.assertEqual(rows[0]["review_bucket"], "ready_for_bridge_qa")
        self.assertEqual(rows[1]["failure_reasons"], ["low_anchor_count", "low_real_overlap"])
        self.assertIn("low_ransac_inlier_fraction", rows[2]["failure_reasons"])
        self.assertIn("pair_scale_outlier", rows[2]["failure_reasons"])
        for row in rows:
            self.assertNotIn("a", row)
            self.assertNotIn("b", row)
            self.assertEqual(row["overlap_metadata_action"], "none_scalar_only")

    def test_write_markdown_states_read_only_boundary(self):
        payload = {
            "generated_at_utc": "2026-05-08T00:00:00+00:00",
            "source_tool": "vc_merge_tifxyz",
            "source_summary": "summary.json",
            "source_public_context": "public.md",
            "surface_count": 3,
            "edge_count": 1,
            "strip_count": 1,
            "ready_for_bridge_qa_count": 1,
            "needs_upstream_merge_review_count": 0,
            "metadata_write_action": "none_scalar_only",
            "upstream_schema_reference": {
                "repository": "ScrollPrize/villa",
                "commit": "c257b402fce35b5967fee752e415067a18d281c4",
                "path": "volume-cartographer/apps/src/vc_merge_tifxyz.cpp",
            },
            "decision": "adapter_schema_ready_for_current_main_summary",
            "edge_rows": [
                {
                    "edge_index": 0,
                    "anchor_count": 480,
                    "ransac_inlier_fraction": 0.875,
                    "pair_scale_delta": 0.02,
                    "real_overlap_min": 1980,
                    "review_bucket": "ready_for_bridge_qa",
                    "failure_reasons": [],
                }
            ],
            "recommended_next_steps": ["Keep this adapter read-only."],
            "storage_policy": "Scalar only.",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "summary.md"
            summary.write_markdown(payload, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("vc_merge_tifxyz", text)
        self.assertIn("read-only", text)
        self.assertIn("none_scalar_only", text)
        self.assertIn("not an `overlapping.json` writer", text)
        self.assertIn("not an unwrap", text)
        self.assertIn("fields not copied", text)


if __name__ == "__main__":
    unittest.main()
