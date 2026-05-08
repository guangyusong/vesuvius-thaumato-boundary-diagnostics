import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "plot_thaumato_patch_graph_growth_qa.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("plot_thaumato_patch_graph_growth_qa", SCRIPT_PATH)
plotter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plotter)


class PlotThaumatoPatchGraphGrowthQATests(unittest.TestCase):
    def test_two_chart_rigid_gate_metrics_labels_projection_and_orientation(self):
        base_metrics = {
            "two_chart_rigid_trial_available": True,
            "two_chart_rigid_trial_orthogonal_transform_determinant": 1.0,
            "two_chart_rigid_trial_bridge_projection_ratio": 0.9,
            "two_chart_rigid_trial_p90_edge_distortion": 0.1,
        }

        success = plotter.two_chart_rigid_trial_gate_metrics(base_metrics, 0.2)
        self.assertTrue(success["two_chart_rigid_trial_selection_gate_pass"])
        self.assertEqual(success["two_chart_rigid_trial_placement_class"], "rigid_cap_success")

        low_projection = plotter.two_chart_rigid_trial_gate_metrics(
            {**base_metrics, "two_chart_rigid_trial_bridge_projection_ratio": 0.55},
            0.2,
        )
        self.assertFalse(low_projection["two_chart_rigid_trial_selection_gate_pass"])
        self.assertEqual(
            low_projection["two_chart_rigid_trial_placement_class"],
            "low_bridge_projection_failure",
        )

        reflection = plotter.two_chart_rigid_trial_gate_metrics(
            {
                **base_metrics,
                "two_chart_rigid_trial_orthogonal_transform_determinant": -1.0,
            },
            0.2,
        )
        self.assertFalse(reflection["two_chart_rigid_trial_selection_gate_pass"])
        self.assertEqual(
            reflection["two_chart_rigid_trial_placement_class"],
            "reflection_like_transform_failure",
        )

    def test_two_chart_rigid_trial_reports_bridge_internal_ratio(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0])},
            {"centroid": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([2.0, 0.0, 1.0])},
            {"centroid": np.array([3.0, 0.0, 2.0])},
        ]
        axes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        chart_frames = {
            0: {"axes": axes, "centroid": np.array([0.0, 0.0, 0.0])},
            2: {"axes": axes, "centroid": np.array([2.0, 0.0, 1.0])},
        }

        metrics = plotter.two_chart_rigid_trial_metrics(
            nodes=nodes,
            base_edges={(2, 3)},
            base_roots=[0, 0, 2, 2],
            component_sizes={0: 2, 2: 2},
            edge=(1, 2),
            chart_frames=chart_frames,
        )

        self.assertTrue(metrics["two_chart_rigid_trial_available"])
        self.assertAlmostEqual(
            metrics["two_chart_rigid_trial_bridge_edge_distortion"],
            metrics["two_chart_rigid_trial_internal_p90_edge_distortion"],
        )
        self.assertAlmostEqual(
            metrics["two_chart_rigid_trial_bridge_to_internal_p90_ratio"],
            1.0,
        )

    def test_multi_chart_atlas_metrics_places_two_bridges_without_exporting_edges(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 4.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 6.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 7.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 6.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        base_edges = {
            (0, 1),
            (0, 2),
            (1, 2),
            (3, 4),
            (3, 5),
            (4, 5),
            (6, 7),
            (6, 8),
            (7, 8),
        }
        grown_edges = set(base_edges)
        grown_edges.update({(2, 3), (5, 6)})

        metrics = plotter.multi_chart_atlas_metrics(nodes, base_edges, grown_edges)

        self.assertTrue(metrics["multi_chart_atlas_available"])
        self.assertEqual(metrics["multi_chart_atlas_placed_component_count"], 3)
        self.assertEqual(metrics["multi_chart_atlas_bridge_edge_count"], 2)
        self.assertEqual(metrics["multi_chart_atlas_unplaced_bridge_edge_count"], 0)
        self.assertAlmostEqual(metrics["multi_chart_atlas_placed_node_fraction"], 1.0)
        self.assertLessEqual(metrics["multi_chart_atlas_bridge_aware_p90_edge_distortion"], 0.01)
        self.assertNotIn("edge", metrics)

    def test_multi_chart_atlas_metrics_places_singleton_endpoint_bridge(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        base_edges = {(0, 1), (0, 2), (1, 2)}
        grown_edges = set(base_edges)
        grown_edges.add((2, 3))

        metrics = plotter.multi_chart_atlas_metrics(nodes, base_edges, grown_edges)

        self.assertTrue(metrics["multi_chart_atlas_available"])
        self.assertEqual(metrics["multi_chart_atlas_placed_component_count"], 2)
        self.assertEqual(metrics["multi_chart_atlas_bridge_edge_count"], 1)
        self.assertEqual(metrics["multi_chart_atlas_singleton_endpoint_placement_count"], 1)
        self.assertEqual(metrics["multi_chart_atlas_too_small_component_placement_count"], 1)
        self.assertEqual(metrics["multi_chart_atlas_unplaced_bridge_edge_count"], 0)
        self.assertAlmostEqual(metrics["multi_chart_atlas_placed_node_fraction"], 1.0)
        self.assertLessEqual(metrics["multi_chart_atlas_bridge_aware_p90_edge_distortion"], 0.01)

    def test_multi_chart_atlas_metrics_places_two_node_endpoint_bridge(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 4.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        base_edges = {(0, 1), (0, 2), (1, 2), (3, 4)}
        grown_edges = set(base_edges)
        grown_edges.add((2, 3))

        metrics = plotter.multi_chart_atlas_metrics(nodes, base_edges, grown_edges)

        self.assertTrue(metrics["multi_chart_atlas_available"])
        self.assertEqual(metrics["multi_chart_atlas_singleton_endpoint_placement_count"], 0)
        self.assertEqual(metrics["multi_chart_atlas_too_small_component_placement_count"], 1)
        self.assertEqual(metrics["multi_chart_atlas_bridge_edge_count"], 1)
        self.assertEqual(metrics["multi_chart_atlas_unplaced_bridge_edge_count"], 0)
        self.assertAlmostEqual(metrics["multi_chart_atlas_placed_node_fraction"], 1.0)
        self.assertLessEqual(metrics["multi_chart_atlas_bridge_aware_p90_edge_distortion"], 0.01)

    def test_multi_chart_atlas_metrics_counts_unplaced_when_no_chart_frame_exists(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 4.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        base_edges = {(0, 1), (2, 3)}
        grown_edges = set(base_edges)
        grown_edges.add((1, 2))

        metrics = plotter.multi_chart_atlas_metrics(nodes, base_edges, grown_edges)

        self.assertFalse(metrics["multi_chart_atlas_available"])
        self.assertEqual(metrics["multi_chart_atlas_singleton_endpoint_placement_count"], 0)
        self.assertEqual(metrics["multi_chart_atlas_too_small_component_placement_count"], 0)
        self.assertEqual(metrics["multi_chart_atlas_unplaced_bridge_edge_count"], 1)
        self.assertEqual(
            metrics["multi_chart_atlas_unplaced_bridge_missing_chart_frame_count"],
            1,
        )
        self.assertEqual(
            metrics["multi_chart_atlas_unplaced_bridge_missing_chart_frame_endpoint_count"],
            2,
        )
        self.assertEqual(
            metrics["multi_chart_atlas_unplaced_bridge_too_small_endpoint_count"],
            2,
        )
        self.assertEqual(metrics["multi_chart_atlas_unplaced_bridge_disconnected_count"], 0)

    def test_build_patch_graph_and_growth_edges(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        points = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 1, 1],
                [0, 4, 0],
                [0, 5, 0],
                [0, 4, 1],
                [0, 5, 1],
                [0, 0, 4],
                [1, 0, 4],
                [0, 0, 5],
                [1, 0, 5],
                [0, 4, 4],
                [1, 4, 4],
                [0, 4, 5],
                [1, 4, 5],
            ],
            dtype=np.float64,
        )
        normals = np.array([[1, 0, 0]] * 16, dtype=np.float64)

        nodes, edges, edge_quality = plotter.build_patch_graph(
            points, normals, cell_size=4.0, min_points_per_cell=4
        )
        base_edges = set(plotter.pruned_edges(edge_quality, 0.75, 0.35))
        grown_edges, added_edges = plotter.grow_distortion_edges(
            nodes, edges, base_edges, edge_quality, target=0.75
        )

        self.assertEqual(len(nodes), 4)
        self.assertEqual(len(edges), 6)
        self.assertLess(plotter.largest_fraction(len(nodes), base_edges), 0.75)
        self.assertGreater(len(added_edges), 0)
        self.assertGreaterEqual(plotter.largest_fraction(len(nodes), grown_edges), 0.75)

    def test_candidate_bridge_diagnostics_reports_cap_compliance(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([1.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        all_edges = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
        base_edges = {(0, 1), (0, 2), (1, 2)}
        edge_quality = [
            {"edge": edge, "normal_agreement": 0.9, "offset_ratio": 0.1}
            for edge in all_edges
        ]

        diagnostics = plotter.candidate_bridge_diagnostics(
            nodes,
            all_edges,
            base_edges,
            edge_quality,
            max_p90_distortion=0.0,
            limit=2,
        )

        self.assertEqual(diagnostics["bridge_candidate_count"], 3)
        self.assertEqual(len(diagnostics["top_by_low_p90"]), 2)
        self.assertEqual(len(diagnostics["top_by_coverage"]), 2)
        self.assertTrue(all("edge" not in row for row in diagnostics["top_by_low_p90"]))
        self.assertLessEqual(
            diagnostics["cap_compliant_bridge_candidate_count"],
            diagnostics["bridge_candidate_count"],
        )

    def test_candidate_bridge_diagnostics_reports_broad_policy(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([1.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([2.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([3.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        all_edges = [(0, 1), (1, 2), (0, 3), (2, 3)]
        base_edges = {(0, 1), (1, 2)}
        edge_quality = [
            {"edge": (0, 1), "normal_agreement": 0.95, "offset_ratio": 0.1},
            {"edge": (1, 2), "normal_agreement": 0.95, "offset_ratio": 0.1},
            {"edge": (0, 3), "normal_agreement": 0.93, "offset_ratio": 0.3},
            {"edge": (2, 3), "normal_agreement": 0.93, "offset_ratio": 0.3},
        ]

        def fake_unwrap_metrics(_nodes, edges):
            edge_set = {tuple(edge) for edge in edges}
            return {
                "patch_unwrap_proxy_largest_component_node_fraction": plotter.largest_fraction(
                    len(nodes),
                    edge_set,
                ),
                "patch_unwrap_proxy_p10_triangle_area_ratio": 0.8,
                "patch_unwrap_proxy_p90_edge_distortion": 0.3
                if (0, 3) in edge_set or (2, 3) in edge_set
                else 0.0,
            }

        def fake_chart_metrics(_nodes, edge, _base_roots, _chart_frames):
            projection = 0.9 if tuple(edge) == (0, 3) else 0.7
            return {
                "bridge_normal_offset_ratio_left": 0.1,
                "bridge_normal_offset_ratio_right": 0.1,
                "bridge_projection_ratio_left": projection,
                "bridge_projection_ratio_right": projection,
                "chart_normal_agreement": 0.9,
                "chart_origin_separation": 1.0,
                "chart_pair_available": True,
                "chart_tangent_singular_max": 1.0,
                "chart_tangent_singular_min": 0.9,
                "left_chart_component_fraction": 0.75,
                "right_chart_component_fraction": 0.25,
            }

        with mock.patch.object(plotter.ablation, "unwrap_proxy_metrics", side_effect=fake_unwrap_metrics), mock.patch.object(
            plotter,
            "bridge_chart_frame_metrics",
            side_effect=fake_chart_metrics,
        ):
            diagnostics = plotter.candidate_bridge_diagnostics(
                nodes,
                all_edges,
                base_edges,
                edge_quality,
                max_p90_distortion=0.2,
                target=0.9,
                limit=4,
            )

        policy = diagnostics["top_coverage_normal_projection_policy"]
        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["pass_count"], 1)
        self.assertAlmostEqual(policy["best_trial_largest_component_fraction"], 1.0)
        row = policy["top_by_policy"][0]
        self.assertAlmostEqual(row["min_bridge_projection_ratio"], 0.9)
        self.assertNotIn("edge", row)

    def test_candidate_diagnostics_only_sequence_marks_growth_skipped(self):
        sequence = plotter.candidate_diagnostics_only_sequence(0.75, 0.5)
        self.assertFalse(sequence["enabled"])
        self.assertEqual(sequence["reason"], "candidate_diagnostics_only")
        self.assertFalse(sequence["reached"])
        self.assertAlmostEqual(sequence["target"], 0.75)
        self.assertAlmostEqual(sequence["final_largest_component_fraction"], 0.5)

        metrics = plotter.candidate_diagnostics_only_local_metrics()
        self.assertIn("local_chart_quality_p90", metrics)
        self.assertIsNone(metrics["local_chart_quality_p90"])

    def test_cap_base_edges_by_incremental_p90_skips_bad_base_edge(self):
        nodes = [{}, {}, {}]
        base_edges = {(0, 1), (1, 2), (0, 2)}
        edge_quality = [
            {"edge": (0, 1), "normal_agreement": 1.0, "offset_ratio": 0.0},
            {"edge": (1, 2), "normal_agreement": 1.0, "offset_ratio": 0.0},
            {"edge": (0, 2), "normal_agreement": 0.5, "offset_ratio": 0.5},
        ]

        def fake_unwrap_metrics(_nodes, edges):
            edge_set = {tuple(edge) for edge in edges}
            p90 = 0.3 if (0, 2) in edge_set else 0.0 if edge_set else None
            return {
                "patch_unwrap_proxy_p90_edge_distortion": p90,
                "patch_unwrap_proxy_largest_component_node_fraction": plotter.largest_fraction(
                    len(nodes),
                    edge_set,
                ),
            }

        with mock.patch.object(plotter.ablation, "unwrap_proxy_metrics", side_effect=fake_unwrap_metrics):
            capped_edges, summary = plotter.cap_base_edges_by_incremental_p90(
                nodes,
                base_edges,
                edge_quality,
                max_p90_distortion=0.2,
            )

        self.assertEqual(capped_edges, {(0, 1), (1, 2)})
        self.assertEqual(summary["input_base_edge_count"], 3)
        self.assertEqual(summary["accepted_base_edge_count"], 2)
        self.assertEqual(summary["skipped_base_edge_count"], 1)
        self.assertAlmostEqual(summary["input_base_p90_edge_distortion"], 0.3)
        self.assertAlmostEqual(summary["capped_base_p90_edge_distortion"], 0.0)
        self.assertEqual(summary["skipped_trial_p90_summary"]["count"], 1)
        self.assertEqual(summary["skipped_cycle_edge_count"], 1)
        self.assertEqual(summary["skipped_bridge_edge_count"], 0)
        self.assertEqual(summary["skipped_p90_cap_gap_summary"]["count"], 1)
        self.assertAlmostEqual(summary["skipped_p90_cap_gap_summary"]["min"], 0.1)
        skipped = summary["top_skipped_base_edges_by_low_p90"][0]
        self.assertEqual(skipped["component_relation"], "cycle")
        self.assertAlmostEqual(skipped["trial_largest_component_fraction"], 1.0)
        self.assertAlmostEqual(skipped["trial_p90_edge_distortion"], 0.3)
        self.assertAlmostEqual(skipped["p90_cap_gap"], 0.1)
        self.assertNotIn("edge", skipped)
        self.assertFalse(summary["skipped_alternate_route_probe"]["enabled"])
        self.assertEqual(summary["skipped_alternate_route_probe"]["reason"], "not_requested")
        self.assertFalse(summary["skipped_bridge_component_probe"]["enabled"])
        self.assertEqual(summary["skipped_bridge_component_probe"]["reason"], "not_requested")

    def test_cap_base_alternate_route_probe_reports_scalar_alternative(self):
        nodes = [{}, {}, {}, {}]
        base_edges = {(0, 1), (2, 3), (1, 2)}
        all_edges = [(0, 1), (2, 3), (1, 2), (0, 3)]
        edge_quality = [
            {"edge": (0, 1), "normal_agreement": 1.0, "offset_ratio": 0.0},
            {"edge": (2, 3), "normal_agreement": 0.99, "offset_ratio": 0.0},
            {"edge": (1, 2), "normal_agreement": 0.9, "offset_ratio": 0.1},
            {"edge": (0, 3), "normal_agreement": 0.95, "offset_ratio": 0.05},
        ]

        def fake_unwrap_metrics(_nodes, edges):
            edge_set = {tuple(edge) for edge in edges}
            if (1, 2) in edge_set:
                p90 = 0.3
            elif (0, 3) in edge_set:
                p90 = 0.1
            else:
                p90 = 0.0 if edge_set else None
            return {
                "patch_unwrap_proxy_p90_edge_distortion": p90,
                "patch_unwrap_proxy_largest_component_node_fraction": plotter.largest_fraction(
                    len(nodes),
                    edge_set,
                ),
                "patch_unwrap_proxy_p10_triangle_area_ratio": 0.8,
            }

        with mock.patch.object(plotter.ablation, "unwrap_proxy_metrics", side_effect=fake_unwrap_metrics):
            capped_edges, summary = plotter.cap_base_edges_by_incremental_p90(
                nodes,
                base_edges,
                edge_quality,
                max_p90_distortion=0.2,
                all_edges=all_edges,
                alternate_route_max_length=2,
                alternate_route_branching=4,
                diagnostic_limit=3,
            )

        self.assertEqual(capped_edges, {(0, 1), (2, 3)})
        probe = summary["skipped_alternate_route_probe"]
        self.assertTrue(probe["enabled"])
        self.assertEqual(probe["skipped_edge_count"], 1)
        self.assertEqual(probe["target_connecting_path_count"], 1)
        self.assertEqual(probe["cap_compliant_target_path_count"], 1)
        self.assertAlmostEqual(probe["best_cap_compliant_trial_fraction"], 1.0)
        self.assertAlmostEqual(probe["best_cap_compliant_p90"], 0.1)
        skipped_summary = probe["skipped_edge_route_summaries"][0]
        self.assertEqual(skipped_summary["component_relation"], "bridge")
        self.assertEqual(skipped_summary["target_connecting_path_count"], 1)
        row = skipped_summary["top_by_low_p90"][0]
        self.assertEqual(row["path_length"], 1)
        self.assertTrue(row["cap_compliant"])
        self.assertAlmostEqual(row["trial_p90_edge_distortion"], 0.1)
        self.assertNotIn("edge", row)
        self.assertNotIn("node_index", row)

    def test_cap_base_component_probe_reports_two_chart_surrogate(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 4.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        left_edges = {(0, 1), (0, 2), (1, 2)}
        right_edges = {(3, 4), (3, 5), (4, 5)}
        bridge = (2, 3)
        base_edges = {*left_edges, *right_edges, bridge}
        edge_quality = [
            {"edge": edge, "normal_agreement": 1.0, "offset_ratio": 0.0}
            for edge in sorted(left_edges | right_edges)
        ]
        edge_quality.append({"edge": bridge, "normal_agreement": 0.844132, "offset_ratio": 0.190131})

        def fake_unwrap_metrics(_nodes, edges):
            edge_set = {tuple(edge) for edge in edges}
            if bridge in edge_set:
                p90 = 0.234847
            elif edge_set == left_edges:
                p90 = 0.01
            elif edge_set == right_edges:
                p90 = 0.02
            else:
                p90 = 0.0 if edge_set else None
            return {
                "patch_unwrap_proxy_p90_edge_distortion": p90,
                "patch_unwrap_proxy_largest_component_node_fraction": plotter.largest_fraction(
                    len(nodes),
                    edge_set,
                ),
                "patch_unwrap_proxy_p10_triangle_area_ratio": 0.8,
            }

        with mock.patch.object(plotter.ablation, "unwrap_proxy_metrics", side_effect=fake_unwrap_metrics):
            capped_edges, summary = plotter.cap_base_edges_by_incremental_p90(
                nodes,
                base_edges,
                edge_quality,
                max_p90_distortion=0.2,
                component_probe=True,
                diagnostic_limit=3,
            )

        self.assertEqual(capped_edges, left_edges | right_edges)
        probe = summary["skipped_bridge_component_probe"]
        self.assertTrue(probe["enabled"])
        self.assertEqual(probe["skipped_bridge_edge_count"], 1)
        self.assertEqual(probe["global_trial_cap_compliant_count"], 0)
        self.assertEqual(probe["two_chart_quality_cap_compliant_count"], 1)
        self.assertEqual(probe["two_chart_rigid_trial_cap_compliant_count"], 1)
        self.assertEqual(probe["two_chart_rigid_trial_selection_gate_pass_count"], 1)
        self.assertEqual(
            probe["two_chart_rigid_trial_placement_class_counts"],
            {"rigid_cap_success": 1},
        )
        self.assertAlmostEqual(probe["best_two_chart_quality"], 0.190131)
        self.assertLessEqual(probe["two_chart_rigid_trial_p90_summary"]["p90"], 0.2)
        row = probe["top_by_two_chart_quality"][0]
        self.assertTrue(row["two_chart_quality_cap_compliant"])
        self.assertTrue(row["two_chart_rigid_trial_cap_compliant"])
        self.assertTrue(row["two_chart_rigid_trial_selection_gate_pass"])
        self.assertEqual(row["two_chart_rigid_trial_placement_class"], "rigid_cap_success")
        self.assertFalse(row["global_trial_cap_compliant"])
        self.assertAlmostEqual(row["global_trial_p90_edge_distortion"], 0.234847)
        self.assertLessEqual(row["two_chart_rigid_trial_p90_edge_distortion"], 0.2)
        self.assertAlmostEqual(row["transition_offset_ratio"], 0.190131)
        self.assertTrue(row["chart_pair_available"])
        self.assertNotIn("edge", row)
        self.assertNotIn("node_index", row)

    def test_base_cap_boundary_transition_prototype_accepts_local_boundary(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 4.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        left_edges = {(0, 1), (0, 2), (1, 2)}
        right_edges = {(3, 4), (3, 5), (4, 5)}
        bridge = (2, 3)
        base_edges = {*left_edges, *right_edges, bridge}
        edge_quality = [
            {"edge": edge, "normal_agreement": 1.0, "offset_ratio": 0.0}
            for edge in sorted(left_edges | right_edges)
        ]
        edge_quality.append({"edge": bridge, "normal_agreement": 0.844132, "offset_ratio": 0.190131})

        def fake_unwrap_metrics(_nodes, edges):
            edge_set = {tuple(edge) for edge in edges}
            if bridge in edge_set:
                p90 = 0.234847
            elif edge_set == left_edges:
                p90 = 0.01
            elif edge_set == right_edges:
                p90 = 0.02
            else:
                p90 = 0.0 if edge_set else None
            return {
                "patch_unwrap_proxy_p90_edge_distortion": p90,
                "patch_unwrap_proxy_largest_component_node_fraction": plotter.largest_fraction(
                    len(nodes),
                    edge_set,
                ),
                "patch_unwrap_proxy_p10_triangle_area_ratio": 0.8,
            }

        with mock.patch.object(plotter.ablation, "unwrap_proxy_metrics", side_effect=fake_unwrap_metrics):
            capped_edges, summary = plotter.cap_base_edges_by_incremental_p90(
                nodes,
                base_edges,
                edge_quality,
                max_p90_distortion=0.2,
                boundary_prototype=True,
                boundary_target=0.9,
                diagnostic_limit=3,
            )

        self.assertEqual(capped_edges, left_edges | right_edges)
        prototype = summary["skipped_bridge_boundary_transition_prototype"]
        self.assertTrue(prototype["enabled"])
        self.assertFalse(prototype["requires_two_chart_rigid_trial_selection_gate"])
        self.assertEqual(prototype["boundary_transition_candidate_count"], 1)
        self.assertEqual(prototype["cap_compliant_boundary_transition_candidate_count"], 1)
        self.assertEqual(prototype["direct_global_trial_cap_compliant_count"], 0)
        self.assertEqual(prototype["normal_guided_global_trial_cap_compliant_count"], 1)
        self.assertEqual(prototype["two_chart_rigid_trial_cap_compliant_count"], 1)
        self.assertEqual(prototype["two_chart_rigid_trial_selection_gate_pass_count"], 1)
        self.assertEqual(
            prototype["two_chart_rigid_trial_placement_class_counts"],
            {"rigid_cap_success": 1},
        )
        self.assertEqual(prototype["accepted_boundary_transition_count"], 1)
        self.assertTrue(prototype["reaches_target"])
        self.assertAlmostEqual(prototype["final_boundary_largest_component_fraction"], 1.0)
        self.assertAlmostEqual(prototype["final_boundary_local_quality_p90"], 0.190131)
        self.assertLessEqual(prototype["normal_guided_global_trial_p90_summary"]["p90"], 0.2)
        self.assertLessEqual(prototype["two_chart_rigid_trial_p90_summary"]["p90"], 0.2)
        blocker_summary = prototype["global_reconciliation_blocker_summary"]
        self.assertEqual(blocker_summary["candidate_count"], 1)
        self.assertEqual(blocker_summary["local_cap_global_blocked_count"], 1)
        self.assertEqual(blocker_summary["transition_normal_ok_count"], 1)
        self.assertEqual(blocker_summary["frame_degenerate_count"], 0)
        self.assertEqual(blocker_summary["normal_guided_resolved_blocker_count"], 1)
        self.assertEqual(blocker_summary["two_chart_rigid_resolved_blocker_count"], 1)
        self.assertAlmostEqual(
            blocker_summary["direct_global_p90_cap_gap_summary"]["p90"],
            0.034847,
        )
        row = prototype["top_by_boundary_quality"][0]
        self.assertTrue(row["boundary_transition_cap_compliant"])
        self.assertFalse(row["direct_global_trial_cap_compliant"])
        self.assertTrue(row["normal_guided_global_trial_cap_compliant"])
        self.assertTrue(row["two_chart_rigid_trial_cap_compliant"])
        self.assertTrue(row["two_chart_rigid_trial_selection_gate_pass"])
        self.assertEqual(row["two_chart_rigid_trial_placement_class"], "rigid_cap_success")
        self.assertAlmostEqual(row["boundary_transition_quality"], 0.190131)
        self.assertLessEqual(row["normal_guided_global_trial_p90_edge_distortion"], 0.2)
        self.assertLessEqual(row["two_chart_rigid_trial_p90_edge_distortion"], 0.2)
        self.assertLess(row["normal_guided_global_trial_p90_delta_vs_direct"], 0.0)
        self.assertNotIn("edge", row)
        self.assertNotIn("node_index", row)
        self.assertNotIn("root", row)
        step = prototype["accepted_steps"][0]
        self.assertAlmostEqual(step["coverage_delta"], 0.5)
        self.assertNotIn("edge", step)
        self.assertNotIn("node_index", step)
        self.assertNotIn("root", step)

    def test_base_cap_boundary_transition_can_require_two_chart_gate(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 4.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        left_edges = {(0, 1), (0, 2), (1, 2)}
        right_edges = {(3, 4), (3, 5), (4, 5)}
        bridge = (2, 3)
        edge_quality = [
            {"edge": edge, "normal_agreement": 1.0, "offset_ratio": 0.0}
            for edge in sorted(left_edges | right_edges)
        ]
        edge_quality.append({"edge": bridge, "normal_agreement": 0.844132, "offset_ratio": 0.190131})

        def fake_unwrap_metrics(_nodes, edges):
            edge_set = {tuple(edge) for edge in edges}
            if bridge in edge_set:
                p90 = 0.234847
            elif edge_set == left_edges:
                p90 = 0.01
            elif edge_set == right_edges:
                p90 = 0.02
            else:
                p90 = 0.0 if edge_set else None
            return {
                "patch_unwrap_proxy_p90_edge_distortion": p90,
                "patch_unwrap_proxy_largest_component_node_fraction": plotter.largest_fraction(
                    len(nodes),
                    edge_set,
                ),
                "patch_unwrap_proxy_p10_triangle_area_ratio": 0.8,
            }

        def reject_two_chart_gate(_rigid_trial, _max_p90_distortion):
            return {
                "two_chart_rigid_trial_orientation_gate_pass": True,
                "two_chart_rigid_trial_orientation_determinant_threshold": 0,
                "two_chart_rigid_trial_placement_class": "low_bridge_projection_failure",
                "two_chart_rigid_trial_projection_gate_pass": False,
                "two_chart_rigid_trial_projection_gate_threshold": 0.7,
                "two_chart_rigid_trial_selection_gate_pass": False,
            }

        with mock.patch.object(plotter.ablation, "unwrap_proxy_metrics", side_effect=fake_unwrap_metrics):
            with mock.patch.object(
                plotter,
                "two_chart_rigid_trial_gate_metrics",
                side_effect=reject_two_chart_gate,
            ):
                prototype = plotter.base_cap_boundary_transition_prototype(
                    nodes,
                    left_edges | right_edges,
                    [bridge],
                    edge_quality,
                    max_p90_distortion=0.2,
                    target=0.9,
                    diagnostic_limit=3,
                    require_two_chart_selection_gate=True,
                )

        self.assertTrue(prototype["requires_two_chart_rigid_trial_selection_gate"])
        self.assertEqual(
            prototype["method"],
            "greedy_local_chart_boundary_transition_two_chart_gate",
        )
        self.assertEqual(prototype["boundary_transition_candidate_count"], 1)
        self.assertEqual(prototype["cap_compliant_boundary_transition_candidate_count"], 1)
        self.assertEqual(prototype["two_chart_rigid_trial_selection_gate_pass_count"], 0)
        self.assertEqual(prototype["accepted_boundary_transition_count"], 0)
        self.assertFalse(prototype["reaches_target"])
        self.assertEqual(
            prototype["two_chart_rigid_trial_placement_class_counts"],
            {"low_bridge_projection_failure": 1},
        )

    def test_candidate_bridge_diagnostics_reports_chart_frame_scalars(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 4.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        base_edges = {(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)}
        all_edges = [*sorted(base_edges), (1, 3)]
        edge_quality = [
            {"edge": edge, "normal_agreement": 0.9, "offset_ratio": 0.1}
            for edge in all_edges
        ]

        diagnostics = plotter.candidate_bridge_diagnostics(
            nodes,
            all_edges,
            base_edges,
            edge_quality,
            max_p90_distortion=0.2,
            limit=4,
        )

        self.assertEqual(diagnostics["bridge_candidate_count"], 1)
        self.assertEqual(diagnostics["chart_pair_available_count"], 1)
        self.assertEqual(diagnostics["chart_normal_ge_050_count"], 1)
        row = diagnostics["top_by_coverage"][0]
        self.assertTrue(row["chart_pair_available"])
        self.assertAlmostEqual(row["chart_normal_agreement"], 1.0)
        self.assertIn("chart_origin_separation", row)
        self.assertNotIn("edge", row)

    def test_chart_transition_budget_reports_aggregate_component_coverage(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 4.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        base_edges = {(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)}
        all_edges = [*sorted(base_edges), (1, 3)]
        edge_quality = [
            {"edge": edge, "normal_agreement": 0.9, "offset_ratio": 0.1}
            for edge in all_edges
        ]

        budget = plotter.chart_transition_budget(
            nodes,
            all_edges,
            base_edges,
            edge_quality,
            max_p90_distortion=None,
            rotation_thresholds=(15.0,),
        )

        self.assertEqual(budget["bridge_candidate_count"], 1)
        self.assertEqual(budget["chart_pair_available_count"], 1)
        capped = budget["rows"][0]
        self.assertEqual(capped["eligible_component_pair_count"], 1)
        self.assertEqual(capped["largest_component_fraction"], 1.0)
        self.assertTrue(capped["reaches_90"])
        self.assertNotIn("left_root", capped)
        staged = budget["two_stage_rows"][0]
        self.assertEqual(staged["stage1_component_pair_count"], 1)
        self.assertEqual(staged["stage1_largest_component_fraction"], 1.0)
        self.assertEqual(staged["frame_missing_cleanup_edge_count"], 0)
        self.assertEqual(staged["after_all_p90_cleanup_fraction"], 1.0)
        self.assertNotIn("left_root", staged)

    def test_chart_transition_budget_reports_frame_missing_cleanup(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        base_edges = {(0, 1), (0, 2), (1, 2)}
        all_edges = [*sorted(base_edges), (1, 3)]
        edge_quality = [
            {"edge": edge, "normal_agreement": 0.9, "offset_ratio": 0.1}
            for edge in all_edges
        ]

        budget = plotter.chart_transition_budget(
            nodes,
            all_edges,
            base_edges,
            edge_quality,
            max_p90_distortion=None,
            rotation_thresholds=(15.0,),
        )

        staged = budget["two_stage_rows"][0]
        self.assertEqual(staged["stage1_component_pair_count"], 0)
        self.assertEqual(staged["stage1_largest_component_fraction"], 0.75)
        self.assertEqual(staged["frame_missing_cleanup_edge_count"], 1)
        self.assertEqual(staged["after_frame_missing_cleanup_fraction"], 1.0)
        self.assertTrue(staged["reaches_90_after_frame_missing_cleanup"])
        self.assertNotIn("left_root", staged)

    def test_two_stage_chart_growth_reports_stateful_frame_missing_cleanup(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        base_edges = {(0, 1), (0, 2), (1, 2)}
        all_edges = [*sorted(base_edges), (1, 3)]
        edge_quality = [
            {"edge": edge, "normal_agreement": 0.9, "offset_ratio": 0.1}
            for edge in all_edges
        ]

        grown_edges, added_edges, metrics, sequence = plotter.grow_two_stage_chart_edges_with_sequence(
            nodes,
            all_edges,
            base_edges,
            edge_quality,
            target=1.0,
            max_p90_distortion=None,
            rotation_threshold_degrees=90.0,
            cleanup_score="p90",
        )

        self.assertEqual(added_edges, {(1, 3)})
        self.assertEqual(plotter.largest_fraction(len(nodes), grown_edges), 1.0)
        self.assertEqual(metrics["local_chart_largest_component_fraction"], 1.0)
        self.assertTrue(sequence["reached"])
        self.assertEqual(sequence["cleanup_score"], "p90")
        self.assertEqual(sequence["stage1_edge_count"], 0)
        self.assertEqual(sequence["frame_missing_cleanup_edge_count"], 1)
        self.assertEqual(sequence["high_rotation_cleanup_edge_count"], 0)
        self.assertNotIn("edge", sequence["stage_summaries"][0])

    def test_two_stage_path_cleanup_allows_final_p90_compliant_path(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([1.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([2.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([9.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        base_edges: set[tuple[int, int]] = set()
        all_edges = [(0, 1), (1, 2)]
        edge_quality = [
            {"edge": edge, "normal_agreement": 0.9, "offset_ratio": 0.1}
            for edge in all_edges
        ]

        def fake_unwrap_metrics(_nodes, edges):
            edge_set = {plotter.sorted_edge(edge) for edge in edges}
            if (0, 1) in edge_set and (1, 2) not in edge_set:
                p90 = 0.3
            elif {(0, 1), (1, 2)}.issubset(edge_set):
                p90 = 0.1
            else:
                p90 = 0.0 if edge_set else None
            return {
                "patch_unwrap_proxy_largest_component_node_fraction": plotter.largest_fraction(
                    len(nodes), edge_set
                ),
                "patch_unwrap_proxy_p90_edge_distortion": p90,
                "patch_unwrap_proxy_p10_triangle_area_ratio": 1.0,
            }

        with mock.patch.object(plotter.ablation, "unwrap_proxy_metrics", side_effect=fake_unwrap_metrics):
            p90_edges, _p90_added, _p90_metrics, p90_sequence = (
                plotter.grow_two_stage_chart_edges_with_sequence(
                    nodes,
                    all_edges,
                    base_edges,
                    edge_quality,
                    target=0.75,
                    max_p90_distortion=0.2,
                    rotation_threshold_degrees=90.0,
                    cleanup_score="p90",
                )
            )
            path_edges, _path_added, _path_metrics, path_sequence = (
                plotter.grow_two_stage_chart_edges_with_sequence(
                    nodes,
                    all_edges,
                    base_edges,
                    edge_quality,
                    target=0.75,
                    max_p90_distortion=0.2,
                    rotation_threshold_degrees=90.0,
                    cleanup_score="path",
                    cleanup_lookahead=2,
                    cleanup_branching=8,
                )
            )

        self.assertEqual(plotter.largest_fraction(len(nodes), p90_edges), 0.25)
        self.assertFalse(p90_sequence["reached"])
        self.assertEqual(plotter.largest_fraction(len(nodes), path_edges), 0.75)
        self.assertTrue(path_sequence["reached"])
        self.assertEqual(path_sequence["cleanup_score"], "path")
        self.assertEqual(path_sequence["path_cleanup_step_count"], 1)
        self.assertEqual(path_sequence["path_cleanup_max_length"], 2)
        self.assertEqual(path_sequence["frame_missing_cleanup_edge_count"], 2)
        p90_frontier = p90_sequence["stall_frontier"]
        self.assertEqual(p90_frontier["remaining_candidate_count"], 2)
        self.assertEqual(p90_frontier["frontier_candidate_count"], 1)
        self.assertEqual(p90_frontier["disconnected_candidate_count"], 1)
        self.assertEqual(p90_frontier["frontier_cap_fail_count"], 1)
        self.assertEqual(p90_frontier["frontier_cap_pass_count"], 0)
        self.assertEqual(p90_frontier["best_frontier_relaxed_trial_fraction"], 0.5)
        self.assertEqual(p90_frontier["best_frontier_p90_blocked_trial_fraction"], 0.5)
        self.assertEqual(p90_frontier["best_frontier_p90_blocked_p90"], 0.3)
        self.assertIsNone(p90_frontier["min_target_reaching_frontier_p90"])
        self.assertIsNone(p90_frontier["min_target_reaching_frontier_trial_fraction"])
        path_frontier = path_sequence["stall_frontier"]
        self.assertEqual(path_frontier["remaining_candidate_count"], 0)
        self.assertEqual(path_frontier["frontier_candidate_count"], 0)
        path_search = path_sequence["path_search_summaries"][0]
        self.assertEqual(path_search["stage"], "frame_missing_cleanup")
        self.assertEqual(path_search["visited_path_count"], 2)
        self.assertEqual(path_search["improving_path_count"], 2)
        self.assertEqual(path_search["p90_blocked_improving_path_count"], 1)
        self.assertEqual(path_search["strict_cap_viable_path_count"], 1)
        self.assertEqual(path_search["selected_path_length"], 2)
        self.assertEqual(path_search["selected_path_summary"]["path_length"], 2)
        self.assertEqual(path_search["selected_path_summary"]["trial_fraction"], 0.75)
        self.assertEqual(path_search["selected_path_summary"]["target_gap"], 0.0)
        self.assertEqual(path_search["path_length_summary"]["max"], 2)
        self.assertEqual(path_search["improving_p90_summary"]["count"], 2)
        self.assertEqual(path_search["p90_blocked_p90_summary"]["count"], 1)
        self.assertEqual(path_search["p90_blocked_p90_summary"]["max"], 0.3)
        self.assertNotIn("edge", path_search)
        self.assertNotIn("path_signature", path_search)
        self.assertNotIn("edge", path_search["selected_path_summary"])
        self.assertNotIn("path_signature", path_search["selected_path_summary"])
        self.assertNotIn("edge", path_sequence["stage_summaries"][0])
        self.assertNotIn("edge", p90_frontier)

    def test_two_stage_path_quality_cleanup_prioritizes_local_quality(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([1.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([2.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        base_edges: set[tuple[int, int]] = set()
        all_edges = [(0, 1), (1, 2), (0, 3)]
        edge_quality = [
            {"edge": edge, "normal_agreement": 0.9, "offset_ratio": 0.1}
            for edge in all_edges
        ]

        def fake_unwrap_metrics(_nodes, edges):
            edge_set = {plotter.sorted_edge(edge) for edge in edges}
            has_chain = {(0, 1), (1, 2)}.issubset(edge_set)
            has_quality_bridge = (0, 3) in edge_set
            p90 = None
            if has_quality_bridge and any(edge in edge_set for edge in ((0, 1), (1, 2))):
                p90 = 0.9
            elif has_chain or has_quality_bridge or (0, 1) in edge_set:
                p90 = 0.1
            return {
                "patch_unwrap_proxy_largest_component_node_fraction": plotter.largest_fraction(
                    len(nodes), edge_set
                ),
                "patch_unwrap_proxy_p90_edge_distortion": p90,
                "patch_unwrap_proxy_p10_triangle_area_ratio": 1.0,
            }

        def fake_local_metrics(_nodes, grown_edges, _base_edges, _edge_quality):
            edge_set = {plotter.sorted_edge(edge) for edge in grown_edges}
            if (0, 3) in edge_set and not any(edge in edge_set for edge in ((0, 1), (1, 2))):
                quality = 0.2
            elif {(0, 1), (1, 2)}.issubset(edge_set):
                quality = 0.7
            elif (0, 1) in edge_set:
                quality = 0.8
            else:
                quality = None
            return {
                "local_chart_largest_component_fraction": plotter.rounded(
                    plotter.largest_fraction(len(nodes), edge_set)
                ),
                "local_chart_p90_internal_edge_distortion": None,
                "local_chart_bridge_edge_count": len(edge_set),
                "local_chart_p90_bridge_offset_ratio": None,
                "local_chart_p10_bridge_normal_agreement": None,
                "local_chart_quality_p90": quality,
            }

        with mock.patch.object(plotter.ablation, "unwrap_proxy_metrics", side_effect=fake_unwrap_metrics), mock.patch.object(
            plotter, "component_local_chart_metrics", side_effect=fake_local_metrics
        ):
            path_edges, _path_added, _path_metrics, path_sequence = (
                plotter.grow_two_stage_chart_edges_with_sequence(
                    nodes,
                    all_edges,
                    base_edges,
                    edge_quality,
                    target=0.75,
                    max_p90_distortion=0.2,
                    rotation_threshold_degrees=90.0,
                    cleanup_score="path",
                    cleanup_lookahead=2,
                    cleanup_branching=8,
                )
            )
            quality_edges, _quality_added, _quality_metrics, quality_sequence = (
                plotter.grow_two_stage_chart_edges_with_sequence(
                    nodes,
                    all_edges,
                    base_edges,
                    edge_quality,
                    target=0.75,
                    max_p90_distortion=0.2,
                    rotation_threshold_degrees=90.0,
                    cleanup_score="path_quality",
                    cleanup_lookahead=2,
                    cleanup_branching=8,
                )
            )

        self.assertEqual(path_edges, {(0, 1), (1, 2)})
        self.assertEqual(plotter.largest_fraction(len(nodes), path_edges), 0.75)
        self.assertTrue(path_sequence["reached"])
        self.assertEqual(path_sequence["cleanup_score"], "path")
        self.assertEqual(path_sequence["path_cleanup_max_length"], 2)

        self.assertEqual(quality_edges, {(0, 3)})
        self.assertEqual(plotter.largest_fraction(len(nodes), quality_edges), 0.5)
        self.assertFalse(quality_sequence["reached"])
        self.assertEqual(quality_sequence["cleanup_score"], "path_quality")
        self.assertEqual(quality_sequence["path_cleanup_max_length"], 1)
        self.assertEqual(quality_sequence["final_local_chart_quality_p90"], 0.2)

    def test_two_stage_path_target_quality_cleanup_prioritizes_target_p90(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([1.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([2.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        base_edges: set[tuple[int, int]] = set()
        all_edges = [(0, 1), (1, 2), (0, 3)]
        edge_quality = [
            {"edge": edge, "normal_agreement": 0.9, "offset_ratio": 0.1}
            for edge in all_edges
        ]

        def fake_unwrap_metrics(_nodes, edges):
            edge_set = {plotter.sorted_edge(edge) for edge in edges}
            has_chain = {(0, 1), (1, 2)}.issubset(edge_set)
            has_target_bridge = (0, 3) in edge_set
            p90 = None
            if has_target_bridge and any(edge in edge_set for edge in ((0, 1), (1, 2))):
                p90 = 0.9
            elif has_chain:
                p90 = 0.18
            elif has_target_bridge or (0, 1) in edge_set:
                p90 = 0.1
            return {
                "patch_unwrap_proxy_largest_component_node_fraction": plotter.largest_fraction(
                    len(nodes), edge_set
                ),
                "patch_unwrap_proxy_p90_edge_distortion": p90,
                "patch_unwrap_proxy_p10_triangle_area_ratio": 1.0,
            }

        def fake_local_metrics(_nodes, grown_edges, _base_edges, _edge_quality):
            edge_set = {plotter.sorted_edge(edge) for edge in grown_edges}
            if (0, 3) in edge_set and not any(edge in edge_set for edge in ((0, 1), (1, 2))):
                quality = 0.2
            elif {(0, 1), (1, 2)}.issubset(edge_set):
                quality = 0.7
            elif (0, 1) in edge_set:
                quality = 0.8
            else:
                quality = None
            return {
                "local_chart_largest_component_fraction": plotter.rounded(
                    plotter.largest_fraction(len(nodes), edge_set)
                ),
                "local_chart_p90_internal_edge_distortion": None,
                "local_chart_bridge_edge_count": len(edge_set),
                "local_chart_p90_bridge_offset_ratio": None,
                "local_chart_p10_bridge_normal_agreement": None,
                "local_chart_quality_p90": quality,
            }

        with mock.patch.object(plotter.ablation, "unwrap_proxy_metrics", side_effect=fake_unwrap_metrics), mock.patch.object(
            plotter, "component_local_chart_metrics", side_effect=fake_local_metrics
        ):
            path_edges, _path_added, _path_metrics, path_sequence = (
                plotter.grow_two_stage_chart_edges_with_sequence(
                    nodes,
                    all_edges,
                    base_edges,
                    edge_quality,
                    target=0.5,
                    max_p90_distortion=0.2,
                    rotation_threshold_degrees=90.0,
                    cleanup_score="path",
                    cleanup_lookahead=2,
                    cleanup_branching=8,
                )
            )
            target_edges, _target_added, _target_metrics, target_sequence = (
                plotter.grow_two_stage_chart_edges_with_sequence(
                    nodes,
                    all_edges,
                    base_edges,
                    edge_quality,
                    target=0.5,
                    max_p90_distortion=0.2,
                    rotation_threshold_degrees=90.0,
                    cleanup_score="path_target_quality",
                    cleanup_lookahead=2,
                    cleanup_branching=8,
                )
            )

        self.assertEqual(path_edges, {(0, 1), (1, 2)})
        self.assertEqual(plotter.largest_fraction(len(nodes), path_edges), 0.75)
        self.assertTrue(path_sequence["reached"])
        self.assertEqual(path_sequence["cleanup_score"], "path")

        self.assertEqual(target_edges, {(0, 3)})
        self.assertEqual(plotter.largest_fraction(len(nodes), target_edges), 0.5)
        self.assertTrue(target_sequence["reached"])
        self.assertEqual(target_sequence["cleanup_score"], "path_target_quality")
        self.assertEqual(target_sequence["final_global_p90_edge_distortion"], 0.1)
        self.assertEqual(target_sequence["final_local_chart_quality_p90"], 0.2)

    def test_two_stage_path_quality_normal_penalizes_weak_normals(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([1.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([2.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        base_edges: set[tuple[int, int]] = set()
        all_edges = [(0, 1), (1, 2), (0, 3)]
        edge_quality = [
            {"edge": (0, 1), "normal_agreement": 0.9, "offset_ratio": 0.4},
            {"edge": (1, 2), "normal_agreement": 0.9, "offset_ratio": 0.4},
            {"edge": (0, 3), "normal_agreement": 0.05, "offset_ratio": 0.05},
        ]

        def fake_unwrap_metrics(_nodes, edges):
            edge_set = {plotter.sorted_edge(edge) for edge in edges}
            has_chain_edge = any(edge in edge_set for edge in ((0, 1), (1, 2)))
            has_chain = {(0, 1), (1, 2)}.issubset(edge_set)
            has_weak_bridge = (0, 3) in edge_set
            if has_weak_bridge and has_chain_edge:
                p90 = 0.9
            elif has_chain:
                p90 = 0.18
            elif has_weak_bridge or has_chain_edge:
                p90 = 0.1
            else:
                p90 = None
            return {
                "patch_unwrap_proxy_largest_component_node_fraction": plotter.largest_fraction(
                    len(nodes), edge_set
                ),
                "patch_unwrap_proxy_p90_edge_distortion": p90,
                "patch_unwrap_proxy_p10_triangle_area_ratio": 1.0,
            }

        def fake_local_metrics(_nodes, grown_edges, _base_edges, _edge_quality):
            edge_set = {plotter.sorted_edge(edge) for edge in grown_edges}
            if (0, 3) in edge_set and not any(edge in edge_set for edge in ((0, 1), (1, 2))):
                quality = 0.05
            elif {(0, 1), (1, 2)}.issubset(edge_set):
                quality = 0.4
            elif (0, 1) in edge_set:
                quality = 0.6
            else:
                quality = None
            return {
                "local_chart_largest_component_fraction": plotter.rounded(
                    plotter.largest_fraction(len(nodes), edge_set)
                ),
                "local_chart_p90_internal_edge_distortion": None,
                "local_chart_bridge_edge_count": len(edge_set),
                "local_chart_p90_bridge_offset_ratio": None,
                "local_chart_p10_bridge_normal_agreement": None,
                "local_chart_quality_p90": quality,
            }

        with mock.patch.object(plotter.ablation, "unwrap_proxy_metrics", side_effect=fake_unwrap_metrics), mock.patch.object(
            plotter, "component_local_chart_metrics", side_effect=fake_local_metrics
        ):
            quality_edges, _quality_added, _quality_metrics, quality_sequence = (
                plotter.grow_two_stage_chart_edges_with_sequence(
                    nodes,
                    all_edges,
                    base_edges,
                    edge_quality,
                    target=0.75,
                    max_p90_distortion=0.2,
                    rotation_threshold_degrees=90.0,
                    cleanup_score="path_quality",
                    cleanup_lookahead=2,
                    cleanup_branching=8,
                )
            )
            normal_edges, _normal_added, _normal_metrics, normal_sequence = (
                plotter.grow_two_stage_chart_edges_with_sequence(
                    nodes,
                    all_edges,
                    base_edges,
                    edge_quality,
                    target=0.75,
                    max_p90_distortion=0.2,
                    rotation_threshold_degrees=90.0,
                    cleanup_score="path_quality_normal",
                    cleanup_lookahead=2,
                    cleanup_branching=8,
                    cleanup_normal_penalty=2.0,
                )
            )
            minimax_edges, _minimax_added, _minimax_metrics, minimax_sequence = (
                plotter.grow_two_stage_chart_edges_with_sequence(
                    nodes,
                    all_edges,
                    base_edges,
                    edge_quality,
                    target=0.75,
                    max_p90_distortion=0.2,
                    rotation_threshold_degrees=90.0,
                    cleanup_score="path_quality_minimax",
                    cleanup_lookahead=2,
                    cleanup_branching=8,
                )
            )

        self.assertEqual(quality_edges, {(0, 3)})
        self.assertFalse(quality_sequence["reached"])
        self.assertEqual(quality_sequence["cleanup_score"], "path_quality")

        self.assertEqual(normal_edges, {(0, 1), (1, 2)})
        self.assertTrue(normal_sequence["reached"])
        self.assertEqual(normal_sequence["cleanup_score"], "path_quality_normal")
        self.assertEqual(normal_sequence["cleanup_normal_penalty"], 2.0)
        self.assertEqual(normal_sequence["path_cleanup_max_length"], 2)

        self.assertEqual(minimax_edges, {(0, 1), (1, 2)})
        self.assertTrue(minimax_sequence["reached"])
        self.assertEqual(minimax_sequence["cleanup_score"], "path_quality_minimax")
        self.assertEqual(minimax_sequence["path_cleanup_max_length"], 2)

    def test_local_atlas_growth_uses_bridge_quality_thresholds(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([3.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        all_edges = [(0, 1), (0, 2), (1, 2), (0, 3)]
        base_edges = {(0, 1), (0, 2), (1, 2)}
        edge_quality = [
            {"edge": (0, 1), "normal_agreement": 0.9, "offset_ratio": 0.1},
            {"edge": (0, 2), "normal_agreement": 0.9, "offset_ratio": 0.1},
            {"edge": (1, 2), "normal_agreement": 0.9, "offset_ratio": 0.1},
            {"edge": (0, 3), "normal_agreement": 0.8, "offset_ratio": 0.05},
        ]

        grown_edges, added_edges, metrics = plotter.grow_local_atlas_edges(
            nodes,
            all_edges,
            base_edges,
            edge_quality,
            target=1.0,
            max_bridge_offset_ratio=0.2,
            min_bridge_normal_agreement=0.5,
        )

        self.assertEqual(added_edges, {(0, 3)})
        self.assertEqual(plotter.largest_fraction(len(nodes), grown_edges), 1.0)
        self.assertEqual(metrics["local_chart_bridge_edge_count"], 1)
        self.assertEqual(metrics["local_chart_p90_bridge_offset_ratio"], 0.05)

    def test_local_atlas_broad_candidate_policy_can_seed_high_offset_bridge(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([3.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        all_edges = [(0, 1), (0, 2), (1, 2), (0, 3)]
        base_edges = {(0, 1), (0, 2), (1, 2)}
        edge_quality = [
            {"edge": (0, 1), "normal_agreement": 0.95, "offset_ratio": 0.1},
            {"edge": (0, 2), "normal_agreement": 0.95, "offset_ratio": 0.1},
            {"edge": (1, 2), "normal_agreement": 0.95, "offset_ratio": 0.1},
            {"edge": (0, 3), "normal_agreement": 0.95, "offset_ratio": 0.5},
        ]

        def high_projection_metrics(_nodes, _edge, _base_roots, _chart_frames):
            return {
                "bridge_normal_offset_ratio_left": 0.1,
                "bridge_normal_offset_ratio_right": 0.1,
                "bridge_projection_ratio_left": 0.9,
                "bridge_projection_ratio_right": 0.95,
                "chart_normal_agreement": 0.9,
                "chart_origin_separation": 1.0,
                "chart_pair_available": True,
                "chart_tangent_singular_max": 1.0,
                "chart_tangent_singular_min": 0.9,
                "left_chart_component_fraction": 0.75,
                "right_chart_component_fraction": 0.25,
            }

        _grown, added_without_policy, _metrics, sequence_without_policy = (
            plotter.grow_local_atlas_edges_with_sequence(
                nodes,
                all_edges,
                base_edges,
                edge_quality,
                target=1.0,
                max_bridge_offset_ratio=0.2,
                min_bridge_normal_agreement=0.5,
            )
        )
        self.assertFalse(added_without_policy)
        self.assertFalse(sequence_without_policy["reached"])

        with mock.patch.object(
            plotter,
            "bridge_chart_frame_metrics",
            side_effect=high_projection_metrics,
        ):
            grown, added, _metrics, sequence = plotter.grow_local_atlas_edges_with_sequence(
                nodes,
                all_edges,
                base_edges,
                edge_quality,
                target=1.0,
                max_bridge_offset_ratio=0.2,
                min_bridge_normal_agreement=0.5,
                use_broad_candidate_policy=True,
            )

        self.assertEqual(added, {(0, 3)})
        self.assertEqual(plotter.largest_fraction(len(nodes), grown), 1.0)
        self.assertTrue(sequence["reached"])
        self.assertTrue(sequence["broad_candidate_policy_enabled"])
        self.assertEqual(sequence["broad_candidate_policy_step_count"], 1)
        step = sequence["steps"][0]
        self.assertEqual(step["selection_source"], "broad_candidate_policy")
        self.assertTrue(step["touches_base_largest_component"])
        self.assertTrue(step["broad_candidate_policy_pass"])
        self.assertAlmostEqual(step["min_bridge_projection_ratio"], 0.9)
        self.assertNotIn("edge", step)

    def test_local_atlas_broad_candidate_policy_requires_base_largest_touch(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {
                "centroid": np.array([float(index), 0.0, 0.0]),
                "normal": np.array([1.0, 0.0, 0.0]),
            }
            for index in range(7)
        ]
        all_edges = [(0, 1), (1, 2), (3, 4), (5, 6), (3, 5)]
        base_edges = {(0, 1), (1, 2), (3, 4), (5, 6)}
        edge_quality = [
            {"edge": (0, 1), "normal_agreement": 0.95, "offset_ratio": 0.1},
            {"edge": (1, 2), "normal_agreement": 0.95, "offset_ratio": 0.1},
            {"edge": (3, 4), "normal_agreement": 0.95, "offset_ratio": 0.1},
            {"edge": (5, 6), "normal_agreement": 0.95, "offset_ratio": 0.1},
            {"edge": (3, 5), "normal_agreement": 0.95, "offset_ratio": 0.5},
        ]

        def high_projection_metrics(_nodes, _edge, _base_roots, _chart_frames):
            return {
                "chart_pair_available": True,
                "bridge_projection_ratio_left": 0.9,
                "bridge_projection_ratio_right": 0.9,
            }

        with mock.patch.object(
            plotter,
            "bridge_chart_frame_metrics",
            side_effect=high_projection_metrics,
        ):
            _grown, added, _metrics, sequence = plotter.grow_local_atlas_edges_with_sequence(
                nodes,
                all_edges,
                base_edges,
                edge_quality,
                target=0.5,
                max_bridge_offset_ratio=0.2,
                min_bridge_normal_agreement=0.5,
                bridge_only=False,
                use_broad_candidate_policy=True,
            )

        self.assertFalse(added)
        self.assertFalse(sequence["reached"])
        self.assertEqual(sequence["broad_candidate_policy_step_count"], 0)

    def test_local_atlas_two_chart_boundary_gate_can_seed_bridge(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 4.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        left_edges = {(0, 1), (0, 2), (1, 2)}
        right_edges = {(3, 4), (3, 5), (4, 5)}
        bridge = (2, 3)
        all_edges = [*sorted(left_edges | right_edges), bridge]
        base_edges = left_edges | right_edges
        edge_quality = [
            {"edge": edge, "normal_agreement": 1.0, "offset_ratio": 0.0}
            for edge in sorted(base_edges)
        ]
        edge_quality.append({"edge": bridge, "normal_agreement": 0.844132, "offset_ratio": 0.190131})

        def fake_unwrap_metrics(_nodes, edges):
            edge_set = {tuple(edge) for edge in edges}
            return {
                "patch_unwrap_proxy_p90_edge_distortion": 0.234847 if bridge in edge_set else 0.02,
                "patch_unwrap_proxy_largest_component_node_fraction": plotter.largest_fraction(
                    len(nodes),
                    edge_set,
                ),
                "patch_unwrap_proxy_p10_triangle_area_ratio": 0.8,
            }

        with mock.patch.object(plotter.ablation, "unwrap_proxy_metrics", side_effect=fake_unwrap_metrics):
            _grown_without_gate, added_without_gate, _metrics_without_gate, sequence_without_gate = (
                plotter.grow_local_atlas_edges_with_sequence(
                    nodes,
                    all_edges,
                    base_edges,
                    edge_quality,
                    target=1.0,
                    max_bridge_offset_ratio=0.2,
                    min_bridge_normal_agreement=0.95,
                    max_p90_distortion=0.2,
                )
            )
            grown, added, _metrics, sequence = plotter.grow_local_atlas_edges_with_sequence(
                nodes,
                all_edges,
                base_edges,
                edge_quality,
                target=1.0,
                max_bridge_offset_ratio=0.2,
                min_bridge_normal_agreement=0.95,
                max_p90_distortion=0.2,
                use_two_chart_boundary_gate=True,
                two_chart_boundary_gate_only=True,
            )

        self.assertFalse(added_without_gate)
        self.assertFalse(sequence_without_gate["reached"])
        self.assertEqual(added, {bridge})
        self.assertEqual(plotter.largest_fraction(len(nodes), grown), 1.0)
        self.assertTrue(sequence["reached"])
        self.assertTrue(sequence["two_chart_boundary_gate_enabled"])
        self.assertTrue(sequence["two_chart_boundary_gate_only"])
        self.assertEqual(sequence["two_chart_boundary_gate_step_count"], 1)
        step = sequence["steps"][0]
        self.assertEqual(step["selection_source"], "two_chart_boundary_gate")
        self.assertTrue(step["two_chart_boundary_gate_pass"])
        self.assertTrue(step["boundary_transition_cap_compliant"])
        self.assertTrue(step["two_chart_rigid_trial_selection_gate_pass"])
        self.assertEqual(step["two_chart_rigid_trial_placement_class"], "rigid_cap_success")
        self.assertAlmostEqual(step["boundary_transition_quality"], 0.190131)
        self.assertLessEqual(step["two_chart_rigid_trial_p90_edge_distortion"], 0.2)
        self.assertTrue(step["two_chart_rigid_trial_available"])
        self.assertIsNotNone(step["two_chart_rigid_trial_all_edge_p90_distortion"])
        self.assertNotIn("edge", step)
        atlas = sequence["two_chart_atlas_reconciliation"]
        self.assertTrue(atlas["enabled"])
        self.assertEqual(
            atlas["decision"],
            "two_chart_atlas_resolves_direct_global_blocker",
        )
        self.assertEqual(atlas["scope"], "first_two_chart_boundary_gate_step")
        self.assertEqual(atlas["two_chart_boundary_gate_step_count"], 1)
        atlas_step = atlas["first_boundary_step"]
        self.assertAlmostEqual(atlas_step["direct_global_p90_edge_distortion_after"], 0.234847)
        self.assertFalse(atlas_step["direct_global_cap_compliant_after"])
        self.assertTrue(atlas_step["two_chart_cap_compliant"])
        self.assertNotIn("edge", atlas_step)
        multi = sequence["multi_chart_atlas_reconciliation"]
        self.assertTrue(multi["enabled"])
        self.assertEqual(
            multi["decision"],
            "multi_chart_atlas_resolves_direct_global_blocker",
        )
        self.assertEqual(multi["scope"], "final_local_atlas_result")
        self.assertEqual(multi["multi_chart_atlas_bridge_edge_count"], 1)
        self.assertFalse(multi["direct_global_cap_compliant"])
        self.assertTrue(multi["multi_chart_cap_compliant"])
        self.assertNotIn("edge", multi)

    def test_local_atlas_two_chart_boundary_gate_rejects_failed_gate(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 4.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        left_edges = {(0, 1), (0, 2), (1, 2)}
        right_edges = {(3, 4), (3, 5), (4, 5)}
        bridge = (2, 3)
        all_edges = [*sorted(left_edges | right_edges), bridge]
        base_edges = left_edges | right_edges
        edge_quality = [
            {"edge": edge, "normal_agreement": 1.0, "offset_ratio": 0.0}
            for edge in sorted(base_edges)
        ]
        edge_quality.append({"edge": bridge, "normal_agreement": 0.844132, "offset_ratio": 0.190131})

        def reject_two_chart_gate(_rigid_trial, _max_p90_distortion):
            return {
                "two_chart_rigid_trial_orientation_gate_pass": True,
                "two_chart_rigid_trial_orientation_determinant_threshold": 0,
                "two_chart_rigid_trial_placement_class": "low_bridge_projection_failure",
                "two_chart_rigid_trial_projection_gate_pass": False,
                "two_chart_rigid_trial_projection_gate_threshold": 0.7,
                "two_chart_rigid_trial_selection_gate_pass": False,
            }

        with mock.patch.object(
            plotter,
            "two_chart_rigid_trial_gate_metrics",
            side_effect=reject_two_chart_gate,
        ):
            _grown, added, _metrics, sequence = plotter.grow_local_atlas_edges_with_sequence(
                nodes,
                all_edges,
                base_edges,
                edge_quality,
                target=1.0,
                max_bridge_offset_ratio=0.2,
                min_bridge_normal_agreement=0.95,
                max_p90_distortion=0.2,
                use_two_chart_boundary_gate=True,
            )

        self.assertFalse(added)
        self.assertFalse(sequence["reached"])
        self.assertTrue(sequence["two_chart_boundary_gate_enabled"])
        self.assertEqual(sequence["two_chart_boundary_gate_step_count"], 0)
        self.assertEqual(sequence["stop_reason"], "no_eligible_bridge")

    def test_local_atlas_two_chart_boundary_gate_only_blocks_local_fallback(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 4.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        left_edges = {(0, 1), (0, 2), (1, 2)}
        right_edges = {(3, 4), (3, 5), (4, 5)}
        bridge = (2, 3)
        all_edges = [*sorted(left_edges | right_edges), bridge]
        base_edges = left_edges | right_edges
        edge_quality = [
            {"edge": edge, "normal_agreement": 1.0, "offset_ratio": 0.0}
            for edge in sorted(base_edges)
        ]
        edge_quality.append({"edge": bridge, "normal_agreement": 0.844132, "offset_ratio": 0.190131})

        def reject_two_chart_gate(_rigid_trial, _max_p90_distortion):
            return {
                "two_chart_rigid_trial_orientation_gate_pass": True,
                "two_chart_rigid_trial_orientation_determinant_threshold": 0,
                "two_chart_rigid_trial_placement_class": "low_bridge_projection_failure",
                "two_chart_rigid_trial_projection_gate_pass": False,
                "two_chart_rigid_trial_projection_gate_threshold": 0.7,
                "two_chart_rigid_trial_selection_gate_pass": False,
            }

        with mock.patch.object(
            plotter,
            "two_chart_rigid_trial_gate_metrics",
            side_effect=reject_two_chart_gate,
        ):
            _fallback_grown, fallback_added, _fallback_metrics, fallback_sequence = (
                plotter.grow_local_atlas_edges_with_sequence(
                    nodes,
                    all_edges,
                    base_edges,
                    edge_quality,
                    target=1.0,
                    max_bridge_offset_ratio=0.2,
                    min_bridge_normal_agreement=0.5,
                    max_p90_distortion=0.2,
                    use_two_chart_boundary_gate=True,
                )
            )
            _only_grown, only_added, _only_metrics, only_sequence = (
                plotter.grow_local_atlas_edges_with_sequence(
                    nodes,
                    all_edges,
                    base_edges,
                    edge_quality,
                    target=1.0,
                    max_bridge_offset_ratio=0.2,
                    min_bridge_normal_agreement=0.5,
                    max_p90_distortion=0.2,
                    use_two_chart_boundary_gate=True,
                    two_chart_boundary_gate_only=True,
                )
            )

        self.assertEqual(fallback_added, {bridge})
        self.assertEqual(
            fallback_sequence["steps"][0]["selection_source"],
            "local_atlas_bridge_gate",
        )
        self.assertEqual(fallback_sequence["two_chart_boundary_gate_step_count"], 0)
        self.assertFalse(only_added)
        self.assertFalse(only_sequence["reached"])
        self.assertTrue(only_sequence["two_chart_boundary_gate_only"])
        self.assertEqual(only_sequence["two_chart_boundary_gate_step_count"], 0)
        self.assertEqual(only_sequence["stop_reason"], "no_eligible_bridge")

    def test_local_atlas_sequence_reports_scalar_steps_without_edges(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([3.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([6.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        all_edges = [(0, 1), (0, 2), (1, 2), (0, 3), (3, 4)]
        base_edges = {(0, 1), (0, 2), (1, 2)}
        edge_quality = [
            {"edge": (0, 1), "normal_agreement": 0.9, "offset_ratio": 0.1},
            {"edge": (0, 2), "normal_agreement": 0.9, "offset_ratio": 0.1},
            {"edge": (1, 2), "normal_agreement": 0.9, "offset_ratio": 0.1},
            {"edge": (0, 3), "normal_agreement": 0.8, "offset_ratio": 0.05},
            {"edge": (3, 4), "normal_agreement": 0.7, "offset_ratio": 0.06},
        ]

        _grown_edges, _added_edges, _metrics, sequence = plotter.grow_local_atlas_edges_with_sequence(
            nodes,
            all_edges,
            base_edges,
            edge_quality,
            target=1.0,
            max_bridge_offset_ratio=0.2,
            min_bridge_normal_agreement=0.5,
        )

        self.assertTrue(sequence["reached"])
        self.assertEqual(sequence["step_count"], 2)
        self.assertEqual(sequence["stop_reason"], "target_reached")
        self.assertEqual(sequence["steps"][0]["step"], 1)
        self.assertGreater(sequence["steps"][0]["coverage_delta"], 0.0)
        self.assertTrue(all("edge" not in step for step in sequence["steps"]))

    def test_local_atlas_sequence_reports_chart_frame_scalars(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 4.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        all_edges = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (1, 3)]
        base_edges = {(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)}
        edge_quality = [
            {"edge": edge, "normal_agreement": 0.9, "offset_ratio": 0.1}
            for edge in base_edges
        ]
        edge_quality.append({"edge": (1, 3), "normal_agreement": 0.8, "offset_ratio": 0.05})

        _grown_edges, _added_edges, _metrics, sequence = plotter.grow_local_atlas_edges_with_sequence(
            nodes,
            all_edges,
            base_edges,
            edge_quality,
            target=1.0,
            max_bridge_offset_ratio=0.2,
            min_bridge_normal_agreement=0.5,
        )

        step = sequence["steps"][0]
        self.assertTrue(step["chart_pair_available"])
        self.assertAlmostEqual(step["chart_normal_agreement"], 1.0)
        self.assertGreaterEqual(step["chart_tangent_singular_min"], 0.0)
        self.assertLessEqual(step["chart_tangent_singular_max"], 1.0)
        self.assertIsNotNone(step["chart_origin_separation"])
        self.assertAlmostEqual(step["left_chart_component_fraction"], 0.5)
        self.assertAlmostEqual(step["right_chart_component_fraction"], 0.5)

    def test_local_atlas_sequence_reports_first_transition_reconciliation(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([3.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        all_edges = [(0, 1), (0, 2), (1, 2), (0, 3)]
        base_edges = {(0, 1), (0, 2), (1, 2)}
        edge_quality = [
            {"edge": (0, 1), "normal_agreement": 0.9, "offset_ratio": 0.1},
            {"edge": (0, 2), "normal_agreement": 0.9, "offset_ratio": 0.1},
            {"edge": (1, 2), "normal_agreement": 0.9, "offset_ratio": 0.1},
            {"edge": (0, 3), "normal_agreement": 0.8, "offset_ratio": 0.05},
        ]

        def fake_unwrap_metrics(_nodes, edges):
            edge_set = {tuple(edge) for edge in edges}
            p90 = 0.3 if (0, 3) in edge_set else 0.0 if edge_set else None
            return {
                "patch_unwrap_proxy_p90_edge_distortion": p90,
                "patch_unwrap_proxy_largest_component_node_fraction": plotter.largest_fraction(
                    len(nodes),
                    edge_set,
                ),
                "patch_unwrap_proxy_p10_triangle_area_ratio": 0.8,
            }

        with mock.patch.object(plotter.ablation, "unwrap_proxy_metrics", side_effect=fake_unwrap_metrics):
            _grown_edges, _added_edges, _metrics, sequence = plotter.grow_local_atlas_edges_with_sequence(
                nodes,
                all_edges,
                base_edges,
                edge_quality,
                target=1.0,
                max_bridge_offset_ratio=0.2,
                min_bridge_normal_agreement=0.5,
                max_p90_distortion=0.2,
            )

        step = sequence["steps"][0]
        self.assertAlmostEqual(step["transition_normal_risk"], 0.2)
        self.assertAlmostEqual(step["boundary_local_quality_p90_after"], 0.2)
        self.assertAlmostEqual(step["global_p90_cap_gap_after"], 0.1)
        self.assertTrue(step["local_quality_cap_compliant_after"])
        self.assertTrue(step["boundary_quality_cap_compliant_after"])
        self.assertFalse(step["global_p90_cap_compliant_after"])
        self.assertTrue(step["global_blocked_despite_local_cap_after"])
        self.assertTrue(step["global_blocked_despite_boundary_cap_after"])
        self.assertNotIn("edge", step)
        reconciliation = sequence["first_transition_reconciliation"]
        self.assertTrue(reconciliation["enabled"])
        self.assertEqual(
            reconciliation["decision"],
            "first_transition_boundary_chart_resolves_local_but_not_global",
        )
        self.assertNotIn("edge", reconciliation["first_step"])

    def test_local_atlas_normal_risk_quality_rejects_weak_normal_bridge(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        nodes = [
            {"centroid": np.array([0.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 1.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 0.0, 1.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([3.0, 0.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
            {"centroid": np.array([0.0, 3.0, 0.0]), "normal": np.array([1.0, 0.0, 0.0])},
        ]
        all_edges = [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4)]
        base_edges = {(0, 1), (0, 2), (1, 2)}
        edge_quality = [
            {"edge": (0, 1), "normal_agreement": 0.9, "offset_ratio": 0.1},
            {"edge": (0, 2), "normal_agreement": 0.9, "offset_ratio": 0.1},
            {"edge": (1, 2), "normal_agreement": 0.9, "offset_ratio": 0.1},
            {"edge": (0, 3), "normal_agreement": 0.6, "offset_ratio": 0.01},
            {"edge": (0, 4), "normal_agreement": 0.85, "offset_ratio": 0.18},
        ]

        _grown_edges, _added_edges, _metrics, sequence = plotter.grow_local_atlas_edges_with_sequence(
            nodes,
            all_edges,
            base_edges,
            edge_quality,
            target=0.8,
            max_bridge_offset_ratio=0.2,
            min_bridge_normal_agreement=0.5,
            use_transition_normal_risk_quality=True,
        )

        self.assertTrue(sequence["selection_uses_transition_normal_risk"])
        step = sequence["steps"][0]
        self.assertTrue(step["selection_uses_transition_normal_risk"])
        self.assertAlmostEqual(step["normal_agreement"], 0.85)
        self.assertAlmostEqual(step["transition_normal_risk"], 0.15)
        self.assertAlmostEqual(step["offset_ratio"], 0.18)
        self.assertAlmostEqual(step["selection_quality"], 0.18)
        self.assertNotIn("edge", step)

    def test_patch_graph_neighbor_radius_adds_wider_cell_edges(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is only required for experiment metrics")

        points = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 1, 1],
                [0, 8, 0],
                [0, 9, 0],
                [0, 8, 1],
                [0, 9, 1],
            ],
            dtype=np.float64,
        )
        normals = np.array([[1, 0, 0]] * 8, dtype=np.float64)

        _nodes1, edges1, _quality1 = plotter.build_patch_graph(
            points, normals, cell_size=4.0, min_points_per_cell=4, neighbor_radius=1
        )
        _nodes2, edges2, _quality2 = plotter.build_patch_graph(
            points, normals, cell_size=4.0, min_points_per_cell=4, neighbor_radius=2
        )

        self.assertEqual(len(edges1), 0)
        self.assertEqual(len(edges2), 1)

    def test_png_dimensions_rejects_non_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "not.png"
            path.write_bytes(b"not a png")
            with self.assertRaises(ValueError):
                plotter.png_dimensions(path)


if __name__ == "__main__":
    unittest.main()
