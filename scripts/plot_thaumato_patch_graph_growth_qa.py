#!/usr/bin/env python3
"""Render a raster-only QA view of one Thaumato patch-graph growth case."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import benchmark_thaumato_blur as base
import run_thaumato_surface_detection_ablation as ablation


TWO_CHART_MIN_BRIDGE_PROJECTION_RATIO = 0.7
TWO_CHART_MIN_ORIENTATION_DETERMINANT = 0.0
BROAD_CANDIDATE_MIN_NORMAL_AGREEMENT = 0.9
BROAD_CANDIDATE_MIN_BRIDGE_PROJECTION_RATIO = 0.75


def import_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - exercised by CLI environment checks.
        raise SystemExit(
            "Missing dependency: pillow. Install experiment dependencies with "
            "`python3 -m pip install -r requirements-experiments.txt`."
        ) from exc
    return Image, ImageDraw, ImageFont


def build_patch_graph(
    points,
    normals,
    cell_size: float,
    min_points_per_cell: int,
    neighbor_radius: int = 1,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]], list[dict[str, float | tuple[int, int]]]]:
    import numpy as np

    if neighbor_radius < 1:
        raise ValueError("neighbor_radius must be positive")

    cell_indices = np.floor(points / cell_size).astype(np.int64)
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, cell in enumerate(cell_indices):
        cells[(int(cell[0]), int(cell[1]), int(cell[2]))].append(index)

    kept_cells = sorted(
        (cell, indices) for cell, indices in cells.items() if len(indices) >= min_points_per_cell
    )
    nodes: list[dict[str, Any]] = []
    cell_to_node: dict[tuple[int, int, int], int] = {}
    for node_index, (cell, indices) in enumerate(kept_cells):
        cell_points = points[indices].astype(np.float64, copy=False)
        centered = cell_points - cell_points.mean(axis=0)
        covariance = centered.T @ centered / max(1, cell_points.shape[0])
        _eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        plane_normal = eigenvectors[:, 0]
        if normals is not None:
            cell_normals = normals[indices].astype(np.float64, copy=False)
            lengths = np.linalg.norm(cell_normals, axis=1)
            valid = lengths > 1e-6
            if np.any(valid):
                mean_normal = np.mean(cell_normals[valid] / lengths[valid, None], axis=0)
                if np.dot(plane_normal, mean_normal) < 0:
                    plane_normal = -plane_normal
        nodes.append({"cell": cell, "centroid": cell_points.mean(axis=0), "normal": plane_normal})
        cell_to_node[cell] = node_index

    neighbor_offsets = [
        (dz, dy, dx)
        for dz in range(-neighbor_radius, neighbor_radius + 1)
        for dy in range(-neighbor_radius, neighbor_radius + 1)
        for dx in range(-neighbor_radius, neighbor_radius + 1)
        if (dz, dy, dx) != (0, 0, 0)
    ]
    edges: list[tuple[int, int]] = []
    edge_quality: list[dict[str, float | tuple[int, int]]] = []
    for node_index, node in enumerate(nodes):
        z, y, x = node["cell"]
        for dz, dy, dx in neighbor_offsets:
            other_index = cell_to_node.get((z + dz, y + dy, x + dx))
            if other_index is None or other_index <= node_index:
                continue
            other = nodes[other_index]
            delta = other["centroid"] - node["centroid"]
            gap = float(np.linalg.norm(delta))
            if gap <= 0:
                continue
            normal_agreement = float(abs(np.dot(node["normal"], other["normal"])))
            plane_offset = float(
                0.5
                * (
                    abs(np.dot(node["normal"], delta))
                    + abs(np.dot(other["normal"], -delta))
                )
            )
            edge = (node_index, other_index)
            edges.append(edge)
            edge_quality.append(
                {
                    "edge": edge,
                    "normal_agreement": normal_agreement,
                    "offset_ratio": plane_offset / gap,
                }
            )
    return nodes, edges, edge_quality


def sorted_edge(edge: tuple[int, int]) -> tuple[int, int]:
    return tuple(sorted(edge))


def component_roots(node_count: int, edges: set[tuple[int, int]]) -> tuple[list[int], dict[int, int]]:
    parent = list(range(node_count))
    component_size = [1] * node_count

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if component_size[left_root] < component_size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        component_size[left_root] += component_size[right_root]

    for left, right in edges:
        union(left, right)
    roots = [find(index) for index in range(node_count)]
    sizes: dict[int, int] = {}
    for root in roots:
        sizes[root] = sizes.get(root, 0) + 1
    return roots, sizes


def largest_fraction(node_count: int, edges: set[tuple[int, int]]) -> float:
    if node_count == 0:
        return 0.0
    _roots, sizes = component_roots(node_count, edges)
    return max(sizes.values()) / node_count if sizes else 0.0


def pruned_edges(
    edge_quality: list[dict[str, float | tuple[int, int]]],
    min_normal_agreement: float,
    max_offset_ratio: float,
) -> list[tuple[int, int]]:
    return [
        sorted_edge(quality["edge"])
        for quality in edge_quality
        if isinstance(quality.get("edge"), tuple)
        and quality["normal_agreement"] >= min_normal_agreement
        and quality["offset_ratio"] <= max_offset_ratio
    ]


def cap_base_edges_by_incremental_p90(
    nodes: list[dict[str, Any]],
    base_edges: set[tuple[int, int]],
    edge_quality: list[dict[str, float | tuple[int, int]]],
    max_p90_distortion: float,
    diagnostic_limit: int = 24,
    all_edges: list[tuple[int, int]] | None = None,
    alternate_route_max_length: int = 0,
    alternate_route_branching: int = 64,
    component_probe: bool = False,
    boundary_prototype: bool = False,
    boundary_require_two_chart_gate: bool = False,
    boundary_target: float | None = None,
) -> tuple[set[tuple[int, int]], dict[str, Any]]:
    """Rebuild base edges greedily while enforcing an incremental p90 cap."""

    if diagnostic_limit < 1:
        raise ValueError("diagnostic_limit must be positive")
    if alternate_route_max_length < 0:
        raise ValueError("alternate_route_max_length must be non-negative")
    if alternate_route_branching < 1:
        raise ValueError("alternate_route_branching must be positive")

    quality_by_edge = {
        sorted_edge(quality["edge"]): quality
        for quality in edge_quality
        if isinstance(quality.get("edge"), tuple)
    }

    input_chart = ablation.unwrap_proxy_metrics(nodes, sorted(base_edges))
    accepted_edges: set[tuple[int, int]] = set()
    skipped_p90_values: list[float] = []
    skipped_trial_fractions: list[float] = []
    skipped_cap_gaps: list[float] = []
    skipped_larger_component_fractions: list[float] = []
    skipped_smaller_component_fractions: list[float] = []
    skipped_edges: list[tuple[int, int]] = []
    skipped_rows: list[dict[str, Any]] = []
    skipped_bridge_count = 0
    skipped_cycle_count = 0
    missing_quality_count = 0

    def edge_key(edge: tuple[int, int]) -> tuple[float, float, tuple[int, int]]:
        quality = quality_by_edge.get(edge)
        if quality is None:
            return (float("inf"), float("inf"), edge)
        return (
            -float(quality["normal_agreement"]),
            float(quality["offset_ratio"]),
            edge,
        )

    for edge in sorted(base_edges, key=edge_key):
        quality = quality_by_edge.get(edge)
        if quality is None:
            missing_quality_count += 1
        trial_edges = set(accepted_edges)
        trial_edges.add(edge)
        trial_chart = ablation.unwrap_proxy_metrics(nodes, sorted(trial_edges))
        p90 = trial_chart.get("patch_unwrap_proxy_p90_edge_distortion")
        if p90 is None or (isinstance(p90, (int, float)) and float(p90) <= max_p90_distortion):
            accepted_edges.add(edge)
        elif isinstance(p90, (int, float)):
            roots, sizes = component_roots(len(nodes), accepted_edges)
            left, right = edge
            left_root = roots[left]
            right_root = roots[right]
            is_bridge = left_root != right_root
            if is_bridge:
                skipped_bridge_count += 1
                component_sizes = sorted((sizes[left_root], sizes[right_root]), reverse=True)
            else:
                skipped_cycle_count += 1
                component_sizes = [sizes[left_root], sizes[left_root]]
            larger_fraction = component_sizes[0] / len(nodes) if nodes else 0.0
            smaller_fraction = component_sizes[-1] / len(nodes) if nodes else 0.0
            trial_fraction = trial_chart.get("patch_unwrap_proxy_largest_component_node_fraction")
            trial_fraction_value = float(trial_fraction) if isinstance(trial_fraction, (int, float)) else None
            p90_value = float(p90)
            cap_gap = p90_value - max_p90_distortion
            skipped_p90_values.append(p90_value)
            skipped_cap_gaps.append(cap_gap)
            skipped_larger_component_fractions.append(larger_fraction)
            skipped_smaller_component_fractions.append(smaller_fraction)
            skipped_edges.append(edge)
            if trial_fraction_value is not None:
                skipped_trial_fractions.append(trial_fraction_value)
            row = {
                "component_relation": "bridge" if is_bridge else "cycle",
                "larger_component_fraction_before": rounded(larger_fraction),
                "smaller_component_fraction_before": rounded(smaller_fraction),
                "trial_largest_component_fraction": rounded(trial_fraction_value),
                "coverage_delta_from_capped_so_far": rounded(
                    trial_fraction_value - largest_fraction(len(nodes), accepted_edges)
                    if trial_fraction_value is not None
                    else None
                ),
                "trial_p90_edge_distortion": rounded(p90_value),
                "p90_cap_gap": rounded(cap_gap),
                "trial_p10_triangle_area_ratio": rounded(
                    trial_chart.get("patch_unwrap_proxy_p10_triangle_area_ratio")
                ),
                "normal_agreement": rounded(quality.get("normal_agreement") if quality else None),
                "offset_ratio": rounded(quality.get("offset_ratio") if quality else None),
            }
            skipped_rows.append(row)

    capped_chart = ablation.unwrap_proxy_metrics(nodes, sorted(accepted_edges))
    skipped_by_p90 = sorted(
        skipped_rows,
        key=lambda row: (
            float("inf")
            if row["trial_p90_edge_distortion"] is None
            else float(row["trial_p90_edge_distortion"]),
            -float(row["trial_largest_component_fraction"] or 0.0),
            -float(row["normal_agreement"] or 0.0),
            float(row["offset_ratio"] or 0.0),
        ),
    )
    skipped_by_coverage = sorted(
        skipped_rows,
        key=lambda row: (
            -float(row["trial_largest_component_fraction"] or 0.0),
            float("inf")
            if row["trial_p90_edge_distortion"] is None
            else float(row["trial_p90_edge_distortion"]),
            -float(row["normal_agreement"] or 0.0),
            float(row["offset_ratio"] or 0.0),
        ),
    )

    def ranked(rows_in_order: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"rank": index + 1, **row} for index, row in enumerate(rows_in_order[:diagnostic_limit])]

    alternate_route_summary: dict[str, Any] = {
        "enabled": False,
        "reason": "not_requested",
        "max_length": alternate_route_max_length,
        "branching": alternate_route_branching,
    }
    if all_edges is not None and alternate_route_max_length >= 2:
        alternate_route_summary = base_cap_alternate_route_probe(
            nodes=nodes,
            all_edges=all_edges,
            accepted_edges=accepted_edges,
            skipped_edges=skipped_edges,
            edge_quality=edge_quality,
            max_p90_distortion=max_p90_distortion,
            max_length=alternate_route_max_length,
            branching=alternate_route_branching,
            diagnostic_limit=diagnostic_limit,
        )
    elif all_edges is not None:
        alternate_route_summary = {
            "enabled": False,
            "reason": "max_length_below_2",
            "max_length": alternate_route_max_length,
            "branching": alternate_route_branching,
        }
    component_probe_summary: dict[str, Any] = {
        "enabled": False,
        "reason": "not_requested",
    }
    if component_probe:
        component_probe_summary = base_cap_bridge_component_probe(
            nodes=nodes,
            accepted_edges=accepted_edges,
            skipped_edges=skipped_edges,
            edge_quality=edge_quality,
            max_p90_distortion=max_p90_distortion,
            diagnostic_limit=diagnostic_limit,
        )
    boundary_prototype_summary: dict[str, Any] = {
        "enabled": False,
        "reason": "not_requested",
    }
    if boundary_prototype:
        boundary_prototype_summary = base_cap_boundary_transition_prototype(
            nodes=nodes,
            accepted_edges=accepted_edges,
            skipped_edges=skipped_edges,
            edge_quality=edge_quality,
            max_p90_distortion=max_p90_distortion,
            target=boundary_target,
            diagnostic_limit=diagnostic_limit,
            require_two_chart_selection_gate=boundary_require_two_chart_gate,
        )

    return accepted_edges, {
        "enabled": True,
        "max_p90_distortion": rounded(max_p90_distortion),
        "input_base_edge_count": len(base_edges),
        "accepted_base_edge_count": len(accepted_edges),
        "skipped_base_edge_count": len(base_edges) - len(accepted_edges),
        "missing_quality_edge_count": missing_quality_count,
        "input_base_largest_component_fraction": rounded(largest_fraction(len(nodes), base_edges)),
        "input_base_p90_edge_distortion": rounded(input_chart["patch_unwrap_proxy_p90_edge_distortion"]),
        "capped_base_largest_component_fraction": rounded(largest_fraction(len(nodes), accepted_edges)),
        "capped_base_p90_edge_distortion": rounded(capped_chart["patch_unwrap_proxy_p90_edge_distortion"]),
        "skipped_trial_p90_summary": numeric_summary(skipped_p90_values),
        "skipped_trial_largest_component_fraction_summary": numeric_summary(skipped_trial_fractions),
        "skipped_p90_cap_gap_summary": numeric_summary(skipped_cap_gaps),
        "skipped_bridge_edge_count": skipped_bridge_count,
        "skipped_cycle_edge_count": skipped_cycle_count,
        "skipped_larger_component_fraction_summary": numeric_summary(
            skipped_larger_component_fractions
        ),
        "skipped_smaller_component_fraction_summary": numeric_summary(
            skipped_smaller_component_fractions
        ),
        "top_skipped_base_edges_by_low_p90": ranked(skipped_by_p90),
        "top_skipped_base_edges_by_coverage": ranked(skipped_by_coverage),
        "skipped_alternate_route_probe": alternate_route_summary,
        "skipped_bridge_component_probe": component_probe_summary,
        "skipped_bridge_boundary_transition_prototype": boundary_prototype_summary,
    }


def candidate_edges(
    all_edges: list[tuple[int, int]],
    base_edges: set[tuple[int, int]],
    edge_quality: list[dict[str, float | tuple[int, int]]],
) -> list[dict[str, float | tuple[int, int]]]:
    all_edge_set = {sorted_edge(edge) for edge in all_edges}
    candidates = [
        quality
        for quality in edge_quality
        if isinstance(quality.get("edge"), tuple)
        and sorted_edge(quality["edge"]) in all_edge_set
        and sorted_edge(quality["edge"]) not in base_edges
    ]
    candidates.sort(
        key=lambda item: (
            -float(item["normal_agreement"]),
            float(item["offset_ratio"]),
            sorted_edge(item["edge"]),
        )
    )
    return candidates


def grow_normal_edges(
    node_count: int,
    all_edges: list[tuple[int, int]],
    base_edges: set[tuple[int, int]],
    edge_quality: list[dict[str, float | tuple[int, int]]],
    target: float,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    grown_edges = set(base_edges)
    added_edges: set[tuple[int, int]] = set()
    for quality in candidate_edges(all_edges, base_edges, edge_quality):
        if largest_fraction(node_count, grown_edges) >= target:
            break
        edge = sorted_edge(quality["edge"])
        grown_edges.add(edge)
        added_edges.add(edge)
    return grown_edges, added_edges


def grow_distortion_edges(
    nodes: list[dict[str, Any]],
    all_edges: list[tuple[int, int]],
    base_edges: set[tuple[int, int]],
    edge_quality: list[dict[str, float | tuple[int, int]]],
    target: float,
    max_p90_distortion: float | None = None,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    node_count = len(nodes)
    grown_edges = set(base_edges)
    added_edges: set[tuple[int, int]] = set()
    candidates = candidate_edges(all_edges, base_edges, edge_quality)
    chart = ablation.unwrap_proxy_metrics(nodes, sorted(grown_edges))
    while chart["patch_unwrap_proxy_largest_component_node_fraction"] < target and candidates:
        roots, _sizes = component_roots(node_count, grown_edges)
        best_index = None
        best_score = None
        best_chart = None
        for candidate_index, quality in enumerate(candidates):
            edge = sorted_edge(quality["edge"])
            left, right = edge
            if roots[left] == roots[right]:
                continue
            trial_edges = set(grown_edges)
            trial_edges.add(edge)
            trial_chart = ablation.unwrap_proxy_metrics(nodes, sorted(trial_edges))
            trial_fraction = trial_chart["patch_unwrap_proxy_largest_component_node_fraction"]
            if trial_fraction <= chart["patch_unwrap_proxy_largest_component_node_fraction"]:
                continue
            p90 = trial_chart["patch_unwrap_proxy_p90_edge_distortion"]
            if max_p90_distortion is not None and (p90 is None or float(p90) > max_p90_distortion):
                continue
            p10_area = trial_chart["patch_unwrap_proxy_p10_triangle_area_ratio"]
            score = (
                float("inf") if p90 is None else float(p90),
                1.0 if p10_area is None else max(0.0, 1.0 - float(p10_area)),
                -float(trial_fraction),
                -float(quality["normal_agreement"]),
                float(quality["offset_ratio"]),
                edge,
            )
            if best_score is None or score < best_score:
                best_index = candidate_index
                best_score = score
                best_chart = trial_chart
        if best_index is None or best_chart is None:
            break
        chosen = candidates.pop(best_index)
        edge = sorted_edge(chosen["edge"])
        grown_edges.add(edge)
        added_edges.add(edge)
        chart = best_chart
    return grown_edges, added_edges


def rounded(value: Any, digits: int = 6) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return round(float(value), digits)
    return None


def percentile_from_sorted(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[int(position)]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def numeric_summary(values: list[Any]) -> dict[str, Any]:
    finite = sorted(
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    )
    return {
        "count": len(finite),
        "min": rounded(finite[0]) if finite else None,
        "p50": rounded(percentile_from_sorted(finite, 50)),
        "p90": rounded(percentile_from_sorted(finite, 90)),
        "max": rounded(finite[-1]) if finite else None,
    }


def categorical_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get(key) or "missing")
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def scalar_max(*values: Any) -> float | None:
    finite = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return max(finite) if finite else None


def finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def normal_guided_unwrap_proxy_metrics(
    nodes: list[dict[str, Any]],
    edges: list[tuple[int, int]],
) -> dict[str, Any]:
    """Project the largest component with a chart plane normal guided by node normals."""

    import numpy as np

    prefix = "normal_guided_global"
    base_result = {
        f"{prefix}_node_count": 0,
        f"{prefix}_edge_count": 0,
        f"{prefix}_largest_component_node_fraction": 0.0,
        "normal_guided_chart_normal_agreement": None,
        f"{prefix}_mean_edge_distortion": None,
        f"{prefix}_p90_edge_distortion": None,
    }
    node_count = len(nodes)
    edge_set: set[tuple[int, int]] = set()
    for edge in edges:
        left, right = edge
        if left == right:
            continue
        if left < 0 or right < 0 or left >= node_count or right >= node_count:
            continue
        edge_set.add(sorted_edge((left, right)))

    if node_count < 3 or not edge_set:
        return base_result

    adjacency: list[set[int]] = [set() for _ in range(node_count)]
    for left, right in edge_set:
        adjacency[left].add(right)
        adjacency[right].add(left)

    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(node_count):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)

    largest_nodes = sorted(max(components, key=len))
    largest_node_set = set(largest_nodes)
    component_edges = [
        edge
        for edge in sorted(edge_set)
        if edge[0] in largest_node_set and edge[1] in largest_node_set
    ]
    if len(largest_nodes) < 3 or not component_edges:
        return {
            **base_result,
            f"{prefix}_node_count": len(largest_nodes),
            f"{prefix}_edge_count": len(component_edges),
            f"{prefix}_largest_component_node_fraction": rounded(
                len(largest_nodes) / node_count if node_count else 0.0
            ),
        }

    centroids = np.array([nodes[index]["centroid"] for index in largest_nodes], dtype=np.float64)
    centered = centroids - centroids.mean(axis=0)
    covariance = centered.T @ centered / max(1, centered.shape[0])
    _eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    pca_normal = eigenvectors[:, 0]

    def normalized(vector) -> np.ndarray | None:
        array = np.asarray(vector, dtype=np.float64)
        length = float(np.linalg.norm(array))
        if length <= 1e-9:
            return None
        return array / length

    chart_normal = pca_normal
    chart_normal_agreement = None
    node_normals = np.array([nodes[index]["normal"] for index in largest_nodes], dtype=np.float64)
    normal_lengths = np.linalg.norm(node_normals, axis=1)
    valid_normals = normal_lengths > 1e-6
    if np.any(valid_normals):
        unit_normals = node_normals[valid_normals] / normal_lengths[valid_normals, None]
        oriented_normals = unit_normals.copy()
        opposite = (oriented_normals @ pca_normal) < 0
        oriented_normals[opposite] *= -1.0
        mean_normal = normalized(np.mean(oriented_normals, axis=0))
        if mean_normal is not None:
            chart_normal = mean_normal
        chart_normal_agreement = float(np.mean(np.abs(unit_normals @ chart_normal)))

    def projected_axis(candidate) -> np.ndarray | None:
        vector = np.asarray(candidate, dtype=np.float64)
        projected = vector - float(np.dot(vector, chart_normal)) * chart_normal
        return normalized(projected)

    axis0 = projected_axis(eigenvectors[:, -1])
    if axis0 is None:
        axis0 = projected_axis(eigenvectors[:, -2])
    if axis0 is None:
        basis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(chart_normal)))]
        axis0 = projected_axis(basis)
    if axis0 is None:
        return {
            **base_result,
            f"{prefix}_node_count": len(largest_nodes),
            f"{prefix}_edge_count": len(component_edges),
            f"{prefix}_largest_component_node_fraction": rounded(
                len(largest_nodes) / node_count if node_count else 0.0
            ),
            "normal_guided_chart_normal_agreement": rounded(chart_normal_agreement),
        }

    axis1 = normalized(np.cross(chart_normal, axis0))
    if axis1 is None:
        return {
            **base_result,
            f"{prefix}_node_count": len(largest_nodes),
            f"{prefix}_edge_count": len(component_edges),
            f"{prefix}_largest_component_node_fraction": rounded(
                len(largest_nodes) / node_count if node_count else 0.0
            ),
            "normal_guided_chart_normal_agreement": rounded(chart_normal_agreement),
        }

    chart_axes = np.stack([axis0, axis1], axis=1)
    chart_coords = centered @ chart_axes
    local_index = {node_index: index for index, node_index in enumerate(largest_nodes)}

    edge_distortions: list[float] = []
    for left, right in component_edges:
        left_local = local_index[left]
        right_local = local_index[right]
        length_3d = float(np.linalg.norm(centroids[right_local] - centroids[left_local]))
        if length_3d <= 1e-9:
            continue
        length_2d = float(np.linalg.norm(chart_coords[right_local] - chart_coords[left_local]))
        edge_distortions.append(abs(length_2d / length_3d - 1.0))

    return {
        f"{prefix}_node_count": len(largest_nodes),
        f"{prefix}_edge_count": len(component_edges),
        f"{prefix}_largest_component_node_fraction": rounded(
            len(largest_nodes) / node_count if node_count else 0.0
        ),
        "normal_guided_chart_normal_agreement": rounded(chart_normal_agreement),
        f"{prefix}_mean_edge_distortion": rounded(
            float(np.mean(np.array(edge_distortions, dtype=np.float64))) if edge_distortions else None
        ),
        f"{prefix}_p90_edge_distortion": rounded(
            float(np.percentile(np.array(edge_distortions, dtype=np.float64), 90))
            if edge_distortions
            else None
        ),
    }


def two_chart_rigid_trial_metrics(
    nodes: list[dict[str, Any]],
    base_edges: set[tuple[int, int]],
    base_roots: list[int],
    component_sizes: dict[int, int],
    edge: tuple[int, int],
    chart_frames: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Measure a scalar two-chart rigid placement for one component bridge."""

    import numpy as np

    prefix = "two_chart_rigid_trial"
    base_result = {
        f"{prefix}_available": False,
        f"{prefix}_node_fraction": None,
        f"{prefix}_edge_count": 0,
        f"{prefix}_anchor_component_fraction": None,
        f"{prefix}_moving_component_fraction": None,
        f"{prefix}_chart_tangent_singular_min": None,
        f"{prefix}_chart_tangent_singular_max": None,
        f"{prefix}_orthogonal_transform_determinant": None,
        f"{prefix}_bridge_projection_ratio": None,
        f"{prefix}_bridge_edge_distortion": None,
        f"{prefix}_internal_p90_edge_distortion": None,
        f"{prefix}_bridge_to_internal_p90_ratio": None,
        f"{prefix}_all_edge_p90_distortion": None,
        f"{prefix}_p90_edge_distortion": None,
    }
    node_count = len(nodes)
    left, right = sorted_edge(edge)
    if left >= node_count or right >= node_count:
        return base_result
    left_root = base_roots[left]
    right_root = base_roots[right]
    if left_root == right_root:
        return base_result

    left_frame = chart_frames.get(left_root)
    right_frame = chart_frames.get(right_root)
    if left_frame is None or right_frame is None:
        return base_result

    left_size = component_sizes.get(left_root, 0)
    right_size = component_sizes.get(right_root, 0)
    if left_size >= right_size:
        anchor_root = left_root
        moving_root = right_root
        anchor_endpoint = left
        moving_endpoint = right
        anchor_frame = left_frame
        moving_frame = right_frame
    else:
        anchor_root = right_root
        moving_root = left_root
        anchor_endpoint = right
        moving_endpoint = left
        anchor_frame = right_frame
        moving_frame = left_frame

    anchor_size = component_sizes.get(anchor_root, 0)
    moving_size = component_sizes.get(moving_root, 0)
    bridge_delta = nodes[moving_endpoint]["centroid"] - nodes[anchor_endpoint]["centroid"]
    bridge_length_3d = float(np.linalg.norm(bridge_delta))
    if bridge_length_3d <= 1e-9:
        return {
            **base_result,
            f"{prefix}_anchor_component_fraction": rounded(
                anchor_size / node_count if node_count else 0.0
            ),
            f"{prefix}_moving_component_fraction": rounded(
                moving_size / node_count if node_count else 0.0
            ),
        }

    tangent_matrix = anchor_frame["axes"].T @ moving_frame["axes"]
    singular_values = np.linalg.svd(tangent_matrix, compute_uv=False)
    u, _s, vt = np.linalg.svd(tangent_matrix)
    transform = u @ vt
    bridge_vector_2d = anchor_frame["axes"].T @ bridge_delta
    bridge_length_2d = float(np.linalg.norm(bridge_vector_2d))
    bridge_distortion = abs(bridge_length_2d / bridge_length_3d - 1.0)

    def local_coord(frame: dict[str, Any], node_index: int):
        return frame["axes"].T @ (nodes[node_index]["centroid"] - frame["centroid"])

    anchor_endpoint_coord = local_coord(anchor_frame, anchor_endpoint)
    moving_endpoint_coord = local_coord(moving_frame, moving_endpoint)
    translation = anchor_endpoint_coord + bridge_vector_2d - transform @ moving_endpoint_coord

    def placed_coord(node_index: int):
        root = base_roots[node_index]
        if root == anchor_root:
            return local_coord(anchor_frame, node_index)
        if root == moving_root:
            return transform @ local_coord(moving_frame, node_index) + translation
        return None

    internal_distortions: list[float] = []
    all_distortions: list[float] = [bridge_distortion]
    trial_edge_count = 1
    for base_edge in sorted(base_edges):
        edge_left, edge_right = base_edge
        if base_roots[edge_left] not in (anchor_root, moving_root):
            continue
        if base_roots[edge_left] != base_roots[edge_right]:
            continue
        left_coord = placed_coord(edge_left)
        right_coord = placed_coord(edge_right)
        if left_coord is None or right_coord is None:
            continue
        length_3d = float(
            np.linalg.norm(nodes[edge_right]["centroid"] - nodes[edge_left]["centroid"])
        )
        if length_3d <= 1e-9:
            continue
        length_2d = float(np.linalg.norm(right_coord - left_coord))
        distortion = abs(length_2d / length_3d - 1.0)
        internal_distortions.append(distortion)
        all_distortions.append(distortion)
        trial_edge_count += 1

    def percentile_or_none(values: list[float], percentile: float) -> float | None:
        return float(np.percentile(np.array(values, dtype=np.float64), percentile)) if values else None

    internal_p90 = percentile_or_none(internal_distortions, 90)
    all_edge_p90 = percentile_or_none(all_distortions, 90)
    quality_p90 = scalar_max(internal_p90, bridge_distortion)
    bridge_to_internal_ratio = (
        bridge_distortion / internal_p90
        if internal_p90 is not None and internal_p90 > 1e-12
        else None
    )

    return {
        f"{prefix}_available": True,
        f"{prefix}_node_fraction": rounded(
            (anchor_size + moving_size) / node_count if node_count else 0.0
        ),
        f"{prefix}_edge_count": trial_edge_count,
        f"{prefix}_anchor_component_fraction": rounded(
            anchor_size / node_count if node_count else 0.0
        ),
        f"{prefix}_moving_component_fraction": rounded(
            moving_size / node_count if node_count else 0.0
        ),
        f"{prefix}_chart_tangent_singular_min": rounded(float(np.min(singular_values))),
        f"{prefix}_chart_tangent_singular_max": rounded(float(np.max(singular_values))),
        f"{prefix}_orthogonal_transform_determinant": rounded(float(np.linalg.det(transform))),
        f"{prefix}_bridge_projection_ratio": rounded(bridge_length_2d / bridge_length_3d),
        f"{prefix}_bridge_edge_distortion": rounded(bridge_distortion),
        f"{prefix}_internal_p90_edge_distortion": rounded(internal_p90),
        f"{prefix}_bridge_to_internal_p90_ratio": rounded(bridge_to_internal_ratio),
        f"{prefix}_all_edge_p90_distortion": rounded(all_edge_p90),
        f"{prefix}_p90_edge_distortion": rounded(quality_p90),
    }


def two_chart_rigid_trial_gate_metrics(
    rigid_trial: dict[str, Any],
    max_p90_distortion: float,
) -> dict[str, Any]:
    """Classify a two-chart rigid trial with the scalar gates learned so far."""

    prefix = "two_chart_rigid_trial"
    rigid_p90 = finite_float(rigid_trial.get(f"{prefix}_p90_edge_distortion"))
    projection = finite_float(rigid_trial.get(f"{prefix}_bridge_projection_ratio"))
    determinant = finite_float(rigid_trial.get(f"{prefix}_orthogonal_transform_determinant"))
    cap_compliant = rigid_p90 is not None and rigid_p90 <= max_p90_distortion
    projection_gate = (
        projection is not None and projection >= TWO_CHART_MIN_BRIDGE_PROJECTION_RATIO
    )
    orientation_gate = (
        determinant is not None and determinant > TWO_CHART_MIN_ORIENTATION_DETERMINANT
    )

    if rigid_trial.get(f"{prefix}_available") is not True:
        placement_class = "unavailable"
    elif projection is not None and projection < TWO_CHART_MIN_BRIDGE_PROJECTION_RATIO:
        placement_class = "low_bridge_projection_failure"
    elif determinant is not None and determinant <= TWO_CHART_MIN_ORIENTATION_DETERMINANT:
        placement_class = "reflection_like_transform_failure"
    elif cap_compliant:
        placement_class = "rigid_cap_success"
    else:
        placement_class = "rigid_cap_failure"

    return {
        f"{prefix}_projection_gate_threshold": rounded(
            TWO_CHART_MIN_BRIDGE_PROJECTION_RATIO
        ),
        f"{prefix}_orientation_determinant_threshold": rounded(
            TWO_CHART_MIN_ORIENTATION_DETERMINANT
        ),
        f"{prefix}_projection_gate_pass": projection_gate,
        f"{prefix}_orientation_gate_pass": orientation_gate,
        f"{prefix}_selection_gate_pass": (
            cap_compliant and projection_gate and orientation_gate
        ),
        f"{prefix}_placement_class": placement_class,
    }


def multi_chart_atlas_metrics(
    nodes: list[dict[str, Any]],
    base_edges: set[tuple[int, int]],
    grown_edges: set[tuple[int, int]],
) -> dict[str, Any]:
    """Score a scalar rigid atlas over original base components and accepted bridges."""

    import numpy as np

    prefix = "multi_chart_atlas"
    base_result = {
        f"{prefix}_available": False,
        f"{prefix}_placed_component_count": 0,
        f"{prefix}_placed_node_fraction": None,
        f"{prefix}_internal_edge_count": 0,
        f"{prefix}_bridge_edge_count": 0,
        f"{prefix}_all_edge_count": 0,
        f"{prefix}_internal_p90_edge_distortion": None,
        f"{prefix}_bridge_p90_edge_distortion": None,
        f"{prefix}_bridge_max_edge_distortion": None,
        f"{prefix}_all_edge_p90_distortion": None,
        f"{prefix}_bridge_to_internal_p90_ratio": None,
        f"{prefix}_bridge_aware_p90_edge_distortion": None,
        f"{prefix}_singleton_endpoint_placement_count": 0,
        f"{prefix}_too_small_component_placement_count": 0,
        f"{prefix}_unplaced_bridge_edge_count": 0,
        f"{prefix}_unplaced_bridge_missing_chart_frame_count": 0,
        f"{prefix}_unplaced_bridge_missing_chart_frame_endpoint_count": 0,
        f"{prefix}_unplaced_bridge_too_small_endpoint_count": 0,
        f"{prefix}_unplaced_bridge_disconnected_count": 0,
    }
    node_count = len(nodes)
    if node_count < 3 or not base_edges:
        return base_result

    base_roots, base_sizes = component_roots(node_count, base_edges)
    component_nodes: dict[int, list[int]] = defaultdict(list)
    for node_index, root in enumerate(base_roots):
        component_nodes[root].append(node_index)
    _frame_roots, chart_frames = component_chart_frames(nodes, base_edges)
    bridge_edges = [
        sorted_edge(edge)
        for edge in sorted(grown_edges)
        if edge not in base_edges and base_roots[edge[0]] != base_roots[edge[1]]
    ]
    if not bridge_edges:
        return base_result

    root_graph: dict[int, list[tuple[tuple[int, int], int]]] = defaultdict(list)
    roots_with_bridges: set[int] = set()
    for edge in bridge_edges:
        left, right = edge
        left_root = base_roots[left]
        right_root = base_roots[right]
        roots_with_bridges.add(left_root)
        roots_with_bridges.add(right_root)
        root_graph[left_root].append((edge, right_root))
        root_graph[right_root].append((edge, left_root))

    start_candidates = [
        root
        for root in roots_with_bridges
        if root in chart_frames and base_sizes.get(root, 0) >= 3
    ]
    if not start_candidates:
        return {
            **base_result,
            f"{prefix}_bridge_edge_count": len(bridge_edges),
            f"{prefix}_unplaced_bridge_edge_count": len(bridge_edges),
            f"{prefix}_unplaced_bridge_missing_chart_frame_count": len(bridge_edges),
            f"{prefix}_unplaced_bridge_missing_chart_frame_endpoint_count": 2 * len(bridge_edges),
            f"{prefix}_unplaced_bridge_too_small_endpoint_count": sum(
                1
                for edge in bridge_edges
                for root in (base_roots[edge[0]], base_roots[edge[1]])
                if base_sizes.get(root, 0) < 3
            ),
        }
    def local_coord(root: int, node_index: int):
        frame = chart_frames[root]
        return frame["axes"].T @ (nodes[node_index]["centroid"] - frame["centroid"])

    def placed_coord(root: int, node_index: int):
        placement = placements[root]
        point_coords = placement.get("point_coords")
        if isinstance(point_coords, dict):
            return point_coords.get(node_index)
        return placement["rotation"] @ local_coord(root, node_index) + placement["translation"]

    placements: dict[int, dict[str, Any]] = {}
    for start_root in sorted(
        start_candidates,
        key=lambda root: base_sizes.get(root, 0),
        reverse=True,
    ):
        if start_root in placements:
            continue
        placements[start_root] = {
            "kind": "frame",
            "axes": chart_frames[start_root]["axes"],
            "rotation": np.eye(2, dtype=np.float64),
            "translation": np.zeros(2, dtype=np.float64),
        }
        queue = [start_root]
        while queue:
            anchor_root = queue.pop(0)
            anchor_frame = chart_frames.get(anchor_root)
            if anchor_frame is None:
                continue
            for edge, other_root in root_graph.get(anchor_root, []):
                if other_root in placements:
                    continue
                moving_frame = chart_frames.get(other_root)
                if moving_frame is None:
                    continue
                left, right = edge
                if base_roots[left] == anchor_root and base_roots[right] == other_root:
                    anchor_endpoint = left
                    moving_endpoint = right
                elif base_roots[right] == anchor_root and base_roots[left] == other_root:
                    anchor_endpoint = right
                    moving_endpoint = left
                else:
                    continue
                bridge_delta = nodes[moving_endpoint]["centroid"] - nodes[anchor_endpoint]["centroid"]
                bridge_vector_2d = anchor_frame["axes"].T @ bridge_delta
                tangent_matrix = anchor_frame["axes"].T @ moving_frame["axes"]
                u, _s, vt = np.linalg.svd(tangent_matrix)
                local_transform = u @ vt
                anchor_rotation = placements[anchor_root]["rotation"]
                moving_rotation = anchor_rotation @ local_transform
                translation = (
                    placed_coord(anchor_root, anchor_endpoint)
                    + anchor_rotation @ bridge_vector_2d
                    - moving_rotation @ local_coord(other_root, moving_endpoint)
                )
                placements[other_root] = {
                    "kind": "frame",
                    "axes": moving_frame["axes"],
                    "rotation": moving_rotation,
                    "translation": translation,
                }
                queue.append(other_root)

    singleton_endpoint_placement_count = 0
    too_small_component_placement_count = 0
    progress = True
    while progress:
        progress = False
        for edge in bridge_edges:
            left, right = edge
            left_root = base_roots[left]
            right_root = base_roots[right]
            if left_root in placements and right_root in placements:
                continue
            if left_root in placements and 0 < base_sizes.get(right_root, 0) < 3:
                source_root = left_root
                source_endpoint = left
                target_root = right_root
                target_endpoint = right
            elif right_root in placements and 0 < base_sizes.get(left_root, 0) < 3:
                source_root = right_root
                source_endpoint = right
                target_root = left_root
                target_endpoint = left
            else:
                continue
            source_coord = placed_coord(source_root, source_endpoint)
            if source_coord is None:
                continue
            source_placement = placements[source_root]
            point_coords = {}
            for member in component_nodes.get(target_root, [target_endpoint]):
                bridge_delta = nodes[member]["centroid"] - nodes[source_endpoint]["centroid"]
                point_coords[member] = source_coord + source_placement["rotation"] @ (
                    source_placement["axes"].T @ bridge_delta
                )
            placements[target_root] = {
                "kind": "too_small_component",
                "axes": source_placement["axes"],
                "rotation": source_placement["rotation"],
                "point_coords": point_coords,
            }
            too_small_component_placement_count += 1
            if base_sizes.get(target_root, 0) == 1:
                singleton_endpoint_placement_count += 1
            progress = True

    internal_distortions: list[float] = []
    for edge in sorted(base_edges):
        left, right = edge
        root = base_roots[left]
        if root != base_roots[right] or root not in placements:
            continue
        left_coord = placed_coord(root, left)
        right_coord = placed_coord(root, right)
        if left_coord is None or right_coord is None:
            continue
        length_3d = float(np.linalg.norm(nodes[right]["centroid"] - nodes[left]["centroid"]))
        if length_3d <= 1e-9:
            continue
        length_2d = float(np.linalg.norm(right_coord - left_coord))
        internal_distortions.append(abs(length_2d / length_3d - 1.0))

    bridge_distortions: list[float] = []
    unplaced_bridge_count = 0
    unplaced_missing_frame_count = 0
    unplaced_missing_frame_endpoint_count = 0
    unplaced_too_small_endpoint_count = 0
    unplaced_disconnected_count = 0
    for edge in bridge_edges:
        left, right = edge
        left_root = base_roots[left]
        right_root = base_roots[right]
        if left_root not in placements or right_root not in placements:
            unplaced_bridge_count += 1
            left_has_frame = left_root in chart_frames
            right_has_frame = right_root in chart_frames
            missing_endpoint_count = int(not left_has_frame) + int(not right_has_frame)
            unplaced_missing_frame_endpoint_count += missing_endpoint_count
            if missing_endpoint_count:
                unplaced_missing_frame_count += 1
            else:
                unplaced_disconnected_count += 1
            unplaced_too_small_endpoint_count += int(base_sizes.get(left_root, 0) < 3)
            unplaced_too_small_endpoint_count += int(base_sizes.get(right_root, 0) < 3)
            continue
        left_coord = placed_coord(left_root, left)
        right_coord = placed_coord(right_root, right)
        if left_coord is None or right_coord is None:
            unplaced_bridge_count += 1
            unplaced_disconnected_count += 1
            continue
        length_3d = float(np.linalg.norm(nodes[right]["centroid"] - nodes[left]["centroid"]))
        if length_3d <= 1e-9:
            continue
        length_2d = float(np.linalg.norm(right_coord - left_coord))
        bridge_distortions.append(abs(length_2d / length_3d - 1.0))

    def percentile_or_none(values: list[float], percentile: float) -> float | None:
        return float(np.percentile(np.array(values, dtype=np.float64), percentile)) if values else None

    internal_p90 = percentile_or_none(internal_distortions, 90)
    bridge_p90 = percentile_or_none(bridge_distortions, 90)
    all_distortions = [*internal_distortions, *bridge_distortions]
    all_edge_p90 = percentile_or_none(all_distortions, 90)
    bridge_max = max(bridge_distortions) if bridge_distortions else None
    bridge_aware_p90 = scalar_max(internal_p90, bridge_p90)
    placed_node_count = sum(base_sizes.get(root, 0) for root in placements)
    bridge_to_internal_ratio = (
        bridge_p90 / internal_p90
        if internal_p90 is not None and bridge_p90 is not None and internal_p90 > 1e-12
        else None
    )
    return {
        f"{prefix}_available": bool(placements and bridge_distortions),
        f"{prefix}_placed_component_count": len(placements),
        f"{prefix}_placed_node_fraction": rounded(
            placed_node_count / node_count if node_count else 0.0
        ),
        f"{prefix}_internal_edge_count": len(internal_distortions),
        f"{prefix}_bridge_edge_count": len(bridge_distortions),
        f"{prefix}_all_edge_count": len(all_distortions),
        f"{prefix}_internal_p90_edge_distortion": rounded(internal_p90),
        f"{prefix}_bridge_p90_edge_distortion": rounded(bridge_p90),
        f"{prefix}_bridge_max_edge_distortion": rounded(bridge_max),
        f"{prefix}_all_edge_p90_distortion": rounded(all_edge_p90),
        f"{prefix}_bridge_to_internal_p90_ratio": rounded(bridge_to_internal_ratio),
        f"{prefix}_bridge_aware_p90_edge_distortion": rounded(bridge_aware_p90),
        f"{prefix}_singleton_endpoint_placement_count": singleton_endpoint_placement_count,
        f"{prefix}_too_small_component_placement_count": too_small_component_placement_count,
        f"{prefix}_unplaced_bridge_edge_count": unplaced_bridge_count,
        f"{prefix}_unplaced_bridge_missing_chart_frame_count": unplaced_missing_frame_count,
        f"{prefix}_unplaced_bridge_missing_chart_frame_endpoint_count": (
            unplaced_missing_frame_endpoint_count
        ),
        f"{prefix}_unplaced_bridge_too_small_endpoint_count": unplaced_too_small_endpoint_count,
        f"{prefix}_unplaced_bridge_disconnected_count": unplaced_disconnected_count,
    }


def bridge_projection_min(row: dict[str, Any]) -> float | None:
    values = [
        row.get("bridge_projection_ratio_left"),
        row.get("bridge_projection_ratio_right"),
    ]
    if not all(finite_float(value) for value in values):
        return None
    return min(float(value) for value in values)


def broad_candidate_policy_summary(
    rows: list[dict[str, Any]],
    target: float | None,
    limit: int,
) -> dict[str, Any]:
    if not finite_float(target):
        return {
            "enabled": False,
            "reason": "target_not_available",
        }
    target_value = float(target)

    def policy_pass(row: dict[str, Any]) -> bool:
        projection = bridge_projection_min(row)
        return (
            bool(row.get("chart_pair_available"))
            and bool(row.get("touches_base_largest_component"))
            and finite_float(row.get("trial_largest_component_fraction"))
            and float(row["trial_largest_component_fraction"]) >= target_value
            and finite_float(row.get("normal_agreement"))
            and float(row["normal_agreement"]) >= BROAD_CANDIDATE_MIN_NORMAL_AGREEMENT
            and projection is not None
            and projection >= BROAD_CANDIDATE_MIN_BRIDGE_PROJECTION_RATIO
        )

    def policy_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        p90 = row.get("trial_p90_edge_distortion")
        return (
            -float(row.get("trial_largest_component_fraction") or 0.0),
            -float(row.get("normal_agreement") or 0.0),
            -float(bridge_projection_min(row) or 0.0),
            float("inf") if p90 is None else float(p90),
        )

    passing_rows = [row for row in rows if policy_pass(row)]
    top_rows = []
    for index, row in enumerate(sorted(passing_rows, key=policy_key)[:limit], start=1):
        top_rows.append(
            {
                "rank": index,
                **row,
                "min_bridge_projection_ratio": rounded(bridge_projection_min(row)),
            }
        )
    return {
        "enabled": True,
        "target": rounded(target_value),
        "normal_agreement_min": rounded(BROAD_CANDIDATE_MIN_NORMAL_AGREEMENT),
        "bridge_projection_ratio_min": rounded(
            BROAD_CANDIDATE_MIN_BRIDGE_PROJECTION_RATIO
        ),
        "requires_chart_pair_available": True,
        "requires_touches_base_largest_component": True,
        "pass_count": len(passing_rows),
        "best_trial_largest_component_fraction": rounded(
            max(
                (
                    float(row["trial_largest_component_fraction"])
                    for row in passing_rows
                ),
                default=0.0,
            )
        ),
        "best_trial_p90_edge_distortion": rounded(
            min(
                (
                    float(row["trial_p90_edge_distortion"])
                    for row in passing_rows
                    if row.get("trial_p90_edge_distortion") is not None
                ),
                default=float("inf"),
            )
        ),
        "top_by_policy": top_rows,
    }


def candidate_diagnostics_only_sequence(
    target: float | None,
    base_fraction: float,
    reason: str = "candidate_diagnostics_only",
) -> dict[str, Any]:
    return {
        "enabled": False,
        "reason": reason,
        "target": rounded(target),
        "reached": False,
        "final_largest_component_fraction": rounded(base_fraction),
        "final_quality_p90": None,
        "final_global_p90_edge_distortion": None,
    }


def candidate_diagnostics_only_local_metrics() -> dict[str, Any]:
    return {
        "local_chart_quality_p90": None,
        "local_chart_p90_bridge_offset_ratio": None,
        "local_chart_p10_bridge_normal_agreement": None,
    }


def base_cap_alternate_route_probe(
    nodes: list[dict[str, Any]],
    all_edges: list[tuple[int, int]],
    accepted_edges: set[tuple[int, int]],
    skipped_edges: list[tuple[int, int]],
    edge_quality: list[dict[str, float | tuple[int, int]]],
    max_p90_distortion: float,
    max_length: int,
    branching: int,
    diagnostic_limit: int,
) -> dict[str, Any]:
    """Search scalar summaries of alternate component routes for skipped base bridges."""

    if max_length < 2:
        return {"enabled": False, "reason": "max_length_below_2"}
    if branching < 1:
        raise ValueError("branching must be positive")
    if diagnostic_limit < 1:
        raise ValueError("diagnostic_limit must be positive")

    node_count = len(nodes)
    roots, sizes = component_roots(node_count, accepted_edges)
    quality_by_edge = {
        sorted_edge(quality["edge"]): quality
        for quality in edge_quality
        if isinstance(quality.get("edge"), tuple)
    }
    skipped_edge_set = {sorted_edge(edge) for edge in skipped_edges}
    adjacency: dict[int, list[tuple[int, tuple[int, int], dict[str, Any]]]] = defaultdict(list)
    for raw_edge in all_edges:
        edge = sorted_edge(raw_edge)
        if edge in accepted_edges or edge in skipped_edge_set:
            continue
        left, right = edge
        left_root = roots[left]
        right_root = roots[right]
        if left_root == right_root:
            continue
        quality = quality_by_edge.get(edge)
        if quality is None:
            continue
        adjacency[left_root].append((right_root, edge, quality))
        adjacency[right_root].append((left_root, edge, quality))

    def route_sort_key(item: tuple[int, tuple[int, int], dict[str, Any]]) -> tuple[float, float, tuple[int, int]]:
        _other, edge, quality = item
        return (
            -float(quality["normal_agreement"]),
            float(quality["offset_ratio"]),
            edge,
        )

    for root in list(adjacency):
        adjacency[root].sort(key=route_sort_key)

    path_limit = max(diagnostic_limit, diagnostic_limit * max_length * branching)
    per_skipped: list[dict[str, Any]] = []
    all_route_rows: list[dict[str, Any]] = []

    for skipped_edge in skipped_edges:
        left, right = sorted_edge(skipped_edge)
        source_root = roots[left]
        target_root = roots[right]
        if source_root == target_root:
            per_skipped.append(
                {
                    "component_relation": "cycle",
                    "source_component_fraction": rounded(sizes.get(source_root, 0) / node_count if node_count else 0.0),
                    "target_component_fraction": rounded(sizes.get(target_root, 0) / node_count if node_count else 0.0),
                    "visited_path_count": 0,
                    "target_connecting_path_count": 0,
                    "cap_compliant_target_path_count": 0,
                    "truncated": False,
                    "top_by_low_p90": [],
                    "top_by_coverage": [],
                }
            )
            continue

        route_rows: list[dict[str, Any]] = []
        visited_path_count = 0
        truncated = False
        stack: list[tuple[int, tuple[int, ...], list[tuple[int, int]], list[dict[str, Any]]]] = [
            (source_root, (source_root,), [], [])
        ]
        while stack and visited_path_count < path_limit:
            current_root, visited_roots, path_edges, path_qualities = stack.pop()
            if len(path_edges) >= max_length:
                continue
            for next_root, edge, quality in list(reversed(adjacency.get(current_root, [])[:branching])):
                if next_root in visited_roots:
                    continue
                next_edges = [*path_edges, edge]
                next_qualities = [*path_qualities, quality]
                visited_path_count += 1
                if next_root == target_root:
                    trial_edges = set(accepted_edges)
                    trial_edges.update(next_edges)
                    trial_chart = ablation.unwrap_proxy_metrics(nodes, sorted(trial_edges))
                    p90 = trial_chart.get("patch_unwrap_proxy_p90_edge_distortion")
                    trial_fraction = trial_chart.get("patch_unwrap_proxy_largest_component_node_fraction")
                    p90_value = float(p90) if isinstance(p90, (int, float)) else None
                    cap_compliant = p90_value is not None and p90_value <= max_p90_distortion
                    normal_values = [float(item["normal_agreement"]) for item in next_qualities]
                    offset_values = [float(item["offset_ratio"]) for item in next_qualities]
                    row = {
                        "path_length": len(next_edges),
                        "trial_largest_component_fraction": rounded(trial_fraction),
                        "trial_p90_edge_distortion": rounded(p90_value),
                        "p90_cap_gap": rounded(
                            p90_value - max_p90_distortion if p90_value is not None else None
                        ),
                        "trial_p10_triangle_area_ratio": rounded(
                            trial_chart.get("patch_unwrap_proxy_p10_triangle_area_ratio")
                        ),
                        "cap_compliant": bool(cap_compliant),
                        "mean_normal_agreement": rounded(sum(normal_values) / len(normal_values)),
                        "min_normal_agreement": rounded(min(normal_values)),
                        "mean_offset_ratio": rounded(sum(offset_values) / len(offset_values)),
                        "max_offset_ratio": rounded(max(offset_values)),
                    }
                    route_rows.append(row)
                    all_route_rows.append(row)
                else:
                    stack.append(
                        (
                            next_root,
                            (*visited_roots, next_root),
                            next_edges,
                            next_qualities,
                        )
                    )
                if visited_path_count >= path_limit:
                    truncated = True
                    break

        def p90_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
            p90 = row["trial_p90_edge_distortion"]
            return (
                float("inf") if p90 is None else float(p90),
                -float(row["trial_largest_component_fraction"] or 0.0),
                float(row["path_length"]),
                float(row["max_offset_ratio"] or 0.0),
            )

        def coverage_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
            p90 = row["trial_p90_edge_distortion"]
            return (
                -float(row["trial_largest_component_fraction"] or 0.0),
                float("inf") if p90 is None else float(p90),
                float(row["path_length"]),
                float(row["max_offset_ratio"] or 0.0),
            )

        def ranked(rows_in_order: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [{"rank": index + 1, **row} for index, row in enumerate(rows_in_order[:diagnostic_limit])]

        cap_rows = [row for row in route_rows if row["cap_compliant"]]
        per_skipped.append(
            {
                "component_relation": "bridge",
                "larger_component_fraction": rounded(
                    max(sizes.get(source_root, 0), sizes.get(target_root, 0)) / node_count
                    if node_count
                    else 0.0
                ),
                "smaller_component_fraction": rounded(
                    min(sizes.get(source_root, 0), sizes.get(target_root, 0)) / node_count
                    if node_count
                    else 0.0
                ),
                "visited_path_count": visited_path_count,
                "target_connecting_path_count": len(route_rows),
                "cap_compliant_target_path_count": len(cap_rows),
                "truncated": truncated,
                "best_cap_compliant_trial_fraction": rounded(
                    max((float(row["trial_largest_component_fraction"]) for row in cap_rows), default=0.0)
                ),
                "best_cap_compliant_p90": rounded(
                    min((float(row["trial_p90_edge_distortion"]) for row in cap_rows), default=float("inf"))
                ),
                "target_path_p90_summary": numeric_summary(
                    [row["trial_p90_edge_distortion"] for row in route_rows]
                ),
                "target_path_length_summary": numeric_summary([row["path_length"] for row in route_rows]),
                "top_by_low_p90": ranked(sorted(route_rows, key=p90_key)),
                "top_by_coverage": ranked(sorted(route_rows, key=coverage_key)),
            }
        )

    cap_route_rows = [row for row in all_route_rows if row["cap_compliant"]]
    return {
        "enabled": True,
        "max_length": max_length,
        "branching": branching,
        "path_limit_per_skipped_edge": path_limit,
        "skipped_edge_count": len(skipped_edges),
        "target_connecting_path_count": len(all_route_rows),
        "cap_compliant_target_path_count": len(cap_route_rows),
        "best_cap_compliant_trial_fraction": rounded(
            max((float(row["trial_largest_component_fraction"]) for row in cap_route_rows), default=0.0)
        ),
        "best_cap_compliant_p90": rounded(
            min((float(row["trial_p90_edge_distortion"]) for row in cap_route_rows), default=float("inf"))
        ),
        "target_path_p90_summary": numeric_summary(
            [row["trial_p90_edge_distortion"] for row in all_route_rows]
        ),
        "target_path_length_summary": numeric_summary([row["path_length"] for row in all_route_rows]),
        "skipped_edge_route_summaries": per_skipped,
    }


def base_cap_bridge_component_probe(
    nodes: list[dict[str, Any]],
    accepted_edges: set[tuple[int, int]],
    skipped_edges: list[tuple[int, int]],
    edge_quality: list[dict[str, float | tuple[int, int]]],
    max_p90_distortion: float,
    diagnostic_limit: int,
) -> dict[str, Any]:
    """Measure skipped bridges as separate local components plus scalar transition quality."""

    if diagnostic_limit < 1:
        raise ValueError("diagnostic_limit must be positive")

    node_count = len(nodes)
    roots, sizes = component_roots(node_count, accepted_edges)
    base_roots, chart_frames = component_chart_frames(nodes, accepted_edges)
    quality_by_edge = {
        sorted_edge(quality["edge"]): quality
        for quality in edge_quality
        if isinstance(quality.get("edge"), tuple)
    }

    component_edge_cache: dict[int, set[tuple[int, int]]] = {}

    def component_edges(root: int) -> set[tuple[int, int]]:
        cached = component_edge_cache.get(root)
        if cached is not None:
            return cached
        edges = {
            edge
            for edge in accepted_edges
            if roots[edge[0]] == root and roots[edge[1]] == root
        }
        component_edge_cache[root] = edges
        return edges

    def component_chart(root: int) -> dict[str, Any]:
        return ablation.unwrap_proxy_metrics(nodes, sorted(component_edges(root)))

    rows: list[dict[str, Any]] = []
    for skipped_edge in skipped_edges:
        edge = sorted_edge(skipped_edge)
        left, right = edge
        left_root = roots[left]
        right_root = roots[right]
        if left_root == right_root:
            rows.append(
                {
                    "component_relation": "cycle",
                    "component_fraction": rounded(sizes.get(left_root, 0) / node_count if node_count else 0.0),
                    "global_trial_cap_compliant": False,
                    "two_chart_quality_cap_compliant": False,
                }
            )
            continue

        root_rows = [
            (left_root, sizes.get(left_root, 0), component_chart(left_root)),
            (right_root, sizes.get(right_root, 0), component_chart(right_root)),
        ]
        root_rows.sort(key=lambda item: item[1], reverse=True)
        larger_root, larger_size, larger_chart = root_rows[0]
        smaller_root, smaller_size, smaller_chart = root_rows[1]
        larger_edges = component_edges(larger_root)
        smaller_edges = component_edges(smaller_root)
        trial_edges = set(accepted_edges)
        trial_edges.add(edge)
        trial_chart = ablation.unwrap_proxy_metrics(nodes, sorted(trial_edges))
        trial_p90 = trial_chart.get("patch_unwrap_proxy_p90_edge_distortion")
        trial_p90_value = float(trial_p90) if isinstance(trial_p90, (int, float)) else None
        quality = quality_by_edge.get(edge) or {}
        normal_agreement = quality.get("normal_agreement")
        offset_ratio = quality.get("offset_ratio")
        normal_risk = 1.0 - float(normal_agreement) if isinstance(normal_agreement, (int, float)) else None
        larger_p90 = larger_chart.get("patch_unwrap_proxy_p90_edge_distortion")
        smaller_p90 = smaller_chart.get("patch_unwrap_proxy_p90_edge_distortion")
        local_values = [
            float(value)
            for value in (larger_p90, smaller_p90, offset_ratio, normal_risk)
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        two_chart_quality = max(local_values) if local_values else None
        rigid_trial = two_chart_rigid_trial_metrics(
            nodes,
            accepted_edges,
            base_roots,
            sizes,
            edge,
            chart_frames,
        )
        rigid_p90 = rigid_trial.get("two_chart_rigid_trial_p90_edge_distortion")
        rigid_p90_value = float(rigid_p90) if isinstance(rigid_p90, (int, float)) else None
        row = {
            "component_relation": "bridge",
            "larger_component_fraction": rounded(larger_size / node_count if node_count else 0.0),
            "smaller_component_fraction": rounded(smaller_size / node_count if node_count else 0.0),
            "larger_component_edge_count": len(larger_edges),
            "smaller_component_edge_count": len(smaller_edges),
            "larger_component_p90_edge_distortion": rounded(larger_p90),
            "smaller_component_p90_edge_distortion": rounded(smaller_p90),
            "larger_component_p10_triangle_area_ratio": rounded(
                larger_chart.get("patch_unwrap_proxy_p10_triangle_area_ratio")
            ),
            "smaller_component_p10_triangle_area_ratio": rounded(
                smaller_chart.get("patch_unwrap_proxy_p10_triangle_area_ratio")
            ),
            "transition_normal_agreement": rounded(normal_agreement),
            "transition_normal_risk": rounded(normal_risk),
            "transition_offset_ratio": rounded(offset_ratio),
            "two_chart_quality": rounded(two_chart_quality),
            "two_chart_quality_cap_compliant": (
                isinstance(two_chart_quality, (int, float)) and two_chart_quality <= max_p90_distortion
            ),
            "global_trial_largest_component_fraction": rounded(
                trial_chart.get("patch_unwrap_proxy_largest_component_node_fraction")
            ),
            "global_trial_p90_edge_distortion": rounded(trial_p90_value),
            "global_trial_p90_cap_gap": rounded(
                trial_p90_value - max_p90_distortion if trial_p90_value is not None else None
            ),
            "global_trial_cap_compliant": (
                trial_p90_value is not None and trial_p90_value <= max_p90_distortion
            ),
            **rigid_trial,
            **two_chart_rigid_trial_gate_metrics(rigid_trial, max_p90_distortion),
            "two_chart_rigid_trial_p90_cap_gap": rounded(
                rigid_p90_value - max_p90_distortion if rigid_p90_value is not None else None
            ),
            "two_chart_rigid_trial_cap_compliant": (
                rigid_p90_value is not None and rigid_p90_value <= max_p90_distortion
            ),
            "global_trial_p10_triangle_area_ratio": rounded(
                trial_chart.get("patch_unwrap_proxy_p10_triangle_area_ratio")
            ),
            **bridge_chart_frame_metrics(nodes, edge, base_roots, chart_frames),
        }
        rows.append(row)

    def quality_key(row: dict[str, Any]) -> tuple[float, float, float]:
        quality = row.get("two_chart_quality")
        trial_gap = row.get("global_trial_p90_cap_gap")
        return (
            float("inf") if quality is None else float(quality),
            float("inf") if trial_gap is None else float(trial_gap),
            -float(row.get("global_trial_largest_component_fraction") or 0.0),
        )

    bridge_rows = [row for row in rows if row.get("component_relation") == "bridge"]
    two_chart_rows = [row for row in bridge_rows if row.get("two_chart_quality_cap_compliant")]
    global_cap_rows = [row for row in bridge_rows if row.get("global_trial_cap_compliant")]
    rigid_cap_rows = [
        row for row in bridge_rows if row.get("two_chart_rigid_trial_cap_compliant")
    ]
    rigid_selection_gate_rows = [
        row for row in bridge_rows if row.get("two_chart_rigid_trial_selection_gate_pass")
    ]
    return {
        "enabled": True,
        "skipped_edge_count": len(skipped_edges),
        "skipped_bridge_edge_count": len(bridge_rows),
        "two_chart_quality_cap_compliant_count": len(two_chart_rows),
        "global_trial_cap_compliant_count": len(global_cap_rows),
        "two_chart_rigid_trial_cap_compliant_count": len(rigid_cap_rows),
        "two_chart_rigid_trial_selection_gate_pass_count": len(
            rigid_selection_gate_rows
        ),
        "two_chart_rigid_trial_placement_class_counts": categorical_counts(
            bridge_rows,
            "two_chart_rigid_trial_placement_class",
        ),
        "best_two_chart_quality": rounded(
            min((float(row["two_chart_quality"]) for row in bridge_rows if row.get("two_chart_quality") is not None), default=float("inf"))
        ),
        "global_trial_p90_summary": numeric_summary(
            [row.get("global_trial_p90_edge_distortion") for row in bridge_rows]
        ),
        "two_chart_rigid_trial_p90_summary": numeric_summary(
            [row.get("two_chart_rigid_trial_p90_edge_distortion") for row in bridge_rows]
        ),
        "two_chart_quality_summary": numeric_summary([row.get("two_chart_quality") for row in bridge_rows]),
        "top_by_two_chart_quality": [
            {"rank": index + 1, **row}
            for index, row in enumerate(sorted(rows, key=quality_key)[:diagnostic_limit])
        ],
    }


def base_cap_boundary_transition_prototype(
    nodes: list[dict[str, Any]],
    accepted_edges: set[tuple[int, int]],
    skipped_edges: list[tuple[int, int]],
    edge_quality: list[dict[str, float | tuple[int, int]]],
    max_p90_distortion: float,
    target: float | None,
    diagnostic_limit: int,
    require_two_chart_selection_gate: bool = False,
) -> dict[str, Any]:
    """Greedily accept skipped bridges as local chart-boundary transitions."""

    if diagnostic_limit < 1:
        raise ValueError("diagnostic_limit must be positive")
    if target is not None and not 0 <= target <= 1:
        raise ValueError("target must be between 0 and 1")

    node_count = len(nodes)
    roots, sizes = component_roots(node_count, accepted_edges)
    base_roots, chart_frames = component_chart_frames(nodes, accepted_edges)
    quality_by_edge = {
        sorted_edge(quality["edge"]): quality
        for quality in edge_quality
        if isinstance(quality.get("edge"), tuple)
    }

    component_edge_cache: dict[int, set[tuple[int, int]]] = {}
    component_chart_cache: dict[int, dict[str, Any]] = {}

    def component_edges(root: int) -> set[tuple[int, int]]:
        cached = component_edge_cache.get(root)
        if cached is not None:
            return cached
        edges = {
            edge
            for edge in accepted_edges
            if roots[edge[0]] == root and roots[edge[1]] == root
        }
        component_edge_cache[root] = edges
        return edges

    def component_chart(root: int) -> dict[str, Any]:
        cached = component_chart_cache.get(root)
        if cached is not None:
            return cached
        chart = ablation.unwrap_proxy_metrics(nodes, sorted(component_edges(root)))
        component_chart_cache[root] = chart
        return chart

    def scalar_max(*values: Any) -> float | None:
        finite = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        return max(finite) if finite else None

    def transition_quality_for(
        larger_p90: Any,
        smaller_p90: Any,
        offset_ratio: Any,
        normal_agreement: Any,
    ) -> tuple[float | None, float | None]:
        normal_risk = (
            1.0 - float(normal_agreement)
            if isinstance(normal_agreement, (int, float)) and math.isfinite(float(normal_agreement))
            else None
        )
        return scalar_max(larger_p90, smaller_p90, offset_ratio, normal_risk), normal_risk

    candidates: list[dict[str, Any]] = []
    cycle_count = 0
    for skipped_edge in skipped_edges:
        edge = sorted_edge(skipped_edge)
        left, right = edge
        left_root = roots[left]
        right_root = roots[right]
        if left_root == right_root:
            cycle_count += 1
            continue

        root_rows = [
            (left_root, sizes.get(left_root, 0), component_chart(left_root)),
            (right_root, sizes.get(right_root, 0), component_chart(right_root)),
        ]
        root_rows.sort(key=lambda item: item[1], reverse=True)
        larger_root, larger_size, larger_chart = root_rows[0]
        smaller_root, smaller_size, smaller_chart = root_rows[1]
        quality = quality_by_edge.get(edge) or {}
        normal_agreement = quality.get("normal_agreement")
        offset_ratio = quality.get("offset_ratio")
        larger_p90 = larger_chart.get("patch_unwrap_proxy_p90_edge_distortion")
        smaller_p90 = smaller_chart.get("patch_unwrap_proxy_p90_edge_distortion")
        transition_quality, normal_risk = transition_quality_for(
            larger_p90,
            smaller_p90,
            offset_ratio,
            normal_agreement,
        )
        trial_edges = set(accepted_edges)
        trial_edges.add(edge)
        trial_chart = ablation.unwrap_proxy_metrics(nodes, sorted(trial_edges))
        trial_p90 = trial_chart.get("patch_unwrap_proxy_p90_edge_distortion")
        trial_p90_value = float(trial_p90) if isinstance(trial_p90, (int, float)) else None
        trial_fraction = trial_chart.get("patch_unwrap_proxy_largest_component_node_fraction")
        trial_fraction_value = (
            float(trial_fraction) if isinstance(trial_fraction, (int, float)) else None
        )
        normal_guided_trial = normal_guided_unwrap_proxy_metrics(nodes, sorted(trial_edges))
        normal_guided_p90 = normal_guided_trial.get(
            "normal_guided_global_p90_edge_distortion"
        )
        normal_guided_p90_value = (
            float(normal_guided_p90) if isinstance(normal_guided_p90, (int, float)) else None
        )
        rigid_trial = two_chart_rigid_trial_metrics(
            nodes,
            accepted_edges,
            base_roots,
            sizes,
            edge,
            chart_frames,
        )
        rigid_p90 = rigid_trial.get("two_chart_rigid_trial_p90_edge_distortion")
        rigid_p90_value = float(rigid_p90) if isinstance(rigid_p90, (int, float)) else None
        public_row = {
            "component_relation": "boundary_transition",
            "larger_component_fraction": rounded(larger_size / node_count if node_count else 0.0),
            "smaller_component_fraction": rounded(smaller_size / node_count if node_count else 0.0),
            "larger_component_edge_count": len(component_edges(larger_root)),
            "smaller_component_edge_count": len(component_edges(smaller_root)),
            "larger_component_p90_edge_distortion": rounded(larger_p90),
            "smaller_component_p90_edge_distortion": rounded(smaller_p90),
            "transition_normal_agreement": rounded(normal_agreement),
            "transition_normal_risk": rounded(normal_risk),
            "transition_offset_ratio": rounded(offset_ratio),
            "boundary_transition_quality": rounded(transition_quality),
            "boundary_transition_cap_compliant": (
                isinstance(transition_quality, (int, float))
                and transition_quality <= max_p90_distortion
            ),
            "direct_global_trial_largest_component_fraction": rounded(trial_fraction_value),
            "direct_global_trial_p90_edge_distortion": rounded(trial_p90_value),
            "direct_global_trial_p90_cap_gap": rounded(
                trial_p90_value - max_p90_distortion if trial_p90_value is not None else None
            ),
            "direct_global_trial_cap_compliant": (
                trial_p90_value is not None and trial_p90_value <= max_p90_distortion
            ),
            "normal_guided_global_trial_p90_edge_distortion": rounded(normal_guided_p90_value),
            "normal_guided_global_trial_p90_cap_gap": rounded(
                normal_guided_p90_value - max_p90_distortion
                if normal_guided_p90_value is not None
                else None
            ),
            "normal_guided_global_trial_cap_compliant": (
                normal_guided_p90_value is not None
                and normal_guided_p90_value <= max_p90_distortion
            ),
            "normal_guided_global_trial_p90_delta_vs_direct": rounded(
                normal_guided_p90_value - trial_p90_value
                if normal_guided_p90_value is not None and trial_p90_value is not None
                else None
            ),
            "normal_guided_global_chart_normal_agreement": normal_guided_trial.get(
                "normal_guided_chart_normal_agreement"
            ),
            **rigid_trial,
            **two_chart_rigid_trial_gate_metrics(rigid_trial, max_p90_distortion),
            "two_chart_rigid_trial_p90_cap_gap": rounded(
                rigid_p90_value - max_p90_distortion if rigid_p90_value is not None else None
            ),
            "two_chart_rigid_trial_cap_compliant": (
                rigid_p90_value is not None and rigid_p90_value <= max_p90_distortion
            ),
            "two_chart_rigid_trial_p90_delta_vs_direct": rounded(
                rigid_p90_value - trial_p90_value
                if rigid_p90_value is not None and trial_p90_value is not None
                else None
            ),
            **bridge_chart_frame_metrics(nodes, edge, base_roots, chart_frames),
        }
        candidates.append(
            {
                "edge": edge,
                "left_root": left_root,
                "right_root": right_root,
                "normal_agreement": normal_agreement,
                "offset_ratio": offset_ratio,
                "normal_risk": normal_risk,
                "transition_quality": transition_quality,
                "public": public_row,
            }
        )

    component_roots_sorted = sorted(sizes)
    root_to_index = {root: index for index, root in enumerate(component_roots_sorted)}
    parent = list(range(len(component_roots_sorted)))
    coverage = [sizes[root] for root in component_roots_sorted]

    def boundary_find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def boundary_union(left: int, right: int) -> bool:
        left_root = boundary_find(left)
        right_root = boundary_find(right)
        if left_root == right_root:
            return False
        if coverage[left_root] < coverage[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        coverage[left_root] += coverage[right_root]
        return True

    def boundary_fraction() -> float:
        if not node_count:
            return 0.0
        return max(
            (coverage[boundary_find(index)] for index in range(len(coverage))),
            default=0,
        ) / node_count

    def target_reached() -> bool:
        return target is not None and boundary_fraction() >= target

    def candidate_quality_sort(candidate: dict[str, Any]) -> float:
        quality = candidate.get("transition_quality")
        return float("inf") if quality is None else float(quality)

    remaining = list(candidates)
    accepted_edges_internal: set[tuple[int, int]] = set()
    accepted_qualities: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    stop_reason = "target_already_reached" if target_reached() else "no_candidates"
    while not target_reached() and remaining:
        current_fraction = boundary_fraction()
        best_index = None
        best_score = None
        best_after_fraction = None
        best_left_fraction = None
        best_right_fraction = None
        bridge_candidate_count = 0
        eligible_candidate_count = 0
        for candidate_index, candidate in enumerate(remaining):
            left_index = root_to_index[candidate["left_root"]]
            right_index = root_to_index[candidate["right_root"]]
            left_boundary_root = boundary_find(left_index)
            right_boundary_root = boundary_find(right_index)
            if left_boundary_root == right_boundary_root:
                continue
            bridge_candidate_count += 1
            public = candidate["public"]
            if not public["boundary_transition_cap_compliant"]:
                continue
            if (
                require_two_chart_selection_gate
                and not public["two_chart_rigid_trial_selection_gate_pass"]
            ):
                continue
            eligible_candidate_count += 1
            merged_fraction = (
                coverage[left_boundary_root] + coverage[right_boundary_root]
            ) / node_count if node_count else 0.0
            after_fraction = max(current_fraction, merged_fraction)
            score = (
                -after_fraction,
                candidate_quality_sort(candidate),
                -float(candidate.get("normal_agreement") or 0.0),
                float(candidate.get("offset_ratio") or 0.0),
                candidate_index,
            )
            if best_score is None or score < best_score:
                best_index = candidate_index
                best_score = score
                best_after_fraction = after_fraction
                best_left_fraction = coverage[left_boundary_root] / node_count if node_count else 0.0
                best_right_fraction = coverage[right_boundary_root] / node_count if node_count else 0.0
        if best_index is None:
            stop_reason = "no_cap_compliant_boundary_transition"
            break

        chosen = remaining.pop(best_index)
        boundary_union(
            root_to_index[chosen["left_root"]],
            root_to_index[chosen["right_root"]],
        )
        accepted_edges_internal.add(chosen["edge"])
        accepted_qualities.append(chosen)
        boundary_edges = set(accepted_edges)
        boundary_edges.update(accepted_edges_internal)
        local_metrics = component_local_chart_metrics(nodes, boundary_edges, accepted_edges, edge_quality)
        accepted_normal_risks = [candidate["normal_risk"] for candidate in accepted_qualities]
        normal_risk_p90 = percentile_from_sorted(
            sorted(float(value) for value in accepted_normal_risks if isinstance(value, (int, float))),
            90,
        )
        boundary_quality_after = scalar_max(
            local_metrics["local_chart_p90_internal_edge_distortion"],
            local_metrics["local_chart_p90_bridge_offset_ratio"],
            normal_risk_p90,
        )
        steps.append(
            {
                "step": len(steps) + 1,
                "boundary_candidate_count_before": bridge_candidate_count,
                "eligible_boundary_candidate_count_before": eligible_candidate_count,
                "atlas_largest_component_fraction_before": rounded(current_fraction),
                "atlas_largest_component_fraction_after": rounded(
                    best_after_fraction if best_after_fraction is not None else boundary_fraction()
                ),
                "coverage_delta": rounded(
                    (
                        best_after_fraction
                        if best_after_fraction is not None
                        else boundary_fraction()
                    )
                    - current_fraction
                ),
                "left_boundary_component_fraction_before": rounded(best_left_fraction),
                "right_boundary_component_fraction_before": rounded(best_right_fraction),
                "transition_normal_agreement": rounded(chosen["normal_agreement"]),
                "transition_normal_risk": rounded(chosen["normal_risk"]),
                "transition_offset_ratio": rounded(chosen["offset_ratio"]),
                "boundary_transition_quality": chosen["public"]["boundary_transition_quality"],
                "boundary_local_quality_p90_after": rounded(boundary_quality_after),
                "boundary_p90_transition_normal_risk_after": rounded(normal_risk_p90),
                "boundary_p90_transition_offset_ratio_after": local_metrics[
                    "local_chart_p90_bridge_offset_ratio"
                ],
                "direct_global_trial_p90_edge_distortion": chosen["public"][
                    "direct_global_trial_p90_edge_distortion"
                ],
                "direct_global_trial_p90_cap_gap": chosen["public"][
                    "direct_global_trial_p90_cap_gap"
                ],
                "direct_global_trial_cap_compliant": chosen["public"][
                    "direct_global_trial_cap_compliant"
                ],
                "two_chart_rigid_trial_p90_edge_distortion": chosen["public"][
                    "two_chart_rigid_trial_p90_edge_distortion"
                ],
                "two_chart_rigid_trial_p90_cap_gap": chosen["public"][
                    "two_chart_rigid_trial_p90_cap_gap"
                ],
                "two_chart_rigid_trial_cap_compliant": chosen["public"][
                    "two_chart_rigid_trial_cap_compliant"
                ],
            }
        )
        stop_reason = "target_reached" if target_reached() else "continuing"

    if not target_reached() and not remaining and stop_reason == "continuing":
        stop_reason = "no_candidates"

    final_boundary_edges = set(accepted_edges)
    final_boundary_edges.update(accepted_edges_internal)
    final_local_metrics = component_local_chart_metrics(
        nodes,
        final_boundary_edges,
        accepted_edges,
        edge_quality,
    )
    final_normal_risk_p90 = percentile_from_sorted(
        sorted(
            float(candidate["normal_risk"])
            for candidate in accepted_qualities
            if isinstance(candidate.get("normal_risk"), (int, float))
        ),
        90,
    )
    final_quality_p90 = scalar_max(
        final_local_metrics["local_chart_p90_internal_edge_distortion"],
        final_local_metrics["local_chart_p90_bridge_offset_ratio"],
        final_normal_risk_p90,
    )
    cap_candidates = [
        candidate for candidate in candidates if candidate["public"]["boundary_transition_cap_compliant"]
    ]
    direct_global_cap_candidates = [
        candidate for candidate in candidates if candidate["public"]["direct_global_trial_cap_compliant"]
    ]
    normal_guided_cap_candidates = [
        candidate
        for candidate in candidates
        if candidate["public"]["normal_guided_global_trial_cap_compliant"]
    ]
    rigid_cap_candidates = [
        candidate
        for candidate in candidates
        if candidate["public"]["two_chart_rigid_trial_cap_compliant"]
    ]
    rigid_selection_gate_candidates = [
        candidate
        for candidate in candidates
        if candidate["public"]["two_chart_rigid_trial_selection_gate_pass"]
    ]
    public_rows = [candidate["public"] for candidate in candidates]

    def quality_key(row: dict[str, Any]) -> tuple[float, float, float]:
        quality = row.get("boundary_transition_quality")
        gap = row.get("direct_global_trial_p90_cap_gap")
        return (
            float("inf") if quality is None else float(quality),
            float("inf") if gap is None else float(gap),
            -float(row.get("direct_global_trial_largest_component_fraction") or 0.0),
        )

    local_cap_global_blocked = [
        row
        for row in public_rows
        if row.get("boundary_transition_cap_compliant") is True
        and row.get("direct_global_trial_cap_compliant") is False
        and isinstance(row.get("direct_global_trial_p90_cap_gap"), (int, float))
        and float(row["direct_global_trial_p90_cap_gap"]) > 0
    ]
    frame_degenerate = [
        row
        for row in public_rows
        if isinstance(row.get("chart_normal_agreement"), (int, float))
        and isinstance(row.get("chart_tangent_singular_min"), (int, float))
        and float(row["chart_normal_agreement"]) < 0.02
        and float(row["chart_tangent_singular_min"]) < 0.02
    ]
    transition_normal_ok = [
        row
        for row in public_rows
        if isinstance(row.get("transition_normal_agreement"), (int, float))
        and float(row["transition_normal_agreement"]) >= 0.8
    ]
    local_cap_global_blocked_frame_degenerate = [
        row
        for row in local_cap_global_blocked
        if row in frame_degenerate
    ]
    normal_guided_resolved_blockers = [
        row
        for row in local_cap_global_blocked
        if row.get("normal_guided_global_trial_cap_compliant") is True
    ]
    rigid_resolved_blockers = [
        row
        for row in local_cap_global_blocked
        if row.get("two_chart_rigid_trial_cap_compliant") is True
    ]

    return {
        "enabled": True,
        "method": (
            "greedy_local_chart_boundary_transition_two_chart_gate"
            if require_two_chart_selection_gate
            else "greedy_local_chart_boundary_transition"
        ),
        "requires_two_chart_rigid_trial_selection_gate": bool(
            require_two_chart_selection_gate
        ),
        "target": rounded(target),
        "skipped_edge_count": len(skipped_edges),
        "skipped_bridge_edge_count": len(candidates),
        "skipped_cycle_edge_count": cycle_count,
        "base_component_count": len(component_roots_sorted),
        "base_largest_component_fraction": rounded(
            max(sizes.values()) / node_count if sizes and node_count else 0.0
        ),
        "boundary_transition_candidate_count": len(candidates),
        "cap_compliant_boundary_transition_candidate_count": len(cap_candidates),
        "direct_global_trial_cap_compliant_count": len(direct_global_cap_candidates),
        "normal_guided_global_trial_cap_compliant_count": len(normal_guided_cap_candidates),
        "two_chart_rigid_trial_cap_compliant_count": len(rigid_cap_candidates),
        "two_chart_rigid_trial_selection_gate_pass_count": len(
            rigid_selection_gate_candidates
        ),
        "two_chart_rigid_trial_placement_class_counts": categorical_counts(
            public_rows,
            "two_chart_rigid_trial_placement_class",
        ),
        "accepted_boundary_transition_count": len(accepted_qualities),
        "final_boundary_largest_component_fraction": rounded(boundary_fraction()),
        "reaches_target": bool(target_reached()),
        "stop_reason": stop_reason if not target_reached() else "target_reached",
        "final_boundary_local_quality_p90": rounded(final_quality_p90),
        "final_boundary_p90_transition_normal_risk": rounded(final_normal_risk_p90),
        "final_boundary_p90_transition_offset_ratio": final_local_metrics[
            "local_chart_p90_bridge_offset_ratio"
        ],
        "boundary_transition_quality_summary": numeric_summary(
            [row.get("boundary_transition_quality") for row in public_rows]
        ),
        "accepted_boundary_transition_quality_summary": numeric_summary(
            [
                candidate["public"].get("boundary_transition_quality")
                for candidate in accepted_qualities
            ]
        ),
        "direct_global_trial_p90_summary": numeric_summary(
            [row.get("direct_global_trial_p90_edge_distortion") for row in public_rows]
        ),
        "normal_guided_global_trial_p90_summary": numeric_summary(
            [row.get("normal_guided_global_trial_p90_edge_distortion") for row in public_rows]
        ),
        "two_chart_rigid_trial_p90_summary": numeric_summary(
            [row.get("two_chart_rigid_trial_p90_edge_distortion") for row in public_rows]
        ),
        "global_reconciliation_blocker_summary": {
            "candidate_count": len(public_rows),
            "local_cap_global_blocked_count": len(local_cap_global_blocked),
            "frame_degenerate_count": len(frame_degenerate),
            "transition_normal_ok_count": len(transition_normal_ok),
            "local_cap_global_blocked_frame_degenerate_count": len(
                local_cap_global_blocked_frame_degenerate
            ),
            "normal_guided_resolved_blocker_count": len(normal_guided_resolved_blockers),
            "two_chart_rigid_resolved_blocker_count": len(rigid_resolved_blockers),
            "direct_global_p90_cap_gap_summary": numeric_summary(
                [row.get("direct_global_trial_p90_cap_gap") for row in local_cap_global_blocked]
            ),
            "normal_guided_p90_delta_vs_direct_summary": numeric_summary(
                [
                    row.get("normal_guided_global_trial_p90_delta_vs_direct")
                    for row in local_cap_global_blocked
                ]
            ),
            "two_chart_rigid_p90_delta_vs_direct_summary": numeric_summary(
                [
                    row.get("two_chart_rigid_trial_p90_delta_vs_direct")
                    for row in local_cap_global_blocked
                ]
            ),
            "chart_normal_agreement_summary": numeric_summary(
                [row.get("chart_normal_agreement") for row in local_cap_global_blocked]
            ),
            "chart_tangent_singular_min_summary": numeric_summary(
                [row.get("chart_tangent_singular_min") for row in local_cap_global_blocked]
            ),
        },
        "top_by_boundary_quality": [
            {"rank": index + 1, **row}
            for index, row in enumerate(sorted(public_rows, key=quality_key)[:diagnostic_limit])
        ],
        "accepted_steps": steps,
        "storage_policy": (
            "Scalar summaries only; no raw chunks, point clouds, component IDs, bridge endpoints, "
            "edge IDs, path signatures, coordinates, meshes, predictions, letters, or titles."
        ),
    }


def candidate_bridge_diagnostics(
    nodes: list[dict[str, Any]],
    all_edges: list[tuple[int, int]],
    base_edges: set[tuple[int, int]],
    edge_quality: list[dict[str, float | tuple[int, int]]],
    max_p90_distortion: float | None = None,
    target: float | None = None,
    limit: int = 24,
) -> dict[str, Any]:
    """Rank bridge candidates by scalar trial-chart metrics without saving edge tables."""

    if limit < 1:
        raise ValueError("limit must be positive")
    node_count = len(nodes)
    base_chart = ablation.unwrap_proxy_metrics(nodes, sorted(base_edges))
    base_fraction = float(base_chart["patch_unwrap_proxy_largest_component_node_fraction"] or 0.0)
    roots, sizes = component_roots(node_count, base_edges)
    largest_root = max(sizes, key=sizes.get) if sizes else None
    base_roots, chart_frames = component_chart_frames(nodes, base_edges)

    rows: list[dict[str, Any]] = []
    for quality in candidate_edges(all_edges, base_edges, edge_quality):
        edge = sorted_edge(quality["edge"])
        left, right = edge
        if roots[left] == roots[right]:
            continue
        trial_edges = set(base_edges)
        trial_edges.add(edge)
        trial_chart = ablation.unwrap_proxy_metrics(nodes, sorted(trial_edges))
        p90 = trial_chart["patch_unwrap_proxy_p90_edge_distortion"]
        trial_fraction = float(trial_chart["patch_unwrap_proxy_largest_component_node_fraction"] or 0.0)
        cap_compliant = max_p90_distortion is None or (
            p90 is not None and float(p90) <= max_p90_distortion
        )
        rows.append(
            {
                "normal_agreement": rounded(quality["normal_agreement"]),
                "offset_ratio": rounded(quality["offset_ratio"]),
                "trial_largest_component_fraction": rounded(trial_fraction),
                "coverage_delta": rounded(trial_fraction - base_fraction),
                "trial_p90_edge_distortion": rounded(p90),
                "trial_p10_triangle_area_ratio": rounded(
                    trial_chart["patch_unwrap_proxy_p10_triangle_area_ratio"]
                ),
                "cap_compliant": bool(cap_compliant),
                "touches_base_largest_component": bool(
                    largest_root is not None and (roots[left] == largest_root or roots[right] == largest_root)
                ),
                **bridge_chart_frame_metrics(nodes, edge, base_roots, chart_frames),
            }
        )

    def p90_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        p90 = row["trial_p90_edge_distortion"]
        return (
            float("inf") if p90 is None else float(p90),
            -float(row["trial_largest_component_fraction"]),
            -float(row["normal_agreement"]),
            float(row["offset_ratio"]),
        )

    def coverage_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        p90 = row["trial_p90_edge_distortion"]
        return (
            -float(row["trial_largest_component_fraction"]),
            float("inf") if p90 is None else float(p90),
            -float(row["normal_agreement"]),
            float(row["offset_ratio"]),
        )

    def ranked(rows_in_order: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"rank": index + 1, **row} for index, row in enumerate(rows_in_order[:limit])]

    cap_rows = [row for row in rows if row["cap_compliant"]]
    bridge_to_largest = [row for row in rows if row["touches_base_largest_component"]]
    cap_bridge_to_largest = [row for row in bridge_to_largest if row["cap_compliant"]]
    chart_rows = [row for row in rows if row["chart_pair_available"]]
    chart_ge_001 = [
        row
        for row in chart_rows
        if row["chart_normal_agreement"] is not None and float(row["chart_normal_agreement"]) >= 0.01
    ]
    chart_ge_050 = [
        row
        for row in chart_rows
        if row["chart_normal_agreement"] is not None and float(row["chart_normal_agreement"]) >= 0.50
    ]
    return {
        "base_largest_component_fraction": rounded(base_fraction),
        "max_p90_distortion": rounded(max_p90_distortion),
        "bridge_candidate_count": len(rows),
        "cap_compliant_bridge_candidate_count": len(cap_rows),
        "touches_base_largest_component_count": len(bridge_to_largest),
        "cap_compliant_touches_base_largest_component_count": len(cap_bridge_to_largest),
        "chart_pair_available_count": len(chart_rows),
        "chart_normal_ge_001_count": len(chart_ge_001),
        "chart_normal_ge_050_count": len(chart_ge_050),
        "best_cap_compliant_trial_fraction": rounded(
            max((float(row["trial_largest_component_fraction"]) for row in cap_rows), default=0.0)
        ),
        "best_cap_compliant_coverage_delta": rounded(
            max((float(row["coverage_delta"]) for row in cap_rows), default=0.0)
        ),
        "best_chart_normal_ge_001_trial_fraction": rounded(
            max((float(row["trial_largest_component_fraction"]) for row in chart_ge_001), default=0.0)
        ),
        "best_chart_normal_ge_050_trial_fraction": rounded(
            max((float(row["trial_largest_component_fraction"]) for row in chart_ge_050), default=0.0)
        ),
        "top_coverage_normal_projection_policy": broad_candidate_policy_summary(
            rows,
            target,
            limit,
        ),
        "top_by_low_p90": ranked(sorted(rows, key=p90_key)),
        "top_by_coverage": ranked(sorted(rows, key=coverage_key)),
    }


def chart_transition_budget(
    nodes: list[dict[str, Any]],
    all_edges: list[tuple[int, int]],
    base_edges: set[tuple[int, int]],
    edge_quality: list[dict[str, float | tuple[int, int]]],
    max_p90_distortion: float | None,
    rotation_thresholds: tuple[float, ...] = (15.0, 45.0, 75.0, 90.0),
) -> dict[str, Any]:
    """Aggregate component-level transition coverage without exporting component IDs."""

    node_count = len(nodes)
    roots, sizes = component_roots(node_count, base_edges)
    base_roots, chart_frames = component_chart_frames(nodes, base_edges)
    component_sizes = {root: sizes[root] for root in sorted(sizes)}
    root_to_index = {root: index for index, root in enumerate(component_sizes)}
    candidates: list[dict[str, Any]] = []

    for quality in candidate_edges(all_edges, base_edges, edge_quality):
        edge = sorted_edge(quality["edge"])
        left, right = edge
        left_root = roots[left]
        right_root = roots[right]
        if left_root == right_root:
            continue
        trial_edges = set(base_edges)
        trial_edges.add(edge)
        trial_chart = ablation.unwrap_proxy_metrics(nodes, sorted(trial_edges))
        p90 = trial_chart["patch_unwrap_proxy_p90_edge_distortion"]
        chart_metrics = bridge_chart_frame_metrics(nodes, edge, base_roots, chart_frames)
        chart_normal = chart_metrics.get("chart_normal_agreement")
        rotation_degrees = None
        if isinstance(chart_normal, (int, float)):
            clipped = min(1.0, max(0.0, float(chart_normal)))
            rotation_degrees = math.degrees(math.acos(clipped))
        candidates.append(
            {
                "left_root": left_root,
                "right_root": right_root,
                "p90": p90,
                "cap_compliant": (
                    max_p90_distortion is None
                    or (isinstance(p90, (int, float)) and float(p90) <= max_p90_distortion)
                ),
                "rotation_degrees": rotation_degrees,
                "chart_pair_available": bool(chart_metrics.get("chart_pair_available")),
            }
        )

    def summarize(threshold: float | None, require_cap: bool) -> dict[str, Any]:
        parent = list(range(len(root_to_index)))
        coverage = [component_sizes[root] for root in component_sizes]

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> bool:
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                return False
            if coverage[left_root] < coverage[right_root]:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root
            coverage[left_root] += coverage[right_root]
            return True

        eligible = []
        for candidate in candidates:
            if require_cap and not candidate["cap_compliant"]:
                continue
            rotation = candidate["rotation_degrees"]
            if threshold is not None and not isinstance(rotation, (int, float)):
                continue
            if threshold is not None and float(rotation) > threshold:
                continue
            eligible.append(candidate)

        unique_pairs: set[tuple[int, int]] = set()
        cycle_edge_count = 0
        for candidate in eligible:
            left = root_to_index[candidate["left_root"]]
            right = root_to_index[candidate["right_root"]]
            pair = tuple(sorted((left, right)))
            if pair in unique_pairs:
                cycle_edge_count += 1
                continue
            unique_pairs.add(pair)
            if not union(left, right):
                cycle_edge_count += 1
        largest = max((coverage[find(index)] for index in range(len(parent))), default=0)
        rotations = [
            float(candidate["rotation_degrees"])
            for candidate in eligible
            if isinstance(candidate.get("rotation_degrees"), (int, float))
        ]
        p90_values = [
            float(candidate["p90"])
            for candidate in eligible
            if isinstance(candidate.get("p90"), (int, float))
        ]
        return {
            "rotation_threshold_degrees": rounded(threshold),
            "require_p90_cap": require_cap,
            "eligible_bridge_count": len(eligible),
            "eligible_component_pair_count": len(unique_pairs),
            "cycle_edge_count": cycle_edge_count,
            "largest_component_fraction": rounded(largest / node_count if node_count else 0.0),
            "reaches_75": largest / node_count >= 0.75 if node_count else False,
            "reaches_90": largest / node_count >= 0.90 if node_count else False,
            "max_rotation_degrees": rounded(max(rotations) if rotations else None),
            "max_p90_edge_distortion": rounded(max(p90_values) if p90_values else None),
        }

    def two_stage_summarize(threshold: float) -> dict[str, Any]:
        parent = list(range(len(root_to_index)))
        coverage = [component_sizes[root] for root in component_sizes]

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> bool:
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                return False
            if coverage[left_root] < coverage[right_root]:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root
            coverage[left_root] += coverage[right_root]
            return True

        def fraction() -> float:
            largest = max((coverage[find(index)] for index in range(len(parent))), default=0)
            return largest / node_count if node_count else 0.0

        def largest_index() -> int | None:
            if not parent:
                return None
            return max(range(len(parent)), key=lambda index: coverage[find(index)])

        def component_pair(candidate: dict[str, Any]) -> tuple[int, int]:
            return (
                root_to_index[candidate["left_root"]],
                root_to_index[candidate["right_root"]],
            )

        def is_stage1(candidate: dict[str, Any]) -> bool:
            rotation = candidate["rotation_degrees"]
            return (
                bool(candidate["cap_compliant"])
                and isinstance(rotation, (int, float))
                and float(rotation) <= threshold
            )

        stage1_pairs: set[tuple[int, int]] = set()
        stage1_cycle_edges = 0
        for candidate in candidates:
            if not is_stage1(candidate):
                continue
            left, right = component_pair(candidate)
            pair = tuple(sorted((left, right)))
            if pair in stage1_pairs:
                stage1_cycle_edges += 1
                continue
            stage1_pairs.add(pair)
            if not union(left, right):
                stage1_cycle_edges += 1

        stage1_fraction = fraction()
        anchor = largest_index()
        frame_missing_edges = 0
        frame_missing_pairs: set[tuple[int, int]] = set()
        frame_missing_p90: list[float] = []

        def attach_cleanup(
            predicate,
            used_pairs: set[tuple[int, int]],
        ) -> tuple[int, list[float], list[float]]:
            nonlocal anchor
            edge_count = 0
            p90_values: list[float] = []
            rotations: list[float] = []
            if anchor is None:
                return edge_count, p90_values, rotations
            changed = True
            while changed:
                changed = False
                anchor_root = find(anchor)
                for candidate in candidates:
                    if not bool(candidate["cap_compliant"]) or is_stage1(candidate):
                        continue
                    if not predicate(candidate):
                        continue
                    left, right = component_pair(candidate)
                    if find(left) == find(right):
                        continue
                    if find(left) != anchor_root and find(right) != anchor_root:
                        continue
                    pair = tuple(sorted((left, right)))
                    if pair in used_pairs:
                        continue
                    used_pairs.add(pair)
                    if union(left, right):
                        anchor = find(anchor)
                        anchor_root = find(anchor)
                        edge_count += 1
                        changed = True
                        p90 = candidate.get("p90")
                        if isinstance(p90, (int, float)):
                            p90_values.append(float(p90))
                        rotation = candidate.get("rotation_degrees")
                        if isinstance(rotation, (int, float)):
                            rotations.append(float(rotation))
            return edge_count, p90_values, rotations

        frame_missing_edges, frame_missing_p90, _frame_missing_rotations = attach_cleanup(
            lambda candidate: not isinstance(candidate.get("rotation_degrees"), (int, float)),
            frame_missing_pairs,
        )
        frame_missing_fraction = fraction()
        high_rotation_pairs: set[tuple[int, int]] = set()
        high_rotation_edges, high_rotation_p90, high_rotation_values = attach_cleanup(
            lambda candidate: (
                isinstance(candidate.get("rotation_degrees"), (int, float))
                and float(candidate["rotation_degrees"]) > threshold
            ),
            high_rotation_pairs,
        )
        final_fraction = fraction()
        cleanup_p90_values = [*frame_missing_p90, *high_rotation_p90]
        return {
            "rotation_threshold_degrees": rounded(threshold),
            "stage1_component_pair_count": len(stage1_pairs),
            "stage1_cycle_edge_count": stage1_cycle_edges,
            "stage1_largest_component_fraction": rounded(stage1_fraction),
            "frame_missing_cleanup_edge_count": frame_missing_edges,
            "after_frame_missing_cleanup_fraction": rounded(frame_missing_fraction),
            "high_rotation_cleanup_edge_count": high_rotation_edges,
            "after_all_p90_cleanup_fraction": rounded(final_fraction),
            "reaches_75_after_frame_missing_cleanup": frame_missing_fraction >= 0.75 if node_count else False,
            "reaches_90_after_frame_missing_cleanup": frame_missing_fraction >= 0.90 if node_count else False,
            "reaches_75_after_all_p90_cleanup": final_fraction >= 0.75 if node_count else False,
            "reaches_90_after_all_p90_cleanup": final_fraction >= 0.90 if node_count else False,
            "max_high_rotation_cleanup_degrees": rounded(
                max(high_rotation_values) if high_rotation_values else None
            ),
            "max_cleanup_p90_edge_distortion": rounded(
                max(cleanup_p90_values) if cleanup_p90_values else None
            ),
        }

    rows = [summarize(None, require_cap=True)]
    rows.extend(summarize(threshold, require_cap=True) for threshold in rotation_thresholds)
    rows.extend(summarize(threshold, require_cap=False) for threshold in rotation_thresholds)
    return {
        "base_component_count": len(component_sizes),
        "bridge_candidate_count": len(candidates),
        "chart_pair_available_count": sum(1 for candidate in candidates if candidate["chart_pair_available"]),
        "max_p90_distortion": rounded(max_p90_distortion),
        "rows": rows,
        "two_stage_rows": [two_stage_summarize(threshold) for threshold in rotation_thresholds],
    }


def component_local_chart_metrics(
    nodes: list[dict[str, Any]],
    grown_edges: set[tuple[int, int]],
    base_edges: set[tuple[int, int]],
    edge_quality: list[dict[str, float | tuple[int, int]]],
) -> dict[str, Any]:
    """Measure a multi-chart atlas proxy anchored to original pruned components."""

    import numpy as np

    node_count = len(nodes)
    if node_count == 0:
        return {
            "local_chart_largest_component_fraction": 0.0,
            "local_chart_p90_internal_edge_distortion": None,
            "local_chart_bridge_edge_count": 0,
            "local_chart_p90_bridge_offset_ratio": None,
            "local_chart_p10_bridge_normal_agreement": None,
            "local_chart_quality_p90": None,
        }

    base_roots, _base_sizes = component_roots(node_count, base_edges)
    edge_quality_by_edge = {
        sorted_edge(quality["edge"]): quality
        for quality in edge_quality
        if isinstance(quality.get("edge"), tuple)
    }

    component_nodes: dict[int, list[int]] = defaultdict(list)
    for node_index, root in enumerate(base_roots):
        component_nodes[root].append(node_index)

    internal_distortions: list[float] = []
    for root, component in component_nodes.items():
        component_set = set(component)
        component_edges = sorted(
            edge
            for edge in grown_edges
            if edge[0] in component_set and edge[1] in component_set
        )
        if len(component) < 3 or not component_edges:
            continue
        centroids = np.array([nodes[index]["centroid"] for index in component], dtype=np.float64)
        centered = centroids - centroids.mean(axis=0)
        covariance = centered.T @ centered / max(1, centered.shape[0])
        _eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        chart_coords = centered @ eigenvectors[:, -2:]
        local_index = {node_index: index for index, node_index in enumerate(component)}
        for left, right in component_edges:
            left_local = local_index[left]
            right_local = local_index[right]
            length_3d = float(np.linalg.norm(centroids[right_local] - centroids[left_local]))
            if length_3d <= 1e-9:
                continue
            length_2d = float(np.linalg.norm(chart_coords[right_local] - chart_coords[left_local]))
            internal_distortions.append(abs(length_2d / length_3d - 1.0))

    bridge_offsets: list[float] = []
    bridge_normals: list[float] = []
    for edge in sorted(grown_edges):
        left, right = edge
        if base_roots[left] == base_roots[right]:
            continue
        quality = edge_quality_by_edge.get(edge)
        if quality is None:
            continue
        bridge_offsets.append(float(quality["offset_ratio"]))
        bridge_normals.append(float(quality["normal_agreement"]))

    def percentile_or_none(values: list[float], percentile: float) -> float | None:
        return float(np.percentile(np.array(values, dtype=np.float64), percentile)) if values else None

    internal_p90 = percentile_or_none(internal_distortions, 90)
    bridge_offset_p90 = percentile_or_none(bridge_offsets, 90)
    quality_values = [
        value for value in (internal_p90, bridge_offset_p90) if isinstance(value, (int, float))
    ]
    return {
        "local_chart_largest_component_fraction": rounded(largest_fraction(node_count, grown_edges)),
        "local_chart_p90_internal_edge_distortion": rounded(internal_p90),
        "local_chart_bridge_edge_count": len(bridge_offsets),
        "local_chart_p90_bridge_offset_ratio": rounded(bridge_offset_p90),
        "local_chart_p10_bridge_normal_agreement": rounded(percentile_or_none(bridge_normals, 10)),
        "local_chart_quality_p90": rounded(max(quality_values) if quality_values else None),
    }


def component_chart_frames(
    nodes: list[dict[str, Any]],
    base_edges: set[tuple[int, int]],
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    """Build PCA frames for original pruned components without exporting geometry."""

    import numpy as np

    node_count = len(nodes)
    base_roots, _base_sizes = component_roots(node_count, base_edges)
    component_nodes: dict[int, list[int]] = defaultdict(list)
    for node_index, root in enumerate(base_roots):
        component_nodes[root].append(node_index)

    frames: dict[int, dict[str, Any]] = {}
    for root, component in component_nodes.items():
        if len(component) < 3:
            continue
        centroids = np.array([nodes[index]["centroid"] for index in component], dtype=np.float64)
        centered = centroids - centroids.mean(axis=0)
        covariance = centered.T @ centered / max(1, centered.shape[0])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, 0]
        node_normals = np.array([nodes[index]["normal"] for index in component], dtype=np.float64)
        normal_lengths = np.linalg.norm(node_normals, axis=1)
        valid_normals = normal_lengths > 1e-6
        if np.any(valid_normals):
            mean_normal = np.mean(node_normals[valid_normals] / normal_lengths[valid_normals, None], axis=0)
            if np.dot(normal, mean_normal) < 0:
                normal = -normal
        frames[root] = {
            "centroid": centroids.mean(axis=0),
            "axes": eigenvectors[:, -2:],
            "normal": normal,
            "node_count": len(component),
            "planarity_ratio": (
                float(max(0.0, eigenvalues[0]) / max(float(np.sum(np.maximum(eigenvalues, 0.0))), 1e-12))
            ),
        }
    return base_roots, frames


def bridge_chart_frame_metrics(
    nodes: list[dict[str, Any]],
    edge: tuple[int, int],
    base_roots: list[int],
    frames: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Return scalar-only compatibility metrics for the two local charts joined by an edge."""

    import numpy as np

    left, right = edge
    left_frame = frames.get(base_roots[left])
    right_frame = frames.get(base_roots[right])
    if left_frame is None or right_frame is None:
        return {
            "chart_pair_available": False,
            "chart_normal_agreement": None,
            "chart_tangent_singular_min": None,
            "chart_tangent_singular_max": None,
            "bridge_projection_ratio_left": None,
            "bridge_projection_ratio_right": None,
            "bridge_normal_offset_ratio_left": None,
            "bridge_normal_offset_ratio_right": None,
            "chart_origin_separation": None,
            "left_chart_component_fraction": None,
            "right_chart_component_fraction": None,
        }

    delta = nodes[right]["centroid"] - nodes[left]["centroid"]
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        unit_delta = np.zeros(3, dtype=np.float64)
    else:
        unit_delta = delta / length
    tangent_matrix = left_frame["axes"].T @ right_frame["axes"]
    singular_values = np.linalg.svd(tangent_matrix, compute_uv=False)
    node_count = len(nodes)

    def projection_ratio(frame: dict[str, Any], vector) -> float | None:
        if length <= 1e-9:
            return None
        return float(np.linalg.norm(frame["axes"].T @ vector))

    def normal_offset_ratio(frame: dict[str, Any], vector) -> float | None:
        if length <= 1e-9:
            return None
        return abs(float(np.dot(frame["normal"], vector)))

    return {
        "chart_pair_available": True,
        "chart_normal_agreement": rounded(abs(float(np.dot(left_frame["normal"], right_frame["normal"])))),
        "chart_tangent_singular_min": rounded(float(np.min(singular_values))),
        "chart_tangent_singular_max": rounded(float(np.max(singular_values))),
        "bridge_projection_ratio_left": rounded(projection_ratio(left_frame, unit_delta)),
        "bridge_projection_ratio_right": rounded(projection_ratio(right_frame, unit_delta)),
        "bridge_normal_offset_ratio_left": rounded(normal_offset_ratio(left_frame, unit_delta)),
        "bridge_normal_offset_ratio_right": rounded(normal_offset_ratio(right_frame, unit_delta)),
        "chart_origin_separation": rounded(
            float(np.linalg.norm(right_frame["centroid"] - left_frame["centroid"]))
        ),
        "left_chart_component_fraction": rounded(left_frame["node_count"] / node_count if node_count else 0.0),
        "right_chart_component_fraction": rounded(right_frame["node_count"] / node_count if node_count else 0.0),
    }


def grow_local_atlas_edges(
    nodes: list[dict[str, Any]],
    all_edges: list[tuple[int, int]],
    base_edges: set[tuple[int, int]],
    edge_quality: list[dict[str, float | tuple[int, int]]],
    target: float,
    max_bridge_offset_ratio: float,
    min_bridge_normal_agreement: float,
    bridge_only: bool = True,
    use_broad_candidate_policy: bool = False,
    max_p90_distortion: float | None = None,
    use_two_chart_boundary_gate: bool = False,
    two_chart_boundary_gate_only: bool = False,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], dict[str, Any]]:
    """Grow using bridge-local quality, treating merged components as an atlas."""

    grown_edges, added_edges, metrics, _sequence = grow_local_atlas_edges_with_sequence(
        nodes,
        all_edges,
        base_edges,
        edge_quality,
        target=target,
        max_bridge_offset_ratio=max_bridge_offset_ratio,
        min_bridge_normal_agreement=min_bridge_normal_agreement,
        bridge_only=bridge_only,
        max_p90_distortion=max_p90_distortion,
        use_broad_candidate_policy=use_broad_candidate_policy,
        use_two_chart_boundary_gate=use_two_chart_boundary_gate,
        two_chart_boundary_gate_only=two_chart_boundary_gate_only,
    )
    return grown_edges, added_edges, metrics


def local_atlas_first_transition_reconciliation(
    steps: list[dict[str, Any]],
    max_p90_distortion: float | None,
) -> dict[str, Any]:
    """Package first-step local/global attribution without exposing the edge."""

    if not steps:
        return {
            "enabled": False,
            "reason": "no_local_atlas_steps",
        }
    first = steps[0]
    local_quality = first.get("local_chart_quality_p90_after")
    boundary_quality = first.get("boundary_local_quality_p90_after")
    global_p90 = first.get("global_p90_edge_distortion_after")
    cap_available = isinstance(max_p90_distortion, (int, float)) and math.isfinite(
        float(max_p90_distortion)
    )
    local_cap_compliant = first.get("local_quality_cap_compliant_after")
    boundary_cap_compliant = first.get("boundary_quality_cap_compliant_after")
    global_cap_compliant = first.get("global_p90_cap_compliant_after")
    global_blocked_despite_local = (
        local_cap_compliant is True and global_cap_compliant is False
    )
    global_blocked_despite_boundary = (
        boundary_cap_compliant is True and global_cap_compliant is False
    )
    if not cap_available:
        decision = "first_transition_cap_not_available"
    elif global_cap_compliant is True:
        decision = "first_transition_global_cap_compliant"
    elif global_blocked_despite_boundary:
        decision = "first_transition_boundary_chart_resolves_local_but_not_global"
    elif global_blocked_despite_local:
        decision = "first_transition_global_blocked_despite_local_cap"
    else:
        decision = "first_transition_not_locally_cap_compliant"
    return {
        "enabled": True,
        "method": "scalar_first_local_atlas_step_attribution",
        "max_p90_distortion": rounded(max_p90_distortion),
        "first_step": {
            "coverage_delta": first.get("coverage_delta"),
            "local_chart_quality_p90_after": local_quality,
            "boundary_local_quality_p90_after": boundary_quality,
            "global_p90_edge_distortion_after": global_p90,
            "local_global_p90_gap_after": first.get("local_global_p90_gap_after"),
            "boundary_global_p90_gap_after": first.get("boundary_global_p90_gap_after"),
            "local_quality_cap_margin_after": first.get("local_quality_cap_margin_after"),
            "boundary_quality_cap_margin_after": first.get("boundary_quality_cap_margin_after"),
            "global_p90_cap_gap_after": first.get("global_p90_cap_gap_after"),
            "local_quality_cap_compliant_after": local_cap_compliant,
            "boundary_quality_cap_compliant_after": boundary_cap_compliant,
            "global_p90_cap_compliant_after": global_cap_compliant,
            "global_blocked_despite_local_cap_after": global_blocked_despite_local,
            "global_blocked_despite_boundary_cap_after": global_blocked_despite_boundary,
            "transition_normal_agreement": first.get("normal_agreement"),
            "transition_normal_risk": first.get("transition_normal_risk"),
            "selection_quality": first.get("selection_quality"),
            "selection_uses_transition_normal_risk": first.get(
                "selection_uses_transition_normal_risk"
            ),
            "chart_pair_available": first.get("chart_pair_available"),
            "chart_normal_agreement": first.get("chart_normal_agreement"),
        },
        "decision": decision,
        "storage_policy": (
            "Scalar first-transition attribution only; no edge ID, endpoint, path, "
            "component ID, coordinate, point cloud, mesh, prediction, letter, or title."
        ),
    }


def local_atlas_two_chart_atlas_reconciliation(
    steps: list[dict[str, Any]],
    max_p90_distortion: float | None,
) -> dict[str, Any]:
    """Compare the first accepted two-chart atlas bridge with direct global PCA."""

    if not steps:
        return {
            "enabled": False,
            "reason": "no_local_atlas_steps",
        }
    boundary_steps = [
        step for step in steps if step.get("selection_source") == "two_chart_boundary_gate"
    ]
    if not boundary_steps:
        return {
            "enabled": False,
            "reason": "no_two_chart_boundary_gate_steps",
            "step_count": len(steps),
        }

    first = boundary_steps[0]

    def number(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        return finite_float(value)

    def ratio(numerator: Any, denominator: Any) -> float | None:
        numerator_value = number(numerator)
        denominator_value = number(denominator)
        if numerator_value is None or denominator_value is None or denominator_value <= 1e-12:
            return None
        return rounded(numerator_value / denominator_value)

    cap = number(max_p90_distortion)
    global_p90 = number(first.get("global_p90_edge_distortion_after"))
    two_chart_p90 = number(first.get("two_chart_rigid_trial_p90_edge_distortion"))
    two_chart_all_edge_p90 = number(
        first.get("two_chart_rigid_trial_all_edge_p90_distortion")
    )
    cap_available = cap is not None
    global_cap_compliant = (
        global_p90 is not None and cap is not None and global_p90 <= cap
    )
    two_chart_cap_compliant = (
        two_chart_p90 is not None and cap is not None and two_chart_p90 <= cap
    )

    if first.get("two_chart_rigid_trial_selection_gate_pass") is not True:
        decision = "first_two_chart_boundary_gate_not_selection_passed"
    elif not cap_available:
        decision = "two_chart_atlas_cap_not_available"
    elif global_cap_compliant is False and two_chart_cap_compliant is True:
        decision = "two_chart_atlas_resolves_direct_global_blocker"
    elif global_cap_compliant is True:
        decision = "direct_global_already_cap_compliant"
    elif two_chart_cap_compliant is False:
        decision = "two_chart_atlas_still_above_cap"
    else:
        decision = "two_chart_atlas_inconclusive"

    return {
        "enabled": True,
        "method": "scalar_first_two_chart_boundary_gate_atlas_reconciliation",
        "scope": "first_two_chart_boundary_gate_step",
        "max_p90_distortion": rounded(cap),
        "step_count": len(steps),
        "two_chart_boundary_gate_step_count": len(boundary_steps),
        "first_boundary_step": {
            "coverage_delta": first.get("coverage_delta"),
            "boundary_local_quality_p90_after": first.get("boundary_local_quality_p90_after"),
            "direct_global_p90_edge_distortion_after": rounded(global_p90),
            "direct_global_p90_cap_gap_after": rounded(
                global_p90 - cap if global_p90 is not None and cap is not None else None
            ),
            "direct_global_cap_compliant_after": global_cap_compliant
            if cap_available and global_p90 is not None
            else None,
            "two_chart_bridge_aware_p90_edge_distortion": rounded(two_chart_p90),
            "two_chart_all_edge_p90_distortion": rounded(two_chart_all_edge_p90),
            "two_chart_internal_p90_edge_distortion": first.get(
                "two_chart_rigid_trial_internal_p90_edge_distortion"
            ),
            "two_chart_bridge_edge_distortion": first.get(
                "two_chart_rigid_trial_bridge_edge_distortion"
            ),
            "two_chart_bridge_to_internal_p90_ratio": first.get(
                "two_chart_rigid_trial_bridge_to_internal_p90_ratio"
            ),
            "two_chart_bridge_projection_ratio": first.get(
                "two_chart_rigid_trial_bridge_projection_ratio"
            ),
            "two_chart_transform_determinant": first.get(
                "two_chart_rigid_trial_orthogonal_transform_determinant"
            ),
            "two_chart_placement_class": first.get(
                "two_chart_rigid_trial_placement_class"
            ),
            "two_chart_cap_margin": rounded(
                cap - two_chart_p90
                if cap is not None and two_chart_p90 is not None
                else None
            ),
            "two_chart_cap_compliant": two_chart_cap_compliant
            if cap_available and two_chart_p90 is not None
            else None,
            "two_chart_delta_vs_direct_global": rounded(
                two_chart_p90 - global_p90
                if two_chart_p90 is not None and global_p90 is not None
                else None
            ),
            "direct_global_delta_vs_two_chart": rounded(
                global_p90 - two_chart_p90
                if two_chart_p90 is not None and global_p90 is not None
                else None
            ),
            "all_edge_hiding_ratio": ratio(two_chart_p90, two_chart_all_edge_p90),
        },
        "decision": decision,
        "storage_policy": (
            "Scalar first-boundary-step atlas reconciliation only; no edge ID, endpoint, "
            "path, component ID, coordinate, point cloud, mesh, prediction, letter, or title."
        ),
    }


def local_atlas_multi_chart_atlas_reconciliation(
    steps: list[dict[str, Any]],
    atlas_metrics: dict[str, Any],
    max_p90_distortion: float | None,
) -> dict[str, Any]:
    """Compare final direct-global PCA with a scalar multi-chart atlas score."""

    if not steps:
        return {
            "enabled": False,
            "reason": "no_local_atlas_steps",
        }
    if atlas_metrics.get("multi_chart_atlas_available") is not True:
        return {
            "enabled": False,
            "reason": "multi_chart_atlas_unavailable",
            "step_count": len(steps),
            **atlas_metrics,
        }

    def number(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        return finite_float(value)

    cap = number(max_p90_distortion)
    final_step = steps[-1]
    direct_global_p90 = number(final_step.get("global_p90_edge_distortion_after"))
    bridge_aware_p90 = number(
        atlas_metrics.get("multi_chart_atlas_bridge_aware_p90_edge_distortion")
    )
    unplaced_bridges = number(
        atlas_metrics.get("multi_chart_atlas_unplaced_bridge_edge_count")
    )
    cap_available = cap is not None
    direct_cap_compliant = (
        direct_global_p90 is not None and cap is not None and direct_global_p90 <= cap
    )
    atlas_cap_compliant = (
        bridge_aware_p90 is not None and cap is not None and bridge_aware_p90 <= cap
    )
    if unplaced_bridges is not None and unplaced_bridges > 0:
        decision = "multi_chart_atlas_partial_bridge_coverage"
    elif not cap_available:
        decision = "multi_chart_atlas_cap_not_available"
    elif direct_cap_compliant is False and atlas_cap_compliant is True:
        decision = "multi_chart_atlas_resolves_direct_global_blocker"
    elif direct_cap_compliant is True:
        decision = "direct_global_already_cap_compliant"
    elif atlas_cap_compliant is False:
        decision = "multi_chart_atlas_still_above_cap"
    else:
        decision = "multi_chart_atlas_inconclusive"

    return {
        "enabled": True,
        "method": "scalar_multi_chart_atlas_reconciliation",
        "scope": "final_local_atlas_result",
        "max_p90_distortion": rounded(cap),
        "step_count": len(steps),
        "direct_global_p90_edge_distortion": rounded(direct_global_p90),
        "direct_global_cap_compliant": direct_cap_compliant
        if cap_available and direct_global_p90 is not None
        else None,
        "direct_global_p90_cap_gap": rounded(
            direct_global_p90 - cap
            if direct_global_p90 is not None and cap is not None
            else None
        ),
        "multi_chart_cap_compliant": atlas_cap_compliant
        if cap_available and bridge_aware_p90 is not None
        else None,
        "multi_chart_cap_margin": rounded(
            cap - bridge_aware_p90
            if cap is not None and bridge_aware_p90 is not None
            else None
        ),
        "direct_global_delta_vs_multi_chart": rounded(
            direct_global_p90 - bridge_aware_p90
            if direct_global_p90 is not None and bridge_aware_p90 is not None
            else None
        ),
        **atlas_metrics,
        "decision": decision,
        "storage_policy": (
            "Scalar final-atlas reconciliation only; no edge ID, endpoint, path, "
            "component ID, coordinate, point cloud, mesh, prediction, letter, or title."
        ),
    }


def grow_local_atlas_edges_with_sequence(
    nodes: list[dict[str, Any]],
    all_edges: list[tuple[int, int]],
    base_edges: set[tuple[int, int]],
    edge_quality: list[dict[str, float | tuple[int, int]]],
    target: float,
    max_bridge_offset_ratio: float,
    min_bridge_normal_agreement: float,
    bridge_only: bool = True,
    max_p90_distortion: float | None = None,
    use_transition_normal_risk_quality: bool = False,
    use_broad_candidate_policy: bool = False,
    use_two_chart_boundary_gate: bool = False,
    two_chart_boundary_gate_only: bool = False,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], dict[str, Any], dict[str, Any]]:
    """Grow using bridge-local quality and keep a scalar-only chosen-bridge trace."""

    node_count = len(nodes)
    grown_edges = set(base_edges)
    added_edges: set[tuple[int, int]] = set()
    candidates = candidate_edges(all_edges, base_edges, edge_quality)
    base_roots, chart_frames = component_chart_frames(nodes, base_edges)
    policy_base_roots, policy_base_sizes = component_roots(node_count, base_edges)
    policy_base_largest_root = (
        max(policy_base_sizes, key=policy_base_sizes.get) if policy_base_sizes else None
    )
    steps: list[dict[str, Any]] = []
    accepted_transition_normal_risks: list[float] = []

    def cap_compliant(value: Any) -> bool | None:
        if max_p90_distortion is None or not isinstance(value, (int, float)):
            return None
        return float(value) <= float(max_p90_distortion)

    def cap_gap(value: Any) -> float | None:
        if max_p90_distortion is None or not isinstance(value, (int, float)):
            return None
        return rounded(float(value) - float(max_p90_distortion))

    def cap_margin(value: Any) -> float | None:
        if max_p90_distortion is None or not isinstance(value, (int, float)):
            return None
        return rounded(float(max_p90_distortion) - float(value))

    def selection_quality(normal_agreement: float, offset_ratio: float) -> float:
        if not use_transition_normal_risk_quality:
            return offset_ratio
        return max(offset_ratio, 1.0 - normal_agreement)

    def boundary_gate_trial(
        edge: tuple[int, int],
        trial_edges: set[tuple[int, int]],
        current_edges: set[tuple[int, int]],
        current_roots: list[int],
        current_sizes: dict[int, int],
        current_chart_frames: dict[int, dict[str, Any]],
        transition_normal_risk: float,
    ) -> dict[str, Any]:
        if max_p90_distortion is None:
            return {
                "two_chart_boundary_gate_available": False,
                "two_chart_boundary_gate_pass": False,
                "two_chart_boundary_gate_reason": "max_p90_distortion_not_set",
            }
        local_metrics = component_local_chart_metrics(nodes, trial_edges, base_edges, edge_quality)
        normal_risk_p90 = percentile_from_sorted(
            sorted([*accepted_transition_normal_risks, transition_normal_risk]),
            90,
        )
        boundary_quality = scalar_max(
            local_metrics["local_chart_p90_internal_edge_distortion"],
            local_metrics["local_chart_p90_bridge_offset_ratio"],
            normal_risk_p90,
        )
        boundary_cap_compliant = cap_compliant(boundary_quality)
        rigid_trial = two_chart_rigid_trial_metrics(
            nodes,
            current_edges,
            current_roots,
            current_sizes,
            edge,
            current_chart_frames,
        )
        rigid_gate = two_chart_rigid_trial_gate_metrics(
            rigid_trial,
            float(max_p90_distortion),
        )
        return {
            "two_chart_boundary_gate_available": True,
            "two_chart_boundary_gate_pass": (
                boundary_cap_compliant is True
                and rigid_gate["two_chart_rigid_trial_selection_gate_pass"] is True
            ),
            "boundary_transition_quality": rounded(boundary_quality),
            "boundary_transition_cap_compliant": boundary_cap_compliant,
            "boundary_p90_transition_normal_risk": rounded(normal_risk_p90),
            "boundary_p90_transition_offset_ratio": local_metrics[
                "local_chart_p90_bridge_offset_ratio"
            ],
            **rigid_trial,
            **rigid_gate,
            "two_chart_rigid_trial_p90_cap_gap": cap_gap(
                rigid_trial.get("two_chart_rigid_trial_p90_edge_distortion")
            ),
            "two_chart_rigid_trial_cap_compliant": cap_compliant(
                rigid_trial.get("two_chart_rigid_trial_p90_edge_distortion")
            ),
        }

    stop_reason = (
        "target_already_reached"
        if largest_fraction(node_count, grown_edges) >= target
        else "no_candidates"
    )
    while largest_fraction(node_count, grown_edges) < target and candidates:
        roots, sizes = component_roots(node_count, grown_edges)
        largest_root = max(sizes, key=sizes.get) if sizes else None
        current_fraction = largest_fraction(node_count, grown_edges)
        best_index = None
        best_score = None
        best_trial_fraction = None
        best_left_fraction = None
        best_right_fraction = None
        best_selection_source = "local_atlas_bridge_gate"
        best_broad_policy_pass = False
        best_touches_base_largest_component = None
        best_two_chart_boundary_gate: dict[str, Any] = {
            "two_chart_boundary_gate_available": False,
            "two_chart_boundary_gate_pass": False,
        }
        eligible_candidate_count = 0
        bridge_candidate_count = 0
        current_chart_roots: list[int] | None = None
        current_chart_frames: dict[int, dict[str, Any]] | None = None
        if use_two_chart_boundary_gate:
            current_chart_roots, current_chart_frames = component_chart_frames(nodes, grown_edges)
        for candidate_index, quality in enumerate(candidates):
            edge = sorted_edge(quality["edge"])
            left, right = edge
            if roots[left] == roots[right]:
                continue
            bridge_candidate_count += 1
            if bridge_only and largest_root not in (roots[left], roots[right]):
                continue
            normal_agreement = float(quality["normal_agreement"])
            offset_ratio = float(quality["offset_ratio"])
            transition_normal_risk = 1.0 - normal_agreement
            bridge_selection_quality = selection_quality(normal_agreement, offset_ratio)
            trial_edges = set(grown_edges)
            trial_edges.add(edge)
            trial_fraction = largest_fraction(node_count, trial_edges)
            if trial_fraction <= current_fraction:
                continue
            local_gate_pass = (
                normal_agreement >= min_bridge_normal_agreement
                and offset_ratio <= max_bridge_offset_ratio
                and (
                    not use_transition_normal_risk_quality
                    or bridge_selection_quality <= max_bridge_offset_ratio
                )
            )
            broad_policy_pass = False
            touches_base_largest_component = bool(
                policy_base_largest_root is not None
                and (
                    policy_base_roots[left] == policy_base_largest_root
                    or policy_base_roots[right] == policy_base_largest_root
                )
            )
            if use_broad_candidate_policy and trial_fraction >= target:
                chart_frame_metrics = bridge_chart_frame_metrics(
                    nodes,
                    edge,
                    base_roots,
                    chart_frames,
                )
                projection = bridge_projection_min(chart_frame_metrics)
                broad_policy_pass = (
                    bool(chart_frame_metrics.get("chart_pair_available"))
                    and touches_base_largest_component
                    and normal_agreement >= BROAD_CANDIDATE_MIN_NORMAL_AGREEMENT
                    and projection is not None
                    and projection >= BROAD_CANDIDATE_MIN_BRIDGE_PROJECTION_RATIO
                )
            boundary_gate = {
                "two_chart_boundary_gate_available": False,
                "two_chart_boundary_gate_pass": False,
            }
            if (
                use_two_chart_boundary_gate
                and current_chart_roots is not None
                and current_chart_frames is not None
            ):
                boundary_gate = boundary_gate_trial(
                    edge,
                    trial_edges,
                    grown_edges,
                    current_chart_roots,
                    sizes,
                    current_chart_frames,
                    transition_normal_risk,
                )
            two_chart_boundary_gate_pass = bool(
                boundary_gate.get("two_chart_boundary_gate_pass")
            )
            if two_chart_boundary_gate_only:
                local_gate_pass = False
                broad_policy_pass = False
            if not local_gate_pass and not broad_policy_pass and not two_chart_boundary_gate_pass:
                continue
            eligible_candidate_count += 1
            if two_chart_boundary_gate_pass:
                selection_source = "two_chart_boundary_gate"
            elif broad_policy_pass:
                selection_source = "broad_candidate_policy"
            else:
                selection_source = "local_atlas_bridge_gate"
            source_rank = {
                "two_chart_boundary_gate": 0,
                "broad_candidate_policy": 1,
                "local_atlas_bridge_gate": 2,
            }[selection_source]
            source_quality = (
                boundary_gate.get("boundary_transition_quality")
                if selection_source == "two_chart_boundary_gate"
                else bridge_selection_quality
            )
            source_quality_value = (
                float("inf")
                if not isinstance(source_quality, (int, float))
                else float(source_quality)
            )
            rigid_p90 = boundary_gate.get("two_chart_rigid_trial_p90_edge_distortion")
            rigid_p90_value = (
                float("inf")
                if not isinstance(rigid_p90, (int, float))
                else float(rigid_p90)
            )
            score = (
                source_rank,
                -trial_fraction,
                source_quality_value,
                rigid_p90_value,
                -normal_agreement,
                edge,
            )
            if best_score is None or score < best_score:
                best_index = candidate_index
                best_score = score
                best_trial_fraction = trial_fraction
                best_left_fraction = sizes.get(roots[left], 0) / node_count if node_count else 0.0
                best_right_fraction = sizes.get(roots[right], 0) / node_count if node_count else 0.0
                best_selection_source = selection_source
                best_broad_policy_pass = broad_policy_pass
                best_touches_base_largest_component = touches_base_largest_component
                best_two_chart_boundary_gate = boundary_gate
        if best_index is None:
            stop_reason = "no_eligible_bridge"
            break
        chosen = candidates.pop(best_index)
        edge = sorted_edge(chosen["edge"])
        grown_edges.add(edge)
        added_edges.add(edge)
        local_metrics_after = component_local_chart_metrics(nodes, grown_edges, base_edges, edge_quality)
        global_chart_after = ablation.unwrap_proxy_metrics(nodes, sorted(grown_edges))
        chart_frame_metrics = bridge_chart_frame_metrics(nodes, edge, base_roots, chart_frames)
        normal_agreement = float(chosen["normal_agreement"])
        offset_ratio = float(chosen["offset_ratio"])
        transition_normal_risk = 1.0 - normal_agreement
        bridge_selection_quality = selection_quality(normal_agreement, offset_ratio)
        accepted_transition_normal_risks.append(transition_normal_risk)
        transition_normal_risk_p90 = percentile_from_sorted(
            sorted(accepted_transition_normal_risks),
            90,
        )
        boundary_quality_after = scalar_max(
            local_metrics_after["local_chart_p90_internal_edge_distortion"],
            local_metrics_after["local_chart_p90_bridge_offset_ratio"],
            transition_normal_risk_p90,
        )
        local_quality_after = local_metrics_after["local_chart_quality_p90"]
        global_p90_after = global_chart_after["patch_unwrap_proxy_p90_edge_distortion"]
        local_quality_cap_compliant = cap_compliant(local_quality_after)
        boundary_quality_cap_compliant = cap_compliant(boundary_quality_after)
        global_p90_cap_compliant = cap_compliant(global_p90_after)
        steps.append(
            {
                "step": len(steps) + 1,
                "bridge_candidate_count_before": bridge_candidate_count,
                "eligible_candidate_count_before": eligible_candidate_count,
                "largest_component_fraction_before": rounded(current_fraction),
                "largest_component_fraction_after": rounded(
                    best_trial_fraction
                    if best_trial_fraction is not None
                    else largest_fraction(node_count, grown_edges)
                ),
                "coverage_delta": rounded(
                    (
                        best_trial_fraction
                        if best_trial_fraction is not None
                        else largest_fraction(node_count, grown_edges)
                    )
                    - current_fraction
                ),
                "left_component_fraction_before": rounded(best_left_fraction),
                "right_component_fraction_before": rounded(best_right_fraction),
                "normal_agreement": rounded(chosen["normal_agreement"]),
                "transition_normal_risk": rounded(transition_normal_risk),
                "offset_ratio": rounded(chosen["offset_ratio"]),
                "selection_quality": rounded(bridge_selection_quality),
                "selection_uses_transition_normal_risk": bool(
                    use_transition_normal_risk_quality
                ),
                "selection_source": best_selection_source,
                "touches_base_largest_component": best_touches_base_largest_component,
                "broad_candidate_policy_pass": best_broad_policy_pass,
                "two_chart_boundary_gate_pass": bool(
                    best_two_chart_boundary_gate.get("two_chart_boundary_gate_pass")
                ),
                "broad_candidate_policy_normal_threshold": rounded(
                    BROAD_CANDIDATE_MIN_NORMAL_AGREEMENT
                ),
                "broad_candidate_policy_projection_threshold": rounded(
                    BROAD_CANDIDATE_MIN_BRIDGE_PROJECTION_RATIO
                ),
                "two_chart_boundary_gate_projection_threshold": rounded(
                    TWO_CHART_MIN_BRIDGE_PROJECTION_RATIO
                ),
                "two_chart_boundary_gate_orientation_determinant_threshold": rounded(
                    TWO_CHART_MIN_ORIENTATION_DETERMINANT
                ),
                **chart_frame_metrics,
                "min_bridge_projection_ratio": rounded(
                    bridge_projection_min(chart_frame_metrics)
                ),
                "boundary_transition_quality": best_two_chart_boundary_gate.get(
                    "boundary_transition_quality"
                ),
                "boundary_transition_cap_compliant": best_two_chart_boundary_gate.get(
                    "boundary_transition_cap_compliant"
                ),
                "two_chart_rigid_trial_p90_edge_distortion": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_p90_edge_distortion"
                ),
                "two_chart_rigid_trial_available": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_available"
                ),
                "two_chart_rigid_trial_node_fraction": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_node_fraction"
                ),
                "two_chart_rigid_trial_edge_count": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_edge_count"
                ),
                "two_chart_rigid_trial_anchor_component_fraction": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_anchor_component_fraction"
                ),
                "two_chart_rigid_trial_moving_component_fraction": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_moving_component_fraction"
                ),
                "two_chart_rigid_trial_chart_tangent_singular_min": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_chart_tangent_singular_min"
                ),
                "two_chart_rigid_trial_chart_tangent_singular_max": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_chart_tangent_singular_max"
                ),
                "two_chart_rigid_trial_orthogonal_transform_determinant": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_orthogonal_transform_determinant"
                ),
                "two_chart_rigid_trial_bridge_projection_ratio": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_bridge_projection_ratio"
                ),
                "two_chart_rigid_trial_bridge_edge_distortion": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_bridge_edge_distortion"
                ),
                "two_chart_rigid_trial_internal_p90_edge_distortion": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_internal_p90_edge_distortion"
                ),
                "two_chart_rigid_trial_bridge_to_internal_p90_ratio": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_bridge_to_internal_p90_ratio"
                ),
                "two_chart_rigid_trial_all_edge_p90_distortion": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_all_edge_p90_distortion"
                ),
                "two_chart_rigid_trial_p90_cap_gap": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_p90_cap_gap"
                ),
                "two_chart_rigid_trial_cap_compliant": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_cap_compliant"
                ),
                "two_chart_rigid_trial_projection_gate_pass": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_projection_gate_pass"
                ),
                "two_chart_rigid_trial_orientation_gate_pass": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_orientation_gate_pass"
                ),
                "two_chart_rigid_trial_selection_gate_pass": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_selection_gate_pass"
                ),
                "two_chart_rigid_trial_placement_class": best_two_chart_boundary_gate.get(
                    "two_chart_rigid_trial_placement_class"
                ),
                "local_chart_quality_p90_after": local_quality_after,
                "local_chart_p90_bridge_offset_ratio_after": local_metrics_after[
                    "local_chart_p90_bridge_offset_ratio"
                ],
                "local_chart_p10_bridge_normal_agreement_after": local_metrics_after[
                    "local_chart_p10_bridge_normal_agreement"
                ],
                "boundary_local_quality_p90_after": rounded(boundary_quality_after),
                "boundary_p90_transition_normal_risk_after": rounded(transition_normal_risk_p90),
                "boundary_quality_cap_margin_after": cap_margin(boundary_quality_after),
                "boundary_quality_cap_compliant_after": boundary_quality_cap_compliant,
                "global_p90_edge_distortion_after": rounded(global_p90_after),
                "global_p90_cap_gap_after": cap_gap(global_p90_after),
                "global_p90_cap_compliant_after": global_p90_cap_compliant,
                "local_quality_cap_margin_after": cap_margin(local_quality_after),
                "local_quality_cap_compliant_after": local_quality_cap_compliant,
                "local_global_p90_gap_after": rounded(
                    float(global_p90_after) - float(local_quality_after)
                    if isinstance(global_p90_after, (int, float))
                    and isinstance(local_quality_after, (int, float))
                    else None
                ),
                "boundary_global_p90_gap_after": rounded(
                    float(global_p90_after) - float(boundary_quality_after)
                    if isinstance(global_p90_after, (int, float))
                    and isinstance(boundary_quality_after, (int, float))
                    else None
                ),
                "global_blocked_despite_local_cap_after": (
                    local_quality_cap_compliant is True and global_p90_cap_compliant is False
                ),
                "global_blocked_despite_boundary_cap_after": (
                    boundary_quality_cap_compliant is True and global_p90_cap_compliant is False
                ),
            }
        )
        stop_reason = "target_reached" if largest_fraction(node_count, grown_edges) >= target else "continuing"
    metrics = component_local_chart_metrics(nodes, grown_edges, base_edges, edge_quality)
    final_multi_chart_atlas = multi_chart_atlas_metrics(nodes, base_edges, grown_edges)
    sequence = {
        "target": rounded(target),
        "reached": largest_fraction(node_count, grown_edges) >= target,
        "stop_reason": stop_reason if largest_fraction(node_count, grown_edges) < target else "target_reached",
        "step_count": len(steps),
        "final_largest_component_fraction": metrics["local_chart_largest_component_fraction"],
        "final_quality_p90": metrics["local_chart_quality_p90"],
        "selection_uses_transition_normal_risk": bool(use_transition_normal_risk_quality),
        "broad_candidate_policy_enabled": bool(use_broad_candidate_policy),
        "broad_candidate_policy_step_count": sum(
            1 for step in steps if step.get("selection_source") == "broad_candidate_policy"
        ),
        "two_chart_boundary_gate_enabled": bool(use_two_chart_boundary_gate),
        "two_chart_boundary_gate_only": bool(two_chart_boundary_gate_only),
        "two_chart_boundary_gate_step_count": sum(
            1 for step in steps if step.get("selection_source") == "two_chart_boundary_gate"
        ),
        "two_chart_boundary_gate_projection_threshold": rounded(
            TWO_CHART_MIN_BRIDGE_PROJECTION_RATIO
        ),
        "two_chart_boundary_gate_orientation_determinant_threshold": rounded(
            TWO_CHART_MIN_ORIENTATION_DETERMINANT
        ),
        "broad_candidate_policy_normal_threshold": rounded(
            BROAD_CANDIDATE_MIN_NORMAL_AGREEMENT
        ),
        "broad_candidate_policy_projection_threshold": rounded(
            BROAD_CANDIDATE_MIN_BRIDGE_PROJECTION_RATIO
        ),
        "steps": steps,
        "first_transition_reconciliation": local_atlas_first_transition_reconciliation(
            steps,
            max_p90_distortion,
        ),
        "two_chart_atlas_reconciliation": local_atlas_two_chart_atlas_reconciliation(
            steps,
            max_p90_distortion,
        ),
        "multi_chart_atlas_reconciliation": local_atlas_multi_chart_atlas_reconciliation(
            steps,
            final_multi_chart_atlas,
            max_p90_distortion,
        ),
    }
    return grown_edges, added_edges, metrics, sequence


def grow_two_stage_chart_edges_with_sequence(
    nodes: list[dict[str, Any]],
    all_edges: list[tuple[int, int]],
    base_edges: set[tuple[int, int]],
    edge_quality: list[dict[str, float | tuple[int, int]]],
    target: float,
    max_p90_distortion: float | None,
    rotation_threshold_degrees: float,
    cleanup_score: str = "coverage",
    cleanup_lookahead: int = 1,
    cleanup_branching: int = 32,
    cleanup_normal_penalty: float = 2.0,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], dict[str, Any], dict[str, Any]]:
    """Greedy stateful chart propagation with scalar-only stage summaries."""

    if cleanup_lookahead < 1:
        raise ValueError("cleanup_lookahead must be positive")
    if cleanup_branching < 1:
        raise ValueError("cleanup_branching must be positive")
    if cleanup_normal_penalty < 0:
        raise ValueError("cleanup_normal_penalty must be non-negative")

    node_count = len(nodes)
    grown_edges = set(base_edges)
    added_edges: set[tuple[int, int]] = set()
    base_roots, chart_frames = component_chart_frames(nodes, base_edges)
    candidates: list[dict[str, Any]] = []
    for quality in candidate_edges(all_edges, base_edges, edge_quality):
        edge = sorted_edge(quality["edge"])
        chart_metrics = bridge_chart_frame_metrics(nodes, edge, base_roots, chart_frames)
        chart_normal = chart_metrics.get("chart_normal_agreement")
        rotation_degrees = None
        if isinstance(chart_normal, (int, float)):
            clipped = min(1.0, max(0.0, float(chart_normal)))
            rotation_degrees = math.degrees(math.acos(clipped))
        candidates.append(
            {
                "edge": edge,
                "normal_agreement": float(quality["normal_agreement"]),
                "offset_ratio": float(quality["offset_ratio"]),
                "chart_pair_available": bool(chart_metrics.get("chart_pair_available")),
                "rotation_degrees": rotation_degrees,
            }
        )

    def trial_chart(edge: tuple[int, int]) -> dict[str, Any]:
        trial_edges = set(grown_edges)
        trial_edges.add(edge)
        return ablation.unwrap_proxy_metrics(nodes, sorted(trial_edges))

    def trial_chart_for_edges(extra_edges: list[tuple[int, int]]) -> dict[str, Any]:
        trial_edges = set(grown_edges)
        trial_edges.update(sorted_edge(edge) for edge in extra_edges)
        return ablation.unwrap_proxy_metrics(nodes, sorted(trial_edges))

    def trial_local_metrics_for_edges(extra_edges: list[tuple[int, int]]) -> dict[str, Any]:
        trial_edges = set(grown_edges)
        trial_edges.update(sorted_edge(edge) for edge in extra_edges)
        return component_local_chart_metrics(nodes, trial_edges, base_edges, edge_quality)

    def p90_allowed(p90: Any) -> bool:
        return max_p90_distortion is None or (
            isinstance(p90, (int, float)) and float(p90) <= max_p90_distortion
        )

    def current_roots() -> tuple[list[int], dict[int, int]]:
        return component_roots(node_count, grown_edges)

    def largest_root(roots: list[int], sizes: dict[int, int]) -> int | None:
        return max(sizes, key=sizes.get) if sizes else None

    def stage_state(stage: str, edge_count: int) -> dict[str, Any]:
        chart = ablation.unwrap_proxy_metrics(nodes, sorted(grown_edges))
        metrics = component_local_chart_metrics(nodes, grown_edges, base_edges, edge_quality)
        return {
            "stage": stage,
            "accepted_edge_count": edge_count,
            "largest_component_fraction": metrics["local_chart_largest_component_fraction"],
            "global_p90_edge_distortion": rounded(chart["patch_unwrap_proxy_p90_edge_distortion"]),
            "local_chart_quality_p90": metrics["local_chart_quality_p90"],
        }

    def add_edge(candidate: dict[str, Any]) -> None:
        edge = sorted_edge(candidate["edge"])
        grown_edges.add(edge)
        added_edges.add(edge)

    path_cleanup_step_count = 0
    path_cleanup_max_length = 0
    path_search_summaries: list[dict[str, Any]] = []

    stage_summaries: list[dict[str, Any]] = []
    stage1_count = 0
    for candidate in sorted(
        candidates,
        key=lambda row: (
            float("inf")
            if not isinstance(row.get("rotation_degrees"), (int, float))
            else float(row["rotation_degrees"]),
            row["offset_ratio"],
            -row["normal_agreement"],
            row["edge"],
        ),
    ):
        rotation = candidate.get("rotation_degrees")
        if not isinstance(rotation, (int, float)) or float(rotation) > rotation_threshold_degrees:
            continue
        roots, _sizes = current_roots()
        left, right = candidate["edge"]
        if roots[left] == roots[right]:
            continue
        chart = trial_chart(candidate["edge"])
        if not p90_allowed(chart.get("patch_unwrap_proxy_p90_edge_distortion")):
            continue
        add_edge(candidate)
        stage1_count += 1
    stage_summaries.append(stage_state("rotation_limited_chart_core", stage1_count))

    def attach_cleanup(stage: str, predicate) -> int:
        nonlocal path_cleanup_step_count, path_cleanup_max_length
        count = 0
        while True:
            roots, sizes = current_roots()
            anchor = largest_root(roots, sizes)
            if anchor is None:
                break
            if cleanup_score in {
                "path",
                "path_quality",
                "path_quality_minimax",
                "path_quality_normal",
                "path_target_quality",
            }:
                current_fraction = largest_fraction(node_count, grown_edges)
                eligible = [
                    candidate
                    for candidate in candidates
                    if candidate["edge"] not in grown_edges
                    and predicate(candidate)
                    and roots[candidate["edge"][0]] != roots[candidate["edge"][1]]
                ]
                if not eligible:
                    break
                best_path: list[dict[str, Any]] | None = None
                best_path_score = None
                improving_rows: list[dict[str, Any]] = []
                blocked_rows: list[dict[str, Any]] = []
                score_rows: list[dict[str, Any]] = []
                visited_path_count = 0
                improving_path_count = 0
                p90_blocked_improving_path_count = 0
                selected_path_summary: dict[str, Any] | None = None

                def extension_key(candidate: dict[str, Any]) -> tuple[float, float, float, tuple[int, int]]:
                    rotation = candidate.get("rotation_degrees")
                    return (
                        candidate["offset_ratio"],
                        -candidate["normal_agreement"],
                        float("inf") if rotation is None else float(rotation),
                        candidate["edge"],
                    )

                def visit(
                    connected_roots: set[int],
                    path_candidates: list[dict[str, Any]],
                    used_edges: set[tuple[int, int]],
                    depth: int,
                ) -> None:
                    nonlocal best_path, best_path_score, visited_path_count, improving_path_count
                    nonlocal p90_blocked_improving_path_count, selected_path_summary
                    if path_candidates:
                        visited_path_count += 1
                        path_edges = [candidate["edge"] for candidate in path_candidates]
                        chart = trial_chart_for_edges(path_edges)
                        p90 = chart.get("patch_unwrap_proxy_p90_edge_distortion")
                        trial_fraction = float(
                            chart.get("patch_unwrap_proxy_largest_component_node_fraction") or 0.0
                        )
                        if trial_fraction > current_fraction:
                            improving_path_count += 1
                            improving_row = {
                                "path_length": len(path_candidates),
                                "target_gap": max(0.0, target - trial_fraction),
                                "trial_fraction": trial_fraction,
                                "p90": p90,
                            }
                            improving_rows.append(improving_row)
                        if trial_fraction > current_fraction and not p90_allowed(p90):
                            p90_blocked_improving_path_count += 1
                            blocked_rows.append(improving_row)
                        if trial_fraction > current_fraction and p90_allowed(p90):
                            p90_score = float("inf") if p90 is None else float(p90)
                            local_score = 0.0
                            if cleanup_score in {
                                "path_quality",
                                "path_quality_minimax",
                                "path_quality_normal",
                                "path_target_quality",
                            }:
                                local_metrics = trial_local_metrics_for_edges(path_edges)
                                local_quality = local_metrics.get("local_chart_quality_p90")
                                local_score = (
                                    float("inf")
                                    if local_quality is None
                                    else float(local_quality)
                                )
                            mean_offset = sum(
                                candidate["offset_ratio"] for candidate in path_candidates
                            ) / len(path_candidates)
                            mean_normal = sum(
                                candidate["normal_agreement"] for candidate in path_candidates
                            ) / len(path_candidates)
                            normal_risk = max(0.0, 1.0 - mean_normal)
                            path_signature = tuple(
                                sorted(candidate["edge"] for candidate in path_candidates)
                            )
                            score_row = {
                                "path_length": len(path_candidates),
                                "target_gap": max(0.0, target - trial_fraction),
                                "trial_fraction": trial_fraction,
                                "p90": p90,
                                "local_quality": local_score,
                                "mean_offset": mean_offset,
                                "mean_normal": mean_normal,
                                "normal_risk": normal_risk,
                            }
                            score_rows.append(score_row)
                            if cleanup_score == "path_quality":
                                score = (
                                    local_score,
                                    p90_score,
                                    -trial_fraction,
                                    len(path_candidates),
                                    mean_offset,
                                    -mean_normal,
                                    path_signature,
                                )
                            elif cleanup_score == "path_quality_minimax":
                                score = (
                                    max(local_score, mean_offset, normal_risk),
                                    local_score,
                                    mean_offset,
                                    normal_risk,
                                    p90_score,
                                    -trial_fraction,
                                    len(path_candidates),
                                    -mean_normal,
                                    path_signature,
                                )
                            elif cleanup_score == "path_quality_normal":
                                score = (
                                    local_score + cleanup_normal_penalty * normal_risk,
                                    local_score,
                                    p90_score,
                                    -trial_fraction,
                                    len(path_candidates),
                                    mean_offset,
                                    -mean_normal,
                                    path_signature,
                                )
                            elif cleanup_score == "path_target_quality":
                                score = (
                                    max(0.0, target - trial_fraction),
                                    p90_score,
                                    local_score,
                                    -trial_fraction,
                                    len(path_candidates),
                                    mean_offset,
                                    -mean_normal,
                                    path_signature,
                                )
                            else:
                                score = (
                                    -trial_fraction,
                                    p90_score,
                                    len(path_candidates),
                                    mean_offset,
                                    -mean_normal,
                                    path_signature,
                                )
                            if best_path_score is None or score < best_path_score:
                                best_path_score = score
                                best_path = list(path_candidates)
                                selected_path_summary = {
                                    key: rounded(value) for key, value in score_row.items()
                                }
                    if depth >= cleanup_lookahead:
                        return
                    frontier: list[dict[str, Any]] = []
                    for candidate in eligible:
                        edge = candidate["edge"]
                        if edge in used_edges:
                            continue
                        left_root = roots[edge[0]]
                        right_root = roots[edge[1]]
                        left_connected = left_root in connected_roots
                        right_connected = right_root in connected_roots
                        if left_connected == right_connected:
                            continue
                        frontier.append(candidate)
                    for candidate in sorted(frontier, key=extension_key)[:cleanup_branching]:
                        edge = candidate["edge"]
                        next_roots = set(connected_roots)
                        next_roots.add(roots[edge[0]])
                        next_roots.add(roots[edge[1]])
                        visit(
                            next_roots,
                            [*path_candidates, candidate],
                            {*used_edges, edge},
                            depth + 1,
                        )

                visit({anchor}, [], set(), 0)
                best_score_public = None
                if best_path_score is not None:
                    best_score_public = [rounded(value) for value in best_path_score[:-1]]
                path_search_summaries.append(
                    {
                        "stage": stage,
                        "attempt": path_cleanup_step_count + 1,
                        "current_largest_component_fraction": rounded(current_fraction),
                        "eligible_candidate_count": len(eligible),
                        "visited_path_count": visited_path_count,
                        "improving_path_count": improving_path_count,
                        "p90_blocked_improving_path_count": p90_blocked_improving_path_count,
                        "strict_cap_viable_path_count": len(score_rows),
                        "selected_path_length": len(best_path) if best_path else 0,
                        "selected_score_prefix": best_score_public,
                        "selected_path_summary": selected_path_summary,
                        "improving_path_length_summary": numeric_summary(
                            [row["path_length"] for row in improving_rows]
                        ),
                        "improving_target_gap_summary": numeric_summary(
                            [row["target_gap"] for row in improving_rows]
                        ),
                        "improving_trial_fraction_summary": numeric_summary(
                            [row["trial_fraction"] for row in improving_rows]
                        ),
                        "improving_p90_summary": numeric_summary([row["p90"] for row in improving_rows]),
                        "p90_blocked_path_length_summary": numeric_summary(
                            [row["path_length"] for row in blocked_rows]
                        ),
                        "p90_blocked_target_gap_summary": numeric_summary(
                            [row["target_gap"] for row in blocked_rows]
                        ),
                        "p90_blocked_trial_fraction_summary": numeric_summary(
                            [row["trial_fraction"] for row in blocked_rows]
                        ),
                        "p90_blocked_p90_summary": numeric_summary([row["p90"] for row in blocked_rows]),
                        "path_length_summary": numeric_summary([row["path_length"] for row in score_rows]),
                        "target_gap_summary": numeric_summary([row["target_gap"] for row in score_rows]),
                        "trial_fraction_summary": numeric_summary([row["trial_fraction"] for row in score_rows]),
                        "p90_summary": numeric_summary([row["p90"] for row in score_rows]),
                        "local_quality_summary": numeric_summary([row["local_quality"] for row in score_rows]),
                        "mean_offset_summary": numeric_summary([row["mean_offset"] for row in score_rows]),
                        "mean_normal_summary": numeric_summary([row["mean_normal"] for row in score_rows]),
                        "normal_risk_summary": numeric_summary([row["normal_risk"] for row in score_rows]),
                    }
                )
                if best_path is None:
                    break
                for candidate in best_path:
                    add_edge(candidate)
                count += len(best_path)
                path_cleanup_step_count += 1
                path_cleanup_max_length = max(path_cleanup_max_length, len(best_path))
                continue

            best_candidate = None
            best_score = None
            for candidate in candidates:
                edge = candidate["edge"]
                if edge in grown_edges:
                    continue
                if not predicate(candidate):
                    continue
                left, right = edge
                if roots[left] == roots[right]:
                    continue
                if anchor not in (roots[left], roots[right]):
                    continue
                chart = trial_chart(edge)
                p90 = chart.get("patch_unwrap_proxy_p90_edge_distortion")
                if not p90_allowed(p90):
                    continue
                trial_fraction = float(
                    chart.get("patch_unwrap_proxy_largest_component_node_fraction") or 0.0
                )
                rotation = candidate.get("rotation_degrees")
                p90_score = float("inf") if p90 is None else float(p90)
                rotation_score = float("inf") if rotation is None else float(rotation)
                if cleanup_score == "p90":
                    score = (
                        p90_score,
                        -trial_fraction,
                        rotation_score,
                        candidate["offset_ratio"],
                        -candidate["normal_agreement"],
                        edge,
                    )
                else:
                    score = (
                        -trial_fraction,
                        p90_score,
                        rotation_score,
                        candidate["offset_ratio"],
                        -candidate["normal_agreement"],
                        edge,
                    )
                if best_score is None or score < best_score:
                    best_score = score
                    best_candidate = candidate
            if best_candidate is None:
                break
            add_edge(best_candidate)
            count += 1
        stage_summaries.append(stage_state(stage, count))
        return count

    frame_missing_count = attach_cleanup(
        "frame_missing_cleanup",
        lambda candidate: not isinstance(candidate.get("rotation_degrees"), (int, float)),
    )
    high_rotation_count = attach_cleanup(
        "high_rotation_cleanup",
        lambda candidate: (
            isinstance(candidate.get("rotation_degrees"), (int, float))
            and float(candidate["rotation_degrees"]) > rotation_threshold_degrees
        ),
    )

    metrics = component_local_chart_metrics(nodes, grown_edges, base_edges, edge_quality)
    global_chart = ablation.unwrap_proxy_metrics(nodes, sorted(grown_edges))
    final_fraction = float(metrics["local_chart_largest_component_fraction"] or 0.0)

    def stall_frontier_summary() -> dict[str, Any]:
        roots, sizes = current_roots()
        anchor = largest_root(roots, sizes)
        summary: dict[str, Any] = {
            "remaining_candidate_count": 0,
            "cycle_candidate_count": 0,
            "frontier_candidate_count": 0,
            "disconnected_candidate_count": 0,
            "frontier_chart_core_count": 0,
            "frontier_frame_missing_count": 0,
            "frontier_high_rotation_count": 0,
            "frontier_cap_pass_count": 0,
            "frontier_cap_fail_count": 0,
            "frontier_missing_p90_count": 0,
            "frontier_cap_reaches_target_count": 0,
            "frontier_relaxed_reaches_target_count": 0,
            "best_frontier_relaxed_trial_fraction": 0.0,
            "best_frontier_relaxed_p90": None,
            "best_frontier_cap_trial_fraction": 0.0,
            "best_frontier_cap_p90": None,
            "best_frontier_p90_blocked_trial_fraction": 0.0,
            "best_frontier_p90_blocked_p90": None,
            "min_target_reaching_frontier_p90": None,
            "min_target_reaching_frontier_trial_fraction": None,
        }
        if anchor is None:
            return summary

        def update_best(prefix: str, trial_fraction: float, p90: Any) -> None:
            best_fraction_key = f"best_frontier_{prefix}_trial_fraction"
            best_p90_key = f"best_frontier_{prefix}_p90"
            current_fraction = float(summary[best_fraction_key] or 0.0)
            current_p90 = summary[best_p90_key]
            p90_score = float("inf") if p90 is None else float(p90)
            current_p90_score = float("inf") if current_p90 is None else float(current_p90)
            if trial_fraction > current_fraction or (
                math.isclose(trial_fraction, current_fraction) and p90_score < current_p90_score
            ):
                summary[best_fraction_key] = rounded(trial_fraction)
                summary[best_p90_key] = rounded(p90)

        def update_min_target_reaching(trial_fraction: float, p90: Any) -> None:
            if trial_fraction < target or not isinstance(p90, (int, float)):
                return
            current_p90 = summary["min_target_reaching_frontier_p90"]
            if current_p90 is None or float(p90) < float(current_p90):
                summary["min_target_reaching_frontier_p90"] = rounded(p90)
                summary["min_target_reaching_frontier_trial_fraction"] = rounded(trial_fraction)

        for candidate in candidates:
            edge = candidate["edge"]
            if edge in grown_edges:
                continue
            summary["remaining_candidate_count"] += 1
            left, right = edge
            left_root = roots[left]
            right_root = roots[right]
            if left_root == right_root:
                summary["cycle_candidate_count"] += 1
                continue
            if anchor not in (left_root, right_root):
                summary["disconnected_candidate_count"] += 1
                continue

            summary["frontier_candidate_count"] += 1
            rotation = candidate.get("rotation_degrees")
            if not isinstance(rotation, (int, float)):
                summary["frontier_frame_missing_count"] += 1
            elif float(rotation) > rotation_threshold_degrees:
                summary["frontier_high_rotation_count"] += 1
            else:
                summary["frontier_chart_core_count"] += 1

            chart = trial_chart(edge)
            p90 = chart.get("patch_unwrap_proxy_p90_edge_distortion")
            trial_fraction = float(chart.get("patch_unwrap_proxy_largest_component_node_fraction") or 0.0)
            if p90 is None:
                summary["frontier_missing_p90_count"] += 1
            if trial_fraction >= target:
                summary["frontier_relaxed_reaches_target_count"] += 1
            update_min_target_reaching(trial_fraction, p90)
            update_best("relaxed", trial_fraction, p90)
            if p90_allowed(p90):
                summary["frontier_cap_pass_count"] += 1
                if trial_fraction >= target:
                    summary["frontier_cap_reaches_target_count"] += 1
                update_best("cap", trial_fraction, p90)
            else:
                summary["frontier_cap_fail_count"] += 1
                update_best("p90_blocked", trial_fraction, p90)

        return summary

    sequence = {
        "target": rounded(target),
        "reached": final_fraction >= target,
        "rotation_threshold_degrees": rounded(rotation_threshold_degrees),
        "cleanup_score": cleanup_score,
        "cleanup_lookahead": cleanup_lookahead,
        "cleanup_branching": cleanup_branching,
        "cleanup_normal_penalty": rounded(cleanup_normal_penalty),
        "path_cleanup_step_count": path_cleanup_step_count,
        "path_cleanup_max_length": path_cleanup_max_length,
        "path_search_summaries": path_search_summaries,
        "max_p90_distortion": rounded(max_p90_distortion),
        "stage1_edge_count": stage1_count,
        "frame_missing_cleanup_edge_count": frame_missing_count,
        "high_rotation_cleanup_edge_count": high_rotation_count,
        "final_largest_component_fraction": metrics["local_chart_largest_component_fraction"],
        "final_global_p90_edge_distortion": rounded(global_chart["patch_unwrap_proxy_p90_edge_distortion"]),
        "final_local_chart_quality_p90": metrics["local_chart_quality_p90"],
        "stall_frontier": stall_frontier_summary(),
        "stage_summaries": stage_summaries,
    }
    return grown_edges, added_edges, metrics, sequence


def project_nodes(nodes: list[dict[str, Any]]):
    import numpy as np

    centroids = np.array([node["centroid"] for node in nodes], dtype=np.float64)
    centered = centroids - centroids.mean(axis=0)
    covariance = centered.T @ centered / max(1, centered.shape[0])
    _eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    coords = centered @ eigenvectors[:, -2:]
    return coords


def scale_points(coords, width: int, height: int, margin: int) -> list[tuple[float, float]]:
    import numpy as np

    if coords.shape[0] == 0:
        return []
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    scaled = (coords - mins) / span
    x_values = margin + scaled[:, 0] * (width - 2 * margin)
    y_values = margin + (1.0 - scaled[:, 1]) * (height - 2 * margin)
    return [(float(x), float(y)) for x, y in zip(x_values, y_values)]


def draw_graph_png(
    nodes: list[dict[str, Any]],
    all_edges: set[tuple[int, int]],
    base_edges: set[tuple[int, int]],
    normal_added_edges: set[tuple[int, int]],
    distortion_added_edges: set[tuple[int, int]],
    local_atlas_added_edges: set[tuple[int, int]],
    title: str,
    out_path: Path,
) -> None:
    Image, ImageDraw, ImageFont = import_pillow()
    width, height = 1180, 860
    margin = 86
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    points = scale_points(project_nodes(nodes), width, height, margin)

    ink = (28, 32, 36)
    candidate_color = (213, 216, 220)
    base_color = (34, 130, 97)
    normal_color = (52, 106, 183)
    distortion_color = (217, 120, 45)
    local_atlas_color = (126, 82, 172)
    node_color = (34, 38, 42)
    largest_color = (184, 54, 54)

    roots, sizes = component_roots(len(nodes), base_edges)
    largest_root = max(sizes, key=sizes.get) if sizes else None
    largest_base_nodes = {index for index, root in enumerate(roots) if root == largest_root}

    def draw_edges(edges: set[tuple[int, int]], color: tuple[int, int, int], width_px: int) -> None:
        for left, right in sorted(edges):
            if left >= len(points) or right >= len(points):
                continue
            draw.line((points[left], points[right]), fill=color, width=width_px)

    draw.text((24, 18), title, fill=ink, font=font)
    draw.text(
        (24, 41),
        "PCA view of patch-cell centroids. Gray: rejected candidates, green: pruned graph, "
        "blue: normal growth, orange: single-chart growth, purple: local-atlas growth.",
        fill=(78, 84, 90),
        font=font,
    )
    draw_edges(all_edges - base_edges, candidate_color, 1)
    draw_edges(base_edges, base_color, 2)
    draw_edges(normal_added_edges, normal_color, 3)
    draw_edges(distortion_added_edges, distortion_color, 3)
    draw_edges(local_atlas_added_edges, local_atlas_color, 4)
    for index, (x, y) in enumerate(points):
        radius = 3 if index in largest_base_nodes else 2
        color = largest_color if index in largest_base_nodes else node_color
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    legend_y = height - 42
    for x, color, label in (
        (24, candidate_color, "candidate"),
        (136, base_color, "pruned"),
        (230, normal_color, "normal added"),
        (360, distortion_color, "single-chart added"),
        (535, local_atlas_color, "local-atlas added"),
        (710, largest_color, "base largest component node"),
    ):
        draw.rectangle((x, legend_y, x + 20, legend_y + 12), fill=color)
        draw.text((x + 28, legend_y - 1), label, fill=ink, font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG file: {path}")
    return struct.unpack(">II", data[16:24])


def select_cluster(points, normals, threshold: int, epsilon: float, sample_limit: int, seed: int):
    import hdbscan
    import numpy as np

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=epsilon)
    labels = clusterer.fit_predict(points)
    labels_unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {
        int(label): int(size) for label, size in zip(labels_unique, counts) if int(label) != -1
    }
    selected = [(label, size) for label, size in cluster_sizes.items() if size >= threshold]
    if not selected:
        raise SystemExit(f"no HDBSCAN cluster reached threshold {threshold}")
    label, cluster_size = max(selected, key=lambda item: (item[1], -item[0]))
    indices = np.flatnonzero(labels == label)
    sample_indices = indices
    if sample_limit > 0 and indices.size > sample_limit:
        rng = np.random.default_rng(seed + label * 1009 + threshold)
        sample_indices = rng.choice(indices, size=sample_limit, replace=False)
    return {
        "label": int(label),
        "cluster_size": int(cluster_size),
        "selected_cluster_count": len(selected),
        "sample_count": int(sample_indices.size),
        "points": points[sample_indices].astype("float64", copy=False),
        "normals": normals[sample_indices].astype("float64", copy=False),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-json", type=Path, default=Path("reports/data_index_2026-05-06.json"))
    parser.add_argument("--sample", default="PHerc1667")
    parser.add_argument("--volume")
    parser.add_argument("--array", dest="array_path", default="3")
    parser.add_argument("--chunk", type=base.parse_chunk, default=[13, 4, 3])
    parser.add_argument("--side", choices=["recto", "verso"], default="recto")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--surface-detection-url", default=ablation.DEFAULT_SURFACE_DETECTION_URL)
    parser.add_argument("--blur-size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sobel-chunks", type=int, default=4)
    parser.add_argument("--sobel-overlap", type=int, default=3)
    parser.add_argument("--threshold-der", type=float, default=0.075)
    parser.add_argument("--threshold-der2", type=float, default=0.002)
    parser.add_argument("--hdbscan-epsilon", type=float, default=20.0)
    parser.add_argument("--hdbscan-threshold", type=int, default=8000)
    parser.add_argument("--hdbscan-patch-sample", type=int, default=2048)
    parser.add_argument("--patch-graph-cell-size", type=float, default=8.0)
    parser.add_argument("--patch-graph-min-cell-points", type=int, default=4)
    parser.add_argument("--patch-graph-neighbor-radius", type=int, default=1)
    parser.add_argument("--mesh-prune-min-normal-agreement", type=float, default=0.75)
    parser.add_argument("--mesh-prune-max-offset-ratio", type=float, default=0.35)
    parser.add_argument(
        "--cap-base-p90",
        action="store_true",
        help="Greedily rebuild the pruned base graph under --distortion-growth-max-p90 before cleanup.",
    )
    parser.add_argument("--growth-target", type=float, default=0.75)
    parser.add_argument("--distortion-growth-max-p90", type=float)
    parser.add_argument("--local-atlas-bridge-min-normal-agreement", type=float, default=0.50)
    parser.add_argument("--local-atlas-bridge-max-offset-ratio", type=float, default=0.20)
    parser.add_argument(
        "--local-atlas-use-transition-normal-risk",
        action="store_true",
        help=(
            "Score and gate local-atlas bridge choices by max(offset_ratio, "
            "1 - normal_agreement) instead of offset ratio alone."
        ),
    )
    parser.add_argument(
        "--local-atlas-use-broad-candidate-policy",
        action="store_true",
        help=(
            "Allow target-reaching broad candidates that pass the normal/projection "
            "policy to seed local-atlas growth even when the offset gate rejects them."
        ),
    )
    parser.add_argument(
        "--local-atlas-use-two-chart-boundary-gate",
        action="store_true",
        help=(
            "Allow cap-compliant local boundary transitions that pass the two-chart "
            "rigid projection/orientation gate to seed local-atlas growth."
        ),
    )
    parser.add_argument(
        "--local-atlas-two-chart-boundary-gate-only",
        action="store_true",
        help=(
            "When using the two-chart boundary gate, reject ordinary local-atlas "
            "and broad-candidate fallback sources."
        ),
    )
    parser.add_argument("--two-stage-chart-rotation-threshold", type=float, default=90.0)
    parser.add_argument(
        "--two-stage-cleanup-score",
        choices=[
            "coverage",
            "p90",
            "path",
            "path_quality",
            "path_quality_minimax",
            "path_quality_normal",
            "path_target_quality",
        ],
        default="coverage",
    )
    parser.add_argument("--two-stage-cleanup-lookahead", type=int, default=1)
    parser.add_argument("--two-stage-cleanup-branching", type=int, default=32)
    parser.add_argument("--two-stage-cleanup-normal-penalty", type=float, default=2.0)
    parser.add_argument(
        "--candidate-diagnostics-only",
        action="store_true",
        help=(
            "Compute the capped base graph and candidate diagnostics, then skip "
            "normal, distortion, local-atlas, and two-stage growth diagnostics."
        ),
    )
    parser.add_argument(
        "--local-atlas-only",
        action="store_true",
        help=(
            "Compute local-atlas growth and candidate diagnostics, then skip "
            "normal, distortion, and two-stage growth diagnostics."
        ),
    )
    parser.add_argument("--candidate-diagnostic-limit", type=int, default=24)
    parser.add_argument(
        "--base-cap-alternate-route-max-length",
        type=int,
        default=0,
        help=(
            "When --cap-base-p90 skips base bridges, search scalar-only alternate component "
            "routes up to this many edges. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--base-cap-alternate-route-branching",
        type=int,
        default=64,
        help="Per-component candidate branch cap for --base-cap-alternate-route-max-length.",
    )
    parser.add_argument(
        "--base-cap-component-probe",
        action="store_true",
        help=(
            "When --cap-base-p90 skips base bridges, record scalar local-component "
            "and transition metrics for a multi-chart bridge-repair diagnostic."
        ),
    )
    parser.add_argument(
        "--base-cap-boundary-prototype",
        action="store_true",
        help=(
            "When --cap-base-p90 skips base bridges, greedily accept cap-compliant "
            "local chart-boundary transitions and report scalar atlas coverage."
        ),
    )
    parser.add_argument(
        "--base-cap-boundary-require-two-chart-gate",
        action="store_true",
        help=(
            "Require skipped bridge boundary transitions to pass the two-chart rigid "
            "projection/orientation selection gate before the boundary prototype accepts them."
        ),
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--png-out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if args.distortion_growth_max_p90 is not None and args.distortion_growth_max_p90 < 0:
        raise SystemExit("--distortion-growth-max-p90 must be non-negative")
    if args.cap_base_p90 and args.distortion_growth_max_p90 is None:
        raise SystemExit("--cap-base-p90 requires --distortion-growth-max-p90")
    if not 0 <= args.local_atlas_bridge_min_normal_agreement <= 1:
        raise SystemExit("--local-atlas-bridge-min-normal-agreement must be between 0 and 1")
    if args.local_atlas_bridge_max_offset_ratio < 0:
        raise SystemExit("--local-atlas-bridge-max-offset-ratio must be non-negative")
    if not 0 <= args.two_stage_chart_rotation_threshold <= 180:
        raise SystemExit("--two-stage-chart-rotation-threshold must be between 0 and 180")
    if args.two_stage_cleanup_lookahead < 1:
        raise SystemExit("--two-stage-cleanup-lookahead must be positive")
    if args.two_stage_cleanup_branching < 1:
        raise SystemExit("--two-stage-cleanup-branching must be positive")
    if args.two_stage_cleanup_normal_penalty < 0:
        raise SystemExit("--two-stage-cleanup-normal-penalty must be non-negative")
    if args.candidate_diagnostic_limit < 1:
        raise SystemExit("--candidate-diagnostic-limit must be positive")
    if args.base_cap_alternate_route_max_length < 0:
        raise SystemExit("--base-cap-alternate-route-max-length must be non-negative")
    if args.base_cap_alternate_route_branching < 1:
        raise SystemExit("--base-cap-alternate-route-branching must be positive")
    if args.base_cap_alternate_route_max_length and not args.cap_base_p90:
        raise SystemExit("--base-cap-alternate-route-max-length requires --cap-base-p90")
    if args.base_cap_component_probe and not args.cap_base_p90:
        raise SystemExit("--base-cap-component-probe requires --cap-base-p90")
    if args.base_cap_boundary_prototype and not args.cap_base_p90:
        raise SystemExit("--base-cap-boundary-prototype requires --cap-base-p90")
    if args.local_atlas_use_two_chart_boundary_gate and not args.cap_base_p90:
        raise SystemExit("--local-atlas-use-two-chart-boundary-gate requires --cap-base-p90")
    if (
        args.local_atlas_two_chart_boundary_gate_only
        and not args.local_atlas_use_two_chart_boundary_gate
    ):
        raise SystemExit(
            "--local-atlas-two-chart-boundary-gate-only requires "
            "--local-atlas-use-two-chart-boundary-gate"
        )
    if (
        args.base_cap_boundary_require_two_chart_gate
        and not args.base_cap_boundary_prototype
    ):
        raise SystemExit(
            "--base-cap-boundary-require-two-chart-gate requires --base-cap-boundary-prototype"
        )
    if args.candidate_diagnostics_only and args.local_atlas_only:
        raise SystemExit("--local-atlas-only cannot be combined with --candidate-diagnostics-only")

    torch = ablation.import_torch()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    source_text = ablation.fetch_text(args.surface_detection_url, args.timeout)
    source_sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    module = ablation.load_surface_detection(source_text)
    index = json.loads(args.index_json.read_text(encoding="utf-8"))
    volume_entry = base.select_volume(index, args.sample, args.volume)
    array = base.select_array(volume_entry, args.array_path)
    if array.get("compressor") is not None:
        raise ValueError(f"compressed Zarr chunks are not supported: {array.get('compressor')!r}")

    url = base.chunk_url(volume_entry["url"], array["path"], args.chunk)
    chunk_bytes = base.fetch_bytes(url, args.timeout)
    chunk_sha256 = hashlib.sha256(chunk_bytes).hexdigest()
    chunk = base.decode_uncompressed_chunk(chunk_bytes, array)
    roi = base.center_crop(chunk, args.crop_size)
    del chunk

    volume = torch.from_numpy(roi).to(device=args.device, dtype=dtype)
    reference_vector = torch.tensor([1.0, 0.0, 0.0], device=args.device, dtype=torch.float32)
    start = time.perf_counter()
    recto, verso = module.surface_detection(
        volume,
        reference_vector,
        blur_size=args.blur_size,
        sobel_chunks=args.sobel_chunks,
        sobel_overlap=args.sobel_overlap,
        window_size=args.window_size,
        stride=args.stride,
        threshold_der=args.threshold_der,
        threshold_der2=args.threshold_der2,
        convert_to_numpy=False,
    )
    if args.device == "cuda":
        torch.cuda.synchronize()
        peak_memory_bytes = int(torch.cuda.max_memory_allocated())
    else:
        peak_memory_bytes = 0
    runtime_seconds = time.perf_counter() - start
    side_output = recto if args.side == "recto" else verso
    points = side_output[0].detach().cpu().numpy()
    normals = side_output[1].detach().cpu().numpy()

    cluster = select_cluster(
        points,
        normals,
        threshold=args.hdbscan_threshold,
        epsilon=args.hdbscan_epsilon,
        sample_limit=args.hdbscan_patch_sample,
        seed=args.seed,
    )
    nodes, edges, edge_quality = build_patch_graph(
        cluster["points"],
        cluster["normals"],
        cell_size=args.patch_graph_cell_size,
        min_points_per_cell=args.patch_graph_min_cell_points,
        neighbor_radius=args.patch_graph_neighbor_radius,
    )
    all_edges = {sorted_edge(edge) for edge in edges}
    raw_base_edges = set(
        pruned_edges(
            edge_quality,
            min_normal_agreement=args.mesh_prune_min_normal_agreement,
            max_offset_ratio=args.mesh_prune_max_offset_ratio,
        )
    )
    base_p90_cap_summary: dict[str, Any] = {
        "enabled": False,
        "input_base_edge_count": len(raw_base_edges),
    }
    if args.cap_base_p90:
        base_edges, base_p90_cap_summary = cap_base_edges_by_incremental_p90(
            nodes,
            raw_base_edges,
            edge_quality,
            max_p90_distortion=float(args.distortion_growth_max_p90),
            diagnostic_limit=args.candidate_diagnostic_limit,
            all_edges=edges,
            alternate_route_max_length=args.base_cap_alternate_route_max_length,
            alternate_route_branching=args.base_cap_alternate_route_branching,
            component_probe=args.base_cap_component_probe,
            boundary_prototype=args.base_cap_boundary_prototype,
            boundary_require_two_chart_gate=args.base_cap_boundary_require_two_chart_gate,
            boundary_target=args.growth_target,
        )
    else:
        base_edges = raw_base_edges
    base_chart = ablation.unwrap_proxy_metrics(nodes, sorted(base_edges))
    base_fraction = largest_fraction(len(nodes), base_edges)
    if args.candidate_diagnostics_only:
        normal_grown_edges = set(base_edges)
        normal_added_edges: set[tuple[int, int]] = set()
        distortion_grown_edges = set(base_edges)
        distortion_added_edges: set[tuple[int, int]] = set()
        local_atlas_grown_edges = set(base_edges)
        local_atlas_added_edges: set[tuple[int, int]] = set()
        two_stage_chart_grown_edges = set(base_edges)
        two_stage_chart_added_edges: set[tuple[int, int]] = set()
        normal_chart = base_chart
        distortion_chart = base_chart
        local_atlas_global_chart = base_chart
        two_stage_chart_global_chart = base_chart
        local_atlas_metrics = candidate_diagnostics_only_local_metrics()
        two_stage_chart_metrics = candidate_diagnostics_only_local_metrics()
        local_atlas_sequence = candidate_diagnostics_only_sequence(
            args.growth_target,
            base_fraction,
        )
        two_stage_chart_sequence = candidate_diagnostics_only_sequence(
            args.growth_target,
            base_fraction,
        )
    else:
        if args.local_atlas_only:
            normal_grown_edges = set(base_edges)
            normal_added_edges: set[tuple[int, int]] = set()
            distortion_grown_edges = set(base_edges)
            distortion_added_edges: set[tuple[int, int]] = set()
        else:
            normal_grown_edges, normal_added_edges = grow_normal_edges(
                len(nodes), edges, base_edges, edge_quality, target=args.growth_target
            )
            distortion_grown_edges, distortion_added_edges = grow_distortion_edges(
                nodes,
                edges,
                base_edges,
                edge_quality,
                target=args.growth_target,
                max_p90_distortion=args.distortion_growth_max_p90,
            )
        (
            local_atlas_grown_edges,
            local_atlas_added_edges,
            local_atlas_metrics,
            local_atlas_sequence,
        ) = grow_local_atlas_edges_with_sequence(
            nodes,
            edges,
            base_edges,
            edge_quality,
            target=args.growth_target,
            max_bridge_offset_ratio=args.local_atlas_bridge_max_offset_ratio,
            min_bridge_normal_agreement=args.local_atlas_bridge_min_normal_agreement,
            max_p90_distortion=args.distortion_growth_max_p90,
            use_transition_normal_risk_quality=args.local_atlas_use_transition_normal_risk,
            use_broad_candidate_policy=args.local_atlas_use_broad_candidate_policy,
            use_two_chart_boundary_gate=args.local_atlas_use_two_chart_boundary_gate,
            two_chart_boundary_gate_only=args.local_atlas_two_chart_boundary_gate_only,
        )
        if args.local_atlas_only:
            two_stage_chart_grown_edges = set(base_edges)
            two_stage_chart_added_edges: set[tuple[int, int]] = set()
            two_stage_chart_metrics = candidate_diagnostics_only_local_metrics()
            two_stage_chart_sequence = candidate_diagnostics_only_sequence(
                args.growth_target,
                base_fraction,
                reason="local_atlas_only",
            )
        else:
            (
                two_stage_chart_grown_edges,
                two_stage_chart_added_edges,
                two_stage_chart_metrics,
                two_stage_chart_sequence,
            ) = grow_two_stage_chart_edges_with_sequence(
                nodes,
                edges,
                base_edges,
                edge_quality,
                target=args.growth_target,
                max_p90_distortion=args.distortion_growth_max_p90,
                rotation_threshold_degrees=args.two_stage_chart_rotation_threshold,
                cleanup_score=args.two_stage_cleanup_score,
                cleanup_lookahead=args.two_stage_cleanup_lookahead,
                cleanup_branching=args.two_stage_cleanup_branching,
                cleanup_normal_penalty=args.two_stage_cleanup_normal_penalty,
            )
        normal_chart = ablation.unwrap_proxy_metrics(nodes, sorted(normal_grown_edges))
        distortion_chart = ablation.unwrap_proxy_metrics(nodes, sorted(distortion_grown_edges))
        local_atlas_global_chart = ablation.unwrap_proxy_metrics(nodes, sorted(local_atlas_grown_edges))
        two_stage_chart_global_chart = ablation.unwrap_proxy_metrics(
            nodes,
            sorted(two_stage_chart_grown_edges),
        )

    title = (
        f"{args.sample} chunk={','.join(map(str, args.chunk))} {args.side} seed={args.seed} "
        f"target={args.growth_target:.2f} neighbor={args.patch_graph_neighbor_radius}"
    )
    draw_graph_png(
        nodes,
        all_edges,
        base_edges,
        normal_added_edges,
        distortion_added_edges,
        local_atlas_added_edges,
        title,
        args.png_out,
    )
    width, height = png_dimensions(args.png_out)

    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "policy": (
            "Fetched one bounded public OME-Zarr chunk transiently, rendered a raster-only patch-graph QA "
            "PNG, and wrote scalar summary JSON only. Raw chunks, extracted volumes, point clouds, HDBSCAN "
            "labels, patch-graph tables, meshes, flattened coordinates, component IDs, bridge endpoints, "
            "predictions, letters, titles, and ink detections were not saved."
        ),
        "source": {
            "index_json": str(args.index_json),
            "sample_id": args.sample,
            "volume_name": volume_entry["name"],
            "volume_url": volume_entry["url"],
            "array_path": array["path"],
            "chunk": args.chunk,
            "chunk_url": url,
            "chunk_byte_count": len(chunk_bytes),
            "chunk_sha256": chunk_sha256,
            "roi_shape": [args.crop_size, args.crop_size, args.crop_size],
        },
        "upstream": {
            "surface_detection_url": args.surface_detection_url,
            "surface_detection_sha256": source_sha256,
        },
        "parameters": {
            "side": args.side,
            "blur_size": args.blur_size,
            "seed": args.seed,
            "hdbscan_epsilon": args.hdbscan_epsilon,
            "hdbscan_threshold": args.hdbscan_threshold,
            "hdbscan_patch_sample": args.hdbscan_patch_sample,
            "patch_graph_cell_size": args.patch_graph_cell_size,
            "patch_graph_min_cell_points": args.patch_graph_min_cell_points,
            "patch_graph_neighbor_radius": args.patch_graph_neighbor_radius,
            "mesh_prune_min_normal_agreement": args.mesh_prune_min_normal_agreement,
            "mesh_prune_max_offset_ratio": args.mesh_prune_max_offset_ratio,
            "cap_base_p90": args.cap_base_p90,
            "growth_target": args.growth_target,
            "distortion_growth_max_p90": args.distortion_growth_max_p90,
            "local_atlas_bridge_min_normal_agreement": args.local_atlas_bridge_min_normal_agreement,
            "local_atlas_bridge_max_offset_ratio": args.local_atlas_bridge_max_offset_ratio,
            "local_atlas_use_transition_normal_risk": args.local_atlas_use_transition_normal_risk,
            "local_atlas_use_broad_candidate_policy": args.local_atlas_use_broad_candidate_policy,
            "local_atlas_use_two_chart_boundary_gate": (
                args.local_atlas_use_two_chart_boundary_gate
            ),
            "local_atlas_two_chart_boundary_gate_only": (
                args.local_atlas_two_chart_boundary_gate_only
            ),
            "two_stage_chart_rotation_threshold": args.two_stage_chart_rotation_threshold,
            "two_stage_cleanup_score": args.two_stage_cleanup_score,
            "two_stage_cleanup_lookahead": args.two_stage_cleanup_lookahead,
            "two_stage_cleanup_branching": args.two_stage_cleanup_branching,
            "two_stage_cleanup_normal_penalty": args.two_stage_cleanup_normal_penalty,
            "candidate_diagnostics_only": args.candidate_diagnostics_only,
            "local_atlas_only": args.local_atlas_only,
            "candidate_diagnostic_limit": args.candidate_diagnostic_limit,
            "base_cap_alternate_route_max_length": args.base_cap_alternate_route_max_length,
            "base_cap_alternate_route_branching": args.base_cap_alternate_route_branching,
            "base_cap_component_probe": args.base_cap_component_probe,
            "base_cap_boundary_prototype": args.base_cap_boundary_prototype,
            "base_cap_boundary_require_two_chart_gate": args.base_cap_boundary_require_two_chart_gate,
        },
        "environment": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": args.device,
            "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
            "dtype": args.dtype,
            "runtime_seconds": runtime_seconds,
            "peak_memory_bytes": peak_memory_bytes,
        },
        "cluster": {
            "side_point_count": int(points.shape[0]),
            "selected_cluster_count": cluster["selected_cluster_count"],
            "selected_label": cluster["label"],
            "selected_cluster_size": cluster["cluster_size"],
            "sampled_point_count": cluster["sample_count"],
        },
        "graph_summary": {
            "node_count": len(nodes),
            "edge_count": len(all_edges),
            "base_edge_count": len(base_edges),
            "base_largest_component_fraction": largest_fraction(len(nodes), base_edges),
            "normal_added_edge_count": len(normal_added_edges),
            "normal_largest_component_fraction": largest_fraction(len(nodes), normal_grown_edges),
            "normal_p90_edge_distortion": normal_chart["patch_unwrap_proxy_p90_edge_distortion"],
            "distortion_added_edge_count": len(distortion_added_edges),
            "distortion_largest_component_fraction": largest_fraction(len(nodes), distortion_grown_edges),
            "distortion_p90_edge_distortion": distortion_chart["patch_unwrap_proxy_p90_edge_distortion"],
            "local_atlas_added_edge_count": len(local_atlas_added_edges),
            "local_atlas_largest_component_fraction": largest_fraction(len(nodes), local_atlas_grown_edges),
            "local_atlas_global_p90_edge_distortion": local_atlas_global_chart[
                "patch_unwrap_proxy_p90_edge_distortion"
            ],
            "two_stage_chart_added_edge_count": len(two_stage_chart_added_edges),
            "two_stage_chart_largest_component_fraction": largest_fraction(
                len(nodes),
                two_stage_chart_grown_edges,
            ),
            "two_stage_chart_global_p90_edge_distortion": two_stage_chart_global_chart[
                "patch_unwrap_proxy_p90_edge_distortion"
            ],
            **local_atlas_metrics,
            "two_stage_chart_local_quality_p90": two_stage_chart_metrics["local_chart_quality_p90"],
            "two_stage_chart_p90_bridge_offset_ratio": two_stage_chart_metrics[
                "local_chart_p90_bridge_offset_ratio"
            ],
            "two_stage_chart_p10_bridge_normal_agreement": two_stage_chart_metrics[
                "local_chart_p10_bridge_normal_agreement"
            ],
            "base_p90_edge_distortion": base_chart["patch_unwrap_proxy_p90_edge_distortion"],
        },
        "base_p90_cap_summary": base_p90_cap_summary,
        "local_atlas_sequence": local_atlas_sequence,
        "two_stage_chart_sequence": two_stage_chart_sequence,
        "candidate_diagnostics": candidate_bridge_diagnostics(
            nodes,
            edges,
            base_edges,
            edge_quality,
            max_p90_distortion=args.distortion_growth_max_p90,
            target=args.growth_target,
            limit=args.candidate_diagnostic_limit,
        ),
        "chart_transition_budget": chart_transition_budget(
            nodes,
            edges,
            base_edges,
            edge_quality,
            max_p90_distortion=args.distortion_growth_max_p90,
        ),
        "png": {
            "path": str(args.png_out),
            "width": width,
            "height": height,
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.png_out} ({width}x{height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
