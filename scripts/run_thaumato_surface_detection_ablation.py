#!/usr/bin/env python3
"""Run seeded upstream Thaumato surface_detection blur-size ablations."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import tempfile
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import benchmark_thaumato_blur as base


DEFAULT_SURFACE_DETECTION_URL = (
    "https://raw.githubusercontent.com/ScrollPrize/villa/"
    "09b1fee00ffbd4b52102f7a9b27809582c52153d/"
    "thaumato-anakalyptor/ThaumatoAnakalyptor/surface_detection.py"
)
UNWRAP_GROWTH_TARGETS = (0.50, 0.75, 0.90)


def unwrap_growth_suffix(target: float) -> str:
    return str(int(round(100 * target)))


def prefixed_growth_metric_names(prefix: str) -> tuple[str, ...]:
    names: list[str] = []
    for target in UNWRAP_GROWTH_TARGETS:
        suffix = unwrap_growth_suffix(target)
        names.extend(
            [
                f"{prefix}_{suffix}_reached",
                f"{prefix}_{suffix}_added_edge_count",
                f"{prefix}_{suffix}_node_count",
                f"{prefix}_{suffix}_edge_count",
                f"{prefix}_{suffix}_triangle_count",
                f"{prefix}_{suffix}_largest_component_node_fraction",
                f"{prefix}_{suffix}_mean_edge_distortion",
                f"{prefix}_{suffix}_p90_edge_distortion",
                f"{prefix}_{suffix}_mean_triangle_area_ratio",
                f"{prefix}_{suffix}_p10_triangle_area_ratio",
                f"{prefix}_{suffix}_mean_added_edge_normal_agreement",
                f"{prefix}_{suffix}_p90_added_edge_offset_ratio",
            ]
        )
    return tuple(names)


def unwrap_growth_metric_names() -> tuple[str, ...]:
    return prefixed_growth_metric_names("patch_unwrap_growth")


def unwrap_distortion_growth_metric_names() -> tuple[str, ...]:
    return prefixed_growth_metric_names("patch_unwrap_distortion_growth")


def local_atlas_growth_metric_names() -> tuple[str, ...]:
    names: list[str] = []
    for target in UNWRAP_GROWTH_TARGETS:
        suffix = unwrap_growth_suffix(target)
        prefix = f"patch_local_atlas_growth_{suffix}"
        names.extend(
            [
                f"{prefix}_reached",
                f"{prefix}_added_edge_count",
                f"{prefix}_node_count",
                f"{prefix}_edge_count",
                f"{prefix}_largest_component_node_fraction",
                f"{prefix}_local_chart_p90_internal_edge_distortion",
                f"{prefix}_bridge_edge_count",
                f"{prefix}_p90_bridge_offset_ratio",
                f"{prefix}_p10_bridge_normal_agreement",
                f"{prefix}_p10_bridge_chart_normal_agreement",
                f"{prefix}_quality_p90",
                f"{prefix}_global_p90_edge_distortion",
            ]
        )
    return tuple(names)


def import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by CLI environment checks.
        raise SystemExit(
            "Missing dependency: torch. Install a CUDA-enabled wheel in the experiment venv, for example "
            "`python -m pip install torch --index-url https://download.pytorch.org/whl/cu128`."
        ) from exc
    return torch


def fetch_text(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "vesuvius-2026-surface-ablation/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def load_surface_detection(source_text: str):
    with tempfile.TemporaryDirectory(prefix="vesuvius-surface-detection-") as tmpdir:
        path = Path(tmpdir) / "surface_detection.py"
        path.write_text(source_text, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("surface_detection_snapshot", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


def parse_int_list(value: str) -> list[int]:
    try:
        values = [int(part) for part in value.split(",") if part]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_float_list(value: str) -> list[float]:
    try:
        values = [float(part) for part in value.split(",") if part]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one number")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all radii must be positive")
    return values


def run_one(module, volume, reference_vector, args, blur_size: int, seed: int) -> dict[str, Any]:
    torch = import_torch()
    torch.manual_seed(seed)
    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    recto, verso = module.surface_detection(
        volume,
        reference_vector,
        blur_size=blur_size,
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
    elapsed = time.perf_counter() - start
    recto_count = int(recto[0].shape[0])
    verso_count = int(verso[0].shape[0])
    return {
        "blur_size": blur_size,
        "seed": seed,
        "runtime_seconds": elapsed,
        "peak_memory_bytes": peak_memory_bytes,
        "recto_count": recto_count,
        "verso_count": verso_count,
        "total_count": recto_count + verso_count,
        "recto_metrics": point_metrics(
            recto[0],
            recto[1],
            volume.shape,
            args.nn_sample,
            seed,
            args.connectivity_radii,
            hdbscan_metrics=args.hdbscan_metrics,
            hdbscan_epsilon=args.hdbscan_epsilon,
            hdbscan_thresholds=args.hdbscan_selected_thresholds,
            hdbscan_patch_quality=args.hdbscan_patch_quality,
            hdbscan_patch_sample=args.hdbscan_patch_sample,
            hdbscan_local_continuity=args.hdbscan_local_continuity,
            hdbscan_continuity_neighbors=args.hdbscan_continuity_neighbors,
            hdbscan_patch_graph=args.hdbscan_patch_graph,
            patch_graph_cell_size=args.hdbscan_patch_graph_cell_size,
            patch_graph_min_cell_points=args.hdbscan_patch_graph_min_cell_points,
            patch_graph_neighbor_radius=args.hdbscan_patch_graph_neighbor_radius,
            hdbscan_mesh_proxy=args.hdbscan_mesh_proxy,
            hdbscan_mesh_prune=args.hdbscan_mesh_prune,
            mesh_prune_min_normal_agreement=args.hdbscan_mesh_prune_min_normal_agreement,
            mesh_prune_max_offset_ratio=args.hdbscan_mesh_prune_max_offset_ratio,
            hdbscan_unwrap_proxy=args.hdbscan_unwrap_proxy,
            hdbscan_unwrap_growth_proxy=args.hdbscan_unwrap_growth_proxy,
            hdbscan_unwrap_distortion_growth_proxy=args.hdbscan_unwrap_distortion_growth_proxy,
            hdbscan_unwrap_distortion_growth_candidate_limit=(
                args.hdbscan_unwrap_distortion_growth_candidate_limit
            ),
            hdbscan_unwrap_distortion_growth_bridge_only=(
                args.hdbscan_unwrap_distortion_growth_bridge_only
            ),
            hdbscan_unwrap_distortion_growth_max_p90=args.hdbscan_unwrap_distortion_growth_max_p90,
            hdbscan_local_atlas_growth_proxy=args.hdbscan_local_atlas_growth_proxy,
            hdbscan_local_atlas_growth_min_normal_agreement=(
                args.hdbscan_local_atlas_growth_min_normal_agreement
            ),
            hdbscan_local_atlas_growth_max_offset_ratio=(
                args.hdbscan_local_atlas_growth_max_offset_ratio
            ),
            hdbscan_local_atlas_growth_min_chart_normal_agreement=(
                args.hdbscan_local_atlas_growth_min_chart_normal_agreement
            ),
        ),
        "verso_metrics": point_metrics(
            verso[0],
            verso[1],
            volume.shape,
            args.nn_sample,
            seed,
            args.connectivity_radii,
            hdbscan_metrics=args.hdbscan_metrics,
            hdbscan_epsilon=args.hdbscan_epsilon,
            hdbscan_thresholds=args.hdbscan_selected_thresholds,
            hdbscan_patch_quality=args.hdbscan_patch_quality,
            hdbscan_patch_sample=args.hdbscan_patch_sample,
            hdbscan_local_continuity=args.hdbscan_local_continuity,
            hdbscan_continuity_neighbors=args.hdbscan_continuity_neighbors,
            hdbscan_patch_graph=args.hdbscan_patch_graph,
            patch_graph_cell_size=args.hdbscan_patch_graph_cell_size,
            patch_graph_min_cell_points=args.hdbscan_patch_graph_min_cell_points,
            patch_graph_neighbor_radius=args.hdbscan_patch_graph_neighbor_radius,
            hdbscan_mesh_proxy=args.hdbscan_mesh_proxy,
            hdbscan_mesh_prune=args.hdbscan_mesh_prune,
            mesh_prune_min_normal_agreement=args.hdbscan_mesh_prune_min_normal_agreement,
            mesh_prune_max_offset_ratio=args.hdbscan_mesh_prune_max_offset_ratio,
            hdbscan_unwrap_proxy=args.hdbscan_unwrap_proxy,
            hdbscan_unwrap_growth_proxy=args.hdbscan_unwrap_growth_proxy,
            hdbscan_unwrap_distortion_growth_proxy=args.hdbscan_unwrap_distortion_growth_proxy,
            hdbscan_unwrap_distortion_growth_candidate_limit=(
                args.hdbscan_unwrap_distortion_growth_candidate_limit
            ),
            hdbscan_unwrap_distortion_growth_bridge_only=(
                args.hdbscan_unwrap_distortion_growth_bridge_only
            ),
            hdbscan_unwrap_distortion_growth_max_p90=args.hdbscan_unwrap_distortion_growth_max_p90,
            hdbscan_local_atlas_growth_proxy=args.hdbscan_local_atlas_growth_proxy,
            hdbscan_local_atlas_growth_min_normal_agreement=(
                args.hdbscan_local_atlas_growth_min_normal_agreement
            ),
            hdbscan_local_atlas_growth_max_offset_ratio=(
                args.hdbscan_local_atlas_growth_max_offset_ratio
            ),
            hdbscan_local_atlas_growth_min_chart_normal_agreement=(
                args.hdbscan_local_atlas_growth_min_chart_normal_agreement
            ),
        ),
    }


def tensor_percentiles(values, percentiles: list[float]) -> dict[str, float]:
    torch = import_torch()
    if values.numel() == 0:
        return {f"p{percentile:g}": 0.0 for percentile in percentiles}
    q = torch.tensor([percentile / 100.0 for percentile in percentiles], device=values.device)
    quantiles = torch.quantile(values.float(), q)
    return {f"p{percentile:g}": float(value.item()) for percentile, value in zip(percentiles, quantiles)}


def point_metrics(
    coords,
    normals,
    roi_shape: tuple[int, int, int],
    sample_limit: int,
    seed: int,
    connectivity_radii: list[float],
    hdbscan_metrics: bool = False,
    hdbscan_epsilon: float = 20.0,
    hdbscan_thresholds: list[int] | None = None,
    hdbscan_patch_quality: bool = False,
    hdbscan_patch_sample: int = 2048,
    hdbscan_local_continuity: bool = False,
    hdbscan_continuity_neighbors: int = 16,
    hdbscan_patch_graph: bool = False,
    patch_graph_cell_size: float = 8.0,
    patch_graph_min_cell_points: int = 4,
    patch_graph_neighbor_radius: int = 1,
    hdbscan_mesh_proxy: bool = False,
    hdbscan_mesh_prune: bool = False,
    mesh_prune_min_normal_agreement: float = 0.75,
    mesh_prune_max_offset_ratio: float = 0.35,
    hdbscan_unwrap_proxy: bool = False,
    hdbscan_unwrap_growth_proxy: bool = False,
    hdbscan_unwrap_distortion_growth_proxy: bool = False,
    hdbscan_unwrap_distortion_growth_candidate_limit: int = 0,
    hdbscan_unwrap_distortion_growth_bridge_only: bool = False,
    hdbscan_unwrap_distortion_growth_max_p90: float | None = None,
    hdbscan_local_atlas_growth_proxy: bool = False,
    hdbscan_local_atlas_growth_min_normal_agreement: float = 0.50,
    hdbscan_local_atlas_growth_max_offset_ratio: float = 0.20,
    hdbscan_local_atlas_growth_min_chart_normal_agreement: float = 0.0,
) -> dict[str, Any]:
    torch = import_torch()
    count = int(coords.shape[0])
    if count == 0:
        return {
            "count": 0,
            "density": 0.0,
            "bbox_min": None,
            "bbox_max": None,
            "centroid": None,
            "orientation_coherence": 0.0,
            "orientation_eigenvalues": [],
            "normal_norm_mean": 0.0,
            "normal_norm_std": 0.0,
            "nearest_neighbor": None,
            "components": None,
            "radius20_components": None,
            "radius_sweep_components": [],
            "hdbscan_components": None,
        }

    coords_f = coords.float()
    normals_f = normals.float()
    normal_norms = torch.linalg.norm(normals_f, dim=1)
    unit_normals = normals_f / torch.clamp(normal_norms[:, None], min=1e-6)
    orientation_matrix = unit_normals.T @ unit_normals / count
    eigenvalues = torch.linalg.eigvalsh(orientation_matrix).float()

    nearest = None
    if count > 1 and sample_limit > 0:
        generator = torch.Generator(device=coords.device)
        generator.manual_seed(seed)
        sample_count = min(sample_limit, count)
        indices = torch.randperm(count, device=coords.device, generator=generator)[:sample_count]
        sample = coords_f[indices]
        distances = torch.cdist(sample, coords_f)
        distances[distances == 0] = float("inf")
        min_distances = distances.min(dim=1).values
        finite = min_distances[torch.isfinite(min_distances)]
        if finite.numel() > 0:
            nearest = {
                "sample_count": int(sample_count),
                "mean": float(finite.mean().item()),
                **tensor_percentiles(finite, [50, 90, 99]),
            }

    coords_list = coords.detach().cpu().tolist()
    radius_coords_list = coords_list
    if sample_limit > 0 and count > sample_limit:
        generator = torch.Generator(device=coords.device)
        generator.manual_seed(seed + 10_000)
        radius_indices = torch.randperm(count, device=coords.device, generator=generator)[:sample_limit]
        radius_coords_list = [coords_list[index] for index in radius_indices.cpu().tolist()]
    radius_sweep = [
        radius_component_metrics(radius_coords_list, radius=radius, min_cluster_size=8000, source_count=count)
        for radius in connectivity_radii
    ]
    radius20 = next((item for item in radius_sweep if item["radius"] == 20.0), None)
    if radius20 is None:
        radius20 = radius_component_metrics(
            radius_coords_list,
            radius=20.0,
            min_cluster_size=8000,
            source_count=count,
        )

    hdbscan_components = None
    if hdbscan_metrics:
        hdbscan_components = hdbscan_component_metrics(
            coords_list,
            epsilon=hdbscan_epsilon,
            selected_thresholds=hdbscan_thresholds or [512, 8000],
            normals=normals.detach().cpu().tolist() if hdbscan_patch_quality else None,
            patch_quality=hdbscan_patch_quality,
            patch_sample_limit=hdbscan_patch_sample,
            local_continuity=hdbscan_local_continuity,
            continuity_neighbors=hdbscan_continuity_neighbors,
            patch_graph=hdbscan_patch_graph,
            patch_graph_cell_size=patch_graph_cell_size,
            patch_graph_min_cell_points=patch_graph_min_cell_points,
            patch_graph_neighbor_radius=patch_graph_neighbor_radius,
            mesh_proxy=hdbscan_mesh_proxy,
            mesh_prune=hdbscan_mesh_prune,
            mesh_prune_min_normal_agreement=mesh_prune_min_normal_agreement,
            mesh_prune_max_offset_ratio=mesh_prune_max_offset_ratio,
            unwrap_proxy=hdbscan_unwrap_proxy,
            unwrap_growth_proxy=hdbscan_unwrap_growth_proxy,
            unwrap_distortion_growth_proxy=hdbscan_unwrap_distortion_growth_proxy,
            unwrap_distortion_growth_candidate_limit=hdbscan_unwrap_distortion_growth_candidate_limit,
            unwrap_distortion_growth_bridge_only=hdbscan_unwrap_distortion_growth_bridge_only,
            unwrap_distortion_growth_max_p90=hdbscan_unwrap_distortion_growth_max_p90,
            local_atlas_growth_proxy=hdbscan_local_atlas_growth_proxy,
            local_atlas_growth_min_normal_agreement=(
                hdbscan_local_atlas_growth_min_normal_agreement
            ),
            local_atlas_growth_max_offset_ratio=hdbscan_local_atlas_growth_max_offset_ratio,
            local_atlas_growth_min_chart_normal_agreement=(
                hdbscan_local_atlas_growth_min_chart_normal_agreement
            ),
            seed=seed,
        )

    return {
        "count": count,
        "density": count / float(roi_shape[0] * roi_shape[1] * roi_shape[2]),
        "bbox_min": [int(value) for value in coords.min(dim=0).values.tolist()],
        "bbox_max": [int(value) for value in coords.max(dim=0).values.tolist()],
        "centroid": [float(value) for value in coords_f.mean(dim=0).tolist()],
        "orientation_coherence": float(eigenvalues[-1].item()),
        "orientation_eigenvalues": [float(value) for value in eigenvalues.tolist()],
        "normal_norm_mean": float(normal_norms.mean().item()),
        "normal_norm_std": float(normal_norms.std(unbiased=False).item()),
        "nearest_neighbor": nearest,
        "components": connected_component_metrics(coords_list),
        "radius20_components": radius20,
        "radius_sweep_components": radius_sweep,
        "hdbscan_components": hdbscan_components,
    }


def connected_component_metrics(coords: list[list[int]]) -> dict[str, Any]:
    points = {(int(z), int(y), int(x)) for z, y, x in coords}
    count = len(points)
    if count == 0:
        return {
            "component_count": 0,
            "largest_component_size": 0,
            "largest_component_fraction": 0.0,
            "singleton_count": 0,
            "singleton_fraction": 0.0,
            "component_size_p50": 0.0,
            "component_size_p90": 0.0,
            "component_size_p99": 0.0,
        }

    offsets = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) != (0, 0, 0)
    ]
    remaining = set(points)
    sizes: list[int] = []
    while remaining:
        start = remaining.pop()
        queue: deque[tuple[int, int, int]] = deque([start])
        size = 1
        while queue:
            z, y, x = queue.popleft()
            for dz, dy, dx in offsets:
                neighbor = (z + dz, y + dy, x + dx)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    size += 1
        sizes.append(size)

    sizes.sort()
    largest = sizes[-1]
    singleton_count = sum(1 for size in sizes if size == 1)
    return {
        "component_count": len(sizes),
        "largest_component_size": largest,
        "largest_component_fraction": largest / count,
        "singleton_count": singleton_count,
        "singleton_fraction": singleton_count / count,
        "component_size_p50": percentile_from_sorted(sizes, 50),
        "component_size_p90": percentile_from_sorted(sizes, 90),
        "component_size_p99": percentile_from_sorted(sizes, 99),
    }


def percentile_from_sorted(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return float(values[lower] * (1.0 - fraction) + values[upper] * fraction)


def radius_component_metrics(
    coords: list[list[int]],
    radius: float,
    min_cluster_size: int,
    source_count: int | None = None,
) -> dict[str, Any]:
    points = list({(int(z), int(y), int(x)) for z, y, x in coords})
    count = len(points)
    source_count = count if source_count is None else source_count
    if count == 0:
        return {
            "radius": radius,
            "sample_count": 0,
            "source_count": source_count,
            "component_count": 0,
            "largest_component_size": 0,
            "largest_component_fraction": 0.0,
            "large_component_count": 0,
            "large_component_fraction": 0.0,
            "min_cluster_size": min_cluster_size,
        }

    parent = list(range(count))
    component_size = [1] * count

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

    cell_size = max(1, int(math.ceil(radius)))
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, (z, y, x) in enumerate(points):
        cells[(z // cell_size, y // cell_size, x // cell_size)].append(index)

    radius2 = radius * radius
    for index, (z, y, x) in enumerate(points):
        cell = (z // cell_size, y // cell_size, x // cell_size)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for other in cells.get((cell[0] + dz, cell[1] + dy, cell[2] + dx), []):
                        if other <= index:
                            continue
                        oz, oy, ox = points[other]
                        distance2 = (z - oz) ** 2 + (y - oy) ** 2 + (x - ox) ** 2
                        if distance2 <= radius2:
                            union(index, other)

    sizes_by_root: dict[int, int] = {}
    for index in range(count):
        root = find(index)
        sizes_by_root[root] = sizes_by_root.get(root, 0) + 1
    sizes = sorted(sizes_by_root.values())
    largest = sizes[-1]
    large_sizes = [size for size in sizes if size >= min_cluster_size]
    return {
        "radius": radius,
        "sample_count": count,
        "source_count": source_count,
        "component_count": len(sizes),
        "largest_component_size": largest,
        "largest_component_fraction": largest / count,
        "large_component_count": len(large_sizes),
        "large_component_fraction": sum(large_sizes) / count,
        "min_cluster_size": min_cluster_size,
        "component_size_p50": percentile_from_sorted(sizes, 50),
        "component_size_p90": percentile_from_sorted(sizes, 90),
        "component_size_p99": percentile_from_sorted(sizes, 99),
    }


def hdbscan_component_metrics(
    coords: list[list[int]],
    epsilon: float,
    selected_thresholds: list[int],
    normals: list[list[float]] | None = None,
    patch_quality: bool = False,
    patch_sample_limit: int = 2048,
    local_continuity: bool = False,
    continuity_neighbors: int = 16,
    patch_graph: bool = False,
    patch_graph_cell_size: float = 8.0,
    patch_graph_min_cell_points: int = 4,
    patch_graph_neighbor_radius: int = 1,
    mesh_proxy: bool = False,
    mesh_prune: bool = False,
    mesh_prune_min_normal_agreement: float = 0.75,
    mesh_prune_max_offset_ratio: float = 0.35,
    unwrap_proxy: bool = False,
    unwrap_growth_proxy: bool = False,
    unwrap_distortion_growth_proxy: bool = False,
    unwrap_distortion_growth_candidate_limit: int = 0,
    unwrap_distortion_growth_bridge_only: bool = False,
    unwrap_distortion_growth_max_p90: float | None = None,
    local_atlas_growth_proxy: bool = False,
    local_atlas_growth_min_normal_agreement: float = 0.50,
    local_atlas_growth_max_offset_ratio: float = 0.20,
    local_atlas_growth_min_chart_normal_agreement: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    try:
        import hdbscan
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by CLI environment checks.
        raise SystemExit(
            "Missing dependency: hdbscan. Install experiment dependencies with "
            "`python -m pip install -r requirements-experiments.txt`."
        ) from exc

    points = np.array(coords, dtype=np.float32)
    normals_array = np.array(normals, dtype=np.float32) if normals is not None else None
    count = int(points.shape[0])
    if count == 0:
        selected = [
            {"threshold": threshold, "selected_count": 0, "selected_fraction": 0.0}
            for threshold in selected_thresholds
        ]
        if patch_quality:
            for item in selected:
                item["patch_quality"] = empty_patch_quality_summary()
        return {
            "epsilon": epsilon,
            "point_count": 0,
            "cluster_count": 0,
            "noise_fraction": 0.0,
            "largest_cluster_size": 0,
            "largest_cluster_fraction": 0.0,
            "selected_thresholds": selected,
        }

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=epsilon)
    labels = clusterer.fit_predict(points)
    labels_unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes_by_label = {int(label): int(size) for label, size in zip(labels_unique, counts) if label != -1}
    cluster_sizes = list(cluster_sizes_by_label.values())
    noise_count = int(counts[labels_unique == -1][0]) if np.any(labels_unique == -1) else 0
    largest = max(cluster_sizes) if cluster_sizes else 0
    selected = []
    for threshold in selected_thresholds:
        selected_count = sum(size for size in cluster_sizes if size >= threshold)
        item = {
            "threshold": threshold,
            "selected_count": selected_count,
            "selected_fraction": selected_count / count,
        }
        if patch_quality:
            item["patch_quality"] = selected_patch_quality_metrics(
                points,
                normals_array,
                labels,
                cluster_sizes_by_label,
                threshold=threshold,
                sample_limit=patch_sample_limit,
                local_continuity=local_continuity,
                continuity_neighbors=continuity_neighbors,
                patch_graph=patch_graph,
                patch_graph_cell_size=patch_graph_cell_size,
                patch_graph_min_cell_points=patch_graph_min_cell_points,
                patch_graph_neighbor_radius=patch_graph_neighbor_radius,
                mesh_proxy=mesh_proxy,
                mesh_prune=mesh_prune,
                mesh_prune_min_normal_agreement=mesh_prune_min_normal_agreement,
                mesh_prune_max_offset_ratio=mesh_prune_max_offset_ratio,
                unwrap_proxy=unwrap_proxy,
                unwrap_growth_proxy=unwrap_growth_proxy,
                unwrap_distortion_growth_proxy=unwrap_distortion_growth_proxy,
                unwrap_distortion_growth_candidate_limit=unwrap_distortion_growth_candidate_limit,
                unwrap_distortion_growth_bridge_only=unwrap_distortion_growth_bridge_only,
                unwrap_distortion_growth_max_p90=unwrap_distortion_growth_max_p90,
                local_atlas_growth_proxy=local_atlas_growth_proxy,
                local_atlas_growth_min_normal_agreement=local_atlas_growth_min_normal_agreement,
                local_atlas_growth_max_offset_ratio=local_atlas_growth_max_offset_ratio,
                local_atlas_growth_min_chart_normal_agreement=(
                    local_atlas_growth_min_chart_normal_agreement
                ),
                seed=seed,
            )
        selected.append(item)
    return {
        "epsilon": epsilon,
        "point_count": count,
        "cluster_count": len(cluster_sizes),
        "noise_fraction": noise_count / count,
        "largest_cluster_size": largest,
        "largest_cluster_fraction": largest / count,
        "selected_thresholds": selected,
    }


def empty_patch_quality_summary() -> dict[str, Any]:
    return {
        "selected_cluster_count": 0,
        "selected_count": 0,
        "sampled_point_count": 0,
        "weighted_plane_rms": None,
        "weighted_planarity_ratio": None,
        "weighted_normal_coherence": None,
        "weighted_bbox_diagonal": None,
        "weighted_local_plane_rms": None,
        "weighted_local_plane_rms_p90": None,
        "weighted_local_planarity_ratio": None,
        "weighted_local_normal_agreement": None,
        "weighted_nearest_neighbor_mean": None,
        "weighted_nearest_neighbor_p90": None,
        "weighted_patch_graph_node_count": None,
        "weighted_patch_graph_edge_count": None,
        "weighted_patch_graph_mean_degree": None,
        "weighted_patch_graph_largest_component_fraction": None,
        "weighted_patch_graph_isolated_node_fraction": None,
        "weighted_patch_graph_mean_edge_normal_agreement": None,
        "weighted_patch_graph_p10_edge_normal_agreement": None,
        "weighted_patch_graph_mean_edge_plane_offset": None,
        "weighted_patch_graph_p90_edge_plane_offset": None,
        "weighted_patch_graph_mean_edge_offset_ratio": None,
        "weighted_patch_graph_p90_edge_offset_ratio": None,
        "weighted_patch_graph_mean_cell_plane_rms": None,
        "weighted_patch_graph_mean_cell_planarity_ratio": None,
        "weighted_patch_mesh_node_count": None,
        "weighted_patch_mesh_edge_count": None,
        "weighted_patch_mesh_triangle_count": None,
        "weighted_patch_mesh_used_edge_fraction": None,
        "weighted_patch_mesh_nonmanifold_edge_fraction": None,
        "weighted_patch_mesh_largest_component_fraction": None,
        "weighted_patch_mesh_largest_component_triangle_fraction": None,
        "weighted_patch_mesh_mean_edge_length": None,
        "weighted_patch_mesh_p90_edge_length": None,
        "weighted_patch_mesh_mean_triangle_area": None,
        "weighted_patch_mesh_p90_triangle_area": None,
        "weighted_patch_mesh_mean_triangle_normal_agreement": None,
        "weighted_patch_mesh_p10_triangle_normal_agreement": None,
        "weighted_patch_mesh_degenerate_triangle_fraction": None,
        "weighted_patch_mesh_pruned_edge_count": None,
        "weighted_patch_mesh_pruned_edge_keep_fraction": None,
        "weighted_patch_mesh_pruned_triangle_count": None,
        "weighted_patch_mesh_pruned_used_edge_fraction": None,
        "weighted_patch_mesh_pruned_nonmanifold_edge_fraction": None,
        "weighted_patch_mesh_pruned_largest_component_fraction": None,
        "weighted_patch_mesh_pruned_largest_component_triangle_fraction": None,
        "weighted_patch_mesh_pruned_mean_edge_length": None,
        "weighted_patch_mesh_pruned_p90_edge_length": None,
        "weighted_patch_mesh_pruned_mean_triangle_area": None,
        "weighted_patch_mesh_pruned_p90_triangle_area": None,
        "weighted_patch_mesh_pruned_mean_triangle_normal_agreement": None,
        "weighted_patch_mesh_pruned_p10_triangle_normal_agreement": None,
        "weighted_patch_mesh_pruned_degenerate_triangle_fraction": None,
        "weighted_patch_unwrap_proxy_node_count": None,
        "weighted_patch_unwrap_proxy_edge_count": None,
        "weighted_patch_unwrap_proxy_triangle_count": None,
        "weighted_patch_unwrap_proxy_largest_component_node_fraction": None,
        "weighted_patch_unwrap_proxy_chart_planarity_ratio": None,
        "weighted_patch_unwrap_proxy_chart_normal_agreement": None,
        "weighted_patch_unwrap_proxy_mean_edge_distortion": None,
        "weighted_patch_unwrap_proxy_p90_edge_distortion": None,
        "weighted_patch_unwrap_proxy_mean_triangle_area_ratio": None,
        "weighted_patch_unwrap_proxy_p10_triangle_area_ratio": None,
        "weighted_patch_unwrap_proxy_degenerate_triangle_fraction": None,
        **{f"weighted_{key}": None for key in unwrap_growth_metric_names()},
        **{f"weighted_{key}": None for key in unwrap_distortion_growth_metric_names()},
        **{f"weighted_{key}": None for key in local_atlas_growth_metric_names()},
        "sample_limit": 0,
        "continuity_neighbor_count": 0,
        "patch_graph_cell_size": None,
        "patch_graph_min_points_per_cell": 0,
        "patch_graph_neighbor_radius": 0,
        "patch_mesh_proxy": False,
        "patch_mesh_prune": False,
        "patch_mesh_prune_min_normal_agreement": None,
        "patch_mesh_prune_max_offset_ratio": None,
        "patch_unwrap_proxy": False,
        "patch_unwrap_proxy_uses_pruned_edges": False,
        "patch_unwrap_growth_proxy": False,
        "patch_unwrap_distortion_growth_proxy": False,
        "patch_unwrap_distortion_growth_candidate_limit": 0,
        "patch_unwrap_distortion_growth_bridge_only": False,
        "patch_unwrap_distortion_growth_max_p90": None,
        "patch_local_atlas_growth_proxy": False,
        "patch_local_atlas_growth_min_normal_agreement": None,
        "patch_local_atlas_growth_max_offset_ratio": None,
        "patch_local_atlas_growth_min_chart_normal_agreement": None,
    }


def selected_patch_quality_metrics(
    points,
    normals,
    labels,
    cluster_sizes_by_label: dict[int, int],
    threshold: int,
    sample_limit: int,
    local_continuity: bool = False,
    continuity_neighbors: int = 16,
    patch_graph: bool = False,
    patch_graph_cell_size: float = 8.0,
    patch_graph_min_cell_points: int = 4,
    patch_graph_neighbor_radius: int = 1,
    mesh_proxy: bool = False,
    mesh_prune: bool = False,
    mesh_prune_min_normal_agreement: float = 0.75,
    mesh_prune_max_offset_ratio: float = 0.35,
    unwrap_proxy: bool = False,
    unwrap_growth_proxy: bool = False,
    unwrap_distortion_growth_proxy: bool = False,
    unwrap_distortion_growth_candidate_limit: int = 0,
    unwrap_distortion_growth_bridge_only: bool = False,
    unwrap_distortion_growth_max_p90: float | None = None,
    local_atlas_growth_proxy: bool = False,
    local_atlas_growth_min_normal_agreement: float = 0.50,
    local_atlas_growth_max_offset_ratio: float = 0.20,
    local_atlas_growth_min_chart_normal_agreement: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    import numpy as np

    selected_labels = [
        label for label, size in sorted(cluster_sizes_by_label.items()) if size >= threshold
    ]
    if not selected_labels:
        summary = empty_patch_quality_summary()
        summary["sample_limit"] = sample_limit
        summary["patch_graph_cell_size"] = patch_graph_cell_size if patch_graph else None
        summary["patch_graph_min_points_per_cell"] = patch_graph_min_cell_points if patch_graph else 0
        summary["patch_graph_neighbor_radius"] = patch_graph_neighbor_radius if patch_graph else 0
        summary["patch_mesh_proxy"] = bool(mesh_proxy)
        summary["patch_mesh_prune"] = bool(mesh_prune)
        summary["patch_mesh_prune_min_normal_agreement"] = (
            mesh_prune_min_normal_agreement if mesh_prune else None
        )
        summary["patch_mesh_prune_max_offset_ratio"] = mesh_prune_max_offset_ratio if mesh_prune else None
        summary["patch_unwrap_proxy"] = bool(unwrap_proxy)
        summary["patch_unwrap_proxy_uses_pruned_edges"] = bool(unwrap_proxy and mesh_prune)
        summary["patch_unwrap_growth_proxy"] = bool(unwrap_growth_proxy)
        summary["patch_unwrap_distortion_growth_proxy"] = bool(unwrap_distortion_growth_proxy)
        summary["patch_unwrap_distortion_growth_candidate_limit"] = (
            unwrap_distortion_growth_candidate_limit if unwrap_distortion_growth_proxy else 0
        )
        summary["patch_unwrap_distortion_growth_bridge_only"] = bool(
            unwrap_distortion_growth_proxy and unwrap_distortion_growth_bridge_only
        )
        summary["patch_unwrap_distortion_growth_max_p90"] = (
            unwrap_distortion_growth_max_p90 if unwrap_distortion_growth_proxy else None
        )
        summary["patch_local_atlas_growth_proxy"] = bool(local_atlas_growth_proxy)
        summary["patch_local_atlas_growth_min_normal_agreement"] = (
            local_atlas_growth_min_normal_agreement if local_atlas_growth_proxy else None
        )
        summary["patch_local_atlas_growth_max_offset_ratio"] = (
            local_atlas_growth_max_offset_ratio if local_atlas_growth_proxy else None
        )
        summary["patch_local_atlas_growth_min_chart_normal_agreement"] = (
            local_atlas_growth_min_chart_normal_agreement if local_atlas_growth_proxy else None
        )
        return summary

    cluster_metrics: list[dict[str, Any]] = []
    for label in selected_labels:
        indices = np.flatnonzero(labels == label)
        if indices.size == 0:
            continue
        sample_indices = indices
        if sample_limit > 0 and indices.size > sample_limit:
            rng = np.random.default_rng(seed + label * 1009 + threshold)
            sample_indices = rng.choice(indices, size=sample_limit, replace=False)
        cluster_points = points[sample_indices].astype(np.float64, copy=False)
        centered = cluster_points - cluster_points.mean(axis=0)
        covariance = centered.T @ centered / max(1, cluster_points.shape[0])
        eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
        total_variance = float(eigenvalues.sum())
        plane_rms = float(np.sqrt(eigenvalues[0]))
        planarity_ratio = float(eigenvalues[0] / total_variance) if total_variance > 0 else 0.0

        normal_coherence = None
        if normals is not None:
            cluster_normals = normals[sample_indices].astype(np.float64, copy=False)
            normal_lengths = np.linalg.norm(cluster_normals, axis=1)
            valid = normal_lengths > 1e-6
            if np.any(valid):
                unit_normals = cluster_normals[valid] / normal_lengths[valid, None]
                orientation_matrix = unit_normals.T @ unit_normals / unit_normals.shape[0]
                normal_coherence = float(np.linalg.eigvalsh(orientation_matrix)[-1])

        bbox_diagonal = float(np.linalg.norm(cluster_points.max(axis=0) - cluster_points.min(axis=0)))
        metrics = {
            "cluster_size": int(indices.size),
            "sample_count": int(sample_indices.size),
            "plane_rms": plane_rms,
            "planarity_ratio": planarity_ratio,
            "normal_coherence": normal_coherence,
            "bbox_diagonal": bbox_diagonal,
        }
        if local_continuity:
            metrics.update(
                local_continuity_metrics(
                    cluster_points,
                    cluster_normals if normals is not None else None,
                    neighbors=continuity_neighbors,
                )
            )
        if patch_graph:
            metrics.update(
                patch_graph_metrics(
                    cluster_points,
                    cluster_normals if normals is not None else None,
                    cell_size=patch_graph_cell_size,
                    min_points_per_cell=patch_graph_min_cell_points,
                    neighbor_radius=patch_graph_neighbor_radius,
                    mesh_proxy=mesh_proxy,
                    mesh_prune=mesh_prune,
                    mesh_prune_min_normal_agreement=mesh_prune_min_normal_agreement,
                    mesh_prune_max_offset_ratio=mesh_prune_max_offset_ratio,
                    unwrap_proxy=unwrap_proxy,
                    unwrap_growth_proxy=unwrap_growth_proxy,
                    unwrap_distortion_growth_proxy=unwrap_distortion_growth_proxy,
                    unwrap_distortion_growth_candidate_limit=unwrap_distortion_growth_candidate_limit,
                    unwrap_distortion_growth_bridge_only=unwrap_distortion_growth_bridge_only,
                    unwrap_distortion_growth_max_p90=unwrap_distortion_growth_max_p90,
                    local_atlas_growth_proxy=local_atlas_growth_proxy,
                    local_atlas_growth_min_normal_agreement=local_atlas_growth_min_normal_agreement,
                    local_atlas_growth_max_offset_ratio=local_atlas_growth_max_offset_ratio,
                    local_atlas_growth_min_chart_normal_agreement=(
                        local_atlas_growth_min_chart_normal_agreement
                    ),
                )
            )
        cluster_metrics.append(metrics)

    if not cluster_metrics:
        summary = empty_patch_quality_summary()
        summary["sample_limit"] = sample_limit
        summary["patch_graph_cell_size"] = patch_graph_cell_size if patch_graph else None
        summary["patch_graph_min_points_per_cell"] = patch_graph_min_cell_points if patch_graph else 0
        summary["patch_graph_neighbor_radius"] = patch_graph_neighbor_radius if patch_graph else 0
        summary["patch_mesh_proxy"] = bool(mesh_proxy)
        summary["patch_mesh_prune"] = bool(mesh_prune)
        summary["patch_mesh_prune_min_normal_agreement"] = (
            mesh_prune_min_normal_agreement if mesh_prune else None
        )
        summary["patch_mesh_prune_max_offset_ratio"] = mesh_prune_max_offset_ratio if mesh_prune else None
        summary["patch_unwrap_proxy"] = bool(unwrap_proxy)
        summary["patch_unwrap_proxy_uses_pruned_edges"] = bool(unwrap_proxy and mesh_prune)
        summary["patch_unwrap_growth_proxy"] = bool(unwrap_growth_proxy)
        summary["patch_unwrap_distortion_growth_proxy"] = bool(unwrap_distortion_growth_proxy)
        summary["patch_unwrap_distortion_growth_candidate_limit"] = (
            unwrap_distortion_growth_candidate_limit if unwrap_distortion_growth_proxy else 0
        )
        summary["patch_unwrap_distortion_growth_bridge_only"] = bool(
            unwrap_distortion_growth_proxy and unwrap_distortion_growth_bridge_only
        )
        summary["patch_unwrap_distortion_growth_max_p90"] = (
            unwrap_distortion_growth_max_p90 if unwrap_distortion_growth_proxy else None
        )
        summary["patch_local_atlas_growth_proxy"] = bool(local_atlas_growth_proxy)
        summary["patch_local_atlas_growth_min_normal_agreement"] = (
            local_atlas_growth_min_normal_agreement if local_atlas_growth_proxy else None
        )
        summary["patch_local_atlas_growth_max_offset_ratio"] = (
            local_atlas_growth_max_offset_ratio if local_atlas_growth_proxy else None
        )
        summary["patch_local_atlas_growth_min_chart_normal_agreement"] = (
            local_atlas_growth_min_chart_normal_agreement if local_atlas_growth_proxy else None
        )
        return summary

    def weighted_mean(key: str) -> float | None:
        value_weights = [
            (float(item[key]), float(item["cluster_size"]))
            for item in cluster_metrics
            if isinstance(item.get(key), (int, float))
        ]
        if not value_weights:
            return None
        weight_sum = sum(weight for _, weight in value_weights)
        return sum(value * weight for value, weight in value_weights) / weight_sum

    selected_count = sum(item["cluster_size"] for item in cluster_metrics)
    return {
        "selected_cluster_count": len(cluster_metrics),
        "selected_count": selected_count,
        "sampled_point_count": sum(item["sample_count"] for item in cluster_metrics),
        "weighted_plane_rms": weighted_mean("plane_rms"),
        "weighted_planarity_ratio": weighted_mean("planarity_ratio"),
        "weighted_normal_coherence": weighted_mean("normal_coherence"),
        "weighted_bbox_diagonal": weighted_mean("bbox_diagonal"),
        "weighted_local_plane_rms": weighted_mean("local_plane_rms_mean"),
        "weighted_local_plane_rms_p90": weighted_mean("local_plane_rms_p90"),
        "weighted_local_planarity_ratio": weighted_mean("local_planarity_ratio_mean"),
        "weighted_local_normal_agreement": weighted_mean("local_normal_agreement_mean"),
        "weighted_nearest_neighbor_mean": weighted_mean("nearest_neighbor_mean"),
        "weighted_nearest_neighbor_p90": weighted_mean("nearest_neighbor_p90"),
        "weighted_patch_graph_node_count": weighted_mean("patch_graph_node_count"),
        "weighted_patch_graph_edge_count": weighted_mean("patch_graph_edge_count"),
        "weighted_patch_graph_mean_degree": weighted_mean("patch_graph_mean_degree"),
        "weighted_patch_graph_largest_component_fraction": weighted_mean(
            "patch_graph_largest_component_fraction"
        ),
        "weighted_patch_graph_isolated_node_fraction": weighted_mean("patch_graph_isolated_node_fraction"),
        "weighted_patch_graph_mean_edge_normal_agreement": weighted_mean(
            "patch_graph_mean_edge_normal_agreement"
        ),
        "weighted_patch_graph_p10_edge_normal_agreement": weighted_mean(
            "patch_graph_p10_edge_normal_agreement"
        ),
        "weighted_patch_graph_mean_edge_plane_offset": weighted_mean("patch_graph_mean_edge_plane_offset"),
        "weighted_patch_graph_p90_edge_plane_offset": weighted_mean("patch_graph_p90_edge_plane_offset"),
        "weighted_patch_graph_mean_edge_offset_ratio": weighted_mean("patch_graph_mean_edge_offset_ratio"),
        "weighted_patch_graph_p90_edge_offset_ratio": weighted_mean("patch_graph_p90_edge_offset_ratio"),
        "weighted_patch_graph_mean_cell_plane_rms": weighted_mean("patch_graph_mean_cell_plane_rms"),
        "weighted_patch_graph_mean_cell_planarity_ratio": weighted_mean(
            "patch_graph_mean_cell_planarity_ratio"
        ),
        "weighted_patch_mesh_node_count": weighted_mean("patch_mesh_node_count"),
        "weighted_patch_mesh_edge_count": weighted_mean("patch_mesh_edge_count"),
        "weighted_patch_mesh_triangle_count": weighted_mean("patch_mesh_triangle_count"),
        "weighted_patch_mesh_used_edge_fraction": weighted_mean("patch_mesh_used_edge_fraction"),
        "weighted_patch_mesh_nonmanifold_edge_fraction": weighted_mean("patch_mesh_nonmanifold_edge_fraction"),
        "weighted_patch_mesh_largest_component_fraction": weighted_mean("patch_mesh_largest_component_fraction"),
        "weighted_patch_mesh_largest_component_triangle_fraction": weighted_mean(
            "patch_mesh_largest_component_triangle_fraction"
        ),
        "weighted_patch_mesh_mean_edge_length": weighted_mean("patch_mesh_mean_edge_length"),
        "weighted_patch_mesh_p90_edge_length": weighted_mean("patch_mesh_p90_edge_length"),
        "weighted_patch_mesh_mean_triangle_area": weighted_mean("patch_mesh_mean_triangle_area"),
        "weighted_patch_mesh_p90_triangle_area": weighted_mean("patch_mesh_p90_triangle_area"),
        "weighted_patch_mesh_mean_triangle_normal_agreement": weighted_mean(
            "patch_mesh_mean_triangle_normal_agreement"
        ),
        "weighted_patch_mesh_p10_triangle_normal_agreement": weighted_mean(
            "patch_mesh_p10_triangle_normal_agreement"
        ),
        "weighted_patch_mesh_degenerate_triangle_fraction": weighted_mean(
            "patch_mesh_degenerate_triangle_fraction"
        ),
        "weighted_patch_mesh_pruned_edge_count": weighted_mean("patch_mesh_pruned_edge_count"),
        "weighted_patch_mesh_pruned_edge_keep_fraction": weighted_mean("patch_mesh_pruned_edge_keep_fraction"),
        "weighted_patch_mesh_pruned_triangle_count": weighted_mean("patch_mesh_pruned_triangle_count"),
        "weighted_patch_mesh_pruned_used_edge_fraction": weighted_mean(
            "patch_mesh_pruned_used_edge_fraction"
        ),
        "weighted_patch_mesh_pruned_nonmanifold_edge_fraction": weighted_mean(
            "patch_mesh_pruned_nonmanifold_edge_fraction"
        ),
        "weighted_patch_mesh_pruned_largest_component_fraction": weighted_mean(
            "patch_mesh_pruned_largest_component_fraction"
        ),
        "weighted_patch_mesh_pruned_largest_component_triangle_fraction": weighted_mean(
            "patch_mesh_pruned_largest_component_triangle_fraction"
        ),
        "weighted_patch_mesh_pruned_mean_edge_length": weighted_mean("patch_mesh_pruned_mean_edge_length"),
        "weighted_patch_mesh_pruned_p90_edge_length": weighted_mean("patch_mesh_pruned_p90_edge_length"),
        "weighted_patch_mesh_pruned_mean_triangle_area": weighted_mean(
            "patch_mesh_pruned_mean_triangle_area"
        ),
        "weighted_patch_mesh_pruned_p90_triangle_area": weighted_mean("patch_mesh_pruned_p90_triangle_area"),
        "weighted_patch_mesh_pruned_mean_triangle_normal_agreement": weighted_mean(
            "patch_mesh_pruned_mean_triangle_normal_agreement"
        ),
        "weighted_patch_mesh_pruned_p10_triangle_normal_agreement": weighted_mean(
            "patch_mesh_pruned_p10_triangle_normal_agreement"
        ),
        "weighted_patch_mesh_pruned_degenerate_triangle_fraction": weighted_mean(
            "patch_mesh_pruned_degenerate_triangle_fraction"
        ),
        "weighted_patch_unwrap_proxy_node_count": weighted_mean("patch_unwrap_proxy_node_count"),
        "weighted_patch_unwrap_proxy_edge_count": weighted_mean("patch_unwrap_proxy_edge_count"),
        "weighted_patch_unwrap_proxy_triangle_count": weighted_mean("patch_unwrap_proxy_triangle_count"),
        "weighted_patch_unwrap_proxy_largest_component_node_fraction": weighted_mean(
            "patch_unwrap_proxy_largest_component_node_fraction"
        ),
        "weighted_patch_unwrap_proxy_chart_planarity_ratio": weighted_mean(
            "patch_unwrap_proxy_chart_planarity_ratio"
        ),
        "weighted_patch_unwrap_proxy_chart_normal_agreement": weighted_mean(
            "patch_unwrap_proxy_chart_normal_agreement"
        ),
        "weighted_patch_unwrap_proxy_mean_edge_distortion": weighted_mean(
            "patch_unwrap_proxy_mean_edge_distortion"
        ),
        "weighted_patch_unwrap_proxy_p90_edge_distortion": weighted_mean(
            "patch_unwrap_proxy_p90_edge_distortion"
        ),
        "weighted_patch_unwrap_proxy_mean_triangle_area_ratio": weighted_mean(
            "patch_unwrap_proxy_mean_triangle_area_ratio"
        ),
        "weighted_patch_unwrap_proxy_p10_triangle_area_ratio": weighted_mean(
            "patch_unwrap_proxy_p10_triangle_area_ratio"
        ),
        "weighted_patch_unwrap_proxy_degenerate_triangle_fraction": weighted_mean(
            "patch_unwrap_proxy_degenerate_triangle_fraction"
        ),
        **{f"weighted_{key}": weighted_mean(key) for key in unwrap_growth_metric_names()},
        **{f"weighted_{key}": weighted_mean(key) for key in unwrap_distortion_growth_metric_names()},
        **{f"weighted_{key}": weighted_mean(key) for key in local_atlas_growth_metric_names()},
        "sample_limit": sample_limit,
        "continuity_neighbor_count": continuity_neighbors if local_continuity else 0,
        "patch_graph_cell_size": patch_graph_cell_size if patch_graph else None,
        "patch_graph_min_points_per_cell": patch_graph_min_cell_points if patch_graph else 0,
        "patch_graph_neighbor_radius": patch_graph_neighbor_radius if patch_graph else 0,
        "patch_mesh_proxy": bool(mesh_proxy),
        "patch_mesh_prune": bool(mesh_prune),
        "patch_mesh_prune_min_normal_agreement": mesh_prune_min_normal_agreement if mesh_prune else None,
        "patch_mesh_prune_max_offset_ratio": mesh_prune_max_offset_ratio if mesh_prune else None,
        "patch_unwrap_proxy": bool(unwrap_proxy),
        "patch_unwrap_proxy_uses_pruned_edges": bool(unwrap_proxy and mesh_prune),
        "patch_unwrap_growth_proxy": bool(unwrap_growth_proxy),
        "patch_unwrap_distortion_growth_proxy": bool(unwrap_distortion_growth_proxy),
        "patch_unwrap_distortion_growth_candidate_limit": (
            unwrap_distortion_growth_candidate_limit if unwrap_distortion_growth_proxy else 0
        ),
        "patch_unwrap_distortion_growth_bridge_only": bool(
            unwrap_distortion_growth_proxy and unwrap_distortion_growth_bridge_only
        ),
        "patch_unwrap_distortion_growth_max_p90": (
            unwrap_distortion_growth_max_p90 if unwrap_distortion_growth_proxy else None
        ),
        "patch_local_atlas_growth_proxy": bool(local_atlas_growth_proxy),
        "patch_local_atlas_growth_min_normal_agreement": (
            local_atlas_growth_min_normal_agreement if local_atlas_growth_proxy else None
        ),
        "patch_local_atlas_growth_max_offset_ratio": (
            local_atlas_growth_max_offset_ratio if local_atlas_growth_proxy else None
        ),
        "patch_local_atlas_growth_min_chart_normal_agreement": (
            local_atlas_growth_min_chart_normal_agreement if local_atlas_growth_proxy else None
        ),
    }


def local_continuity_metrics(points, normals, neighbors: int) -> dict[str, Any]:
    import numpy as np

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - exercised by CLI environment checks.
        raise SystemExit(
            "Missing dependency: scipy. Install experiment dependencies with "
            "`python -m pip install -r requirements-experiments.txt`."
        ) from exc

    count = int(points.shape[0])
    if count < 3:
        return {
            "local_plane_rms_mean": None,
            "local_plane_rms_p90": None,
            "local_planarity_ratio_mean": None,
            "local_normal_agreement_mean": None,
            "nearest_neighbor_mean": None,
            "nearest_neighbor_p90": None,
        }

    query_count = min(max(3, neighbors + 1), count)
    distances, indices = cKDTree(points).query(points, k=query_count)
    if query_count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    neighbor_distances = distances[:, 1:] if query_count > 1 else distances
    nearest = neighbor_distances[:, 0] if neighbor_distances.shape[1] else np.zeros(count)

    plane_rms_values: list[float] = []
    planarity_values: list[float] = []
    for row in indices:
        local_points = points[row]
        centered = local_points - local_points.mean(axis=0)
        covariance = centered.T @ centered / max(1, local_points.shape[0])
        eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
        total_variance = float(eigenvalues.sum())
        plane_rms_values.append(float(np.sqrt(eigenvalues[0])))
        planarity_values.append(float(eigenvalues[0] / total_variance) if total_variance > 0 else 0.0)

    normal_agreement = None
    if normals is not None:
        normal_lengths = np.linalg.norm(normals, axis=1)
        valid = normal_lengths > 1e-6
        if np.any(valid):
            unit_normals = np.zeros_like(normals, dtype=np.float64)
            unit_normals[valid] = normals[valid] / normal_lengths[valid, None]
            agreement_values: list[float] = []
            for center_index, neighbor_indices in enumerate(indices[:, 1:]):
                if not valid[center_index]:
                    continue
                valid_neighbors = [index for index in neighbor_indices if valid[index]]
                if not valid_neighbors:
                    continue
                dots = np.abs(unit_normals[valid_neighbors] @ unit_normals[center_index])
                agreement_values.append(float(np.mean(dots)))
            if agreement_values:
                normal_agreement = float(np.mean(agreement_values))

    plane_rms_array = np.array(plane_rms_values, dtype=np.float64)
    planarity_array = np.array(planarity_values, dtype=np.float64)
    return {
        "local_plane_rms_mean": float(np.mean(plane_rms_array)),
        "local_plane_rms_p90": float(np.percentile(plane_rms_array, 90)),
        "local_planarity_ratio_mean": float(np.mean(planarity_array)),
        "local_normal_agreement_mean": normal_agreement,
        "nearest_neighbor_mean": float(np.mean(nearest)),
        "nearest_neighbor_p90": float(np.percentile(nearest, 90)),
    }


def patch_graph_metrics(
    points,
    normals,
    cell_size: float,
    min_points_per_cell: int,
    neighbor_radius: int = 1,
    mesh_proxy: bool = False,
    mesh_prune: bool = False,
    mesh_prune_min_normal_agreement: float = 0.75,
    mesh_prune_max_offset_ratio: float = 0.35,
    unwrap_proxy: bool = False,
    unwrap_growth_proxy: bool = False,
    unwrap_distortion_growth_proxy: bool = False,
    unwrap_distortion_growth_candidate_limit: int = 0,
    unwrap_distortion_growth_bridge_only: bool = False,
    unwrap_distortion_growth_max_p90: float | None = None,
    local_atlas_growth_proxy: bool = False,
    local_atlas_growth_min_normal_agreement: float = 0.50,
    local_atlas_growth_max_offset_ratio: float = 0.20,
    local_atlas_growth_min_chart_normal_agreement: float = 0.0,
) -> dict[str, Any]:
    import numpy as np

    if cell_size <= 0:
        raise ValueError("patch graph cell_size must be positive")
    if min_points_per_cell < 1:
        raise ValueError("patch graph min_points_per_cell must be positive")
    if neighbor_radius < 1:
        raise ValueError("patch graph neighbor_radius must be positive")

    base = {
        "patch_graph_cell_size": float(cell_size),
        "patch_graph_min_points_per_cell": int(min_points_per_cell),
        "patch_graph_neighbor_radius": int(neighbor_radius),
        "patch_graph_node_count": 0,
        "patch_graph_edge_count": 0,
        "patch_graph_mean_degree": 0.0,
        "patch_graph_largest_component_fraction": 0.0,
        "patch_graph_isolated_node_fraction": 0.0,
        "patch_graph_mean_edge_normal_agreement": None,
        "patch_graph_p10_edge_normal_agreement": None,
        "patch_graph_mean_edge_plane_offset": None,
        "patch_graph_p90_edge_plane_offset": None,
        "patch_graph_mean_edge_offset_ratio": None,
        "patch_graph_p90_edge_offset_ratio": None,
        "patch_graph_mean_cell_plane_rms": None,
        "patch_graph_mean_cell_planarity_ratio": None,
        "patch_mesh_proxy": bool(mesh_proxy),
        "patch_mesh_node_count": 0 if mesh_proxy else None,
        "patch_mesh_edge_count": 0 if mesh_proxy else None,
        "patch_mesh_triangle_count": 0 if mesh_proxy else None,
        "patch_mesh_used_edge_fraction": None,
        "patch_mesh_nonmanifold_edge_fraction": None,
        "patch_mesh_largest_component_fraction": 0.0 if mesh_proxy else None,
        "patch_mesh_largest_component_triangle_fraction": None,
        "patch_mesh_mean_edge_length": None,
        "patch_mesh_p90_edge_length": None,
        "patch_mesh_mean_triangle_area": None,
        "patch_mesh_p90_triangle_area": None,
        "patch_mesh_mean_triangle_normal_agreement": None,
        "patch_mesh_p10_triangle_normal_agreement": None,
        "patch_mesh_degenerate_triangle_fraction": None,
        "patch_mesh_prune": bool(mesh_prune),
        "patch_mesh_prune_min_normal_agreement": mesh_prune_min_normal_agreement if mesh_prune else None,
        "patch_mesh_prune_max_offset_ratio": mesh_prune_max_offset_ratio if mesh_prune else None,
        "patch_mesh_pruned_edge_count": 0 if mesh_prune else None,
        "patch_mesh_pruned_edge_keep_fraction": None,
        "patch_mesh_pruned_triangle_count": 0 if mesh_prune else None,
        "patch_mesh_pruned_used_edge_fraction": None,
        "patch_mesh_pruned_nonmanifold_edge_fraction": None,
        "patch_mesh_pruned_largest_component_fraction": 0.0 if mesh_prune else None,
        "patch_mesh_pruned_largest_component_triangle_fraction": None,
        "patch_mesh_pruned_mean_edge_length": None,
        "patch_mesh_pruned_p90_edge_length": None,
        "patch_mesh_pruned_mean_triangle_area": None,
        "patch_mesh_pruned_p90_triangle_area": None,
        "patch_mesh_pruned_mean_triangle_normal_agreement": None,
        "patch_mesh_pruned_p10_triangle_normal_agreement": None,
        "patch_mesh_pruned_degenerate_triangle_fraction": None,
        "patch_unwrap_proxy": bool(unwrap_proxy),
        "patch_unwrap_proxy_uses_pruned_edges": bool(unwrap_proxy and mesh_prune),
        "patch_unwrap_proxy_node_count": 0 if unwrap_proxy else None,
        "patch_unwrap_proxy_edge_count": 0 if unwrap_proxy else None,
        "patch_unwrap_proxy_triangle_count": 0 if unwrap_proxy else None,
        "patch_unwrap_proxy_largest_component_node_fraction": 0.0 if unwrap_proxy else None,
        "patch_unwrap_proxy_chart_planarity_ratio": None,
        "patch_unwrap_proxy_chart_normal_agreement": None,
        "patch_unwrap_proxy_mean_edge_distortion": None,
        "patch_unwrap_proxy_p90_edge_distortion": None,
        "patch_unwrap_proxy_mean_triangle_area_ratio": None,
        "patch_unwrap_proxy_p10_triangle_area_ratio": None,
        "patch_unwrap_proxy_degenerate_triangle_fraction": None,
        "patch_unwrap_growth_proxy": bool(unwrap_growth_proxy),
        **{key: None for key in unwrap_growth_metric_names()},
        "patch_unwrap_distortion_growth_proxy": bool(unwrap_distortion_growth_proxy),
        "patch_unwrap_distortion_growth_candidate_limit": (
            int(unwrap_distortion_growth_candidate_limit) if unwrap_distortion_growth_proxy else 0
        ),
        "patch_unwrap_distortion_growth_bridge_only": bool(
            unwrap_distortion_growth_proxy and unwrap_distortion_growth_bridge_only
        ),
        "patch_unwrap_distortion_growth_max_p90": (
            unwrap_distortion_growth_max_p90 if unwrap_distortion_growth_proxy else None
        ),
        **{key: None for key in unwrap_distortion_growth_metric_names()},
        "patch_local_atlas_growth_proxy": bool(local_atlas_growth_proxy),
        "patch_local_atlas_growth_min_normal_agreement": (
            local_atlas_growth_min_normal_agreement if local_atlas_growth_proxy else None
        ),
        "patch_local_atlas_growth_max_offset_ratio": (
            local_atlas_growth_max_offset_ratio if local_atlas_growth_proxy else None
        ),
        "patch_local_atlas_growth_min_chart_normal_agreement": (
            local_atlas_growth_min_chart_normal_agreement if local_atlas_growth_proxy else None
        ),
        **{key: None for key in local_atlas_growth_metric_names()},
    }
    if int(points.shape[0]) == 0:
        return base

    cell_indices = np.floor(points / cell_size).astype(np.int64)
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, cell in enumerate(cell_indices):
        cells[(int(cell[0]), int(cell[1]), int(cell[2]))].append(index)

    kept_cells = sorted(
        (cell, indices) for cell, indices in cells.items() if len(indices) >= min_points_per_cell
    )
    if not kept_cells:
        return base

    nodes: list[dict[str, Any]] = []
    cell_to_node: dict[tuple[int, int, int], int] = {}
    for node_index, (cell, indices) in enumerate(kept_cells):
        cell_points = points[indices].astype(np.float64, copy=False)
        centered = cell_points - cell_points.mean(axis=0)
        covariance = centered.T @ centered / max(1, cell_points.shape[0])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        total_variance = float(eigenvalues.sum())
        plane_normal = eigenvectors[:, 0]
        if normals is not None:
            cell_normals = normals[indices].astype(np.float64, copy=False)
            lengths = np.linalg.norm(cell_normals, axis=1)
            valid = lengths > 1e-6
            if np.any(valid):
                mean_normal = np.mean(cell_normals[valid] / lengths[valid, None], axis=0)
                if np.dot(plane_normal, mean_normal) < 0:
                    plane_normal = -plane_normal
        nodes.append(
            {
                "cell": cell,
                "centroid": cell_points.mean(axis=0),
                "normal": plane_normal,
                "plane_rms": float(np.sqrt(eigenvalues[0])),
                "planarity_ratio": float(eigenvalues[0] / total_variance) if total_variance > 0 else 0.0,
            }
        )
        cell_to_node[cell] = node_index

    node_count = len(nodes)
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

    normal_agreements: list[float] = []
    plane_offsets: list[float] = []
    offset_ratios: list[float] = []
    edges: list[tuple[int, int]] = []
    edge_quality: list[dict[str, float | tuple[int, int]]] = []
    degree = [0] * node_count
    neighbor_offsets = [
        (dz, dy, dx)
        for dz in range(-neighbor_radius, neighbor_radius + 1)
        for dy in range(-neighbor_radius, neighbor_radius + 1)
        for dx in range(-neighbor_radius, neighbor_radius + 1)
        if (dz, dy, dx) != (0, 0, 0)
    ]
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
            normal_agreements.append(normal_agreement)
            plane_offsets.append(plane_offset)
            offset_ratios.append(plane_offset / gap)
            edges.append((node_index, other_index))
            edge_quality.append(
                {
                    "edge": (node_index, other_index),
                    "normal_agreement": normal_agreement,
                    "offset_ratio": plane_offset / gap,
                }
            )
            degree[node_index] += 1
            degree[other_index] += 1
            union(node_index, other_index)

    sizes_by_root: dict[int, int] = {}
    for index in range(node_count):
        root = find(index)
        sizes_by_root[root] = sizes_by_root.get(root, 0) + 1

    def mean_or_none(values: list[float]) -> float | None:
        return float(np.mean(np.array(values, dtype=np.float64))) if values else None

    def percentile_or_none(values: list[float], percentile: float) -> float | None:
        return float(np.percentile(np.array(values, dtype=np.float64), percentile)) if values else None

    cell_plane_rms = [float(node["plane_rms"]) for node in nodes]
    cell_planarity = [float(node["planarity_ratio"]) for node in nodes]
    edge_count = len(normal_agreements)
    mesh_metrics = mesh_proxy_metrics(nodes, edges) if mesh_proxy else {}
    unwrap_edges = edges
    if mesh_prune:
        pruned_edges = [
            quality["edge"]
            for quality in edge_quality
            if quality["normal_agreement"] >= mesh_prune_min_normal_agreement
            and quality["offset_ratio"] <= mesh_prune_max_offset_ratio
        ]
        mesh_metrics.update(mesh_proxy_metrics(nodes, pruned_edges, prefix="patch_mesh_pruned"))
        mesh_metrics["patch_mesh_pruned_edge_keep_fraction"] = (
            len(pruned_edges) / edge_count if edge_count else None
        )
        unwrap_edges = pruned_edges
    if unwrap_proxy:
        mesh_metrics.update(unwrap_proxy_metrics(nodes, unwrap_edges))
    if unwrap_growth_proxy:
        mesh_metrics.update(unwrap_growth_metrics(nodes, edges, unwrap_edges, edge_quality))
    if unwrap_distortion_growth_proxy:
        mesh_metrics.update(
            unwrap_distortion_growth_metrics(
                nodes,
                edges,
                unwrap_edges,
                edge_quality,
                candidate_limit=unwrap_distortion_growth_candidate_limit,
                bridge_only=unwrap_distortion_growth_bridge_only,
                max_p90_distortion=unwrap_distortion_growth_max_p90,
            )
        )
    if local_atlas_growth_proxy:
        mesh_metrics.update(
            local_atlas_growth_metrics(
                nodes,
                edges,
                unwrap_edges,
                edge_quality,
                min_normal_agreement=local_atlas_growth_min_normal_agreement,
                max_offset_ratio=local_atlas_growth_max_offset_ratio,
                min_chart_normal_agreement=local_atlas_growth_min_chart_normal_agreement,
            )
        )
    return {
        **base,
        "patch_graph_node_count": node_count,
        "patch_graph_edge_count": edge_count,
        "patch_graph_mean_degree": (2.0 * edge_count / node_count) if node_count else 0.0,
        "patch_graph_largest_component_fraction": max(sizes_by_root.values()) / node_count,
        "patch_graph_isolated_node_fraction": sum(1 for value in degree if value == 0) / node_count,
        "patch_graph_mean_edge_normal_agreement": mean_or_none(normal_agreements),
        "patch_graph_p10_edge_normal_agreement": percentile_or_none(normal_agreements, 10),
        "patch_graph_mean_edge_plane_offset": mean_or_none(plane_offsets),
        "patch_graph_p90_edge_plane_offset": percentile_or_none(plane_offsets, 90),
        "patch_graph_mean_edge_offset_ratio": mean_or_none(offset_ratios),
        "patch_graph_p90_edge_offset_ratio": percentile_or_none(offset_ratios, 90),
        "patch_graph_mean_cell_plane_rms": mean_or_none(cell_plane_rms),
        "patch_graph_mean_cell_planarity_ratio": mean_or_none(cell_planarity),
        **mesh_metrics,
    }


def mesh_proxy_metrics(
    nodes: list[dict[str, Any]],
    edges: list[tuple[int, int]],
    prefix: str = "patch_mesh",
) -> dict[str, Any]:
    import numpy as np

    node_count = len(nodes)
    edge_set = {tuple(sorted(edge)) for edge in edges}
    adjacency: list[set[int]] = [set() for _ in range(node_count)]
    edge_lengths: list[float] = []
    for left, right in edge_set:
        adjacency[left].add(right)
        adjacency[right].add(left)
        edge_lengths.append(float(np.linalg.norm(nodes[right]["centroid"] - nodes[left]["centroid"])))

    triangles: list[tuple[int, int, int]] = []
    for left in range(node_count):
        larger_neighbors = sorted(index for index in adjacency[left] if index > left)
        for pos, middle in enumerate(larger_neighbors):
            for right in larger_neighbors[pos + 1 :]:
                if right in adjacency[middle]:
                    triangles.append((left, middle, right))

    triangle_areas: list[float] = []
    normal_agreements: list[float] = []
    degenerate_count = 0
    edge_triangle_counts: dict[tuple[int, int], int] = {edge: 0 for edge in edge_set}
    component_triangles: dict[int, int] = {}

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

    for left, right in edge_set:
        union(left, right)

    for triangle in triangles:
        left, middle, right = triangle
        a = nodes[left]["centroid"]
        b = nodes[middle]["centroid"]
        c = nodes[right]["centroid"]
        cross = np.cross(b - a, c - a)
        area = float(0.5 * np.linalg.norm(cross))
        triangle_areas.append(area)
        if area <= 1e-9:
            degenerate_count += 1
        else:
            triangle_normal = cross / np.linalg.norm(cross)
            agreements = [
                abs(float(np.dot(triangle_normal, nodes[index]["normal"])))
                for index in triangle
            ]
            normal_agreements.append(float(np.mean(agreements)))
        for edge in ((left, middle), (left, right), (middle, right)):
            edge_triangle_counts[tuple(sorted(edge))] += 1
        root = find(left)
        component_triangles[root] = component_triangles.get(root, 0) + 1

    component_counts: dict[int, int] = {}
    for index in range(node_count):
        root = find(index)
        component_counts[root] = component_counts.get(root, 0) + 1

    def mean_or_none(values: list[float]) -> float | None:
        return float(np.mean(np.array(values, dtype=np.float64))) if values else None

    def percentile_or_none(values: list[float], percentile: float) -> float | None:
        return float(np.percentile(np.array(values, dtype=np.float64), percentile)) if values else None

    triangle_count = len(triangles)
    used_edge_count = sum(1 for count in edge_triangle_counts.values() if count > 0)
    nonmanifold_edge_count = sum(1 for count in edge_triangle_counts.values() if count > 2)
    edge_count = len(edge_set)
    largest_component = max(component_counts.values()) if component_counts else 0
    largest_component_triangles = max(component_triangles.values()) if component_triangles else 0
    return {
        f"{prefix}_node_count": node_count,
        f"{prefix}_edge_count": edge_count,
        f"{prefix}_triangle_count": triangle_count,
        f"{prefix}_used_edge_fraction": used_edge_count / edge_count if edge_count else None,
        f"{prefix}_nonmanifold_edge_fraction": nonmanifold_edge_count / edge_count if edge_count else None,
        f"{prefix}_largest_component_fraction": largest_component / node_count if node_count else 0.0,
        f"{prefix}_largest_component_triangle_fraction": (
            largest_component_triangles / triangle_count if triangle_count else None
        ),
        f"{prefix}_mean_edge_length": mean_or_none(edge_lengths),
        f"{prefix}_p90_edge_length": percentile_or_none(edge_lengths, 90),
        f"{prefix}_mean_triangle_area": mean_or_none(triangle_areas),
        f"{prefix}_p90_triangle_area": percentile_or_none(triangle_areas, 90),
        f"{prefix}_mean_triangle_normal_agreement": mean_or_none(normal_agreements),
        f"{prefix}_p10_triangle_normal_agreement": percentile_or_none(normal_agreements, 10),
        f"{prefix}_degenerate_triangle_fraction": degenerate_count / triangle_count if triangle_count else None,
    }


def unwrap_proxy_metrics(
    nodes: list[dict[str, Any]],
    edges: list[tuple[int, int]],
) -> dict[str, Any]:
    """Project the largest patch-graph component to 2D and summarize local distortion."""

    import numpy as np

    prefix = "patch_unwrap_proxy"
    base = {
        f"{prefix}_node_count": 0,
        f"{prefix}_edge_count": 0,
        f"{prefix}_triangle_count": 0,
        f"{prefix}_largest_component_node_fraction": 0.0,
        f"{prefix}_chart_planarity_ratio": None,
        f"{prefix}_chart_normal_agreement": None,
        f"{prefix}_mean_edge_distortion": None,
        f"{prefix}_p90_edge_distortion": None,
        f"{prefix}_mean_triangle_area_ratio": None,
        f"{prefix}_p10_triangle_area_ratio": None,
        f"{prefix}_degenerate_triangle_fraction": None,
    }
    node_count = len(nodes)
    edge_set = {tuple(sorted(edge)) for edge in edges}
    if node_count < 3 or not edge_set:
        return base

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
        edge for edge in sorted(edge_set) if edge[0] in largest_node_set and edge[1] in largest_node_set
    ]
    if len(largest_nodes) < 3 or not component_edges:
        return {
            **base,
            f"{prefix}_node_count": len(largest_nodes),
            f"{prefix}_edge_count": len(component_edges),
            f"{prefix}_largest_component_node_fraction": len(largest_nodes) / node_count,
        }

    centroids = np.array([nodes[index]["centroid"] for index in largest_nodes], dtype=np.float64)
    centered = centroids - centroids.mean(axis=0)
    covariance = centered.T @ centered / max(1, centered.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total_variance = float(eigenvalues.sum())
    chart_normal = eigenvectors[:, 0]
    chart_axes = eigenvectors[:, -2:]
    chart_coords = centered @ chart_axes
    local_index = {node_index: index for index, node_index in enumerate(largest_nodes)}

    chart_normal_agreement = None
    node_normals = np.array([nodes[index]["normal"] for index in largest_nodes], dtype=np.float64)
    normal_lengths = np.linalg.norm(node_normals, axis=1)
    valid_normals = normal_lengths > 1e-6
    if np.any(valid_normals):
        unit_normals = node_normals[valid_normals] / normal_lengths[valid_normals, None]
        chart_normal_agreement = float(np.mean(np.abs(unit_normals @ chart_normal)))

    edge_distortions: list[float] = []
    for left, right in component_edges:
        left_local = local_index[left]
        right_local = local_index[right]
        length_3d = float(np.linalg.norm(centroids[right_local] - centroids[left_local]))
        if length_3d <= 1e-9:
            continue
        length_2d = float(np.linalg.norm(chart_coords[right_local] - chart_coords[left_local]))
        edge_distortions.append(abs(length_2d / length_3d - 1.0))

    component_edge_set = {tuple(sorted(edge)) for edge in component_edges}
    component_adjacency: dict[int, set[int]] = {index: set() for index in largest_nodes}
    for left, right in component_edge_set:
        component_adjacency[left].add(right)
        component_adjacency[right].add(left)

    triangles: list[tuple[int, int, int]] = []
    for left in largest_nodes:
        larger_neighbors = sorted(index for index in component_adjacency[left] if index > left)
        for pos, middle in enumerate(larger_neighbors):
            for right in larger_neighbors[pos + 1 :]:
                if right in component_adjacency[middle]:
                    triangles.append((left, middle, right))

    area_ratios: list[float] = []
    degenerate_count = 0
    for left, middle, right in triangles:
        a3 = centroids[local_index[left]]
        b3 = centroids[local_index[middle]]
        c3 = centroids[local_index[right]]
        area_3d = float(0.5 * np.linalg.norm(np.cross(b3 - a3, c3 - a3)))
        a2 = chart_coords[local_index[left]]
        b2 = chart_coords[local_index[middle]]
        c2 = chart_coords[local_index[right]]
        area_2d = float(0.5 * abs(np.linalg.det(np.stack([b2 - a2, c2 - a2]))))
        if area_3d <= 1e-9:
            degenerate_count += 1
        else:
            area_ratios.append(area_2d / area_3d)

    def mean_or_none(values: list[float]) -> float | None:
        return float(np.mean(np.array(values, dtype=np.float64))) if values else None

    def percentile_or_none(values: list[float], percentile: float) -> float | None:
        return float(np.percentile(np.array(values, dtype=np.float64), percentile)) if values else None

    return {
        f"{prefix}_node_count": len(largest_nodes),
        f"{prefix}_edge_count": len(component_edges),
        f"{prefix}_triangle_count": len(triangles),
        f"{prefix}_largest_component_node_fraction": len(largest_nodes) / node_count,
        f"{prefix}_chart_planarity_ratio": eigenvalues[0] / total_variance if total_variance > 0 else 0.0,
        f"{prefix}_chart_normal_agreement": chart_normal_agreement,
        f"{prefix}_mean_edge_distortion": mean_or_none(edge_distortions),
        f"{prefix}_p90_edge_distortion": percentile_or_none(edge_distortions, 90),
        f"{prefix}_mean_triangle_area_ratio": mean_or_none(area_ratios),
        f"{prefix}_p10_triangle_area_ratio": percentile_or_none(area_ratios, 10),
        f"{prefix}_degenerate_triangle_fraction": (
            degenerate_count / len(triangles) if triangles else None
        ),
    }


def unwrap_growth_metrics(
    nodes: list[dict[str, Any]],
    all_edges: list[tuple[int, int]],
    base_edges: list[tuple[int, int]],
    edge_quality: list[dict[str, float | tuple[int, int]]],
) -> dict[str, Any]:
    """Add rejected patch-graph edges until fixed coverage targets are reached."""

    import numpy as np

    node_count = len(nodes)
    result = {key: None for key in unwrap_growth_metric_names()}
    if node_count < 3:
        return result

    all_edge_set = {tuple(sorted(edge)) for edge in all_edges}
    grown_edges = {tuple(sorted(edge)) for edge in base_edges}
    quality_by_edge = {
        tuple(sorted(quality["edge"])): quality
        for quality in edge_quality
        if isinstance(quality.get("edge"), tuple)
    }
    candidates = [
        quality
        for edge, quality in quality_by_edge.items()
        if edge in all_edge_set and edge not in grown_edges
    ]
    candidates.sort(
        key=lambda item: (
            -float(item["normal_agreement"]),
            float(item["offset_ratio"]),
            tuple(item["edge"]),
        )
    )

    added_qualities: list[dict[str, float | tuple[int, int]]] = []
    candidate_index = 0

    def largest_fraction(edges: set[tuple[int, int]]) -> float:
        if not edges:
            return 1.0 / node_count
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
        sizes: dict[int, int] = {}
        for index in range(node_count):
            root = find(index)
            sizes[root] = sizes.get(root, 0) + 1
        return max(sizes.values()) / node_count if sizes else 0.0

    def mean_or_none(values: list[float]) -> float | None:
        return float(np.mean(np.array(values, dtype=np.float64))) if values else None

    def percentile_or_none(values: list[float], percentile: float) -> float | None:
        return float(np.percentile(np.array(values, dtype=np.float64), percentile)) if values else None

    for target in UNWRAP_GROWTH_TARGETS:
        while largest_fraction(grown_edges) < target and candidate_index < len(candidates):
            quality = candidates[candidate_index]
            grown_edges.add(tuple(sorted(quality["edge"])))
            added_qualities.append(quality)
            candidate_index += 1

        chart = unwrap_proxy_metrics(nodes, sorted(grown_edges))
        suffix = unwrap_growth_suffix(target)
        prefix = f"patch_unwrap_growth_{suffix}"
        normal_values = [float(item["normal_agreement"]) for item in added_qualities]
        offset_values = [float(item["offset_ratio"]) for item in added_qualities]
        result.update(
            {
                f"{prefix}_reached": chart["patch_unwrap_proxy_largest_component_node_fraction"] >= target,
                f"{prefix}_added_edge_count": len(added_qualities),
                f"{prefix}_node_count": chart["patch_unwrap_proxy_node_count"],
                f"{prefix}_edge_count": chart["patch_unwrap_proxy_edge_count"],
                f"{prefix}_triangle_count": chart["patch_unwrap_proxy_triangle_count"],
                f"{prefix}_largest_component_node_fraction": chart[
                    "patch_unwrap_proxy_largest_component_node_fraction"
                ],
                f"{prefix}_mean_edge_distortion": chart["patch_unwrap_proxy_mean_edge_distortion"],
                f"{prefix}_p90_edge_distortion": chart["patch_unwrap_proxy_p90_edge_distortion"],
                f"{prefix}_mean_triangle_area_ratio": chart["patch_unwrap_proxy_mean_triangle_area_ratio"],
                f"{prefix}_p10_triangle_area_ratio": chart["patch_unwrap_proxy_p10_triangle_area_ratio"],
                f"{prefix}_mean_added_edge_normal_agreement": mean_or_none(normal_values),
                f"{prefix}_p90_added_edge_offset_ratio": percentile_or_none(offset_values, 90),
            }
        )
    return result


def unwrap_distortion_growth_metrics(
    nodes: list[dict[str, Any]],
    all_edges: list[tuple[int, int]],
    base_edges: list[tuple[int, int]],
    edge_quality: list[dict[str, float | tuple[int, int]]],
    candidate_limit: int = 0,
    bridge_only: bool = False,
    max_p90_distortion: float | None = None,
) -> dict[str, Any]:
    """Grow components by choosing the next edge with the least chart distortion."""

    import numpy as np

    node_count = len(nodes)
    result = {key: None for key in unwrap_distortion_growth_metric_names()}
    if node_count < 3:
        return result

    all_edge_set = {tuple(sorted(edge)) for edge in all_edges}
    grown_edges = {tuple(sorted(edge)) for edge in base_edges}
    candidates = [
        quality
        for quality in edge_quality
        if isinstance(quality.get("edge"), tuple)
        and tuple(sorted(quality["edge"])) in all_edge_set
        and tuple(sorted(quality["edge"])) not in grown_edges
    ]
    candidates.sort(
        key=lambda item: (
            -float(item["normal_agreement"]),
            float(item["offset_ratio"]),
            tuple(item["edge"]),
        )
    )

    added_qualities: list[dict[str, float | tuple[int, int]]] = []

    def component_roots(edges: set[tuple[int, int]]) -> tuple[list[int], dict[int, int]]:
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

    def mean_or_none(values: list[float]) -> float | None:
        return float(np.mean(np.array(values, dtype=np.float64))) if values else None

    def percentile_or_none(values: list[float], percentile: float) -> float | None:
        return float(np.percentile(np.array(values, dtype=np.float64), percentile)) if values else None

    def satisfies_p90_cap(chart: dict[str, Any]) -> bool:
        if max_p90_distortion is None:
            return True
        p90 = chart["patch_unwrap_proxy_p90_edge_distortion"]
        return p90 is not None and float(p90) <= max_p90_distortion

    for target in UNWRAP_GROWTH_TARGETS:
        chart = unwrap_proxy_metrics(nodes, sorted(grown_edges))
        while (
            (
                chart["patch_unwrap_proxy_largest_component_node_fraction"] < target
                or not satisfies_p90_cap(chart)
            )
            and candidates
        ):
            roots, sizes = component_roots(grown_edges)
            largest_root = max(sizes, key=sizes.get) if sizes else None
            best_index = None
            best_score = None
            best_chart = None
            search_candidate_indices = []
            for candidate_index, quality in enumerate(candidates):
                edge = tuple(sorted(quality["edge"]))
                left, right = edge
                if roots[left] == roots[right]:
                    continue
                if bridge_only and largest_root not in (roots[left], roots[right]):
                    continue
                search_candidate_indices.append(candidate_index)
            if candidate_limit > 0:
                search_candidate_indices = search_candidate_indices[:candidate_limit]
            for candidate_index in search_candidate_indices:
                quality = candidates[candidate_index]
                edge = tuple(sorted(quality["edge"]))
                trial_edges = set(grown_edges)
                trial_edges.add(edge)
                trial_chart = unwrap_proxy_metrics(nodes, sorted(trial_edges))
                trial_fraction = trial_chart["patch_unwrap_proxy_largest_component_node_fraction"]
                if trial_fraction <= chart["patch_unwrap_proxy_largest_component_node_fraction"]:
                    continue
                p90 = trial_chart["patch_unwrap_proxy_p90_edge_distortion"]
                if (
                    max_p90_distortion is not None
                    and (p90 is None or float(p90) > max_p90_distortion)
                ):
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
            grown_edges.add(tuple(sorted(chosen["edge"])))
            added_qualities.append(chosen)
            chart = best_chart

        suffix = unwrap_growth_suffix(target)
        prefix = f"patch_unwrap_distortion_growth_{suffix}"
        normal_values = [float(item["normal_agreement"]) for item in added_qualities]
        offset_values = [float(item["offset_ratio"]) for item in added_qualities]
        result.update(
            {
                f"{prefix}_reached": (
                    chart["patch_unwrap_proxy_largest_component_node_fraction"] >= target
                    and satisfies_p90_cap(chart)
                ),
                f"{prefix}_added_edge_count": len(added_qualities),
                f"{prefix}_node_count": chart["patch_unwrap_proxy_node_count"],
                f"{prefix}_edge_count": chart["patch_unwrap_proxy_edge_count"],
                f"{prefix}_triangle_count": chart["patch_unwrap_proxy_triangle_count"],
                f"{prefix}_largest_component_node_fraction": chart[
                    "patch_unwrap_proxy_largest_component_node_fraction"
                ],
                f"{prefix}_mean_edge_distortion": chart["patch_unwrap_proxy_mean_edge_distortion"],
                f"{prefix}_p90_edge_distortion": chart["patch_unwrap_proxy_p90_edge_distortion"],
                f"{prefix}_mean_triangle_area_ratio": chart["patch_unwrap_proxy_mean_triangle_area_ratio"],
                f"{prefix}_p10_triangle_area_ratio": chart["patch_unwrap_proxy_p10_triangle_area_ratio"],
                f"{prefix}_mean_added_edge_normal_agreement": mean_or_none(normal_values),
                f"{prefix}_p90_added_edge_offset_ratio": percentile_or_none(offset_values, 90),
            }
        )
    return result


def graph_component_roots(
    node_count: int,
    edges: set[tuple[int, int]],
) -> tuple[list[int], dict[int, int]]:
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


def graph_largest_component_fraction(node_count: int, edges: set[tuple[int, int]]) -> float:
    if node_count == 0:
        return 0.0
    _roots, sizes = graph_component_roots(node_count, edges)
    return max(sizes.values()) / node_count if sizes else 0.0


def local_atlas_percentile_or_none(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    import numpy as np

    return float(np.percentile(np.array(values, dtype=np.float64), percentile))


def component_local_chart_metrics(
    nodes: list[dict[str, Any]],
    grown_edges: set[tuple[int, int]],
    base_edges: set[tuple[int, int]],
    edge_quality: list[dict[str, float | tuple[int, int]]],
    base_roots: list[int] | None = None,
    chart_frames: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Measure a multi-chart atlas proxy anchored to the original pruned components."""

    import numpy as np

    node_count = len(nodes)
    if node_count == 0:
        return {
            "local_chart_largest_component_fraction": 0.0,
            "local_chart_p90_internal_edge_distortion": None,
            "local_chart_bridge_edge_count": 0,
            "local_chart_p90_bridge_offset_ratio": None,
            "local_chart_p10_bridge_normal_agreement": None,
            "local_chart_p10_bridge_chart_normal_agreement": None,
            "local_chart_quality_p90": None,
        }

    if base_roots is None:
        base_roots, _base_sizes = graph_component_roots(node_count, base_edges)
    if chart_frames is None:
        chart_frames = component_chart_frames(nodes, base_roots)
    edge_quality_by_edge = {
        tuple(sorted(quality["edge"])): quality
        for quality in edge_quality
        if isinstance(quality.get("edge"), tuple)
    }

    component_nodes: dict[int, list[int]] = defaultdict(list)
    for node_index, root in enumerate(base_roots):
        component_nodes[root].append(node_index)

    internal_distortions: list[float] = []
    for component in component_nodes.values():
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
    bridge_chart_normals: list[float] = []
    for edge in sorted(grown_edges):
        left, right = edge
        if base_roots[left] == base_roots[right]:
            continue
        quality = edge_quality_by_edge.get(edge)
        if quality is None:
            continue
        bridge_offsets.append(float(quality["offset_ratio"]))
        bridge_normals.append(float(quality["normal_agreement"]))
        chart_normal = bridge_chart_normal_agreement(edge, base_roots, chart_frames)
        if chart_normal is not None:
            bridge_chart_normals.append(chart_normal)

    internal_p90 = local_atlas_percentile_or_none(internal_distortions, 90)
    bridge_offset_p90 = local_atlas_percentile_or_none(bridge_offsets, 90)
    quality_values = [
        value for value in (internal_p90, bridge_offset_p90) if isinstance(value, (int, float))
    ]
    return {
        "local_chart_largest_component_fraction": graph_largest_component_fraction(
            node_count, grown_edges
        ),
        "local_chart_p90_internal_edge_distortion": internal_p90,
        "local_chart_bridge_edge_count": len(bridge_offsets),
        "local_chart_p90_bridge_offset_ratio": bridge_offset_p90,
        "local_chart_p10_bridge_normal_agreement": local_atlas_percentile_or_none(
            bridge_normals, 10
        ),
        "local_chart_p10_bridge_chart_normal_agreement": local_atlas_percentile_or_none(
            bridge_chart_normals, 10
        ),
        "local_chart_quality_p90": max(quality_values) if quality_values else None,
    }


def component_chart_frames(
    nodes: list[dict[str, Any]],
    base_roots: list[int],
) -> dict[int, dict[str, Any]]:
    """Build internal PCA frames for original components without exporting geometry."""

    import numpy as np

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
        _eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, 0]
        node_normals = np.array([nodes[index]["normal"] for index in component], dtype=np.float64)
        normal_lengths = np.linalg.norm(node_normals, axis=1)
        valid_normals = normal_lengths > 1e-6
        if np.any(valid_normals):
            mean_normal = np.mean(node_normals[valid_normals] / normal_lengths[valid_normals, None], axis=0)
            if np.dot(normal, mean_normal) < 0:
                normal = -normal
        frames[root] = {"normal": normal}
    return frames


def bridge_chart_normal_agreement(
    edge: tuple[int, int],
    base_roots: list[int],
    chart_frames: dict[int, dict[str, Any]],
) -> float | None:
    """Return scalar chart-normal agreement for a bridge between original components."""

    left, right = edge
    left_frame = chart_frames.get(base_roots[left])
    right_frame = chart_frames.get(base_roots[right])
    if left_frame is None or right_frame is None:
        return None
    import numpy as np

    return float(abs(np.dot(left_frame["normal"], right_frame["normal"])))


def local_atlas_growth_metrics(
    nodes: list[dict[str, Any]],
    all_edges: list[tuple[int, int]],
    base_edges: list[tuple[int, int]],
    edge_quality: list[dict[str, float | tuple[int, int]]],
    min_normal_agreement: float = 0.50,
    max_offset_ratio: float = 0.20,
    min_chart_normal_agreement: float = 0.0,
) -> dict[str, Any]:
    """Grow components with bridge-local quality while preserving per-component charts."""

    node_count = len(nodes)
    result = {key: None for key in local_atlas_growth_metric_names()}
    if node_count < 3:
        return result

    all_edge_set = {tuple(sorted(edge)) for edge in all_edges}
    base_edge_set = {tuple(sorted(edge)) for edge in base_edges}
    grown_edges = set(base_edge_set)
    base_roots, _base_sizes = graph_component_roots(node_count, base_edge_set)
    chart_frames = component_chart_frames(nodes, base_roots)
    candidates = [
        quality
        for quality in edge_quality
        if isinstance(quality.get("edge"), tuple)
        and tuple(sorted(quality["edge"])) in all_edge_set
        and tuple(sorted(quality["edge"])) not in grown_edges
    ]
    candidates.sort(
        key=lambda item: (
            -float(item["normal_agreement"]),
            float(item["offset_ratio"]),
            tuple(sorted(item["edge"])),
        )
    )
    added_edges: set[tuple[int, int]] = set()

    def quality_cap_satisfied(metrics: dict[str, Any]) -> bool:
        quality_p90 = metrics["local_chart_quality_p90"]
        return quality_p90 is not None and float(quality_p90) <= max_offset_ratio

    for target in UNWRAP_GROWTH_TARGETS:
        local_metrics = component_local_chart_metrics(
            nodes,
            grown_edges,
            base_edge_set,
            edge_quality,
            base_roots=base_roots,
            chart_frames=chart_frames,
        )
        while local_metrics["local_chart_largest_component_fraction"] < target and candidates:
            roots, sizes = graph_component_roots(node_count, grown_edges)
            largest_root = max(sizes, key=sizes.get) if sizes else None
            current_fraction = graph_largest_component_fraction(node_count, grown_edges)
            best_index = None
            best_score = None
            for candidate_index, quality in enumerate(candidates):
                edge = tuple(sorted(quality["edge"]))
                left, right = edge
                if roots[left] == roots[right]:
                    continue
                if largest_root not in (roots[left], roots[right]):
                    continue
                normal_agreement = float(quality["normal_agreement"])
                offset_ratio = float(quality["offset_ratio"])
                if normal_agreement < min_normal_agreement or offset_ratio > max_offset_ratio:
                    continue
                chart_normal_agreement = bridge_chart_normal_agreement(
                    edge, base_roots, chart_frames
                )
                if (
                    min_chart_normal_agreement > 0.0
                    and (
                        chart_normal_agreement is None
                        or chart_normal_agreement < min_chart_normal_agreement
                    )
                ):
                    continue
                trial_edges = set(grown_edges)
                trial_edges.add(edge)
                trial_fraction = graph_largest_component_fraction(node_count, trial_edges)
                if trial_fraction <= current_fraction:
                    continue
                if min_chart_normal_agreement > 0.0:
                    chart_score = chart_normal_agreement if chart_normal_agreement is not None else -1.0
                    score = (
                        -trial_fraction,
                        -chart_score,
                        offset_ratio,
                        -normal_agreement,
                        edge,
                    )
                else:
                    score = (
                        -trial_fraction,
                        offset_ratio,
                        -normal_agreement,
                        edge,
                    )
                if best_score is None or score < best_score:
                    best_index = candidate_index
                    best_score = score
            if best_index is None:
                break
            chosen = candidates.pop(best_index)
            edge = tuple(sorted(chosen["edge"]))
            grown_edges.add(edge)
            added_edges.add(edge)
            local_metrics = component_local_chart_metrics(
                nodes,
                grown_edges,
                base_edge_set,
                edge_quality,
                base_roots=base_roots,
                chart_frames=chart_frames,
            )

        global_chart = unwrap_proxy_metrics(nodes, sorted(grown_edges))
        suffix = unwrap_growth_suffix(target)
        prefix = f"patch_local_atlas_growth_{suffix}"
        result.update(
            {
                f"{prefix}_reached": (
                    local_metrics["local_chart_largest_component_fraction"] >= target
                    and quality_cap_satisfied(local_metrics)
                ),
                f"{prefix}_added_edge_count": len(added_edges),
                f"{prefix}_node_count": global_chart["patch_unwrap_proxy_node_count"],
                f"{prefix}_edge_count": global_chart["patch_unwrap_proxy_edge_count"],
                f"{prefix}_largest_component_node_fraction": local_metrics[
                    "local_chart_largest_component_fraction"
                ],
                f"{prefix}_local_chart_p90_internal_edge_distortion": local_metrics[
                    "local_chart_p90_internal_edge_distortion"
                ],
                f"{prefix}_bridge_edge_count": local_metrics["local_chart_bridge_edge_count"],
                f"{prefix}_p90_bridge_offset_ratio": local_metrics[
                    "local_chart_p90_bridge_offset_ratio"
                ],
                f"{prefix}_p10_bridge_normal_agreement": local_metrics[
                    "local_chart_p10_bridge_normal_agreement"
                ],
                f"{prefix}_p10_bridge_chart_normal_agreement": local_metrics[
                    "local_chart_p10_bridge_chart_normal_agreement"
                ],
                f"{prefix}_quality_p90": local_metrics["local_chart_quality_p90"],
                f"{prefix}_global_p90_edge_distortion": global_chart[
                    "patch_unwrap_proxy_p90_edge_distortion"
                ],
            }
        )
    return result


def mean_nested(items: list[dict[str, Any]], key_path: tuple[str, ...]) -> float:
    values: list[float] = []
    for item in items:
        value: Any = item
        for key in key_path:
            if value is None:
                break
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            values.append(float(value))
    return sum(values) / len(values) if values else 0.0


def mean_radius_sweep(items: list[dict[str, Any]], side: str) -> list[dict[str, float]]:
    by_radius: dict[float, list[dict[str, Any]]] = {}
    key = f"{side}_metrics"
    for item in items:
        metrics = item.get(key, {})
        for radius_metric in metrics.get("radius_sweep_components", []):
            by_radius.setdefault(float(radius_metric["radius"]), []).append(radius_metric)
    result: list[dict[str, float]] = []
    for radius, radius_items in sorted(by_radius.items()):
        result.append(
            {
                "radius": radius,
                "mean_component_count": sum(item["component_count"] for item in radius_items) / len(radius_items),
                "mean_largest_component_fraction": sum(item["largest_component_fraction"] for item in radius_items)
                / len(radius_items),
            }
        )
    return result


def mean_hdbscan_selected(items: list[dict[str, Any]], side: str) -> list[dict[str, float]]:
    by_threshold: dict[int, list[dict[str, Any]]] = {}
    key = f"{side}_metrics"
    for item in items:
        hdbscan_metrics = item.get(key, {}).get("hdbscan_components")
        if not hdbscan_metrics:
            continue
        for threshold_metric in hdbscan_metrics.get("selected_thresholds", []):
            by_threshold.setdefault(int(threshold_metric["threshold"]), []).append(threshold_metric)
    result: list[dict[str, float]] = []
    for threshold, threshold_items in sorted(by_threshold.items()):
        summary = {
            "threshold": threshold,
            "mean_selected_count": sum(item["selected_count"] for item in threshold_items) / len(threshold_items),
            "mean_selected_fraction": sum(item["selected_fraction"] for item in threshold_items)
            / len(threshold_items),
        }
        patch_items = [item.get("patch_quality") for item in threshold_items if item.get("patch_quality")]
        selected_patch_items = [
            item for item in patch_items if item and item.get("selected_cluster_count", 0) > 0
        ]
        if patch_items:
            summary["patch_quality_run_count"] = len(patch_items)
            summary["patch_quality_selected_run_count"] = len(selected_patch_items)
            for key in (
                "weighted_plane_rms",
                "weighted_planarity_ratio",
                "weighted_normal_coherence",
                "weighted_bbox_diagonal",
                "weighted_local_plane_rms",
                "weighted_local_plane_rms_p90",
                "weighted_local_planarity_ratio",
                "weighted_local_normal_agreement",
                "weighted_nearest_neighbor_mean",
                "weighted_nearest_neighbor_p90",
                "weighted_patch_graph_node_count",
                "weighted_patch_graph_edge_count",
                "weighted_patch_graph_mean_degree",
                "weighted_patch_graph_largest_component_fraction",
                "weighted_patch_graph_isolated_node_fraction",
                "weighted_patch_graph_mean_edge_normal_agreement",
                "weighted_patch_graph_p10_edge_normal_agreement",
                "weighted_patch_graph_mean_edge_plane_offset",
                "weighted_patch_graph_p90_edge_plane_offset",
                "weighted_patch_graph_mean_edge_offset_ratio",
                "weighted_patch_graph_p90_edge_offset_ratio",
                "weighted_patch_graph_mean_cell_plane_rms",
                "weighted_patch_graph_mean_cell_planarity_ratio",
                "weighted_patch_mesh_node_count",
                "weighted_patch_mesh_edge_count",
                "weighted_patch_mesh_triangle_count",
                "weighted_patch_mesh_used_edge_fraction",
                "weighted_patch_mesh_nonmanifold_edge_fraction",
                "weighted_patch_mesh_largest_component_fraction",
                "weighted_patch_mesh_largest_component_triangle_fraction",
                "weighted_patch_mesh_mean_edge_length",
                "weighted_patch_mesh_p90_edge_length",
                "weighted_patch_mesh_mean_triangle_area",
                "weighted_patch_mesh_p90_triangle_area",
                "weighted_patch_mesh_mean_triangle_normal_agreement",
                "weighted_patch_mesh_p10_triangle_normal_agreement",
                "weighted_patch_mesh_degenerate_triangle_fraction",
                "weighted_patch_mesh_pruned_edge_count",
                "weighted_patch_mesh_pruned_edge_keep_fraction",
                "weighted_patch_mesh_pruned_triangle_count",
                "weighted_patch_mesh_pruned_used_edge_fraction",
                "weighted_patch_mesh_pruned_nonmanifold_edge_fraction",
                "weighted_patch_mesh_pruned_largest_component_fraction",
                "weighted_patch_mesh_pruned_largest_component_triangle_fraction",
                "weighted_patch_mesh_pruned_mean_edge_length",
                "weighted_patch_mesh_pruned_p90_edge_length",
                "weighted_patch_mesh_pruned_mean_triangle_area",
                "weighted_patch_mesh_pruned_p90_triangle_area",
                "weighted_patch_mesh_pruned_mean_triangle_normal_agreement",
                "weighted_patch_mesh_pruned_p10_triangle_normal_agreement",
                "weighted_patch_mesh_pruned_degenerate_triangle_fraction",
                "weighted_patch_unwrap_proxy_node_count",
                "weighted_patch_unwrap_proxy_edge_count",
                "weighted_patch_unwrap_proxy_triangle_count",
                "weighted_patch_unwrap_proxy_largest_component_node_fraction",
                "weighted_patch_unwrap_proxy_chart_planarity_ratio",
                "weighted_patch_unwrap_proxy_chart_normal_agreement",
                "weighted_patch_unwrap_proxy_mean_edge_distortion",
                "weighted_patch_unwrap_proxy_p90_edge_distortion",
                "weighted_patch_unwrap_proxy_mean_triangle_area_ratio",
                "weighted_patch_unwrap_proxy_p10_triangle_area_ratio",
                "weighted_patch_unwrap_proxy_degenerate_triangle_fraction",
                *(f"weighted_{key}" for key in unwrap_growth_metric_names()),
                *(f"weighted_{key}" for key in unwrap_distortion_growth_metric_names()),
                *(f"weighted_{key}" for key in local_atlas_growth_metric_names()),
            ):
                values = [
                    float(item[key])
                    for item in selected_patch_items
                    if isinstance(item.get(key), (int, float))
                ]
                summary[f"mean_{key}"] = sum(values) / len(values) if values else None
        result.append(summary)
    return result


def first_radius_for_fraction(radius_sweep: list[dict[str, float]], threshold: float) -> float | None:
    for item in radius_sweep:
        if item["mean_largest_component_fraction"] >= threshold:
            return item["radius"]
    return None


def summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["blur_size"], []).append(run)
    summaries: list[dict[str, Any]] = []
    for blur_size, items in grouped.items():
        recto_sweep = mean_radius_sweep(items, "recto")
        verso_sweep = mean_radius_sweep(items, "verso")
        recto_hdbscan_selected = mean_hdbscan_selected(items, "recto")
        verso_hdbscan_selected = mean_hdbscan_selected(items, "verso")
        summaries.append(
            {
                "blur_size": blur_size,
                "repeat_count": len(items),
                "mean_runtime_seconds": sum(item["runtime_seconds"] for item in items) / len(items),
                "mean_recto_count": sum(item["recto_count"] for item in items) / len(items),
                "mean_verso_count": sum(item["verso_count"] for item in items) / len(items),
                "mean_total_count": sum(item["total_count"] for item in items) / len(items),
                "max_peak_memory_bytes": max(item["peak_memory_bytes"] for item in items),
                "mean_recto_orientation_coherence": mean_nested(items, ("recto_metrics", "orientation_coherence")),
                "mean_verso_orientation_coherence": mean_nested(items, ("verso_metrics", "orientation_coherence")),
                "mean_recto_nn": mean_nested(items, ("recto_metrics", "nearest_neighbor", "mean")),
                "mean_verso_nn": mean_nested(items, ("verso_metrics", "nearest_neighbor", "mean")),
                "mean_recto_component_count": mean_nested(items, ("recto_metrics", "components", "component_count")),
                "mean_verso_component_count": mean_nested(items, ("verso_metrics", "components", "component_count")),
                "mean_recto_largest_component_fraction": mean_nested(
                    items, ("recto_metrics", "components", "largest_component_fraction")
                ),
                "mean_verso_largest_component_fraction": mean_nested(
                    items, ("verso_metrics", "components", "largest_component_fraction")
                ),
                "mean_recto_radius20_component_count": mean_nested(
                    items, ("recto_metrics", "radius20_components", "component_count")
                ),
                "mean_verso_radius20_component_count": mean_nested(
                    items, ("verso_metrics", "radius20_components", "component_count")
                ),
                "mean_recto_radius20_largest_fraction": mean_nested(
                    items, ("recto_metrics", "radius20_components", "largest_component_fraction")
                ),
                "mean_verso_radius20_largest_fraction": mean_nested(
                    items, ("verso_metrics", "radius20_components", "largest_component_fraction")
                ),
                "recto_radius_sweep": recto_sweep,
                "verso_radius_sweep": verso_sweep,
                "recto_radius_at_90pct_connected": first_radius_for_fraction(recto_sweep, 0.9),
                "verso_radius_at_90pct_connected": first_radius_for_fraction(verso_sweep, 0.9),
                "mean_recto_hdbscan_largest_fraction": mean_nested(
                    items, ("recto_metrics", "hdbscan_components", "largest_cluster_fraction")
                ),
                "mean_verso_hdbscan_largest_fraction": mean_nested(
                    items, ("verso_metrics", "hdbscan_components", "largest_cluster_fraction")
                ),
                "recto_hdbscan_selected": recto_hdbscan_selected,
                "verso_hdbscan_selected": verso_hdbscan_selected,
            }
        )
    return sorted(summaries, key=lambda item: item["blur_size"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-json", type=Path, default=Path("examples/data_index_PHerc0332.json"))
    parser.add_argument("--sample", default="PHerc0332")
    parser.add_argument("--volume")
    parser.add_argument("--array", dest="array_path", default="3")
    parser.add_argument("--chunk", type=base.parse_chunk, default=[1, 4, 4], help="Chunk coordinate as z,y,x.")
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--surface-detection-url", default=DEFAULT_SURFACE_DETECTION_URL)
    parser.add_argument("--blur-sizes", type=parse_int_list, default=[5, 11])
    parser.add_argument("--seeds", type=parse_int_list, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sobel-chunks", type=int, default=4)
    parser.add_argument("--sobel-overlap", type=int, default=3)
    parser.add_argument("--threshold-der", type=float, default=0.075)
    parser.add_argument("--threshold-der2", type=float, default=0.002)
    parser.add_argument("--nn-sample", type=int, default=512)
    parser.add_argument("--connectivity-radii", type=parse_float_list, default=[4.0, 8.0, 12.0, 16.0, 20.0])
    parser.add_argument("--hdbscan-metrics", action="store_true")
    parser.add_argument("--hdbscan-epsilon", type=float, default=20.0)
    parser.add_argument("--hdbscan-selected-thresholds", type=parse_int_list, default=[512, 8000])
    parser.add_argument("--hdbscan-patch-quality", action="store_true")
    parser.add_argument("--hdbscan-patch-sample", type=int, default=2048)
    parser.add_argument("--hdbscan-local-continuity", action="store_true")
    parser.add_argument("--hdbscan-continuity-neighbors", type=int, default=16)
    parser.add_argument("--hdbscan-patch-graph", action="store_true")
    parser.add_argument("--hdbscan-patch-graph-cell-size", type=float, default=8.0)
    parser.add_argument("--hdbscan-patch-graph-min-cell-points", type=int, default=4)
    parser.add_argument("--hdbscan-patch-graph-neighbor-radius", type=int, default=1)
    parser.add_argument("--hdbscan-mesh-proxy", action="store_true")
    parser.add_argument("--hdbscan-mesh-prune", action="store_true")
    parser.add_argument("--hdbscan-mesh-prune-min-normal-agreement", type=float, default=0.75)
    parser.add_argument("--hdbscan-mesh-prune-max-offset-ratio", type=float, default=0.35)
    parser.add_argument("--hdbscan-unwrap-proxy", action="store_true")
    parser.add_argument("--hdbscan-unwrap-growth-proxy", action="store_true")
    parser.add_argument("--hdbscan-unwrap-distortion-growth-proxy", action="store_true")
    parser.add_argument("--hdbscan-unwrap-distortion-growth-candidate-limit", type=int, default=0)
    parser.add_argument("--hdbscan-unwrap-distortion-growth-bridge-only", action="store_true")
    parser.add_argument("--hdbscan-unwrap-distortion-growth-max-p90", type=float)
    parser.add_argument("--hdbscan-local-atlas-growth-proxy", action="store_true")
    parser.add_argument("--hdbscan-local-atlas-growth-min-normal-agreement", type=float, default=0.50)
    parser.add_argument("--hdbscan-local-atlas-growth-max-offset-ratio", type=float, default=0.20)
    parser.add_argument("--hdbscan-local-atlas-growth-min-chart-normal-agreement", type=float, default=0.0)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if args.hdbscan_patch_quality and not args.hdbscan_metrics:
        raise SystemExit("--hdbscan-patch-quality requires --hdbscan-metrics")
    if args.hdbscan_local_continuity and not args.hdbscan_patch_quality:
        raise SystemExit("--hdbscan-local-continuity requires --hdbscan-patch-quality")
    if args.hdbscan_patch_graph and not args.hdbscan_patch_quality:
        raise SystemExit("--hdbscan-patch-graph requires --hdbscan-patch-quality")
    if args.hdbscan_mesh_proxy and not args.hdbscan_patch_graph:
        raise SystemExit("--hdbscan-mesh-proxy requires --hdbscan-patch-graph")
    if args.hdbscan_mesh_prune and not args.hdbscan_mesh_proxy:
        raise SystemExit("--hdbscan-mesh-prune requires --hdbscan-mesh-proxy")
    if args.hdbscan_unwrap_proxy and not args.hdbscan_patch_graph:
        raise SystemExit("--hdbscan-unwrap-proxy requires --hdbscan-patch-graph")
    if args.hdbscan_unwrap_growth_proxy and not args.hdbscan_mesh_prune:
        raise SystemExit("--hdbscan-unwrap-growth-proxy requires --hdbscan-mesh-prune")
    if args.hdbscan_unwrap_distortion_growth_proxy and not args.hdbscan_mesh_prune:
        raise SystemExit("--hdbscan-unwrap-distortion-growth-proxy requires --hdbscan-mesh-prune")
    if args.hdbscan_unwrap_distortion_growth_candidate_limit < 0:
        raise SystemExit("--hdbscan-unwrap-distortion-growth-candidate-limit must be non-negative")
    if args.hdbscan_unwrap_distortion_growth_bridge_only and not args.hdbscan_unwrap_distortion_growth_proxy:
        raise SystemExit(
            "--hdbscan-unwrap-distortion-growth-bridge-only requires "
            "--hdbscan-unwrap-distortion-growth-proxy"
        )
    if (
        args.hdbscan_unwrap_distortion_growth_max_p90 is not None
        and args.hdbscan_unwrap_distortion_growth_max_p90 < 0
    ):
        raise SystemExit("--hdbscan-unwrap-distortion-growth-max-p90 must be non-negative")
    if args.hdbscan_unwrap_distortion_growth_max_p90 is not None and not args.hdbscan_unwrap_distortion_growth_proxy:
        raise SystemExit(
            "--hdbscan-unwrap-distortion-growth-max-p90 requires "
            "--hdbscan-unwrap-distortion-growth-proxy"
        )
    if args.hdbscan_local_atlas_growth_proxy and not args.hdbscan_mesh_prune:
        raise SystemExit("--hdbscan-local-atlas-growth-proxy requires --hdbscan-mesh-prune")
    if not 0.0 <= args.hdbscan_local_atlas_growth_min_normal_agreement <= 1.0:
        raise SystemExit("--hdbscan-local-atlas-growth-min-normal-agreement must be between 0 and 1")
    if args.hdbscan_local_atlas_growth_max_offset_ratio < 0:
        raise SystemExit("--hdbscan-local-atlas-growth-max-offset-ratio must be non-negative")
    if not 0.0 <= args.hdbscan_local_atlas_growth_min_chart_normal_agreement <= 1.0:
        raise SystemExit(
            "--hdbscan-local-atlas-growth-min-chart-normal-agreement must be between 0 and 1"
        )
    if args.hdbscan_continuity_neighbors < 2:
        raise SystemExit("--hdbscan-continuity-neighbors must be at least 2")
    if args.hdbscan_patch_graph_cell_size <= 0:
        raise SystemExit("--hdbscan-patch-graph-cell-size must be positive")
    if args.hdbscan_patch_graph_min_cell_points < 1:
        raise SystemExit("--hdbscan-patch-graph-min-cell-points must be positive")
    if args.hdbscan_patch_graph_neighbor_radius < 1:
        raise SystemExit("--hdbscan-patch-graph-neighbor-radius must be positive")
    if not 0.0 <= args.hdbscan_mesh_prune_min_normal_agreement <= 1.0:
        raise SystemExit("--hdbscan-mesh-prune-min-normal-agreement must be between 0 and 1")
    if args.hdbscan_mesh_prune_max_offset_ratio < 0:
        raise SystemExit("--hdbscan-mesh-prune-max-offset-ratio must be non-negative")

    torch = import_torch()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]

    source_text = fetch_text(args.surface_detection_url, args.timeout)
    source_sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    module = load_surface_detection(source_text)

    index = json.loads(args.index_json.read_text(encoding="utf-8"))
    volume_entry = base.select_volume(index, args.sample, args.volume)
    array = base.select_array(volume_entry, args.array_path)
    if array.get("compressor") is not None:
        raise ValueError(f"compressed Zarr chunks are not supported by this ablation: {array.get('compressor')!r}")

    url = base.chunk_url(volume_entry["url"], array["path"], args.chunk)
    data = base.fetch_bytes(url, args.timeout)
    chunk_sha256 = hashlib.sha256(data).hexdigest()
    chunk = base.decode_uncompressed_chunk(data, array)
    roi = base.center_crop(chunk, args.crop_size)
    del chunk

    volume = torch.from_numpy(roi).to(device=args.device, dtype=dtype)
    reference_vector = torch.tensor([1.0, 0.0, 0.0], device=args.device, dtype=torch.float32)

    # Initialize cached Sobel kernels before timed runs.
    _ = run_one(module, volume, reference_vector, args, args.blur_sizes[0], args.seeds[0])

    runs = [
        run_one(module, volume, reference_vector, args, blur_size, seed)
        for blur_size in args.blur_sizes
        for seed in args.seeds
    ]
    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "policy": (
            "Fetched one public OME-Zarr chunk and one upstream source file into memory/temp storage, "
            "ran seeded surface_detection blur-size ablations, and wrote point counts/timings only. "
            "Raw chunk bytes, extracted volumes, point clouds, HDBSCAN labels, meshes, and upstream "
            "source snapshots were not saved."
        ),
        "environment": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": args.device,
            "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
            "dtype": args.dtype,
        },
        "upstream": {
            "surface_detection_url": args.surface_detection_url,
            "surface_detection_sha256": source_sha256,
        },
        "source": {
            "index_json": str(args.index_json),
            "sample_id": args.sample,
            "volume_name": volume_entry["name"],
            "volume_url": volume_entry["url"],
            "array_path": array["path"],
            "array_shape": array.get("shape"),
            "array_chunks": array.get("chunks"),
            "array_dtype": array.get("dtype"),
            "chunk": args.chunk,
            "chunk_url": url,
            "chunk_byte_count": len(data),
            "chunk_sha256": chunk_sha256,
            "roi_shape": list(roi.shape),
        },
        "parameters": {
            "blur_sizes": args.blur_sizes,
            "seeds": args.seeds,
            "window_size": args.window_size,
            "stride": args.stride,
            "sobel_chunks": args.sobel_chunks,
            "sobel_overlap": args.sobel_overlap,
            "threshold_der": args.threshold_der,
            "threshold_der2": args.threshold_der2,
            "nn_sample": args.nn_sample,
            "connectivity_radii": args.connectivity_radii,
            "hdbscan_metrics": args.hdbscan_metrics,
            "hdbscan_epsilon": args.hdbscan_epsilon,
            "hdbscan_selected_thresholds": args.hdbscan_selected_thresholds,
            "hdbscan_patch_quality": args.hdbscan_patch_quality,
            "hdbscan_patch_sample": args.hdbscan_patch_sample,
            "hdbscan_local_continuity": args.hdbscan_local_continuity,
            "hdbscan_continuity_neighbors": args.hdbscan_continuity_neighbors,
            "hdbscan_patch_graph": args.hdbscan_patch_graph,
            "hdbscan_patch_graph_cell_size": args.hdbscan_patch_graph_cell_size,
            "hdbscan_patch_graph_min_cell_points": args.hdbscan_patch_graph_min_cell_points,
            "hdbscan_patch_graph_neighbor_radius": args.hdbscan_patch_graph_neighbor_radius,
            "hdbscan_mesh_proxy": args.hdbscan_mesh_proxy,
            "hdbscan_mesh_prune": args.hdbscan_mesh_prune,
            "hdbscan_mesh_prune_min_normal_agreement": args.hdbscan_mesh_prune_min_normal_agreement,
            "hdbscan_mesh_prune_max_offset_ratio": args.hdbscan_mesh_prune_max_offset_ratio,
            "hdbscan_unwrap_proxy": args.hdbscan_unwrap_proxy,
            "hdbscan_unwrap_growth_proxy": args.hdbscan_unwrap_growth_proxy,
            "hdbscan_unwrap_distortion_growth_proxy": args.hdbscan_unwrap_distortion_growth_proxy,
            "hdbscan_unwrap_distortion_growth_candidate_limit": (
                args.hdbscan_unwrap_distortion_growth_candidate_limit
            ),
            "hdbscan_unwrap_distortion_growth_bridge_only": (
                args.hdbscan_unwrap_distortion_growth_bridge_only
            ),
            "hdbscan_unwrap_distortion_growth_max_p90": args.hdbscan_unwrap_distortion_growth_max_p90,
            "hdbscan_local_atlas_growth_proxy": args.hdbscan_local_atlas_growth_proxy,
            "hdbscan_local_atlas_growth_min_normal_agreement": (
                args.hdbscan_local_atlas_growth_min_normal_agreement
            ),
            "hdbscan_local_atlas_growth_max_offset_ratio": (
                args.hdbscan_local_atlas_growth_max_offset_ratio
            ),
            "hdbscan_local_atlas_growth_min_chart_normal_agreement": (
                args.hdbscan_local_atlas_growth_min_chart_normal_agreement
            ),
        },
        "runs": runs,
        "summaries": summarize_runs(runs),
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.json_out}")
    for item in report["summaries"]:
        print(
            f"blur_size={item['blur_size']} mean_total={item['mean_total_count']:.1f} "
            f"mean_recto={item['mean_recto_count']:.1f} mean_verso={item['mean_verso_count']:.1f} "
            f"recto_coh={item['mean_recto_orientation_coherence']:.3f} "
            f"verso_coh={item['mean_verso_orientation_coherence']:.3f} "
            f"recto_largest={item['mean_recto_largest_component_fraction']:.3f} "
            f"verso_largest={item['mean_verso_largest_component_fraction']:.3f} "
            f"recto_r90={item['recto_radius_at_90pct_connected']} "
            f"verso_r90={item['verso_radius_at_90pct_connected']} "
            f"recto_hdbscan_largest={item['mean_recto_hdbscan_largest_fraction']:.3f} "
            f"verso_hdbscan_largest={item['mean_verso_hdbscan_largest_fraction']:.3f} "
            f"mean_runtime={item['mean_runtime_seconds']:.4f}s "
            f"max_peak={item['max_peak_memory_bytes'] / (1024 * 1024):.1f}MiB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
