#!/usr/bin/env python3
"""Benchmark small-volume blur choices before Thaumato-style edge detection.

The script reads one public OME-Zarr chunk into memory, crops a bounded ROI,
compares denoising kernels, and writes metrics plus a small derived montage. It
does not write raw chunk bytes or extracted volumes to disk.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def import_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by CLI environment checks.
        raise SystemExit(
            "Missing dependency: numpy. Install experiment dependencies with "
            "`python3 -m pip install -r requirements-experiments.txt`."
        ) from exc
    return np


def import_pillow():
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - exercised by CLI environment checks.
        raise SystemExit(
            "Missing dependency: pillow. Install experiment dependencies with "
            "`python3 -m pip install -r requirements-experiments.txt`."
        ) from exc
    return Image, ImageDraw


def fetch_bytes(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "vesuvius-2026-thaumato-blur/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_chunk(value: str) -> list[int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("chunk must be z,y,x")
    try:
        chunk = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("chunk coordinates must be integers") from exc
    if any(coord < 0 for coord in chunk):
        raise argparse.ArgumentTypeError("chunk coordinates must be non-negative")
    return chunk


def chunk_url(volume_url: str, array_path: str, chunk: list[int]) -> str:
    return f"{volume_url.rstrip('/')}/{array_path}/{'/'.join(str(coord) for coord in chunk)}"


def select_volume(index: dict[str, Any], sample_id: str, volume_name: str | None) -> dict[str, Any]:
    for sample in index.get("samples", []):
        if sample.get("sample_id") != sample_id:
            continue
        volumes = sample.get("categories", {}).get("volumes", {}).get("entries", [])
        zarr_volumes = [entry for entry in volumes if entry.get("kind") == "zarr"]
        if volume_name is None:
            if not zarr_volumes:
                raise ValueError(f"sample {sample_id} has no indexed zarr volumes")
            return zarr_volumes[0]
        for entry in zarr_volumes:
            if entry.get("name") == volume_name:
                return entry
    raise ValueError(f"could not find sample={sample_id} volume={volume_name or '<first zarr>'}")


def select_array(volume: dict[str, Any], array_path: str | None) -> dict[str, Any]:
    arrays = volume.get("zarr", {}).get("arrays", [])
    if not arrays:
        raise ValueError(f"volume {volume.get('name')} has no indexed arrays")
    if array_path is None:
        return arrays[-1]
    for array in arrays:
        if array.get("path") == array_path:
            return array
    raise ValueError(f"could not find array path {array_path} in volume {volume.get('name')}")


def numpy_dtype(zarr_dtype: str):
    np = import_numpy()
    return np.dtype(zarr_dtype)


def decode_uncompressed_chunk(data: bytes, array: dict[str, Any]):
    np = import_numpy()
    dtype = numpy_dtype(array["dtype"])
    chunks = array.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != 3:
        raise ValueError(f"expected 3D chunk metadata, got {chunks!r}")
    expected = math.prod(chunks) * dtype.itemsize
    if len(data) != expected:
        raise ValueError(
            f"expected an uncompressed full chunk of {expected} bytes from {chunks} {dtype}, "
            f"got {len(data)} bytes"
        )
    return np.frombuffer(data, dtype=dtype).reshape(tuple(chunks)).astype(np.float32)


def center_crop(volume, crop_size: int):
    shape = volume.shape
    if crop_size <= 0:
        raise ValueError("crop size must be positive")
    if any(crop_size > axis for axis in shape):
        raise ValueError(f"crop size {crop_size} exceeds chunk shape {shape}")
    starts = [(axis - crop_size) // 2 for axis in shape]
    return volume[
        starts[0] : starts[0] + crop_size,
        starts[1] : starts[1] + crop_size,
        starts[2] : starts[2] + crop_size,
    ].copy()


def convolve1d_reflect(volume, kernel, axis: int):
    np = import_numpy()
    radius = len(kernel) // 2
    padded = np.pad(volume, [(radius, radius) if idx == axis else (0, 0) for idx in range(3)], mode="reflect")
    out = np.zeros_like(volume, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        slices = [slice(None)] * 3
        slices[axis] = slice(offset, offset + volume.shape[axis])
        out += np.float32(weight) * padded[tuple(slices)]
    return out


def separable_filter(volume, kernel):
    out = volume.astype("float32", copy=False)
    for axis in range(3):
        out = convolve1d_reflect(out, kernel, axis)
    return out


def smooth_other_axes(volume, smooth_kernel, derivative_axis: int):
    out = volume
    for axis in range(3):
        if axis != derivative_axis:
            out = convolve1d_reflect(out, smooth_kernel, axis)
    return out


def box_kernel(size: int):
    np = import_numpy()
    if size % 2 != 1:
        raise ValueError("box kernel size must be odd")
    return np.full(size, 1.0 / size, dtype=np.float32)


def gaussian_kernel(size: int, sigma: float):
    np = import_numpy()
    if size % 2 != 1:
        raise ValueError("gaussian kernel size must be odd")
    x = np.arange(size, dtype=np.float32) - (size // 2)
    weights = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return (weights / weights.sum()).astype(np.float32)


def sobel_like_gradient(volume):
    np = import_numpy()
    derivative = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    smooth = np.array([1.0, 2.0, 1.0], dtype=np.float32)

    gx = smooth_other_axes(convolve1d_reflect(volume, derivative, 2), smooth, derivative_axis=2)
    gy = smooth_other_axes(convolve1d_reflect(volume, derivative, 1), smooth, derivative_axis=1)
    gz = smooth_other_axes(convolve1d_reflect(volume, derivative, 0), smooth, derivative_axis=0)
    return np.sqrt(gx * gx + gy * gy + gz * gz, dtype=np.float32)


def percentile_stats(values, percentiles: list[float]) -> dict[str, float]:
    np = import_numpy()
    result = {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }
    for percentile in percentiles:
        result[f"p{percentile:g}"] = float(np.percentile(values, percentile))
    return result


def normalize_image(values, low: float, high: float):
    np = import_numpy()
    if high <= low:
        high = low + 1.0
    scaled = (values.astype(np.float32) - low) / (high - low)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def write_montage(path: Path, results: list[dict[str, Any]], slice_index: int) -> None:
    np = import_numpy()
    Image, ImageDraw = import_pillow()
    intensity_values = np.concatenate([item["filtered"][slice_index].ravel() for item in results])
    gradient_values = np.concatenate([item["gradient"][slice_index].ravel() for item in results])
    i_low, i_high = np.percentile(intensity_values, [1, 99])
    g_low, g_high = np.percentile(gradient_values, [1, 99])

    tile = results[0]["filtered"].shape[1]
    label_height = 22
    rows = 2
    cols = len(results)
    image = Image.new("RGB", (cols * tile, rows * (tile + label_height)), "white")
    draw = ImageDraw.Draw(image)
    for col, item in enumerate(results):
        intensity = normalize_image(item["filtered"][slice_index], float(i_low), float(i_high))
        gradient = normalize_image(item["gradient"][slice_index], float(g_low), float(g_high))
        for row, (label, array) in enumerate((("intensity", intensity), ("sobel", gradient))):
            x = col * tile
            y = row * (tile + label_height)
            image.paste(Image.fromarray(array, mode="L").convert("RGB"), (x, y + label_height))
            draw.text((x + 4, y + 4), f"{item['name']} {label}", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def benchmark_filters(roi, edge_percentile: float) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    np = import_numpy()
    filters = [
        ("identity", None),
        ("box5", box_kernel(5)),
        ("box11_thaumato", box_kernel(11)),
        ("gaussian11_sigma2", gaussian_kernel(11, 2.0)),
    ]

    results: list[dict[str, Any]] = []
    baseline_gradient = sobel_like_gradient(roi)
    edge_threshold = float(np.percentile(baseline_gradient, edge_percentile))
    baseline_edge_mask = baseline_gradient >= edge_threshold
    baseline_edge_sum = float(np.sum(baseline_gradient[baseline_edge_mask]))
    baseline_metrics = percentile_stats(baseline_gradient, [50, 90, 95, 99])

    for name, kernel in filters:
        start = time.perf_counter()
        filtered = roi.astype(np.float32, copy=True) if kernel is None else separable_filter(roi, kernel)
        gradient = baseline_gradient if kernel is None else sobel_like_gradient(filtered)
        runtime = time.perf_counter() - start
        edge_sum = float(np.sum(gradient[baseline_edge_mask]))
        high_gradient_fraction = float(np.mean(gradient >= edge_threshold))
        rms_delta = float(np.sqrt(np.mean((filtered - roi) ** 2)))
        results.append(
            {
                "name": name,
                "runtime_seconds": runtime,
                "filtered": filtered,
                "gradient": gradient,
                "metrics": {
                    "intensity": percentile_stats(filtered, [1, 50, 99]),
                    "gradient": percentile_stats(gradient, [50, 90, 95, 99]),
                    "edge_threshold_from_identity_p95": edge_threshold,
                    "fraction_above_identity_p95": high_gradient_fraction,
                    "edge_retention_on_identity_p95_mask": edge_sum / baseline_edge_sum
                    if baseline_edge_sum > 0
                    else 0.0,
                    "rms_delta_from_input": rms_delta,
                },
            }
        )
    return results, edge_threshold, baseline_metrics


def public_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": item["name"],
        "runtime_seconds": item["runtime_seconds"],
        "metrics": item["metrics"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-json", type=Path, default=Path("examples/data_index_PHerc0332.json"))
    parser.add_argument("--sample", default="PHerc0332")
    parser.add_argument("--volume")
    parser.add_argument("--array", dest="array_path", default="5")
    parser.add_argument("--chunk", type=parse_chunk, default=[0, 1, 1], help="Chunk coordinate as z,y,x.")
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--edge-percentile", type=float, default=95.0)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--montage-out", type=Path)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    index = json.loads(args.index_json.read_text(encoding="utf-8"))
    volume = select_volume(index, args.sample, args.volume)
    array = select_array(volume, args.array_path)
    if array.get("compressor") is not None:
        raise ValueError(f"compressed Zarr chunks are not supported by this benchmark: {array.get('compressor')!r}")

    url = chunk_url(volume["url"], array["path"], args.chunk)
    data = fetch_bytes(url, args.timeout)
    chunk_sha256 = hashlib.sha256(data).hexdigest()
    chunk = decode_uncompressed_chunk(data, array)
    roi = center_crop(chunk, args.crop_size)
    del chunk

    results, edge_threshold, baseline_metrics = benchmark_filters(roi, args.edge_percentile)
    montage_path = args.montage_out
    if montage_path is not None:
        write_montage(montage_path, results, slice_index=roi.shape[0] // 2)

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "policy": (
            "Fetched one public OME-Zarr chunk into memory, cropped a small ROI, wrote metrics "
            "and an optional derived 2D montage only; raw chunk bytes and extracted volumes were not saved."
        ),
        "source": {
            "index_json": str(args.index_json),
            "sample_id": args.sample,
            "volume_name": volume["name"],
            "volume_url": volume["url"],
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
        "experiment": {
            "track": "Thaumato Anakalyptor initial blur preprocessing",
            "hypothesis": (
                "The current 11^3 uniform blur may suppress useful edge detail before Sobel-like "
                "surface extraction; smaller or smoother kernels can be benchmarked on tiny ROIs "
                "before spending GPU time on full-pipeline runs."
            ),
            "edge_percentile": args.edge_percentile,
            "edge_threshold": edge_threshold,
            "baseline_gradient_metrics": baseline_metrics,
            "filters": [public_result(item) for item in results],
            "montage": str(montage_path) if montage_path is not None else None,
        },
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.json_out}")
    if montage_path is not None:
        print(f"Wrote {montage_path}")
    for item in report["experiment"]["filters"]:
        metrics = item["metrics"]
        print(
            f"{item['name']}: runtime={item['runtime_seconds']:.3f}s "
            f"edge_retention={metrics['edge_retention_on_identity_p95_mask']:.3f} "
            f"fraction_above_identity_p95={metrics['fraction_above_identity_p95']:.3f} "
            f"rms_delta={metrics['rms_delta_from_input']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
