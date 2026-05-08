#!/usr/bin/env python3
"""Create a tiny copied-input VC3D tifxyz smoke volpkg.

The upstream SurfacePatchIndex path can rewrite tifxyz files for mmap. This
helper intentionally copies inputs into the smoke volpkg instead of symlinking
the public fixture cache.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


REQUIRED_TIFXYZ_FILES = ("meta.json", "x.tif", "y.tif", "z.tif")
SURFACE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class SurfaceSpec:
    name: str
    tifxyz_dir: Path


def parse_surface_spec(value: str) -> SurfaceSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("surface must be NAME=/path/to/tifxyz")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not SURFACE_NAME_RE.fullmatch(name):
        raise argparse.ArgumentTypeError(
            "surface name must use only letters, digits, dots, underscores, or dashes"
        )
    path = Path(raw_path).expanduser()
    return SurfaceSpec(name=name, tifxyz_dir=path)


def parse_row(value: str) -> tuple[str, str]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("row must be LEFT_NAME,RIGHT_NAME")
    return (parts[0], parts[1])


def missing_tifxyz_files(tifxyz_dir: Path) -> list[str]:
    return [
        filename
        for filename in REQUIRED_TIFXYZ_FILES
        if not (tifxyz_dir / filename).is_file()
    ]


def validate_plan(surfaces: list[SurfaceSpec], rows: list[tuple[str, str]]) -> None:
    names = [surface.name for surface in surfaces]
    if len(set(names)) != len(names):
        raise ValueError("surface names must be unique")
    if len(surfaces) < 2:
        raise ValueError("at least two surfaces are required")
    missing_by_surface = {
        surface.name: missing_tifxyz_files(surface.tifxyz_dir)
        for surface in surfaces
    }
    missing_by_surface = {
        name: missing
        for name, missing in missing_by_surface.items()
        if missing
    }
    if missing_by_surface:
        details = "; ".join(
            f"{name}: {', '.join(missing)}"
            for name, missing in sorted(missing_by_surface.items())
        )
        raise FileNotFoundError(f"missing required tifxyz files: {details}")
    known = set(names)
    unknown = sorted({name for row in rows for name in row if name not in known})
    if unknown:
        raise ValueError(f"rows reference unknown surfaces: {', '.join(unknown)}")


def build_volpkg(
    out_dir: Path,
    surfaces: list[SurfaceSpec],
    rows: list[tuple[str, str]],
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    validate_plan(surfaces, rows)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(out_dir)

    paths_dir = out_dir / "paths"
    paths_dir.mkdir(parents=True, exist_ok=True)
    copied_files: list[str] = []
    for surface in surfaces:
        surface_out = paths_dir / surface.name
        surface_out.mkdir(parents=True, exist_ok=False)
        for filename in REQUIRED_TIFXYZ_FILES:
            source = surface.tifxyz_dir / filename
            destination = surface_out / filename
            shutil.copy2(source, destination)
            copied_files.append(str(destination))

    merge_json = out_dir / "merge.json"
    payload = {"rows": [[left, right] for left, right in rows]}
    merge_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "out_dir": str(out_dir),
        "merge_json": str(merge_json),
        "surface_count": len(surfaces),
        "row_count": len(rows),
        "copied_file_count": len(copied_files),
        "copied_inputs": True,
        "symlinks_created": False,
    }


def build_run_command(binary: Path, merge_json: Path, *, ransac_iters: int, anchor_cap: int) -> str:
    return (
        f"{binary} --merge {merge_json} --ransac-iters {ransac_iters} "
        f"--ransac-seed 1 --anchor-cap {anchor_cap} --strip-cols 1"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--surface",
        action="append",
        required=True,
        type=parse_surface_spec,
        help="Surface mapping as NAME=/path/to/mesh/tifxyz; repeat for each surface.",
    )
    parser.add_argument(
        "--row",
        action="append",
        required=True,
        type=parse_row,
        help="Merge row as LEFT_NAME,RIGHT_NAME; repeat for each edge row.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--vc-merge-bin",
        type=Path,
        default=Path("external/villa/build/volume-cartographer-gcc12/bin/vc_merge_tifxyz"),
    )
    parser.add_argument("--ransac-iters", type=int, default=200)
    parser.add_argument("--anchor-cap", type=int, default=128)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    summary = build_volpkg(
        args.out_dir,
        list(args.surface),
        list(args.row),
        overwrite=args.overwrite,
    )
    command = build_run_command(
        args.vc_merge_bin,
        Path(summary["merge_json"]),
        ransac_iters=args.ransac_iters,
        anchor_cap=args.anchor_cap,
    )
    summary["example_command"] = command

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
