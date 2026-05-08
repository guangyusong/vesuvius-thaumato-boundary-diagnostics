# VC3D Merge Summary Scalar Adapter

- timestamp UTC: 2026-05-08T02:27:55+00:00
- source tool: `vc_merge_tifxyz`
- source summary: `data/vc3d_merge_smoke/PHerc0139_overlap_r11_r12.volpkg/paths/pherc0139_a_merged/pherc0139_a_merged_summary.json`
- public context: `reports/2026-05-08_public_vc3d_merge_context_refresh.md`
- upstream schema: `ScrollPrize/villa@c257b402fce35b5967fee752e415067a18d281c4` `volume-cartographer/apps/src/vc_merge_tifxyz.cpp`
- script: `scripts/summarize_vc3d_merge_summary_adapter.py`
- machine: current `g2-standard-16` L4 VM
- new data downloads: none
- GPU runtime: none
- cost impact: CPU-only scalar summary generation, negligible incremental cost
- command: `python3 scripts/summarize_vc3d_merge_summary_adapter.py`
- hypothesis: current `vc_merge_tifxyz` outputs can feed a read-only scalar bridge-QA adapter without copying paths, names, geometry, or metadata writes
- visual checks: none; scalar-only JSON and Markdown table output
- saved data policy: scalar aggregate JSON only; no raw chunks, surface paths, names, coordinates, OBJ, tifxyz outputs, meshes, predictions, letters, or titles

## Summary

- surfaces: `2`
- edges: `1`
- strip count: `2`
- ready for bridge QA: `1`
- needs upstream merge review: `0`
- metadata write action: `none_scalar_only`
- decision: `adapter_schema_ready_for_current_main_summary`

## Adapter Contract

- input source: `vc_merge_tifxyz summary.json`
- adapter mode: read-only scalar bridge-candidate review
- overlap metadata action: `none_scalar_only`
- fields not copied: surface names, surface paths, output directories, OBJ paths, coordinates, meshes, tifxyz payloads

## Edge Review Rows

| edge | anchors | inlier frac | pair scale delta | real overlap min | bucket | reasons |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 47534 | 0.705495 | 0.006605 | 46894 | ready_for_bridge_qa | none |

## Integration Boundary

This adapter is read-only. It summarizes current `vc_merge_tifxyz` diagnostics for review and does not reimplement N-way merge, alignment, blending, rasterization, or overlap metadata writing.

It is not an `overlapping.json` writer, does not mutate VC3D metadata, and does not claim to solve an active VC3D issue by itself.

This is not an unwrap, text, letter, title, or ink claim.

## Ranked Next Steps

- Keep this adapter read-only and scalar-only inside the progress-prize package.
- When a real current-main vc_merge_tifxyz summary is available, run this adapter and compare its review buckets to the existing multi-chart bridge-QA rows.
- Only propose upstream code after a bounded current-main reproduction shows a concrete failure not covered by existing VC3D merge tooling.

## Storage Policy

Read vc_merge_tifxyz summary scalars only; do not carry surface names, paths, coordinates, OBJ paths, tifxyz outputs, meshes, overlap metadata, predictions, ink, letters, or titles into the adapter output.
