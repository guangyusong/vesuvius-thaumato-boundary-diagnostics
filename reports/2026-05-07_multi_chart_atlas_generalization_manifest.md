# Multi-Chart Atlas Generalization Manifest

- timestamp UTC: 2026-05-07T23:50:24Z
- code path: `scripts/plot_thaumato_patch_graph_growth_qa.py`
- hypothesis: singleton endpoint placement in the scalar multi-chart atlas should
  generalize beyond the `13,4,3` recto seed `1` six-bridge case. The next two
  bounded checks should score all accepted local-atlas bridges, or report a
  scalar reason for any remaining unplaced bridge.
- data plan: rerun two public PHerc1667 OME-Zarr chunks, array `3`, crop
  `128`, fetched transiently as needed:
  - `13,4,4` verso seed `0`, neighbor radius `2`, target `0.75`; prior local
    atlas reached target with final quality `0.087831` but direct-global p90
    `0.506859`.
  - `14,4,4` verso seed `0`, neighbor radius `3`, target `0.75`; prior cap-base
    local atlas reached target with final quality `0.161402` but direct-global
    p90 `0.300417`.
- expected runtime: under 10 minutes total on the current `g2-standard-16` L4 VM.
- expected incremental VM cost: under USD `0.20`.
- outputs: scalar JSON plus raster PNG QA images only; no raw chunks, extracted
  volumes, point clouds, meshes, flattened coordinates, component IDs, bridge
  endpoints, predictions, text, ink, letters, titles, or model weights.

## Planned Commands

```bash
/usr/bin/time -f 'wall_seconds %e' .venv/bin/python scripts/plot_thaumato_patch_graph_growth_qa.py \
  --index-json reports/data_index_2026-05-06.json \
  --sample PHerc1667 \
  --volume 20231107190228-3.240um-88keV-masked.zarr \
  --array 3 \
  --chunk 13,4,4 \
  --side verso \
  --crop-size 128 \
  --blur-size 11 \
  --seed 0 \
  --device cuda \
  --dtype float32 \
  --window-size 9 \
  --stride 1 \
  --hdbscan-epsilon 20 \
  --hdbscan-threshold 8000 \
  --hdbscan-patch-sample 2048 \
  --patch-graph-cell-size 8 \
  --patch-graph-min-cell-points 4 \
  --patch-graph-neighbor-radius 2 \
  --mesh-prune-min-normal-agreement 0.80 \
  --mesh-prune-max-offset-ratio 0.30 \
  --growth-target 0.75 \
  --distortion-growth-max-p90 0.20 \
  --local-atlas-bridge-min-normal-agreement 0.50 \
  --local-atlas-bridge-max-offset-ratio 0.20 \
  --candidate-diagnostic-limit 24 \
  --local-atlas-only \
  --json-out reports/thaumato_patch_graph_growth_qa_PHerc1667_13_4_4_verso_seed0_multi_chart_atlas_generalization75_2026-05-07.json \
  --png-out reports/thaumato_patch_graph_growth_qa_PHerc1667_13_4_4_verso_seed0_multi_chart_atlas_generalization75_2026-05-07.png

/usr/bin/time -f 'wall_seconds %e' .venv/bin/python scripts/plot_thaumato_patch_graph_growth_qa.py \
  --index-json reports/data_index_2026-05-06.json \
  --sample PHerc1667 \
  --volume 20231107190228-3.240um-88keV-masked.zarr \
  --array 3 \
  --chunk 14,4,4 \
  --side verso \
  --crop-size 128 \
  --blur-size 11 \
  --seed 0 \
  --device cuda \
  --dtype float32 \
  --window-size 9 \
  --stride 1 \
  --hdbscan-epsilon 20 \
  --hdbscan-threshold 8000 \
  --hdbscan-patch-sample 2048 \
  --patch-graph-cell-size 8 \
  --patch-graph-min-cell-points 4 \
  --patch-graph-neighbor-radius 3 \
  --mesh-prune-min-normal-agreement 0.80 \
  --mesh-prune-max-offset-ratio 0.30 \
  --cap-base-p90 \
  --growth-target 0.75 \
  --distortion-growth-max-p90 0.20 \
  --local-atlas-bridge-min-normal-agreement 0.50 \
  --local-atlas-bridge-max-offset-ratio 0.20 \
  --candidate-diagnostic-limit 64 \
  --local-atlas-only \
  --json-out reports/thaumato_patch_graph_growth_qa_PHerc1667_14_4_4_verso_seed0_multi_chart_atlas_generalization75_2026-05-07.json \
  --png-out reports/thaumato_patch_graph_growth_qa_PHerc1667_14_4_4_verso_seed0_multi_chart_atlas_generalization75_2026-05-07.png
```

## Evaluation Result

| case | wall | reached | steps | final fraction | final local quality | direct-global p90 | multi-chart p90 | placed components | placed node fraction | scored bridges | singleton placements | tiny-component placements | unplaced bridges | decision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PHerc1667 `13,4,4` verso seed `0` | `24.07s` | true | `2` | `0.751244` | `0.087831` | `0.506859` | `0.010780` | `3` | `0.751244` | `2` | `1` | `1` | `0` | `multi_chart_atlas_resolves_direct_global_blocker` |
| PHerc1667 `14,4,4` verso seed `0` | `189.71s` | true | `3` | `0.758865` | `0.161402` | `0.300417` | `0.039103` | `4` | `0.758865` | `3` | `0` | `2` | `0` | `multi_chart_atlas_resolves_direct_global_blocker` |

The first `14,4,4` rerun, before the tiny-component placement fix, was useful
failure analysis: it scored one of three accepted bridges and left two bridges
unplaced, both because the newly attached original components had fewer than
three nodes and therefore no PCA chart frame. The scorer now projects every node
in a too-small original component into the neighboring placed chart frame,
counting these through
`multi_chart_atlas_too_small_component_placement_count`. This preserves full
scalar reporting while avoiding edge IDs, component IDs, coordinates, geometry,
predictions, letters, or titles.

Across the three current real-data direct-global blockers:

- `13,4,3` recto seed `1`: direct-global p90 `0.230291`, multi-chart p90
  `0.005552`, six bridges scored, five singleton placements.
- `13,4,4` verso seed `0`: direct-global p90 `0.506859`, multi-chart p90
  `0.010780`, two bridges scored, one singleton placement.
- `14,4,4` verso seed `0`: direct-global p90 `0.300417`, multi-chart p90
  `0.039103`, three bridges scored, two two-node component placements.

## Next Action

Add package verifier checks for the two new generalization JSON/PNG artifacts,
then summarize the three-case scalar atlas evidence as the next progress-prize
candidate claim. A fourth held-out chunk is useful after packaging, but lower
leverage than making this result reproducible and guarded.
