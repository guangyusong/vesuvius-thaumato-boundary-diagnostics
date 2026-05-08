# Multi-Chart Atlas Real-Data Validation Manifest

- timestamp UTC: 2026-05-07T23:35:23Z
- code path: `scripts/plot_thaumato_patch_graph_growth_qa.py`
- hypothesis: the new scalar
  `local_atlas_sequence.multi_chart_atlas_reconciliation` field can score a
  real PHerc1667 local-atlas result with more than one accepted bridge. The
  chosen case is the previously saved ordinary local-atlas `13,4,3` recto seed
  `1` success, which historically reached target through six local-atlas
  bridges.
- data plan: rerun one public PHerc1667 OME-Zarr chunk, array `3`, chunk
  `13,4,3`; fetch transiently as needed.
- expected runtime: under 5 minutes on the current `g2-standard-16` L4 VM.
- expected incremental VM cost: under USD `0.10`.
- outputs: scalar JSON plus raster PNG QA image only; no raw chunks, extracted
  volumes, point clouds, meshes, predictions, text, ink, letters, titles, model
  weights, endpoint IDs, component IDs, edge IDs, or coordinate dumps.

## Planned Command

```bash
/usr/bin/time -f 'wall_seconds %e' .venv/bin/python scripts/plot_thaumato_patch_graph_growth_qa.py \
  --index-json reports/data_index_2026-05-06.json \
  --sample PHerc1667 \
  --volume 20231107190228-3.240um-88keV-masked.zarr \
  --array 3 \
  --chunk 13,4,3 \
  --side recto \
  --crop-size 128 \
  --blur-size 11 \
  --seed 1 \
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
  --growth-target 0.90 \
  --distortion-growth-max-p90 0.20 \
  --local-atlas-bridge-min-normal-agreement 0.50 \
  --local-atlas-bridge-max-offset-ratio 0.20 \
  --candidate-diagnostic-limit 24 \
  --json-out reports/thaumato_patch_graph_growth_qa_PHerc1667_13_4_3_recto_seed1_multi_chart_atlas_realdata90_2026-05-07.json \
  --png-out reports/thaumato_patch_graph_growth_qa_PHerc1667_13_4_3_recto_seed1_multi_chart_atlas_realdata90_2026-05-07.png
```

## Evaluation Result

| case | wall | reached | steps | final fraction | final local quality | direct-global p90 | multi-chart p90 | placed components | placed node fraction | scored bridges | singleton placements | unplaced bridges | missing-frame bridges | too-small endpoints | disconnected bridges | decision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PHerc1667 `13,4,3` recto seed `1` | `232.73s` | true | `6` | `0.902703` | `0.127353` | `0.230291` | `0.005552` | `7` | `0.902703` | `6` | `5` | `0` | `0` | `0` | `0` | `multi_chart_atlas_resolves_direct_global_blocker` |

The validation rerun shows that singleton endpoint placement closes the previous
partial-coverage gap. The ordinary local-atlas case still reaches target through
six accepted bridges, and all six bridges are now represented in the scalar
multi-chart atlas score. Five endpoints are original singleton components that
inherit the neighboring chart frame only for their bridge endpoint coordinate;
no edge IDs, component IDs, point clouds, coordinates, predictions, letters, or
titles are stored. The direct-global p90 remains above the `0.20` cap at
`0.230291`, while the bridge-aware multi-chart p90 is `0.005552`, so this case is
now classified as a resolved direct-global blocker for this scalar QA metric.

## Next Action

Test the singleton endpoint placement on at least two more bounded PHerc1667 or
PHerc0332 chunks before using the multi-chart atlas score as a package-level
claim. Prioritize cases where the ordinary local-atlas path reaches target but
direct-global PCA exceeds the p90 cap, because those are the most relevant
progress-prize diagnostics.
