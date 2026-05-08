# PHercParis4 Multi-Chart Atlas Control Manifest

- timestamp UTC: 2026-05-08T00:09:29Z
- code path: `scripts/plot_thaumato_patch_graph_growth_qa.py`
- machine: current `g2-standard-16` L4 VM
- objective: add one bounded non-PHerc1667 specificity/control check for the
  scalar multi-chart atlas bridge-repair QA claim before doing any broader data
  sweep

## Rationale

The current package claim is supported by three bounded PHerc1667 direct-global
blockers. A useful next check is not another same-region PHerc1667 positive; it
is a small out-of-sample control where the detector found selected clusters but
the prior mesh-pruning report did not show strong reconstruction quality.

The prior PHercParis4 report
`reports/2026-05-06_third_region_mesh_prune.md` found that blur `11` produced
selected recto/verso clusters in most central chunks, but conservative pruning
left only about two triangle proxies per side. This makes PHercParis4
`3,4,4` recto seed `0` a good bounded control: if local-atlas growth does not
reach target or the multi-chart score does not produce a direct-global blocker
resolution, that improves specificity framing. If it unexpectedly reaches
target with the same scalar pattern, it becomes a higher-priority generalization
case.

## Data Plan

- sample: `PHercParis4`
- volume: `20260323153942-2.400um-0.2m-137keV-masked.zarr`
- array: `3`
- chunk: `3,4,4`
- side/seed: `recto` / `0`
- crop: `128`
- expected transient read: one public OME-Zarr chunk, about `2 MB`
- outputs: scalar JSON plus raster PNG QA only

No raw chunks, extracted volumes, point clouds, HDBSCAN labels, patch graphs,
meshes, flattened coordinates, endpoint IDs, component IDs, model weights,
predictions, ink, letters, titles, or text will be saved.

## Cost

Expected runtime is under 5 minutes on the current L4 VM. Expected incremental
VM cost is under USD `0.10`. No larger machine, public endpoint, paid service,
or cloud resource mutation is needed.

## Command

```bash
/usr/bin/time -f 'wall_seconds %e' .venv/bin/python scripts/plot_thaumato_patch_graph_growth_qa.py \
  --index-json reports/data_index_2026-05-06.json \
  --sample PHercParis4 \
  --volume 20260323153942-2.400um-0.2m-137keV-masked.zarr \
  --array 3 \
  --chunk 3,4,4 \
  --side recto \
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
  --json-out reports/thaumato_patch_graph_growth_qa_PHercParis4_3_4_4_recto_seed0_multi_chart_atlas_control75_2026-05-08.json \
  --png-out reports/thaumato_patch_graph_growth_qa_PHercParis4_3_4_4_recto_seed0_multi_chart_atlas_control75_2026-05-08.png
```

## Expected Decision

Treat this as a specificity/control result first. The most useful outcome is a
clear scalar failure reason: no target reach, no direct-global blocker, or
multi-chart bridge-aware p90 not under cap. A positive direct-global blocker
resolution would be stronger but should not be claimed until at least one more
PHercParis4 or non-PHerc1667 control is checked.

## Evaluation Result

The run completed in `5.27s` wall time and wrote a `1180x860` PNG. The selected
cluster had `11043` points, producing a 99-node patch graph with 132 edges. The
local-atlas control did not reach target:

| scalar | value |
| --- | ---: |
| base largest-component fraction | `0.040404` |
| base p90 edge distortion | `0.000239` |
| local-atlas reached target | `false` |
| local-atlas step count | `1` |
| local-atlas stop reason | `no_eligible_bridge` |
| local-atlas final fraction | `0.060606` |
| local-atlas final quality p90 | `0.089678` |
| local-atlas direct-global p90 | `0.090298` |
| bridge candidates | `111` |
| cap-compliant bridge candidates | `109` |
| multi-chart decision | `direct_global_already_cap_compliant` |
| multi-chart bridge p90 | `0.000168` |
| multi-chart bridge count | `1` |
| multi-chart internal p90 | `0.265047` |
| placed node fraction | `0.060606` |
| unplaced bridges | `0` |

Interpretation: this is a useful non-PHerc1667 specificity control. It does not
create a new direct-global blocker-resolution claim, because the local atlas
does not reach the `0.75` target and the direct-global p90 is already below the
`0.20` cap. The high internal multi-chart p90 reflects the weak PHercParis4
patch graph/mesh-pruning quality already seen in the third-region report; it
does not contradict the PHerc1667 blocker cases because this control is not a
target-reaching direct-global blocker.

## Outputs

- JSON:
  `reports/thaumato_patch_graph_growth_qa_PHercParis4_3_4_4_recto_seed0_multi_chart_atlas_control75_2026-05-08.json`
- JSON SHA256:
  `2088e3526c21df4db43adbac280b1426d135bcf117e74cc9fef02187bed81d0b`
- PNG:
  `reports/thaumato_patch_graph_growth_qa_PHercParis4_3_4_4_recto_seed0_multi_chart_atlas_control75_2026-05-08.png`
- PNG SHA256:
  `03a8c4c7e2517090a87d00a9a2e456274f70acf3bd8aff030a7a7ee114f0bf46`

## Next Action

Add this PHercParis4 control to the package manifest/verifier as a scalar
specificity guard, then rebuild the Thaumato bundle. Do not run a wider
PHercParis4 sweep unless the public submission needs another non-PHerc1667
control after publication review.
