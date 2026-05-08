# PHercParis4 3,4,4 Verso Multi-Chart Control Manifest

- timestamp UTC: 2026-05-08T01:06:11Z
- code path: `scripts/plot_thaumato_patch_graph_growth_qa.py`
- machine: current `g2-standard-16` L4 VM
- objective: add one bounded PHercParis4 specificity/control check for the
  scalar multi-chart bridge-representation package decision

## Rationale

The package now has three PHerc1667 direct-global blocker positives, one
PHercParis4 recto control, and two route-substitution negatives. A second
non-PHerc1667 control is the next smallest useful check because it tests whether
the PHercParis4 result is one-sided or stable across recto/verso selected
clusters from the same bounded chunk.

PHercParis4 `3,4,4` verso seed `0` is a conservative choice: the prior third
region mesh-prune report showed selected `blur=11` clusters on both sides, but
weak post-pruning mesh proxies. The expected useful outcome is either no target
reach or an already-cap-compliant direct-global result, which strengthens the
specificity boundary without adding a same-region PHerc1667 positive.

## Data Plan

- sample: `PHercParis4`
- volume: `20260323153942-2.400um-0.2m-137keV-masked.zarr`
- array: `3`
- chunk: `3,4,4`
- side/seed: `verso` / `0`
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
  --json-out reports/thaumato_patch_graph_growth_qa_PHercParis4_3_4_4_verso_seed0_multi_chart_atlas_control75_2026-05-08.json \
  --png-out reports/thaumato_patch_graph_growth_qa_PHercParis4_3_4_4_verso_seed0_multi_chart_atlas_control75_2026-05-08.png
```

## Expected Decision

Treat this as a specificity/control result. If it does not create a
target-reaching direct-global blocker, add it as a second non-PHerc1667 control
to the adapter contract and method-decision summary. If it unexpectedly becomes
a target-reaching blocker-like result, keep it separate as review-needed
generalization evidence before changing the package claim.

## Evaluation Result

The run completed in `5.43s` wall time and wrote a `1180x860` PNG. The selected
cluster produced a 120-node patch graph with 243 edges. The local-atlas control
accepted no bridge steps and did not reach target:

| scalar | value |
| --- | ---: |
| base largest-component fraction | `0.058333` |
| base p90 edge distortion | `0.002423` |
| local-atlas reached target | `false` |
| local-atlas step count | `0` |
| local-atlas stop reason | `no_eligible_bridge` |
| local-atlas final fraction | `0.058333` |
| local-atlas final quality p90 | `0.015020` |
| local-atlas direct-global p90 | `0.002423` |
| multi-chart reconciliation | `no_local_atlas_steps` |

Interpretation: this is a clean second PHercParis4 specificity control. It does
not create a direct-global blocker-resolution claim because no local-atlas
bridge was accepted and the direct graph is already far below the strict `0.20`
p90 cap. The result strengthens the package boundary: PHerc1667 blocker cases
are target-reaching multi-chart positives, while this PHercParis4 verso control
is a no-eligible-bridge negative.

## Outputs

- JSON:
  `reports/thaumato_patch_graph_growth_qa_PHercParis4_3_4_4_verso_seed0_multi_chart_atlas_control75_2026-05-08.json`
- JSON SHA256:
  `ace902f1869f5638d4c5d3e901adf1f42b72fb7353041a86e5b5a4a243af2ce3`
- PNG:
  `reports/thaumato_patch_graph_growth_qa_PHercParis4_3_4_4_verso_seed0_multi_chart_atlas_control75_2026-05-08.png`
- PNG SHA256:
  `029af8d0d501364a259a2436cf001c98cdb45ba6b11bca8d18e0cb0d81145408`

## Next Action

Add this control to the VC3D adapter-contract and multi-chart method-decision
summaries, then rebuild the preserved local package bundle.
