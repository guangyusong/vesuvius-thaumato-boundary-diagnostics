# PHerc1667 Two-Chart Boundary Source Validation

- timestamp UTC: 2026-05-07T20:20:00Z
- method: `pherc1667_two_chart_boundary_source_localatlas_validation`
- decision: `package_normalrisk_two_chart_boundary_source`
- data read: six bounded public PHerc1667 OME-Zarr chunks, array `3`, fetched transiently
- approximate wall time: `1278s`
- approximate incremental cost: USD `0.41`
- saved data policy: scalar JSON plus raster PNG QA images only; no raw chunks, point clouds, edge IDs, endpoints, component IDs, coordinates, meshes, predictions, letters, titles, ink detections, or model weights

## Summary

The executable local-atlas two-chart boundary source reproduces the
diagnostics-only split only when paired with the stricter normal-risk local
gate. In that configuration, both boundary-positive PHerc1667 cases reach the
target through exactly one `two_chart_boundary_gate` step, while the
boundary-negative/control accepts no local-atlas step.

The permissive local-atlas fallback is an important failure analysis: the
negative/control reaches target through ordinary `local_atlas_bridge_gate`
steps even though the new boundary source accepts zero steps. Package the
normal-risk configuration, not the permissive fallback.

## Cases

| chunk | side | seed | role | normal risk | reached | steps | boundary-source steps | final fraction | final quality p90 | first source | boundary quality | rigid p90 | placement | global p90 | wall |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| `13,4,4` | verso | 0 | boundary-positive | false | True | 1 | 1 | `0.756219` | `0.181851` | `two_chart_boundary_gate` | `0.183677` | `0.049563` | `rigid_cap_success` | `0.478186` | `~217s` |
| `13,4,3` | recto | 1 | boundary-positive | false | True | 1 | 1 | `0.908108` | `0.190131` | `two_chart_boundary_gate` | `0.190131` | `0.141786` | `rigid_cap_success` | `0.235783` | `207.43s` |
| `14,4,4` | verso | 0 | boundary-negative/control | false | True | 3 | 0 | `0.758865` | `0.161402` | `local_atlas_bridge_gate` | `0.380893` | `0.032322` | `reflection_like_transform_failure` | `0.300417` | `197.85s` |
| `13,4,4` | verso | 0 | boundary-positive | true | True | 1 | 1 | `0.756219` | `0.181851` | `two_chart_boundary_gate` | `0.183677` | `0.049563` | `rigid_cap_success` | `0.478186` | `258.23s` |
| `13,4,3` | recto | 1 | boundary-positive | true | True | 1 | 1 | `0.908108` | `0.190131` | `two_chart_boundary_gate` | `0.190131` | `0.141786` | `rigid_cap_success` | `0.235783` | `206.76s` |
| `14,4,4` | verso | 0 | boundary-negative/control | true | False | 0 | 0 | `0.425532` | `0.000802` | none | n/a | n/a | n/a | `0.000631` | `190.63s` |

## Interpretation

The two-chart boundary source is now executable, not only diagnostic. The useful
configuration is:

```text
--local-atlas-use-transition-normal-risk
--local-atlas-use-two-chart-boundary-gate
--base-cap-boundary-prototype
--base-cap-boundary-require-two-chart-gate
```

The result remains a bridge-repair and QA method result. It is not an unwrap,
text, letter, title, or ink claim. The global p90 remains above the strict cap
for the two positive target-reaching cases, so the method should be presented as
local chart-boundary recovery evidence plus explicit global-reconciliation
caveat.

## Ranked Next Steps

1. Update the package verifier to require the normal-risk boundary-source split
   and the permissive fallback caveat.
2. Package this as the primary bridge-repair method improvement for a monthly
   progress-prize submission candidate.
3. Before any broader GPU screen, decide whether the method needs a
   boundary-source-only mode or whether normal-risk fallback is sufficient.
