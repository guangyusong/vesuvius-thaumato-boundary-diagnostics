# VC3D Multi-Chart Adapter Contract

- timestamp UTC: 2026-05-08T01:07:49+00:00
- public context: `reports/2026-05-07_public_vc3d_overlap_context_refresh.md`
- script: `scripts/summarize_vc3d_multi_chart_adapter_contract.py`
- machine: current `g2-standard-16` L4 VM
- new data downloads: none
- GPU runtime: none
- cost impact: CPU-only scalar summary generation, negligible incremental cost
- command: `python3 scripts/summarize_vc3d_multi_chart_adapter_contract.py`
- hypothesis: the multi-chart atlas result is useful to VC3D as a read-only bridge-quality adapter after SurfacePatchIndex candidate discovery, not as an overlap metadata writer
- visual checks: none; scalar-only JSON and Markdown table output
- saved data policy: scalar aggregate JSON only; no raw chunks, endpoints, geometry payloads, meshes, predictions, letters, or titles

## Summary

- cases: `5`
- positive blockers: `3`
- specificity controls: `2`
- metadata write action: `none_scalar_only`
- decision: `package_read_only_multi_chart_bridge_quality_adapter`

## Adapter Contract

- candidate source: `SurfacePatchIndex_or_existing_overlap_writer`
- adapter mode: read-only scalar bridge-quality scoring
- overlap metadata action: `none_scalar_only`
- output fields: `bridge_quality_action`, `review_bucket`, `overlap_metadata_action`, `non_claims`

- Current VC3D main uses SurfacePatchIndex-style spatial queries for candidate discovery.
- Current overlap writers own overlap metadata writes and update both source and target sides.
- This adapter is read-only and applies a scalar multi-chart bridge-quality gate after candidate discovery.

## Cases

| case | role | target | reached | direct p90 | direct cap | multi bridge-aware p90 | multi cap | bridges | placed | unplaced | action |
| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- |
| pherc1667_13_4_3_recto_seed1 | direct_global_blocker_positive | 0.900000 | True | 0.230291 | False | 0.005552 | True | 6 | 0.902703 | 0 | quality_gate_pass_review_candidate |
| pherc1667_13_4_4_verso_seed0 | direct_global_blocker_positive | 0.750000 | True | 0.506859 | False | 0.010780 | True | 2 | 0.751244 | 0 | quality_gate_pass_review_candidate |
| pherc1667_14_4_4_verso_seed0 | direct_global_blocker_positive | 0.750000 | True | 0.300417 | False | 0.039103 | True | 3 | 0.758865 | 0 | quality_gate_pass_review_candidate |
| phercparis4_3_4_4_recto_seed0 | specificity_control | 0.750000 | False | 0.090298 | True | 0.265047 | False | 1 | 0.060606 | 0 | no_positive_action_control |
| phercparis4_3_4_4_verso_seed0 | specificity_control | 0.750000 | False | n/a | False | n/a | False | 0 | n/a | 0 | no_positive_action_control |

## Integration Boundary

This summary is a read-only adapter contract. `SurfacePatchIndex` or the current overlap writer should remain responsible for candidate discovery; this package only scores saved scalar bridge-quality evidence after a candidate exists.

It is not an `overlapping.json` writer, does not mutate VC3D metadata, and does not claim to solve an active VC3D issue by itself.

This is not an unwrap, text, letter, title, or ink claim.

## Ranked Next Steps

- Package the read-only adapter contract with the Thaumato progress-prize candidate.
- If public repository credentials become available, submit this as a scalar QA artifact before proposing VC3D code changes.
- Only write a VC3D patch after a current-main reproduction shows a concrete candidate-discovery or metadata-writing defect.

## Storage Policy

Read saved scalar JSON only; no raw chunks, point clouds, geometry payloads, patch identifiers, bridge endpoints, overlap metadata, meshes, predictions, letters, or titles.
