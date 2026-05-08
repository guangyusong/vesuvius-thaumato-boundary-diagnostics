# PHerc1667 Multi-Chart Method Decision

- timestamp UTC: 2026-05-08T01:08:17+00:00
- script: `scripts/summarize_pherc1667_multi_chart_method_decision.py`
- machine: current `g2-standard-16` L4 VM
- new data downloads: none
- GPU runtime: none; scalar-only summary generation
- cost impact: negligible CPU-only package bookkeeping
- command: `python3 scripts/summarize_pherc1667_multi_chart_method_decision.py`
- hypothesis: if short-route substitution is the right next method, saved blocker cases should expose at least one strict-cap-compliant target path; if not, package the multi-chart bridge representation instead
- visual checks: none; scalar-only JSON and Markdown table output
- saved data policy: scalar aggregate JSON only; no raw chunks, endpoints, path signatures, component identifiers, coordinates, meshes, predictions, letters, or titles

## Summary

- decision: `package_multi_chart_bridge_representation_defer_route_substitution`
- metadata write action: `none_scalar_only`
- multi-chart direct-global blocker positives: `3`
- multi-chart specificity controls: `2`
- route-substitution cases: `2`
- route-substitution target paths: `139`
- route-substitution cap paths: `0`
- shared positive/route-negative cases: `2`

The package decision is to emphasize multi-chart bridge representation, while keeping route substitution as bounded failure analysis. The adapter contract has 3 PHerc1667 direct-global blocker positives plus 2 specificity controls with `none_scalar_only` metadata action. The route audit finds `139` target-connecting short routes across two blocker cases and zero cap-compliant target paths under the strict `0.20` p90 cap.

## Shared Cases

| case | direct p90 | multi-chart p90 | multi-chart cap | route target paths | route cap paths | route best p90 | route p90 gap |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| pherc1667_13_4_3_recto_seed1 | 0.230291 | 0.005552 | True | 91 | 0 | 0.232185 | 0.032185 |
| pherc1667_14_4_4_verso_seed0 | 0.300417 | 0.039103 | True | 48 | 0 | 0.301094 | 0.101094 |

## Multi-Chart Positive Cases

| case | direct p90 | multi-chart p90 | bridges | unplaced | action |
| --- | ---: | ---: | ---: | ---: | --- |
| pherc1667_13_4_3_recto_seed1 | 0.230291 | 0.005552 | 6 | 0 | quality_gate_pass_review_candidate |
| pherc1667_13_4_4_verso_seed0 | 0.506859 | 0.010780 | 2 | 0 | quality_gate_pass_review_candidate |
| pherc1667_14_4_4_verso_seed0 | 0.300417 | 0.039103 | 3 | 0 | quality_gate_pass_review_candidate |

## Route-Substitution Audit

| case | target paths | cap paths | best p90 | p90 gap | classification |
| --- | ---: | ---: | ---: | ---: | --- |
| pherc1667_13_4_3_recto_seed1 | 91 | 0 | 0.232185 | 0.032185 | route_substitution_negative |
| pherc1667_14_4_4_verso_seed0 | 48 | 0 | 0.301094 | 0.101094 | route_substitution_negative |

## Interpretation

This is the reviewer-facing method decision that ties the positive and negative evidence together. Multi-chart bridge representation resolves the saved direct-global blocker cases at the scalar QA level, while route substitution provides zero cap-compliant alternatives on the two overlapping blocker cases. The package should therefore present route substitution as a rejected branch, not as the main contribution.

This is not an unwrap, text, ink, letter, title, `overlapping.json` writer, or VC3D metadata mutation claim.

## Ranked Next Steps

- Package the multi-chart bridge representation as the reviewer-facing method decision.
- Keep route substitution as bounded failure analysis unless a future representation lowers the strict p90 gap.
- If continuing technically, add one current-public-repo adapter dry-run or one additional specificity control before proposing code changes.

## Storage Policy

Read saved scalar JSON only; no raw chunks, endpoints, path signatures, patch identifiers, component identifiers, coordinates, meshes, predictions, ink, letters, or titles.
