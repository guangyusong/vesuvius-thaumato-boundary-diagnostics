# Vesuvius Thaumato Boundary Diagnostics

Small research repo for one surface-unwrapping issue: local surface bridges can
look worse than they are when every component is forced into one global PCA
chart. The code here checks that behavior with a bridge-aware multi-chart atlas
score and keeps a few representative outputs.

## Result Snapshot

The retained PHerc1667 checks compare direct-global PCA p90 distortion against a
bridge-aware multi-chart atlas score:

| case | direct-global p90 | multi-chart p90 | accepted bridges |
| --- | ---: | ---: | ---: |
| `13,4,3` recto seed `1` | `0.230291` | `0.005552` | `6` |
| `13,4,4` verso seed `0` | `0.506859` | `0.010780` | `2` |
| `14,4,4` verso seed `0` | `0.300417` | `0.039103` | `3` |

Two PHercParis4 controls are included for comparison.

## What Is Included

- `scripts/plot_thaumato_patch_graph_growth_qa.py`: bounded patch-graph QA and
  multi-chart atlas scoring prototype.
- `scripts/summarize_*`: small report builders for the VC3D adapter and method
  decision.
- `reports/`: curated manifests, JSON metrics, and PNG QA images for the main
  positive cases and controls.
- `examples/vc_merge_tifxyz_summary_minimal.json`: tiny synthetic input for the
  adapter summary code.
- `tests/`: focused unit tests for the retained scripts.

## Reproduce The Checks

Install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-experiments.txt
```

Run tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Run the lightweight public package verifier:

```bash
python3 scripts/verify_public_package.py
```

The full real-data reruns need public Vesuvius OME-Zarr access. The checked-in
files are just the code, small derived reports, and QA images.
