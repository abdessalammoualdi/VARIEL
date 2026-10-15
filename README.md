# VARIEL

**Resource-aware, safeguarded maintenance of deployed predictive systems**

VARIEL is one maintenance architecture built around three decision subsystems. The **Maintenance Action Cabinet** chooses HOLD, sparse repair, evidence escalation, FULL, or FULL+REPAIR. The **Resource Cabinet** decides how much information is worth acquiring before a decision. The **Safeguard Cabinet** constrains intervention liability, controls high-information decisions, and preserves specified operating behavior. Probability-Action Geometry (V-PAG), residual refinement, Temporal Actionability, and causal label reuse connect the decisions across deployment windows.

## Project structure

| Directory | Contents |
| --- | --- |
| `variel/frozen/` | Original, SHA-256-locked algorithms, grouped into sparse, resource, high-information, preservation, multiclass and temporal components |
| `variel/infrastructure/` | Original data loading, preprocessing, model training, and configurations |
| `scripts/` | Run selection, dataset acquisition, training entry point, and source verification |
| `data/` | The 10 principal and 3 locked sparse-confirmation dataset records; original URLs and file checksums where established |
| `expected/` | Frozen result summaries |
| `figures/` | Figure 1, main Figures 2–11, supplementary S1–S8, Matplotlib source and numerical plotting inputs |
| `tests/` | Checks for key probability-action mathematical identities |
| `reproducibility/` | Source hash ledger and experiment input contracts |

## Install

Python 3.10+ recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

FT-Transformer experiments also need a compatible PyTorch installation and the `rtdl-revisiting-models==0.0.2` package used by the original model-training code. Hardware-specific PyTorch/CUDA installation follows the upstream PyTorch documentation.

## Verify and inspect

```bash
python scripts/verify.py
python scripts/verify_input_linkage.py
python scripts/verify_racs_linkage.py
python scripts/verify_terminal_chain.py
python scripts/verify_unseen_data.py
python -m unittest discover -s tests -v
python scripts/data.py list
python scripts/run.py --list
```

The verification script checks the unmodified original source and figure-input SHA-256 values. The three unit tests check probability geometry and exact prospective liability. These checks do not execute training.

For one source-and-input integrity command, use `python scripts/verify_release.py` (no experiment fitting).

## Datasets

See **[data/SOURCE_PROVENANCE.md](data/SOURCE_PROVENANCE.md)** for each original download location, expected raw/prepared form, and processing boundary. The original downloader for Bank, Electricity and Gas verifies all extracted bytes:

```bash
python scripts/data.py fetch-core
```

The original Rialto and Spam files are available with checksum verification:

```bash
python scripts/data.py fetch rialto
python scripts/data.py fetch spam
```

The full protocol is dataset-specific; use the original locked conversion/split where the public download is not the final experiment input. The repository does not redistribute the 13 datasets.

## Source and terminal inputs

The terminal Resource Cabinet source, its original upstream SHARP and CPPI-GREG tables, RACS inputs and 1,080-decision output are bundled under `reproducibility/resource_cabinet/racs/`. Validate their linkage without model training:

```bash
python scripts/verify_racs_linkage.py
python scripts/stage_racs_inputs.py
```

The machine-readable [input manifest](reproducibility/INPUT_LINKAGE.json) traces component runners to the original files and identifies dataset-specific preparation and endpoint inputs that are not part of the distributed source. The source-lock checks and results summaries are distinct from an entire clean-environment experiment replay.

## Training and maintenance

The original model-training infrastructure accepts Bank, Electricity and Gas with XGBoost, CatBoost, and FT-Transformer. To inspect a training command without executing it:

```bash
python scripts/train.py --datasets bank --models xgboost --seeds 0 --profile smoke
```

Add `--execute` to run that source-training command after setting up the original data and environment. Other datasets and regimes retain their own terminal-source runners:

```bash
python scripts/run.py --list
python scripts/run.py --component high_information
```

These commands identify the preserved source implementation. Original runner scripts expect their specific input directories, prepared data and upstream outputs, as described in `reproducibility/EXECUTION_SCOPE.md`. The frozen implementation has not been refactored into a new algorithm or replaced with illustrative pseudocode.

## Figures and reported evidence

```bash
python figures/main/generate_variel_python_final.py
python figures/supplementary/plot_all.py
```

The figures are drawn from the supplied final numeric source tables. They are separate from model fitting. Source-to-figure integrity and core identities can be checked without re-running the original study.

The principal evaluation includes ten deployment streams. MHEALTH, Older Activity, and Multivariate Gait form a **separate locked sparse-confirmation experiment**. The `expected/` summaries preserve the corresponding experiment-level scope rather than treating every dataset as part of every comparison.

## Frozen input provenance

Run `python scripts/verify_release.py` to check the exact Resource Cabinet and fresh-stream input identities and 1,080 recorded decisions without fitting models. Use `reproducibility/INPUT_LINKAGE.json` to locate frozen source inputs and see which prediction/endpoint inputs still require original preparation.

## Baselines and attribution

The maintenance architecture is VARIEL; MCGrad, LTT, Periodic-5%, rolling FULL, UPF and Severity-Aware are external comparison methods or regimes. Original comparator authorship and software licenses remain distinct. Refer to the accompanying manuscript and its citations for comparator definitions, statistical analyses and experimental design.

For the exact software-source and input audit, see `reproducibility/RELEASE_SOURCE_INPUT_AUDIT.md`, `reproducibility/INPUT_LINKAGE.json`, and `reproducibility/SOURCE_LOCK.csv`.

## Source validation

```bash
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

This checks the frozen source-to-result records, prepared dataset identities and final Resource Cabinet decision sequence without model training. The full verification scope is in [RELEASE_CHECKS.md](reproducibility/RELEASE_CHECKS.md).

## Repository access
