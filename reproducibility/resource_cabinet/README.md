# Resource Cabinet: terminal RACS evidence

`racs/` contains the **two exact CSV inputs** used by the byte-locked `run_racs_closure.py`: `CPPI_GREG_SELECTED_ALL_SIX.csv` (the upstream CPPI-GREG decisions and residual estimates) and `BOSA_SELECTED_DEPLOYMENT_SEEDWISE.csv` (the frozen resource-ground-truth/threshold record). The original RACS output, rescue decisions, assessment and cross-domain weight table are alongside those inputs. Their checksums and containing nested ZIP hash are in `racs/PROVENANCE.json`.

`python scripts/verify_racs_linkage.py` checks all original input/output checksums and directly verifies the 1,080 decision ledger and 21 triggered cases, including the fixed 3/8 CPPI-GREG consensus weight. It uses only frozen CSVs; it does not train models or change the terminal record.

For a fresh execution of the original RACS source program (not necessary for source verification), stage the exact two CSVs in the script's execution directory because that original file reads them using `HERE`. Other upstream PPARC/SHARP/CPPI-GREG training-to-case-cache pipelines require the documented case-cache and configuration records, which are not all in the final manuscript or plotting archives.
