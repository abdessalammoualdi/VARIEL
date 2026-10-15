# Source-to-input audit — 10 October 2026

This report separates verified terminal inputs from prepared data and endpoint files not distributed in the compact source repository.

## Resolved from existing evidence

**Resource Cabinet — terminal RACS.** Recovered `CPPI_GREG_SELECTED_ALL_SIX.csv`, `BOSA_SELECTED_DEPLOYMENT_SEEDWISE.csv`, `RACS_SELECTED_ALL_SIX.csv`, the rescue ledger and final assessment from the identical `DTRAR_ASPIRE_RACS_TERMINAL_CLOSURE.zip` that also contains the original `run_racs_closure.py`. The source byte hash matches the locked code. The static ledger checker confirmed 1,080 aligned decisions, 21 RACS-triggered rows, all 1,080 operationally valid, and 1,075 exact oracle actions. Fixed CPPI-GREG weight: 0.375. Source and CSV hashes are in `resource_cabinet/racs/PROVENANCE.json`.

**Three locked external sparse confirmations.** Found original `LIABILITY_FRESH_LOCK.json` and three prepared CSV files in `DTRAR_LiabilityFreshAudit_Batch16_Review.zip`, nested in the authoritative evidence package. All three prepared CSVs were SHA-256 checked against that original lock. The original Batch13 structural scripts, Batch16 lock script and Batch17 evaluation runner are preserved under `variel/frozen/sparse/`, with immutable source hashes. The three exact prepared stream SHA-256 values are in `fresh_confirmation/SOURCE_TO_PREPARED_DATA.json` and `data/sources.json`. Full datasets are excluded from the public source bundle.

**Preservation V24 and multiclass V13.** The original V21, V10 and Compass-SR predecessor ZIPs have been added under `terminal_archives/`, with their SHA-256 hashes in `INPUT_LINKAGE.json`. The terminal source and earlier-numbered support modules are included unchanged. Their large per-seed endpoint and job caches are *not* in this compact repository.

**Temporal Actionability.** The authoritative terminal numeric archive is now included; the candidate Python runner's exact originating run/source record remains unconfirmed because that numeric ZIP does not contain the script or its hash.

## Unresolved source/input boundaries

1. **Occupancy Detection v2 and EEG Eye State:** the original audit archive, the exact prepared-file SHA-256 values, preparation script and pre-performance 50/15/35 split protocol are included under `unseen_prepared_audit/`. The larger prepared CSV bytes were found on the research workstation but were excluded from the one-shot ZIP by size limits; no different public dataset will be silently substituted.
2. **Upstream Resource Cabinet:** recovered original PPARC/SHARP and CPPI-GREG terminal reviews, the final SHARP-selected and unresolved tables, original CPPI case results, BOSA selected table and terminal RACS records. The 32 large tail-case binaries and some endpoint-prediction caches are not distributed as GitHub code; their original paths and SHA-256s are recorded in `EXTERNAL_INPUT_HASHES.csv`.
3. **V24 preservation:** original V21 is available, but terminal endpoint/candidate caches (`artifacts/variel_cross_backbone_all_final/endpoint_cache`, specialist candidate cache, V24 job inputs) were excluded from the source-only archive.
4. **V13 multiclass:** exact V10 and Compass inputs are available, but the V12 XGBoost static cache required by `run_xgboost_ltt_mc_v13.py` was not included.
5. **HI 80/80:** the original terminal aggregate is present; the trained rolling endpoints, probability arrays and complete per-seed inputs are not packaged. The original runner refers to their original `artifacts/` locations.
6. **Original dataset distribution:** exact public URLs and raw/prepared hashes are included where established. Public downloads for some principal streams do not automatically reconstruct the prepared experimental inputs. No alternative dataset will be silently substituted.

Large model/endpoint binaries are kept outside the distributed source repository; their original paths and reported SHA-256s are indexed. The terminal input tables, original source runners and structural audit identities described above are linked. No training or evaluation runs were carried out during this source-linkage audit. The reference files and source checkers can be inspected without any model computation.
