# Source and result verification

The original frozen numerical evidence is maintained alongside immutable source files and archive fingerprints. This repository distributes the method, its original experimental runners, exact dataset-source identities, frozen evaluation summaries, final publication figures and source-to-input records.

## Checks performed for this release

- Original Python source identities: **60/60** SHA-256 matches. The Resource Cabinet BOSA and burst sources were copied directly from their terminal experiment archives.
- Supplementary plot inputs: **20/20** frozen SHA-256 matches.
- Resource Cabinet: the original SHARP -> CPPI-GREG -> RACS decision ledgers contain 1,080 aligned cases; 43 SHARP-unresolved, 21 CPPI-unresolved, zero operationally unresolved after RACS. The final decisions have 1,080/1,080 operational validity and 1,075/1,080 exact oracle matches. Those two denominators have different meanings.
- RACS input/output bytes: identical to the terminal closure archive; 21 consensus cases checked by `verify_racs_linkage.py`.
- Occupancy v2 and EEG Eye State: original pre-performance blind-audit file, conversion script, split counts and prepared stream SHA-256 fingerprints included.
- Fresh MHEALTH, Older Activity and Multivariate Gait: exact original Batch16 prepared stream hashes and counts verified against their frozen record.
- Additional V24, V13, Temporal Actionability and HI evidence archives: source/terminal identities recorded under `reproducibility/`.
- Main Figures 2–11 and supplementary S1–S8: all 18 rendered from the packaged plotting code and inputs, with identical 72-dpi raster output to the frozen PDFs.
- Three mathematical geometry/liability tests passed.

Run `python scripts/verify_release.py` for hash, metadata and frozen decision-chain checks. The check does not train any models.

## Experimental execution scope

Large training endpoints and cached probabilities are derived assets, not the algorithm source. Their exact original workstation paths, sizes and SHA-256s are indexed in `reproducibility/EXTERNAL_INPUT_HASHES.csv`. Dataset downloads are specified in `data/sources.json`, with input-specific transformations and hashes where frozen. Those external binary assets are not bundled in this source repository.

An independent clean-environment training-to-terminal replay of every backbone, dataset and seed has not been carried out as part of code packaging. Source integrity and the fixed published experimental records are verified separately from such a replay.

Earlier-numbered source modules retained by the terminal implementations are intentional dependencies, not superseded results. The original source algorithms have not been altered in this release.
