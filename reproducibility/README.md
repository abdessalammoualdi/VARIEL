# Reproducibility records

- `SOURCE_LOCK.csv`: exact hashes and archival origin of the original method source files.
- `resource_cabinet/`: terminal SHARP, CPPI-GREG, BOSA and RACS tables and source-matched review archives.
- `unseen_prepared_audit/`: pre-performance Occupancy v2 and EEG structural audit, conversion source, exact prepared input SHA-256.
- `fresh_confirmation/`: 3 prepared frozen dataset hashes and source split counts.
- `temporal/`, `high_information/`, `preservation/`, `multiclass/`: final subsystem evidence.
- `EXTERNAL_INPUT_HASHES.csv`: hashes and original local paths of large model endpoints and prepared files, not included as public source content.
- `UPSTREAM_SHA256.json`: immutable byte provenance for distributed terminal evidence.

The data acquisition routes and stream-specific preprocessing are in `data/` and the original immutable experiment scripts.
