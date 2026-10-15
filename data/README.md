# Datasets and preparation

VARIEL uses ten principal dataset streams and three separate locked sparse-confirmation datasets. See `sources.json` for the exact dataset versions and public links.

The original Bank/Electricity/Gas downloader and its SHA-256 manifest are preserved under `variel/infrastructure/`. It can be invoked from there; downloaded data is saved locally, never committed. The original external-reference source provides pinned checksums for Rialto, Spam and BLE RSSI. A similarly named dataset is not automatically an equivalent replacement.

Use `python scripts/data.py list` to inspect the 13 sources. Use `python scripts/data.py fetch DATASET` for datasets with an exact direct-source URL and an expected SHA-256; it will refuse incorrect bytes. `python scripts/data.py fetch-core` invokes the original Bank/Electricity/Gas acquisition script. Other public dataset landing pages can be opened from the catalog and need source-specific preprocessing verification.

Data must retain the stream ordering, source/target boundary, model-specific training folds, canary/bulk separation, and seed schedule from each frozen script. Do not randomly shuffle or normalize the full stream before splitting. The official raw data pages alone are not sufficient to reproduce all published endpoints.
