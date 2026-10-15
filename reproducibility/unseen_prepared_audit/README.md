# Occupancy Detection v2 and EEG Eye State

This folder contains the original *pre-performance* dataset audit, prepared-file SHA-256 identities and conversion script. The original source records identify the UCI dataset versions, row order and 50/15/35 contiguous split. Occupancy v2 removes the two rows with invalid targets before the split; EEG Eye State retains its UCI chronological row order.

The prepared CSV bytes are intentionally excluded from the code release. When the published downloader reconstructs each prepared stream, it must match the exact SHA-256 recorded in `PREPARED_FILES.json` before that stream can be used to assess maintenance performance. `python scripts/verify_unseen_data.py` checks the existing identities and optionally checks locally present prepared CSVs.
