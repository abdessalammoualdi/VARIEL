# VARIEL source-to-input map

The JSON manifest `INPUT_LINKAGE.json` maps frozen runners to their named input files and archive sources. `PRESENT_FROZEN_EXACT` means the exact recorded bytes are bundled; `UPSTREAM_ARCHIVE_PRESENT` identifies original prior terminal summaries; `NOT_BUNDLED` means original trained predictions or per-case model outputs are not distributed.

The terminal RACS input gap is closed. Both CSVs, the final result, and the original RACS source originate in the same frozen archive; `python scripts/verify_racs_linkage.py` verifies 1,080 cases and 21 consensus repairs without retraining.

For V24, V13 and the high-information audit, the terminal code and compact upstream ZIPs are present. Their trained endpoints, individual case caches, or precise input preparation cannot be inferred from the available summary ZIPs. The code's original `DTRAR_PROJECT` layout is preserved; moving its inputs into different directories without an explicit, verified mapping would break execution.

**Dataset source pages are not always the prepared experimental streams.** The catalog identifies the exact file hashes where known; the three locked fresh confirmations have now been linked to their exact original prepared CSV hashes and original Batch13/16 preparation sources; the original Occupancy v2 and EEG preparation audit now supplies the exact prepared CSV fingerprints and split counts. Dataset-specific conversions must use their original frozen source procedures rather than substituting a similarly named public download.
