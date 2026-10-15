# VARIEL dataset provenance

The original links below are the public sources recorded in the frozen code or the dataset publishers' own repositories. An **original public dataset** and a **prepared experimental stream** are different objects. The frozen algorithms are unchanged in this release.

| Dataset | Original source | Exact research input |
| --- | --- | --- |
| Bank | [UCI 222](https://archive.ics.uci.edu/dataset/222/bank+marketing), original Bank ZIP | `bank-full.csv`, SHA-256 pinned in `raw_manifest.json` |
| Electricity | [OpenML 151](https://www.openml.org/d/151), download ID 2419 | `electricity-normalized.arff`, SHA-256 pinned in `raw_manifest.json` |
| Gas | [UCI 224](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset) | `batch1.dat`–`batch10.dat`, all 10 hashes in `raw_manifest.json` |
| Airlines | [scikit-multiflow streaming-datasets](https://github.com/scikit-multiflow/streaming-datasets) | `airlines.csv`, 539,383 rows, fixed column schema in original HI runner |
| Occupancy Detection v2 | [UCI 357](https://archive.ics.uci.edu/dataset/357/occupancy+detection) | Project-prepared v2 CSV; exact original blind-audit lock included under `reproducibility/unseen_prepared_audit/` |
| EEG Eye State | [UCI 264](https://archive.ics.uci.edu/dataset/264/eeg+eye+state) | Project-prepared locked CSV; exact original blind-audit lock included under `reproducibility/unseen_prepared_audit/` |
| Rialto | [original drift-stream repository](https://github.com/ogozuacik/concept-drift-datasets-scikit-multiflow) | `real-world/rialto.csv`, fixed SHA-256 in `sources.json` |
| Spam | [original drift-stream repository](https://github.com/ogozuacik/concept-drift-datasets-scikit-multiflow) | `real-world/spam.csv`, fixed SHA-256. This is **not** UCI Spambase |
| INSECTS | [original LABIC INSECTS source route](https://sites.labic.icmc.usp.br/vsouza/repository/creme/INSECTS-abrupt_balanced_norm.arff) | Prepared `INSECTS.csv`, SHA-256 pinned in the frozen HI source; original ARFF retrieval/transform still needs confirmation |
| BLE RSSI | [UCI 586](https://archive.ics.uci.edu/dataset/586/ble+rssi+dataset+for+indoor+localization) | `raw_rssi.csv`, not `filtered_rssi.csv`, SHA-256 pinned in frozen source |
| MHEALTH | [UCI 319](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) | Prepared 250,000-row stream, SHA-256 `041e9079373f...`; original Batch16/B17 lock and preparation code included |
| Older Activity | [UCI 427](https://archive.ics.uci.edu/dataset/427/activity+recognition+with+healthy+older+people+using+a+batteryless+wearable+sensor) | Prepared 75,128-row stream, SHA-256 `86e470a9d38d...`; Batch13 conversion and Batch16 frozen lock included |
| Multivariate Gait | [UCI 760](https://archive.ics.uci.edu/dataset/760/multivariate+gait+data) | Prepared 181,800-row stream, SHA-256 `5c6a50090384...`; Batch13 conversion and Batch16 frozen lock included; final result is HOLD |

## Commands

`python scripts/data.py list` prints all 13 public source locations. `python scripts/data.py fetch-core` uses the original downloader and verifies Bank, Electricity and Gas file fingerprints. `python scripts/data.py fetch rialto` and `python scripts/data.py fetch spam` fetch exact original drift-stream CSVs, with SHA-256 validation. Other sources have the official landing pages, recorded expected names and frozen per-stream protocol under `variel/frozen/`. The code-only handoff does not contain every prepared stream manifest.

**Do not replace:** Rialto or Spam with a similarly named benchmark; raw BLE RSSI with the filtered variant; the INSECTS prepared numeric CSV with an unverified ARFF conversion; Occupancy v2 or EEG locked CSV with untracked different concatenations; or Batch17 converted fresh streams with arbitrary ordering. Those changes alter the experiment.

The three fresh-confirmation prepared CSVs were located in the original nested Batch16 archive and all three hashes were verified. The full bytes are not redistributed here; the exact filenames, code, stream ordering, split counts, and lock records are in `reproducibility/fresh_confirmation/`.
