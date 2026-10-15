# Resource Cabinet terminal sequence

The development cases and the original selected decision tables are linked in the following order: PPARC/SHARP (1,037 operational decisions), CPPI-GREG (1,059), and RACS (1,080); the final RACS consensus handles the 21 unresolved cases. These are distinct operational-validity and exact-oracle-match counts. The final record has 1,075 exact oracle matches.

The original runner versions are under `variel/frozen/resource/`. Their source SHA-256s are preserved in `SOURCE_LOCK.csv`, and identical runner bytes are in the corresponding review ZIPs in this folder. The original 504 CPPI-GREG geometric case JSON files are in `CPPI_GREG_CASE_RECORDS_V2.zip`. The larger terminal tail/prediction binaries are not distributed with source code; their paths and SHA-256s are indexed in `EXTERNAL_INPUT_HASHES.csv`.

Run `python scripts/verify_terminal_chain.py` to audit the exact final decisions and confirmed identities without model computation.
