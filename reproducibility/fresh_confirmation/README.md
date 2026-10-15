# Fresh sparse confirmation: exact frozen input identity

The three data inputs are already identifiable from the original Batch16 archive. `LIABILITY_FRESH_LOCK.json` is copied unchanged from the archive and lists exact SHA-256 hashes, stream order, 50% train / 15% source / 35% target division, 12 target windows, target-class mapping, seeds and label budget.

`SOURCE_TO_PREPARED_DATA.json` records the result of directly SHA-checking each archived prepared CSV against its frozen lock. All three match. Those full CSVs are in the original `DTRAR_LiabilityFreshAudit_Batch16_Review.zip`, nested within the authors' authoritative source archive; they are not redistributed in this GitHub code package.

The original preparation script `variel/frozen/sparse/audit_dtrar_liability_fresh_batch16.py` is included byte-for-byte and checks the MHEALTH official ZIP before conversion. The earlier Batch13 structural converter and its official-UCI continuation script are also preserved as original source (`audit_dtrar_final_holdouts_batch13.py` and `audit_dtrar_final_holdouts_batch13_continue_v2.py`). UCI 760 was converted by the initial Batch13 script; UCI 427 uses the official UCI static-archive continuation script. The output hashes are checked against Batch16. The exact prepared CSV bytes, nevertheless, were located and verified.

The Batch17 final assessment is a separate numerical result. No new model training or evaluation was performed while assembling this manifest.
