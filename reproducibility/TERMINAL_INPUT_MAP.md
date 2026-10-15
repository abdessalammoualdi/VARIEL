# Frozen source-to-input map

| Mechanism | Original frozen source | Input / terminal evidence | Archive-link status |
|---|---|---|---|
| Occupancy v2 / EEG | `unseen_prepared_audit/audit_dtrar_unseen_datasets_v2.py` | `UNSEEN_DATASET_AUDIT_V2.json`, `PREPARED_FILES.json` | Original prepared rows and hashes recorded; original CSV bytes not distributed |
| PPARC/SHARP | `variel/frozen/resource/run_pparc_sharp.py` | `resource_cabinet/terminal_upstream/SHARP_SELECTED_ALL_SIX.csv`, `SHARP_UNRESOLVED_CASES.csv` | Final SHA-256-linked source, 1,080 decisions / 43 unresolved |
| CPPI-GREG | `variel/frozen/resource/run_cppigreg_terminal.py` | `CPPI_GREG_SELECTED_ALL_SIX.csv`, `CPPI_GREG_UNRESOLVED_CASES.csv`, `CPPI_GREG_CASE_RECORDS_V2.zip` | Final SHA-256-linked source, 1,080 decisions / 21 unresolved |
| RACS | `variel/frozen/resource/run_racs_closure.py` | `resource_cabinet/racs/` | Frozen matched inputs and 1,080 operational decisions, 21 rescue cases |
| BOSA | `variel/frozen/resource/run_bosa_nmcs_all_six.py` | `BOSA_SELECTED_DEPLOYMENT_SEEDWISE.csv` | Original six-stream results and source hash retained |
| Temporal Actionability | `variel/frozen/temporal/run_dtrar_temporal_actionability.py` | `temporal/DTRAR_TemporalActionability_Final_Review.zip` and final summary inputs | Result identity preserved; terminal runner hash not recorded in final ZIP |
| High-information | `variel/frozen/highinfo/run_dominance_guard.py` | Original development/confirmation archives in `high_information/` | Frozen sources and summaries; trained endpoint cache inputs indexed separately |
| V24 preservation | `variel/frozen/preservation/run_v24.py`, `pg19.py`, `pg20.py`, `secondary_v18.py` | Original final V24 review, main summary, ten per-job JSON records | Frozen sources and assessments; large input probability caches indexed separately |
| V13 multiclass | `variel/frozen/multiclass/run_xgboost_ltt_mc_v13.py` | V13 review plus V10/Compass archives | Frozen summary/input references; V12 static cache hashes indexed separately |

Source/figure hashes can be verified without executing experimental code using `scripts/verify.py`, `scripts/verify_input_linkage.py`, `scripts/verify_racs_linkage.py` and `scripts/verify_terminal_chain.py`. These integrity checks are not substitutions for model fitting or independent experimental reproduction.
