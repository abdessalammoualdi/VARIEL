#!/usr/bin/env python3
# Compatibility bridge used by the exact frozen dtrar_final_core.py.
# It exposes the exact RAPID helper snapshot as `core`; the complete historical
# risk-optimal source is bundled in evidence/ with its original SHA-256.
import run_dtrar_rapid_pilot as core
BASE=core.BASE
