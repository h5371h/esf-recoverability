#!/bin/bash
# Usage: regenerate.sh <repo_clone_dir> <v2_results_dir>   — places v2 CSVs under the v1 names and reruns every downstream script.
set -e; R="$1"; S="$2"; PY="${PY:-python3}"
mkdir -p "$R/data/v1_archive"; for f in sweep_latest.csv per_recording_predictions_latest.csv sweep_tuab_final.csv per_recording_predictions_tuab_final.csv; do [ -f "$R/data/$f" ] && cp -n "$R/data/$f" "$R/data/v1_archive/" || true; done
cp "$S/sweep_v2.csv" "$R/data/sweep_tuab_final.csv"; cp "$S/per_recording_predictions_v2.csv" "$R/data/per_recording_predictions_tuab_final.csv"
cp "$S/sweep_v2.csv" "$R/data/sweep_latest.csv";     cp "$S/per_recording_predictions_v2.csv" "$R/data/per_recording_predictions_latest.csv"
cd "$R"; $PY src/paper_figures.py; $PY src/failure_taxonomy.py; $PY src/validation_compute.py; $PY src/run_all_advanced_stats.py --per-rec-csv data/per_recording_predictions_latest.csv --sweep-csv data/sweep_latest.csv --out-json data/advanced_stats_results.json --out-summary data/advanced_stats_summary.md
echo REGEN_DONE
