# #!/usr/bin/env bash
# # =============================================================================
# # verify_datasets.sh
# # Run this first. It checks every expected raw file for all 14 benchmark
# # datasets and prints a clear PASS / FAIL / SKIP report.
# # =============================================================================

# set -euo pipefail

# DATA_ROOT="${DATA_ROOT:-/mnt/hdd1/TSFM/dataset}"
# RAW="$DATA_ROOT/raw"

# RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[1;34m'; NC='\033[0m'
# pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
# fail() { echo -e "${RED}[FAIL]${NC} $*"; }
# warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
# info() { echo -e "${BLUE}[INFO]${NC} $*"; }

# ERRORS=0
# WARNINGS=0

# check_file() {
#   local path="$1"; local desc="$2"
#   if [[ -f "$path" ]]; then
#     pass "$desc  →  $path  ($(du -sh "$path" 2>/dev/null | cut -f1))"
#   else
#     fail "$desc  →  $path NOT FOUND"; ((ERRORS++)) || true
#   fi
# }

# check_dir() {
#   local path="$1"; local desc="$2"; local min_files="${3:-1}"
#   if [[ -d "$path" ]]; then
#     local n=$(find "$path" -maxdepth 1 -type f 2>/dev/null | wc -l)
#     if [[ $n -ge $min_files ]]; then
#       pass "$desc  →  $path ($n files)"
#     else
#       warn "$desc  →  $path exists but only $n files (expected ≥$min_files)"; ((WARNINGS++)) || true
#     fi
#   else
#     fail "$desc  →  $path NOT FOUND"; ((ERRORS++)) || true
#   fi
# }

# echo "══════════════════════════════════════════════════════════════════"
# info "Verifying TSFM benchmark datasets under $RAW"
# info "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
# echo "══════════════════════════════════════════════════════════════════"
# echo

# # ── 1–4. ETT Family ─────────────────────────────────────────────────────────
# echo ">>> 1–4. ETT Family (ETTh1, ETTh2, ETTm1, ETTm2)"
# ETT_DIR="$RAW/energy/ETT-small"
# if [[ -d "$RAW/energy/ETTDataset/ETT-small" && ! -d "$ETT_DIR" ]]; then
#   ETT_DIR="$RAW/energy/ETTDataset/ETT-small"
# fi
# for f in ETTh1 ETTh2 ETTm1 ETTm2; do
#   check_file "$ETT_DIR/${f}.csv" "$f"
# done
# echo

# # ── 5. Electricity (ECL) ────────────────────────────────────────────────────
# echo ">>> 5. Electricity (ECL)"
# # May live in a few places depending on how it was downloaded
# ECL_CANDIDATES=(
#   "$RAW/energy/electricity/electricity.csv"
#   "$RAW/energy/hf_mirror/electricity/electricity.csv"
#   "$RAW/energy/ETTDataset/electricity/electricity.csv"
# )
# ECL_FOUND=0
# for c in "${ECL_CANDIDATES[@]}"; do
#   if [[ -f "$c" ]]; then
#     pass "Electricity  →  $c  ($(du -sh "$c" | cut -f1))"; ECL_FOUND=1; break
#   fi
# done
# [[ $ECL_FOUND -eq 0 ]] && { fail "Electricity  →  not found in any expected location"; ((ERRORS++)) || true; }
# echo

# # ── 6. Solar-Energy ─────────────────────────────────────────────────────────
# echo ">>> 6. Solar-Energy"
# SOLAR_CANDIDATES=(
#   "$RAW/energy/solar_energy/solar_AL.txt"
#   "$RAW/energy/solar_energy/solar_AL.txt.gz"
#   "$RAW/energy/multivariate-time-series-data/solar-energy/solar_AL.txt"
#   "$RAW/energy/solar_source/solar_AL.txt"
# )
# SOLAR_FOUND=0
# for c in "${SOLAR_CANDIDATES[@]}"; do
#   if [[ -f "$c" ]]; then
#     pass "Solar-Energy  →  $c  ($(du -sh "$c" | cut -f1))"; SOLAR_FOUND=1; break
#   fi
# done
# [[ $SOLAR_FOUND -eq 0 ]] && { fail "Solar-Energy  →  not found in any expected location"; ((ERRORS++)) || true; }
# echo

# # ── 7. Traffic ─────────────────────────────────────────────────────────────
# echo ">>> 7. Traffic"
# TRAFFIC_CANDIDATES=(
#   "$RAW/traffic/traffic/traffic.csv"
#   "$RAW/traffic/iTransformer/data/traffic.csv"
#   "$RAW/traffic/iTransformer/data_provider/traffic.csv"
# )
# TRAFFIC_FOUND=0
# for c in "${TRAFFIC_CANDIDATES[@]}"; do
#   if [[ -f "$c" ]]; then
#     pass "Traffic  →  $c  ($(du -sh "$c" | cut -f1))"; TRAFFIC_FOUND=1; break
#   fi
# done
# [[ $TRAFFIC_FOUND -eq 0 ]] && { fail "Traffic  →  not found in any expected location"; ((ERRORS++)) || true; }
# echo

# # ── 8–10. PEMS03, PEMS04, PEMS08 ──────────────────────────────────────────
# echo ">>> 8–10. PEMS (03, 04, 08)"
# for ds in PEMS03 PEMS04 PEMS08; do
#   check_file "$RAW/traffic/PEMS/${ds}.npz" "$ds"
# done
# echo

# # ── 11. Weather (Jena) ────────────────────────────────────────────────────
# echo ">>> 11. Weather (Jena)"
# WEATHER_CANDIDATES=(
#   "$RAW/weather/weather/weather.csv"
#   "$RAW/weather/weather.csv"
# )
# WEATHER_FOUND=0
# for c in "${WEATHER_CANDIDATES[@]}"; do
#   if [[ -f "$c" ]]; then
#     pass "Weather  →  $c  ($(du -sh "$c" | cut -f1))"; WEATHER_FOUND=1; break
#   fi
# done
# [[ $WEATHER_FOUND -eq 0 ]] && { fail "Weather  →  not found in any expected location"; ((ERRORS++)) || true; }
# echo

# # ── 12. PTB-XL ─────────────────────────────────────────────────────────────
# echo ">>> 12. PTB-XL"
# check_file "$RAW/ecg/ptbxl/ptbxl_database.csv" "PTB-XL metadata"
# check_file "$RAW/ecg/ptbxl/scp_statements.csv" "PTB-XL label definitions"
# check_dir  "$RAW/ecg/ptbxl/records100" "PTB-XL 100Hz records" 1000
# echo

# # ── 13. GIFT-Eval ─────────────────────────────────────────────────────────
# echo ">>> 13. GIFT-Eval (subset or full)"
# GIFT_DIR="$RAW/multi_domain/gift_eval"
# if [[ -d "$GIFT_DIR" ]]; then
#   n=$(find "$GIFT_DIR" -maxdepth 1 -type d | wc -l)
#   if [[ $n -ge 5 ]]; then
#     pass "GIFT-Eval  →  $GIFT_DIR ($n sub-directories)"
#   else
#     warn "GIFT-Eval  →  $GIFT_DIR only $n sub-dirs (looks incomplete)"; ((WARNINGS++)) || true
#   fi
# else
#   fail "GIFT-Eval  →  $GIFT_DIR NOT FOUND"; ((ERRORS++)) || true
# fi
# echo

# # ── 14. Monash Archive ────────────────────────────────────────────────────
# echo ">>> 14. Monash Archive (20 selected .tsf files)"
# MONASH_DIR="$RAW/multi_domain/monash"
# if [[ -d "$MONASH_DIR" ]]; then
#   n=$(ls "$MONASH_DIR"/*.tsf 2>/dev/null | wc -l)
#   if [[ $n -eq 20 ]]; then
#     pass "Monash  →  $MONASH_DIR ($n/20 .tsf files)"
#   else
#     warn "Monash  →  $MONASH_DIR only $n/20 .tsf files"; ((WARNINGS++)) || true
#   fi
# else
#   fail "Monash  →  $MONASH_DIR NOT FOUND"; ((ERRORS++)) || true
# fi
# echo

# # ── Wearables (WESAD) — tracked but not required for Minimal tier ─────────
# echo ">>> Wearables (WESAD) — deferred/optional"
# if [[ -d "$RAW/wearables/wesad" ]]; then
#   n=$(find "$RAW/wearables/wesad" -maxdepth 1 -type d | wc -l)
#   pass "WESAD  →  $RAW/wearables/wesad ($n sub-dirs)"
# else
#   warn "WESAD  →  not present (deferred per BENCHMARK_SPEC.md, not a blocker)"; ((WARNINGS++)) || true
# fi
# echo

# # ── Summary ────────────────────────────────────────────────────────────────
# echo "══════════════════════════════════════════════════════════════════"
# if [[ $ERRORS -eq 0 && $WARNINGS -eq 0 ]]; then
#   pass "ALL CHECKS PASSED — every required dataset is present."
# elif [[ $ERRORS -eq 0 ]]; then
#   warn "ALL REQUIRED DATASETS PRESENT — $WARNINGS non-critical warning(s)."
# else
#   fail "$ERRORS missing item(s), $WARNINGS warning(s). Run download_missing.sh next."
# fi
# echo "══════════════════════════════════════════════════════════════════"

#!/usr/bin/env bash
# =============================================================================
# verify_datasets.sh
# Run this FIRST on your server. It checks every expected file for all datasets
# listed in the TSFM Catalog Excel sheet across multiple possible locations.
# =============================================================================

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/hdd1/TSFM/dataset}"
RAW="$DATA_ROOT/raw"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[1;34m'; NC='\033[0m'
pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
info() { echo -e "${BLUE}[INFO]${NC} $*"; }

ERRORS=0
WARNINGS=0

# Helper: check file across multiple candidate paths
check_multi() {
  local desc="$1"; shift
  local found=0
  for p in "$@"; do
    if [[ -f "$p" ]]; then
      pass "$desc  →  $p  ($(du -sh "$p" 2>/dev/null | cut -f1))"
      found=1; break
    fi
  done
  if [[ $found -eq 0 ]]; then
    fail "$desc  →  NOT FOUND in any expected location"; ((ERRORS++)) || true
  fi
}

# Helper: check directory with minimum file count
check_dir() {
  local desc="$1"; local path="$2"; local min="${3:-1}"
  if [[ -d "$path" ]]; then
    local n=$(find "$path" -maxdepth 1 -type f 2>/dev/null | wc -l)
    if [[ $n -ge $min ]]; then
      pass "$desc  →  $path ($n files)"
    else
      warn "$desc  →  $path exists but only $n files (expected ≥$min)"; ((WARNINGS++)) || true
    fi
  else
    fail "$desc  →  $path NOT FOUND"; ((ERRORS++)) || true
  fi
}

echo "══════════════════════════════════════════════════════════════════════"
info "Verifying TSFM benchmark datasets under $RAW"
info "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "══════════════════════════════════════════════════════════════════════"
echo

# ═══════════════════════════════════════════════════════════════════════════
# TIER 1 — Phase 1 Minimal (Required)
# ═══════════════════════════════════════════════════════════════════════════
echo ">>> TIER 1 — Phase 1 Minimal (Required)"

# 1–4. ETT Family
echo "  [1–4] ETT Family"
ETT_DIR="$RAW/energy/ETT-small"
[[ -d "$RAW/energy/ETTDataset/ETT-small" ]] && ETT_DIR="$RAW/energy/ETTDataset/ETT-small"
for f in ETTh1 ETTh2 ETTm1 ETTm2; do
  check_multi "$f.csv" "$ETT_DIR/${f}.csv" "$RAW/energy/ett_source/${f}.csv" "$RAW/energy/hf_mirror/ETT-small/${f}.csv"
done

# 5. Electricity (ECL)
echo "  [5] Electricity (ECL)"
check_multi "electricity.csv" \
  "$RAW/energy/electricity/electricity.csv" \
  "$RAW/energy/hf_mirror/electricity/electricity.csv" \
  "$RAW/energy/ETTDataset/electricity/electricity.csv" \
  "$RAW/energy/multivariate-time-series-data/electricity/electricity.txt"

# 6. Solar-Energy
echo "  [6] Solar-Energy"
check_multi "solar_AL.txt" \
  "$RAW/energy/solar_energy/solar_AL.txt" \
  "$RAW/energy/solar_source/solar_AL.txt" \
  "$RAW/energy/multivariate-time-series-data/solar-energy/solar_AL.txt"

# 7. Traffic
echo "  [7] Traffic"
check_multi "traffic.csv" \
  "$RAW/traffic/traffic/traffic.csv" \
  "$RAW/traffic/iTransformer/data/traffic.csv" \
  "$RAW/traffic/iTransformer/data_provider/traffic.csv" \
  "$RAW/traffic/hf_mirror/traffic/traffic.csv"

# 8–10. PEMS
echo "  [8–10] PEMS03, PEMS04, PEMS08"
for ds in PEMS03 PEMS04 PEMS08; do
  check_multi "${ds}.npz" "$RAW/traffic/PEMS/${ds}.npz" "$RAW/traffic/pems_source/${ds}.npz" "$RAW/traffic/iTransformer/data/PEMS/${ds}.npz"
done

# 11. Weather
echo "  [11] Weather (Jena)"
check_multi "weather.csv" \
  "$RAW/weather/weather/weather.csv" \
  "$RAW/weather/weather.csv" \
  "$RAW/weather/hf_mirror/weather/weather.csv"

# 12. PTB-XL
echo "  [12] PTB-XL"
check_multi "ptbxl_database.csv" "$RAW/ecg/ptbxl/ptbxl_database.csv"
check_multi "scp_statements.csv" "$RAW/ecg/ptbxl/scp_statements.csv"
check_dir  "PTB-XL 100Hz records" "$RAW/ecg/ptbxl/records100" 1000
echo

# ═══════════════════════════════════════════════════════════════════════════
# TIER 2 — Phase 1 Recommended
# ═══════════════════════════════════════════════════════════════════════════
echo ">>> TIER 2 — Phase 1 Recommended"

# 13. GIFT-Eval
echo "  [13] GIFT-Eval"
GIFT_DIR="$RAW/multi_domain/gift_eval"
if [[ -d "$GIFT_DIR" ]]; then
  n=$(find "$GIFT_DIR" -maxdepth 1 -type d 2>/dev/null | wc -l)
  if [[ $n -ge 5 ]]; then
    pass "GIFT-Eval  →  $GIFT_DIR ($n sub-directories)"
  else
    warn "GIFT-Eval  →  $GIFT_DIR only $n sub-dirs (looks incomplete)"; ((WARNINGS++)) || true
  fi
else
  fail "GIFT-Eval  →  $GIFT_DIR NOT FOUND"; ((ERRORS++)) || true
fi

# 14. Monash Archive
echo "  [14] Monash Archive (20 selected .tsf files)"
MONASH_DIR="$RAW/multi_domain/monash"
if [[ -d "$MONASH_DIR" ]]; then
  n=$(ls "$MONASH_DIR"/*.tsf 2>/dev/null | wc -l)
  if [[ $n -eq 20 ]]; then
    pass "Monash  →  $MONASH_DIR ($n/20 .tsf files)"
  else
    warn "Monash  →  $MONASH_DIR only $n/20 .tsf files"; ((WARNINGS++)) || true
  fi
else
  fail "Monash  →  $MONASH_DIR NOT FOUND"; ((ERRORS++)) || true
fi

# 15. Solar (Monash duplicate — flagged for dedup scanner)
echo "  [15] solar_10_minutes (Monash, flagged 3-way duplicate)"
check_multi "solar_10_minutes_dataset.tsf" "$MONASH_DIR/solar_10_minutes_dataset.tsf"
echo

# ═══════════════════════════════════════════════════════════════════════════
# TIER 3 — Phase 2/3 Extension (from Excel sheet)
# ═══════════════════════════════════════════════════════════════════════════
echo ">>> TIER 3 — Phase 2/3 Extension (from Excel sheet)"

# Wearables
echo "  Wearables"
check_dir "WESAD" "$RAW/wearables/wesad" 10

# ECG Extension
echo "  ECG Extension"
check_multi "MIMIC-IV-ECG metadata" "$RAW/ecg/mimic_iv_ecg/RECORDS" 2>/dev/null || true

# EEG
echo "  EEG (Optional/Full tier)"
for d in tuab tuev seed shhs bci_iv eeg_motor; do
  [[ -d "$RAW/eeg/$d" ]] && pass "$d  →  present" || warn "$d  →  not present (optional)"
done

# CGM
echo "  CGM (Future Phase)"
for d in ohio_t1dm shanghai_t1dm shanghai_t2dm d1namo big_ideas; do
  [[ -d "$RAW/cgm/$d" ]] && pass "$d  →  present" || warn "$d  →  not present (optional)"
done

# IMU / Activity Recognition
echo "  IMU / HAR (Team extension)"
for d in uci_har pamap2 opportunity wisdm unimib_shar mhealth realworld usc_had; do
  [[ -d "$RAW/imu/$d" ]] && pass "$d  →  present" || warn "$d  →  not present (optional)"
done

# Other
echo "  Other"
check_multi "C-MAPSS" "$RAW/industrial/cmapss/"* 2>/dev/null || warn "C-MAPSS  →  not present (optional)"
check_multi "ILI" "$RAW/health/illness/illness.csv" "$RAW/health/illness.csv" 2>/dev/null || warn "ILI  →  not present (optional)"
check_multi "Exchange Rate" \
  "$RAW/finance/exchange_rate/exchange_rate.txt" \
  "$RAW/energy/multivariate-time-series-data/exchange_rate/exchange_rate.txt" \
  "$RAW/energy/hf_mirror/exchange_rate/exchange_rate.txt" 2>/dev/null || warn "Exchange Rate  →  not present (optional)"
echo

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
echo "══════════════════════════════════════════════════════════════════════"
if [[ $ERRORS -eq 0 && $WARNINGS -eq 0 ]]; then
  pass "ALL CHECKS PASSED — every required dataset is present."
elif [[ $ERRORS -eq 0 ]]; then
  warn "ALL REQUIRED DATASETS PRESENT — $WARNINGS non-critical warning(s)."
else
  fail "$ERRORS missing item(s), $WARNINGS warning(s). Run download_all_datasets.sh next."
fi
echo "══════════════════════════════════════════════════════════════════════"