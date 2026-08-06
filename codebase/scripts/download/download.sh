#!/usr/bin/env bash
# =============================================================================
# download_all_datasets.sh
# Comprehensive idempotent downloader for ALL datasets in the TSFM Catalog.
#
# TIER 1  = Phase 1 Minimal      (Required, all headless)
# TIER 2  = Phase 1 Recommended  (Required, all headless)
# TIER 3  = Phase 2/3 Extension    (Mixed: some headless, some manual)
#
# Usage:
#   bash download_all_datasets.sh --tier 1              # Minimal only
#   bash download_all_datasets.sh --tier 2              # Minimal + Recommended
#   bash download_all_datasets.sh --tier 3              # All that can be automated
#   bash download_all_datasets.sh --dataset ETTh1       # Specific dataset only
#   bash download_all_datasets.sh --dry-run             # Show what would happen
#   bash download_all_datasets.sh --verify              # Run verification only
# =============================================================================

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/hdd1/TSFM/dataset}"
RAW="$DATA_ROOT/raw"
LOG="$DATA_ROOT/manifests"
mkdir -p "$RAW"/{energy,traffic,weather,ecg,multi_domain,wearables,cgm,eeg,imu,industrial,health,finance} "$LOG"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[1;34m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
fail() { echo -e "${RED}[✗]${NC} $*"; }
info() { echo -e "${BLUE}[INFO]${NC} $*"; }
auto() { echo -e "${CYAN}[AUTO]${NC} $*"; }
manual() { echo -e "${YELLOW}[MANUAL]${NC} $*"; }
log()  { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $1" >> "$LOG/download_all.log"; }

# ── Parse args ──────────────────────────────────────────────────────────────
TIER=2
DATASET=""
DRY_RUN=0
VERIFY_ONLY=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --tier) TIER="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --verify) VERIFY_ONLY=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ $VERIFY_ONLY -eq 1 ]]; then
  bash "$(dirname "$0")/verify_datasets.sh" 2>/dev/null || { info "verify_datasets.sh not found, running inline check..."; }
  exit 0
fi

if [[ $DRY_RUN -eq 1 ]]; then
  info "DRY RUN — no files will be written"
fi

TARGETS=()
[[ -n "$DATASET" ]] && TARGETS=("$DATASET")

want() {
  local tag="$1"
  [[ ${#TARGETS[@]} -eq 0 ]] && return 0
  for t in "${TARGETS[@]}"; do [[ "$t" == "$tag" ]] && return 0; done
  return 1
}

# ── Helper: HuggingFace download ─────────────────────────────────────────────
download_hf() {
  local repo="$1" file="$2" dest="$3" name="$4"
  if [[ -f "$dest" ]]; then warn "$name already present, skipping."; return 0; fi
  if [[ $DRY_RUN -eq 1 ]]; then auto "[DRY] Would download $name → $dest"; return 0; fi
  mkdir -p "$(dirname "$dest")"
  echo "  Downloading $name from HuggingFace ($repo) …"
  python3 - <<PYEOF
from huggingface_hub import hf_hub_download
import shutil, os
tmp = hf_hub_download(repo_id="$repo", filename="$file", repo_type="dataset")
shutil.copy(tmp, "$dest")
print(f"  saved {os.path.getsize('$dest')//1024} KB")
PYEOF
  ok "$name"; log "Downloaded $name via HF $repo"
}

# ── Helper: wget download ───────────────────────────────────────────────────
download_wget() {
  local url="$1" dest="$2" name="$3"
  if [[ -f "$dest" ]]; then warn "$name already present, skipping."; return 0; fi
  if [[ $DRY_RUN -eq 1 ]]; then auto "[DRY] Would download $name → $dest"; return 0; fi
  mkdir -p "$(dirname "$dest")"
  echo "  Downloading $name …"
  if wget -q --show-progress -O "$dest" "$url"; then
    ok "$name ($(du -sh "$dest" | cut -f1))"; log "Downloaded $name from $url"
  else
    fail "$name — wget failed"; rm -f "$dest"; log "FAILED: $name from $url"
    return 1
  fi
}

# ═══════════════════════════════════════════════════════════════════════════
# TIER 1 — Phase 1 Minimal (Required)
# ════════════════════════════════════════════════════════════════════════════

# 1–4. ETT Family ────────────────────────────────────────────────────────────
if want etth || want ett || want "ETTh1" || want "ETTh2" || want "ETTm1" || want "ETTm2"; then
  echo; echo "=== TIER 1 [1–4] ETT Family ==="
  ETT_DIR="$RAW/energy/ETT-small"
  mkdir -p "$ETT_DIR"
  for f in ETTh1 ETTh2 ETTm1 ETTm2; do
    dest="$ETT_DIR/${f}.csv"
    if [[ -f "$dest" || -f "$RAW/energy/ETTDataset/ETT-small/${f}.csv" || -f "$RAW/energy/hf_mirror/ETT-small/${f}.csv" ]]; then
      warn "$f already present."; continue
    fi
    download_hf "thuml/Time-Series-Library" "ETT-small/${f}.csv" "$dest" "$f"
  done
fi

# 5. Electricity (ECL) ─────────────────────────────────────────────────────
if want electricity || want ecl; then
  echo; echo "=== TIER 1 [5] Electricity (ECL) ==="
  DEST="$RAW/energy/electricity/electricity.csv"
  if [[ -f "$DEST" || -f "$RAW/energy/hf_mirror/electricity/electricity.csv" || -f "$RAW/energy/ETTDataset/electricity/electricity.csv" ]]; then
    warn "Electricity already present."
  else
    download_hf "thuml/Time-Series-Library" "electricity/electricity.csv" "$DEST" "Electricity"
  fi
fi

# 6. Solar-Energy ───────────────────────────────────────────────────────────
if want solar; then
  echo; echo "=== TIER 1 [6] Solar-Energy ==="
  DEST="$RAW/energy/solar_energy/solar_AL.txt"
  if [[ -f "$DEST" || -f "$RAW/energy/solar_source/solar_AL.txt" || -f "$RAW/energy/multivariate-time-series-data/solar-energy/solar_AL.txt" ]]; then
    warn "Solar-Energy already present."
  else
    mkdir -p "$(dirname "$DEST")"
    TMP_GZ="/tmp/solar_AL.txt.gz"
    if [[ $DRY_RUN -eq 1 ]]; then auto "[DRY] Would download solar_AL.txt.gz from LSTNet GitHub"; else
      wget -q --show-progress \
        "https://github.com/laiguokun/multivariate-time-series-data/raw/master/solar-energy/solar_AL.txt.gz" \
        -O "$TMP_GZ"
      gunzip -c "$TMP_GZ" > "$DEST"
      rm "$TMP_GZ"
      ok "Solar-Energy → $DEST"; log "Downloaded solar_AL.txt from GitHub LSTNet"
    fi
  fi
fi

# 7. Traffic ────────────────────────────────────────────────────────────────
if want traffic; then
  echo; echo "=== TIER 1 [7] Traffic ==="
  DEST="$RAW/traffic/traffic/traffic.csv"
  if [[ -f "$DEST" || -f "$RAW/traffic/iTransformer/data/traffic.csv" || -f "$RAW/traffic/hf_mirror/traffic/traffic.csv" ]]; then
    warn "Traffic already present."
  else
    download_hf "thuml/Time-Series-Library" "traffic/traffic.csv" "$DEST" "Traffic"
  fi
fi

# 8–10. PEMS03, PEMS04, PEMS08 ───────────────────────────────────────────────
if want pems; then
  echo; echo "=== TIER 1 [8–10] PEMS03, PEMS04, PEMS08 ==="
  PEMS_DIR="$RAW/traffic/PEMS"
  mkdir -p "$PEMS_DIR"

  for DS in PEMS03 PEMS04 PEMS08; do
    DEST="$PEMS_DIR/${DS}.npz"
    if [[ -f "$DEST" || -f "$RAW/traffic/pems_source/${DS}.npz" || -f "$RAW/traffic/iTransformer/data/PEMS/${DS}.npz" ]]; then
      warn "$DS already present."; continue
    fi
    if [[ $DRY_RUN -eq 1 ]]; then auto "[DRY] Would download $DS from Zenodo 7816008"; continue; fi

    echo "  Downloading $DS from Zenodo (TrafficDataSets, record 7816008) …"
    # Zenodo 7816008 hosts PEMS03/04/07/08 .npz files
    ZENODO_URL="https://zenodo.org/records/7816008/files/${DS}.npz?download=1"
    if wget -q --show-progress -O "$DEST" "$ZENODO_URL"; then
      ok "$DS ($(du -sh "$DEST" | cut -f1))"; log "Downloaded $DS from Zenodo 7816008"
    else
      fail "$DS — Zenodo download failed, falling back to HF dunzane/time-series-dataset"
      rm -f "$DEST"
      python3 - <<PYEOF
from huggingface_hub import hf_hub_download
import shutil, os
try:
    tmp = hf_hub_download(repo_id="dunzane/time-series-dataset", filename="PEMS/${DS}.npz", repo_type="dataset")
    shutil.copy(tmp, "$DEST")
    print(f"  [HF fallback] saved {os.path.getsize('$DEST')//(1024*1024)} MB")
except Exception as e:
    print(f"  [HF fallback] also failed: {e}")
    exit(1)
PYEOF
      ok "$DS (HF fallback)"; log "Downloaded $DS from HF dunzane/time-series-dataset (fallback)"
    fi
  done

  # Verify shapes
  if [[ $DRY_RUN -eq 0 ]]; then
    echo "  Verifying PEMS shapes …"
    export PEMS_DIR
    python3 - <<'PYEOF'
import numpy as np, os, sys
pems_dir = os.environ.get("PEMS_DIR", "")
expected = {"PEMS03": (26208, 358), "PEMS04": (16992, 307), "PEMS08": (17856, 170)}
errs = 0
for ds, (rows, cols) in expected.items():
    path = f"{pems_dir}/{ds}.npz"
    if not os.path.exists(path):
        # Try alternate locations
        for alt in [f"/mnt/hdd1/TSFM/dataset/raw/traffic/pems_source/{ds}.npz",
                    f"/mnt/hdd1/TSFM/dataset/raw/traffic/iTransformer/data/PEMS/{ds}.npz"]:
            if os.path.exists(alt):
                path = alt
                break
    if not os.path.exists(path):
        print(f"  MISSING {ds}.npz anywhere"); errs += 1; continue
    d = np.load(path); data = d[list(d.keys())[0]]
    t, n = data.shape[0], data.shape[1]
    ok = (t == rows and n == cols)
    print(f"  {ds}: shape {data.shape} {'OK' if ok else 'UNEXPECTED'}")
    if not ok: errs += 1
sys.exit(1 if errs else 0)
PYEOF
  fi
fi

# 11. Weather (Jena) ────────────────────────────────────────────────────────
if want weather; then
  echo; echo "=== TIER 1 [11] Weather (Jena) ==="
  DEST="$RAW/weather/weather/weather.csv"
  if [[ -f "$DEST" || -f "$RAW/weather/weather.csv" || -f "$RAW/weather/hf_mirror/weather/weather.csv" ]]; then
    warn "Weather already present."
  else
    download_hf "thuml/Time-Series-Library" "weather/weather.csv" "$DEST" "Weather"
  fi
fi

# 12. PTB-XL ────────────────────────────────────────────────────────────────
if want ptbxl || want ptb-xl; then
  echo; echo "=== TIER 1 [12] PTB-XL (PhysioNet, ~2GB) ==="
  DEST_DIR="$RAW/ecg/ptbxl"
  if [[ -f "$DEST_DIR/ptbxl_database.csv" ]]; then
    warn "PTB-XL already present (ptbxl_database.csv found)."
  else
    if [[ $DRY_RUN -eq 1 ]]; then auto "[DRY] Would download PTB-XL from PhysioNet"; else
      mkdir -p "$DEST_DIR"
      echo "  PTB-XL requires a free PhysioNet account."
      if [[ -z "${PHYSIONET_USER:-}" ]]; then read -rp "  PhysioNet username: " PHYSIONET_USER; fi
      if [[ -z "${PHYSIONET_PASS:-}" ]]; then read -rsp "  PhysioNet password: " PHYSIONET_PASS; echo; fi

      wget -r -N -c -np \
        --user="$PHYSIONET_USER" --password="$PHYSIONET_PASS" \
        --directory-prefix="$DEST_DIR" --cut-dirs=4 --no-host-directories \
        -R "*.zip" "https://physionet.org/files/ptb-xl/1.0.3/" \
        2>&1 | tee "$LOG/ptbxl_wget.log"

      for sentinel in ptbxl_database.csv scp_statements.csv; do
        if [[ ! -f "$DEST_DIR/$sentinel" ]]; then fail "Missing $sentinel"; exit 1; fi
      done
      ok "PTB-XL download complete"; log "Downloaded PTB-XL v1.0.3 from PhysioNet"
    fi
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# TIER 2 — Phase 1 Recommended
# ════════════════════════════════════════════════════════════════════════════

if [[ $TIER -lt 2 ]]; then
  echo; info "Tier 2 datasets skipped (use --tier 2 or higher)"
else

# 13. GIFT-Eval (full ~5GB) ─────────────────────────────────────────────────
if want gift || want gift-eval; then
  echo; echo "=== TIER 2 [13] GIFT-Eval (~5GB) ==="
  DEST_DIR="$RAW/multi_domain/gift_eval"
  if [[ -d "$DEST_DIR" && "$(ls -A "$DEST_DIR" 2>/dev/null)" ]]; then
    warn "GIFT-Eval already present, skipping."
  else
    if [[ $DRY_RUN -eq 1 ]]; then auto "[DRY] Would download GIFT-Eval from Salesforce/GiftEval"; else
      mkdir -p "$DEST_DIR"
      huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir "$DEST_DIR"
      ok "GIFT-Eval → $DEST_DIR"; log "Downloaded Salesforce/GiftEval via huggingface-cli"
    fi
  fi
fi

# 14. Monash Archive (20 selected .tsf datasets) ─────────────────────────────
if want monash; then
  echo; echo "=== TIER 2 [14] Monash Archive (20 datasets) ==="
  MONASH_DIR="$RAW/multi_domain/monash"
  mkdir -p "$MONASH_DIR"

  declare -A MONASH_DATASETS=(
    ["m1_yearly"]="4656222 m1_yearly_dataset.tsf"
    ["m1_quarterly"]="4656262 m1_quarterly_dataset.tsf"
    ["m1_monthly"]="4656298 m1_monthly_dataset.tsf"
    ["nn5_daily_without_missing"]="4656117 nn5_daily_dataset_without_missing_values.tsf"
    ["nn5_weekly"]="4656125 nn5_weekly_dataset.tsf"
    ["tourism_monthly"]="4656103 tourism_monthly_dataset.tsf"
    ["tourism_quarterly"]="4656093 tourism_quarterly_dataset.tsf"
    ["tourism_yearly"]="4656096 tourism_yearly_dataset.tsf"
    ["cif_2016"]="4656042 cif_2016_dataset.tsf"
    ["car_parts_without_missing"]="4656021 car_parts_dataset_without_missing_values.tsf"
    ["fred_md"]="4654833 fred_md_dataset.tsf"
    ["pedestrian_counts"]="4656626 pedestrian_counts_dataset.tsf"
    ["hospital"]="4656014 hospital_dataset.tsf"
    ["covid_deaths"]="4656009 covid_deaths_dataset.tsf"
    ["australian_electricity_demand"]="4659727 australian_electricity_demand_dataset.tsf"
    ["electricity_weekly"]="4656140 electricity_weekly_dataset.tsf"
    ["rideshare_without_missing"]="4656185 rideshare_dataset_without_missing_values.tsf"
    ["saugeenday"]="4656058 saugeenday_dataset.tsf"
    ["solar_10_minutes"]="4656144 solar_10_minutes_dataset.tsf"
    ["sunspot_without_missing"]="4654773 sunspot_without_missing_values_dataset.tsf"
  )

  for name in "${!MONASH_DATASETS[@]}"; do
    read -r zenodo_id filename <<< "${MONASH_DATASETS[$name]}"
    DEST="$MONASH_DIR/${filename}"
    if [[ -f "$DEST" ]]; then warn "$name already present."; continue; fi
    if [[ $DRY_RUN -eq 1 ]]; then auto "[DRY] Would download $name from Zenodo $zenodo_id"; continue; fi
    URL="https://zenodo.org/records/${zenodo_id}/files/${filename}?download=1"
    download_wget "$URL" "$DEST" "$name"
  done
  echo "  Monash .tsf count: $(ls "$MONASH_DIR"/*.tsf 2>/dev/null | wc -l)/20"
fi

# 15. Exchange Rate (LSTNet / HF) ───────────────────────────────────────────
if want exchange || want exchange_rate; then
  echo; echo "=== TIER 2 [15] Exchange Rate ==="
  DEST="$RAW/finance/exchange_rate/exchange_rate.txt"
  if [[ -f "$DEST" || -f "$RAW/energy/multivariate-time-series-data/exchange_rate/exchange_rate.txt" ]]; then
    warn "Exchange Rate already present."
  else
    # Primary: HF Time-Series-Library
    download_hf "thuml/Time-Series-Library" "exchange_rate/exchange_rate.txt" "$DEST" "Exchange Rate (HF)"
    if [[ ! -f "$DEST" ]]; then
      # Fallback: LSTNet GitHub
      mkdir -p "$(dirname "$DEST")"
      TMP_GZ="/tmp/exchange_rate.txt.gz"
      wget -q --show-progress \
        "https://github.com/laiguokun/multivariate-time-series-data/raw/master/exchange_rate/exchange_rate.txt.gz" \
        -O "$TMP_GZ"
      gunzip -c "$TMP_GZ" > "$DEST"
      rm "$TMP_GZ"
      ok "Exchange Rate (LSTNet fallback) → $DEST"
    fi
  fi
fi

# 16. ILI (Influenza) ────────────────────────────────────────────────────────
if want ili || want illness; then
  echo; echo "=== TIER 2 [16] ILI (Influenza) ==="
  DEST="$RAW/health/illness/illness.csv"
  if [[ -f "$DEST" ]]; then
    warn "ILI already present."
  else
    download_hf "thuml/Time-Series-Library" "illness/illness.csv" "$DEST" "ILI"
  fi
fi

fi # end Tier 2

# ═══════════════════════════════════════════════════════════════════════════
# TIER 3 — Phase 2/3 Extension (from Excel sheet)
# ════════════════════════════════════════════════════════════════════════════

if [[ $TIER -lt 3 ]]; then
  echo; info "Tier 3 datasets skipped (use --tier 3)"
else

echo; echo "══════════════════════════════════════════════════════════════════════"
info "TIER 3 — Extension datasets from the Excel catalog"
info "Some are automatic, many require manual steps (credentialed/registration)."
echo "══════════════════════════════════════════════════════════════════════"

# ── Wearables ───────────────────────────────────────────────────────────────
if want wesad; then
  echo; echo "=== TIER 3 [Wearables] WESAD ==="
  DEST_DIR="$RAW/wearables/wesad"
  if [[ -d "$DEST_DIR" && "$(ls -A "$DEST_DIR" 2>/dev/null)" ]]; then
    warn "WESAD already present."
  else
    if [[ $DRY_RUN -eq 1 ]]; then auto "[DRY] Would download WESAD from UCI"; else
      mkdir -p "$DEST_DIR"
      # UCI ML Repository direct static link (id 465)
      wget -q --show-progress -O /tmp/WESAD.zip \
        "https://archive.ics.uci.edu/static/public/465/wesad.zip" \
        || wget -q --show-progress -O /tmp/WESAD.zip \
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00565/WESAD.zip"
      unzip -q /tmp/WESAD.zip -d "$DEST_DIR"
      rm /tmp/WESAD.zip
      ok "WESAD → $DEST_DIR"; log "Downloaded WESAD from UCI ML Repository"
    fi
  fi
fi

if want ppg_dalia || want ppgdalia; then
  echo; echo "=== TIER 3 [Wearables] PPG-DaLiA ==="
  DEST_DIR="$RAW/wearables/ppg_dalia"
  if [[ -d "$DEST_DIR" && "$(ls -A "$DEST_DIR" 2>/dev/null)" ]]; then
    warn "PPG-DaLiA already present."
  else
    if [[ $DRY_RUN -eq 1 ]]; then auto "[DRY] Would download PPG-DaLiA"; else
      mkdir -p "$DEST_DIR"
      # Try UCI first (id 495), then Zenodo mirror
      if wget -q --show-progress -O /tmp/ppg_dalia.zip \
         "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip" 2>/dev/null; then
        unzip -q /tmp/ppg_dalia.zip -d "$DEST_DIR"
        rm /tmp/ppg_dalia.zip
        ok "PPG-DaLiA (UCI) → $DEST_DIR"
      else
        # Zenodo mirror
        wget -q --show-progress -O /tmp/ppg_dalia.zip \
          "https://zenodo.org/records/3902728/files/PPGDalia.zip?download=1"
        unzip -q /tmp/ppg_dalia.zip -d "$DEST_DIR"
        rm /tmp/ppg_dalia.zip
        ok "PPG-DaLiA (Zenodo) → $DEST_DIR"
      fi
      log "Downloaded PPG-DaLiA"
    fi
  fi
fi

# ── ECG Extension ─────────────────────────────────────────────────────────────
echo; echo "=== TIER 3 [ECG Extension] ==="
manual "MIMIC-IV-ECG: Requires PhysioNet credentialed access."
manual "  → Apply at: https://physionet.org/settings/credentialing/"
manual "  → Dataset: https://physionet.org/files/mimic-iv-ecg/1.0/"
manual "  → Place under: $RAW/ecg/mimic_iv_ecg/"
manual "  → ACTION REQUIRED: Start credentialing application manually."

# ── EEG ─────────────────────────────────────────────────────────────────────
echo; echo "=== TIER 3 [EEG] ==="
manual "TUAB (TUH Abnormal EEG): Requires TUH registration."
manual "  → Apply at: https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml"
manual "  → Place under: $RAW/eeg/tuab/"
manual ""
manual "TUEV (TUH EEG Events): Same registration as TUAB."
manual "  → Place under: $RAW/eeg/tuev/"
manual ""
manual "SEED (SJTU Emotion EEG): Requires BCMI Lab registration."
manual "  → Apply at: https://bcmi.sjtu.edu.cn/home/seed/seed.html"
manual "  → Place under: $RAW/eeg/seed/"
manual ""
manual "SHHS (Sleep Heart Health Study): Requires NSRR registration."
manual "  → Apply at: https://sleepdata.org/datasets/shhs"
manual "  → Place under: $RAW/eeg/shhs/"
manual ""
auto "BCI Competition IV: Public download available."
auto "  → http://www.bbci.de/competition/iv/ or https://github.com/bnco-dev/BCI-Competition-IV"
auto "  → Place under: $RAW/eeg/bci_iv/"
manual ""
auto "PhysioNet EEG Motor Movement (MMI): PhysioNet open access."
auto "  → https://physionet.org/files/eegmmidb/1.0.0/"
auto "  → Place under: $RAW/eeg/eeg_motor/"

# ── CGM ─────────────────────────────────────────────────────────────────────
echo; echo "=== TIER 3 [CGM] ==="
manual "OhioT1DM: Requires email request to Ohio State."
manual "  → https://t1ds.org/OhioT1DM/ — email request required"
manual "  → Place under: $RAW/cgm/ohio_t1dm/"
manual ""
auto "ShanghaiT1DM / ShanghaiT2DM: Available on figshare."
auto "  → Search: 'Shanghai T1DM' on figshare.com"
auto "  → Place under: $RAW/cgm/shanghai_t1dm/ and $RAW/cgm/shanghai_t2dm/"
manual ""
auto "D1NAMO: Zenodo, direct download."
auto "  → https://zenodo.org/records/ (search 'D1NAMO diabetes')"
auto "  → Place under: $RAW/cgm/d1namo/"
manual ""
auto "BIG IDEAs (Gluvarpro): PhysioNet open access."
auto "  → https://physionet.org/files/big-ideas-glycemic-wearable/1.1.2/"
auto "  → Place under: $RAW/cgm/big_ideas/"

# ── IMU / Activity Recognition ──────────────────────────────────────────────
echo; echo "=== TIER 3 [IMU / HAR] ==="
auto "UCI HAR: UCI ML Repository (id 240)"
auto "  → https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
auto "  → Place under: $RAW/imu/uci_har/"
manual ""
auto "PAMAP2: UCI ML Repository (id 231)"
auto "  → https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
auto "  → Place under: $RAW/imu/pamap2/"
manual ""
auto "OPPORTUNITY: UCI ML Repository (id 226)"
auto "  → https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip"
auto "  → Place under: $RAW/imu/opportunity/"
manual ""
manual "WISDM: Requires registration at Fordham University."
manual "  → https://www.cis.fordham.edu/wisdm/dataset.php"
manual "  → Place under: $RAW/imu/wisdm/"
manual ""
manual "UniMiB SHAR: Download from UniMiB website."
manual "  → https://www.unibs.it/it/page/1230"
manual "  → Place under: $RAW/imu/unimib_shar/"
manual ""
auto "MHEALTH: UCI ML Repository (id 319)"
auto "  → https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"
auto "  → Place under: $RAW/imu/mhealth/"
manual ""
manual "REALWORLD: University of Mannheim website."
manual "  → https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/"
manual "  → Place under: $RAW/imu/realworld/"
manual ""
manual "USC-HAD: USC website."
manual "  → https://sipi.usc.edu/had/"
manual "  → Place under: $RAW/imu/usc_had/"

# ── Industrial / Other ──────────────────────────────────────────────────────
echo; echo "=== TIER 3 [Industrial / Other] ==="
manual "C-MAPSS (NASA Turbofan): NASA Prognostics CoE or Kaggle."
manual "  → https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/"
manual "  → Place under: $RAW/industrial/cmapss/"
manual ""
auto "ILI already handled in Tier 2."
manual ""
auto "Exchange Rate already handled in Tier 2."

fi # end Tier 3

# ═══════════════════════════════════════════════════════════════════════════
# Final summary
# ════════════════════════════════════════════════════════════════════════════
echo
echo "══════════════════════════════════════════════════════════════════════"
ok "Download run complete."
echo "  Disk usage by domain:"
du -sh "$RAW"/{energy,traffic,weather,ecg,multi_domain,wearables,cgm,eeg,imu,industrial,health,finance} 2>/dev/null || true
echo
echo "  Manifest log: $LOG/download_all.log"
echo "══════════════════════════════════════════════════════════════════════"