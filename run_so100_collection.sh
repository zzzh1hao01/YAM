#!/usr/bin/env bash
# =============================================================================
# SO100-leader -> YAM data collection, end to end.
#
# Wraps the full operator flow from docs/so100_collection_finetune_plan.md:
#   preflight checks -> CAN/watchdog startup -> (calibration if needed) ->
#   teleop collection -> post-session reminders.
#
# Assumes the SO100 leader arms are ALREADY CALIBRATED: the calibration JSONs
# referenced by agent.calibration_path in the so100 yamls must exist
# (lerobot-style per-motor calibration files are also accepted by the agent).
#
# Usage (on the rig machine, any directory):
#   ./run_so100_collection.sh              # full flow: checks, CAN, collect
#   ./run_so100_collection.sh replay       # open-loop replay gate for recorded episodes
#   ./run_so100_collection.sh --skip-can   # skip CAN reset/watchdog (already done this boot)
#
# Keypad during collection (focus must be ON the color-pad window!):
#   s = start episode    a = save + end    b = discard + end
# Exit the launcher normally (not ctrl+c) so auto-convert/upload runs.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GELLO="$REPO_ROOT/gello_software"
LEFT_CFG="$GELLO/configs/yam_left_so100.yaml"
RIGHT_CFG="$GELLO/configs/yam_right_so100.yaml"
CAL_LEFT="$GELLO/configs/so100_calibration_left.json"
CAL_RIGHT="$GELLO/configs/so100_calibration_right.json"

MODE="collect"
SKIP_CAN=0
for arg in "$@"; do
  case "$arg" in
    replay|collect) MODE="$arg" ;;
    --skip-can) SKIP_CAN=1 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $arg (try --help)"; exit 1 ;;
  esac
done

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mFAIL: %s\033[0m\n' "$*"; exit 1; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- preflight --
say "Preflight checks"

[ "${CONDA_DEFAULT_ENV:-}" = "ai2_yam" ] \
  || die "conda env 'ai2_yam' is not active. Run: conda activate ai2_yam"

python - <<'PY' || die "feetech-servo-sdk not installed. Run: pip install feetech-servo-sdk"
import scservo_sdk  # noqa: F401
PY

for cfg in "$LEFT_CFG" "$RIGHT_CFG"; do
  [ -f "$cfg" ] || die "missing config: $cfg"
  if grep -q 'usb-\.\.\.' "$cfg"; then
    die "SO100 serial port is still a placeholder in $cfg
  Find yours with:  ls /dev/serial/by-id/
  and put the full path in the agent.port field."
  fi
done

# Stale storage values inherited from the previous task -- refuse to record
# with a wrong instruction, warn about the rest.
INSTR=$(grep -E '^\s*language_instruction:' "$LEFT_CFG" | head -1)
case "$INSTR" in
  *"can into the basket"*) die "language_instruction in $LEFT_CFG is still the stale default:
  $INSTR
  Set it to your actual task (e.g. 'fold the towel in half') before collecting." ;;
esac
grep -Eq '^\s*task_directory:\s*"?shirt' "$LEFT_CFG" \
  && warn "task_directory is still 'shirt' in $LEFT_CFG -- did you mean fold_laundry?"

BASE_DIR=$(python - "$LEFT_CFG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["storage"]["base_dir"])
PY
)
[ -d "$BASE_DIR" ] || die "storage.base_dir does not exist on this machine: $BASE_DIR
  Edit it in $LEFT_CFG"

AUTO_UPLOAD=$(python - "$LEFT_CFG" <<'PY'
import sys, yaml
print(str(yaml.safe_load(open(sys.argv[1]))["lerobot"].get("auto_upload", False)).lower())
PY
)
if [ "$AUTO_UPLOAD" = "true" ]; then
  python - <<'PY' || die "auto_upload is on but no HuggingFace login. Run: hf auth login  (token is in the shared .env)"
from huggingface_hub import whoami
whoami()
PY
else
  warn "lerobot.auto_upload is false -- episodes will convert locally but not push to HF."
fi

# ------------------------------------------------------------ CAN + watchdog --
if [ "$SKIP_CAN" -eq 1 ]; then
  say "Skipping CAN reset/watchdog (--skip-can)"
else
  say "CAN reset + 400ms motor watchdog (per-boot sequence)"
  ip link show can_leader_l  >/dev/null 2>&1 || die "CAN interface can_leader_l not found. Check cabling; see: ip link show | grep can"
  ip link show can_follower_r >/dev/null 2>&1 || die "CAN interface can_follower_r not found."
  sh "$REPO_ROOT/i2rt/scripts/reset_all_can.sh"
  python "$REPO_ROOT/i2rt/i2rt/motor_config_tool/set_timeout.py" --channel can_leader_l  --timeout
  python "$REPO_ROOT/i2rt/i2rt/motor_config_tool/set_timeout.py" --channel can_follower_r --timeout
fi

# -------------------------------------------------- calibration (pre-done) --
for cal in "$CAL_LEFT" "$CAL_RIGHT"; do
  [ -f "$cal" ] || die "calibration file missing: $cal
  The arms are assumed pre-calibrated. Point agent.calibration_path in the
  so100 yamls at your existing calibration files (lerobot-style per-motor
  JSONs are also accepted by SO100LeaderAgent)."
done

# ------------------------------------------------------------------ collect --
cd "$GELLO"
if [ "$MODE" = "replay" ]; then
  say "Open-loop replay gate: recorded actions drive the followers with no teleop.
    The demo should reproduce cleanly -- jitter/drift here means fix calibration
    or smoothing BEFORE collecting at scale."
  exec python experiments/launch_yaml_replay.py \
      --left_config_path  configs/yam_left_so100.yaml \
      --right_config_path configs/yam_right_so100.yaml
fi

say "Starting collection.
  Keypad ('s' start / 'a' save / 'b' discard) needs focus on the color-pad window.
  Exit normally at session end -- ctrl+c SKIPS the convert/upload pipeline.
  First session? Record 2-3 throwaway episodes, then: $0 replay --skip-can"
python experiments/launch_yaml_collect_data.py \
    --left_config_path  configs/yam_left_so100.yaml \
    --right_config_path configs/yam_right_so100.yaml

say "Session complete. Post-session checklist:
  1. Check the converted dataset's meta/info.json:
       robot_type == bi_yam_follower, cameras top/left/right, (14,) state/action, fps 30
  2. Open-loop replay a random converted episode:  $0 replay --skip-can
  3. When happy, set lerobot.auto_upload: true in $LEFT_CFG so future
     sessions push straight to the hackathon HF dataset."
