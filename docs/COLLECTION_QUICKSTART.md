# SO100 → YAM Data Collection Quickstart

From fresh clone to uploaded LeRobot dataset. Assumes: a bimanual YAM rig with
two CAN buses + three RealSense cameras, two **pre-calibrated** SO100 leader
arms on USB serial, and Ubuntu with miniconda.

Background/design context: `docs/so100_collection_finetune_plan.md`.

## 1. Clone + install (once)

```bash
git clone https://github.com/zzzh1hao01/YAM.git
cd YAM
./setup_so100.sh          # creates conda env ai2_yam, installs i2rt/gello/lerobot + feetech SDK
conda activate ai2_yam
```

## 2. Fill in the machine-specific config (once)

All placeholders live in `gello_software/configs/yam_left_so100.yaml` (and the
agent block of `yam_right_so100.yaml`):

| Field | Set to |
|---|---|
| `agent.port` (both yamls) | your SO100 serial devices — `ls /dev/serial/by-id/` |
| `agent.calibration_path` (both) | your existing SO100 calibration JSONs (lerobot-style per-motor files accepted) |
| `robot.channel` | verify against `ip link show \| grep can` (defaults: `can_leader_l` left, `can_follower_r` right) |
| `storage.base_dir` | a real directory on this machine for episode data |
| `storage.task_directory` | e.g. `fold_laundry` |
| `storage.language_instruction` | the actual task instruction, e.g. `"fold the towel in half"` |
| `storage.episodes` | episodes per session |
| `lerobot.hf_repo_id` | the team dataset repo, e.g. `<hf-user>/hackathon` |

Open decision points to settle on the rig (see the plan doc §1.2/§Open):
the SO100→YAM joint correspondence + pinned wrist joint (`agent` block knobs)
and the gripper mapping.

## 3. Credentials (once)

Get the shared `.env` from a teammate (never via git). For uploads the rig
needs a HuggingFace login:

```bash
hf auth login --token $HF_ACCESS_TOKEN
```

## 4. Collect

```bash
./run_so100_collection.sh              # preflight checks -> CAN/watchdog -> teleop collection
```

The wrapper refuses to start on stale/placeholder config and walks you through
what to fix. During collection, keyboard focus must be on the **color-pad
window**: `s` start episode, `a` save, `b` discard. Exit normally at the end of
the session — `ctrl+c` skips the convert/upload pipeline.

Same-boot later sessions: `./run_so100_collection.sh --skip-can`.

## 5. Validate before collecting at scale (first session only)

1. Record 2–3 throwaway episodes.
2. Open-loop replay them: `./run_so100_collection.sh replay --skip-can` —
   the arms must reproduce the demo cleanly from recorded actions alone.
   Jitter or drift here means fix calibration/smoothing first; the model
   consumes 30-step action chunks open-loop, so replay quality is training
   quality.
3. Check one converted episode's `meta/info.json`:
   `robot_type: bi_yam_follower`, features `observation.images.{top,left,right}`,
   `(14,)` state/action, `fps: 30`.
4. Then set `lerobot.auto_upload: true` and collect for real. Vary garment
   placement/appearance across episodes; keep demos smooth and unhesitating.

## Troubleshooting

- **Both CAN buses report `loss communication` at once** — a long GIL-holding
  operation ran while motors were live; keep heavy init before motors
  energize (see `CLAUDE.md` on the 400 ms watchdog).
- **Gripper drifted after power-cycle** — re-zero motor 7 (commands in
  `CLAUDE.md` startup section).
- **Upload failed / skipped** — run the converter manually from the repo root:
  `python molmoact_to_lerobot_v30.py --config_path gello_software/configs/yam_left_so100.yaml`
  (individual CLI flags override the config values).
