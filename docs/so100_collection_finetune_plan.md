# Plan: SO100-Leader Teleop Data Collection → MolmoAct2-BimanualYAM Fine-Tuning (Modal)

Goal: collect teleoperated demonstrations of a bimanual task (e.g. laundry folding) on the
bimanual YAM using **SO100 leader arms** as the teleop input device (instead of GELLO), convert
to LeRobot v3.0, fine-tune `allenai/MolmoAct2-BimanualYAM` on Modal, and deploy back to this rig.

The guiding principle: **reuse the existing gello_software runtime end-to-end** (env loop,
cameras, data saver, keypad, auto-convert/upload, eval launchers). The only genuinely new
hardware code is an SO100 leader *agent* that plugs into the existing `Agent` protocol.

---

## 0. Data contract (what training needs, regardless of teleop device)

The model never sees the teleop device — only what is recorded. Every episode must satisfy:

| Field | Requirement |
|---|---|
| `observation.state` | (14,) float32 **YAM follower** joints: `[left 6 joints + gripper, right 6 joints + gripper]`. Never leader joints. |
| `action` | (14,) **absolute YAM joint targets** (`action_mode: next_joint_fields` — the commanded next joint state, not deltas, not leader positions). |
| Cameras | `observation.images.top` (base/front cam), `.left` (left wrist), `.right` (right wrist). Order and left/right arm assignment must be consistent. |
| Meta | `fps: 30`, `robot_type: "bi_yam_follower"` (verified against released allenai YAM datasets' `meta/info.json`), a `language_instruction` per episode. |
| Norm tag | Training mixture reuses tag `yam_dual_molmoact2` (matches the checkpoint and the inference server's `NORM_TAG`). |

⚠️ **Known key mismatch to fix:** `molmoact_to_lerobot_v30.py` currently emits
`observation.images.camera_{left,right,front}`, but the training mixture
(`experiments/launch_scripts/data_mixtures.py`, tag `yam_dual_molmoact2`) and the released YAM
datasets use `observation.images.{top,left,right}`. Phase 2 includes patching the converter's
feature naming (`camera_front` → `top`, `camera_left` → `left`, `camera_right` → `right`).

---

## Phase 1 — SO100 leader agent (new code)

### 1.1 `gello_software/gello/agents/so100_agent.py`

New `SO100LeaderAgent` implementing the existing `Agent` protocol
(`gello/agents/agent.py`: `act(obs) -> np.ndarray`), one instance per arm, composed with the
existing `BimanualAgent` — exactly how `GelloAgent` is wired today.

- **Bus read:** SO100 leaders are Feetech STS3215 servos on a serial bus. Read positions
  directly via `scservo_sdk` (`feetech-servo-sdk` pip package) — 6 values per arm
  (5 joints + gripper trigger). Avoid importing the local `lerobot/` checkout into the runtime;
  it is conversion-side only. Support loading an existing lerobot-style calibration JSON if one
  exists so hand-tuned ranges aren't redone.
- **Output:** 7-D absolute YAM joint target per arm `[j1..j6, gripper]`, so `BimanualAgent`
  yields the 14-D action the env/data pipeline already expects.
- **Torque off** on the leader bus at init (leader is passive, moved by hand).

### 1.2 Retargeting (decision point)

SO100 is 5-DoF + gripper; each YAM arm is 6-DoF + gripper. Two options:

1. **Joint-space mapping with one YAM joint pinned** *(recommended start)* — map SO100's
  5 joints onto 5 YAM joints via per-joint `sign * scale * (leader - offset) + home`, hold the
  remaining YAM wrist joint at a fixed pose. Simple, deterministic, low-latency; costs one
  wrist DoF, which is often acceptable for folding-style tabletop tasks.
2. **EE-space retargeting (FK on SO100 → IK on YAM)** — faithful 6-DoF control but requires
  URDF/IK plumbing (mujoco menagerie model is in `third_party/`) and careful smoothness work.
  Fall back to this only if option 1 proves unusable for the task.

### 1.3 `gello_software/scripts/calibrate_so100_leader.py` (new)

Interactive per-arm calibration producing `configs/so100_calibration_{left,right}.json`:

- For each joint: move leader to two reference poses matching known YAM poses → solve
  sign/scale/offset. For the gripper: capture open/closed trigger extremes → map to the YAM
  gripper range (`normalize_gripper=False` in training, so raw consistency across episodes
  matters more than any particular range).
- Sanity print: live leader→YAM target readout without commanding motors.

### 1.4 Config + launcher wiring

- Add an `agent` block variant to `configs/yam_left.yaml` / `yam_right.yaml` using
  `_target_: gello.agents.so100_agent.SO100LeaderAgent` with `port` (serial by-id path) and
  `calibration_path`. Add `"so100"` to the `storage.teleop_device` options in
  `launch_yaml_collect_data.py` (metadata only — device selection is driven by the agent block).
- **Safety rails (reuse, don't rewrite):** keep the existing start-position interpolation and
  max-joint-delta guard from the launch flow; keep the 400 ms motor watchdog startup sequence;
  keep any heavy init (bus scans, camera warmup) *before* motors are energized (GIL/watchdog
  caveat in `CLAUDE.md`).
- **Smoothing:** SO100 reads can be noisier than GELLO's Dynamixels — add an EMA filter +
  per-step joint-delta clamp inside the agent, tunable from config.

### 1.5 Bench validation before any data collection

- Leader-only dry run (motors off): live plot of retargeted targets, confirm ranges/signs.
- Single-arm teleop at low speed, then bimanual (`launch_yaml.py`).
- Record 2–3 throwaway episodes and **replay them open-loop**
  (`launch_yaml_replay.py` / `launch_yaml_open_loop.py`): the recorded `action` stream must
  reproduce the demo cleanly. This directly validates the training targets, because the model
  consumes 30-step action chunks open-loop.

---

## Phase 2 — Converter fix + collection at scale

### 2.1 Patch `molmoact_to_lerobot_v30.py` camera keys

Emit `observation.images.top` / `.left` / `.right` (mapping `front_rgb/` → `top`). Verify one
converted episode's `meta/info.json` features against the released
`allenai/MolmoAct2-BimanualYAM` datasets before collecting at scale.

### 2.2 Collect with the existing pipeline (unchanged)

`launch_yaml_collect_data.py` with `configs/yam_left.yaml`:

```yaml
storage:
  episodes: <target>          # collect in sessions; ~100+ demos for laundry folding
  task_directory: "fold_laundry"
  language_instruction: "fold the towel in half"   # vary phrasing across sessions if desired
  teleop_device: "so100"
  save_format: "json"
lerobot:
  auto_convert: true
  auto_upload: true
  hf_repo_id: "<user>/yam_fold_laundry_lerobot_v30"
  fps: 30
  robot_type: "bi_yam_follower"
  action_mode: "next_joint_fields"
```

Operational notes: keypad `s` start / `a` save / `b` discard needs focus on the color pad
window; `ctrl+c` skips convert/upload. Per-boot CAN + watchdog sequence per `CLAUDE.md`
(`can_leader_l` / `can_follower_r` on this machine). Vary garment placement/appearance across
episodes; keep demos smooth and unhesitating.

### 2.3 QC gate before training

- Visualize episodes (`molmoact2/experiments/scripts/dataset_visualize.py`).
- Check shapes ((14,) state/action), camera viewpoints/keys, instruction presence, fps.
- Open-loop replay a random sample of *converted* episodes.

---

## Phase 3 — Register mixture + smoke test

In `molmoact2/experiments/launch_scripts/data_mixtures.py`: copy `build_molmoact2_yam`
(line ~357) → `build_molmoact2_yam_fold`, pointing `repo_ids` at the new HF dataset, **reusing
tag `yam_dual_molmoact2`** and keeping `action_horizon=30`, `n_action_steps=30`,
`control_mode="absolute joint pose"`, `normalize_gripper=False`. Register in
`MOLMOACT2_LEROBOT_MIXTURES`.

Smoke test (1 GPU, 20 steps, `--packing=false --dynamic_seq_len=true`, recipe in
`molmoact2/experiments/README.md`) with start checkpoint `allenai/MolmoAct2-BimanualYAM`.

---

## Phase 4 — Fine-tune on Modal

- **Mode:** LoRA (recommended: single task, same embodiment) — LoRA on VLM path + fully
  trained action expert, `--lora_rank=64`, LRs per the README recipe. Action-expert-only is the
  cheaper fallback; full FT only if the dataset grows multi-task.
- **Modal app** (`modal_train.py`, new): image based on `molmoact2/experiments/Dockerfile` deps
  (CUDA devel + `pip install -e "experiments[all]" -e "experiments/lerobot[async]"`); Modal
  Volume mounted for `HF_HOME`, `LEROBOT_DATA_ROOT`, and `--save_folder`; Secrets for
  `HF_ACCESS_TOKEN` (private dataset pull) and `WANDB_API_KEY`; `gpu="H100:N"` and `torchrun
  --standalone --nproc-per-node=N launch_scripts/train_lerobot.py allenai/MolmoAct2-BimanualYAM
  yam_fold ...`. Keep `--global_batch_size=64` by trading `device_batch_size` vs GPU count.
- **Long-run safety:** high function `timeout`, `--save_interval` checkpointing to the Volume,
  periodic volume commits, resume-from-checkpoint on relaunch.

---

## Phase 5 — Merge, deploy, evaluate

1. Merge LoRA (`molmoact2/experiments/scripts/merge_lora.py`); ensure the exported checkpoint
   ships `norm_stats.json` containing the `yam_dual_molmoact2` tag.
2. Open-loop sanity check against held-out episodes
   (`experiments/scripts/run_open_loop_inference_lerobot.py`) before touching the robot.
3. Serve with `molmoact2/examples/yam/host_server_yam.py` pointed at the fine-tuned checkpoint
   (keep the upstream bf16/schema workarounds; YAM uses `inference_action_mode="continuous"`).
4. Evaluate on-robot with the existing eval launcher in **server mode**:
   `launch_yaml_eval_molmoact.py` with `eval.mode: server`, `eval.molmoact_server: <host:8202>`
   — success/failure labeling and eval→LeRobot conversion come for free.
5. Measure folding success over held-out garment placements; iterate on data (more demos where
   it fails) rather than hyperparameters first.

---

## Open decision points

1. **Retargeting scheme** (Phase 1.2): joint-space with pinned wrist joint vs EE-space IK —
   start with joint-space; revisit only if the task needs the sixth DoF.
2. **Which YAM joint to pin** and the exact SO100→YAM joint correspondence — settle during
   calibration on the real rig.
3. **Gripper mapping curve** (linear vs thresholded) — pick during bench validation.
4. **Instruction diversity** — single fixed string vs varied phrasings per session (varied is
   closer to how the checkpoint was trained with annotated instructions).

## Likeliest failure points (check early, they're silent)

- Camera key naming mismatch (the converter patch in 2.1) — breaks training key lookup.
- Recording leader joints instead of follower joints anywhere in the pipeline.
- Jittery retargeted actions → jittery open-loop chunks (caught by the replay gate in 1.5).
- Inconsistent gripper convention across sessions.
- Missing `yam_dual_molmoact2` entry in the deployed checkpoint's `norm_stats.json`.
