import atexit
from math import inf
from multiprocessing import Process
import os
import signal
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, Optional

import tyro
import zmq.error
from omegaconf import OmegaConf

from gello.utils.launch_utils import instantiate_from_dict, move_to_start_position
from gello.dynamixel.driver import DynamixelDriver
import numpy as np

from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.data_utils.data_saver import DataSaver
from gello.data_utils.keyboard_interface import KBReset
from gello.utils.control_utils import run_control_loop_prior
from gello.zmq_core.camera_node import ZMQClientCamera, ZMQServerCamera

# Accepted values for storage.teleop_device. Metadata only: it is recorded
# with each episode; the actual input device is selected by the agent block
# (agent._target_) in the config.
TELEOP_DEVICES = ("oculus", "keyboard", "gello", "so100", "none")

# Global variables for cleanup
active_threads = []
active_servers = []
cleanup_in_progress = False

_env = None
_bimanual = False
_left_cfg = None
_right_cfg = None
_agent = None
_robot = None
_robot_client = None
_cameras = None
_data_saver = None
_kb_interface = None
_rerun_collector = None  # optional Rerun take recorder (see --rerun / YAM_RERUN=1)


def _call_cleanup_methods(resource, resource_name: str, methods: list[str]) -> None:
    """Best-effort cleanup helper for heterogeneous resources."""
    if resource is None:
        return
    for method_name in methods:
        if hasattr(resource, method_name):
            try:
                getattr(resource, method_name)()
            except Exception as e:
                print(f"Error calling {resource_name}.{method_name}(): {e}")
            return


def _close_realsense_camera(camera, camera_name: str) -> None:
    """Stop RealSense capture thread/pipeline if camera has no public close()."""
    if camera is None:
        return
    if hasattr(camera, "close"):
        _call_cleanup_methods(camera, camera_name, ["close"])
        return
    try:
        if hasattr(camera, "_stop_event"):
            camera._stop_event.set()
        if hasattr(camera, "_capture_thread") and camera._capture_thread is not None:
            camera._capture_thread.join(timeout=2)
        if hasattr(camera, "_pipeline") and camera._pipeline is not None:
            camera._pipeline.stop()
    except Exception as e:
        print(f"Error closing camera {camera_name}: {e}")


def cleanup():
    """Clean up resources before exit."""
    global cleanup_in_progress
    global _env, _agent, _robot, _robot_client, _cameras, _data_saver, _kb_interface
    global _rerun_collector
    if cleanup_in_progress:
        return
    cleanup_in_progress = True

    print("Cleaning up resources...")

    # Close any in-progress Rerun take first (fast: stamps properties + flushes the
    # .rrd footer) so an interrupted episode is still a valid recording on disk.
    if _rerun_collector is not None:
        try:
            _rerun_collector.close()
        except Exception as e:
            print(f"Error closing rerun collector: {e}")
        _rerun_collector = None

    try:
        if _env is not None and _left_cfg is not None:
            if _bimanual:
                move_to_start_position(_env, _bimanual, _left_cfg, _right_cfg)
            else:
                move_to_start_position(_env, _bimanual, _left_cfg)
    except Exception as e:
        print(f"Warning: failed to move robot to start position during cleanup: {e}")

    # Stop server loops first so background threads can exit.
    for server in active_servers:
        try:
            if hasattr(server, "stop"):
                server.stop()
        except Exception as e:
            print(f"Error stopping server: {e}")

    for server in active_servers:
        try:
            if hasattr(server, "close"):
                server.close()
        except Exception as e:
            print(f"Error closing server: {e}")

    for thread in active_threads:
        if thread.is_alive():
            thread.join(timeout=5)

    _call_cleanup_methods(_robot_client, "robot_client", ["close", "stop", "shutdown"])
    _call_cleanup_methods(_robot, "robot", ["close", "stop", "shutdown"])
    _call_cleanup_methods(_agent, "agent", ["close", "stop", "shutdown"])
    _call_cleanup_methods(_env, "env", ["close", "stop", "shutdown"])
    _call_cleanup_methods(_data_saver, "data_saver", ["close", "stop", "shutdown"])

    if isinstance(_cameras, dict):
        for camera_name, camera in _cameras.items():
            _close_realsense_camera(camera, camera_name)

    if _kb_interface is not None:
        _call_cleanup_methods(_kb_interface, "kb_interface", ["close", "stop", "shutdown"])
        try:
            import pygame

            pygame.quit()
        except Exception as e:
            print(f"Error quitting pygame: {e}")

    active_servers.clear()
    active_threads.clear()
    _robot_client = None
    _robot = None
    _agent = None
    _env = None
    _cameras = None
    _data_saver = None
    _kb_interface = None

    print("Cleanup completed.")


def wait_for_server_ready(port, host="127.0.0.1", timeout_seconds=5):
    """Wait for ZMQ server to be ready with retry logic."""
    from gello.zmq_core.robot_node import ZMQClientRobot

    attempts = int(timeout_seconds * 10)  # 0.1s intervals
    for attempt in range(attempts):
        try:
            client = ZMQClientRobot(port=port, host=host)
            time.sleep(0.1)
            return True
        except (zmq.error.ZMQError, Exception):
            time.sleep(0.1)
        finally:
            if "client" in locals():
                client.close()
            time.sleep(0.1)
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"Server failed to start on {host}:{port} within {timeout_seconds} seconds"
                )
    return False


@dataclass
class Args:
    left_config_path: str
    """Path to the left arm configuration YAML file."""

    right_config_path: Optional[str] = None
    """Path to the right arm configuration YAML file (for bimanual operation)."""

    rerun: bool = False
    """Also record each episode as a Rerun take (.rrd) via yam_rerun (molmoact2 repo
    root). Same as setting YAM_RERUN=1. Requires rerun-sdk==0.34.1 importable; if the
    import fails a warning is printed and collection runs unchanged. The existing
    JSON/PNG save path stays ON in parallel either way."""

    # use_save_interface: bool = False
    # """Enable saving data with keyboard interface."""


# --------------------------------------------------------------------------------------
# Optional Rerun take recording (--rerun / YAM_RERUN=1).
#
# Everything below is import-guarded: rerun-sdk and the yam_rerun package (which lives at
# the molmoact2 repo root, three levels above this file) are only imported inside
# _make_rerun_collector(), so this launcher runs unchanged on machines without them.
#
# Per plan (rerun-yam-port-plan.md Phase 1), each saved-or-discarded episode becomes a
# "take": a RecordingStream teed to recordings/<dataset>/<episode>.rrd (+ the live gRPC
# proxy on :9876 when yam_rerun.server is running), stamped with dataset/task/tag
# properties, compacted with `rerun rrd optimize`, and registered into the local catalog
# on :51234. Entity paths follow molmoact_to_lerobot_v30.py's contract:
# camera/{top,left,right} (front camera == top), {left,right}_arm/{position,goal,velocity}
# with the 14-D state split [left_joint1..6, left_gripper, right_joint1..6, right_gripper].
#
# GIL note (YAM/CLAUDE.md): take begin/finish are millisecond-scale; the expensive parts
# (`rrd optimize` subprocess + catalog registration) run on a background thread, never on
# the control loop while the 250 Hz CAN thread depends on the main thread yielding.
# --------------------------------------------------------------------------------------

_RERUN_CAMERA_PATHS = (
    ("front_camera_rgb", "camera/top"),  # base/front camera -> observation.images.top
    ("left_camera_rgb", "camera/left"),
    ("right_camera_rgb", "camera/right"),
)
_RERUN_LEFT_DIMS = [f"left_joint{i}" for i in range(1, 7)] + ["left_gripper"]
_RERUN_RIGHT_DIMS = [f"right_joint{i}" for i in range(1, 7)] + ["right_gripper"]


def rerun_enabled(args: "Args") -> bool:
    return bool(args.rerun) or os.environ.get("YAM_RERUN", "") == "1"


class _RerunUrdf:
    """Guarded adapter around yam_rerun.urdf_yam.DualYam.

    ``DualYam.create()`` parses the YAM URDF once (called from _make_rerun_collector,
    i.e. before the motors go live). Every logging call is guarded so a runtime failure
    degrades to a one-line warning + no URDF animation, never a broken collection run.
    """

    def __init__(self, module) -> None:
        self._dual = module.DualYam.create()
        self.visual_paths = [arm.visual_geometries_path for arm in self._dual.arms]
        self._disabled = False

    def _guard(self, fn, *fn_args) -> None:
        if self._disabled:
            return
        try:
            fn(*fn_args)
        except Exception as err:
            self._disabled = True
            print(f"[rerun]     urdf_yam call failed ({type(err).__name__}: {err}); URDF logging disabled", flush=True)

    def log_static(self, rec) -> None:
        self._guard(self._dual.log_static, rec)

    def log_joints(self, rec, q14) -> None:
        self._guard(self._dual.log_state, rec, q14)


class _RerunCollector:
    """Records each episode as a Rerun take, driven by three transparent wrappers.

    The collection loop itself lives in gello.utils.control_utils.run_control_loop_prior
    and is not modified; instead the objects this launcher passes into it are wrapped:

    * kb_interface.update() results drive the take lifecycle: "start" begins a take,
      "save" finishes it tagged "Good episode", "discard" tagged "Bad episode".
    * env.step() logs one Rerun tick per control tick while a take is open (cameras,
      position, goal = the gello leader command as actually sent to the robot, i.e.
      minus the dynamic offset, velocities, URDF joints).
    * data_saver.add_observation() counts ticks so an episode that hits
      max_episode_length without a keypress (which run_control_loop_prior discards)
      still closes its take (tagged "Needs review") BEFORE the between-episode
      move_to_start ticks would pollute it.
    """

    def __init__(self, rr, takes, blueprint_mod, urdf, cfg: dict) -> None:
        self._rr = rr
        self._takes = takes
        self._blueprint_mod = blueprint_mod
        self._urdf = urdf
        storage = cfg.get("storage", {})
        self.dataset = takes.sanitize_name(str(storage.get("task_directory", "yam_dataset")))
        self.task = str(storage.get("language_instruction", ""))
        self.recordings_dir = takes.DEFAULT_RECORDINGS_DIR
        self.catalog_uri = takes.DEFAULT_CATALOG_URI
        self.jpeg_quality = int(cfg.get("rerun", {}).get("jpeg_quality", 75))
        self.max_ticks = int(cfg.get("collection", {}).get("max_episode_length", 0)) or None
        self.proxy_uri = self._probe_proxy(takes.DEFAULT_GRPC_PORT)
        self._take = None
        self._episode = None
        self._path = None
        self._ticks = 0
        self._pending: list[threading.Thread] = []
        self._warned_tick = False
        print(
            f"[rerun]     takes -> {self.recordings_dir / self.dataset}/ "
            f"(dataset '{self.dataset}', live proxy {'ON' if self.proxy_uri else 'OFF -- start yam_rerun.server for live view'})",
            flush=True,
        )

    @staticmethod
    def _probe_proxy(port: int) -> Optional[str]:
        import socket

        try:
            with socket.create_connection(("localhost", port), timeout=0.3):
                return f"rerun+http://localhost:{port}/proxy"
        except OSError:
            return None

    # --- wrappers ---------------------------------------------------------------

    def wrap_env(self, env):
        collector = self

        class _Env:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def step(self, joints, reset: Optional[bool] = False):
                obs = self._inner.step(joints, reset=reset)
                if collector._take is not None:
                    offset = getattr(self._inner, "_dynamic_offset", None)
                    goal = joints if offset is None else joints - offset
                    collector.log_tick(goal, obs)
                return obs

        return _Env(env)

    def wrap_kb(self, kb):
        collector = self

        class _KB:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def update(self, dashboard_data: Optional[Dict[str, Any]] = None) -> str:
                result = self._inner.update(dashboard_data)
                phase = (dashboard_data or {}).get("phase")
                if result == "start" and phase == "waiting_start":
                    collector.begin()
                elif result == "save":
                    collector.finish("Good episode")
                elif result == "discard":
                    collector.finish("Bad episode")
                return result

        return _KB(kb)

    def wrap_data_saver(self, saver):
        collector = self

        class _Saver:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def add_observation(self, obs):
                self._inner.add_observation(obs)
                collector.note_saved_tick()

        return _Saver(saver)

    # --- take lifecycle ---------------------------------------------------------

    def begin(self) -> None:
        if self._take is not None:
            return
        rr, takes = self._rr, self._takes
        try:
            episode = takes.next_episode(self.recordings_dir, self.dataset)
            path = takes.episode_path(self.recordings_dir, self.dataset, episode)
            rec = takes.begin_take(path, episode=episode, dataset=self.dataset, task=self.task, proxy_uri=self.proxy_uri)
            # Static per-take data: series names (plot legends) + the URDF meshes.
            for arm, names in (("left_arm", _RERUN_LEFT_DIMS), ("right_arm", _RERUN_RIGHT_DIMS)):
                rec.log(f"{arm}/position", rr.SeriesLines(names=names), static=True)
                rec.log(f"{arm}/goal", rr.SeriesLines(names=[f"{n} goal" for n in names]), static=True)
                rec.log(f"{arm}/velocity", rr.SeriesLines(names=names), static=True)
            if self._urdf is not None:
                self._urdf.log_static(rec)
            self._take = rec
            self._episode = episode
            self._path = path
            self._ticks = 0
            print(f"[rerun]     take started: {path}", flush=True)
        except Exception as err:
            self._take = None
            print(f"[rerun]     failed to start take ({type(err).__name__}: {err}); episode not recorded to rerun", flush=True)

    def log_tick(self, goal14, obs) -> None:
        rec = self._take
        if rec is None:
            return
        rr = self._rr
        try:
            rec.set_time("time", timestamp=time.time())
            q = np.asarray(obs["joint_positions"], dtype=np.float64)
            rec.log("left_arm/position", rr.Scalars(q[:7]))
            rec.log("right_arm/position", rr.Scalars(q[7:14]))
            if goal14 is not None:
                g = np.asarray(goal14, dtype=np.float64)
                if g.shape[0] >= 14:
                    rec.log("left_arm/goal", rr.Scalars(g[:7]))
                    rec.log("right_arm/goal", rr.Scalars(g[7:14]))
            vel = obs.get("joint_velocities")
            if vel is not None:
                v = np.asarray(vel, dtype=np.float64)
                if v.shape[0] >= 14:
                    rec.log("left_arm/velocity", rr.Scalars(v[:7]))
                    rec.log("right_arm/velocity", rr.Scalars(v[7:14]))
            # The RealSense frames arrive as raw RGB arrays here (in-process camera path;
            # no ZMQ JPEG bytes exist in this launcher), so encode once to JPEG.
            for obs_key, entity_path in _RERUN_CAMERA_PATHS:
                image = obs.get(obs_key)
                if image is not None:
                    rec.log(entity_path, rr.Image(image).compress(jpeg_quality=self.jpeg_quality))
            if self._urdf is not None:
                self._urdf.log_joints(rec, q)
        except Exception as err:
            if not self._warned_tick:
                self._warned_tick = True
                print(f"[rerun]     tick logging failed ({type(err).__name__}: {err}); further errors suppressed", flush=True)

    def note_saved_tick(self) -> None:
        if self._take is None:
            return
        self._ticks += 1
        if self.max_ticks is not None and self._ticks >= self.max_ticks:
            # Episode hit max_episode_length with no keypress: run_control_loop_prior
            # discards it, and the next thing it does is drive the arms home -- close
            # the take now so those ticks don't leak into the recording.
            self.finish("Needs review")

    def finish(self, tag: str, *, background: bool = True) -> None:
        rec, self._take = self._take, None
        if rec is None:
            return
        takes = self._takes
        path, episode = self._path, self._episode
        try:
            takes.finish_take(rec, dataset=self.dataset, task=self.task, tag=tag, proxy_uri=None)
        except Exception as err:
            print(f"[rerun]     failed to finalize take {path} ({type(err).__name__}: {err})", flush=True)
            return
        print(f"[rerun]     take stopped: {episode} (tag: {tag})", flush=True)

        def _postprocess() -> None:
            # Off the control loop: `rrd optimize` (subprocess) + catalog registration
            # (gRPC). Registration failure is fine -- the server rescan picks it up.
            try:
                takes.optimize_rrd(path)
            except Exception as err:
                print(f"[rerun]     optimize failed for {path} ({type(err).__name__}: {err})", flush=True)
            try:
                registration = takes.register_rrd(self.catalog_uri, self.dataset, path)
                print(f"[rerun]     registered {episode} in dataset '{self.dataset}' (segments: {registration['segment_ids']})", flush=True)
                if self._blueprint_mod is not None:
                    visual_paths = self._urdf.visual_paths if self._urdf is not None else None
                    if self._blueprint_mod.register_dataset_blueprint(
                        self.catalog_uri, self.recordings_dir, self.dataset, visual_paths=visual_paths
                    ):
                        print(f"[rerun]     default blueprint set for dataset '{self.dataset}'", flush=True)
            except Exception as err:
                print(
                    f"[rerun]     catalog registration skipped for {path} ({type(err).__name__}: {err}) -- "
                    "yam_rerun.server rescans recordings/ on startup",
                    flush=True,
                )

        if background:
            worker = threading.Thread(target=_postprocess, name="rerun-postprocess", daemon=True)
            worker.start()
            self._pending.append(worker)
        else:
            _postprocess()

    def close(self) -> None:
        # An interrupted episode is worth keeping for triage -> "Needs review".
        # Postprocess synchronously: we're exiting, daemon threads won't survive.
        self.finish("Needs review", background=False)
        for worker in self._pending:
            worker.join(timeout=60)
        self._pending.clear()


def _make_rerun_collector(cfg: dict) -> Optional[_RerunCollector]:
    """Build the collector, or return None (with a warning) if rerun isn't available."""
    repo_root = Path(__file__).resolve().parents[3]  # molmoact2/ (this file: molmoact2/YAM/gello_software/experiments/)
    if (repo_root / "yam_rerun").is_dir() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        import rerun as rr
        from yam_rerun import takes
    except Exception as err:
        print(f"[rerun]     disabled: {type(err).__name__}: {err} (need rerun-sdk==0.34.1 + yam_rerun on sys.path)", flush=True)
        return None
    try:
        from yam_rerun import blueprint as blueprint_mod
    except Exception as err:
        print(f"[rerun]     blueprint module unavailable ({type(err).__name__}: {err}); recording without a default blueprint", flush=True)
        blueprint_mod = None
    urdf = None
    try:
        from yam_rerun import urdf_yam

        urdf = _RerunUrdf(urdf_yam)
    except Exception as err:
        print(f"[rerun]     urdf_yam unavailable ({type(err).__name__}: {err}); recording without URDF animation", flush=True)
    return _RerunCollector(rr, takes, blueprint_mod, urdf, cfg)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    cleanup()
    import os

    os._exit(0)

def get_joint_offsets(
    cfg: dict, port: str
):
    """Get joint offsets using the same logic as gello_get_offset.py."""
    joint_ids = list(cfg["agent"]["dynamixel_config"]["joint_ids"])
    driver = DynamixelDriver(joint_ids, port=port, baudrate=57600)

    def get_error(offset: float, index: int, joint_state: np.ndarray) -> float:
        joint_sign_i = cfg["agent"]["dynamixel_config"]["joint_signs"][index]
        joint_i = joint_sign_i * (joint_state[index] - offset)
        start_i = cfg["agent"]["start_joints"][index]
        return np.abs(joint_i - start_i)

    # Warmup
    for _ in range(10):
        driver.get_joints()

    best_offsets = []
    curr_joints = driver.get_joints()

    for i in range(len(joint_ids)):
        best_offset = 0
        best_error = float('inf')
        for offset in np.linspace(-8 * np.pi, 8 * np.pi, 500):
            error = get_error(offset, i, curr_joints)
            if error < best_error:
                best_error = error
                best_offset = offset
        best_offsets.append(best_offset)

    driver.close()
    return best_offsets

def update_offsets(cfg):
    if "dynamixel_config" not in cfg["agent"]:
        # Non-Dynamixel leader (e.g. gello.agents.so100_agent.SO100LeaderAgent):
        # offsets come from the agent's own calibration file.
        return cfg
    joint_offsets = get_joint_offsets(cfg, cfg["agent"]["port"])
    cfg["agent"]["dynamixel_config"]["joint_offsets"] = joint_offsets
    return cfg


def run_post_collection_pipeline(cfg: dict) -> None:
    """Run optional post-collection conversion/upload/tag pipeline."""
    storage_cfg = cfg.get("storage", {})
    lerobot_cfg = cfg.get("lerobot", {})
    auto_convert = bool(lerobot_cfg.get("auto_convert", False))
    auto_upload = bool(lerobot_cfg.get("auto_upload", False))
    if not auto_convert and not auto_upload:
        return
    if auto_upload and not auto_convert:
        print(
            "Skipping post-collection upload because lerobot.auto_convert is false. "
            "Enable lerobot.auto_convert to run conversion+upload pipeline."
        )
        return

    base_dir = Path(storage_cfg["base_dir"]).expanduser()
    task_directory = storage_cfg["task_directory"]
    json_data_dir = base_dir / task_directory
    lerobot_dir = base_dir / f"{task_directory}_lerobot_v30"
    repo_id = lerobot_cfg.get("hf_repo_id", storage_cfg.get("hf_repo_id"))
    if auto_upload and not repo_id:
        raise ValueError(
            "lerobot.hf_repo_id is required when lerobot.auto_upload is true."
        )

    converter_script = Path(__file__).resolve().parents[2] / "molmoact_to_lerobot_v30.py"
    if not converter_script.exists():
        raise FileNotFoundError(f"Converter script not found: {converter_script}")
    if not json_data_dir.exists():
        raise FileNotFoundError(f"Collected json directory not found: {json_data_dir}")
    if lerobot_dir.exists():
        remove_dir = input(
            f"The LeRobot output directory {lerobot_dir} already exists. "
            "Do you want to remove it and continue? (y/n): "
        ).strip().lower()
        if remove_dir == "y":
            shutil.rmtree(lerobot_dir)
            lerobot_dir.mkdir(parents=True, exist_ok=True)
            print(f"Removed and recreated output directory: {lerobot_dir}")
        elif remove_dir == "n":
            print("Conversion canceled by user because output directory already exists.")
            return
        else:
            print("Invalid input. Conversion canceled.")
            return

    convert_cmd = [
        sys.executable,
        str(converter_script),
        "--data_dir",
        str(json_data_dir),
        "--output_dir",
        str(lerobot_dir),
        "--repo_id",
        str(repo_id or "molmoact_v30"),
        "--fps",
        str(lerobot_cfg.get("fps", storage_cfg.get("lerobot_fps", cfg.get("hz", 30)))),
        "--robot_type",
        str(
            lerobot_cfg.get(
                "robot_type", storage_cfg.get("lerobot_robot_type", "molmoact_dual_arm")
            )
        ),
        "--skip_initial_frames",
        str(lerobot_cfg.get("skip_initial_frames", storage_cfg.get("lerobot_skip_initial_frames", 0))),
        "--action_mode",
        str(
            lerobot_cfg.get(
                "action_mode", storage_cfg.get("lerobot_action_mode", "next_joint_fields")
            )
        ),
        "--task_instruction",
        str(storage_cfg.get("language_instruction", "perform the task")),
        "--sanitize_online_viz_meta",
        str(
            int(
                bool(
                    lerobot_cfg.get(
                        "sanitize_online_viz_meta",
                        storage_cfg.get("sanitize_online_viz_meta", True),
                    )
                )
            )
        ),
        "--vcodec",
        str(lerobot_cfg.get("vcodec", "h264")),
        "--image_writer_processes",
        str(int(lerobot_cfg.get("image_writer_processes", 8))),
        "--image_writer_threads",
        str(int(lerobot_cfg.get("image_writer_threads", 8))),
        "--parallel_encoding",
        str(int(bool(lerobot_cfg.get("parallel_encoding", True)))),
        "--upload_to_hf",
        str(int(auto_upload)),
        "--delete_local_after_upload",
        str(
            int(
                bool(
                    lerobot_cfg.get(
                        "delete_local_after_upload",
                        storage_cfg.get("delete_local_after_upload", True),
                    )
                )
            )
        ),
    ]
    print(f"Running post-collection pipeline: {' '.join(convert_cmd)}")
    subprocess.run(convert_cmd, check=True)

    print("Post-collection pipeline completed successfully.")

def main():
    # Register cleanup handlers
    # If terminated without cleanup, can leave ZMQ sockets bound causing "address in use" errors or resource leaks

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = tyro.cli(Args)

    # left, right front camera (the device id order is based on the plugged in order on the adapter)
    ids = get_device_ids()
    print(f"Found {len(ids)} camera devices")
    print(ids)

    bimanual = args.right_config_path is not None

    # Load configs
    left_cfg = OmegaConf.to_container(
        OmegaConf.load(args.left_config_path), resolve=True
    )
    teleop_device = left_cfg.get("storage", {}).get("teleop_device")
    if teleop_device is not None and teleop_device not in TELEOP_DEVICES:
        print(
            f"Warning: storage.teleop_device={teleop_device!r} is not one of "
            f"{TELEOP_DEVICES}; recording it as-is."
        )
    left_cfg = update_offsets(left_cfg)
    if bimanual:
        right_cfg = OmegaConf.to_container(
            OmegaConf.load(args.right_config_path), resolve=True
        )
        right_cfg = update_offsets(right_cfg)

    # Initialize data saver and keyboard interface
    data_saver = DataSaver(
        save_dir=left_cfg["storage"]["base_dir"],
        task_directory=left_cfg["storage"]["task_directory"],
        language_instruction=left_cfg["storage"]["language_instruction"],
        saver_max_workers=left_cfg["storage"].get("saver_max_workers"),
        png_compress_level=left_cfg["storage"].get("png_compress_level", 1),
    )
    kb_interface = KBReset()

    # Build the Rerun collector (if enabled) BEFORE any robot is constructed: importing
    # rerun-sdk takes seconds and holds the GIL, which would starve the 250 Hz CAN
    # thread once the motors are live (see YAM/CLAUDE.md's watchdog note).
    global _rerun_collector
    if rerun_enabled(args):
        _rerun_collector = _make_rerun_collector(left_cfg)

    camera_cfg = left_cfg["sensors"]["cameras"]
    cameras = {
        "left_camera": RealSenseCamera(camera_cfg["left_camera"]["device_id"]),
        "front_camera": RealSenseCamera(camera_cfg["front_camera"]["device_id"]),
        "right_camera": RealSenseCamera(camera_cfg["right_camera"]["device_id"]),
    }

    # Create agent
    if bimanual:
        from gello.agents.agent import BimanualAgent

        agent = BimanualAgent(
            agent_left=instantiate_from_dict(left_cfg["agent"]),
            agent_right=instantiate_from_dict(right_cfg["agent"]),
        )
    else:
        agent = instantiate_from_dict(left_cfg["agent"])

    # Create robot(s)
    left_robot_cfg = left_cfg["robot"]
    if isinstance(left_robot_cfg.get("config"), str):
        left_robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(left_robot_cfg["config"]), resolve=True
        )

    left_robot = instantiate_from_dict(left_robot_cfg)

    if bimanual:
        from gello.robots.robot import BimanualRobot

        right_robot_cfg = right_cfg["robot"]
        if isinstance(right_robot_cfg.get("config"), str):
            right_robot_cfg["config"] = OmegaConf.to_container(
                OmegaConf.load(right_robot_cfg["config"]), resolve=True
            )

        right_robot = instantiate_from_dict(right_robot_cfg)
        robot = BimanualRobot(left_robot, right_robot)

        # For bimanual, use the left config for general settings (hz, etc.)
        cfg = left_cfg
    else:
        robot = left_robot
        cfg = left_cfg

    # Handle different robot types
    if hasattr(robot, "serve"):  # MujocoRobotServer or ZMQServerRobot
        print("Starting robot server...")
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot

        # Get server configuration
        server_port = cfg["robot"].get("port", 5556)
        server_host = cfg["robot"].get("host", "127.0.0.1")

        # Start server in background (non-daemon for proper cleanup)
        server_thread = threading.Thread(target=robot.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(robot)

        # Wait for server to be ready
        print(f"Waiting for server to start on {server_host}:{server_port}...")
        wait_for_server_ready(server_port, server_host)
        print("Server ready!")

        # Create client to communicate with server using port and host from config
        robot_client = ZMQClientRobot(port=server_port, host=server_host)
    else:  # Direct robot (hardware)
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot

        # Get server configuration (use a different default port for hardware)
        hardware_port = cfg.get("hardware_server_port", 6001)
        hardware_host = "127.0.0.1"

        # Create ZMQ server for the hardware robot
        server = ZMQServerRobot(robot, port=hardware_port, host=hardware_host)
        server_thread = threading.Thread(target=server.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(server)

        # Wait for server to be ready
        print(
            f"Waiting for hardware server to start on {hardware_host}:{hardware_port}..."
        )
        wait_for_server_ready(hardware_port, hardware_host)
        print("Hardware server ready!")

        # Create client to communicate with hardware
        robot_client = ZMQClientRobot(port=hardware_port, host=hardware_host)

    env = RobotEnv(robot_client, control_rate_hz=cfg.get("hz", 30), camera_dict=cameras)

    # Store global variables for cleanup
    global _env, _bimanual, _left_cfg, _right_cfg
    global _agent, _robot, _robot_client, _cameras, _data_saver, _kb_interface
    _env = env
    _bimanual = bimanual
    _left_cfg = left_cfg
    _right_cfg = right_cfg if bimanual else None
    _agent = agent
    _robot = robot
    _robot_client = robot_client
    _cameras = cameras
    _data_saver = data_saver
    _kb_interface = kb_interface

    # Move robot to start_joints position if specified in config
    from gello.utils.launch_utils import move_to_start_position

    if bimanual:
        move_to_start_position(env, bimanual, left_cfg, right_cfg)
    else:
        move_to_start_position(env, bimanual, left_cfg)

    print(
        f"Launching robot: {robot.__class__.__name__}, agent: {agent.__class__.__name__}"
    )
    print(f"Control loop: {cfg.get('hz', 30)} Hz")

    # Optional Rerun take recording: wrap the objects fed into the (unmodified)
    # collection loop. The collector itself was built before the robot went live;
    # wrapping is allocation-only, and per-tick logging is native/cheap.
    if _rerun_collector is not None:
        env = _rerun_collector.wrap_env(env)
        data_saver = _rerun_collector.wrap_data_saver(data_saver)
        kb_interface = _rerun_collector.wrap_kb(kb_interface)

    # from gello.utils.control_utils import SaveInterface, run_control_loop

    # Initialize save interface if requested
    # save_interface = None
    # if args.use_save_interface:
    #     save_interface = SaveInterface(
    #         data_dir=Path(args.left_config_path).parents[1] / "data",
    #         agent_name=agent.__class__.__name__,
    #         expand_user=True,
    #     )

    # # Run main control loop
    # run_control_loop(env, agent, save_interface)

    # Run main control loop
    if bimanual:
        run_control_loop_prior(env, agent, left_cfg=left_cfg, right_cfg=right_cfg, data_saver=data_saver, kb_interface=kb_interface)
    else:
        run_control_loop_prior(env, agent, left_cfg=left_cfg, data_saver=data_saver, kb_interface=kb_interface)

    cleanup()
    run_post_collection_pipeline(left_cfg)
    print("All tasks completed.")


if __name__ == "__main__":
    main()
