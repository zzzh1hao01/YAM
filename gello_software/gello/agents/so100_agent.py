"""SO100 leader arm agent for YAM teleoperation.

Reads a passive SO100 leader arm (6x Feetech STS3215 servos: 5 joints + a
gripper trigger) over serial via ``scservo_sdk`` and retargets it into a 7-D
absolute YAM joint command ``[j1..j6, gripper]``. One instance drives one arm;
compose two with :class:`gello.agents.agent.BimanualAgent` for the bimanual
rig, exactly how ``GelloAgent`` is wired today.

Retargeting is joint-space (plan option 1): each of the 5 SO100 joints maps to
one YAM joint via ``sign * scale * (leader_ticks - offset) + home``; the
remaining YAM wrist joint is pinned to a fixed value. The gripper trigger is
mapped linearly from its calibrated open/closed extremes onto a configured YAM
gripper range.

Calibration comes from a JSON file produced by
``scripts/calibrate_so100_leader.py`` (native format, see
:func:`load_so100_calibration`), or from a lerobot-style SO100 calibration
file (per-motor ``homing_offset`` / ``range_min`` / ``range_max``) detected by
its keys.

The ``scservo_sdk`` import is deferred to construction time so this module
imports cleanly without the ``feetech-servo-sdk`` package installed. Do NOT
import the local ``lerobot/`` checkout here; it is conversion-side only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from gello.agents.agent import Agent

# Feetech STS3215 control-table addresses.
ADDR_TORQUE_ENABLE = 40
ADDR_PRESENT_POSITION = 56

# STS3215 encoder resolution: 4096 ticks per revolution.
TICKS_PER_REV = 4096
TICKS_TO_RAD = 2.0 * np.pi / TICKS_PER_REV
# Encoder mid-point; lerobot homing offsets center the range here.
TICKS_CENTER = 2048

# Canonical SO100 motor order used by lerobot calibration files. Index 0-4 are
# the arm joints, index 5 is the gripper trigger.
LEROBOT_SO100_MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Native calibration format tag written by scripts/calibrate_so100_leader.py.
CALIBRATION_FORMAT = "so100_yam_v1"


def _import_scservo_sdk():
    """Import scservo_sdk lazily with an actionable error message."""
    try:
        import scservo_sdk
    except ImportError as e:
        raise ImportError(
            "SO100LeaderAgent requires the 'scservo_sdk' module from the "
            "'feetech-servo-sdk' package. Install it with:\n"
            "    pip install feetech-servo-sdk\n"
            "(The leader bus is read directly over serial; do not import the "
            "local lerobot/ checkout for this.)"
        ) from e
    return scservo_sdk


class FeetechBus:
    """Minimal read-only serial bus wrapper for Feetech STS3215 servos.

    Only implements what the SO100 leader needs: open the port, disable
    torque (the leader is passive and moved by hand), and read present
    positions in raw encoder ticks.
    """

    def __init__(
        self,
        port: str,
        servo_ids: Sequence[int],
        baudrate: int = 1_000_000,
    ):
        scs = _import_scservo_sdk()
        self._scs = scs
        self._servo_ids = [int(i) for i in servo_ids]
        self._port = port
        self._port_handler = scs.PortHandler(port)
        # Protocol end 0 is the STS/SMS servo protocol.
        self._packet_handler = scs.PacketHandler(0)
        if not self._port_handler.openPort():
            raise RuntimeError(f"Failed to open serial port: {port}")
        if not self._port_handler.setBaudRate(int(baudrate)):
            self._port_handler.closePort()
            raise RuntimeError(f"Failed to set baudrate {baudrate} on {port}")

    def set_torque(self, enabled: bool) -> None:
        """Enable/disable torque on every servo on the bus."""
        value = 1 if enabled else 0
        for servo_id in self._servo_ids:
            comm, error = self._packet_handler.write1ByteTxRx(
                self._port_handler, servo_id, ADDR_TORQUE_ENABLE, value
            )
            if comm != self._scs.COMM_SUCCESS or error != 0:
                raise RuntimeError(
                    f"Failed to set torque={value} on servo {servo_id} "
                    f"({self._port}): comm={comm}, error={error}"
                )

    def read_positions(self, retries: int = 2) -> np.ndarray:
        """Read present positions for all servos, in raw encoder ticks.

        Returns:
            (len(servo_ids),) float64 array of raw ticks (0..4095, may exceed
            one turn on multi-turn joints).
        """
        positions = np.zeros(len(self._servo_ids), dtype=np.float64)
        for i, servo_id in enumerate(self._servo_ids):
            last_comm, last_error = None, None
            for _ in range(retries + 1):
                pos, comm, error = self._packet_handler.read2ByteTxRx(
                    self._port_handler, servo_id, ADDR_PRESENT_POSITION
                )
                if comm == self._scs.COMM_SUCCESS and error == 0:
                    positions[i] = float(pos)
                    break
                last_comm, last_error = comm, error
            else:
                raise RuntimeError(
                    f"Failed to read position of servo {servo_id} "
                    f"({self._port}): comm={last_comm}, error={last_error}"
                )
        return positions

    def close(self) -> None:
        try:
            self._port_handler.closePort()
        except Exception:
            pass


def _resolve_calibration_path(path: str) -> Path:
    """Resolve a calibration path relative to cwd or the gello_software root."""
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate
    # gello_software root (this file is gello_software/gello/agents/so100_agent.py).
    repo_root = Path(__file__).resolve().parents[2]
    fallback = repo_root / path
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"SO100 calibration file not found: {path} "
        f"(also tried {fallback}). Run scripts/calibrate_so100_leader.py first."
    )


def _is_lerobot_calibration(data: Dict[str, Any]) -> bool:
    """Detect a lerobot-style per-motor calibration dict by its keys."""
    if not isinstance(data, dict) or not data:
        return False
    values = list(data.values())
    return all(
        isinstance(v, dict) and "homing_offset" in v and "range_min" in v
        for v in values
    )


def load_so100_calibration(path: str) -> Dict[str, Any]:
    """Load an SO100 leader calibration file into a normalized dict.

    Supports two formats, detected by keys:

    1. Native (written by ``scripts/calibrate_so100_leader.py``)::

        {
          "format": "so100_yam_v1",
          "arm": "left",
          "joints": [                       # exactly 5, in servo order
            {"servo_id": 1, "sign": 1.0, "scale": 0.001534, "offset": 2048.0},
            ...
          ],
          "home": [0.0, ...],               # 6-D YAM home pose (radians)
          "gripper": {"servo_id": 6, "open": 2100.0, "closed": 3300.0}
        }

       ``offset`` is in raw leader ticks, ``scale`` in rad/tick, and the
       gripper ``open``/``closed`` values are raw trigger ticks.

    2. lerobot-style SO100 calibration (per-motor dict with ``homing_offset``
       and ``range_min``/``range_max``). Joint offsets are derived from the
       homing offsets (raw tick at center = 2048 - homing_offset), sign from
       ``drive_mode``, and scale defaults to one encoder tick in radians.
       The YAM ``home`` pose defaults to zeros; gripper extremes come from
       the gripper motor's range (open=range_min, closed=range_max).

    Returns:
        dict with keys ``signs`` (5,), ``scales`` (5,), ``offsets`` (5,)
        float64 arrays, ``home`` (6,) float64 array, ``gripper_open`` and
        ``gripper_closed`` floats.
    """
    resolved = _resolve_calibration_path(path)
    with open(resolved) as f:
        data = json.load(f)

    if isinstance(data, dict) and data.get("format") == CALIBRATION_FORMAT:
        joints = data["joints"]
        if len(joints) != 5:
            raise ValueError(
                f"{resolved}: expected 5 joint entries, got {len(joints)}"
            )
        home = np.asarray(data["home"], dtype=np.float64)
        if home.shape != (6,):
            raise ValueError(f"{resolved}: 'home' must have 6 values")
        return {
            "signs": np.array([j["sign"] for j in joints], dtype=np.float64),
            "scales": np.array([j["scale"] for j in joints], dtype=np.float64),
            "offsets": np.array([j["offset"] for j in joints], dtype=np.float64),
            "home": home,
            "gripper_open": float(data["gripper"]["open"]),
            "gripper_closed": float(data["gripper"]["closed"]),
            # Correspondence recorded at calibration time (provenance only).
            "yam_joint_indices": data.get("yam_joint_indices"),
        }

    if _is_lerobot_calibration(data):
        # Order motors canonically; fall back to their 'id' field for
        # non-standard names.
        names = [n for n in LEROBOT_SO100_MOTOR_ORDER if n in data]
        if len(names) != len(data):
            names = sorted(data, key=lambda n: data[n].get("id", 0))
        if len(names) != 6:
            raise ValueError(
                f"{resolved}: lerobot calibration must describe 6 motors "
                f"(5 joints + gripper), got {len(names)}"
            )
        joint_names, gripper_name = names[:5], names[5]
        signs = np.array(
            [-1.0 if data[n].get("drive_mode", 0) else 1.0 for n in joint_names]
        )
        offsets = np.array(
            [TICKS_CENTER - float(data[n]["homing_offset"]) for n in joint_names]
        )
        return {
            "signs": signs,
            "scales": np.full(5, TICKS_TO_RAD, dtype=np.float64),
            "offsets": offsets,
            "home": np.zeros(6, dtype=np.float64),
            "gripper_open": float(data[gripper_name]["range_min"]),
            "gripper_closed": float(data[gripper_name]["range_max"]),
            "yam_joint_indices": None,
        }

    raise ValueError(
        f"Unrecognized SO100 calibration format in {resolved}. Expected the "
        f"native '{CALIBRATION_FORMAT}' format or a lerobot-style per-motor "
        "calibration (homing_offset / range_min / range_max keys)."
    )


class SO100LeaderAgent(Agent):
    """Agent that turns a passive SO100 leader arm into YAM joint targets.

    Output of :meth:`act` is a 7-D absolute YAM joint target
    ``[j1..j6, gripper]``; ``BimanualAgent`` concatenates two of these into
    the 14-D action the env/data pipeline expects.
    """

    def __init__(
        self,
        port: str,
        calibration_path: str,
        yam_joint_indices: Sequence[int] = (0, 1, 2, 3, 4),
        pinned_joint_index: int = 5,
        pinned_joint_value: float = 0.0,
        gripper_range: Sequence[float] = (0.0, 1.0),
        ema_alpha: float = 0.4,
        max_joint_delta: float = 0.1,
        servo_ids: Sequence[int] = (1, 2, 3, 4, 5, 6),
        baudrate: int = 1_000_000,
        start_joints: Optional[Sequence[float]] = None,
    ):
        """
        Args:
            port: Serial port of the leader bus (use /dev/serial/by-id/...).
            calibration_path: JSON produced by scripts/calibrate_so100_leader.py
                (or a lerobot-style SO100 calibration; see
                load_so100_calibration).
            yam_joint_indices: For each of the 5 SO100 joints (in servo order),
                the YAM joint index (0-5) it drives. Settled during rig
                calibration; do not assume the identity mapping is correct.
            pinned_joint_index: The YAM joint (0-5) not driven by the leader.
            pinned_joint_value: Fixed value (radians) held on the pinned joint.
            gripper_range: (open, closed) YAM gripper command values that the
                trigger extremes map onto linearly.
            ema_alpha: EMA smoothing factor in (0, 1]; 1.0 disables smoothing.
            max_joint_delta: Per-act() clamp on each arm joint's change
                (radians). <= 0 disables the clamp. Not applied to the gripper.
            servo_ids: The 6 servo ids on the leader bus, 5 joints then the
                gripper trigger, matching the calibration file's order.
            baudrate: Serial baudrate of the leader bus.
            start_joints: Accepted for launcher compatibility (the env uses
                the config's start_joints to home the follower); unused here.
        """
        yam_joint_indices = [int(i) for i in yam_joint_indices]
        if len(yam_joint_indices) != 5:
            raise ValueError(
                f"yam_joint_indices must have 5 entries, got {yam_joint_indices}"
            )
        covered = set(yam_joint_indices) | {int(pinned_joint_index)}
        if covered != set(range(6)):
            raise ValueError(
                "yam_joint_indices + pinned_joint_index must cover YAM joints "
                f"0-5 exactly once, got {yam_joint_indices} + "
                f"{pinned_joint_index}"
            )
        if len(servo_ids) != 6:
            raise ValueError(f"servo_ids must have 6 entries, got {servo_ids}")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")

        self._yam_joint_indices = np.array(yam_joint_indices, dtype=np.int64)
        self._pinned_joint_index = int(pinned_joint_index)
        self._pinned_joint_value = float(pinned_joint_value)
        self._gripper_range = (float(gripper_range[0]), float(gripper_range[1]))
        self._ema_alpha = float(ema_alpha)
        self._max_joint_delta = float(max_joint_delta)
        self._start_joints = (
            np.asarray(start_joints, dtype=np.float64)
            if start_joints is not None
            else None
        )

        calib = load_so100_calibration(calibration_path)
        self._signs = calib["signs"]
        self._scales = calib["scales"]
        self._offsets = calib["offsets"]
        self._home = calib["home"]
        self._gripper_open = calib["gripper_open"]
        self._gripper_closed = calib["gripper_closed"]
        if self._gripper_closed == self._gripper_open:
            raise ValueError(
                "Calibration gripper open/closed extremes are identical; "
                "re-run scripts/calibrate_so100_leader.py."
            )
        recorded = calib.get("yam_joint_indices")
        if recorded is not None and list(recorded) != yam_joint_indices:
            print(
                f"WARNING: SO100 calibration file was recorded with "
                f"yam_joint_indices={list(recorded)} but this agent is "
                f"configured with {yam_joint_indices}; retargeted joints will "
                "not match the calibration poses."
            )

        # Open the bus and make the leader passive.
        self._bus = FeetechBus(port=port, servo_ids=servo_ids, baudrate=baudrate)
        self._bus.set_torque(False)

        self._last_target: Optional[np.ndarray] = None

    def num_dofs(self) -> int:
        return 7

    def _retarget(self, ticks: np.ndarray) -> np.ndarray:
        """Map raw leader ticks (6,) to an unsmoothed 7-D YAM target."""
        joints = self._home.copy()
        mapped = (
            self._signs * self._scales * (ticks[:5] - self._offsets)
            + self._home[self._yam_joint_indices]
        )
        joints[self._yam_joint_indices] = mapped
        joints[self._pinned_joint_index] = self._pinned_joint_value

        frac = (ticks[5] - self._gripper_open) / (
            self._gripper_closed - self._gripper_open
        )
        frac = min(max(0.0, frac), 1.0)
        gripper = self._gripper_range[0] + frac * (
            self._gripper_range[1] - self._gripper_range[0]
        )
        return np.concatenate([joints, [gripper]])

    def _smooth(self, target: np.ndarray) -> np.ndarray:
        """EMA filter + per-step joint-delta clamp against the last output."""
        if self._last_target is None:
            self._last_target = target
            return target
        smoothed = (
            self._ema_alpha * target + (1.0 - self._ema_alpha) * self._last_target
        )
        if self._max_joint_delta > 0:
            delta = smoothed[:6] - self._last_target[:6]
            np.clip(delta, -self._max_joint_delta, self._max_joint_delta, out=delta)
            smoothed[:6] = self._last_target[:6] + delta
        self._last_target = smoothed
        return smoothed

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        ticks = self._bus.read_positions()
        return self._smooth(self._retarget(ticks))

    def raw_ticks(self) -> np.ndarray:
        """Read raw leader positions (ticks), for calibration/debug readouts."""
        return self._bus.read_positions()

    def close(self) -> None:
        """Release the serial port (leader torque stays off)."""
        self._bus.close()
