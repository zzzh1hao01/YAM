"""Interactive per-arm calibration for an SO100 leader arm driving a YAM arm.

Produces ``configs/so100_calibration_{left,right}.json`` in exactly the format
:func:`gello.agents.so100_agent.load_so100_calibration` expects
(``format: "so100_yam_v1"``).

Flow (leader is passive the whole time; this script NEVER energizes or
commands any motor):

1. Torque is disabled on the leader bus at startup.
2. The operator poses the leader to match the YAM *home* reference pose;
   all 5 joint positions are sampled at once (this fixes each joint's
   ``offset``).
3. For each joint in turn, the operator moves ONLY that joint to match a
   second, displaced YAM reference value; ``sign``/``scale`` are solved from
   the two samples.
4. Gripper trigger open/closed extremes are captured.
5. The JSON is written, then a live dry-run readout loop prints the
   leader -> YAM 7-D targets (via a real ``SO100LeaderAgent`` loading the
   file just written) until Ctrl-C. Nothing is commanded to any motor.

YAM-side reference values come from the embedded table below (edit it to
match poses you can visually reproduce on the follower) or from the
``--home-pose`` / ``--displaced-pose`` CLI args.

Run from ``gello_software/``:

    python scripts/calibrate_so100_leader.py --arm left \\
        --port /dev/serial/by-id/usb-...
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tyro

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

from gello.agents.so100_agent import (  # noqa: E402
    CALIBRATION_FORMAT,
    FeetechBus,
    SO100LeaderAgent,
)

# ---------------------------------------------------------------------------
# YAM-side reference poses (radians, 6 joints). EDIT ME to match poses you can
# reliably reproduce by eye on the follower arm (e.g. drive the follower there
# with experiments/reset_to_home.py variants and photograph it).
#
# HOME_POSE is the pose the leader is matched to in step 2 (all joints at
# once). DISPLACED_POSE[k] is the value YAM joint k is moved to (from home)
# when calibrating the SO100 joint that drives it; only that joint moves.
# ---------------------------------------------------------------------------
YAM_REFERENCE_POSES = {
    "left": {
        "home": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "displaced": (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    },
    "right": {
        "home": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "displaced": (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    },
}

# Minimum leader travel (ticks) between the two reference samples for a
# usable sign/scale solve. 4096 ticks = one full turn.
MIN_TICK_TRAVEL = 20.0


@dataclass
class Args:
    port: str
    """Serial port of the SO100 leader bus (use /dev/serial/by-id/...)."""

    arm: str = "left"
    """Which arm this leader drives: "left" or "right"."""

    output_path: Optional[str] = None
    """Output JSON path. Defaults to configs/so100_calibration_{arm}.json."""

    servo_ids: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    """The 6 leader servo ids: 5 joints then the gripper trigger."""

    yam_joint_indices: Tuple[int, ...] = (0, 1, 2, 3, 4)
    """YAM joint index (0-5) driven by each SO100 joint, in servo order."""

    baudrate: int = 1_000_000
    """Serial baudrate of the leader bus."""

    home_pose: Optional[Tuple[float, ...]] = None
    """6 YAM joint values (radians) for the home reference pose. Overrides
    the embedded YAM_REFERENCE_POSES table."""

    displaced_pose: Optional[Tuple[float, ...]] = None
    """6 YAM joint values (radians) for the displaced reference pose.
    Overrides the embedded YAM_REFERENCE_POSES table."""

    num_samples: int = 20
    """Bus reads averaged per captured pose."""

    dry_run_hz: float = 10.0
    """Print rate of the final live readout loop."""

    def __post_init__(self):
        assert self.arm in ("left", "right"), f"arm must be left/right: {self.arm}"
        assert len(self.servo_ids) == 6, "need 6 servo ids (5 joints + gripper)"
        assert len(self.yam_joint_indices) == 5, "need 5 YAM joint indices"
        for pose in (self.home_pose, self.displaced_pose):
            assert pose is None or len(pose) == 6, "reference poses need 6 values"


def sample_positions(bus: FeetechBus, num_samples: int) -> np.ndarray:
    """Average several bus reads to reduce encoder noise."""
    samples = []
    for _ in range(num_samples):
        samples.append(bus.read_positions())
        time.sleep(0.01)
    return np.mean(samples, axis=0)


def prompt_and_sample(bus: FeetechBus, message: str, num_samples: int) -> np.ndarray:
    input(f"\n{message}\n  ... then press Enter to capture: ")
    ticks = sample_positions(bus, num_samples)
    print(f"  captured ticks: {np.round(ticks, 1)}")
    return ticks


def solve_joint(
    ticks_home: float,
    ticks_displaced: float,
    yam_home: float,
    yam_displaced: float,
) -> Tuple[float, float, float]:
    """Solve (sign, scale, offset) for one joint from two reference samples.

    The retargeting model is ``yam = sign * scale * (ticks - offset) + home``,
    so the home sample pins ``offset`` and the displaced sample gives the
    signed slope.
    """
    tick_travel = ticks_displaced - ticks_home
    if abs(tick_travel) < MIN_TICK_TRAVEL:
        raise ValueError(
            f"Leader barely moved between reference poses "
            f"({tick_travel:.1f} ticks). Re-run and move the joint further."
        )
    slope = (yam_displaced - yam_home) / tick_travel
    sign = 1.0 if slope >= 0 else -1.0
    return sign, abs(slope), ticks_home


def dry_run_readout(args: Args, calibration_path: Path) -> None:
    """Live leader -> YAM target readout. Reads only; commands nothing."""
    print("\n--- Dry-run readout (Ctrl-C to stop; no motors are commanded) ---")
    agent = SO100LeaderAgent(
        port=args.port,
        calibration_path=str(calibration_path),
        yam_joint_indices=args.yam_joint_indices,
        servo_ids=args.servo_ids,
        baudrate=args.baudrate,
    )
    period = 1.0 / args.dry_run_hz
    try:
        while True:
            target = agent.act({})
            joints = ", ".join(f"{x:+.3f}" for x in target[:6])
            print(f"\ryam target: [{joints}]  gripper: {target[6]:.3f}   ", end="")
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nDry run stopped.")
    finally:
        agent.close()


def calibrate(args: Args) -> None:
    refs = YAM_REFERENCE_POSES[args.arm]
    home_pose = np.asarray(
        args.home_pose if args.home_pose is not None else refs["home"],
        dtype=np.float64,
    )
    displaced_pose = np.asarray(
        args.displaced_pose if args.displaced_pose is not None else refs["displaced"],
        dtype=np.float64,
    )

    output_path = Path(
        args.output_path
        if args.output_path is not None
        else Path(__file__).resolve().parents[1]
        / "configs"
        / f"so100_calibration_{args.arm}.json"
    )

    print(f"Calibrating {args.arm} SO100 leader on {args.port}")
    print(f"YAM home reference pose:      {home_pose}")
    print(f"YAM displaced reference pose: {displaced_pose}")

    bus = FeetechBus(port=args.port, servo_ids=args.servo_ids, baudrate=args.baudrate)
    try:
        bus.set_torque(False)
        print("Leader torque disabled (arm is passive).")

        # Warmup reads.
        for _ in range(5):
            bus.read_positions()

        # Step 2: home pose, all joints at once.
        ticks_home = prompt_and_sample(
            bus,
            f"Pose the WHOLE leader arm to match the YAM home pose "
            f"{np.round(home_pose, 3)}",
            args.num_samples,
        )

        # Step 3: one displaced sample per joint.
        joints = []
        for i, yam_idx in enumerate(args.yam_joint_indices):
            servo_id = args.servo_ids[i]
            ticks_displaced = prompt_and_sample(
                bus,
                f"Joint {i + 1}/5 (servo {servo_id} -> YAM joint {yam_idx}): "
                f"move ONLY this joint so YAM joint {yam_idx} would read "
                f"{displaced_pose[yam_idx]:.3f} rad (others stay at home)",
                args.num_samples,
            )
            sign, scale, offset = solve_joint(
                ticks_home[i],
                ticks_displaced[i],
                home_pose[yam_idx],
                displaced_pose[yam_idx],
            )
            print(
                f"  servo {servo_id}: sign={sign:+.0f}, scale={scale:.6f} "
                f"rad/tick, offset={offset:.1f} ticks"
            )
            joints.append(
                {
                    "servo_id": int(servo_id),
                    "yam_joint_index": int(yam_idx),
                    "sign": sign,
                    "scale": scale,
                    "offset": float(offset),
                }
            )

        # Step 4: gripper trigger extremes.
        gripper_id = args.servo_ids[5]
        ticks_open = prompt_and_sample(
            bus, f"Hold the gripper trigger fully OPEN (servo {gripper_id})",
            args.num_samples,
        )
        ticks_closed = prompt_and_sample(
            bus, f"Hold the gripper trigger fully CLOSED (servo {gripper_id})",
            args.num_samples,
        )
        if abs(ticks_closed[5] - ticks_open[5]) < MIN_TICK_TRAVEL:
            raise ValueError(
                "Gripper open/closed extremes are nearly identical; re-run "
                "and squeeze the trigger through its full travel."
            )

        calibration = {
            "format": CALIBRATION_FORMAT,
            "arm": args.arm,
            "port": args.port,
            "servo_ids": list(args.servo_ids),
            "yam_joint_indices": list(args.yam_joint_indices),
            "joints": joints,
            "home": home_pose.tolist(),
            "gripper": {
                "servo_id": int(gripper_id),
                "open": float(ticks_open[5]),
                "closed": float(ticks_closed[5]),
            },
        }
    finally:
        bus.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(calibration, f, indent=2)
        f.write("\n")
    print(f"\nWrote calibration to {output_path}")

    # Step 5: live sanity readout through the real agent + file just written.
    dry_run_readout(args, output_path)


def main(args: Args) -> None:
    calibrate(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
