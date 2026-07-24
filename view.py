"""Interactive ("head mode") viewer for the vectorized mjwarp G1 env.

The physics still runs batched on the GPU (``num_worlds`` copies); this just
mirrors ONE world into a live MuJoCo window each frame via ``mjw.get_data_into``,
so you can watch it, orbit with the mouse, etc. Needs a local display (run it on
your workstation, not a headless node).

    python view.py --worlds 64 --world 0

Keys: standard mujoco.viewer controls. Ctrl-C or close the window to quit.
"""
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import warp as wp

import mujoco_warp as mjw

from g1_env import G1Env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=64,
                   help="how many worlds to simulate on the GPU (only one is shown)")
    p.add_argument("--world", type=int, default=0, help="which world to mirror")
    p.add_argument("--substeps", type=int, default=1)
    p.add_argument("--realtime", action="store_true",
                   help="sleep to run at wall-clock dt (else run as fast as possible)")
    args = p.parse_args()

    env = G1Env(num_worlds=args.worlds, cameras=[], n_substeps=args.substeps,
                backend="warp")
    print(env)
    env.reset()
    act = wp.zeros((env.num_worlds, env.nu), dtype=wp.float32)

    wid = max(0, min(args.world, env.num_worlds - 1))
    # seed the viewer's MjData with world `wid` so the first frame is correct.
    mjw.get_data_into(env.mjd, env.mjm, env.d, world_id=wid)

    with mujoco.viewer.launch_passive(env.mjm, env.mjd) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()
            env.step(act)
            mjw.get_data_into(env.mjd, env.mjm, env.d, world_id=wid)
            viewer.sync()
            if args.realtime:
                time.sleep(max(0.0, env.dt - (time.perf_counter() - t0)))


if __name__ == "__main__":
    main()
