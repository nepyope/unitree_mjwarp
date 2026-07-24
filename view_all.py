"""Render ALL simulated worlds at once, tiled on a grid (Isaac-Gym style).

Physics runs batched on the GPU via ``G1Env`` (num_worlds copies). For display we
build a single MuJoCo model containing N copies of the robot laid out on a grid,
and each frame copy every world's qpos into its copy, then show them together in
one ``mujoco.viewer`` window.

Needs a local display. Keep N modest (tens to low hundreds) -- the viewer renders
all copies on the CPU/GL, so it's much heavier than the headless batch renderer.

    python view_all.py --worlds 36 --spacing 1.0 --realtime
"""
from __future__ import annotations

import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np
import warp as wp

import mujoco_warp as mjw

from g1_env import G1Env, DEFAULT_G1_XML

# The robot-only MJCF that the scene includes (no floor/skybox of its own).
ROBOT_XML = DEFAULT_G1_XML.replace("scene_29dof.xml", "g1_29dof_no_hand.xml")


def build_display_model(n: int, spacing: float, robot_xml: str):
    """Compile one model with `n` grid copies of the robot. Returns
    (mjModel, per_instance_nq, per_instance_nv)."""
    cols = int(math.ceil(math.sqrt(n)))
    disp = mujoco.MjSpec()
    disp.add_texture(
        name="groundtex", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.15, 0.2], width=300, height=300)
    disp.add_material(name="groundplane", textures=["", "groundtex"],
                      texrepeat=[5, 5], texuniform=True)
    floor = disp.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.material = "groundplane"
    light = disp.worldbody.add_light()
    light.pos = [0.0, 0.0, 4.0]
    light.dir = [0.0, 0.0, -1.0]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    nq1 = nv1 = None
    for i in range(n):
        child = mujoco.MjSpec.from_file(robot_xml)   # fresh copy each time
        if nq1 is None:
            cm = child.compile()
            nq1, nv1 = cm.nq, cm.nv
            child = mujoco.MjSpec.from_file(robot_xml)
        gx = (i % cols) * spacing
        gy = (i // cols) * spacing
        frame = disp.worldbody.add_frame()
        frame.pos = [gx, gy, 0.0]
        disp.attach(child, prefix=f"w{i}_", frame=frame)

    return disp.compile(), nq1, nv1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=36)
    p.add_argument("--spacing", type=float, default=1.0, help="grid spacing (m)")
    p.add_argument("--substeps", type=int, default=1)
    p.add_argument("--realtime", action="store_true")
    args = p.parse_args()

    env = G1Env(num_worlds=args.worlds, cameras=[], n_substeps=args.substeps,
                backend="warp")
    print(env)
    env.reset()
    act = wp.zeros((env.num_worlds, env.nu), dtype=wp.float32)

    disp_m, nq1, nv1 = build_display_model(args.worlds, args.spacing, ROBOT_XML)
    disp_d = mujoco.MjData(disp_m)
    assert disp_m.nq == args.worlds * nq1, (disp_m.nq, args.worlds, nq1)
    assert nq1 == env.nq, (nq1, env.nq)
    print(f"display model: {args.worlds} copies, nq/copy={nq1}, total nq={disp_m.nq}")

    # A free-floating base ignores the attach-frame offset (its qpos is the
    # absolute world pose), so we tile by adding a per-copy xy offset to the
    # base position directly.
    cols = int(math.ceil(math.sqrt(args.worlds)))
    off = np.zeros((args.worlds, nq1), np.float32)
    for i in range(args.worlds):
        off[i, 0] = (i % cols) * args.spacing
        off[i, 1] = (i // cols) * args.spacing

    def sync_poses():
        q = env.d.qpos.numpy() + off              # (N, nq1), tiled on grid
        disp_d.qpos[:] = q.reshape(-1)            # instances are contiguous
        mujoco.mj_forward(disp_m, disp_d)

    sync_poses()
    with mujoco.viewer.launch_passive(disp_m, disp_d) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()
            env.step(act)
            sync_poses()
            viewer.sync()
            if args.realtime:
                time.sleep(max(0.0, env.dt - (time.perf_counter() - t0)))


if __name__ == "__main__":
    main()
