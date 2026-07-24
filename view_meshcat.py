"""Headless, browser-based viewer for the G1 env using meshcat.

Unlike ``view.py`` / ``view_all.py`` (which use ``mujoco.viewer`` and need a local
display), this renders in a **web browser** over a websocket, so it works inside a
container / on a headless node -- just forward the printed port. Physics still runs
in ``G1Env`` (CPU or GPU); this only mirrors geometry.

It builds the robot geometry straight from the compiled model (mesh vertices/faces
+ primitives -- no STL file lookups needed) and, each frame, pushes every geom's
world transform, tiled across worlds.

    python view_meshcat.py --device cpu  --worlds 1
    python view_meshcat.py --device cuda:0 --worlds 16 --spacing 1.0

Then open the printed URL (e.g. http://127.0.0.1:7000/static/) in your browser.
In Docker: publish/forward the port and set --zmq-url / open the web url on the host.
"""
from __future__ import annotations

import argparse
import math
import time

import mujoco
import numpy as np

import meshcat
import meshcat.geometry as mg

from g1_env import G1Env


def _rgba_to_hex(rgba) -> int:
    r, g, b = (int(np.clip(c, 0, 1) * 255) for c in rgba[:3])
    return (r << 16) | (g << 8) | b


def build_geom_table(m: mujoco.MjModel):
    """One meshcat geometry per renderable model geom. Returns a list of
    (geom_id, meshcat_geometry, hex_color, opacity). Planes are skipped
    (meshcat draws its own grid)."""
    table = []
    for i in range(m.ngeom):
        gtype = m.geom_type[i]
        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        size = m.geom_size[i]
        rgba = m.geom_rgba[i]
        geom = None
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            did = m.geom_dataid[i]
            va, vn = m.mesh_vertadr[did], m.mesh_vertnum[did]
            fa, fn = m.mesh_faceadr[did], m.mesh_facenum[did]
            verts = m.mesh_vert[va:va + vn].reshape(-1, 3).astype(np.float32)
            faces = m.mesh_face[fa:fa + fn].reshape(-1, 3).astype(np.uint32)
            geom = mg.TriangularMeshGeometry(verts, faces)
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            geom = mg.Box([2 * size[0], 2 * size[1], 2 * size[2]])
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            geom = mg.Sphere(size[0])
        elif gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            geom = mg.Cylinder(2 * size[1], size[0])  # (height, radius)
        else:
            continue
        table.append((i, geom, _rgba_to_hex(rgba), float(rgba[3])))
    return table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--worlds", type=int, default=1)
    p.add_argument("--spacing", type=float, default=1.0)
    p.add_argument("--substeps", type=int, default=1)
    p.add_argument("--zmq-url", type=str, default=None,
                   help="connect to an existing meshcat-server (else start one)")
    p.add_argument("--fps", type=float, default=30.0, help="max display update rate")
    p.add_argument("--steps", type=int, default=0, help="0 = run until Ctrl-C")
    args = p.parse_args()

    backend = "numpy" if args.device.startswith("cpu") else "warp"
    env = G1Env(num_worlds=args.worlds, cameras=[], n_substeps=args.substeps,
                backend=backend, device=args.device)
    print(env)
    env.reset()
    act = (np.zeros((env.num_worlds, env.nu), np.float32) if env.is_cpu
           else __import__("warp").zeros((env.num_worlds, env.nu),
                                         dtype=__import__("warp").float32))

    vis = meshcat.Visualizer(zmq_url=args.zmq_url) if args.zmq_url \
        else meshcat.Visualizer()
    print(f"\n>>> open meshcat in your browser:\n    {vis.url()}\n", flush=True)

    table = build_geom_table(env.mjm)
    cols = int(math.ceil(math.sqrt(args.worlds)))
    grid = np.zeros((args.worlds, 3), np.float32)
    for w in range(args.worlds):
        grid[w] = [(w % cols) * args.spacing, (w // cols) * args.spacing, 0.0]

    # instantiate geometry: one copy per (world, geom)
    for w in range(args.worlds):
        for gid, geom, color, opacity in table:
            mat = mg.MeshLambertMaterial(color=color, opacity=opacity,
                                         transparent=opacity < 1.0)
            vis[f"w{w}/g{gid}"].set_object(geom, mat)

    # buffer for pulling one world's state back from the GPU
    scratch = env.mjd if not env.is_cpu else None

    def world_data(w):
        if env.is_cpu:
            return env._datas[w]
        import mujoco_warp as mjw
        mjw.get_data_into(scratch, env.mjm, env.d, world_id=w)
        mujoco.mj_forward(env.mjm, scratch)
        return scratch

    def push():
        for w in range(args.worlds):
            d = world_data(w)
            off = grid[w]
            for gid, *_ in table:
                T = np.eye(4)
                T[:3, :3] = d.geom_xmat[gid].reshape(3, 3)
                T[:3, 3] = d.geom_xpos[gid] + off
                vis[f"w{w}/g{gid}"].set_transform(T)

    push()
    period = 1.0 / args.fps
    k = 0
    try:
        while args.steps == 0 or k < args.steps:
            t0 = time.perf_counter()
            env.step(act)
            push()
            k += 1
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
