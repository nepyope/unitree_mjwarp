"""Smoke test for the standalone mjwarp G1 env.

Runs a batched rollout on the GPU, prints observation/camera shapes, and
measures throughput (env-steps/s and world-steps/s). Also dumps one head-cam
frame to PNG so we can eyeball the renderer.

    python smoke_test.py --worlds 4096 --steps 200 --cameras head track
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp

from g1_env import G1Env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=2048)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--substeps", type=int, default=1)
    p.add_argument("--cameras", nargs="*", default=[])
    p.add_argument("--save-png", type=str, default=None,
                   help="save world-0 RGB of the first listed camera here")
    args = p.parse_args()

    env = G1Env(num_worlds=args.worlds, cameras=args.cameras,
                n_substeps=args.substeps, backend="warp")
    print(env)
    print("actuators:", env.actuator_names())

    obs = env.reset()
    print("--- obs shapes ---")
    for k, v in obs.items():
        print(f"  {k:12s} {tuple(v.shape)} {v.dtype}")

    # zero-torque action of the right shape.
    act = wp.zeros((env.num_worlds, env.nu), dtype=wp.float32)

    # warmup (JIT kernels compile on first call)
    env.step(act)
    if args.cameras:
        env.render()
    wp.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.steps):
        env.step(act)
    wp.synchronize()
    dt = time.perf_counter() - t0
    sps = args.steps / dt
    print(f"\nphysics: {args.steps} steps x {args.worlds} worlds in {dt:.3f}s")
    print(f"  {sps:,.0f} env-steps/s  ->  {sps * args.worlds:,.0f} world-steps/s")

    if args.cameras:
        env.render()
        wp.synchronize()
        t0 = time.perf_counter()
        R = 30
        for _ in range(R):
            imgs = env.render()
        wp.synchronize()
        dt = time.perf_counter() - t0
        print(f"\nrender: {R} frames x {args.worlds} worlds x {len(args.cameras)} cam "
              f"in {dt:.3f}s -> {R / dt:,.1f} render-calls/s")
        for name, e in imgs.items():
            for kind, arr in e.items():
                print(f"  {name}/{kind:5s} {tuple(arr.shape)} {arr.dtype}")

        if args.save_png:
            name = args.cameras[0]
            rgb = env.render()[name]["rgb"].numpy()[0]  # (H,W,3) vec3f -> float
            img = np.clip(rgb, 0, 1) if rgb.max() <= 1.0 else np.clip(rgb / 255.0, 0, 1)
            try:
                from PIL import Image
                Image.fromarray((img * 255).astype(np.uint8)).save(args.save_png)
                print(f"saved {args.save_png}")
            except ImportError:
                np.save(args.save_png + ".npy", img)
                print(f"PIL missing; saved {args.save_png}.npy")

    print("\nOK")


if __name__ == "__main__":
    main()
