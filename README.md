# unitree_mjwarp

A standalone, **GPU-vectorized Unitree G1 environment on [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)** (`mjwarp`).

It runs thousands of copies of the robot in parallel on a single GPU and can
return **batched camera images (RGB + depth)** via mjwarp's GPU batch renderer,
in addition to proprioceptive state. It's deliberately task-agnostic — just
`reset` / `step` / `render` over raw physics state — so it can back RL/WBC work
*and* general Unitree experiments.

The G1 model (`assets/g1/`, XML + STL meshes) is bundled, so nothing external is
needed beyond the Python deps.

## Install

Requires an NVIDIA GPU. The bundled `warp-lang` ships CUDA 12.9 and supports
Blackwell (sm_120, e.g. RTX 5090) — you just need a recent driver (≥ 570).

```bash
python -m venv .venv && source .venv/bin/activate     # or a conda env, python 3.10–3.12
pip install -r requirements.txt

python -c "import warp as wp; wp.init()"              # should list your GPU as cuda:0
```

## Run the smoke test

```bash
# physics only — throughput check
python smoke_test.py --worlds 2048 --steps 200

# with a first-person head cam + third-person chase cam, dump world-0 RGB
python smoke_test.py --worlds 512 --steps 200 --cameras head track --save-png head0.png
```

## Use it as a library

```python
import warp as wp
from g1_env import G1Env, CameraSpec

env = G1Env(num_worlds=4096, cameras=["head"], backend="warp")  # "warp"|"torch"|"numpy"
obs = env.reset()                       # {"qpos","qvel","sensordata","time"}
act = wp.zeros((env.num_worlds, env.nu), dtype=wp.float32)
for _ in range(100):
    obs = env.step(act)                 # torque control on the actuated joints
imgs = env.render()                     # {"head": {"rgb": (N,H,W,3), "depth": (N,H,W)}}
```

### Backends
- `warp` — mjwarp `wp.array`, zero-copy on-GPU (fastest; for RL).
- `torch` — `torch.Tensor` sharing the same GPU memory (`pip install torch`).
- `numpy` — host copies (slow; for inspection / a request-response server).

### Cameras
Pass preset names (`"head"`, `"track"`) or `CameraSpec(...)` for custom
mount/pose/resolution/FOV. Cameras dominate VRAM: budget roughly
`worlds × H × W × cams × ~16 bytes` for the render buffers. On a 24 GB card start
at `--worlds 512` with 224×224 cameras and scale up while watching `nvidia-smi`.

## Notes
- Actuators are the torque motors defined in `assets/g1/g1.xml` (legs + waist yaw
  + arms; hands are disabled). `env.actuator_names()` lists them in order.
- `qpos`/`qvel` include the floating base first (`qpos[:7] = pos3 + quat4`,
  `qvel[:6] = linvel3 + angvel3`), then the actuated joints.
- Override the model with `G1_XML=/path/to/model.xml` (e.g. to use
  `assets/g1/g1_gear_wbc.xml` or your own MJCF).

## Layout
```
g1_env.py        # the vectorized env
smoke_test.py    # batched rollout + camera render + throughput benchmark
assets/g1/       # bundled G1 MJCF + meshes
```
