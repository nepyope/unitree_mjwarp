"""Standalone, vectorized Unitree G1 environment on MuJoCo Warp (mjwarp).

This mirrors the single-instance MuJoCo sim in ``HIW-500-controoler`` but runs
``nworld`` copies of the robot in parallel on the GPU, and optionally returns
batched camera images (RGB + depth) via mjwarp's GPU batch renderer.

It is deliberately task-agnostic: it exposes ``reset`` / ``step`` / ``render``
returning raw proprioceptive state (qpos/qvel/sensors) and camera tensors, so it
can back the GR00T RL work *and* general Unitree experiments. Task-specific
rewards / observation packing live on top of this, not inside it.

Backends for the returned arrays:
  * "warp"  -> mjwarp ``wp.array`` (zero-copy, on-GPU; fastest, for RL)
  * "torch" -> ``torch.Tensor`` sharing the same GPU memory (needs torch)
  * "numpy" -> host numpy copy (slow; for inspection / a request-response server)

Example
-------
    env = G1Env(num_worlds=4096, cameras=["head"], backend="warp")
    obs = env.reset()
    for _ in range(100):
        act = wp.zeros((env.num_worlds, env.nu), dtype=wp.float32)
        obs = env.step(act)
    imgs = env.render()          # {"head": {"rgb": (N,H,W,3), "depth": (N,H,W)}}
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw

_BUNDLED_G1_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "assets", "g1", "scene_29dof.xml")
DEFAULT_G1_XML = os.environ.get("G1_XML", _BUNDLED_G1_XML)


@dataclass
class CameraSpec:
    """A camera to add to the scene. ``mount`` is a body name to attach to
    (first-person); if ``None`` the camera is a world camera that tracks
    ``target`` (third-person). ``pos``/``quat`` are in the mount/world frame."""

    name: str
    width: int = 128
    height: int = 128
    mount: str | None = None
    target: str | None = "pelvis"
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    fovy: float = 58.0


# Convenience aliases. "head"/"track" map to the cameras the lerobot G1 sim
# scene already ships (head_camera, global_view); if a requested camera name is
# already defined in the loaded MJCF we reuse it in place (no re-add), otherwise
# a matching CameraSpec here is added to the scene.
_PRESET_CAMERAS: dict[str, CameraSpec] = {
    "head": CameraSpec(name="head_camera", width=224, height=224),
    "track": CameraSpec(name="global_view", width=256, height=256),
}


@dataclass
class G1EnvConfig:
    xml_path: str = DEFAULT_G1_XML
    num_worlds: int = 1024
    n_substeps: int = 1                 # physics steps per env.step()
    add_floor: bool = True
    backend: str = "warp"               # "warp" | "torch" | "numpy"
    device: str = "cuda:0"
    cameras: list = field(default_factory=list)   # names (presets) or CameraSpec
    render_rgb: bool = True
    render_depth: bool = True
    depth_scale: float = 1.0
    njmax: int | None = 256     # per-world constraint cap (avoids nefc overflow)
    nconmax: int | None = None


def _add_scene_extras(spec: mujoco.MjSpec, cfg: G1EnvConfig,
                      cams: list[CameraSpec]) -> None:
    """Add the requested cameras (and a floor/light only if the model lacks
    them) onto the loaded G1 spec. Many G1 MJCFs already ship a ground plane,
    skybox and light, so we add those idempotently."""
    geom_names = {g.name for g in spec.geoms}
    tex_names = {t.name for t in spec.textures}
    mat_names = {m.name for m in spec.materials}

    if cfg.add_floor and "floor" not in geom_names:
        if "groundtex" not in tex_names:
            spec.add_texture(
                name="groundtex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.15, 0.2],
                width=300, height=300,
            )
        if "groundplane" not in mat_names:
            spec.add_material(
                name="groundplane", textures=["", "groundtex"],
                texrepeat=[5, 5], texuniform=True, reflectance=0.0,
            )
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [0.0, 0.0, 0.05]
        floor.material = "groundplane"

    if not spec.lights:
        light = spec.worldbody.add_light()
        light.pos = [0.0, 0.0, 3.0]
        light.dir = [0.0, 0.0, -1.0]
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    bodies = {b.name for b in spec.bodies}
    existing = {c.name for c in spec.cameras}
    for c in cams:
        if c.name in existing:  # reuse a camera already defined in the MJCF
            continue
        if c.mount is not None and c.mount in bodies:
            body = next(b for b in spec.bodies if b.name == c.mount)
            cam = body.add_camera()
        else:
            cam = spec.worldbody.add_camera()
        cam.name = c.name
        cam.fovy = c.fovy
        cam.pos = list(c.pos)
        cam.quat = list(c.quat)
        if c.mount is None and c.target is not None and c.target in bodies:
            cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
            cam.targetbody = c.target


class G1Env:
    """Vectorized G1 physics env on mjwarp with optional batched cameras."""

    def __init__(self, num_worlds: int = 1024, cameras=None, **kwargs):
        cfg = G1EnvConfig(num_worlds=num_worlds,
                          cameras=list(cameras) if cameras else [], **kwargs)
        self.cfg = cfg
        self.num_worlds = cfg.num_worlds
        self.backend = cfg.backend

        self.device = cfg.device
        if not str(self.device).startswith("cpu"):
            wp.init()

        cams: list[CameraSpec] = []
        for c in cfg.cameras:
            if isinstance(c, CameraSpec):
                cams.append(c)
            elif c in _PRESET_CAMERAS:
                cams.append(_PRESET_CAMERAS[c])
            else:
                raise KeyError(f"unknown camera preset {c!r}; "
                               f"known: {list(_PRESET_CAMERAS)} or pass a CameraSpec")
        self._cams = cams

        spec = mujoco.MjSpec.from_file(cfg.xml_path)
        _add_scene_extras(spec, cfg, cams)
        self.mjm = spec.compile()
        self.mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, self.mjd)

        self.nq = self.mjm.nq
        self.nv = self.mjm.nv
        self.nu = self.mjm.nu
        self._qpos0 = np.array(self.mjm.qpos0, dtype=np.float32)

        # CPU uses the native MuJoCo engine (mj_step) over a list of MjData --
        # the right tool for 1 (or a few) envs. GPU uses mjwarp for vectorized
        # physics + batched rendering. Warp arrays don't apply on CPU, so the
        # "warp" backend falls back to numpy there.
        self.is_cpu = str(self.device).startswith("cpu")
        if self.is_cpu and self.backend == "warp":
            self.backend = "numpy"

        self._rc = None
        self._rgb_out: dict = {}
        self._depth_out: dict = {}
        self._want_render = bool(cams) and (cfg.render_rgb or cfg.render_depth)

        if self.is_cpu:
            self._datas = [mujoco.MjData(self.mjm) for _ in range(cfg.num_worlds)]
            for d in self._datas:
                mujoco.mj_forward(self.mjm, d)
            if self._want_render:
                self._setup_renderer_cpu()
        else:
            with wp.ScopedDevice(self.device):
                self.m = mjw.put_model(self.mjm)
                self.d = mjw.put_data(self.mjm, self.mjd, nworld=cfg.num_worlds,
                                      njmax=cfg.njmax, nconmax=cfg.nconmax)
            if self._want_render:
                self._setup_renderer()

    # -- rendering setup ---------------------------------------------------
    def _setup_renderer(self) -> None:
        cfg = self.cfg
        self._cam_index = {}
        for c in self._cams:
            cid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_CAMERA, c.name)
            if cid < 0:
                raise RuntimeError(f"camera {c.name!r} not found after compile")
            self._cam_index[c.name] = cid

        # The batch renderer indexes every buffer by *model camera id*, so all
        # per-camera config lists must be full length (ncam), not just the
        # cameras we asked for. Inactive cameras get a tiny res and no output.
        ncam = self.mjm.ncam
        res_by_id = [(c.width, c.height) for c in self._cams]  # placeholder default
        cam_res = [(8, 8)] * ncam
        render_rgb = [False] * ncam
        render_depth = [False] * ncam
        active = [False] * ncam
        # resolution requested per camera (default 128 if a camera was requested
        # by name only, i.e. a preset/spec without explicit size still carries one)
        self._cam_res = {}
        for c in self._cams:
            cid = self._cam_index[c.name]
            cam_res[cid] = (c.width, c.height)
            render_rgb[cid] = cfg.render_rgb
            render_depth[cid] = cfg.render_depth
            active[cid] = True
            self._cam_res[c.name] = (c.width, c.height)

        with wp.ScopedDevice(self.device):
            self._rc = mjw.create_render_context(
                self.mjm, nworld=cfg.num_worlds,
                cam_res=cam_res,
                render_rgb=render_rgb, render_depth=render_depth,
                cam_active=active,
            )
            for c in self._cams:
                w, h = self._cam_res[c.name]
                if cfg.render_rgb:
                    self._rgb_out[c.name] = wp.zeros(
                        (cfg.num_worlds, h, w), dtype=wp.vec3f)
                if cfg.render_depth:
                    self._depth_out[c.name] = wp.zeros(
                        (cfg.num_worlds, h, w), dtype=wp.float32)

    def _setup_renderer_cpu(self) -> None:
        """Native-MuJoCo offscreen renderers (one per unique resolution). Needs a
        GL backend; on a headless box set MUJOCO_GL=egl (or osmesa)."""
        self._cam_index = {}
        self._cam_res = {}
        for c in self._cams:
            cid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_CAMERA, c.name)
            if cid < 0:
                raise RuntimeError(f"camera {c.name!r} not found after compile")
            self._cam_index[c.name] = cid
            self._cam_res[c.name] = (c.width, c.height)
        try:
            self._renderers = {}
            for c in self._cams:
                key = (c.height, c.width)
                if key not in self._renderers:
                    self._renderers[key] = mujoco.Renderer(self.mjm, c.height, c.width)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "CPU camera rendering needs a GL context. On a headless machine "
                "set MUJOCO_GL=egl (or osmesa) before importing. "
                f"Underlying error: {e}") from e

    # -- conversion helper -------------------------------------------------
    def _out(self, arr):
        """arr is a wp.array (GPU) or np.ndarray (CPU)."""
        if isinstance(arr, np.ndarray):
            if self.backend == "torch":
                import torch
                return torch.as_tensor(arr)
            return arr
        if self.backend == "warp":
            return arr
        if self.backend == "numpy":
            return arr.numpy()
        if self.backend == "torch":
            return wp.to_torch(arr)
        raise ValueError(self.backend)

    # -- core API ----------------------------------------------------------
    def reset(self, qpos: np.ndarray | None = None) -> dict:
        """Reset all worlds to the model reference pose (or a supplied qpos)."""
        if self.is_cpu:
            q0 = self._qpos0 if qpos is None else np.asarray(qpos, np.float32)
            q0 = np.broadcast_to(q0, (self.num_worlds, self.nq))
            for i, d in enumerate(self._datas):
                mujoco.mj_resetData(self.mjm, d)
                d.qpos[:] = q0[i]
                d.qvel[:] = 0.0
                d.ctrl[:] = 0.0
                mujoco.mj_forward(self.mjm, d)
            return self.observe()
        with wp.ScopedDevice(self.device):
            q0 = self._qpos0 if qpos is None else np.asarray(qpos, np.float32)
            q0 = np.broadcast_to(q0, (self.num_worlds, self.nq)).copy()
            self.d.qpos.assign(q0)
            self.d.qvel.zero_()
            self.d.ctrl.zero_()
            mjw.forward(self.m, self.d)
        return self.observe()

    def step(self, action) -> dict:
        """Apply ``action`` (nworld, nu) to the actuators and advance physics."""
        if self.is_cpu:
            a = action.numpy() if isinstance(action, wp.array) else np.asarray(action)
            a = np.ascontiguousarray(a, np.float32)
            if a.ndim == 1:
                a = a[None]
            for i, d in enumerate(self._datas):
                d.ctrl[:] = a[i]
                for _ in range(self.cfg.n_substeps):
                    mujoco.mj_step(self.mjm, d)
            return self.observe()
        with wp.ScopedDevice(self.device):
            ctrl_t = wp.to_torch(self.d.ctrl) if self.backend == "torch" else None
            if isinstance(action, np.ndarray):
                self.d.ctrl.assign(np.ascontiguousarray(action, np.float32))
            elif isinstance(action, wp.array):
                wp.copy(self.d.ctrl, action)
            elif ctrl_t is not None:  # torch tensor
                ctrl_t.copy_(action)
            else:
                self.d.ctrl.assign(np.ascontiguousarray(
                    np.asarray(action), np.float32))
            for _ in range(self.cfg.n_substeps):
                mjw.step(self.m, self.d)
        return self.observe()

    def observe(self) -> dict:
        """Raw proprioceptive observation. Task-specific packing goes on top."""
        if self.is_cpu:
            qpos = np.stack([d.qpos for d in self._datas]).astype(np.float32)
            qvel = np.stack([d.qvel for d in self._datas]).astype(np.float32)
            sd = np.stack([d.sensordata for d in self._datas]).astype(np.float32)
            t = np.array([d.time for d in self._datas], np.float32)
            return {"qpos": self._out(qpos), "qvel": self._out(qvel),
                    "sensordata": self._out(sd), "time": self._out(t)}
        return {
            "qpos": self._out(self.d.qpos),      # (N, nq)  free base = [pos3, quat4, joints]
            "qvel": self._out(self.d.qvel),      # (N, nv)  free base = [linvel3, angvel3, joints]
            "sensordata": self._out(self.d.sensordata),
            "time": self._out(self.d.time),
        }

    def render(self) -> dict:
        """Render all requested cameras for all worlds. Returns
        {cam_name: {"rgb": (N,H,W,3) uint8, "depth": (N,H,W) float32}}."""
        if not self._want_render:
            raise RuntimeError("no cameras configured; pass cameras=[...] to G1Env")
        if self.is_cpu:
            out: dict[str, dict] = {}
            for c in self._cams:
                cid = self._cam_index[c.name]
                r = self._renderers[(c.height, c.width)]
                entry: dict = {}
                if self.cfg.render_rgb:
                    rgbs = []
                    for d in self._datas:
                        r.disable_depth_rendering()
                        r.update_scene(d, camera=cid)
                        rgbs.append(r.render().copy())
                    entry["rgb"] = self._out(np.stack(rgbs).astype(np.uint8))
                if self.cfg.render_depth:
                    depths = []
                    for d in self._datas:
                        r.enable_depth_rendering()
                        r.update_scene(d, camera=cid)
                        depths.append((r.render() * self.cfg.depth_scale).copy())
                    r.disable_depth_rendering()
                    entry["depth"] = self._out(np.stack(depths).astype(np.float32))
                out[c.name] = entry
            return out
        out: dict[str, dict] = {}
        with wp.ScopedDevice(self.device):
            mjw.refit_bvh(self.m, self.d, self._rc)
            mjw.render(self.m, self.d, self._rc)
            for c in self._cams:
                cid = self._cam_index[c.name]
                entry: dict = {}
                if c.name in self._rgb_out:
                    mjw.get_rgb(self._rc, cid, self._rgb_out[c.name])
                    entry["rgb"] = self._out(self._rgb_out[c.name])
                if c.name in self._depth_out:
                    mjw.get_depth(self._rc, cid, self.cfg.depth_scale,
                                  self._depth_out[c.name])
                    entry["depth"] = self._out(self._depth_out[c.name])
                out[c.name] = entry
        return out

    # -- introspection -----------------------------------------------------
    @property
    def dt(self) -> float:
        return float(self.mjm.opt.timestep) * self.cfg.n_substeps

    @property
    def camera_names(self) -> list[str]:
        return [c.name for c in self._cams]

    def actuator_names(self) -> list[str]:
        return [mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                for i in range(self.nu)]

    def __repr__(self) -> str:
        return (f"G1Env(worlds={self.num_worlds}, nq={self.nq}, nv={self.nv}, "
                f"nu={self.nu}, cams={[c.name for c in self._cams]}, "
                f"backend={self.backend}, dt={self.dt:.4f}s)")
