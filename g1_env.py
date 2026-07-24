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
                               "assets", "g1", "g1.xml")
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


# A couple of sensible defaults so `cameras=["head","track"]` just works.
_PRESET_CAMERAS: dict[str, CameraSpec] = {
    # First-person head camera looking forward (+x), mounted on the head link.
    "head": CameraSpec(
        name="head", mount="head_link", target=None,
        pos=(0.08, 0.0, 0.0), quat=(0.5, 0.5, -0.5, -0.5), fovy=70.0,
        width=224, height=224,
    ),
    # Third-person chase camera tracking the pelvis.
    "track": CameraSpec(
        name="track", mount=None, target="pelvis",
        pos=(2.5, -2.0, 1.5), width=256, height=256, fovy=45.0,
    ),
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


def _add_scene_extras(spec: mujoco.MjSpec, cfg: G1EnvConfig,
                      cams: list[CameraSpec]) -> None:
    """Add a floor, a light, and the requested cameras onto the loaded G1 spec."""
    if cfg.add_floor:
        # checker ground so the batch renderer's textures/depth have structure.
        spec.add_texture(
            name="groundtex", type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.15, 0.2],
            width=300, height=300,
        )
        spec.add_material(
            name="groundplane", textures=["", "groundtex"],
            texrepeat=[5, 5], texuniform=True, reflectance=0.0,
        )
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [0.0, 0.0, 0.05]
        floor.material = "groundplane"

    light = spec.worldbody.add_light()
    light.pos = [0.0, 0.0, 3.0]
    light.dir = [0.0, 0.0, -1.0]
    light.directional = True

    bodies = {b.name for b in spec.bodies}
    for c in cams:
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

        wp.init()
        self.device = cfg.device

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

        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(self.mjm)
            self.d = mjw.put_data(self.mjm, self.mjd, nworld=cfg.num_worlds)

        self._rc = None
        self._rgb_out: dict[str, wp.array] = {}
        self._depth_out: dict[str, wp.array] = {}
        if cams and (cfg.render_rgb or cfg.render_depth):
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

        cam_res = [(c.width, c.height) for c in self._cams]
        active = [False] * self.mjm.ncam
        for c in self._cams:
            active[self._cam_index[c.name]] = True

        with wp.ScopedDevice(self.device):
            self._rc = mjw.create_render_context(
                self.mjm, nworld=cfg.num_worlds,
                cam_res=cam_res if len(set(cam_res)) > 1 else cam_res[0],
                render_rgb=cfg.render_rgb, render_depth=cfg.render_depth,
                cam_active=active if self.mjm.ncam > len(self._cams) else None,
            )
            for c in self._cams:
                if cfg.render_rgb:
                    self._rgb_out[c.name] = wp.zeros(
                        (cfg.num_worlds, c.height, c.width), dtype=wp.vec3f)
                if cfg.render_depth:
                    self._depth_out[c.name] = wp.zeros(
                        (cfg.num_worlds, c.height, c.width), dtype=wp.float32)

    # -- conversion helper -------------------------------------------------
    def _out(self, arr: wp.array):
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
        return {
            "qpos": self._out(self.d.qpos),      # (N, nq)  free base = [pos3, quat4, joints]
            "qvel": self._out(self.d.qvel),      # (N, nv)  free base = [linvel3, angvel3, joints]
            "sensordata": self._out(self.d.sensordata),
            "time": self._out(self.d.time),
        }

    def render(self) -> dict:
        """Render all requested cameras for all worlds. Returns
        {cam_name: {"rgb": (N,H,W,3) uint8, "depth": (N,H,W) float32}}."""
        if self._rc is None:
            raise RuntimeError("no cameras configured; pass cameras=[...] to G1Env")
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

    def actuator_names(self) -> list[str]:
        return [mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                for i in range(self.nu)]

    def __repr__(self) -> str:
        return (f"G1Env(worlds={self.num_worlds}, nq={self.nq}, nv={self.nv}, "
                f"nu={self.nu}, cams={[c.name for c in self._cams]}, "
                f"backend={self.backend}, dt={self.dt:.4f}s)")
