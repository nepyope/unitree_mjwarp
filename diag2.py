import warp as wp, traceback
from g1_env import G1Env
def run(tag, **kw):
    try:
        env = G1Env(**kw); env.reset()
        act = wp.zeros((env.num_worlds, env.nu), dtype=wp.float32)
        for _ in range(20): env.step(act)
        wp.synchronize()
        if env._rc is not None:
            for _ in range(5): env.render()
            wp.synchronize()
        print(f"[OK] {tag}", flush=True)
    except Exception as e:
        print(f"[FAIL] {tag}: {e}", flush=True); traceback.print_exc()
run("track_16", num_worlds=16, cameras=["track"])
run("headtrack_16", num_worlds=16, cameras=["head","track"])
run("head_512", num_worlds=512, cameras=["head"])
run("headtrack_512", num_worlds=512, cameras=["head","track"])
