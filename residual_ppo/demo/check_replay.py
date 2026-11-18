r"""M9 Step 7 check: a replay .usda reproduces its .npz series EXACTLY at 5 sample times. NOT eval-v1.

    # Directory: <repo root>\residual_ppo
    & <isaac>\python.bat demo\check_replay.py --case indoor_I1_aisle_w7

For each controller and each of 5 time codes (first, 25%, 50%, 75%, last row): the authored xformOp:translate:m9 and
xformOp:orient:m9 samples equal the .npz row (np.array_equal on float64), and the COMPOSED world position of the
copy's body prim (UsdGeom ComputeLocalToWorldTransform, through the referenced iris.usd) equals the .npz position.
Also: the scene sublayer resolves, /World/iris is inactive, and no Physics*/Physx* schema is left on a copy.
"""
import argparse
import json
import sys
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESIDUAL_DIR))

import numpy as np  # noqa: E402

CONTROLLERS = ("pid", "pid_ppo", "ppo_direct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="indoor_I1_aisle_w7")
    args = ap.parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from pxr import Usd, UsdGeom

        replay = RESIDUAL_DIR / "results/m9_demo/replay" / f"{args.case}.usda"
        manifest = json.loads((replay.parent / "replay_manifest.json").read_text(encoding="utf-8"))
        entry = next(f for f in manifest["files"] if f["case"] == args.case)
        stage = Usd.Stage.Open(str(replay), load=Usd.Stage.LoadNone)
        layers = [Path(l.realPath).name for l in stage.GetLayerStack() if l.realPath]
        ok_all = True
        out = {"case": args.case, "layer_stack": layers,
               "iris_active": stage.GetPrimAtPath("/World/iris").IsActive(), "samples": {}}
        print(f"[M9.check_replay] layer stack {layers}; /World/iris active = {out['iris_active']}")
        ok_all &= ("dij3d_boxes.usd" in layers or "forest.usd" in layers) and not out["iris_active"]
        for c in CONTROLLERS:
            with np.load(RESIDUAL_DIR / entry["sources"][c]) as z:
                rec = z["rec60"]
            drone = stage.GetPrimAtPath(f"/World/M9Replay/{c}/drone")
            body = stage.GetPrimAtPath(f"/World/M9Replay/{c}/drone/body")
            tr = drone.GetAttribute("xformOp:translate:m9")
            orient = drone.GetAttribute("xformOp:orient:m9")
            physics_left = [str(p.GetPath()) for p in Usd.PrimRange(drone)
                            if any(s.startswith(("Physics", "Physx")) for s in p.GetAppliedSchemas())]
            ks = sorted({0, len(rec) // 4, len(rec) // 2, 3 * len(rec) // 4, len(rec) - 1})
            rows = []
            for k in ks:
                p_usd = np.array(tr.Get(Usd.TimeCode(k)), dtype=np.float64)
                q = orient.Get(Usd.TimeCode(k))
                q_usd = np.array([q.GetReal(), *q.GetImaginary()], dtype=np.float64)
                world = np.array(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode(k))
                                 .ExtractTranslation(), dtype=np.float64)
                p_npz, q_npz = rec[k, 1:4], rec[k, 4:8]
                same = (np.array_equal(p_usd, p_npz), np.array_equal(q_usd, q_npz), np.array_equal(world, p_npz))
                rows.append({"time_code": k, "t_s": k / 60, "npz_position": p_npz.tolist(),
                             "usd_translate_equal": same[0], "usd_orient_equal": same[1],
                             "composed_body_world_equal": same[2],
                             "composed_max_abs_diff": float(np.max(np.abs(world - p_npz)))})
                ok_all &= all(same)
                print(f"[M9.check_replay] {c:10s} k={k:5d} (t={k / 60:6.3f} s) npz {np.round(p_npz, 6).tolist()} "
                      f"translate {same[0]} orient {same[1]} composed-body {same[2]}")
            ok_all &= not physics_left
            out["samples"][c] = {"rows": rows, "physics_schemas_left": physics_left}
        out["result"] = "PASS" if ok_all else "FAIL"
        (replay.parent / f"check_{args.case}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"[M9.check_replay] RESULT: {out['result']}", flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        print("[M9.check_replay] RESULT: FAIL", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
