r"""M9 Step 7b: render a replay .usda to MP4, headless, from its fixed OverviewCamera. DESCRIPTIVE DEMO, NOT eval-v1.

    # Directory: <repo root>\residual_ppo
    & <isaac>\python.bat demo\render_replay.py --case indoor_I1_aisle_w7
    & <isaac>\python.bat demo\render_replay.py --case forest_F3_weave_w7
    & <isaac>\python.bat demo\render_replay.py --case indoor_I1_aisle_w7 --stills <dir>   (3 PNGs only, framing check)

1920x1080, 30 fps, real time: video frame i shows time code 2i of the 60 Hz replay (t = i / 30 s). The timeline is
never played (no physics); each frame is captured with omni.replicator at a held time (set_current_time, then
orchestrator.step(delta_time=0.0)). The legend and caption are drawn into every frame by code (OpenCV putText; no
ffmpeg needed). Encoding: Kit's omni.videoencoding extension (NVENC H.264 into MP4), fed RGBA frames from Python;
this OpenCV build has no video writer (no ffmpeg / MSMF backend; probed). The MP4 is verified by parsing its boxes
(sample count, duration, codec). Output: results/m9_demo/video/<case>.mp4 plus <case>.json.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESIDUAL_DIR))

import numpy as np  # noqa: E402

from demo import demo_config  # noqa: E402
from demo.make_replay import CONTROLLERS  # noqa: E402  (single source of the tint colours)


def _bgr255(rgb):
    return tuple(int(round(255 * c)) for c in reversed(rgb))


LEGEND = [(label, _bgr255(rgb)) for label, rgb in CONTROLLERS.values()] + [("reference path", (255, 255, 255))]
SCENE_TEXT = {"indoor": "indoor warehouse", "forest": "forest"}


def case_text(case):
    """Caption from demo_config, e.g. 'indoor warehouse, I1 aisle, 7 m/s mean wind, gust scale 0.75'."""
    scene, rest = case.split("_", 1)
    track, wind = rest.rsplit("_", 1)
    spec = demo_config.WIND_LEVELS[wind]
    wind_txt = "calm (0 m/s, no gusts)" if spec["speed"] == 0 else \
        f"{spec['speed']:g} m/s mean wind, gust scale {spec['gust_scale']:g}"
    return f"{SCENE_TEXT[scene]}, {track.replace('_', ' ')}, {wind_txt}"


def overlay(frame, case, t):
    import cv2

    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    panel = frame.copy()
    cv2.rectangle(panel, (24, 24), (24 + 430, 24 + 40 + 38 * len(LEGEND)), (25, 25, 25), -1)
    cv2.rectangle(panel, (0, h - 92), (w, h), (25, 25, 25), -1)
    frame[:] = cv2.addWeighted(panel, 0.62, frame, 0.38, 0)
    for i, (label, color) in enumerate(LEGEND):
        y = 24 + 38 + 38 * i
        cv2.line(frame, (44, y - 8), (104, y - 8), color, 6, cv2.LINE_AA)
        cv2.putText(frame, label, (120, y), font, 0.85, (240, 240, 240), 2, cv2.LINE_AA)
    scene_note = ("  |  indoor wind: artificial uniform disturbance (no sheltering)" if case.startswith("indoor")
                  else "  |  uniform wind field (no sheltering by trees)")
    cv2.putText(frame, f"M9 - DESCRIPTIVE DEMO, NOT EVAL-V1  |  {case_text(case)}  |  t = {t:5.2f} s",
                (24, h - 54), font, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(frame, "crosswind toward obstacles (worst side)  |  no controller perceives obstacles" + scene_note,
                (24, h - 18), font, 0.7, (210, 210, 210), 2, cv2.LINE_AA)
    return frame


def mp4_facts(path):
    """Minimal ISO-BMFF walk: media timescale + duration (mdhd), codec (stsd), and the frame count: stsz in a plain
    MP4, or the sum of trun sample counts in a fragmented MP4 (what omni.videoencoding writes; its stsz is empty)."""
    data = Path(path).read_bytes()
    facts = {"trun_samples": 0, "fragments": 0}

    def walk(start, end):
        i = start
        while i + 8 <= end:
            size = int.from_bytes(data[i:i + 4], "big")
            kind = data[i + 4:i + 8].decode("latin-1")
            header = 8
            if size == 1:
                size, header = int.from_bytes(data[i + 8:i + 16], "big"), 16
            elif size == 0:
                size = end - i
            body = i + header
            if kind in ("moov", "trak", "mdia", "minf", "stbl", "moof", "traf"):
                facts["fragments"] += int(kind == "moof")
                walk(body, i + size)
            elif kind == "trun":
                facts["trun_samples"] += int.from_bytes(data[body + 4:body + 8], "big")
            elif kind == "mdhd":
                v = data[body]
                if v == 1:
                    facts["timescale"] = int.from_bytes(data[body + 20:body + 24], "big")
                    facts["duration"] = int.from_bytes(data[body + 24:body + 32], "big")
                else:
                    facts["timescale"] = int.from_bytes(data[body + 12:body + 16], "big")
                    facts["duration"] = int.from_bytes(data[body + 16:body + 20], "big")
            elif kind == "stsz":
                facts["samples"] = int.from_bytes(data[body + 8:body + 12], "big")
            elif kind == "stsd":
                facts["codec"] = data[body + 12:body + 16].decode("latin-1")
            if size < header:
                break
            i += size

    walk(0, len(data))
    if "timescale" in facts and facts.get("timescale"):
        facts["seconds"] = facts["duration"] / facts["timescale"]
    facts["frames"] = facts.get("samples") or facts["trun_samples"]
    return facts


def verify(case, fps, size):
    """Check an existing MP4 (no rendering) and rewrite its JSON facts."""
    video_dir = RESIDUAL_DIR / "results/m9_demo/video"
    mp4, jpath = video_dir / f"{case}.mp4", video_dir / f"{case}.json"
    info = json.loads(jpath.read_text(encoding="utf-8")) if jpath.is_file() else {"case": case}
    facts = mp4_facts(mp4)
    expected = info.get("frames_encoded")
    ok = (facts.get("codec") == "avc1" and facts["frames"] == expected and info.get("frames_failed") == 0
          and abs(facts.get("seconds", 0) - expected / fps) < 1e-6)
    info.update(mp4_boxes=facts, bytes=mp4.stat().st_size, sha256=hashlib.sha256(mp4.read_bytes()).hexdigest(),
                verified=bool(ok))
    jpath.write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"[M9.render] verify {mp4.name}: {facts['frames']} frames (expected {expected}), codec {facts.get('codec')}, "
          f"{facts.get('seconds', 0):.3f} s, {facts['fragments']} fragments, {mp4.stat().st_size / 1e6:.1f} MB -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--rt-subframes", type=int, default=4)
    ap.add_argument("--stills", default=None, help="write 3 PNG stills to this folder instead of a video")
    ap.add_argument("--verify-only", action="store_true", help="check an existing MP4; no Isaac, no rendering")
    args = ap.parse_args()
    if args.verify_only:
        print(f"[M9.render] RESULT: {'PASS' if verify(args.case, args.fps, (args.width, args.height)) else 'FAIL'}")
        return
    replay = RESIDUAL_DIR / "results/m9_demo/replay" / f"{args.case}.usda"
    if not replay.is_file():
        raise SystemExit(f"missing {replay}")

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": args.width, "height": args.height,
                         "renderer": "RaytracedLighting"})
    try:
        import cv2
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.utils.stage import is_stage_loading

        omni.usd.get_context().open_stage(str(replay))
        for _ in range(20):
            app.update()
        while is_stage_loading():
            app.update()
        stage = omni.usd.get_context().get_stage()
        hz = stage.GetTimeCodesPerSecond()
        end_code = int(stage.GetEndTimeCode())
        seconds = end_code / hz
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        timeline.set_time_codes_per_second(hz)
        rep.orchestrator.set_capture_on_play(False)
        rp = rep.create.render_product("/World/M9Replay/OverviewCamera", (args.width, args.height))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rp])
        for _ in range(120):          # warm-up: materials, textures and the RTX caches settle
            app.update()

        def capture(t):
            timeline.set_current_time(t)
            rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0, pause_timeline=True)
            rgba = np.asarray(annot.get_data())
            if rgba.ndim != 3 or rgba.shape[0] != args.height or rgba.shape[1] != args.width:
                raise RuntimeError(f"unexpected frame shape {rgba.shape}")
            return np.ascontiguousarray(rgba[:, :, :3][:, :, ::-1])     # RGB -> BGR

        capture(0.0)                  # discard the first capture after warm-up
        if args.stills:
            out = Path(args.stills)
            out.mkdir(parents=True, exist_ok=True)
            for label, t in (("start", 0.0), ("mid", round(seconds / 2 * args.fps) / args.fps), ("end", seconds)):
                frame = overlay(capture(t), args.case, t)
                cv2.imwrite(str(out / f"{args.case}_{label}.png"), frame)
                print(f"[M9.render] still {label} t={t:.2f} s mean {frame.mean():.1f}", flush=True)
            print("[M9.render] RESULT: DONE (stills)", flush=True)
            return

        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.videoencoding")
        app.update()
        from video_encoding import get_video_encoding_interface

        encoder = get_video_encoding_interface()
        if encoder is None:
            raise RuntimeError("omni.videoencoding interface not available")
        video_dir = RESIDUAL_DIR / "results/m9_demo/video"
        video_dir.mkdir(parents=True, exist_ok=True)
        mp4 = video_dir / f"{args.case}.mp4"
        n_frames = int(np.floor(seconds * args.fps + 1e-9)) + 1
        if not encoder.start_encoding(str(mp4), float(args.fps), n_frames, True):
            raise RuntimeError("omni.videoencoding start_encoding failed")
        means, failed = [], 0
        try:
            for i in range(n_frames):
                t = i / args.fps
                frame = overlay(capture(t), args.case, t)
                means.append(float(frame.mean()))
                rgba = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
                failed += int(not encoder.encode_next_frame_from_buffer(rgba, args.width, args.height))
                if i % 60 == 0:
                    print(f"[M9.render] frame {i}/{n_frames} t={t:.2f} s mean {means[-1]:.1f}", flush=True)
        finally:
            encoder.finalize_encoding()
        facts = mp4_facts(mp4)
        info = {"case": args.case, "file": mp4.name, "encoder": "omni.videoencoding (NVENC H.264)",
                "fps": args.fps, "size": [args.width, args.height], "frames_encoded": n_frames,
                "frames_failed": failed, "mp4_boxes": facts, "seconds": n_frames / args.fps,
                "replay_end_time_code": end_code, "replay_hz": hz,
                "frame_mean_brightness_min_max": [min(means), max(means)], "rt_subframes": args.rt_subframes,
                "bytes": mp4.stat().st_size, "sha256": hashlib.sha256(mp4.read_bytes()).hexdigest(),
                "camera": "/World/M9Replay/OverviewCamera (fixed)", "overlay": "OpenCV putText, drawn by code"}
        (video_dir / f"{args.case}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        print(f"[M9.render] written {mp4.name}: {n_frames} frames encoded ({failed} failed)", flush=True)
        if not verify(args.case, args.fps, (args.width, args.height)):
            raise RuntimeError("video check failed")
        print("[M9.render] RESULT: DONE", flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        print("[M9.render] RESULT: FAIL", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
