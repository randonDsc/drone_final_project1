r"""M9 Step 8: clearer demo videos (v2). DESCRIPTIVE DEMO, NOT eval-v1. No new flights.

    # Directory: <repo root>\residual_ppo
    & <isaac>\python.bat demo\render_replay_v2.py --case indoor_I1_aisle_w7 --stills     (3 stills + offset check)
    & <isaac>\python.bat demo\render_replay_v2.py --case indoor_I1_aisle_w7              (MP4)
    & <isaac>\python.bat demo\render_replay_v2.py --case indoor_I1_aisle_w7 --verify-only (check an MP4; no Isaac)

Inputs (committed, unchanged): results/m9_demo/replay/<case>.usda (make_replay.py) and the 60 Hz .npz series.
Outputs, all under results/m9_demo/video_v2/:
- <case>.usda: a NEW layer that sublayers ../replay/<case>.usda and only adds opinions: drone tints = the legend's
  sRGB colours converted to linear (v1 authored the sRGB numbers as linear diffuse, plus emissive 0.9, so the
  orange PID drone rendered yellow), emissive 0; the faint flown-path curves deactivated (trails are now drawn growing);
  a non-physical light-blue 3D wind arrow (video_v2_layout.place_wind_arrow); CameraV2 from video_v2_layout's search.
  Before rendering, the composed stage is checked: 60 time codes per second, body world position == .npz EXACTLY
  at 5 time codes per controller, flown paths inactive, no Physics*/Physx* schema on the arrow.
- <case>_layout.json: camera, framing assert (every reference / flown / ring / arrow point in frame and clear of the
  overlay panels), occlusion, inset.
- stills/<case>_{start,mid,end}.png and <case>_stills.json: at t = mid, per controller, the rendered drone is located
  by rendering the frame again with that drone hidden (difference mask, largest component); its centroid is compared
  with the projected body position (pixel offset), and its median hue with the legend hue.
- <case>.mp4 + <case>.json: 2 s hold at t = 0 (start ring + "All three drones start here, same wind"), then real time
  (frame i shows t = i / 30 s). Encoder and checks as render_replay.py (Kit omni.videoencoding, NVENC H.264; MP4
  box parse via render_replay.mp4_facts). MP4s are git-ignored.
Overlay (OpenCV, drawn by code, NOT occluded by the scene): growing trails, name tags, current reference point,
start ring, wind panel (projected wind direction), live distance readout, top-view inset.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESIDUAL_DIR))

import numpy as np  # noqa: E402

from demo import video_v2_layout as L  # noqa: E402
from demo.render_replay import case_text, mp4_facts  # noqa: E402

OUT = L.OUT
REFERENCE_PRIM = "/World/M9Replay/reference"   # make_replay.py's white reference curve
HOLD_SECONDS = 2.0
HOLD_CAPTION = "All three drones start here, same wind"
TAG_OFFSETS = {"pid": (-130, -85), "pid_ppo": (0, -120), "ppo_direct": (130, -85)}
EMISSIVE_V2 = 0.2          # x the SAME linear colour: lifts dark shading without shifting hue (v1: 0.9 x sRGB)
ROUGHNESS_V2 = 0.7         # less white specular on the tinted body (v1: 0.5)
WHITE = (255, 255, 255)
DARK = (25, 25, 25)


# ---------------------------------------------------------------- USD layer

def author_layer(plan, data):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    case = plan["case"]
    replay = RESIDUAL_DIR / "results/m9_demo/replay" / f"{case}.usda"
    src = Usd.Stage.Open(str(replay), load=Usd.Stage.LoadNone)
    tcps, start, end = src.GetTimeCodesPerSecond(), src.GetStartTimeCode(), src.GetEndTimeCode()
    path = OUT / f"{case}.usda"
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    root = stage.GetRootLayer()
    root.subLayerPaths.append(f"../replay/{case}.usda")
    # The ROOT layer's time codes per second rule the stage; without these the 60 Hz sublayer would be retimed.
    stage.SetTimeCodesPerSecond(tcps)
    stage.SetFramesPerSecond(src.GetFramesPerSecond())
    stage.SetStartTimeCode(start)
    stage.SetEndTimeCode(end)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root.customLayerData = {"m9": f"{case}; video v2 overlay layer; descriptive demo, not eval-v1; "
                                  f"no controller perceives obstacles; crosswind toward obstacles (worst side)"}
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    for c in L.ORDER:
        sh = stage.GetPrimAtPath(f"/World/M9Replay/Looks/tint_{c}/Shader")
        lin = L.srgb_to_linear(L.COLORS_SRGB[c])
        sh.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*map(float, lin)))
        sh.GetAttribute("inputs:emissiveColor").Set(Gf.Vec3f(*map(float, EMISSIVE_V2 * lin)))
        sh.GetAttribute("inputs:roughness").Set(ROUGHNESS_V2)
        stage.GetPrimAtPath(f"/World/M9Replay/{c}/flown_path").SetActive(False)

    looks = UsdGeom.Scope.Define(stage, "/World/M9Replay/LooksV2")
    mat = UsdShade.Material.Define(stage, f"{looks.GetPath()}/wind_arrow")
    sh = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    lin = L.srgb_to_linear(L.WIND_ARROW_SRGB)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, lin)))
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, 0.25 * lin)))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    tail = np.array(plan["wind_arrow"]["tail"])
    tip = np.array(plan["wind_arrow"]["tip"])
    yaw = float(np.degrees(np.arctan2(tip[1] - tail[1], tip[0] - tail[0])))
    arrow = UsdGeom.Xform.Define(stage, "/World/M9Replay/WindArrowV2")
    arrow.GetPrim().SetCustomDataByKey("note", "non-physical marker of the mean wind direction")
    arrow.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, tail)))
    arrow.AddRotateZOp().Set(yaw)
    shaft_len = L.ARROW_LENGTH - L.ARROW_HEAD_LENGTH
    shaft = UsdGeom.Cylinder.Define(stage, f"{arrow.GetPath()}/shaft")
    shaft.CreateAxisAttr(UsdGeom.Tokens.x)
    shaft.CreateHeightAttr(shaft_len)
    shaft.CreateRadiusAttr(L.ARROW_SHAFT_RADIUS)
    shaft.AddTranslateOp().Set(Gf.Vec3d(shaft_len / 2, 0, 0))
    head = UsdGeom.Cone.Define(stage, f"{arrow.GetPath()}/head")
    head.CreateAxisAttr(UsdGeom.Tokens.x)
    head.CreateHeightAttr(L.ARROW_HEAD_LENGTH)
    head.CreateRadiusAttr(L.ARROW_HEAD_RADIUS)
    head.AddTranslateOp().Set(Gf.Vec3d(shaft_len + L.ARROW_HEAD_LENGTH / 2, 0, 0))   # cone apex at +height/2
    for g in (shaft, head):
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)

    cam = L.camera_from_plan(plan)
    cprim = UsdGeom.Camera.Define(stage, "/World/M9Replay/CameraV2")
    m = Gf.Matrix4d(*cam["r"], 0, *cam["u"], 0, *(-cam["f"]), 0, *cam["pos"], 1)
    cprim.AddTransformOp().Set(m)
    cprim.CreateFocalLengthAttr(float(plan["camera"]["focal_mm"]))
    cprim.CreateHorizontalApertureAttr(L.H_APERTURE)
    cprim.CreateVerticalApertureAttr(L.H_APERTURE / L.ASPECT)
    cprim.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500.0))
    root.Save()
    return path


def check_layer(path, data):
    """Composed v2 stage == the .npz exactly (5 time codes per controller); 60 tc/s; flown paths off; arrow
    non-physical; camera transform == plan."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    out = {"time_codes_per_second": stage.GetTimeCodesPerSecond(), "samples": {}, "ok": True}
    out["ok"] &= stage.GetTimeCodesPerSecond() == L.HZ
    for c in L.ORDER:
        rec = data["series"][c]["rec60"]
        body = stage.GetPrimAtPath(f"/World/M9Replay/{c}/drone/body")
        ks = sorted({0, len(rec) // 4, len(rec) // 2, 3 * len(rec) // 4, len(rec) - 1})
        diffs = []
        for k in ks:
            w = np.array(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode(k)).ExtractTranslation())
            diffs.append(float(np.max(np.abs(w - rec[k, 1:4]))))
        inactive = not stage.GetPrimAtPath(f"/World/M9Replay/{c}/flown_path").IsActive()
        out["samples"][c] = {"time_codes": ks, "max_abs_diff": diffs, "flown_path_inactive": inactive}
        out["ok"] &= all(d == 0.0 for d in diffs) and inactive
    phys = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath("/World/M9Replay/WindArrowV2"))
            if any(s.startswith(("Physics", "Physx")) for s in p.GetAppliedSchemas())]
    out["wind_arrow_physics_schemas"] = phys
    out["ok"] &= not phys
    out["ok"] = bool(out["ok"])
    return out


# ---------------------------------------------------------------- overlay

def _text(img, s, org, scale, color, thick=2):
    import cv2

    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, DARK, thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _fit_scale(s, width, scale, thick=2):
    import cv2

    while scale > 0.4 and cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] > width:
        scale -= 0.02
    return scale


def _panel(frame, rect, alpha=0.72):
    """Darken a rectangle, clipped to the frame (a negative index would wrap around)."""
    import cv2

    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in rect)
    x0, x1 = max(x0, 0), min(x1, w)
    y0, y1 = max(y0, 0), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = cv2.addWeighted(np.full_like(sub, DARK), alpha, sub, 1 - alpha, 0)


def _arrow(img, p, q, color, thick, tip=0.25):
    import cv2

    p = tuple(int(round(v)) for v in p)
    q = tuple(int(round(v)) for v in q)
    cv2.arrowedLine(img, p, q, DARK, thick + 4, cv2.LINE_AA, tipLength=tip)
    cv2.arrowedLine(img, p, q, color, thick, cv2.LINE_AA, tipLength=tip)


def _rect_hits_panels(x0, y0, x1, y1):
    if x0 < L.EDGE or y0 < L.EDGE or x1 > L.W - L.EDGE or y1 > L.H - L.EDGE:
        return True
    return any(not (x1 < a or x0 > c or y1 < b or y0 > d) for a, b, c, d in L.PANELS.values())


class Overlay:
    """Everything drawn on top of a rendered frame. Projections precomputed once per case."""

    def __init__(self, plan, data):
        self.plan, self.data, self.case = plan, data, plan["case"]
        self.cam = L.camera_from_plan(plan)
        self.n = min(len(data["series"][c]["rec60"]) for c in L.ORDER)
        self.px = {c: L.project(data["series"][c]["rec60"][:, 1:4], self.cam)[0] for c in L.ORDER}
        self.ref_now_px = L.project(data["series"]["pid"]["rec60"][:, 14:17], self.cam)[0]
        self.dist = {c: np.linalg.norm(data["series"][c]["rec60"][:, 1:4] - data["series"][c]["rec60"][:, 14:17],
                                       axis=1) for c in L.ORDER}
        self.ring_px = L.project(L.ring_points(data["start"]), self.cam)[0]
        self.start_px = L.project(data["start"][None], self.cam)[0][0]
        tail, tip = data["arrow"]
        mid = 0.5 * (tail + tip)
        a, b = L.project(np.stack([mid, mid + np.array([data["wind_dir"][0], data["wind_dir"][1], 0.0])]), self.cam)[0]
        v = b - a
        self.wind_screen = v / np.linalg.norm(v)
        self.wind_screen_px_per_m = float(np.linalg.norm(v))
        self.tf = plan["inset"]["transform"]
        self.inset_px = {c: L.to_inset(data["series"][c]["rec60"][:, 1:3], self.tf) for c in L.ORDER}
        self.inset_ref = L.to_inset(data["ref_path"][:, :2], self.tf)
        self.inset_arrow = L.to_inset(np.stack([tail[:2], tip[:2]]), self.tf)
        self.inset_start = L.to_inset(data["start"][None, :2], self.tf)[0]
        self.colors = {c: L.bgr255(L.COLORS_SRGB[c]) for c in L.ORDER}
        self.tag_clamped = 0

    def k_at(self, t):
        return int(min(round(t * L.HZ), self.n - 1))

    def draw(self, frame, t, hold=False):
        import cv2

        k = self.k_at(t)
        # start ring
        ring = self.ring_px.round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [ring], True, DARK, 7 if hold else 5, cv2.LINE_AA)
        cv2.polylines(frame, [ring], True, WHITE, 4 if hold else 2, cv2.LINE_AA)
        # growing trails (up to now)
        for c in L.ORDER:
            pts = self.px[c][:k + 1].round().astype(np.int32).reshape(-1, 1, 2)
            if len(pts) > 1:
                cv2.polylines(frame, [pts], False, DARK, 7, cv2.LINE_AA)
                cv2.polylines(frame, [pts], False, self.colors[c], 4, cv2.LINE_AA)
        # where the reference is now
        rp = tuple(int(round(v)) for v in self.ref_now_px[k])
        cv2.circle(frame, rp, 9, DARK, -1, cv2.LINE_AA)
        cv2.circle(frame, rp, 6, WHITE, -1, cv2.LINE_AA)
        # name tags (each avoids the panels, the hold caption and the tags already placed in this frame)
        self._placed = []
        if hold:
            self._placed.append(self._hold_caption(frame))
        for c in L.ORDER:
            self._tag(frame, c, self.px[c][k])
        self._legend(frame, k)
        self._wind_panel(frame)
        self._inset(frame, k)
        self._caption(frame, t, hold)
        return frame

    def _tag(self, frame, c, p):
        import cv2

        label = L.LABELS[c]
        scale = 0.85
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        ox, oy = TAG_OFFSETS[c]
        side = np.sign(ox) if ox else 1.0
        hw, hh = tw / 2 + 10, th / 2 + 9
        # preferred offset, its mirror images, then beside and below the drone
        offsets = [(ox, oy), (ox, -oy), (-ox, oy), (-ox, -oy), (side * (hw + 60), 0), (-side * (hw + 60), 0),
                   (side * (hw + 40), 70), (-side * (hw + 40), 70), (0, 110), (side * (hw + 40), 140),
                   (-side * (hw + 40), 140), (0, 180), (side * (hw + 60), -60), (-side * (hw + 60), -60)]
        offsets += [(r * np.cos(a), r * np.sin(a)) for r in (120, 170, 230, 300)
                    for a in np.radians(np.arange(0, 360, 30))]

        def free(x0, y0, x1, y1):
            return not _rect_hits_panels(x0, y0, x1, y1) and all(
                x1 + 4 < a or x0 - 4 > c or y1 + 4 < b or y0 - 4 > d for a, b, c, d in self._placed)

        choice = None
        for dx, dy in offsets:
            cx, cy = p[0] + dx, p[1] + dy
            if free(cx - hw, cy - hh, cx + hw, cy + hh):
                choice = (cx, cy)
                break
        if choice is None:      # no free spot: keep the preferred offset, shifted into the frame
            self.tag_clamped += 1
            choice = (float(np.clip(p[0] + ox, L.EDGE + hw, L.W - L.EDGE - hw)),
                      float(np.clip(p[1] + oy, L.EDGE + hh, L.H - L.EDGE - hh)))
        cx, cy = choice
        x0, y0, x1, y1 = cx - hw, cy - hh, cx + hw, cy + hh
        self._placed.append((x0, y0, x1, y1))
        anchor = (int(round(np.clip(p[0], x0, x1))), int(round(np.clip(p[1], y0, y1))))
        cv2.line(frame, tuple(int(round(v)) for v in p), anchor, DARK, 5, cv2.LINE_AA)
        cv2.line(frame, tuple(int(round(v)) for v in p), anchor, self.colors[c], 2, cv2.LINE_AA)
        x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
        _panel(frame, (x0, y0, x1, y1), 0.78)
        cv2.rectangle(frame, (x0, y0), (x1, y1), self.colors[c], 2, cv2.LINE_AA)
        cv2.putText(frame, label, (int(round(cx - tw / 2)), int(round(cy + th / 2))), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, WHITE, 2, cv2.LINE_AA)

    def _hold_caption(self, frame):
        import cv2

        scale = 1.0
        (tw, th), _ = cv2.getTextSize(HOLD_CAPTION, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        ring_bottom = float(self.ring_px[:, 1].max())
        for cy in (ring_bottom + 70, float(self.ring_px[:, 1].min()) - 170, ring_bottom + 140):
            cx = float(np.clip(self.start_px[0], L.EDGE + tw / 2 + 20, L.W - L.EDGE - tw / 2 - 20))
            box = (cx - tw / 2 - 16, cy - th / 2 - 14, cx + tw / 2 + 16, cy + th / 2 + 14)
            if not _rect_hits_panels(*box):
                break
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        anchor = (int(round(np.clip(self.start_px[0], x0, x1))), y0 if cy > self.start_px[1] else y1)
        target = (int(round(self.start_px[0])), int(round(ring_bottom if cy > self.start_px[1]
                                                         else self.ring_px[:, 1].min())))
        cv2.line(frame, anchor, target, WHITE, 2, cv2.LINE_AA)
        _panel(frame, (max(x0, 0), max(y0, 0), x1, y1), 0.82)
        cv2.rectangle(frame, (x0, y0), (x1, y1), WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, HOLD_CAPTION, (int(round(cx - tw / 2)), int(round(cy + th / 2))),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 2, cv2.LINE_AA)
        return box

    def _legend(self, frame, k):
        import cv2

        x0, y0, x1, y1 = L.PANELS["legend"]
        _panel(frame, (x0, y0, x1, y1))
        cv2.putText(frame, "distance to reference now", (x0 + 20, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 2, cv2.LINE_AA)
        clr = self.data["clearance"]
        for i, c in enumerate(L.ORDER):
            y = y0 + 74 + 60 * i
            cv2.line(frame, (x0 + 20, y - 9), (x0 + 74, y - 9), self.colors[c], 7, cv2.LINE_AA)
            cv2.putText(frame, L.LABELS[c], (x0 + 92, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2, cv2.LINE_AA)
            cv2.putText(frame, f"{self.dist[c][k]:5.2f} m", (x0 + 330, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        WHITE, 2, cv2.LINE_AA)
            # static: whole-flight minimum (summary_table.csv), answers "did it hit?" since trails are drawn on top
            cv2.putText(frame, f"closest obstacle (whole flight): {clr[c]['min_clearance_m']:.1f} m",
                        (x0 + 92, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (205, 205, 205), 2, cv2.LINE_AA)
        y = y0 + 250
        cv2.line(frame, (x0 + 20, y - 9), (x0 + 74, y - 9), WHITE, 3, cv2.LINE_AA)
        cv2.circle(frame, (x0 + 47, y - 9), 6, WHITE, -1, cv2.LINE_AA)
        cv2.putText(frame, "reference path / reference now", (x0 + 92, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (220, 220, 220), 2, cv2.LINE_AA)
        hit = [L.LABELS[c] for c in L.ORDER if clr[c]["obstacle_contact"]]
        note = ("obstacle distance from drone centre; "
                + ("no obstacle contact" if not hit else "obstacle contact: " + ", ".join(hit)))
        cv2.putText(frame, note, (x0 + 20, y0 + 276), cv2.FONT_HERSHEY_SIMPLEX,
                    _fit_scale(note, x1 - x0 - 40, 0.55, 1), (190, 190, 190), 1, cv2.LINE_AA)

    def _wind_panel(self, frame):
        import cv2

        x0, y0, x1, y1 = L.PANELS["wind"]
        _panel(frame, (x0, y0, x1, y1))
        d = self.data
        s = (f"Wind {d['wind_speed']:g} m/s, gusts {d['gust_scale']:g}, toward obstacles")
        sc = _fit_scale(s, x1 - x0 - 40, 0.85)
        cv2.putText(frame, s, (x0 + 20, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX, sc, WHITE, 2, cv2.LINE_AA)
        c = np.array([x0 + 120, y0 + 108])
        _arrow(frame, c - 70 * self.wind_screen, c + 70 * self.wind_screen, L.bgr255(L.WIND_ARROW_SRGB), 6, 0.32)
        cv2.putText(frame, "mean wind direction", (x0 + 240, y0 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                    (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(frame, "(as seen by this camera;", (x0 + 240, y0 + 128), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, " blue arrow in the scene)", (x0 + 240, y0 + 152), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (190, 190, 190), 1, cv2.LINE_AA)

    def _inset(self, frame, k):
        import cv2

        x0, y0, x1, y1 = L.INSET
        _panel(frame, (x0, y0, x1, y1), 0.82)
        sub = frame[y0:y1, x0:x1].copy()
        off = np.array([x0, y0], dtype=float)

        def P(p):
            return tuple(int(round(v)) for v in np.asarray(p) - off)

        s = self.tf["scale_px_per_m"]
        grey = (120, 120, 120)
        fp = self.data["footprints"]
        for a, b, c, d in fp["rects"]:
            p0, p1 = L.to_inset([[a, b], [c, d]], self.tf)
            cv2.rectangle(sub, P([p0[0], p1[1]]), P([p1[0], p0[1]]), grey, -1)
        for cx, cy, r in fp["circles"]:
            cv2.circle(sub, P(L.to_inset([[cx, cy]], self.tf)[0]), max(int(round(r * s)), 1), grey, -1, cv2.LINE_AA)
        for ax, ay, bx, by, wdt in fp["segments"]:
            pa, pb = L.to_inset([[ax, ay], [bx, by]], self.tf)
            cv2.line(sub, P(pa), P(pb), grey, max(int(round(wdt * s)), 1), cv2.LINE_AA)
        cv2.polylines(sub, [(self.inset_ref - off).round().astype(np.int32).reshape(-1, 1, 2)], False, WHITE, 2,
                      cv2.LINE_AA)
        for c in L.ORDER:
            pts = (self.inset_px[c][:k + 1] - off).round().astype(np.int32).reshape(-1, 1, 2)
            if len(pts) > 1:
                cv2.polylines(sub, [pts], False, self.colors[c], 3, cv2.LINE_AA)
            cv2.circle(sub, P(self.inset_px[c][k]), 6, self.colors[c], -1, cv2.LINE_AA)
            cv2.circle(sub, P(self.inset_px[c][k]), 6, DARK, 1, cv2.LINE_AA)
        cv2.circle(sub, P(self.inset_start), max(int(round(L.RING_RADIUS * s)), 5), WHITE, 2, cv2.LINE_AA)
        a, b = self.inset_arrow - off
        _arrow(sub, a, b, L.bgr255(L.WIND_ARROW_SRGB), 3, 0.3)
        # title, north and scale bar
        cv2.rectangle(sub, (0, 0), (x1 - x0, L.INSET_TITLE), DARK, -1)
        cv2.putText(sub, "top view, north up; grey = obstacles at flight height", (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        bar_m = 1.0 if s * 2 > 200 else 2.0 if s * 5 > 200 else 5.0
        bx, by = 16, y1 - y0 - 14
        cv2.line(sub, (bx, by), (bx + int(round(bar_m * s)), by), WHITE, 3)
        cv2.putText(sub, f"{bar_m:g} m", (bx + int(round(bar_m * s)) + 8, by + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    WHITE, 1, cv2.LINE_AA)
        frame[y0:y1, x0:x1] = sub
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (90, 90, 90), 1)

    def _caption(self, frame, t, hold):
        import cv2

        x0, y0, x1, y1 = L.PANELS["caption"]
        _panel(frame, (x0, y0, x1, y1), 0.75)
        tt = f"t = {t:5.2f} s" + ("  (held 2 s)" if hold else "")
        cv2.putText(frame, f"M9 - DESCRIPTIVE DEMO, NOT EVAL-V1  |  {case_text(self.case)}  |  {tt}",
                    (24, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
        note = ("indoor wind: artificial uniform disturbance (no sheltering)" if self.case.startswith("indoor")
                else "uniform wind field (no sheltering by trees)")
        cv2.putText(frame, f"crosswind toward obstacles (worst side)  |  no controller perceives obstacles  |  {note}"
                           f"  |  trails and tags drawn on top", (24, y0 + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (210, 210, 210), 1, cv2.LINE_AA)


# ---------------------------------------------------------------- checks on rendered frames

def locate_drone(full, hidden, thresh=30):
    """Rendered drone = largest connected component of |full - hidden| > thresh. Returns (centroid, bbox, area,
    mask) or None."""
    import cv2

    diff = np.abs(full.astype(np.int16) - hidden.astype(np.int16)).sum(axis=2)
    mask = (diff > thresh).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[i]
    return cent[i], (int(x), int(y), int(w), int(h)), int(area), lab == i


def hue_deg_of(bgr_pixels):
    import cv2

    hsv = cv2.cvtColor(bgr_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat = hsv[(hsv[:, 1] > 80) & (hsv[:, 2] > 50)]
    if len(sat) < 10:
        return None, int(len(sat))
    ang = np.radians(sat[:, 0].astype(float) * 2)
    return float(np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360), int(len(sat))


def legend_hue_deg(c):
    import cv2

    b, g, r = L.bgr255(L.COLORS_SRGB[c])
    return float(cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0, 0]) * 2


def _hue_gap(a, b):
    return float(abs((a - b + 180) % 360 - 180))


# ---------------------------------------------------------------- verify (mirrors render_replay.verify)

def verify(case, fps):
    mp4, jpath = OUT / f"{case}.mp4", OUT / f"{case}.json"
    info = json.loads(jpath.read_text(encoding="utf-8")) if jpath.is_file() else {"case": case}
    facts = mp4_facts(mp4)
    expected = info.get("frames_encoded")
    ok = (facts.get("codec") == "avc1" and facts["frames"] == expected and info.get("frames_failed") == 0
          and abs(facts.get("seconds", 0) - expected / fps) < 1e-6)
    info.update(mp4_boxes=facts, bytes=mp4.stat().st_size, sha256=hashlib.sha256(mp4.read_bytes()).hexdigest(),
                verified=bool(ok))
    jpath.write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"[M9.v2.render] verify {mp4.name}: {facts['frames']} frames (expected {expected}), codec "
          f"{facts.get('codec')}, {facts.get('seconds', 0):.3f} s, {facts['fragments']} fragments, "
          f"{mp4.stat().st_size / 1e6:.1f} MB -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--rt-subframes", type=int, default=32)
    ap.add_argument("--stills", action="store_true", help="3 stills + offset/hue check instead of a video")
    ap.add_argument("--still-times", default="start,mid,end", help="with --stills: which of start,mid,end")
    ap.add_argument("--verify-only", action="store_true", help="check an existing MP4; no Isaac, no rendering")
    args = ap.parse_args()
    if args.verify_only:
        print(f"[M9.v2.render] RESULT: {'PASS' if verify(args.case, args.fps) else 'FAIL'}")
        return

    plan, data, _ = L.plan_case(args.case)          # framing assert (raises on failure) BEFORE Isaac starts
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.case}_layout.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": L.W, "height": L.H, "renderer": "RaytracedLighting"})
    try:
        import cv2
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.utils.stage import is_stage_loading
        from pxr import UsdGeom

        layer = author_layer(plan, data)
        layer_check = check_layer(layer, data)
        print(f"[M9.v2.render] layer {layer.name}: composed == npz, 60 tc/s, flown paths off, arrow non-physical "
              f"-> {'PASS' if layer_check['ok'] else 'FAIL'}", flush=True)
        if not layer_check["ok"]:
            raise RuntimeError(f"layer check failed: {layer_check}")

        omni.usd.get_context().open_stage(str(layer))
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
        rp = rep.create.render_product("/World/M9Replay/CameraV2", (L.W, L.H))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rp])
        for _ in range(120):
            app.update()

        def capture(t):
            timeline.set_current_time(t)
            rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0, pause_timeline=True)
            rgba = np.asarray(annot.get_data())
            if rgba.ndim != 3 or rgba.shape[0] != L.H or rgba.shape[1] != L.W:
                raise RuntimeError(f"unexpected frame shape {rgba.shape}")
            return np.ascontiguousarray(rgba[:, :, :3][:, :, ::-1])

        capture(0.0)
        ov = Overlay(plan, data)

        if args.stills:
            sdir = OUT / "stills"
            sdir.mkdir(parents=True, exist_ok=True)
            t_mid = round(seconds / 2 * args.fps) / args.fps
            wanted = args.still_times.split(",")
            if not wanted or set(wanted) - {"start", "mid", "end"}:
                raise ValueError(f"--still-times {args.still_times}")
            record = {"case": args.case, "rt_subframes": args.rt_subframes, "layer_check": layer_check,
                      "still_times": wanted, "stills": {}, "offsets_mid": {}}
            for label, t in (("start", 0.0), ("mid", t_mid), ("end", seconds)):
                if label not in wanted:
                    continue
                raw = capture(t)
                cv2.imwrite(str(sdir / f"{args.case}_{label}_raw.png"), raw)
                frame = ov.draw(raw.copy(), t, hold=(label == "start"))
                cv2.imwrite(str(sdir / f"{args.case}_{label}.png"), frame)
                record["stills"][label] = {"t_s": t, "frame_mean": float(frame.mean())}
                print(f"[M9.v2.render] still {label} t={t:.2f} s", flush=True)
                if label != "mid":
                    continue
                k = ov.k_at(t)
                dbg = raw.copy()
                drones = {c: UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/M9Replay/{c}/drone")) for c in L.ORDER}
                # The white reference curve is hidden in the CHECK renders only (stills and video keep it): where it
                # crosses a drone it splits the difference mask, and the largest component then misses part of the
                # drone (I2 pid_ppo 15.6 px with it visible). Restored before anything else is captured.
                reference = UsdGeom.Imageable(stage.GetPrimAtPath(REFERENCE_PRIM))
                if not reference.GetPrim().IsValid():
                    raise RuntimeError(f"{REFERENCE_PRIM} missing")
                reference.MakeInvisible()
                for d in drones.values():
                    d.MakeInvisible()
                background = capture(t)               # no drone at all
                isolated = {}
                for c in L.ORDER:                     # one drone at a time: no overlap between the masks
                    drones[c].MakeVisible()
                    isolated[c] = capture(t)
                    drones[c].MakeInvisible()
                for d in drones.values():
                    d.MakeVisible()
                reference.MakeVisible()
                record["check_reference_hidden"] = True
                for c in L.ORDER:
                    cv2.imwrite(str(sdir / f"{args.case}_mid_only_{c}.png"), isolated[c])
                    found = locate_drone(isolated[c], background)
                    raw_c = isolated[c]
                    proj = ov.px[c][k]
                    entry = {"projected_px": proj.tolist(), "time_code": k}
                    if found is None:
                        entry["found"] = False
                    else:
                        cen, bbox, area, mask = found
                        # colour from the mask eroded by 1 px: edge pixels are blended with the background
                        core = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                        if core.sum() < 30:
                            core = mask
                        hue, n_px = hue_deg_of(raw_c[core])
                        med = np.median(raw_c[core].reshape(-1, 3), axis=0)[::-1]       # RGB
                        x, y, w, h = bbox
                        entry.update(found=True, rendered_centroid_px=[float(cen[0]), float(cen[1])],
                                     offset_px=[float(cen[0] - proj[0]), float(cen[1] - proj[1])],
                                     offset_norm_px=float(np.hypot(cen[0] - proj[0], cen[1] - proj[1])),
                                     bbox_xywh=list(bbox), area_px=area,
                                     projected_inside_bbox=bool(x <= proj[0] <= x + w and y <= proj[1] <= y + h),
                                     rendered_hue_deg=hue, saturated_pixels=n_px,
                                     rendered_median_rgb=[float(v) for v in med],
                                     legend_rgb=[round(255 * v) for v in L.COLORS_SRGB[c]],
                                     legend_hue_deg=legend_hue_deg(c),
                                     hue_gap_deg=None if hue is None else _hue_gap(hue, legend_hue_deg(c)))
                        dbg[mask] = (0.5 * dbg[mask] + 0.5 * np.array(ov.colors[c])).astype(np.uint8)
                        cv2.rectangle(dbg, (x, y), (x + w, y + h), ov.colors[c], 1)
                        cv2.drawMarker(dbg, (int(round(proj[0])), int(round(proj[1]))), WHITE, cv2.MARKER_CROSS,
                                       24, 2)
                    record["offsets_mid"][c] = entry
                    print(f"[M9.v2.render] mid {c:10s} projected {np.round(proj, 1).tolist()} -> "
                          + (f"rendered {np.round(entry['rendered_centroid_px'], 1).tolist()}, offset "
                             f"{entry['offset_norm_px']:.1f} px, area {entry['area_px']}, inside bbox "
                             f"{entry['projected_inside_bbox']}, hue {entry['rendered_hue_deg']} vs legend "
                             f"{entry['legend_hue_deg']:.0f} (gap {entry['hue_gap_deg']}), median RGB "
                             f"{np.round(entry['rendered_median_rgb']).astype(int).tolist()} vs legend "
                             f"{entry['legend_rgb']}"
                             if entry["found"] else "NOT FOUND"), flush=True)
                cv2.imwrite(str(sdir / f"{args.case}_mid_offsets.png"), dbg)
            record["tags_without_free_spot"] = ov.tag_clamped
            suffix = ("" if wanted == ["start", "mid", "end"] else "_" + "_".join(wanted)) + "_refhidden"
            (OUT / f"{args.case}_stills{suffix}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
            print("[M9.v2.render] RESULT: DONE (stills)", flush=True)
            return

        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.videoencoding")
        app.update()
        from video_encoding import get_video_encoding_interface

        encoder = get_video_encoding_interface()
        if encoder is None:
            raise RuntimeError("omni.videoencoding interface not available")
        mp4 = OUT / f"{args.case}.mp4"
        n_hold = int(round(HOLD_SECONDS * args.fps))
        n_play = int(np.floor(seconds * args.fps + 1e-9)) + 1
        n_frames = n_hold + n_play
        if not encoder.start_encoding(str(mp4), float(args.fps), n_frames, True):
            raise RuntimeError("omni.videoencoding start_encoding failed")
        means, failed = [], 0
        try:
            hold_frame = ov.draw(capture(0.0), 0.0, hold=True)
            hold_rgba = np.ascontiguousarray(cv2.cvtColor(hold_frame, cv2.COLOR_BGR2RGBA))
            for _ in range(n_hold):          # same time code: one render, encoded n_hold times
                failed += int(not encoder.encode_next_frame_from_buffer(hold_rgba, L.W, L.H))
            means.append(float(hold_frame.mean()))
            for i in range(n_play):
                t = i / args.fps
                frame = ov.draw(capture(t), t)
                means.append(float(frame.mean()))
                rgba = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
                failed += int(not encoder.encode_next_frame_from_buffer(rgba, L.W, L.H))
                if i % 60 == 0:
                    print(f"[M9.v2.render] frame {i}/{n_play} t={t:.2f} s mean {means[-1]:.1f}", flush=True)
        finally:
            encoder.finalize_encoding()
        facts = mp4_facts(mp4)
        info = {"case": args.case, "file": mp4.name, "label": L.demo_config.LABEL,
                "encoder": "omni.videoencoding (NVENC H.264)", "fps": args.fps, "size": [L.W, L.H],
                "frames_encoded": n_frames, "hold_frames": n_hold, "hold_seconds": HOLD_SECONDS,
                "play_frames": n_play, "frames_failed": failed, "mp4_boxes": facts, "seconds": n_frames / args.fps,
                "replay_end_time_code": end_code, "replay_hz": hz, "rt_subframes": args.rt_subframes,
                "frame_mean_brightness_min_max": [min(means), max(means)],
                "layer": f"results/m9_demo/video_v2/{layer.name}",
                "layer_sha256": hashlib.sha256(layer.read_bytes()).hexdigest(), "layer_check": layer_check,
                "layout": f"results/m9_demo/video_v2/{args.case}_layout.json",
                "camera": "/World/M9Replay/CameraV2 (fixed)",
                "overlay": "OpenCV, drawn by code on top of the render (not occluded by the scene)",
                "tags_without_free_spot": ov.tag_clamped,
                "bytes": mp4.stat().st_size, "sha256": hashlib.sha256(mp4.read_bytes()).hexdigest()}
        (OUT / f"{args.case}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        print(f"[M9.v2.render] written {mp4.name}: {n_frames} frames ({n_hold} hold + {n_play}), {failed} failed",
              flush=True)
        if not verify(args.case, args.fps):
            raise RuntimeError("video check failed")
        print("[M9.v2.render] RESULT: DONE", flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        print("[M9.v2.render] RESULT: FAIL", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
