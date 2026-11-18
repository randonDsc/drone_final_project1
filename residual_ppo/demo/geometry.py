"""Distances to the exported obstacle shapes (demo/scene_geometry.py). NumPy only. DESCRIPTIVE DEMO, NOT eval-v1.

- clearance(points): exact Euclidean distance from each point to the nearest obstacle SURFACE (0 inside).
- contacts(points): the team's collision convention (path_sac_config.NAVIGATION["collision_half_size"], an
  axis-aligned box of half size 0.25 / 0.25 / 0.06 m around the body origin; body tilt ignored) against the exact
  shapes. Test: the obstacle point closest to the body origin lies inside the box, OR one of 27 box sample points
  (corners, edge midpoints, face centres, centre) lies inside the obstacle. Grazing contacts between those samples
  can be missed; it never reports contact where there is none.
"""
import json
from pathlib import Path

import numpy as np


class Obstacles:
    def __init__(self, shapes):
        self.shapes = shapes
        self.paths = [s["path"] for s in shapes]
        by = {k: [s for s in shapes if s["kind"] == k] for k in ("box", "cylinder", "sphere")}
        b, c, s = by["box"], by["cylinder"], by["sphere"]
        self.box_idx = np.array([i for i, x in enumerate(shapes) if x["kind"] == "box"], dtype=int)
        self.cyl_idx = np.array([i for i, x in enumerate(shapes) if x["kind"] == "cylinder"], dtype=int)
        self.sph_idx = np.array([i for i, x in enumerate(shapes) if x["kind"] == "sphere"], dtype=int)
        self.box_c = np.array([x["center"] for x in b]).reshape(-1, 3)
        self.box_u = np.array([x["axes"] for x in b]).reshape(-1, 3, 3)
        self.box_h = np.array([x["half"] for x in b]).reshape(-1, 3)
        self.cyl_c = np.array([x["center"] for x in c]).reshape(-1, 3)
        self.cyl_a = np.array([x["axis"] for x in c]).reshape(-1, 3)
        self.cyl_h = np.array([x["half_height"] for x in c], dtype=float)
        self.cyl_r = np.array([x["radius"] for x in c], dtype=float)
        self.sph_c = np.array([x["center"] for x in s]).reshape(-1, 3)
        self.sph_r = np.array([x["radius"] for x in s], dtype=float)
        self.aabb = np.array([x["aabb"] for x in shapes], dtype=float).reshape(-1, 2, 3)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["shapes"])

    # ----- closest points, shape (n_points, n_shapes, 3) per kind -----

    def _closest_box(self, p):
        d = p[:, None, :] - self.box_c[None]                              # (n, m, 3)
        local = np.einsum("nmk,mjk->nmj", d, self.box_u)                  # coordinates along the box axes
        q = np.clip(local, -self.box_h[None], self.box_h[None])
        return self.box_c[None] + np.einsum("nmj,mjk->nmk", q, self.box_u)

    def _closest_cyl(self, p):
        d = p[:, None, :] - self.cyl_c[None]
        t = np.einsum("nmk,mk->nm", d, self.cyl_a)
        radial = d - t[..., None] * self.cyl_a[None]
        rn = np.linalg.norm(radial, axis=-1)
        scale = np.where(rn > self.cyl_r[None], self.cyl_r[None] / np.maximum(rn, 1e-12), 1.0)
        tc = np.clip(t, -self.cyl_h[None], self.cyl_h[None])
        return self.cyl_c[None] + tc[..., None] * self.cyl_a[None] + radial * scale[..., None]

    def _closest_sph(self, p):
        d = p[:, None, :] - self.sph_c[None]
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        scale = np.where(n > self.sph_r[None, :, None], self.sph_r[None, :, None] / np.maximum(n, 1e-12), 1.0)
        return self.sph_c[None] + d * scale

    def _per_kind(self, p, chunk=256):
        """Yield (indices into shapes, closest points (n, m, 3)) per kind, chunked over points."""
        for idx, fn, n in ((self.box_idx, self._closest_box, len(self.box_c)),
                           (self.cyl_idx, self._closest_cyl, len(self.cyl_c)),
                           (self.sph_idx, self._closest_sph, len(self.sph_c))):
            if n:
                yield idx, fn

    def clearance(self, points, chunk=128):
        """Distance to the nearest obstacle surface [m] (0 inside) and that obstacle's index, per point."""
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        best = np.full(len(p), np.inf)
        who = np.full(len(p), -1, dtype=int)
        for start in range(0, len(p), chunk):
            sl = slice(start, start + chunk)
            for idx, fn in self._per_kind(p):
                dist = np.linalg.norm(p[sl][:, None, :] - fn(p[sl]), axis=-1)       # (c, m)
                j = np.argmin(dist, axis=1)
                dj = dist[np.arange(len(j)), j]
                better = dj < best[sl]
                best[sl] = np.where(better, dj, best[sl])
                who[sl] = np.where(better, idx[j], who[sl])
        return best, who

    def contacts(self, positions, half_size, chunk=64):
        """Per position: index of an obstacle in contact with the body box, or -1."""
        p = np.asarray(positions, dtype=float).reshape(-1, 3)
        h = np.asarray(half_size, dtype=float).reshape(3)
        grid = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")).reshape(3, -1).T * h
        hit = np.full(len(p), -1, dtype=int)
        lo_all, hi_all = self.aabb[:, 0], self.aabb[:, 1]
        for i0 in range(0, len(p), chunk):
            block = p[i0:i0 + chunk]
            # Cheap AABB prefilter: obstacles whose box overlaps the body box of any point in the block.
            blo, bhi = block.min(0) - h, block.max(0) + h
            near = np.where(np.all(lo_all <= bhi, axis=1) & np.all(hi_all >= blo, axis=1))[0]
            if near.size == 0:
                continue
            sub = Obstacles([self.shapes[k] for k in near])
            for k, q in enumerate(block):
                if hit[i0 + k] >= 0:
                    continue
                for idx, fn in sub._per_kind(q[None]):
                    c = fn(q[None])[0]                                                 # (m, 3)
                    inside = np.all(np.abs(c - q) <= h + 1e-12, axis=1)
                    if inside.any():
                        hit[i0 + k] = near[idx[np.argmax(inside)]]
                        break
                    samples = q + grid
                    cs = fn(samples)                                                   # (27, m, 3)
                    within = np.linalg.norm(cs - samples[:, None, :], axis=-1) <= 1e-9
                    if within.any():
                        hit[i0 + k] = near[idx[np.argmax(within.any(axis=0))]]
                        break
        return hit
