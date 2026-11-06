"""Self-contained static collision scenes assembled around the unchanged X2."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .config import ExperimentConfig, SceneConfig, TaskConfig


ROBOT_DIR = Path(__file__).parent/"models"/"skydio_x2"
SCENE_DIR = Path(__file__).parent/"models"/"scenes"


def load_manifest(config):
    if config.name == "open":
        return None
    path = Path(config.manifest).expanduser() if config.name == "custom" else SCENE_DIR/(config.name+".json")
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get("schema_version") != 1 or value.get("units") != "metres" or value.get("up_axis") != "Z":
        raise ValueError("Scene manifest must use schema 1, metres and Z-up")
    names = set()
    for kind in ("boxes", "planes"):
        if not isinstance(value.get(kind), list):
            raise ValueError("Manifest requires boxes and planes lists")
        for obstacle in value[kind]:
            if not isinstance(obstacle.get("path"), str) or obstacle["path"] in names:
                raise ValueError("Obstacle paths must be unique strings")
            names.add(obstacle["path"])
            if kind == "boxes":
                lower, upper = np.asarray(obstacle["lower"]), np.asarray(obstacle["upper"])
                if lower.shape != (3,) or upper.shape != (3,) or not np.all(np.isfinite([lower, upper])) or np.any(upper <= lower):
                    raise ValueError("Box bounds must be finite, ordered XYZ vectors")
                rgba = obstacle.get("rgba", [.6, .5, .35, 1])
                if len(rgba) != 4 or not np.all(np.isfinite(rgba)) or np.any(np.array(rgba) < 0) or np.any(np.array(rgba) > 1):
                    raise ValueError("RGBA colors must contain four values in [0,1]")
            else:
                normal, offset = np.asarray(obstacle["normal"], float), obstacle["offset"]
                if normal.shape != (3,) or not np.isfinite(offset) or not np.all(np.isfinite(normal)) or not np.isclose(np.linalg.norm(normal), 1):
                    raise ValueError("Plane normals must be finite unit vectors")
                if not np.isclose(abs(normal).max(), 1):
                    raise ValueError("Tilted infinite planes are unsupported")
    return dict(data=value, path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest())


def build_model(config):
    manifest = load_manifest(config)
    if manifest is None:
        return mujoco.MjModel.from_xml_path(str(ROBOT_DIR/"scene.xml")), None, {}
    root = ET.parse(ROBOT_DIR/"x2.xml").getroot()
    root.set("model", "Skydio X2 / "+manifest["data"]["name"])
    root.find("compiler").set("assetdir", str((ROBOT_DIR/"assets").resolve()))
    scene = ET.parse(ROBOT_DIR/"scene.xml").getroot()
    for child in scene:
        if child.tag == "include":
            continue
        child = deepcopy(child)
        if child.tag == "worldbody":
            for geom in list(child.findall("geom")):
                child.remove(geom)  # The manifest explicitly supplies its floor.
        root.append(child)
    world = ET.SubElement(root, "worldbody")
    sources = {}
    for i, box in enumerate(manifest["data"]["boxes"]):
        lower, upper = np.array(box["lower"]), np.array(box["upper"])
        name = f"environment_box_{i}"
        ET.SubElement(world, "geom", name=name, type="box", pos=_numbers((upper+lower)/2),
                      size=_numbers((upper-lower)/2), rgba=_numbers(box.get("rgba", [.6, .5, .35, 1])))
        sources[name] = box["path"]
    for i, plane in enumerate(manifest["data"]["planes"]):
        normal = np.array(plane["normal"])
        rotation = _plane_quaternion(normal)
        name = f"environment_plane_{i}"
        ET.SubElement(world, "geom", name=name, type="plane", pos=_numbers(normal*plane["offset"]),
                      quat=_numbers(rotation), size="0 0 .05", material="groundplane")
        sources[name] = plane["path"]
    # Use metre-scale clipping, independent of the warehouse's overall extent.
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "map", znear="0.001", zfar="10")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.vis.map.znear = .01/model.stat.extent
    model.vis.map.zfar = 100/model.stat.extent
    return model, manifest, sources


def _numbers(value):
    return " ".join(format(float(x), ".17g") for x in value)


def _plane_quaternion(normal):
    # Rotation carrying the local +Z plane normal onto the requested axis.
    if np.allclose(normal, [0, 0, -1]):
        return [0, 1, 0, 0]
    q = np.r_[1+normal[2], np.cross([0, 0, 1], normal)]
    return q/np.linalg.norm(q)


def navigation_preset(scene="blocks"):
    scene_config = SceneConfig(name=scene)
    manifest = load_manifest(scene_config)
    if manifest is None:
        raise ValueError("Navigation presets require an obstacle scene")
    mission = manifest["data"]["mission"]
    warehouse = scene == "warehouse"
    task = TaskConfig(kind="dijkstra", start=tuple(mission["start"]), goal=tuple(mission["goal"]),
                      speed=.8 if warehouse else .6, acceleration=.5 if warehouse else .4,
                      start_hold=1 if warehouse else 3, end_hold=5 if warehouse else 8,
                      initial_position_range=(0, 0, 0), initial_velocity_range=(0, 0, 0),
                      initial_angle_range_degrees=(0, 0, 0), initial_angular_velocity_range=(0, 0, 0))
    return ExperimentConfig(scene=scene_config, task=task, max_seconds=600 if warehouse else 120)
