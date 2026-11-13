"""M1.4: every shared module must resolve inside residual_ppo/, never the old drone_code/.

No Isaac Sim needed: importlib.util.find_spec locates a module without executing it, so this
also works for drone_sim (which imports omni at module level). Names such as controller,
sim_config, motor and drone_diagnostics also exist in the old drone_code/ PD experiments;
importing one of those silently would invalidate the experiment.

Run from residual_ppo/ with either interpreter:
    python -m unittest discover -s tests -p "test_*.py" -v
"""
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RESIDUAL_DIR.parent
sys.path.insert(0, str(RESIDUAL_DIR))

SHARED = [
    "path_sac_env", "path_sac_policy", "path_sac_config", "path_reference",
    "hover_env", "hover_task", "hover_ppo_config", "sensor_config",
    "state_estimation", "model_contract", "controller", "sim_config",
    "drone_sim", "motor", "wind", "drone_diagnostics",
    "hover_sac_policy", "hover_sac_config",
]
# Same names in the old PD experiments (drone_code/); these are the dangerous ones.
SHADOWED = ["controller", "sim_config", "motor", "drone_diagnostics"]


def load_manifest():
    return json.loads((RESIDUAL_DIR / "SHARED_MANIFEST.json").read_text(encoding="utf-8"))


def git_blob(rel_path):
    out = subprocess.run(["git", "-C", str(REPO_DIR), "hash-object", "--", rel_path],
                         capture_output=True, check=True)
    return out.stdout.decode("ascii").strip()


class ImportPathTests(unittest.TestCase):
    def test_residual_ppo_is_first_on_sys_path(self):
        self.assertEqual(Path(sys.path[0]).resolve(), RESIDUAL_DIR)

    def test_every_shared_module_resolves_inside_residual_ppo(self):
        for name in SHARED:
            with self.subTest(module=name):
                spec = importlib.util.find_spec(name)
                self.assertIsNotNone(spec, f"{name} not found")
                self.assertEqual(Path(spec.origin).resolve().parent, RESIDUAL_DIR, spec.origin)

    def test_shadowed_names_are_not_the_old_pd_code(self):
        old = REPO_DIR / "drone_code"
        for name in SHADOWED:
            with self.subTest(module=name):
                origin = Path(importlib.util.find_spec(name).origin).resolve()
                self.assertNotEqual(origin.parent, old.resolve(), origin)

    def test_no_drone_code_on_sys_path(self):
        bad = [p for p in sys.path if p.rstrip("\\/").lower().endswith("drone_code")]
        self.assertEqual(bad, [])

    def test_already_imported_shared_modules_come_from_residual_ppo(self):
        # Catches a module imported earlier in this process (e.g. by another test) from elsewhere.
        for name in SHARED:
            mod = sys.modules.get(name)
            if mod is None:
                continue
            with self.subTest(module=name):
                self.assertEqual(Path(mod.__file__).resolve().parent, RESIDUAL_DIR, mod.__file__)


class ManifestTests(unittest.TestCase):
    def test_controller_blob_matches_manifest(self):
        manifest = load_manifest()
        self.assertEqual(git_blob("residual_ppo/controller.py"), manifest["files"]["controller.py"])
        self.assertEqual(manifest["files"]["controller.py"], "e34bd6193d0df880ed672fc3eb7947b8f7da4bb4")

    def test_full_manifest_check_passes(self):
        spec = importlib.util.spec_from_file_location("sync_shared", RESIDUAL_DIR / "tools" / "sync_shared.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["sync_shared"] = module  # dataclasses need the module registered before exec
        spec.loader.exec_module(module)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = module.check()
        self.assertEqual(rc, 0, buf.getvalue())


if __name__ == "__main__":
    unittest.main()
