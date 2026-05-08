import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_public_package.py"
spec = importlib.util.spec_from_file_location("verify_public_package", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class VerifyPublicPackageTests(unittest.TestCase):
    def test_expected_cases_are_curated(self):
        self.assertEqual(len(module.EXPECTED_CASES), 3)

    def test_forbidden_suffixes_cover_raw_and_model_artifacts(self):
        self.assertIn(".zarr", module.FORBIDDEN_SUFFIXES)
        self.assertIn(".safetensors", module.FORBIDDEN_SUFFIXES)


if __name__ == "__main__":
    unittest.main()
