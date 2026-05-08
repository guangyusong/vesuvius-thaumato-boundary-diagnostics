import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "create_vc3d_tifxyz_smoke_volpkg.py"
)
SPEC = importlib.util.spec_from_file_location("create_vc3d_tifxyz_smoke_volpkg", SCRIPT_PATH)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def write_tifxyz_fixture(root: pathlib.Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text('{"format":"tifxyz"}\n', encoding="utf-8")
    for axis in ("x", "y", "z"):
        (root / f"{axis}.tif").write_bytes(f"{axis}-fixture".encode("ascii"))


class CreateVC3DTifxyzSmokeVolpkgTests(unittest.TestCase):
    def test_build_volpkg_copies_inputs_and_writes_merge_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            left = root / "left_tifxyz"
            right = root / "right_tifxyz"
            write_tifxyz_fixture(left)
            write_tifxyz_fixture(right)

            out_dir = root / "smoke.volpkg"
            summary = builder.build_volpkg(
                out_dir,
                [
                    builder.SurfaceSpec("left_surface", left),
                    builder.SurfaceSpec("right_surface", right),
                ],
                [("left_surface", "right_surface")],
            )

            self.assertEqual(summary["surface_count"], 2)
            self.assertEqual(summary["row_count"], 1)
            self.assertEqual(summary["copied_file_count"], 8)
            self.assertTrue(summary["copied_inputs"])
            self.assertFalse(summary["symlinks_created"])
            merge = json.loads((out_dir / "merge.json").read_text(encoding="utf-8"))
            self.assertEqual(merge, {"rows": [["left_surface", "right_surface"]]})
            copied_x = out_dir / "paths" / "left_surface" / "x.tif"
            self.assertTrue(copied_x.is_file())
            self.assertFalse(copied_x.is_symlink())
            self.assertEqual(copied_x.read_bytes(), b"x-fixture")

    def test_validate_plan_rejects_missing_tifxyz_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            incomplete = root / "incomplete"
            incomplete.mkdir()
            (incomplete / "meta.json").write_text("{}\n", encoding="utf-8")
            complete = root / "complete"
            write_tifxyz_fixture(complete)

            with self.assertRaises(FileNotFoundError) as ctx:
                builder.validate_plan(
                    [
                        builder.SurfaceSpec("a", incomplete),
                        builder.SurfaceSpec("b", complete),
                    ],
                    [("a", "b")],
                )
            self.assertIn("x.tif", str(ctx.exception))
            self.assertIn("y.tif", str(ctx.exception))
            self.assertIn("z.tif", str(ctx.exception))

    def test_parse_surface_spec_rejects_path_like_names(self):
        with self.assertRaises(Exception):
            builder.parse_surface_spec("../bad=/tmp/input")

    def test_build_run_command_uses_bounded_smoke_defaults(self):
        command = builder.build_run_command(
            pathlib.Path("bin/vc_merge_tifxyz"),
            pathlib.Path("smoke.volpkg/merge.json"),
            ransac_iters=200,
            anchor_cap=128,
        )
        self.assertIn("--merge smoke.volpkg/merge.json", command)
        self.assertIn("--ransac-iters 200", command)
        self.assertIn("--ransac-seed 1", command)
        self.assertIn("--anchor-cap 128", command)
        self.assertIn("--strip-cols 1", command)


if __name__ == "__main__":
    unittest.main()
