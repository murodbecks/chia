"""Artifact-level evaluator tests; run in an existing NumPy runtime."""
import json
from pathlib import Path
import tempfile
import unittest
try:
    import numpy as np
except ImportError:
    np = None
from assess import assess


@unittest.skipIf(np is None, "NumPy tests run on the trusted TT host")
class ArtifactTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        (root / "output").mkdir()
        case = dict(name="x", kind="boundary", numerical=True, texts=["hello"],
                    target_id=256047, max_new_tokens=8)
        spec = dict(stage="development", gates=dict(nrmse=.04), cases=[case])
        arrays = {"x__encoder": np.ones((1, 4, 2), np.float32),
                  "x__logits": np.ones((1, 6), np.float32),
                  "x__tokens": np.array([[2, 256047, 42, 2]])}
        (root / "suite.json").write_text(json.dumps(spec))
        np.savez(root / "expected.npz", **arrays)
        np.savez(root / "output/actual.npz", **arrays)
        (root / "output/measurements.json").write_text(json.dumps({"cases": [dict(name="x", seconds=[1, 2, 3, 4, 5])]}))
        (root / "reference.json").write_text(json.dumps({"cases": []}))
        return arrays, spec

    def test_matching_artifacts_pass(self):
        with tempfile.TemporaryDirectory() as root:
            self.fixture(root)
            self.assertTrue(assess(root)["passed"])

    def test_large_error_and_nonfinite_fail(self):
        for value in (3, float("nan"), float("inf")):
            with tempfile.TemporaryDirectory() as root:
                arrays, _ = self.fixture(root)
                arrays["x__encoder"][:] = value
                np.savez(Path(root) / "output/actual.npz", **arrays)
                self.assertFalse(assess(root)["passed"])

    def test_missing_array_fails(self):
        with tempfile.TemporaryDirectory() as root:
            arrays, _ = self.fixture(root)
            del arrays["x__logits"]
            np.savez(Path(root) / "output/actual.npz", **arrays)
            self.assertFalse(assess(root)["passed"])

    def test_wrong_shape_fails(self):
        with tempfile.TemporaryDirectory() as root:
            arrays, _ = self.fixture(root)
            arrays["x__encoder"] = np.ones((1, 3, 2))
            np.savez(Path(root) / "output/actual.npz", **arrays)
            self.assertFalse(assess(root)["passed"])

    def test_quality_cannot_pass_without_scoring(self):
        with tempfile.TemporaryDirectory() as root:
            _, spec = self.fixture(root)
            spec["stage"] = "full"
            (Path(root) / "suite.json").write_text(json.dumps(spec))
            result = assess(root)
            self.assertFalse(result["passed"])
            self.assertFalse(result["quality_evaluated"])

    def test_fabricated_timing_fails(self):
        with tempfile.TemporaryDirectory() as root:
            self.fixture(root)
            (Path(root) / "output/measurements.json").write_text(json.dumps({"cases": [dict(name="x", seconds=[0])]}))
            self.assertFalse(assess(root)["passed"])


if __name__ == "__main__":
    unittest.main()
