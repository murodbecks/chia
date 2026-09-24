"""Checkpoint identity tests use tiny fake bytes, never download model weights."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import models


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = self.root / "registry.json"
        self.spec = dict(repo_id="test/model", revision="a" * 40,
                         checkpoint_format="pytorch_bin_sharded", weight_files=["one.bin", "two.bin"],
                         weight_index="index.json", architecture={"d_model": 4}, files={})
        self.write("config.json", b'{"d_model":4}')
        self.write("one.bin", b"first weights")
        self.write("two.bin", b"other weights")
        self.write("index.json", b'{"weight_map":{"encoder":"one.bin","decoder":"two.bin"}}')

    def write(self, name, data):
        (self.root / name).write_bytes(data)
        self.spec["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    def save(self):
        self.registry.write_text(json.dumps({"schema_version": 1, "models": {"tiny": self.spec}}))

    def verify(self):
        self.save()
        return models.verify_checkpoint(self.root, "tiny", registry=self.registry)

    def test_real_registry_has_three_distinct_pinned_architectures(self):
        small, medium, large = [models.model_spec(name) for name in ("600m", "1.3b-distilled", "3.3b")]
        self.assertEqual(small["architecture"]["encoder_layers"], 12)
        self.assertEqual(medium["architecture"]["encoder_layers"], 24)
        self.assertEqual(large["architecture"]["d_model"], 2048)
        self.assertEqual(len(large["weight_files"]), 3)
        self.assertIn("distilled", medium["repo_id"])
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            models.model_spec("1.3b")

    def test_sharded_and_single_file_receipts(self):
        self.assertEqual(len(self.verify()["verified_files"]), 4)
        self.spec.update(checkpoint_format="pytorch_bin", weight_files=["one.bin"], weight_index=None)
        self.assertEqual(set(self.verify()["verified_files"]), {"config.json", "one.bin"})

    def test_same_length_corruption_and_missing_shard(self):
        (self.root / "one.bin").write_bytes(b"wrong weights")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.verify()
        (self.root / "one.bin").unlink()
        with self.assertRaisesRegex(ValueError, "Missing file"):
            self.verify()

    def test_bad_index_membership_and_duplicate_tensor_names(self):
        for data in (b'{"weight_map":{"a":"../escape.bin"}}', b'{"weight_map":{}}',
                     b'{"weight_map":{"a":"one.bin","a":"two.bin"}}',
                     b'{"weight_map":{"a":["one.bin"]}}'):
            with self.subTest(index=data), self.assertRaises(ValueError):
                self.write("index.json", data)
                self.verify()

    def test_architecture_mismatch_even_with_matching_file_hash(self):
        self.write("config.json", b'{"d_model":8}')
        with self.assertRaisesRegex(ValueError, "Architecture mismatch"):
            self.verify()

    def test_unsafe_filename_unpinned_revision_and_missing_hash(self):
        self.spec["files"]["../outside"] = self.spec["files"]["one.bin"]
        with self.assertRaisesRegex(ValueError, "Invalid checkpoint filename"):
            self.verify()
        del self.spec["files"]["../outside"]
        self.spec["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "immutable commit"):
            self.verify()
        self.spec["revision"] = "a" * 40
        del self.spec["files"]["two.bin"]
        with self.assertRaisesRegex(ValueError, "missing a required file"):
            self.verify()

    def test_hf_style_cache_symlink_is_verified_by_content(self):
        (self.root / "one.bin").rename(self.root / "blob")
        (self.root / "one.bin").symlink_to(self.root / "blob")
        self.assertIn("one.bin", self.verify()["verified_files"])

    def test_download_is_pinned_anonymous_and_rejects_corrupt_cache(self):
        download = Mock(return_value=str(self.root))
        with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(snapshot_download=download)}), \
             patch.object(models, "model_spec", return_value=self.spec):
            receipt = models.download_checkpoint(self.root / "cache", "tiny")
            self.assertEqual(len(receipt["verified_files"]), 4)
            self.assertEqual(download.call_args.kwargs["revision"], self.spec["revision"])
            self.assertEqual(set(download.call_args.kwargs["allow_patterns"]), set(self.spec["files"]))
            self.assertIs(download.call_args.kwargs["token"], False)
            (self.root / "two.bin").write_bytes(b"wrong weights")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                models.download_checkpoint(self.root / "cache", "tiny")


if __name__ == "__main__":
    unittest.main()
