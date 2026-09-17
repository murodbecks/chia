"""Recovery preserves provenance and never invents successful/pending work."""
import json
from pathlib import Path
import tempfile
import unittest

from common import digest, write_json
from recovery import capacity_failure, load_checkpoint, next_round, restore_jobs, retry_delay, recover_capacity_error


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def job(self, name, result=None):
        source = {"backend.py": "unchanged"}
        write_json(self.root / "trials" / name / "source.json", source)
        with (self.root / "events.jsonl").open("a") as f:
            f.write(json.dumps({"event": "started", "job": name, "kind": "smoke",
                                "source_sha256": digest(source)}) + "\n")
        if result is not None:
            write_json(self.root / "trials" / name / "result.json", result)

    def test_restore_keeps_success_and_failure_evidence(self):
        good = {"returncode": 0, "evaluation": {"passed": True}}
        bad = {"returncode": 124}
        self.job("good", good)
        self.job("bad", bad)
        states = restore_jobs(self.root)
        self.assertEqual(states["good"]["result"], good)
        self.assertEqual(states["bad"]["result"], bad)
        self.assertNotIn("ref", states["good"])

    def test_lost_job_is_interrupted_without_overwriting_history(self):
        self.job("lost")
        self.assertTrue(restore_jobs(self.root)["lost"]["result"]["interrupted"])
        self.assertFalse((self.root / "trials/lost/result.json").exists())

    def test_tampered_source_is_rejected(self):
        self.job("job")
        write_json(self.root / "trials/job/source.json", {"backend.py": "changed"})
        with self.assertRaises(ValueError):
            restore_jobs(self.root)

    def test_round_numbers_cannot_overwrite_previous_records(self):
        write_json(self.root / "rounds/015.json", {})
        write_json(self.root / "rounds/003.json", {})
        self.assertEqual(next_round(self.root), 16)

    def test_only_capacity_failures_get_capacity_retry(self):
        self.assertTrue(capacity_failure({"success": False, "error": "Selected model is at capacity"}))
        for record in ({"success": True, "error": "at capacity"},
                       {"success": False, "error": "Authentication failed"},
                       {"success": False, "error": "device failed"}):
            self.assertFalse(capacity_failure(record))
        self.assertEqual([retry_delay(i) for i in range(1, 7)], [30, 60, 120, 240, 300, 300])

    def test_capacity_error_survives_truncated_wrapper_exception(self):
        record = {"success": False, "round": 15, "error": "unknown: truncated preamble"}
        logs = self.root / "agent-logs/15"
        logs.mkdir(parents=True)
        (logs / "codex.log").write_text("[Response]\nat capacity is an example\n")
        self.assertFalse(capacity_failure(recover_capacity_error(self.root, record)))
        (logs / "codex.log").write_text("[Error]\nSelected model is at capacity. Please try a different model.\n")
        self.assertTrue(capacity_failure(recover_capacity_error(self.root, record)))
        self.assertEqual(recover_capacity_error(self.root, {**record, "success": True}),
                         {**record, "success": True})

    def test_checkpoint_verifies_corpus_evaluator_and_submission(self):
        corpus = {"data": "original"}
        workspace = self.root / "workspace"
        workspace.mkdir()
        frozen = self.root / "harness"
        frozen.mkdir()
        for name in ("suite.py", "assess.py", "candidate_worker.py", "reference_worker.py", "remote.py"):
            (frozen / name).write_bytes((Path(__file__).parent / name).read_bytes())
        write_json(self.root / "manifest.json", {"workspace": str(workspace),
            "corpus_sha256": digest(corpus),
            "harness_sha256": digest({p.name: p.read_text() for p in frozen.glob("*.py")})})
        self.assertEqual(load_checkpoint(self.root, corpus)[1], workspace)
        with self.assertRaises(ValueError):
            load_checkpoint(self.root, {"data": "changed"})
        write_json(self.root / "submission.json", {})
        with self.assertRaises(ValueError):
            load_checkpoint(self.root, corpus)
        (self.root / "submission.json").unlink()
        (frozen / "assess.py").write_text("weakened")
        with self.assertRaises(ValueError):
            load_checkpoint(self.root, corpus)


if __name__ == "__main__":
    unittest.main()
