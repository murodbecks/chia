"""Host-only contract, isolation and orchestration regression tests (stdlib)."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from assess import canonical_tokens, latency_summary
from common import digest, safe_path, snapshot
import suite
from workflow import IsolatedCodex, PortingTools, codex_options
from candidate_worker import memory_snapshot
from types import SimpleNamespace
from replay import load_trial


class BoundaryTests(unittest.TestCase):
    def test_traversal_and_absolute_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for path in ("../x", "/etc/passwd", "a/../../x", "a\\b", ""):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    safe_path(root, path)

    def test_symlink_read_and_write_rejected(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as other:
            (Path(root) / "link").symlink_to(other, target_is_directory=True)
            with self.assertRaises(ValueError):
                safe_path(root, "link/secret")
            with self.assertRaises(ValueError):
                snapshot(root)

    def test_nested_paths_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(safe_path(root, "src/backend.py"), Path(root).resolve() / "src/backend.py")

    def test_snapshot_determinism(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "backend.py").write_text("x = 1\n")
            (Path(root) / ".private").write_text("not part of source")
            self.assertEqual(snapshot(root), {"backend.py": "x = 1\n"})
            self.assertEqual(digest(snapshot(root)), digest(snapshot(root)))

    def test_nonfinite_manifest_rejected(self):
        with self.assertRaises(ValueError):
            digest({"gate": float("nan")})

    def test_codex_has_no_bypass_or_host_tools(self):
        agent = IsolatedCodex(model="gpt-6-astra", work_dir="/tmp/blank",
            dangerously_bypass_approvals_and_sandbox=False, ephemeral=True,
            approval_policy="never", extra_cli_args=codex_options("/tmp/blank"))
        cmd = agent._build_cmd()
        self.assertNotIn("--sandbox", cmd)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", cmd)
        for name in ("shell_tool", "unified_exec", "view_image", "apps", "plugins", "hooks"):
            self.assertIn(f"features.{name}=false", cmd)
        self.assertIn('default_permissions="portforge"', cmd)
        self.assertIn("--ignore-user-config", cmd)


class AcceptanceTests(unittest.TestCase):
    def test_memory_accounts_for_every_bank(self):
        view = SimpleNamespace(num_banks=8, total_bytes_allocated_per_bank=4096,
                               total_bytes_free_per_bank=8192)
        fake = SimpleNamespace(BufferType=SimpleNamespace(DRAM=0, L1=1, TRACE=2),
                               device=SimpleNamespace(get_memory_view=lambda device, kind: view))
        memory = memory_snapshot(fake, None)
        self.assertEqual(memory["DRAM"]["total_allocated_bytes"], 32768)
        self.assertEqual(memory["TRACE"]["num_banks"], 8)

    def test_valid_eos_padding(self):
        self.assertEqual(canonical_tokens([2, 256047, 42, 2, 1], 256047, 8), [2, 256047, 42, 2])

    def test_invalid_sequences(self):
        for row in ([2, 3, 42, 2], [2, 256047, 42], [2, 256047, 1, 2],
                    [2, 256047, 42, 2, 43], [2, 256047, 999999, 2]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                canonical_tokens(row, 256047, 8)

    def test_valid_length_cap(self):
        self.assertEqual(canonical_tokens([2, 256047, 42], 256047, 2), [2, 256047, 42])

    def test_invalid_timings_fail(self):
        for sample in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                latency_summary(sample)

    def test_percentiles_are_explicit(self):
        self.assertEqual(latency_summary([1, 2, 3, 4, 5]),
                         {"p50_seconds": 3, "p95_seconds": 5, "samples": 5})


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.corpus = {l: [f"{l} row {i}" for i in range(1012)]
                       for l in ("eng_Latn", *suite.LANGUAGES)}
        self.expected = digest(self.corpus)

    def make(self, stage):
        with patch.object(suite, "CORPUS_SHA256", self.expected):
            return suite.make_suite(self.corpus, stage, tuple(range(32)))

    def test_sample_sizes(self):
        for stage, rows in (("smoke", 1), ("development", 32), ("qualification", 1024), ("full", 8096)):
            plan = self.make(stage)
            self.assertEqual(sum(len(c["texts"]) for c in plan["cases"] if c["kind"] == "quality"), rows)

    def test_behavior_targets_exist_and_align(self):
        for stage in ("development", "final"):
            plan = self.make(stage)
            names = {c["name"]: c for c in plan["cases"]}
            self.assertEqual(len(names), len(plan["cases"]))
            for case in plan["cases"]:
                if "compare_to" in case:
                    self.assertEqual(case["texts"], [names[case["compare_to"]]["texts"][i]
                                                   for i in case["compare_rows"]])

    def test_final_inputs_are_new_after_full(self):
        full = {text for c in self.make("full")["cases"] for text in c["texts"]}
        final = {text for c in self.make("final")["cases"] if c["kind"] == "quality" for text in c["texts"]}
        self.assertFalse(full & final)

    def test_corpus_tamper_rejected(self):
        with self.assertRaises(ValueError):
            suite.make_suite(self.corpus, "smoke")

    def test_gates_cannot_change_across_stages(self):
        self.assertTrue(all(self.make(s)["gates"] == suite.GATES
                            for s in ("smoke", "development", "qualification", "full", "final")))


class ToolStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace = self.root / "source"
        self.workspace.mkdir()
        # Exercise tool implementation without starting an MCP/Ray server.
        self.tools = PortingTools.__new__(PortingTools)
        self.tools.workspace = self.workspace
        self.tools.run_dir = self.root
        self.tools.jobs = {}
        self.tools.final_requested = False

    def tearDown(self):
        self.temp.cleanup()

    def test_source_snapshot_survives_later_edits(self):
        self.tools.write("backend.py", "version = 1\n")
        files = snapshot(self.workspace)
        job = self.tools.start("run", lambda job: "ref", files)["job_id"]
        self.tools.write("backend.py", "version = 2\n")
        archived = json.loads((self.root / "trials" / job / "source.json").read_text())
        self.assertEqual(archived["backend.py"], "version = 1\n")

    def test_replay_rejects_tampered_source(self):
        job = self.tools.start("smoke", lambda job: "ref", {"backend.py": "original"})["job_id"]
        self.assertEqual(load_trial(self.root, job), ({"backend.py": "original"}, "smoke"))
        (self.root / "trials" / job / "source.json").write_text(json.dumps({"backend.py": "tampered"}))
        with self.assertRaises(ValueError):
            load_trial(self.root, job)

    def test_poll_does_not_resubmit_or_recollect(self):
        job = self.tools.start("run", lambda job: "ref", {})["job_id"]
        with patch("workflow.ray.wait", return_value=(["ref"], [])), patch("workflow.get", return_value={"returncode": 0}) as collect:
            self.assertEqual(self.tools.poll(job)["status"], "complete")
            self.assertEqual(self.tools.poll(job)["status"], "complete")
            collect.assert_called_once()

    def test_poll_pending_is_nonblocking(self):
        job = self.tools.start("run", lambda job: "ref", {})["job_id"]
        with patch("workflow.ray.wait", return_value=([], ["ref"])) as wait:
            self.assertEqual(self.tools.poll(job)["status"], "running")
            wait.assert_called_once_with(["ref"], timeout=0)

    def test_post_submission_write_rejected(self):
        self.tools.final_requested = True
        with self.assertRaises(ValueError):
            self.tools.write("backend.py", "pass")

    def test_seed_policy_is_read_only(self):
        for name in ("AGENTS.md", "TASK.md", "CONTRACT.md", "TT_PORTING_GUIDE.md"):
            with self.assertRaises(ValueError):
                self.tools.write(name, "lower the test thresholds")

    def test_submit_cannot_bypass_broad_gates(self):
        self.tools.write("backend.py", "pass")
        self.tools.write("REPORT.md", "I claim success")
        with self.assertRaises(ValueError):
            self.tools.submit()
        self.assertFalse((self.root / "submission.json").exists())


if __name__ == "__main__":
    unittest.main()
