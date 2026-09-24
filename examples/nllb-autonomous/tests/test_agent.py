"""Protocol, isolation settings and bounded role handoff; no live credentials."""
import ast
import inspect
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import agent


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {'NLLB_REPOSITORY_ROOT': '/workspace/original-repository'})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_archived_agent_preserves_original_repository_deny(self):
        with patch.object(agent, 'ROOT', Path('/tmp/run/harness')):
            self.assertEqual(agent.repository_root(), Path('/workspace/original-repository'))
            self.assertIn('"/workspace/original-repository"="deny"', ' '.join(agent.options()))

    def test_supervisor_protocol_accepts_only_advisory_decisions(self):
        valid = {"action": "continue", "instruction": "Run one tiny independent correctness check."}
        self.assertEqual(agent.parse_decision(json.dumps(valid)), valid)
        for value in (dict(valid, action="reset"), dict(valid, command="shell"),
                      dict(valid, instruction=" "), dict(valid, instruction="x" * 2001), []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                agent.parse_decision(json.dumps(value))
        with self.assertRaises(ValueError):
            agent.parse_decision('{"action":"continue","action":"pause","instruction":"x"}')

    def test_isolation_disables_native_tools_memory_and_repository_access(self):
        settings = agent.options(False)
        for name in ("shell_tool", "unified_exec", "apps", "plugins", "memories", "multi_agent"):
            self.assertIn(f"features.{name}=false", settings)
        self.assertIn('web_search="disabled"', settings)
        filesystem = next(s for s in settings if s.startswith("permissions.nllb_agent.filesystem="))
        for path in (agent.repository_root(), Path.home() / ".ssh", Path.home() / ".codex"):
            self.assertIn(json.dumps(str(path)) + '="deny"', filesystem)
        self.assertIn("permissions.nllb_agent.network.enabled=false", settings)

    def test_named_profile_removes_only_conflicting_sandbox_flag(self):
        with patch.object(agent.CodexLLM, "_build_cmd", return_value=["codex", "exec", "--sandbox", "read-only", "--ephemeral"]):
            instance = object.__new__(agent.IsolatedCodex)
            self.assertEqual(instance._build_cmd(), ["codex", "exec", "--ephemeral"])

    def test_error_categories(self):
        for text, category in (("Selected model is at capacity.", "capacity"),
                               ("Authentication failed", "authentication"),
                               ("subprocess timed out", "timeout"), ("other", "provider_error")):
            self.assertEqual(agent.error_kind(text), category)

    def test_capacity_recovered_from_terminal_log_without_echoing_it(self):
        with tempfile.TemporaryDirectory() as root:
            log = Path(root) / "agent-logs/worker/1/error.log"
            log.parent.mkdir(parents=True)
            log.write_text("[Error]\nSelected model is at capacity. Secret CLI context omitted.")
            with patch.object(agent, "IsolatedCodex") as provider:
                provider.return_value.prompt.return_value = SimpleNamespace(success=False, result="", stderr="truncated")
                result = agent.query(root, root, "worker", 1, "task", [], True, 600)
            self.assertEqual(result["error"], "capacity")
            self.assertNotIn("Secret", json.dumps(result))

    def test_supervisor_no_tools_empty_workspace_fail_closed_and_durable(self):
        with tempfile.TemporaryDirectory() as root:
            captured = {}
            def query(workspace, run_dir, role, turn, prompt, tools, web, timeout):
                captured.update(files=list(Path(workspace).iterdir()), tools=tools, web=web, timeout=timeout)
                return {"turn": turn, "success": True, "response": "not json", "error": ""}
            with patch.object(agent, "query", side_effect=query):
                result = agent.supervisor_node._chia_original(root, root, 3, {"pending_jobs": 1})
            self.assertEqual(captured, {"files": [], "tools": [], "web": False, "timeout": 120})
            self.assertEqual(result["decision"]["action"], "pause")
            self.assertFalse(result["success"])
            self.assertEqual(result["error"], "invalid_decision")
            self.assertEqual(len(result["attempt_records"]), 3)
            for relative in result["attempt_records"]:
                self.assertEqual(json.loads((Path(root) / relative).read_text())["response"], "not json")
            self.assertEqual(json.loads((Path(root) / "supervisor-turns/003.json").read_text()), result)

    def test_overlong_reply_is_preserved_and_repaired_without_truncation(self):
        original = json.dumps({"action": "continue", "instruction": "Bounded next step. " * 130})
        repaired = {"action": "continue", "instruction": "Collect the existing pending job; do not resubmit."}
        responses = [original, json.dumps(repaired)]
        with tempfile.TemporaryDirectory() as root:
            def query(workspace, run_dir, role, turn, prompt, tools, web, timeout):
                if role != "supervisor":
                    archived = json.loads((Path(root) / "supervisor-attempt-0-turns/003.json").read_text())
                    self.assertEqual(archived["response"], original)
                    self.assertTrue(archived["provider_success"])
                    self.assertFalse(archived["success"])
                    self.assertIn(json.dumps(original), prompt)
                    self.assertIn('"pending_jobs": 12', prompt)
                    self.assertIn(str(len(json.loads(original)['instruction'])) + ' characters', prompt)
                    self.assertTrue(prompt.endswith(agent.SUPERVISOR_OUTPUT))
                return {"turn": turn, "success": True, "response": responses.pop(0), "error": ""}
            with patch.object(agent, "query", side_effect=query) as mocked:
                result = agent.supervisor_node._chia_original(root, root, 3, {"pending_jobs": 12})
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual([c.args[2] for c in mocked.call_args_list], ["supervisor", "supervisor-repair-1"])
            self.assertEqual(result["decision"], repaired)
            self.assertTrue(result["success"])
            self.assertEqual(result["response"], json.dumps(repaired))

    def test_invalid_repairs_exhaust_bound_without_synthetic_continue(self):
        replies = ['{"action":"reset","instruction":"x"}',
                   '{"action":"continue","action":"pause","instruction":"x"}',
                   json.dumps({"action": "continue", "instruction": "x" * 2001})]
        with tempfile.TemporaryDirectory() as root, patch.object(agent, "query") as mocked:
            mocked.side_effect = [{"turn": 1, "success": True, "response": text, "error": ""} for text in replies]
            result = agent.supervisor_node._chia_original(root, root, 1, {})
            self.assertEqual(mocked.call_count, 3)
            self.assertFalse(result["success"])
            self.assertEqual(result["decision"]["action"], "pause")
            self.assertEqual(result["error"], "invalid_decision")
            self.assertEqual([json.loads((Path(root) / p).read_text())["response"]
                              for p in result["attempt_records"]], replies)

    def test_valid_pause_is_honored_initially_and_after_repair(self):
        pause = {"action": "pause", "instruction": "Hardware needs external intervention."}
        for repair in (False, True):
            with self.subTest(repair=repair), tempfile.TemporaryDirectory() as root:
                replies = (["not JSON"] if repair else []) + [json.dumps(pause)]
                with patch.object(agent, "query") as mocked:
                    mocked.side_effect = [{"turn": 1, "success": True, "response": text, "error": ""}
                                          for text in replies]
                    result = agent.supervisor_node._chia_original(root, root, 1, {})
                self.assertEqual(mocked.call_count, len(replies))
                self.assertTrue(result["success"])
                self.assertEqual(result["decision"], pause)

    def test_provider_failures_do_not_trigger_format_retries(self):
        for kind in ("capacity", "authentication", "timeout", "provider_error"):
            for repair in (False, True):
                with self.subTest(kind=kind, repair=repair), tempfile.TemporaryDirectory() as root:
                    replies = ([{"turn": 1, "success": True, "response": "bad", "error": ""}] if repair else [])
                    replies += [{"turn": 1, "success": False, "response": "incomplete", "error": kind}]
                    with patch.object(agent, "query", side_effect=replies) as mocked:
                        result = agent.supervisor_node._chia_original(root, root, 1, {})
                    self.assertEqual(mocked.call_count, len(replies))
                    self.assertEqual(result["error"], kind)
                    self.assertFalse(result["success"])
                    self.assertEqual(result["decision"]["action"], "pause")

    def test_repair_provider_logs_are_separate(self):
        with tempfile.TemporaryDirectory() as root, patch.object(agent, "IsolatedCodex") as provider:
            provider.return_value.prompt.return_value = SimpleNamespace(success=True, result="{}")
            for role in ("supervisor", "supervisor-repair-1", "supervisor-repair-2"):
                agent.query(root, root, role, 3, "task", [], False, 120)
            paths = [c.kwargs["log_dir"] for c in provider.call_args_list]
            self.assertEqual(len(set(paths)), 3)
            self.assertTrue(all(Path(path).parent.parent == Path(root) / "agent-logs" for path in paths))

    def test_schema_only_for_supervisor_and_repairs_with_exact_shape(self):
        with tempfile.TemporaryDirectory() as root, patch.object(agent, "IsolatedCodex") as provider:
            provider.return_value.prompt.return_value = SimpleNamespace(success=True, result="{}")
            for role in ("worker", "supervisor", "supervisor-repair-1", "supervisor-repair-2"):
                agent.query(root, root, role, 3, "task", [], False, 120)
                cli = provider.call_args.kwargs['extra_cli_args']
                if role == 'worker':
                    self.assertNotIn('--output-schema', cli)
                    continue
                path = Path(cli[cli.index('--output-schema') + 1])
                self.assertEqual(path.parent, (Path(root) / 'agent-logs' / role / '3').resolve())
                schema = json.loads(path.read_text())
                self.assertEqual(schema, agent.SUPERVISOR_SCHEMA)
                self.assertEqual(schema['properties']['action']['enum'], ['continue', 'pause'])
                self.assertEqual(schema['properties']['instruction'], {'type': 'string'})
                self.assertEqual(set(schema['required']), {'action', 'instruction'})
                self.assertFalse(schema['additionalProperties'])

    def test_custom_phase_instructions_cannot_erase_final_short_protocol(self):
        # Match phase_agent's instruction-assignment replacement, but execute
        # only this function so the test never imports a frozen campaign.
        tree = ast.parse(inspect.getsource(agent.supervisor_node._chia_original))
        node = tree.body[0]
        node.decorator_list = []
        node.body[0] = ast.parse('instructions = "CUSTOM PHASE: qualify all twelve gates.\\n"').body[0]
        ast.fix_missing_locations(tree)
        namespace = dict(vars(agent))
        exec(compile(tree, '<custom-phase-supervisor>', 'exec'), namespace)
        overlong = json.dumps({'action': 'continue', 'instruction': 'x' * 2196})
        concise = json.dumps({'action': 'continue', 'instruction': 'Repair benchmark checkpoint routing and test it.'})
        with tempfile.TemporaryDirectory() as root, patch.dict(namespace, {'query': unittest.mock.Mock()}) as _:
            mocked = namespace['query']
            mocked.side_effect = [{'turn': 1, 'success': True, 'response': text, 'error': ''}
                                  for text in (overlong, concise)]
            result = namespace['supervisor_node'](root, root, 1, {'objective': 'ENTIRE ROADMAP'})
            self.assertTrue(result['success'])
            for call in mocked.call_args_list:
                prompt = call.args[4]
                self.assertTrue(prompt.startswith('CUSTOM PHASE:'))
                self.assertTrue(prompt.endswith(agent.SUPERVISOR_OUTPUT))
                self.assertGreater(prompt.index('FINAL RESPONSE PROTOCOL'), prompt.index('ENTIRE ROADMAP'))
                self.assertIn('200-700 characters', prompt)
                self.assertIn('ONE bounded', prompt)
                self.assertIn('four sequential gates or six minutes', prompt)
            self.assertIn('2196 characters', mocked.call_args_list[1].args[4])
            failed = json.loads((Path(root) / result['attempt_records'][0]).read_text())
            self.assertEqual(failed['instruction_characters'], 2196)
            self.assertEqual(failed['response'], overlong)

    def test_schema_provider_error_does_not_retry_without_schema(self):
        with tempfile.TemporaryDirectory() as root, patch.object(agent, 'IsolatedCodex') as provider:
            provider.return_value.prompt.return_value = SimpleNamespace(
                success=False, result='', stderr='Unsupported response schema')
            result = agent.supervisor_node._chia_original(root, root, 1, {})
            self.assertEqual(provider.call_count, 1)
            self.assertIn('--output-schema', provider.call_args.kwargs['extra_cli_args'])
            self.assertFalse(result['success'])
            self.assertEqual(result['error'], 'provider_error')

    def test_three_json_length_failures_remain_fail_closed(self):
        lengths = (2196, 2044, 2035)  # Observed stopped0530 replies, without task content.
        replies = [json.dumps({'action': 'continue', 'instruction': 'x' * n}) for n in lengths]
        with tempfile.TemporaryDirectory() as root, patch.object(agent, 'query') as mocked:
            mocked.side_effect = [{'turn': 1, 'success': True, 'response': text, 'error': ''} for text in replies]
            result = agent.supervisor_node._chia_original(root, root, 1, {})
            self.assertEqual(mocked.call_count, 3)
            self.assertFalse(result['success'])
            self.assertEqual(result['decision']['action'], 'pause')
            records = [json.loads((Path(root) / p).read_text()) for p in result['attempt_records']]
            self.assertEqual([r['instruction_characters'] for r in records], list(lengths))
            self.assertEqual([r['response'] for r in records], replies)

    def test_worker_handoff_has_bounded_timeout_and_state_instruction(self):
        with tempfile.TemporaryDirectory() as root, patch.object(agent, "query") as query:
            query.return_value = {"turn": 2, "success": True, "response": "saved", "error": ""}
            agent.worker_node._chia_original(root, root, ["scoped-tool"], 2, "resume")
            arguments = query.call_args.args
            self.assertIn("STATE.md", arguments[4])
            self.assertTrue(arguments[4].endswith(agent.WORKER_HANDOFF))
            self.assertEqual(arguments[5:], (["scoped-tool"], True, 600))

    def test_phase_worker_retains_compact_batched_handoff_and_scoped_tools(self):
        tree = ast.parse(inspect.getsource(agent.worker_node._chia_original))
        node = tree.body[0]
        node.decorator_list = []
        node.body[0] = ast.parse('instructions = "CUSTOM PHASE: complete frozen gates.\\n"').body[0]
        ast.fix_missing_locations(tree)
        namespace = dict(vars(agent))
        exec(compile(tree, '<custom-phase-worker>', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as root, patch.dict(namespace, {'query': unittest.mock.Mock()}):
            query = namespace['query']
            query.return_value = {'turn': 2, 'success': True, 'response': 'saved', 'error': ''}
            namespace['worker_node'](root, root, ['scoped-tool'], 2, 'pending handle example')
            arguments = query.call_args.args
            self.assertTrue(arguments[4].startswith('CUSTOM PHASE:'))
            self.assertTrue(arguments[4].endswith(agent.WORKER_HANDOFF))
            self.assertLess(arguments[4].index('pending handle example'),
                            arguments[4].index('FINAL WORKING PROTOCOL'))
            self.assertEqual(arguments[5:], (['scoped-tool'], True, 600))


if __name__ == "__main__":
    unittest.main()
