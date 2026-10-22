"""Offline contracts for pinned native coverage and independent PPU recovery."""

import ast
import copy
import json
import os
import re
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci/utils"))
import ppu_rerun as recovery
import upstream_e2e_inventory as inventory


def workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


class InventoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = inventory.load()

    def test_offline_source_and_reference_verification(self):
        inventory.validate(self.manifest)

    def test_serialization_preserves_all_source_names_and_identities(self):
        source = Mock()
        source.read_text.return_value = inventory.render(self.manifest)
        self.assertEqual(inventory.load(source), self.manifest)

    def test_identity_provenance_and_supporting_coverage(self):
        units = self.manifest["units"]
        self.assertEqual(len({u["id"] for u in units}), len(units))
        self.assertEqual(len([u for u in units if u["kind"] == "configured-e2e"]), 70)
        self.assertEqual(
            {u["origin"] for u in units}, {"upstream", "ppu", "fork-addition"}
        )
        self.assertTrue(
            any(u["origin"] == "upstream" and u["locally_modified"] for u in units)
        )
        self.assertTrue(any(u["kind"] == "supporting" for u in units))
        self.assertTrue(
            any(u["origin"] == "upstream" and u.get("disabled") for u in units)
        )
        self.assertTrue(
            any(u["kind"] == "collector-rule" and u["declared_case_ids"] for u in units)
        )
        self.assertIn("not selection or execution evidence", self.manifest["semantics"])

    def test_duplicate_and_drift_fail_closed(self):
        duplicate = copy.deepcopy(self.manifest)
        duplicate["units"].append(duplicate["units"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            inventory.validate(duplicate)
        drift = copy.deepcopy(self.manifest)
        first = next(iter(drift["assets"]))
        drift["assets"][first]["local_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "drift"):
            inventory.validate(drift)

    def test_omitted_units_fail_offline_verification(self):
        for prefix in ("registry:", "workflow:", "collector:", "file:", "ppu:"):
            manifest = copy.deepcopy(self.manifest)
            victim = next(u for u in manifest["units"] if u["id"].startswith(prefix))
            manifest["units"].remove(victim)
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                inventory.validate(manifest)

    def test_pinned_baseline_verification_is_required_in_ci(self):
        steps = workflow("ci.yaml")["jobs"]["registration-validation"]["steps"]
        command = next(
            s["run"]
            for s in steps
            if s.get("name") == "Verify inventory against the pinned upstream tree"
        )
        self.assertIn(inventory.BASELINE, command)
        self.assertIn("--verify-baseline", command)
        self.assertLess(len(inventory.render(self.manifest).encode()), 1500 * 1024)

    def test_native_dispatch_rejects_ppu_only_and_keeps_multi_backend(self):
        tree = ast.parse(
            (ROOT / "scripts/ci/utils/slash_command_handler.py").read_text()
        )
        names = {
            "detect_suite",
            "_dispatch_err",
            "_extract_runner_configs",
            "_extract_legacy_suites",
        }
        functions = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        scope = {
            "re": re,
            "_resolve_runner_config": lambda rc, path, suite: {
                "error": None,
                "suite": suite,
                "is_cpu": False,
            },
        }
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), "handler", "exec"),
            scope,
        )
        ppu_cpu = 'register_cpu_ci(1, suite="stage-a-test-cpu-ppu")\n'
        ppu_hw = 'register_ppu_ci(1, suite="stage-a-test-1-gpu-ppu")\n'
        cuda = 'register_cuda_ci(1, stage="base-a", runner_config="1-gpu-small")\n'
        cpu = 'register_cpu_ci(1, suite="base-a-test-cpu")\n'
        for content, rejected in (
            (ppu_cpu, True),
            (ppu_hw, True),
            ("pass\n", True),
            (cuda, False),
            (cpu, False),
            (ppu_hw + cuda, False),
            (ppu_cpu + cpu, False),
        ):
            with self.subTest(content=content), patch(
                "builtins.open", mock_open(read_data=content)
            ):
                result = scope["detect_suite"]("registered/ppu/test_case.py")
                self.assertEqual(bool(result[0]["error"]), rejected)
                if rejected:
                    self.assertIn("/run-ppu-e2e", result[0]["error"])

    def test_registration_parser_rejects_unresolvable_arguments(self):
        with self.assertRaises(ValueError):
            inventory.registrations("test.py", "register_cuda_ci(1, suite=get_suite())")

    def test_native_selector_and_registration_contracts_preserved(self):
        paths = [
            ".github/workflows/_pr-test-check-changes.yml",
            ".github/workflows/_pr-test-stage.yml",
            ".github/workflows/_pr-test-stage-cpu.yml",
            "scripts/ci/runner_configs.yml",
            "test/run_suite.py",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(
                    inventory.digest((ROOT / path).read_bytes()),
                    (
                        self.manifest["assets"][path]["upstream_sha256"]
                        if path != "test/run_suite.py"
                        else self.manifest["assets"][path]["local_sha256"]
                    ),
                )
        for name in ("pr-test.yml", "pr-test-extra.yml"):
            wf = workflow(name)
            self.assertNotIn("ppu", json.dumps(wf).lower())
            self.assertIn(
                "github.event.repository.default_branch",
                wf["jobs"]["notify-pr-states"]["steps"][0]["run"],
            )

    def test_ppu_registrations_do_not_change_native_partitions(self):
        import compute_partitions as partitions

        tests = partitions.collect_tests(
            partitions.discover_files(str(ROOT)), sanity_check=False
        )
        native = [
            t
            for t in tests
            if t.backend.name != "PPU" and not t.effective_suite.endswith("-ppu")
        ]
        for name in ("pr-test.yml", "pr-test-extra.yml"):
            timeouts = partitions.load_run_timeouts(ROOT / ".github/workflows" / name)
            for full in (False, True):
                with self.subTest(workflow=name, full_parallel=full):
                    result = partitions.compute_partitions(
                        tests, str(ROOT), timeouts, full_parallel=full
                    )
                    self.assertEqual(
                        result,
                        partitions.compute_partitions(
                            native, str(ROOT), timeouts, full_parallel=full
                        ),
                    )
                    self.assertTrue(result)
                    self.assertFalse(any(s.endswith("-ppu") for s in result))
                    for info in result.values():
                        self.assertEqual(info["arr"], list(range(info["size"])))
                        self.assertLessEqual(info["max_parallel"], info["size"])

    def test_display_names_do_not_change_ppu_machine_identity(self):
        ci = workflow("ci.yaml")
        self.assertEqual(ci["name"], "CI Pipeline")
        self.assertEqual(ci["jobs"]["ci"]["name"], "PPU Full CI")
        self.assertEqual(ci["jobs"]["cpu-precheck"]["name"], "PPU CPU Shortlist")
        self.assertIn("PPU E2E Gate", ci["jobs"]["e2e-gate"]["name"])
        self.assertEqual(
            workflow("pr-test-ppu-e2e-entry.yml")["jobs"]["execute"]["name"],
            "PPU E2E (${{ inputs.entry }})",
        )
        self.assertNotIn("pr-test.yml", json.dumps(ci))

    def test_shared_gate_defaults_and_nonpreempting_commands(self):
        gate = workflow("pr-gate.yml")[True]["workflow_call"]["inputs"]
        self.assertEqual(gate["cool-down-minutes"]["default"], 120)
        self.assertEqual(gate["rate-limit-workflow-id"]["default"], "pr-test.yml")
        commands = workflow("ppu-e2e-commands.yml")
        self.assertEqual(commands[True]["workflow_run"]["workflows"], ["CI Pipeline"])
        generic = workflow("slash-command-handler.yml")["jobs"]["slash_command"]
        self.assertFalse(generic["concurrency"]["cancel-in-progress"])
        self.assertIn("ppu-e2e-commands-", generic["concurrency"]["group"])

    def test_native_path_filters_with_actual_picomatch(self):
        check = workflow("_pr-test-check-changes.yml")["jobs"]["run"]
        step = next(
            s
            for s in check["steps"]
            if s.get("uses", "").startswith("dorny/paths-filter@")
        )
        filters = yaml.safe_load(step["with"]["filters"])
        script = "const p=require('picomatch'); const x=JSON.parse(process.argv[1]); console.log(JSON.stringify(Object.entries(x.filters).filter(([k,v])=>x.paths.some(f=>v.some(g=>p(g,{dot:true})(f)))).map(([k])=>k).sort()));"
        cases = [
            (["python/sglang/srt/models/qwen3.py"], {"main_package"}),
            (["python/sglang/kernels/aot/csrc/test.cu"], {"sgl_kernel"}),
            (["python/sglang/kernels/jit/test.py"], {"main_package", "jit_kernel"}),
            (["python/sglang/multimodal_gen/runtime/test.py"], {"multimodal_gen"}),
            (["test/registered/ppu/test_new.py"], {"main_package"}),
            (["docs/test.md"], set()),
            (
                [
                    "test/registered/npu/test_old.py",
                    "test/registered/npu/test_renamed.py",
                ],
                {"main_package"},
            ),
        ]
        for paths, expected in cases:
            result = subprocess.run(
                [
                    "node",
                    "-e",
                    script,
                    json.dumps({"filters": filters, "paths": paths}),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            with self.subTest(paths=paths):
                self.assertEqual(set(json.loads(result.stdout)), expected)


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.pr = {
            "number": 5,
            "state": "open",
            "draft": False,
            "labels": [{"name": "run-ci"}],
            "user": {"login": "author"},
            "head": {
                "sha": "a" * 40,
                "ref": "feature",
                "repo": {"full_name": "fork/repo"},
            },
            "base": {"sha": "b" * 40, "repo": {"full_name": "owner/repo"}},
        }
        self.run = {
            "id": 20,
            "run_attempt": 2,
            "path": recovery.PIPELINE,
            "event": "pull_request",
            "head_sha": self.pr["head"]["sha"],
            "head_branch": "feature",
            "head_repository": {"full_name": "fork/repo"},
            "pull_requests": [{"number": 5}],
            "status": "completed",
            "conclusion": "failure",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.source = {
            "issue_url": "https://api.github.com/repos/owner/repo/issues/5",
            "id": 9,
            "body": "/rerun-failed-ci",
            "user": {"login": "author", "type": "User"},
        }
        self.api = Mock(repository="owner/repo", base="https://api.github.com")
        self.api.comments.return_value = []
        self.api.pages.return_value = [self.run]
        self.api.get.side_effect = lambda path: copy.deepcopy(
            self.pr
            if path == "pulls/5"
            else self.source if path == "issues/comments/9" else self.run
        )
        self.plan = {
            "pr": 5,
            "head_sha": self.pr["head"]["sha"],
            "base_sha": self.pr["base"]["sha"],
        }
        self.env = patch.dict(
            os.environ, {"GITHUB_RUN_ID": "100", "GITHUB_RUN_ATTEMPT": "1"}
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.request = patch.object(recovery, "resolve_request", return_value=None)
        self.request.start()
        self.addCleanup(self.request.stop)
        self.artifact = patch.object(recovery, "attempt_plan", return_value=self.plan)
        self.artifact.start()
        self.addCleanup(self.artifact.stop)

    def recover(self):
        return recovery.recover(self.api, 5, 9, self.pr["head"]["sha"], "author")

    def test_full_rerun_only_and_intent_precedes_submission(self):
        ok, message = self.recover()
        self.assertTrue(ok)
        calls = self.api.post.call_args_list
        self.assertEqual(calls[0].args[0], "issues/5/comments")
        self.assertEqual(calls[1].args, ("actions/runs/20/rerun", {}))
        self.assertIn("selection unchanged", message)

    def test_no_active_cancellation_or_fork_approval_bypass(self):
        for field, value in (
            ("status", "in_progress"),
            ("conclusion", "action_required"),
            ("conclusion", "success"),
        ):
            with self.subTest(value=value):
                original = self.run[field]
                self.run[field] = value
                self.assertFalse(self.recover()[0])
                self.api.post.assert_not_called()
                self.run[field] = original

    def test_no_run_and_cross_pr_source_are_explicit(self):
        self.api.pages.return_value = []
        self.assertIn("No eligible", self.recover()[1])
        self.source["issue_url"] = "https://api.github.com/repos/owner/repo/issues/6"
        with self.assertRaisesRegex(ValueError, "different PR"):
            self.recover()
        self.api.post.assert_not_called()

    def test_combined_command_keeps_selection_unchanged(self):
        self.source["body"] = "/tag-and-rerun-ci extra"
        self.assertTrue(self.recover()[0])
        self.assertEqual(
            [c.args[0] for c in self.api.post.call_args_list],
            ["issues/5/comments", "actions/runs/20/rerun"],
        )

    def test_stale_base_or_unrelated_workflow_never_rerun(self):
        self.plan["base_sha"] = "stale"
        self.assertFalse(self.recover()[0])
        self.plan["base_sha"] = self.pr["base"]["sha"]
        self.run["path"] = ".github/workflows/pr-test.yml"
        self.assertFalse(self.recover()[0])
        self.api.post.assert_not_called()

    def test_absent_label_draft_revoked_author_fail_closed(self):
        for change in (
            {"labels": []},
            {"draft": True},
            {"user": {"login": "someone-else"}},
        ):
            original = copy.deepcopy(self.pr)
            self.pr.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.recover()
            self.pr = original
        self.api.post.assert_not_called()

    def test_selected_e2e_revalidates_manual_label(self):
        with patch.object(
            recovery, "resolve_request", return_value={"selectors": ["test-qwen"]}
        ), patch.object(
            recovery, "label_authorization", return_value=["labeler revoked"]
        ):
            with self.assertRaisesRegex(ValueError, "revoked"):
                self.recover()
        self.api.post.assert_not_called()

    def test_submission_deduplication_is_bound_to_trusted_handler(self):
        self.recover()
        receipt = self.api.post.call_args_list[0].args[1]["body"]
        self.api.post.reset_mock()
        self.api.comments.return_value = [
            {"body": receipt, "user": {"login": "github-actions[bot]", "type": "Bot"}}
        ]
        self.api.handler_run.return_value = {
            "path": recovery.HANDLER,
            "event": "issue_comment",
            "head_branch": "release",
        }
        self.api.metadata.return_value = {"default_branch": "release"}
        self.assertFalse(self.recover()[0])
        self.api.post.assert_not_called()

    def test_attempt_race_fails_closed(self):
        original = self.api.get.side_effect
        self.api.get.side_effect = lambda path: (
            {**self.run, "run_attempt": 3}
            if path.startswith("actions/runs/")
            else original(path)
        )
        with self.assertRaisesRegex(ValueError, "run changed"):
            self.recover()
        self.api.post.assert_not_called()

    def test_generic_native_partial_rerun_is_separate(self):
        tree = ast.parse(
            (ROOT / "scripts/ci/utils/slash_command_handler.py").read_text()
        )
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "handle_rerun_failed_ci"
        )
        scope = {"has_sgl_kernel_changes": lambda pr: False}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), "handler", "exec"),
            scope,
        )
        native = Mock(
            path=".github/workflows/pr-test.yml",
            status="completed",
            conclusion="failure",
        )
        ppu = Mock(path=recovery.PIPELINE, status="completed", conclusion="failure")
        repo = Mock()
        repo.get_workflow_runs.return_value = [native, ppu]
        with patch.object(recovery, "api_from_environment"), patch.object(
            recovery, "recover", return_value=(True, "full rerun")
        ):
            self.assertTrue(
                scope[function.name](
                    repo, Mock(), Mock(), {"can_rerun_failed_ci": True}
                )
            )
        native.rerun_failed_jobs.assert_called_once()
        ppu.rerun_failed_jobs.assert_not_called()
        ppu.rerun.assert_not_called()


class StatusTest(unittest.TestCase):
    def test_trusted_single_writer_workflow(self):
        wf = workflow("pr-states.yml")
        self.assertIn("PR Test Base", wf[True]["workflow_run"]["workflows"])
        self.assertIn("CI Pipeline", wf[True]["workflow_run"]["workflows"])
        writer = wf["jobs"]["update-pr-body"]
        self.assertEqual(
            writer["concurrency"]["group"],
            "pr-states-${{ needs.resolve-pr.outputs.number }}",
        )
        self.assertFalse(writer["concurrency"]["cancel-in-progress"])
        self.assertEqual(
            writer["steps"][0]["with"]["ref"],
            "${{ github.event.repository.default_branch }}",
        )

    def test_status_association_attempts_disabled_and_ignored_runs(self):
        script = r"""
const assert = require('node:assert/strict');
const status = require(process.argv[1]);
const pr = {number:5, head:{sha:'head',ref:'feature',repo:{full_name:'fork/repo'}},base:{sha:'base'}};
const run = {id:1,run_attempt:2,path:'.github/workflows/ci.yaml',event:'pull_request',head_sha:'head',head_branch:'feature',head_repository:{full_name:'fork/repo'},pull_requests:[{number:5,head:{sha:'head'},base:{sha:'base'}}],status:'completed',conclusion:'success',html_url:'https://github.com/owner/repo/actions/runs/1'};
assert(status.associated(run,pr,'ci.yaml'));
for(const change of [{head_sha:'old'},{event:'push'},{pull_requests:[]},{head_repository:{full_name:'other/repo'}},{pull_requests:[{number:5,head:{sha:'head'},base:{sha:'old'}}]}]) assert(!status.associated({...run,...change},pr,'ci.yaml'));
const api={rest:{actions:{getWorkflow:async()=>({data:{id:10,state:'active'}}),listWorkflowRuns:'runs',listJobsForWorkflowRunAttempt:'jobs'}},paginate:async(method,opts)=>method==='runs'?[run]:[{name:'PPU E2E Gate',conclusion:'success'}]};
(async()=>{
  const context={repo:{owner:'owner',repo:'repo'}};
  assert.match(await status.workflowStatus(api,context,pr,'ci.yaml',true),/attempt 2/);
  api.paginate=async(method)=>method==='runs'?[run]:[{name:'PPU E2E Ignored Event',conclusion:'success'}];
  assert.match(await status.workflowStatus(api,context,pr,'ci.yaml',true),/No verified run/);
  api.paginate=async(method)=>method==='runs'?[{...run,pull_requests:[],check_suite_id:7}]:[{name:'PPU E2E Gate',conclusion:'success'}];
  api.rest.checks={getSuite:async()=>({data:{head_sha:'merge'}})};
  api.rest.repos={getCommit:async()=>({data:{parents:[{sha:'base'},{sha:'head'}]}})};
  assert.match(await status.workflowStatus(api,context,pr,'ci.yaml',true),/attempt 2/);
  api.rest.repos.getCommit=async()=>({data:{parents:[{sha:'old-base'},{sha:'head'}]}});
  assert.match(await status.workflowStatus(api,context,pr,'ci.yaml',true),/No verified run/);
  api.rest.actions.getWorkflow=async()=>({data:{state:'disabled_manually'}});
  assert.match(await status.workflowStatus(api,context,pr,'ci.yaml',true),/activation required/);
  await assert.rejects(status.resolvePr(api,{eventName:'workflow_dispatch',payload:{inputs:{pr_number:'5bad'}}}));
  let reads=0, body;
  api.rest.pulls={get:async()=>({data:{...pr,state:'open',body:++reads===1?'old':'user edit\n<!-- pr-states:start -->old status<!-- pr-states:end -->\nafter'}}),update:async p=>{body=p.body;}};
  await status.update(api,context,{info:()=>{}},5);
  assert(body.startsWith('user edit'));
  assert(body.endsWith('\nafter'));
  assert.equal((body.match(/pr-states:start/g)||[]).length,1);
  assert(body.includes('Source: PPU-configured'));
  assert(body.includes('Upstream native CI'));
  body=undefined; reads=0;
  api.rest.pulls.get=async()=>({data:{...pr,state:'open',head:{...pr.head,sha:++reads===1?'head':'new'}}});
  await status.update(api,context,{info:()=>{}},5);
  assert.equal(body,undefined);
})().catch(error=>{console.error(error);process.exit(1);});
"""
        subprocess.run(
            ["node", "-e", script, str(ROOT / "scripts/ci/utils/pr_status.js")],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
