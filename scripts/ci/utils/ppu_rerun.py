"""Full-attempt PPU recovery for the trusted generic slash-command handler.

This adapter never edits selection receipts. It shares the command scheduler's
per-PR concurrency lock and uses only the exact PPU workflow identity.
"""

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ppu"))
from pr_e2e_commands import (  # noqa: E402
    PIPELINE,
    GitHub,
    associated_run,
    attempt_plan,
    body_digest,
    require,
    resolve_request,
)
from select_pr_e2e import label_authorization, snapshot  # noqa: E402

MARKER = "<!-- ppu-generic-rerun:v1 -->\n"
HANDLER = ".github/workflows/slash-command-handler.yml"
COMMAND = re.compile(r"^/(?:rerun-failed-ci|tag-and-rerun-ci)(?:\s+extra)?$")


def api_from_environment():
    api = GitHub.__new__(GitHub)
    api.repository = os.environ["REPO_FULL_NAME"]
    api.base = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    api.token = os.environ["GITHUB_TOKEN"]
    api._runs = {}
    api._metadata = None
    return api


def recover(api, number, source_id, expected_head, allowed_login):
    """Recover at most one current-revision run; return an explanatory result.

    Generic command authorization is resolved by the trusted caller. Selection
    and hardware authorization are independently resolved again here and in CI.
    """
    source = api.get(f"issues/comments/{source_id}")
    require(
        source.get("issue_url") == f"{api.base}/repos/{api.repository}/issues/{number}",
        "Recovery command belongs to a different PR",
    )
    require(
        source["user"].get("type") == "User"
        and source["user"]["login"] == allowed_login,
        "PPU recovery requires the authorized human command author",
    )
    require(
        COMMAND.fullmatch(source["body"].strip()),
        "Use a standalone /rerun-failed-ci or /tag-and-rerun-ci command for PPU recovery",
    )
    pr = api.get(f"pulls/{number}")
    require(
        pr["state"] == "open"
        and not pr["draft"]
        and pr["head"]["sha"] == expected_head,
        "PPU recovery requires an open, non-draft current-revision PR",
    )

    def authorize(live):
        permissions = json.loads(
            (
                Path(__file__).resolve().parents[3] / ".github/CI_PERMISSIONS.json"
            ).read_text()
        )
        require(
            live["user"]["login"] == allowed_login
            or permissions.get(allowed_login, {}).get("can_rerun_failed_ci") is True,
            "Generic rerun permission is no longer valid",
        )
        require(
            "run-ci" in {label["name"] for label in live["labels"]},
            "PPU recovery requires existing run-ci authorization",
        )
        request = resolve_request(api, number)
        if request and request["selectors"]:
            errors = label_authorization(api, snapshot(live))
            require(not errors, "; ".join(errors))
        return request

    request = authorize(pr)
    query = (
        f"actions/workflows/ci.yaml/runs?event=pull_request&head_sha={expected_head}"
    )
    runs = sorted(
        (r for r in api.pages(query, "workflow_runs") if associated_run(api, r, pr)),
        key=lambda r: r["id"],
        reverse=True,
    )
    if any(r["status"] != "completed" for r in runs):
        return (
            False,
            "PPU CI is active/queued; no cancellation or duplicate recovery was requested.",
        )
    if runs and runs[0]["conclusion"] == "action_required":
        return False, "PPU CI requires GitHub Actions approval before recovery."
    candidate = None
    for run in runs:
        plan = attempt_plan(api, run)
        if not plan or (plan.get("pr"), plan.get("head_sha"), plan.get("base_sha")) != (
            number,
            expected_head,
            pr["base"]["sha"],
        ):
            continue
        # A newer successful executable attempt must not resurrect an old failure.
        if run["conclusion"] not in {"failure", "skipped"}:
            return (
                False,
                "Latest eligible PPU attempt does not need failed-CI recovery.",
            )
        candidate = run
        break
    if candidate is None:
        return (
            False,
            "No eligible current-revision PPU run exists; push a new commit to start PR CI.",
        )
    created = datetime.fromisoformat(candidate["created_at"].replace("Z", "+00:00"))
    require(
        datetime.now(timezone.utc) - created < timedelta(days=30),
        "PPU run is too old to rerun",
    )
    for comment in api.comments(number):
        if (
            comment.get("user", {}).get("login") != "github-actions[bot]"
            or comment["user"].get("type") != "Bot"
            or not comment.get("body", "").startswith(MARKER)
        ):
            continue
        receipt = json.loads(comment["body"][len(MARKER) :])
        if (
            receipt.get("repository"),
            receipt.get("pr"),
            receipt.get("source_id"),
            receipt.get("head_sha"),
            receipt.get("base_sha"),
        ) != (api.repository, number, source_id, expected_head, pr["base"]["sha"]):
            continue
        handler = api.handler_run(receipt["handler_run_id"], receipt["handler_attempt"])
        if (
            handler.get("path") == HANDLER
            and handler.get("event") == "issue_comment"
            and handler.get("head_branch") == api.metadata()["default_branch"]
        ):
            return (
                False,
                "This command already submitted PPU recovery; use a new comment to retry.",
            )
    live = api.get(f"pulls/{number}")
    require(snapshot(live) == snapshot(pr), "PR changed during PPU recovery")
    require(authorize(live) == request, "PPU selection changed during recovery")
    current_source = api.get(f"issues/comments/{source_id}")
    require(
        current_source["body"] == source["body"]
        and current_source["user"] == source["user"],
        "Recovery command changed",
    )
    current = api.get(f"actions/runs/{candidate['id']}")
    require(
        associated_run(api, current, live)
        and current["status"] == "completed"
        and current["conclusion"] == candidate["conclusion"]
        and current["run_attempt"] == candidate["run_attempt"],
        "PPU run changed or is being rerun",
    )
    latest = [
        r for r in api.pages(query, "workflow_runs") if associated_run(api, r, live)
    ]
    require(
        not any(
            r["status"] != "completed" or r["id"] not in {old["id"] for old in runs}
            for r in latest
        ),
        "New PPU work appeared during recovery",
    )
    receipt = {
        "repository": api.repository,
        "pr": number,
        "source_id": source_id,
        "source_digest": body_digest(source["body"]),
        "head_sha": expected_head,
        "base_sha": live["base"]["sha"],
        "target_run_id": current["id"],
        "target_attempt": current["run_attempt"] + 1,
        "handler_run_id": int(os.environ["GITHUB_RUN_ID"]),
        "handler_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
    }
    # Persist before submitting: an ambiguous API failure cannot form a retry loop.
    api.post(
        f"issues/{number}/comments",
        {"body": MARKER + json.dumps(receipt, sort_keys=True)},
    )
    api.post(f"actions/runs/{current['id']}/rerun", {})
    return (
        True,
        f"PPU recovery: requested all jobs for run {current['id']}, attempt {receipt['target_attempt']}; selection unchanged.",
    )
