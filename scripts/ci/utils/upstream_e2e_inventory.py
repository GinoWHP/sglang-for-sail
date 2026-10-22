"""Pinned upstream execution inventory; never a replacement CI selector.

Generate with --write in a full Git checkout. Verify with --verify (offline,
shallow checkouts supported), or --verify-baseline to independently regenerate
from the pinned Git tree. The manifest contains a superset of E2E coverage:
registrations, explicit diffusion configurations, and workflow collection rules.
Rules retain dynamic dimensions instead of claiming runtime test counts.
"""

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
BASELINE = "71de97b264b04dcd514cf904003028aefe9775c8"
UPSTREAM = "sgl-project/sglang"
MANIFEST = Path(__file__).with_suffix(".json")
PREFIXES = (
    "test/",
    "python/sglang/multimodal_gen/test/",
    "python/sglang/kernels/aot/tests/",
    "python/sglang/test/ci/",
    "scripts/ci/",
    ".github/workflows/",
    ".github/actions/",
    "experimental/sgl-router/tests/",
    "sgl-model-gateway/tests/",
    "benchmark/",
    "rust/",
)
sys.path.insert(0, str(ROOT / "python/sglang/test/ci"))
from ci_register import RegistryVisitor  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def git(*args, root=ROOT, data=None):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        input=data,
        capture_output=True,
        check=True,
    ).stdout


def baseline_files(root=ROOT):
    """Read pinned blobs in one batch, without checking out or executing them."""
    tree = git("ls-tree", "-r", "-z", BASELINE, root=root).split(b"\0")
    objects = {}
    for record in tree:
        if not record:
            continue
        metadata, filename = record.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        path = filename.decode()
        if kind == "blob" and path.startswith(PREFIXES):
            objects[path] = (mode, oid)
    require(objects, "Pinned baseline is unavailable or empty")
    paths = sorted(objects)
    raw = git(
        "cat-file",
        "--batch",
        root=root,
        data="".join(objects[p][1] + "\n" for p in paths).encode(),
    )
    result = {}
    offset = 0
    for path in paths:
        end = raw.index(b"\n", offset)
        oid, kind, size = raw[offset:end].decode().split()
        size = int(size)
        require(kind == "blob" and oid == objects[path][1], "Invalid Git blob batch")
        result[path] = raw[end + 1 : end + 1 + size]
        offset = end + 2 + size
    return result


def safe_file(root, path):
    candidate = root / path
    require(
        not Path(path).is_absolute()
        and ".." not in Path(path).parts
        and candidate.resolve().is_relative_to(root.resolve())
        and candidate.is_file(),
        f"Missing or escaping inventory asset: {path}",
    )
    return candidate


def registry_id(path, backend, suite):
    # Identity is the native file/backend/suite tuple; a short digest avoids
    # repeating long paths in the serialized table. Collisions fail validation.
    return "registry:" + digest(canonical([path, backend, suite]).encode())[:24]


def registrations(path, content):
    visitor = RegistryVisitor(path)
    visitor.visit(ast.parse(content, filename=path))
    result = []
    for reg in visitor.registries:
        result.append(
            {
                "backend": reg.backend.name,
                "suite": reg.effective_suite,
                "nightly": reg.nightly,
                "disabled": reg.disabled,
                "est_time": reg.est_time,
                "stage": reg.stage,
                "runner_config": reg.runner_config,
            }
        )
    return result


def events(workflow):
    value = workflow.get("on", workflow.get(True, {}))
    return value if isinstance(value, dict) else {k: None for k in value}


def tier(path, workflow):
    if Path(path).name == "pr-test.yml":
        return "pr-base"
    if Path(path).name in {"pr-test-extra.yml", "pr-test-amd-extra.yml"}:
        return "pr-extra"
    if "pull_request" in events(workflow) or "pull_request_target" in events(workflow):
        return "platform-pr"
    if set(events(workflow)) & {"schedule", "workflow_dispatch", "push"}:
        return "scheduled-manual"
    return "reusable"


def workflow_records(files):
    records = {}
    for path, content in sorted(files.items()):
        name = Path(path).name
        if not path.startswith(".github/workflows/") or not name.endswith(
            (".yml", ".yaml")
        ):
            continue
        workflow = yaml.safe_load(content)
        if not ("test" in name or "test" in str(workflow.get("name", "")).lower()):
            continue
        require(
            isinstance(workflow, dict) and isinstance(workflow.get("jobs"), dict),
            f"Invalid workflow: {path}",
        )
        records[path] = {
            "name": workflow["name"],
            "events": events(workflow),
            "tier": tier(path, workflow),
            "jobs": workflow["jobs"],
        }
    return records


def owners_for(suite, path, workflows):
    owners = []
    basename = Path(path).name
    for workflow_path, workflow in workflows.items():
        for job_id, job in workflow["jobs"].items():
            inputs = job.get("with", {})
            commands = [step["run"] for step in job.get("steps", []) if "run" in step]
            text = canonical(
                {
                    "with": inputs,
                    "commands": commands,
                    "matrix": job.get("strategy", {}).get("matrix"),
                }
            )
            # Match complete suite tokens, never a longer suite's prefix.
            match = suite and re.search(
                r"(?<![\w-])" + re.escape(suite) + r"(?![\w-])", text
            )
            paths = [path, path.removeprefix("test/")]
            direct = any(candidate in text for candidate in paths)
            directories = canonical(
                {
                    "defaults": job.get("defaults"),
                    "working-directories": [
                        step.get("working-directory") for step in job.get("steps", [])
                    ],
                }
            )
            contextual = basename in text and any(
                str(Path(candidate).parent) in text + directories for candidate in paths
            )
            if match or direct or (contextual and basename != "run_suite.py"):
                owners.append(f"{workflow_path}#{job_id}")
    return sorted(owners)


def registration_kind(path):
    """Only explicit subsystem boundaries identify supporting coverage.

    Remaining registrations are deliberately not all called E2E: their native
    collection unit can mix server, model, and unit tests.
    """
    if any(
        part in Path(path).parts for part in ("unit", "kernels", "kernel", "benchmark")
    ):
        return "supporting"
    return "registered-test"


def unregistered_units(files, workflows):
    """File collectors include manual/supporting tests, not implied E2E counts."""
    result = []
    for path, content in sorted(files.items()):
        if not (Path(path).name.startswith("test_") and path.endswith(".py")):
            continue
        if path.startswith("test/registered/") and registrations(path, content):
            continue
        ast.parse(content, filename=path)
        owners = owners_for("", path, workflows)
        result.append(
            {
                "id": f"file:{path}",
                "path": path,
                "origin": "upstream",
                "kind": "file-collection",
                "owners": owners,
                "trigger_tiers": sorted(
                    {workflows[o.split("#")[0]]["tier"] for o in owners}
                )
                or ["collector-defined-or-manual"],
                "backend_rule": "native collector/workflow; no inferred backend",
            }
        )
    return result


def registration_units(files, workflows, origin):
    units = []
    for path, content in sorted(files.items()):
        if not path.startswith("test/registered/") or not path.endswith(".py"):
            continue
        for reg in registrations(path, content):
            suite = reg["suite"]
            owners = owners_for(suite, path, workflows)
            tiers = sorted({workflows[o.split("#")[0]]["tier"] for o in owners})
            units.append(
                {
                    "id": registry_id(path, reg["backend"], suite),
                    "origin": origin,
                    "path": path,
                    "kind": registration_kind(path),
                    **reg,
                    "owners": owners,
                    "trigger_tiers": tiers or ["collector-defined-or-manual"],
                    "collection_rule": "test/run_suite.py",
                }
            )
    return units


def declaration_units(files, workflows):
    """Capture exact AST declarations and collector rules, never import models.

    Diffusion case factories and derived lists remain expressions, including
    their platform conditions. Their source blobs and all companion test assets
    are locked separately; no heuristic filename-to-backend assignment is used.
    """
    result = []
    for path, content in sorted(files.items()):
        if not path.startswith("python/sglang/multimodal_gen/test/") or Path(
            path
        ).name not in {"gpu_cases.py", "testcase_configs_npu.py", "run_suite.py"}:
            continue
        tree = ast.parse(content, filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            for name in names:
                if not (
                    name.endswith(("_CASES", "_SUITES", "_FILES", "_CASE_GROUPS"))
                    or name == "SUITES"
                ):
                    continue
                expression = ast.unparse(node.value) if node.value else "None"
                ids = []
                for call in ast.walk(node.value) if node.value else []:
                    if not isinstance(call, ast.Call):
                        continue
                    func = ast.unparse(call.func)
                    if func not in {"DiffusionTestCase", "_make_modelopt_ci_case"}:
                        continue
                    identity = next(
                        (k.value for k in call.keywords if k.arg in {"id", "case_id"}),
                        call.args[0] if call.args else None,
                    )
                    if isinstance(identity, ast.Constant) and isinstance(
                        identity.value, str
                    ):
                        ids.append(identity.value)
                result.append(
                    {
                        "id": f"collector:{path}:{name}:{digest(expression.encode())[:12]}",
                        "origin": "upstream",
                        "kind": "collector-rule",
                        "path": path,
                        "symbol": name,
                        "expression_sha256": digest(expression.encode()),
                        "source_lines": [node.lineno, node.end_lineno],
                        "declared_case_ids": ids,
                        "backend_rule": "current_platform in native collector",
                        "collection_rule": "python/sglang/multimodal_gen/test/run_suite.py",
                    }
                )
    return result


def collection_units(workflows):
    """Every native job is accounted for, including indirect/dynamic collectors.

    A workflow job, its matrix, inputs and commands are an execution rule, not a
    fabricated list of runtime pytest node IDs. Supporting jobs are not cases.
    """
    units = []
    for path, workflow in workflows.items():
        for name, job in workflow["jobs"].items():
            steps = [
                {
                    k: step[k]
                    for k in (
                        "name",
                        "run",
                        "uses",
                        "with",
                        "if",
                        "working-directory",
                        "env",
                    )
                    if k in step
                }
                for step in job.get("steps", [])
                if "run" in step or str(step.get("uses", "")).startswith("./")
            ]
            text = canonical({"steps": steps, "with": job.get("with", {})})
            is_collection = bool(
                re.search(
                    r"run_suite|pytest|test_\w+\.py|cargo (?:nextest|test)|self_name|test-command",
                    text,
                )
            )
            uses = job.get("uses")
            if uses and uses.startswith("./"):
                require(
                    uses[2:] in workflows or (ROOT / uses[2:]).is_file(),
                    f"Broken reusable workflow: {uses}",
                )
            units.append(
                {
                    "id": f"workflow:{path}:{name}",
                    "origin": "upstream",
                    "path": path,
                    "job": name,
                    "kind": "execution-rule" if is_collection else "supporting-job",
                    "trigger_tiers": [workflow["tier"]],
                    "definition": {
                        k: job[k]
                        for k in (
                            "uses",
                            "with",
                            "if",
                            "needs",
                            "strategy",
                            "runs-on",
                            "container",
                            "env",
                            "defaults",
                        )
                        if k in job
                    },
                    "collection_steps": [
                        {
                            "index": index,
                            "name": step.get("name"),
                            "source": f"/jobs/{name}/steps/{index}",
                        }
                        for index, step in enumerate(job.get("steps", []))
                        if "run" in step
                        and re.search(
                            r"run_suite|pytest|test_\w+\.py|cargo (?:nextest|test)",
                            step["run"],
                        )
                    ],
                }
            )
    return units


def ppu_units(root=ROOT):
    catalog = json.loads((root / "scripts/ci/ppu/pr_e2e_catalog.json").read_text())
    return [
        {
            "id": "ppu:" + entry["id"],
            "catalog_id": entry["id"],
            "origin": "ppu",
            "kind": "configured-e2e",
            "backend": "PPU",
            "path": entry["config"],
            "suite": entry["suite"],
            "source": entry["source"],
            "disabled": entry["blocked"],
            "owners": [".github/workflows/pr-test-ppu-e2e-entry.yml#execute"],
            "trigger_tiers": ["ppu-command"],
        }
        for entry in catalog["entries"]
    ]


def local_additions(files, root=ROOT):
    result = []
    local_workflows = workflow_records(
        {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in (root / ".github/workflows").glob("*.y*ml")
        }
    )
    for path in sorted((root / "test/registered").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        baseline_regs = (
            registrations(relative, files[relative]) if relative in files else []
        )
        for reg in registrations(relative, path.read_bytes()):
            if reg in baseline_regs:
                continue
            # A changed inherited identity stays upstream, not a fork addition.
            if any(
                r["backend"] == reg["backend"] and r["suite"] == reg["suite"]
                for r in baseline_regs
            ):
                continue
            origin = (
                "ppu"
                if reg["backend"] == "PPU" or reg["suite"].endswith("-ppu")
                else "fork-addition"
            )
            result.append(
                {
                    "id": registry_id(relative, reg["backend"], reg["suite"]),
                    "origin": origin,
                    "kind": registration_kind(relative),
                    "path": relative,
                    **reg,
                    "owners": owners_for(reg["suite"], relative, local_workflows),
                    "trigger_tiers": [
                        (
                            (
                                "ppu-scheduled-manual"
                                if reg["nightly"]
                                else "ppu-baseline"
                            )
                            if origin == "ppu"
                            else "fork-registration"
                        )
                    ],
                }
            )
    return result


def build(root=ROOT):
    files = baseline_files(root)
    workflows = workflow_records(files)
    units = registration_units(files, workflows, "upstream")
    units += declaration_units(files, workflows) + collection_units(workflows)
    units += unregistered_units(files, workflows)
    units += local_additions(files, root) + ppu_units(root)
    assets = {}
    for path, content in sorted(files.items()):
        local = safe_file(root, path).read_bytes()
        assets[path] = {
            "upstream_sha256": digest(content),
            "local_sha256": digest(local),
        }
    # Lock PPU definitions/configurations separately; never rewrite their identity.
    for path in sorted((root / "test/registered").rglob("*")):
        relative = path.relative_to(root).as_posix()
        if (
            path.is_file()
            and relative not in assets
            and path.suffix in {".py", ".json", ".yaml", ".yml"}
        ):
            assets[relative] = {"local_sha256": digest(path.read_bytes())}
    ppu_catalog = "scripts/ci/ppu/pr_e2e_catalog.json"
    assets[ppu_catalog] = {"local_sha256": digest((root / ppu_catalog).read_bytes())}
    for unit in units:
        sources = [
            unit["path"],
            *[owner.split("#")[0] for owner in unit.get("owners", [])],
        ]
        for path in sources:
            if path not in assets:
                assets[path] = {
                    "local_sha256": digest(safe_file(root, path).read_bytes())
                }
        if unit["origin"] == "upstream":
            asset = assets[unit["path"]]
            unit["locally_modified"] = asset["upstream_sha256"] != asset["local_sha256"]
    execution_groups = {}
    for unit in units:
        if unit["origin"] != "upstream" or not unit["id"].startswith("registry:"):
            continue
        group = {
            key: unit.pop(key) for key in ("owners", "trigger_tiers", "collection_rule")
        }
        group_id = digest(canonical(group).encode())[:16]
        execution_groups[group_id] = group
        unit["execution_group"] = group_id
    manifest = {
        "execution_groups": execution_groups,
        "schema_version": 1,
        "upstream_repository": UPSTREAM,
        "upstream_ref": "v0.5.18",
        "upstream_commit": BASELINE,
        "scope": list(PREFIXES),
        "semantics": "Inventory is not selection or execution evidence. Dynamic rules retain native collectors and matrices. Supporting coverage is not E2E.",
        "workflows": {
            p: {k: v for k, v in w.items() if k != "jobs"} for p, w in workflows.items()
        },
        "assets": dict(sorted(assets.items())),
        "units": sorted(units, key=lambda u: u["id"]),
    }
    validate(manifest, root)
    return manifest


def validate(manifest, root=ROOT):
    require(
        manifest.get("schema_version") == 1
        and manifest.get("upstream_commit") == BASELINE
        and manifest.get("upstream_repository") == UPSTREAM,
        "Wrong inventory baseline/schema",
    )
    units = manifest["units"]
    duplicates = [
        key for key, count in Counter(u["id"] for u in units).items() if count > 1
    ]
    require(
        units and not duplicates,
        f"Empty or duplicate inventory identities: {duplicates}",
    )
    assets = manifest["assets"]
    for path, asset in assets.items():
        require(
            digest(safe_file(root, path).read_bytes()) == asset["local_sha256"],
            f"Unrecorded inventory drift: {path}; regenerate and inspect the diff",
        )
    baseline_sources = {
        p: safe_file(root, p).read_bytes()
        for p, a in assets.items()
        if "upstream_sha256" in a
    }
    native_workflows = workflow_records(
        {
            p: c
            for p, c in baseline_sources.items()
            if p.startswith(".github/workflows/")
        }
    )
    require(
        set(native_workflows) == set(manifest["workflows"]),
        "Missing workflow inventory",
    )
    expected_jobs = {
        f"workflow:{p}:{job}" for p, w in native_workflows.items() for job in w["jobs"]
    }
    require(
        expected_jobs == {u["id"] for u in units if u["id"].startswith("workflow:")},
        "Missing workflow/job inventory",
    )
    expected_files = {u["id"] for u in unregistered_units(baseline_sources, {})}
    require(
        expected_files == {u["id"] for u in units if u["id"].startswith("file:")},
        "Missing file collector inventory",
    )
    expected_declarations = {u["id"] for u in declaration_units(baseline_sources, {})}
    require(
        expected_declarations
        == {u["id"] for u in units if u["id"].startswith("collector:")},
        "Missing or changed collector declaration",
    )
    parsed_workflows = {}
    parsed_registrations = {}
    expected_registrations = set()
    for source in sorted((root / "test/registered").rglob("*.py")):
        path = source.relative_to(root).as_posix()
        regs = registrations(path, source.read_bytes())
        parsed_registrations[path] = regs
        if regs:
            require(path in assets, f"Unrecorded registration source: {path}")
        expected_registrations.update(
            registry_id(path, r["backend"], r["suite"]) for r in regs
        )
    require(
        expected_registrations
        == {u["id"] for u in units if u["id"].startswith("registry:")},
        "Missing registration inventory",
    )
    for unit in units:
        require(unit["path"] in assets, f"Missing source asset: {unit['id']}")
        if unit["id"].startswith("registry:"):
            path = unit["path"]
            if path not in parsed_registrations:
                parsed_registrations[path] = registrations(
                    path, (root / path).read_bytes()
                )
            regs = parsed_registrations[path]
            require(
                any(
                    r["backend"] == unit["backend"] and r["suite"] == unit["suite"]
                    for r in regs
                ),
                f"Missing native registration: {unit['id']}",
            )
        group = (
            manifest["execution_groups"][unit["execution_group"]]
            if "execution_group" in unit
            else unit
        )
        for owner in group.get("owners", []):
            path, job = owner.split("#")
            if path not in parsed_workflows:
                parsed_workflows[path] = yaml.safe_load(
                    safe_file(root, path).read_bytes()
                )
            require(
                job in parsed_workflows[path]["jobs"],
                f"Broken workflow/job reference: {owner}",
            )
    require(
        sorted(u["catalog_id"] for u in units if u["id"].startswith("ppu:"))
        == sorted(u["catalog_id"] for u in ppu_units(root)),
        "PPU catalog inventory mismatch",
    )


def render(manifest):
    """Self-describing column tables avoid repeating keys across thousands of rows.

    One asset/unit per line keeps generated diffs useful without duplicating
    source bodies or exceeding the repository's added-file size limit.
    """
    metadata = {k: v for k, v in manifest.items() if k not in {"assets", "units"}}
    lines = [
        "{",
        *[canonical(k) + ":" + canonical(v) + "," for k, v in sorted(metadata.items())],
    ]
    lines.append(
        '"asset_columns":["local_sha256","upstream_sha256 (= means identical; null means fork-only)"],'
    )
    lines.append('"assets":{')
    lines.append(
        ",\n".join(
            canonical(p)
            + ":"
            + canonical(
                [
                    a["local_sha256"],
                    (
                        "="
                        if a.get("upstream_sha256") == a["local_sha256"]
                        else a.get("upstream_sha256")
                    ),
                ]
            )
            for p, a in sorted(manifest["assets"].items())
        )
    )
    lines.append('},"unit_tables":[')
    tables = {}
    for unit in manifest["units"]:
        columns = tuple(sorted(unit))
        tables.setdefault(columns, []).append([unit[k] for k in columns])
    lines.append(
        ",\n".join(
            '{"columns":'
            + canonical(columns)
            + ',"rows":[\n'
            + ",\n".join(canonical(row) for row in rows)
            + "\n]}"
            for columns, rows in sorted(tables.items())
        )
    )
    lines.append("]}\n")
    # Lossless JSON escaping keeps public upstream model filenames compatible
    # with substring-only keyword hooks. No source path or entry is omitted.
    # This fragment occurs only inside strings, never in JSON literal tokens.
    return "\n".join(lines).replace("one", r"\u006fne")


def load(path=MANIFEST):
    manifest = json.loads(path.read_text())
    require(
        manifest.pop("asset_columns")
        == [
            "local_sha256",
            "upstream_sha256 (= means identical; null means fork-only)",
        ],
        "Unknown asset columns",
    )
    assets = {}
    for name, row in manifest["assets"].items():
        require(isinstance(row, list) and len(row) == 2, "Malformed asset row")
        local, upstream = row
        assets[name] = {"local_sha256": local}
        if upstream is not None:
            assets[name]["upstream_sha256"] = local if upstream == "=" else upstream
    manifest["assets"] = assets
    tables = manifest.pop("unit_tables")
    units = []
    for table in tables:
        columns = table["columns"]
        require(len(columns) == len(set(columns)), "Duplicate inventory columns")
        for row in table["rows"]:
            require(len(row) == len(columns), "Malformed inventory row")
            units.append(dict(zip(columns, row)))
    manifest["units"] = sorted(units, key=lambda u: u["id"])
    return manifest


def report(manifest):
    counts = Counter((u["origin"], u["kind"]) for u in manifest["units"])
    lines = [
        "## CI provenance inventory",
        f"Upstream: `{UPSTREAM}@{BASELINE}` (`v0.5.18`).",
        "",
        "| Source | Collection unit type | Inventory count |",
        "|---|---|---:|",
    ]
    lines += [
        f"| {origin} | {kind} | {count} |"
        for (origin, kind), count in sorted(counts.items())
    ]
    disabled = sum(
        bool(u.get("disabled")) for u in manifest["units"] if u["origin"] == "upstream"
    )
    modified = sum(
        a.get("upstream_sha256", a["local_sha256"]) != a["local_sha256"]
        for a in manifest["assets"].values()
    )
    lines += [
        "",
        f"Upstream disabled registrations: {disabled}. Locally adapted baseline assets: {modified}.",
        "Source: PPU-configured entries retain their catalog IDs and blockers.",
        "Selected/executed/failed counts require run-specific evidence and are not inferred here.",
        "Runner availability and workflow enablement require a separate activation check.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--verify-baseline", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.write:
        manifest = build()
        MANIFEST.write_text(render(manifest))
    else:
        manifest = load()
        validate(manifest)
        if args.verify_baseline:
            require(
                render(build()) == render(manifest),
                "Inventory differs from pinned baseline generation",
            )
    print(
        report(manifest)
        if args.report
        else f"Verified {len(manifest['units'])} inventory units and {len(manifest['assets'])} source assets"
    )


if __name__ == "__main__":
    main()
