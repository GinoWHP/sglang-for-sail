// Trusted default-branch status writer. Never import or execute PR code.
const nativeWorkflows = [
  ['pr-test.yml', 'Base'], ['pr-test-extra.yml', 'Extra (opt-in)'],
  ['pr-test-amd.yml', 'AMD ROCm 7.0'], ['pr-test-amd-rocm720.yml', 'AMD ROCm 7.2'],
  ['pr-test-amd-extra.yml', 'AMD Extra'], ['pr-test-arm64.yml', 'Arm64'],
  ['pr-test-mlx.yml', 'MLX'], ['pr-test-musa.yml', 'MUSA'],
  ['pr-test-npu.yml', 'NPU'], ['pr-test-xeon.yml', 'Xeon'],
  ['pr-test-xpu.yml', 'XPU'], ['pr-test-rust.yml', 'SMG'],
  ['pr-test-sgl-router.yml', 'sgl-router'], ['pr-test-rust-exts.yml', 'Rust extensions'],
  ['pr-test-sgl-kernel.yml', 'SGL Kernel', 'Base'],
  ['pr-test-jit-kernel.yml', 'JIT Kernel', 'Base'],
  ['pr-test-multimodal-gen.yml', 'Multimodal generation', 'Base'],
];

function associated(run, pr, path) {
  return run.path === `.github/workflows/${path}` && run.event === 'pull_request' &&
    run.head_sha === pr.head.sha && run.head_branch === pr.head.ref &&
    run.head_repository?.full_name === pr.head.repo?.full_name &&
    (run.pull_requests || []).some(p => p.number === pr.number &&
      p.base?.sha === pr.base.sha && p.head?.sha === pr.head.sha);
}

async function verifiedAssociation(github, context, run, pr, path) {
  if (associated(run, pr, path)) return true;
  // Fork APIs sometimes omit pull_requests. A current commit-to-PR lookup alone
  // cannot prove the base used by an old run: require its immutable merge parents.
  if ((run.pull_requests || []).length || !run.check_suite_id ||
      run.path !== `.github/workflows/${path}` || run.event !== 'pull_request' ||
      run.head_sha !== pr.head.sha || run.head_branch !== pr.head.ref ||
      run.head_repository?.full_name !== pr.head.repo?.full_name) return false;
  try {
    const {data: suite} = await github.rest.checks.getSuite({...context.repo, check_suite_id: run.check_suite_id});
    const {data: commit} = await github.rest.repos.getCommit({...context.repo, ref: suite.head_sha});
    return commit.parents?.length === 2 && commit.parents[0].sha === pr.base.sha &&
      commit.parents[1].sha === pr.head.sha;
  } catch (error) {
    if (error.status === 404) return false;
    throw error;
  }
}

async function resolvePr(github, context) {
  if (context.payload.pull_request) return context.payload.pull_request.number;
  if (context.eventName === 'workflow_dispatch') {
    const raw = context.payload.inputs.pr_number;
    if (!/^[1-9]\d*$/.test(raw)) throw new Error('pr_number must be a positive integer');
    return Number(raw);
  }
  const run = context.payload.workflow_run;
  if (!run || run.event !== 'pull_request' || !run.head_repository || !run.head_branch) return null;
  const prs = await github.paginate(github.rest.pulls.list, {
    ...context.repo, state: 'open', per_page: 100,
    head: `${run.head_repository.owner.login}:${run.head_branch}`,
  });
  const matches = prs.filter(pr => pr.head.sha === run.head_sha &&
    pr.head.repo?.full_name === run.head_repository.full_name &&
    (!(run.pull_requests || []).length || run.pull_requests.some(p => p.number === pr.number)));
  return matches.length === 1 ? matches[0].number : null;
}

function runLink(run) {
  const status = run.status === 'completed' ? run.conclusion : run.status;
  return `[${status}: run ${run.id}, attempt ${run.run_attempt}](${run.html_url})`;
}

async function workflowStatus(github, context, pr, path, ppu = false) {
  let workflow;
  try {
    ({data: workflow} = await github.rest.actions.getWorkflow({...context.repo, workflow_id: path}));
  } catch (error) {
    if (error.status === 404) return '_Workflow unavailable on the default branch_';
    throw error;
  }
  if (workflow.state !== 'active') return `_Workflow ${workflow.state}; activation required_`;
  const runs = await github.paginate(github.rest.actions.listWorkflowRuns, {
    ...context.repo, workflow_id: workflow.id, head_sha: pr.head.sha,
    event: 'pull_request', per_page: 100,
  });
  const candidates = runs.sort((a, b) => b.id - a.id || b.run_attempt - a.run_attempt);
  for (const run of candidates) {
    if (!await verifiedAssociation(github, context, run, pr, path)) continue;
    if (ppu) {
      const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRunAttempt, {
        ...context.repo, run_id: run.id, attempt_number: run.run_attempt, per_page: 100,
      });
      if (jobs.some(job => job.name === 'PPU E2E Ignored Event')) continue;
      const gate = jobs.find(job => job.name === 'PPU E2E Gate');
      // Empty/irrelevant event runs cannot replace a real PPU result with green.
      if (run.status === 'completed' && !gate) continue;
      if (run.conclusion === 'success' && gate?.conclusion !== 'success') {
        return `${runLink(run)} — PPU E2E gate not verified`;
      }
    }
    return runLink(run);
  }
  return '_No verified run for this PR head/base (not selected, pending, or association unavailable)_';
}

async function update(github, context, core, number) {
  if (!Number.isSafeInteger(number) || number <= 0) throw new Error('Invalid PR number');
  const {data: pr} = await github.rest.pulls.get({...context.repo, pull_number: number});
  if (pr.state !== 'open') return;
  const ppu = await workflowStatus(github, context, pr, 'ci.yaml', true);
  const native = [];
  for (const [path, name, parent] of nativeWorkflows) {
    native.push(`- ${name}: ${parent ? `Reusable jobs within ${parent}; see its run for selected jobs` : await workflowStatus(github, context, pr, path)}`);
  }
  const block = [
    '<!-- pr-states:start -->', '---', '### CI States', '',
    `Revision: \`${pr.head.sha.slice(0, 12)}\`; base: \`${pr.base.sha.slice(0, 12)}\`.`, '',
    `Latest PPU CI: <!-- slot:ppu-ci:start -->${ppu}<!-- slot:ppu-ci:end -->`,
    'Source: PPU-configured. Persistent E2E selection and its gate are independent of native upstream CI.', '',
    '<details><summary>Upstream native CI</summary>', '', ...native, '', '</details>', '',
    'Shared `run-ci` authorization; Extra also requires `run-ci-extra`. Native path filters still apply.',
    'These are workflow results, not counts of executed inventory cases. See each run for selection, evidence, and blocked cases.',
    'Manual refresh: dispatch PR States with this PR number on the default branch.',
    '<!-- pr-states:end -->',
  ].join('\n');
  // Re-fetch after API reads: never overwrite a user's concurrently edited body
  // with the snapshot used for run lookup, or publish a superseded revision.
  const {data: live} = await github.rest.pulls.get({...context.repo, pull_number: number});
  if (live.state !== 'open' || live.head.sha !== pr.head.sha || live.base.sha !== pr.base.sha) {
    core.info('PR revision changed during status lookup; next event will refresh it.');
    return;
  }
  const body = live.body || '';
  const pattern = /\n*(?:---\n+)?<!-- pr-states:start -->[\s\S]*?<!-- pr-states:end -->/;
  const updated = pattern.test(body) ? body.replace(pattern, () => `\n\n${block}`) : `${body}${body ? '\n\n' : ''}${block}`;
  if (updated !== body) await github.rest.pulls.update({...context.repo, pull_number: number, body: updated});
}

module.exports = {nativeWorkflows, associated, resolvePr, workflowStatus, update};
