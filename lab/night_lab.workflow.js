export const meta = {
  name: 'pulsar-night-lab',
  description: 'Run the approved Pulsar.ml lab tests overnight: provision, implement and run by family, verify limits, report, tear down',
  whenToUse: 'After the lab registry has been approved (decisions + run settings on the registry page)',
  phases: [
    { title: 'Preflight', detail: 'check approvals, settings, budget and tools' },
    { title: 'Provision', detail: 'AWS machines or local stack, datasets outside repositories' },
    { title: 'Run', detail: 'one agent per family implements and runs its approved tests on the harness' },
    { title: 'Verify', detail: 'adversarial review of every limit found' },
    { title: 'Report', detail: 'morning report, improvement proposals, registry update' },
    { title: 'Teardown', detail: 'stop every lab instance' },
  ],
}

// args: { approved: [registry test objects], settings: {env, budget_usd, scope, stop, instructions}, stamp: "ISO time", report_file?: path }
const RULES = `
Ground rules (non-negotiable):
- Work in C:/dev/pulsar-ml/pulsar_demo/lab (branch fonction/laboratoire-tests). Tests go in lab/tests/<id>.py and run through
  lab/harness.py (run(ctx) -> metrics; ctx.limit(finding, evidence, proposal, severity) for every limit). Results land in lab/results/.
- Datasets are downloaded to C:/dev/pulsar-ml/data/lab/ (or /data/lab on AWS), never inside a repository. Folktables and
  Diabetes 130 contain real (anonymised) people: no re-identification, no joins with other sources.
- No model of Chinese origin. No gated or non-commercial dataset. Never handle secrets: tokens come from existing env files.
- AWS: region ca-central-1 only, auto-stop at the approved time, never exceed the approved budget.
- Tags: every instance or volume CREATED for the lab gets Project=pulsar-lab. A PRE-EXISTING instance reused by the lab
  is never re-tagged: its Project, Name and other tags stay as they are; add only LabRun=pulsar-lab (removed at
  teardown) so the lab can find it. Teardown targets Project=pulsar-lab OR LabRun=pulsar-lab.
- Data: datasets, caches and models go on a separate EBS data volume per host, created for the lab (tag
  Project=pulsar-lab, LabVolume=data, DeleteOnTermination=true) and mounted on /data/lab; never grow or reuse the root
  volume or a disk of a pre-existing instance. The data volume is detached and DELETED at teardown.
- A limit is a fact about Pulsar.ml found by a test, with numbers. A test bug is NOT a limit: fix the test.
- Do not push, merge or open pull requests. Do not modify pulsar_metrics or pulsar_data_collection; propose changes instead.
`

const FAMILIES = [
  { key: 'ml-streams', prefixes: ['TS-', 'TC-', 'TR-'], host: 'lab-cpu' },
  { key: 'ml-embeddings', prefixes: ['TX-', 'IM-'], host: 'lab-gpu-1' },
  { key: 'fairness', prefixes: ['FA-'], host: 'lab-cpu' },
  { key: 'llm-security', prefixes: ['LI-', 'LP-'], host: 'lab-gpu-1' },
  { key: 'llm-rag-models', prefixes: ['LR-', 'LM-'], host: 'lab-gpu-2' },
  { key: 'agents-bias', prefixes: ['LA-', 'LB-', 'LS-'], host: 'lab-gpu-2' },
  { key: 'platform', prefixes: ['SC-'], host: 'lab-scale + lab-db + lab-soak' },
]

const RESULT = {
  type: 'object',
  properties: {
    tests: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'string' }, status: { type: 'string', enum: ['passed', 'failed', 'limit-found', 'error', 'skipped'] },
      summary: { type: 'string' }, result_file: { type: 'string' },
      limits: { type: 'array', items: { type: 'object', properties: {
        finding: { type: 'string' }, evidence: { type: 'string' }, proposal: { type: 'string' },
        severity: { type: 'string', enum: ['critical', 'major', 'minor'] } }, required: ['finding', 'evidence', 'proposal', 'severity'] } },
    }, required: ['id', 'status', 'summary', 'limits'] } },
    blockers: { type: 'array', items: { type: 'string' } },
  },
  required: ['tests', 'blockers'],
}
const VERDICTS = {
  type: 'object',
  properties: { verdicts: { type: 'array', items: { type: 'object', properties: {
    id: { type: 'string' }, finding: { type: 'string' }, real: { type: 'boolean' }, reason: { type: 'string' },
    proposal_ok: { type: 'boolean' } }, required: ['id', 'finding', 'real', 'reason', 'proposal_ok'] } } },
  required: ['verdicts'],
}

const approved = (args && args.approved) || []
const settings = (args && args.settings) || {}
// Optional settings of a partial run (e.g. the scalability rerun of lab/rerun_scale.md):
// - run_name: suffix of every lab-level result file (run_<family>.<run_name>.json...), so the night's files stay intact;
// - hosts: the only machines to provision; profiles: write profiles to run each test with (LAB_PROFILE, LAB_RUN_TAG);
// - teardown: "terminate" terminates the instances created for the lab (never a pre-existing one); default "stop".
const SUFFIX = settings.run_name ? `.${settings.run_name}` : ''
const PROFILES = settings.profiles || []
if (!approved.length || !settings.env) {
  log('Nothing approved or no run settings: the lab does not start.')
  return { started: false }
}

phase('Preflight')
const preflight = await agent(`${RULES}
Preflight for the Pulsar.ml night lab. Approved tests: ${approved.map(t => t.id).join(', ')}.
Settings: ${JSON.stringify(settings)}.
Check: python env C:/dev/pulsar-ml/validation_e2e/.venv works with pulsar_metrics and pulsar_data_collection (editable);
lab/harness.py imports and "python -m lab.harness --self-check" passes on every host; "python -m lab.check_registry"
passes (no dataset field split by YAML); Docker running; for AWS: credentials work in ca-central-1, quotas (G vCPU >= 8, standard >= 64),
approved budget covers the plan in lab/README.md. Return a short go/no-go with any blocker.`, { label: 'preflight' })
log(`Preflight: ${String(preflight).slice(0, 300)}`)

phase('Provision')
const provision = await agent(`${RULES}
Provision the lab for environment "${settings.env}" (stop time ${settings.stop}, budget ${settings.budget_usd} USD).
Machines to provision: ${settings.hosts ? settings.hosts.join(', ') + ' only; create no other machine' : 'the whole plan'}.
AWS plan (lab/README.md): lab-cpu c7i.4xlarge, lab-scale c7i.8xlarge, lab-db r7i.2xlarge (InfluxDB 2.6.1 + Grafana 11.3.0),
lab-soak c7i.2xlarge, lab-gpu-1/2 g6.xlarge if capacity else g4dn.xlarge (g6 had no capacity on 2026-11-02: try it,
then fall back to g4dn). A pre-existing stopped instance may be restarted only if the operator instructions allow it, and
then follows the pre-existing-instance tag rule (LabRun tag only, original tags untouched).
Every instance: shutdown -h at the stop time, SSH only from the operator IP, Ollama bound to localhost.
Every host: create and attach one data volume (gp3, sized for its family, tags Project=pulsar-lab and LabVolume=data),
mount it on /data/lab, and put datasets, caches, venvs and models there. Return the volume ids.
The GPU AMI ships Python 3.10: install Python 3.12 with uv. Use the Ollama release tarball, not the install script, if
piping a remote script to a shell is not allowed.
Pull the approved models; download the approved datasets outside repositories. Return hostnames/IPs, what failed, and the
hourly cost actually provisioned.`, { label: 'provision' })

phase('Run')
const results = await pipeline(
  FAMILIES.map(f => ({ ...f, tests: approved.filter(t => f.prefixes.some(p => t.id.startsWith(p))) })).filter(f => f.tests.length),
  fam => agent(`${RULES}
You run the "${fam.key}" family of the Pulsar.ml night lab on ${fam.host}. Provisioned environment:
${String(provision).slice(0, 1500)}
Operator instructions: ${settings.instructions || 'none'}
For each approved test below: implement lab/tests/<id>.py on the harness (or use the existing module), do a short dry run,
then the full run; record every limit with numbers and an improvement proposal; never stop the whole family because one
test fails.${PROFILES.length ? `
Run every test once per write profile, in this order: ${PROFILES.join(', ')}; for profile <p> set LAB_PROFILE=<p> and
LAB_RUN_TAG=<p>, which writes lab/results/<id>.<p>.json. Report each profile run as its own entry, id "<id>.<p>".` : ''}
The full specification of each test (method, dataset, licence, pass criteria, expected limit) is in lab/registry.json.
Before returning, write your complete result (the same JSON you return) to lab/results/run_${fam.key}${SUFFIX}.json.
Approved tests of this family: ${fam.tests.map(t => t.id).join(', ')}`, { label: `run:${fam.key}`, phase: 'Run', schema: RESULT }),
  (res, fam) => {
    const limits = (res && res.tests || []).flatMap(t => t.limits.map(l => ({ id: t.id, ...l })))
    if (!limits.length) return { family: fam.key, res, verdicts: [] }
    return agent(`${RULES}
Adversarially review the limits found by the "${fam.key}" family. For each, try to REFUTE it: is it a real property of
Pulsar.ml, reproducible from the evidence and result files in lab/results/, or a defect of the test itself? Default to
real=false when the evidence does not support it. Also judge whether the proposal would fix it. Before returning, write
your complete verdicts (the same JSON you return) to lab/results/verify_${fam.key}${SUFFIX}.json.
Limits: ${JSON.stringify(limits, null, 1)}`, { label: `verify:${fam.key}`, phase: 'Verify', schema: VERDICTS })
      .then(v => ({ family: fam.key, res, verdicts: (v && v.verdicts) || [] }))
  },
)

phase('Report')
// The family results are never inlined in the prompt (the former slice(0, 60000) truncated them). The report agent
// first writes them, complete, to a results file, then reads that file.
const RESULTS_FILE = `C:/dev/pulsar-ml/pulsar_demo/lab/results/night_full_results${SUFFIX}.json`
// The report goes where the caller says (args.report_file); by default next to the results.
const REPORT_FILE = args.report_file || `C:/dev/pulsar-ml/pulsar_demo/lab/results/rapport_nuit${SUFFIX}.md`
const report = await agent(`${RULES}
Write the morning report of the Pulsar.ml night lab, in French, to ${REPORT_FILE}.
Source of truth: ${RESULTS_FILE}. Step 1: write it yourself, as a JSON object with one key "run:<family>" and one key
"verify:<family>" per family, merged from lab/results/run_<family>${SUFFIX}.json and lab/results/verify_<family>${SUFFIX}.json (written by
the family agents; a family without limits has no verify file); never truncate it.
Families: ${results.filter(Boolean).map(r => r.family).join(', ')}. Step 2: read that file in full (by parts if large)
and lab/results/<ID>.json for the numbers. Content: per family and per test the status and key numbers; every CONFIRMED
limit (verdict real=true) with its evidence and improvement proposal, ranked by severity; refuted limits listed apart as
test defects; tests not run and why; cost against the budget of ${settings.budget_usd} USD and health summary. Also write
lab/results/summary${SUFFIX}.json (id -> status).${settings.report_instructions ? `
${settings.report_instructions}` : ''}`, { label: 'report' })

phase('Teardown')
const teardown = await agent(`${RULES}
Tear down the lab in ca-central-1. Approved budget: ${settings.budget_usd} USD; provisioned (from the provision step):
${String(provision).slice(0, 1500)}
1. ${settings.teardown === 'terminate'
    ? 'Terminate every instance tagged Project=pulsar-lab (created for the lab), after its results are copied back; stop (never terminate) an instance tagged LabRun=pulsar-lab.'
    : 'Stop (do not terminate) every instance tagged Project=pulsar-lab or LabRun=pulsar-lab.'}
2. On a pre-existing instance (LabRun tag): remove only the LabRun tag; check its original tags are unchanged.
3. Detach and DELETE every data volume tagged LabVolume=data once its instance is stopped and the results it holds
   have been copied back to lab/results/; never delete a root volume or a volume without that tag.
4. Close SSH tunnels.
Report the final state of each instance and volume, the cost of the night against the budget of ${settings.budget_usd} USD
(say clearly if it was exceeded), and what still costs money (remaining volumes, IPs).`, { label: 'teardown' })

return { started: true, preflight, results: results.filter(Boolean), report, teardown }
