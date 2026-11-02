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

// args: { approved: [registry test objects], settings: {env, budget_usd, scope, stop, instructions}, stamp: "ISO time" }
const RULES = `
Ground rules (non-negotiable):
- Work in C:/dev/pulsar-ml/pulsar_demo/lab (branch fonction/laboratoire-tests). Tests go in lab/tests/<id>.py and run through
  lab/harness.py (run(ctx) -> metrics; ctx.limit(finding, evidence, proposal, severity) for every limit). Results land in lab/results/.
- Datasets are downloaded to C:/dev/pulsar-ml/data/lab/ (or /data/lab on AWS), never inside a repository. Folktables and
  Diabetes 130 contain real (anonymised) people: no re-identification, no joins with other sources.
- No model of Chinese origin. No gated or non-commercial dataset. Never handle secrets: tokens come from existing env files.
- AWS: region ca-central-1 only, tags Project=pulsar-lab, auto-stop at the approved time, never exceed the approved budget.
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
if (!approved.length || !settings.env) {
  log('Nothing approved or no run settings: the lab does not start.')
  return { started: false }
}

phase('Preflight')
const preflight = await agent(`${RULES}
Preflight for the Pulsar.ml night lab. Approved tests: ${approved.map(t => t.id).join(', ')}.
Settings: ${JSON.stringify(settings)}.
Check: python env C:/dev/pulsar-ml/validation_e2e/.venv works with pulsar_metrics and pulsar_data_collection (editable);
lab/harness.py imports; Docker running; for AWS: credentials work in ca-central-1, quotas (G vCPU >= 8, standard >= 64),
approved budget covers the plan in lab/README.md. Return a short go/no-go with any blocker.`, { label: 'preflight' })
log(`Preflight: ${String(preflight).slice(0, 300)}`)

phase('Provision')
const provision = await agent(`${RULES}
Provision the lab for environment "${settings.env}" (stop time ${settings.stop}, budget ${settings.budget_usd} USD).
AWS plan (lab/README.md): lab-cpu c7i.4xlarge, lab-scale c7i.8xlarge, lab-db r7i.2xlarge (InfluxDB 2.6.1 + Grafana 11.3.0),
lab-soak c7i.2xlarge, lab-gpu-1/2 g6.xlarge if capacity else g4dn.xlarge (restart the existing stopped g4dn i-008454b3ee3fa822f first).
Every instance: tag Project=pulsar-lab, shutdown -h at the stop time, SSH only from the operator IP, Ollama bound to localhost.
Pull the approved models; download the approved datasets outside repositories. Return hostnames/IPs, what failed, and the
hourly cost actually provisioned.`, { label: 'provision' })

phase('Run')
const results = await pipeline(
  FAMILIES.map(f => ({ ...f, tests: approved.filter(t => f.prefixes.some(p => t.id.startsWith(p))) })).filter(f => f.tests.length),
  fam => agent(`${RULES}
You run the "${fam.key}" family of the Pulsar.ml night lab on ${fam.host}. Provisioned environment:
${String(provision).slice(0, 1500)}
Operator instructions: ${settings.instructions || 'none'}
For each approved test below: implement lab/tests/<id>.py on the harness, do a short dry run, then the full run; record every
limit with numbers and an improvement proposal; never stop the whole family because one test fails. Tests:
${JSON.stringify(fam.tests, null, 1)}`, { label: `run:${fam.key}`, phase: 'Run', schema: RESULT }),
  (res, fam) => {
    const limits = (res && res.tests || []).flatMap(t => t.limits.map(l => ({ id: t.id, ...l })))
    if (!limits.length) return { family: fam.key, res, verdicts: [] }
    return agent(`${RULES}
Adversarially review the limits found by the "${fam.key}" family. For each, try to REFUTE it: is it a real property of
Pulsar.ml, reproducible from the evidence and result files in lab/results/, or a defect of the test itself? Default to
real=false when the evidence does not support it. Also judge whether the proposal would fix it.
Limits: ${JSON.stringify(limits, null, 1)}`, { label: `verify:${fam.key}`, phase: 'Verify', schema: VERDICTS })
      .then(v => ({ family: fam.key, res, verdicts: (v && v.verdicts) || [] }))
  },
)

phase('Report')
const report = await agent(`${RULES}
Write the morning report of the Pulsar.ml night lab, in French, to G:/My Drive/RocketScience/Pulsar.ml/notes/laboratoire/rapport_nuit.md:
per family and per test the status and key numbers; every CONFIRMED limit (verdict real=true) with its evidence and
improvement proposal, ranked by severity; refuted limits listed apart as test defects; tests not run and why; cost and
health summary. Also write lab/results/summary.json (id -> status). Data:
${JSON.stringify(results.filter(Boolean), null, 1).slice(0, 60000)}`, { label: 'report' })

phase('Teardown')
const teardown = await agent(`${RULES}
Stop every instance tagged Project=pulsar-lab in ca-central-1 (stop, do not terminate), close SSH tunnels, and report the final
state of each instance and an estimate of the night's cost.`, { label: 'teardown' })

return { started: true, preflight, results: results.filter(Boolean), report, teardown }
