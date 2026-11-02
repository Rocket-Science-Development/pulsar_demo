# Pulsar.ml lab

A night-long battery of tests that exercises Pulsar.ml end to end, from drift on time series to prompt-injection
detection on an LLM agent, plus scalability, hardware/software health, data loss under high volume and chaos. Every
limit found is recorded with its evidence and an improvement proposal.

| File | Role |
|---|---|
| `registry_ml.yaml` | Classical ML: time series, tabular classification and regression, text and image embeddings, fairness (21 tests) |
| `registry_llm.yaml` | LLM applications and agents: injection, personal information, RAG faithfulness, model equality, bias probes, agents (14 tests) |
| `registry_platform.yaml` | Scalability, many models per instance, outages, degraded network, soak, crash, schema changes (14 tests) |
| `registry.json` | The three merged, with priorities (P1 = tonight, P2 = if time allows) |
| `harness.py` | Runs one test, samples host health, checks criteria, writes `results/<id>.json` with the limits found |
| `night_lab.workflow.js` | Agent orchestration: preflight, provision, run by family, adversarial verification of limits, report, teardown |

Datasets and models were verified on their source pages (licence, gated flag) on 2026-11-01/02. Downloads go to
`data/lab/` outside the repositories. Resources that are gated, non-commercial or of Chinese origin are excluded.

## Approval

The registry is published as a private page where each test is approved, rejected or sent back for changes, and where
the run settings (environment, AWS budget cap, scope, automatic stop time, instructions) are recorded. The orchestration
starts only from the approved tests and the saved settings.

## Proposed infrastructure (AWS ca-central-1, on demand)

| Machine | Instance | Role | USD/h |
|---|---|---|---|
| lab-cpu | c7i.4xlarge | ML families | 0.78 |
| lab-scale | c7i.8xlarge | load generator, hundreds of Pulse, analyses at scale, network chaos | 1.56 |
| lab-db | r7i.2xlarge | InfluxDB 2.6.1 + Grafana 11.3.0 under load | 0.58 |
| lab-soak | c7i.2xlarge | 8-hour soak | 0.39 |
| lab-gpu x2 | g6.xlarge if capacity, else g4dn.xlarge | LLM, detectors on GPU, agents, canary | 2 x 0.58 to 0.89 |

About 50 USD for 10 hours. Every instance is tagged `Project=pulsar-lab` and stops itself at the approved time.
