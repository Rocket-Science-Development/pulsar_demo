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
| `check_registry.py` | Fails on a YAML value split at commas and on JSON copies that differ from the YAML sources |
| `tests/<id>.py` | One module per test (`run(ctx) -> metrics`, `ctx.limit(...)` for each limit); `_*.py` are shared helpers |
| `approved_tonight.json` | The 29 tests approved for the night of 2026-11-02 |
| `night_lab.workflow.js` | Agent orchestration: preflight, provision, run by family, adversarial verification of limits, report, teardown |
| `rerun_scale.md`, `rerun_scale.json` | Scalability rerun after the 0.10 / 0.11 fixes: plan, cost, and the workflow arguments |

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

About 50 USD for 10 hours. Every instance created for the lab is tagged `Project=pulsar-lab` and stops itself at the
approved time. A pre-existing instance reused by the lab keeps its own tags and only gets `LabRun=pulsar-lab`, removed
at teardown. Datasets, caches and models go on a separate data volume per host (`LabVolume=data`), deleted at teardown.

## Running the tests

The tests target the pulsar_metrics API after the calibration fix (branch `correctifs/calibration-analyzer`):

- `AACalibrator` refuses a reference shorter than 10 windows unless `allow_short=True`, and contiguous draws need at
  least 4 windows;
- it raises `n_draws` automatically (`auto_draws=True`);
- `nan_policy` defaults to `"omit"`;
- performance metrics are evaluated on the labelled rows only.

Where a test shows a limit of the old behaviour, it asks for that behaviour explicitly, so the night's evidence stays
reproducible:

- TC-05, SC-06 and SC-11 pass `auto_draws=False` to show the p-value floor;
- TS-02 passes `nan_policy="propagate"` on its `nan` version, and its `nan_omit` version checks the fix;
- TS-01, TC-05, TS-02, SC-06, SC-11 and LA-03 pass `allow_short=True` on their short references.

The performance evaluation on labelled rows has no switch back to the old behaviour. With it, TS-02's limit "a window
with a missing label is refused" no longer fires.

One test, locally (from `pulsar_demo`, with the validation venv):

```bash
python -m lab.harness --self-check          # the health sampler records on this host
python -m lab.check_registry                # registry fields intact
LAB_DRY=1 python -m lab.harness LP-02       # short dry run (reduced sizes); LAB_ENV=local by default
python -m lab.harness LP-02                 # full run
```

SC-04 (degraded network) needs its own InfluxDB 2.6.1 and a Toxiproxy 2.x container on one Docker network, the
InfluxDB env file in `LAB_PLATFORM_INFLUX_ENV`, and `LAB_TOXIPROXY_URL` / `LAB_TOXI_PROXY_URL` / `LAB_TOXI_UPSTREAM`
(see the docstring of `tests/sc_04.py`). Run it once per write profile (`LAB_PROFILE=default` and `tuned`, with
`LAB_RUN_TAG`). Report: `rapports/rapport_sc04_reseau_degrade.md`.

Each run writes `results/<ID>.json`, which overwrites the previous result of that test. Copy the night's files away
before a local dry run. Datasets are read from `C:/dev/pulsar-ml/data/lab/` (`/data/lab` on AWS), outside the
repositories. Most ML tests need their dataset, and the LLM tests need Ollama or a GPU.

Offline replays recompute a test's statistics from its recorded runs, without the agent or a GPU.
`python -m lab.replay.declencheur_derive` replays LR-03 (`results/LR-03/runs.jsonl`) to measure the false-trigger rate
and the detection delay of the "measured drift" trigger (signals, persistence k, alpha); it needs pulsar_metrics with
`mmd_test(groups=)` and, once, the local Ollama for the passage embeddings. Its report is
`results/rapport_fiabilite_declencheur.md`.

The whole night lab: approve the tests and the run settings on the registry page, then start the `pulsar-night-lab`
workflow (`night_lab.workflow.js`) with `{approved, settings, stamp}`. It runs these steps: preflight, provisioning, one
agent per family (implementation, dry run, full run), adversarial verification of every limit, morning report and
teardown. Each family agent writes `results/run_<family>.json` and `results/verify_<family>.json`. The report agent
merges them into `results/night_full_results.json` and reads that file; nothing is inlined or truncated in its prompt.

## Lessons from the night of 2026-11-02

The report is written to `args.report_file` (default `lab/results/rapport_nuit.md`).

- **Cost.** About 40 USD for 8.8 hours with 6 instances (about 4.50 USD/h plus 0.85 USD of disks), against an
  approved budget of 100 USD. This is an estimate from on-demand prices; Cost Explorer lags by up to 24 h. The EBS
  volumes of the stopped instances kept costing about 2.30 USD per day. Separate data volumes deleted at teardown, and
  an AWS Budgets alarm, avoid that.
- **GPU capacity.** g6.xlarge had no capacity in ca-central-1, so both GPU hosts fell back to g4dn.xlarge. lab-gpu-2
  (16 GB of RAM, no swap) ran out of memory twice, with two families on it. Use one family per GPU host, or
  `OLLAMA_MAX_LOADED_MODELS=1` with a RAM limit.
- **INSECTS has no direct URL** (and its licence is unverified). TC-05 ran on a labelled synthetic substitute; the
  real run needs a manual download after the licence is checked.
- **Python.** The GPU AMI ships Python 3.10; install Python 3.12 with `uv` (`uv python install 3.12`, then
  `uv venv -p 3.12 /data/lab/venv`).
- **Ollama.** Piping the Ollama install script to a shell was not allowed. Install from the release tarball instead,
  and keep Ollama bound to localhost.
- **Health sampler.** It recorded nothing on Linux during the night: `getattr(proc, "num_fds", proc.num_handles)`
  evaluated `num_handles` eagerly. This is fixed; check it with `--self-check` in preflight.
- **Registry.** Unquoted commas inside YAML flow mappings split the `dataset` fields of six tests. They are now quoted,
  and `check_registry.py` catches the case.
- **Tags.** The provisioning agent re-tagged a pre-existing instance of another project (the tags were restored
  afterwards). The workflow now forbids it.
- **Refuted limits.** The 24 refuted limits (report section 3) were test defects. The tests were fixed, for example
  the A/A of IM-02 on a single split, the window sizes of TS-01 aligned on the calibrator's null, and the matched
  benign pools of LI-02. They were not rerun on AWS.

