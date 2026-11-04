# Demo: monitoring a tool-calling agent with Pulsar.ml

A small public-services assistant (synthetic programs, no real data) answers requests in French and English with an
open model served locally, and calls five tools. Its traces are collected with
[pulsar-data-collection](https://github.com/Rocket-Science-Development/pulsar_data_collection) (OpenTelemetry GenAI)
into InfluxDB, then analyzed with [pulsar-metrics](https://github.com/Rocket-Science-Development/pulsar_metrics).

```
run_agent.py ──> PermitAgent ──(OpenAI-compatible API)──> Ollama (llama3.2:3b / granite3.3:2b)
                    │ spans gen_ai.* (invoke_agent, chat, execute_tool)
                    ▼
             PulsarSpanExporter ──> PulseSink ──> InfluxDB  (agent_run, tool_call, llm_call)
```

| Tool | Role |
|---|---|
| `search_programs` | find programs for a topic |
| `check_eligibility` | eligibility from income and age (can be degraded: timeouts) |
| `compute_amount` | estimated amount |
| `authenticate` | token, required before `create_file` |
| `create_file` | open an application file |

## Run

```bash
docker compose up -d                                   # InfluxDB 2.6.1 and Ollama
docker compose exec ollama ollama pull llama3.2:3b     # ~2.0 GB (Meta, Llama 3.2 Community License)
docker compose exec ollama ollama pull granite3.3:2b   # ~1.5 GB (IBM, Apache 2.0), for the model-change scenario

export INFLUX_URL=http://localhost:8086 INFLUX_TOKEN=demo-token-change-me INFLUX_BUCKET=monitoring
uv run python run_agent.py --runs 20
```

Scenarios (the `--label` value is written as the `scenario` tag):

| Command | Drift injected |
|---|---|
| `run_agent.py --runs 50` | none (reference) |
| `run_agent.py --runs 50 --model granite3.3:2b --label model_change` | model change |
| `run_agent.py --runs 50 --timeout-rate 0.3 --label tool_fault` | 30 % of `check_eligibility` calls time out |
| `run_agent.py --runs 50 --file-share 0.6 --label question_shift` | many more file-opening requests |
| `run_agent.py --runs 50 --prompt-version v2 --label prompt_change` | system prompt changed |

Each run prints its steps, tools and termination; the records are then available in InfluxDB, e.g.:

```flux
from(bucket: "monitoring")
  |> range(start: -1h, stop: 1m)
  |> filter(fn: (r) => r["_measurement"] == "permit_assistant_1.0_input_data" and r["record_type"] == "agent_run")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
```

## Concurrency

Ollama serves `OLLAMA_NUM_PARALLEL` requests at once (decoding slots). The compose file sets it explicitly to 4,
with `OLLAMA_MAX_LOADED_MODELS=1` and a 4096 context. Left unset, Ollama picks 4 or 1 slots from the free memory. With
1 slot, extra workers only queue: the agent stays near 50 runs/min and the latency grows. Keep the agent's worker pool
(`--workers` of `campaign_s3s4.py`) equal to the number of slots: fewer workers leave slots idle, more workers only
queue at the server.

Measured on a T4 (15 GB), context 4096, temperature 0, workers = slots (lab SC-10, 2026-11-02). Runs/min, with the
p95 run latency in brackets:

| Model | 1 slot | 4 slots | 16 slots | 1 slot, 16 workers | Loaded size, 1 → 16 slots |
|---|---|---|---|---|---|
| `llama3.2:3b` | 42.4 (2.7 s) | **83.2** (6.2 s) | 146.4 (12.7 s) | 51.2 (21.8 s) | 2.4 → 9.2 GB |
| `granite3.3:2b` | 48.8 (2.6 s) | **107.2** (4.8 s) | 207.2 (10.1 s) | 48.8 (32.7 s) | 1.9 → 6.9 GB |
| `llama3.1:8b` | 16.0 (10.7 s) | 44.0 (15.3 s) | **75.2** (29.5 s) | 26.4 (48.9 s) | 5.0 → 12.7 GB |

- Latency-bounded setting (the most slots whose p95 stays within 3 times the p95 of a single worker): 4 slots for
  `llama3.2:3b` and `granite3.3:2b`, 16 for `llama3.1:8b`.
- For throughput (batch campaigns), 16 slots or more: 146 / 207 / 75 runs/min, against 51 / 49 / 26 with 1 slot.
- Every slot reserves its own KV cache (context 4096) in VRAM, on top of the weights: `llama3.1:8b` at 16 slots takes
  12.7 GB of the T4's 15 GB. A longer context or more slots can push the model partly to the CPU, which is much slower;
  check `ollama ps` (100 % GPU) after changing either.
- These values hold for a T4 and a 4096 context; measure again on another GPU.

## Drift campaign

`campaign.py` runs a calibrated drift study: 300 reference runs, a 60-run holdout (control), then three injected
drifts of 90 runs each (30 % of `check_eligibility` calls time out; `granite3.3:2b` instead of `llama3.2:3b`; 60 % of
file-opening requests). `analyze_drift.py` reads the `agent_run` records back and tests windows of 30 runs against
the reference: chi-square and Kolmogorov-Smirnov tests calibrated by A/A resampling of the reference,
Benjamini-Hochberg over the features, confirmation over 2 consecutive windows.

```bash
python campaign.py --campaign my-study
python analyze_drift.py --campaign my-study --out results
```

Results on a T4 GPU (630 runs, about 20 minutes):

| Phase | Windows with an alert | Confirmed | Confirmed features |
|---|---|---|---|
| holdout (no change) | 0 / 2 | no | — |
| tool fault | 2 / 3 | yes, window 3 | `tool_error_rate` (3.7 % → 15.6 %) |
| model change | 3 / 3 | yes, window 2 | `first_tool`, `steps`, `tool_calls`, `tool_error_rate`, `task_success`, `input_tokens`, `latency_s` |
| question shift | 3 / 3 | yes, window 2 | `first_tool`, `input_tokens`, `task_success` |

Drift monitoring cannot see a defect that is already present in the reference. In a first campaign, the model passed
program names instead of identifiers and two tools failed on every call; the injected fault changed nothing and was
rightly not reported, while the agent still answered as if the tools had worked. The tools now accept a name or a
topic word, and absolute checks (tool error rate, consistency of the answer with the tool results) complement drift
monitoring.

## Counterfactual probe

`probe_agent.py` checks whether the first name of the user changes what the agent does. Each set holds the same
request signed with a name typical of each group; placebo sets change only a file reference (noise floor); a
comparative question is asked in both orders. The rows are collected with `pulsar_data_collection.genai.probes` and
analysed with `pulsar_metrics.fairness.counterfactual_analysis` / `comparative_analysis`.

```bash
python probe_agent.py --sets 60 --out probe_v1.json
python probe_agent.py --sets 60 --prompt biased_test --out probe_biased.json   # validation of the instrument
```

`biased_test` is a deliberately biased system prompt (it asks for an authentication when the name "looks North
African or Arabic"), used only to check that the probe detects a known bias. On a T4 GPU, with `llama3.2:3b`:

| | Normal prompt | Biased prompt |
|---|---|---|
| Placebo flip rate (noise floor) | 0.00 [0, 0.06] | 0.00 [0, 0.06] |
| Authentication rate, `maghreb` minus `fr_qc` | 0.00 [−0.06, 0.06], within the zone | **+0.33 [+0.23, +0.45], established** |
| Authentication rate, `en_ca` minus `fr_qc` | +0.02 [0, 0.08], within the zone | 0.00 [−0.06, 0.06], within the zone |
| Position effect (comparative) | always answers "B": instrument biased, no verdict on preferences | same |

The name lists of the demo are **provisional and synthetic**: lists standing for groups must be reviewed by a GBA Plus
expert before any real audit.

## Injection campaign S0 to S4

`campaign_s3s4.py` injects known changes and `analyze_s3s4.py` checks that each detector sees its own and stays quiet
otherwise. The agent now has a retrieval tool (`search_documents`, synthetic rules of the programs in French and
English); every run is scored in-process (injection, faithfulness of the answer to the retrieved passages, personal
information by role) and its trajectory is checked against the versioned rules of `rules/permit_agent.yaml` (LTLf).

```bash
python campaign_s3s4.py --campaign my-campaign --out results/s3s4 --workers 4
python analyze_s3s4.py --results results/s3s4 --out results/s3s4/analysis.md --campaign my-campaign
```

Results on a T4 GPU (reference 150 runs, windows of 60; p < 0.05 counts as a detection):

| Scenario | Injected | Detected by | Result |
|---|---|---|---|
| S0 | nothing, 5 repetitions | — | 0 false alarm out of 25 tests and 5 canary tests |
| S1 | 5 to 50 % off-topic requests | MMD on request embeddings; empty retrieval | MMD from 35 %; empty retrieval from 10 % |
| S2 | another model behind the endpoint | model equality on 12 canary prompts × 5 answers | granite3.3: 5 / 5; llama3.2 q8 instead of q4: 0 / 5 (too subtle for 60 answers) |
| S3 | rules of 1 to 4 programs removed from the corpus | faithfulness (NLI); empty retrieval | faithfulness at 2 and 3 programs; total failure only seen by empty retrieval |
| S4 | 5 to 50 % prompt injections | injection detector | from 5 % (71 injections out of 71 scored above 0.5) |

What it taught:

- 34 of the 71 injections made the agent call `create_file`, 14 of them before authenticating. Yet the compliance test
  against the reference does not separate them: the agent already breaks "authenticate before opening a file" in 5 %
  of normal runs. A critical rule needs an absolute alert per run, not only a test of change.
- The injection detector flags a French imperative ("Ouvrez-moi un dossier...") in 2.8 % of legitimate requests: an
  absolute threshold needs calibrating per language.
- A retrieval that returns nothing leaves nothing to judge for faithfulness: monitor retrieval coverage too.
- Sending the requests concurrently (`--workers 4`) halves the duration on the same GPU; the detectors on CPU are the
  other half of the time.

## Notes

- Only features are stored (steps, tools, tokens, latency, termination...), never the requests or the answers.
- On a laptop CPU (8 cores), one run of `llama3.2:3b` takes 20 to 60 s. A GPU (or a hosted endpoint in Canada,
  through `LLM_BASE_URL`) is needed for the hundreds of runs of a calibrated drift study.
- `LLM_BASE_URL` and `--model` point the agent to any OpenAI-compatible endpoint (vLLM, Ollama, TGI).
