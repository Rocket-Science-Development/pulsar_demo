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

## Notes

- Only features are stored (steps, tools, tokens, latency, termination...), never the requests or the answers.
- On a laptop CPU (8 cores), one run of `llama3.2:3b` takes 20 to 60 s. A GPU (or a hosted endpoint in Canada,
  through `LLM_BASE_URL`) is needed for the hundreds of runs of a calibrated drift study.
- `LLM_BASE_URL` and `--model` point the agent to any OpenAI-compatible endpoint (vLLM, Ollama, TGI).
