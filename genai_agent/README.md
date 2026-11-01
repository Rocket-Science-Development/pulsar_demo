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

## Notes

- Only features are stored (steps, tools, tokens, latency, termination...), never the requests or the answers.
- On a laptop CPU (8 cores), one run of `llama3.2:3b` takes 20 to 60 s. A GPU (or a hosted endpoint in Canada,
  through `LLM_BASE_URL`) is needed for the hundreds of runs of a calibrated drift study.
- `LLM_BASE_URL` and `--model` point the agent to any OpenAI-compatible endpoint (vLLM, Ollama, TGI).
