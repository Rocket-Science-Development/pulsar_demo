# Trial: consistency of LLM answers

Does a model give the same answer when the same question is asked again? This trial asks 30 questions with known
short answers (`questions.py`: public-services arithmetic, Canadian civics and geography, French and English) several
times to local models, and checks that the consistency metrics of Pulsar.ml separate the conditions and raise no
false alarm on an A/A split. The French report with the environment and the results is
[`rapport_coherence.md`](rapport_coherence.md).

| File | Role |
|---|---|
| `questions.py` | The 30 questions: 10 numbers, 10 yes / no, 10 labels (cities, provinces...), each with its known answer |
| `campaign.py` | Asks the questions through the OpenAI-compatible API of Ollama with `ConsistencyProbe` (pulsar-data-collection), one JSON file per condition |
| `analyze.py` | Per-question agreement, window metrics and drift tests (`pulsar_metrics.genai.consistency`, `consistency_drift`), A/A splits, evidence records |
| `results/coherence-1/` | Answers per condition, per-question frames (CSV), `analysis.json`, `evidence.json`, run log |

| Condition | Model | Temperature | Answers per question |
|---|---|---|---|
| `llama_t07` | llama3.2:3b | 0.7 | 6: answers 0-2 are the reference, 3-5 the A/A holdout |
| `granite_t07` | granite3.3:2b | 0.7 | 3 |
| `llama_t15` | llama3.2:3b | 1.5 | 3 |
| `granite_t15` | granite3.3:2b | 1.5 | 3 |

The questions are synthetic and public: no personal data, and the answers are kept.

## Run

```bash
pip install "pulsar-metrics @ git+https://github.com/Rocket-Science-Development/pulsar_metrics.git" \
            "pulsar-data-collection[genai] @ git+https://github.com/Rocket-Science-Development/pulsar_data_collection.git" scipy
ollama pull llama3.2:3b && ollama pull granite3.3:2b && ollama pull granite-embedding:278m
python campaign.py --campaign coherence-1 --workers 2      450 calls with the next line: under 10 min of LLM time on a 16-core CPU
python analyze.py --campaign coherence-1 --embeddings granite-embedding:278m
```

Use `127.0.0.1` rather than `localhost` in `--base-url` on Windows: `localhost` is tried in IPv6 first and each call
waited about 20 s before the fallback.
