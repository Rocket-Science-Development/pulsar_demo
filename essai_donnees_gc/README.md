# Trial on Government of Canada open data

Pulsar.ml run on real Canadian federal open data, with known changes: a tabular model through the COVID-19 lockdown of
spring 2020, and a retrieval agent on the Government of Canada AI Register with provoked changes. Everything runs on one
workstation (CPU only), with local models and a local InfluxDB. The French report with the environment and the results
is [`rapport_essai_donnees_gc.md`](rapport_essai_donnees_gc.md).

| File | Role |
|---|---|
| `download.py` | Downloads the data and writes `sources.json` (URL, date, size, SHA-256, licence of every file) |
| `no2_data.py` | Hourly NO2 of the Montréal NAPS network joined with the weather of Montréal-Trudeau airport |
| `tabular_replay.py` | Part 1: trains the NO2 model on 2016-2018, replays 2019-2021 week by week through `Pulse` into InfluxDB |
| `tabular_analyze.py` | Part 1: reads the weeks back, calibrated drift on the model error and the inputs, MAE, persistence |
| `register_questions.py` | Part 2: questions with known answers from the structured fields of the register, and their scoring |
| `rag_campaign.py` | Part 2: the agent of `genai_agent/` with a `search_register` tool, reference then three provoked changes |
| `rag_analyze.py` | Part 2: S2 drift analysis of the agent records, MMD on request embeddings, canary model equality, accuracy |
| `influx_env.py` | InfluxDB settings (environment or env file) and creation of the trial bucket with a finite retention |
| `results/` | Versioned results (JSON), the per-run log and the canary answers of the agent campaign |

## Data

| Dataset | Publisher | Licence | Used |
|---|---|---|---|
| [National Air Pollution Surveillance (NAPS)](https://open.canada.ca/data/dataset/1b36a356-defd-4813-acea-47bc3abd859b), hourly NO2 | Environment and Climate Change Canada | Open Government Licence - Canada | 2016-2021, 8 stations of Montréal |
| [Historical climate data](https://climate.weather.gc.ca/), hourly, MONTREAL INTL A (climate ID 7025251) | Environment and Climate Change Canada | [ECCC Data Servers End-use Licence](https://eccc-msc.github.io/open-data/licence/readme_en/) (commercial use allowed) | 2016-2021 |
| [Government of Canada AI Register (MVP)](https://open.canada.ca/data/dataset/fcbc0200-79ba-4fa4-94a6-00e32facea6b) | Treasury Board of Canada Secretariat | Open Government Licence - Canada | 412 systems, April 2026 release |

Only aggregated measurements and descriptions of AI systems are used: no person is involved. From the register, the
vendor and personal-information-bank columns are dropped; the columns kept hold no name, email or phone number. Raw
files stay out of git (`data/`); `sources.json` holds their checksums.

## Run

```bash
pip install "pulsar-metrics @ git+https://github.com/Rocket-Science-Development/pulsar_metrics.git" \
            "pulsar-data-collection[genai,detectors] @ git+https://github.com/Rocket-Science-Development/pulsar_data_collection.git" \
            scikit-learn openai requests
python download.py --data data                       # 79 files, about 60 MB

# InfluxDB 2.6 and Ollama (see genai_agent/docker-compose.yml); INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG in the environment
# or in an env file (--env-file, DOCKER_INFLUXDB_INIT_* keys accepted); the bucket essai_gc is created with 90 days retention
python tabular_replay.py --data data --campaign no2-20261104
python tabular_analyze.py --data data --campaign no2-20261104

ollama pull llama3.2:3b && ollama pull granite3.3:2b && ollama pull nomic-embed-text
python rag_campaign.py --data data --campaign registre-gc-1 --workers 2
python rag_analyze.py --campaign registre-gc-1 --window 8
```

Replace `registre-gc-1` by a new name for another run: records are selected by their `campaign` tag.

## Results (4 November 2026, CPU only)

**Part 1, NO2 of Montréal.** Reference: out-of-fold predictions of 2016-2018 (MAE 2.65 ppb). Stable period: 62 weeks
(January 2019 to early March 2020); changed period: 94 weeks from 16 March 2020. The model error fell from -0.31 ppb
(stable) to -2.60 ppb in spring 2020.

| Reference | Signal | Confirmed false alarms (stable weeks) | First alert after 16 March 2020 | First confirmed | Changed weeks confirmed |
|---|---|---|---|---|---|
| same weeks of 2016-2018 +- 4 | model error (ks_2samp, wasserstein) | 0 / 62 [0, 0.058] | +2 weeks | +3 weeks | 34 / 94 |
| same weeks of 2016-2018 +- 4 | weekly MAE | 0 / 62 | +42 weeks | never | 0 / 94 |
| same weeks of 2016-2018 +- 4 | weather inputs (negative control) | 0 / 62 | +4 weeks | never | 0 / 94 |
| whole of 2016-2018 | model error | 0 / 62 | +0 weeks | +1 week | 13 / 94 |

- The test on the distribution of the error sees the lockdown; the MAE barely moves (2.7 to 3.1 ppb).
- The period reference confirms all 13 weeks of the second quarter of 2020; its 27 windows make the thresholds
  provisional (`reliable=False`).

**Part 2, agent on the AI Register.** 240 runs of `llama3.2:3b` (and `granite3.3:2b` for the model change), windows of
8 runs, confirmation over 2 windows.

| Phase | Accuracy against the register [95 % CI] | Seen by | First confirmed window |
|---|---|---|---|
| reference (80 runs) | 0.90 [0.81, 0.96] | — | — |
| holdout, no change (40) | 0.88 [0.73, 0.96] | nothing confirmed (3 isolated raw alerts) | — |
| half of the questioned systems removed (40) | 0.57 [0.41, 0.73] | faithfulness and retrieval score drop, not confirmed with windows of 8; faithfulness confirmed with windows of 20 | 2 (windows of 20) |
| questions in English (40) | 0.90 [0.76, 0.97] | MMD of the request embeddings; S2: `output_tokens` | 2 (MMD), 4 (S2) |
| granite3.3:2b (40) | 0.25 [0.13, 0.41] | S2 (9 features: first tool, steps, tokens...), retrieval score, canary | 2 |

- Removing documents hurts the user most and is seen last: on the questions whose system had been removed, the agent
  mostly made up an answer instead of saying the system is not in the register.
- granite3.3:2b never called the tool: it answered from memory, in English.
- `FaithfulnessScorer` of pulsar-data-collection 0.11.0 is not thread-safe: called from 4 threads it raised
  `RuntimeError: Already borrowed` in 37 calls out of 40 (1 run out of 240 failed with 2 workers). Run the detectors
  from one thread, or behind a lock.
