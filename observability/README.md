# Pulsar.ml observability: InfluxDB 2 and Grafana

Eight provisioned Grafana dashboards for everything Pulsar.ml measures: drift, performance, agents, fairness,
explanations, personal information, and the health of the capture itself. They replace the 2023 dashboards of the
classic demo (InfluxQL on InfluxDB 1.5, kept frozen at the repository root).

| Dashboard | What it shows |
|---|---|
| **Overview** (home) | Latest verdicts of each model: confirmed drift, disparities, explanation changes, masked personal information, dropped rows; agent runs, tool errors, leaks. Every tile links to its dashboard. |
| **Data drift** | Raw alerts and confirmed drift per feature and window, adjusted p-values (A/A-calibrated, Benjamini-Hochberg), hourly quantile band of the selected feature, latest tests. |
| **Model performance** | Accuracy, precision, recall, F1 by window with the accuracy floor; label coverage (labels arrive late); approval rate and mean score. |
| **Agents and LLM applications** | Runs, latency (median, p95), tool error rate, task success, tokens; first tool, errors by tool, terminations; confirmed drift of the agent per feature. |
| **Fairness** | Latest verdict per group with interval and size, disparate impact ratio against the 4/5 zone, parity difference with its interval, anytime-valid monitor (e-value, alarm line), intersections with small cells withheld, counterfactual probe of the agent. |
| **Explanations** | Share of importance per feature (TreeSHAP), ranking stability (NDCG, top-k overlap), changes of weight with intervals, importance by group. |
| **Personal information** | Findings masked before writing, share of rows concerned, agent personal information by role, leaks and leaked types. |
| **Capture health** | Captured, written, spooled, dropped rows, write errors, completeness, storage availability. |

Conventions: a verdict is always shown with its interval and its size; **▲ established**, **✓ within zone** and
**? inconclusive** carry the fixed status colors, always with a symbol and a label; series use the categorical palette,
never a status color.

## Data model

| Measurement | Written by | Content |
|---|---|---|
| `<model>_<version>_input_data` | `Pulse` (pulsar-data-collection) | captures: features, predictions, labels when known, `pii_count`; agent records (`record_type` agent_run / tool_call / llm_call) |
| `pulsar_results` | `ResultsSink` (pulsar-data-collection) | one point per `MetricResults`: tags `metric_type`, `metric_name`, `feature_name`, `group`, `reference_group`, `test_type`, plus caller tags (`campaign`, `phase`, `scenario`); fields `metric_value`, `ci_low`, `ci_high`, `threshold*`, `n`, `e_value`, `status` (1 established, 0 within, -1 inconclusive), `p_adjusted`, `confirmed` |
| `pulsar_capture_stats` | `StatsReporter` (pulsar-data-collection) | counters of each Pulse |
| `pulsar_feature_quantiles` | `analysis_job.py`, `seed_demo.py` (`analysis.write_hourly_quantiles`, through `ResultsSink.write_stats_many`) | per model and numeric feature, one point per hour stamped at the end of the hour: tags `model_id`, `model_version`, `feature_name`; fields `p10`, `median`, `p90` (exact quantiles of the rows of that hour), `n` |

## Run it

```bash
docker compose up -d                       # InfluxDB 2.6.1 on :8086, Grafana 11.3.0 on :3000
export INFLUX_URL=http://localhost:8086 INFLUX_TOKEN=demo-token-change-me INFLUX_BUCKET=monitoring
python seed_demo.py                        # a week of demo data and its analyses (about a minute)
```

Open http://localhost:3000: the overview is the home page (anonymous read access; admin password in the compose file,
to change before any shared use).

- `seed_demo.py` fills a week: a synthetic eligibility model scored every hour, with an income drop from day 4 and a
  biased production rule from day 5, personal information masked in a free-text field, labels 48 hours late; the
  analyses every 6 hours. `--agent-campaign` analyses the agent runs of a campaign already in the bucket (see
  `genai_agent/campaign.py`); `--probe-rows` adds the counterfactual probe results of `genai_agent/probe_agent.py`.
- `analysis_job.py` is the periodic analysis of a live model: every `--every` minutes it reads the window that just
  closed and writes its results (`--once` for a single run). It also recomputes the hourly quantiles of the last
  `--quantile-hours` closed hours (default 24, so that late rows count; rewriting an hour overwrites it), then
  backfills the older ones: over the last `--backfill-days` (default 30) it finds, in one grouped Flux query, the closed
  hours that have captures but no aggregate, or fewer aggregated values than captured ones (rows that arrived after the
  hour was aggregated), and aggregates the oldest `--backfill-max-hours` of them (default 168 per model and run), in
  reads of at most 24 hours. The history thus fills up over a few runs, and a rerun on complete hours writes nothing.
  The hour being captured is never aggregated. For a model already captured before the job, fill the history at once
  with `--backfill-only` (no analyses, no reference needed; `--backfill-days 365` for a longer history, `--models
  a:1.0,b:2.1` for several models found in the same query).
- `build_dashboards.py` regenerates the dashboards (`grafana/dashboards/*.json`) from their Python description; edit
  it rather than the JSON, then restart Grafana or wait for the provisioning to reload.

## What the demo shows, and what it taught

- Income drift: raw alert from day 4, **confirmed** one window later; no other feature drifts.
- Biased rule: disparate impact ratio of `language=other` about 0.6, **established** on the rolling 24-hour window;
  the anytime-valid monitor crosses its alarm line within a day. The explanations do not move: a rule added after the
  model is invisible to model explanations, as the dashboard says.
- Building it found a defect in `FairnessMonitor` (pairing with old reference decisions turned a change shared by
  every group into a false disparity), fixed in pulsar-metrics 0.9.1, and a calibration artifact of the demo itself
  (a reference of 8 contiguous windows is too short for a contiguous A/A calibration; the demo now draws random rows,
  which suits its independent rows).
- At scale (lab SC-07: 300 models, 30 days of results, 20 concurrent viewers) every panel stays under 3 s at 7 days.
  The model selector no longer puts a negation in its predicate: that made it linear in the number of series (3.8 s
  at 20,000 models, 0.1 s now). The drift feature selector lists numeric features only. The hourly quantile band
  read the raw captures (quantiles are not pushed down to storage in InfluxDB 2.6): about 9 s at 30 days with 20
  viewers. It now reads the hours pre-aggregated by the analysis job (`pulsar_feature_quantiles`); only the hours after
  the last aggregated one (24 at most) are computed from the raw captures. With 20 viewers at 30 days, the band now
  answers in 0.92 s at p95 (8.51 s before), and the whole drift dashboard in 1.92 s (8.79 s before). Older hours never
  aggregated showed as a gap; the job now backfills them (the panel query is unchanged). Measured on a local InfluxDB
  2.6.1 with 300 models x 20 numeric features x 30 days, 12 rows per hour (52 million raw values), the first 10 days
  never aggregated and 2 % of the other hours missing (74,700 model-hours): finding the gaps takes 7.8 s for the 300
  models in one query (13 s with one query per model, 43 ms each), 1.3 s for 50 models, 4.3 s for 300 models over 7
  days. Filling them costs about 20 ms per model-hour (read 5, quantiles 5, write 9 to 28: one write request per
  hour), so 6.6 s per model for a run capped at 168 hours, and about 40 min for the whole history of the 300 models
  at once. A rerun finds no gap and writes nothing; the backfilled values equal numpy's on the raw rows, and the band
  then shows the 720 hours of the month.
- Tests of the aggregation and the backfill: `python -m unittest discover -s tests -v` (from this folder); with
  `PULSAR_TEST_INFLUX_URL` and `PULSAR_TEST_INFLUX_TOKEN` set, one test also runs against a real InfluxDB 2 (it writes
  a model of its own and deletes it).
