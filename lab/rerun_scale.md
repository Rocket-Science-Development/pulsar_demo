# Scalability rerun

Reruns the platform tests of the night of 2026-11-02 on the same machines and at the same sizes, after the fixes of
pulsar_data_collection 0.10.0 and pulsar_metrics 0.11.0. Every test runs in two write profiles, so the gain of the
always-on fixes and the gain of the multi-model configuration are measured apart. The night's results
(`results/<id>.json`) are the baseline.

| Profile | `LAB_PROFILE` | What it measures |
|---|---|---|
| Night (baseline) | — | 0.8 / 0.9 behaviour, already measured |
| `default` | `default` | 0.10 with the default options: one private writer per Pulse. The always-on fixes are included: 1 µs row spacing, `/ping` probe during back-off, 422 split, `uncertain_rows` |
| `tuned` | `tuned` | One `SharedWriter(concurrency=4)` per step for all the Pulse, `spool_compression="gzip"` (SC-03), `durability="wal"` (SC-13) |

Results go to `results/<id>.<profile>.json` (harness: `LAB_RUN_TAG`), next to the night's files, which stay intact.

## Tests and the limits they recheck

| Test | Size (as on the night) | Limits rechecked | Expected |
|---|---|---|---|
| SC-01 | 1 to 500 Pulse, 60 s per level | M31, M32 | tuned: 2 writer threads instead of N+1; p99 and achieved rate at 100 to 500 models; `uncertain_rows` instead of false losses |
| SC-02 | 10k to 4M rows/min, 1 then 100 models, 3 min per step | M33, M34, M31 | default: timestamps no longer ahead of the clock; tuned: higher saturation point |
| SC-03 | outages of 60 s, 10 min, 30 min + black hole | M35, m18, m19 | default: no loss after recovery (probe); tuned: smaller spool (gzip) |
| SC-13 | kill -9, 3 scenarios × 3 repetitions | M36, m24 | tuned (WAL): no loss of the in-memory queue; `.tmp` files recovered |
| SC-05 | max_queue_rows 1k to 1M | m20 | a batch larger than `max_queue_rows` is dropped and counted |
| SC-14 | 7 schema cases | M37, M38 | no valid row lost; counters exact |

SC-05 and SC-14 do not use a shared writer: they run once, with the `default` profile.

## Machines and cost (ca-central-1, on demand)

| Machine | Instance | Role | USD/h |
|---|---|---|---|
| lab-scale | c7i.8xlarge | Pulse processes and load generator | 1.56 |
| lab-db | r7i.2xlarge | InfluxDB 2.6.1 (`ulimit nofile` 65536) + Grafana 11.3.0 | 0.58 |

| Test | Per profile | Runs | Total |
|---|---|---|---|
| SC-01 | about 12 min | 2 | 25 min |
| SC-02 | about 38 min (2177 s on the night) | 2 | 75 min |
| SC-03 | about 1 h 50 (6382 s on the night) | 2 | 3 h 40 |
| SC-13 | about 4 min | 2 | 10 min |
| SC-05, SC-14 | about 4 min | 1 | 5 min |
| Provisioning, dry runs, report | | | about 45 min |

The tests run one after the other because they share InfluxDB. That is about 6 h 20 of two machines: **about 14 USD**,
plus about 0.30 USD of disks. The proposed cap is **25 USD**, with an automatic stop 7 h after the start. The machines
are created for the rerun (tag `Project=pulsar-lab`) and **terminated** at teardown (`teardown: "terminate"`), with
their volumes. No disk is left behind. No GPU host, no lab-soak and no lab-cpu.

To halve the cost, drop the 30-min outage of SC-03 in the `default` profile (`LAB_OUTAGES=60,600`): the total becomes
about 4 h 40, or about 10 USD. The 30-min outage is still measured in the `tuned` profile.

## Before launching

- Approval of the cost cap and of the stop time.
- On each host, install pulsar_data_collection and pulsar_metrics from `main` (0.10.0 and 0.11.0). The test results
  record the versions (`params.write`).
- An AWS Budgets alarm at the cap (recommended, not created by the lab).

## Launch

`rerun_scale.json` holds the workflow arguments: the six tests from `approved_tonight.json` and the settings
(`run_name`, `hosts`, `profiles`, `teardown`). Fill in `stop`, `stamp` and `night_report`, then start
`night_lab.workflow.js` with that file as `args`. The workflow writes `results/run_platform.relance-scale.json`,
`results/night_full_results.relance-scale.json` and the comparison report `results/rapport_relance_scale.md`.

One profile of one test, locally or on a host:

```bash
LAB_PROFILE=tuned LAB_RUN_TAG=tuned python -m lab.harness SC-01
```

`LAB_PLATFORM_INFLUX_ENV` points at another InfluxDB env file than `~/.pulsar-lab/influx2.env`. The file can also be a
docker `.env` (`DOCKER_INFLUXDB_INIT_ADMIN_TOKEN`, `DOCKER_INFLUXDB_INIT_ORG`), with `LAB_INFLUX_URL_OVERRIDE` for the
URL.
