# Remeasure of the write path (pulsar_data_collection 0.11, pulsar_metrics 0.12)

The figures of the scale rerun of 2026-11-03 were measured with pulsar_data_collection 0.10.0. Version 0.11.0 changed
the write path (column-wise line-protocol encoder, `SharedWriter` throughput, uncertain rows after a cut connection),
and the `SharedWriter` gain (5.9k → 24.8k rows/s at 100 models) was measured on a laptop only. The figures cited as
evidence must come from the versions that ship, on the lab machines, so this rerun repeats the write-path tests with
the same machines, sizes and profiles as the 2026-11-03 rerun (`rerun_scale.md`), on the released versions.

Every result now carries an `environment` block (`lab/environment.py`): CPU model, cores, RAM, instance type (EC2
metadata), OS, Python, package versions, InfluxDB / Grafana / Ollama versions, git commits of the checkouts.

## Tests

| Test | Profiles | Why | Duration |
|---|---|---|---|
| SC-01 | default + tuned | capture p99 and achieved rate at 1 to 500 models with the new encoder | about 25 min |
| SC-02 | default + tuned | rows/s per writer and saturation point (encoder, `SharedWriter`) | about 75 min |
| SC-13 | tuned | kill -9 with WAL: still 0 loss after the write-path changes | about 10 min |
| SC-05, SC-14 | default | queue overflow and schema cases: counters still exact | about 5 min |
| SC-03 | tuned, outages 60 s and 10 min | spool and replay after the change of uncertain-row accounting | about 40 min |
| Provisioning, dry run, report | | | about 45 min |

Total about 3 h 20 of the two machines. SC-03 in the default profile and the 30-min outage are not rerun: the 0.11
changes do not touch the outage detection, and the 2026-11-03 figures stand.

## Machines and cost (ca-central-1, on demand)

| Machine | Instance | Role | USD/h |
|---|---|---|---|
| lab-scale | c7i.8xlarge | Pulse processes and load generator | 1.56 |
| lab-db | r7i.2xlarge | InfluxDB 2.6.1 (`ulimit nofile` 65536) | 0.58 |

About 3 h 20 × 2.14 USD/h ≈ **7.2 USD**, plus about 0.20 USD of disks and public IPv4. Proposed cap: **12 USD**, with an
automatic stop 4 h after the start. The machines are created for the rerun (tag `Project=pulsar-lab`) and
**terminated** with their volumes at teardown. No GPU host. The instance `pulsar-gpu-s2` is not touched.

## Before launching

- Approval of the cost cap and of the stop time.
- The 0.11.0 / 0.12.0 merge commits (`e67491a`, `cd18c94`) tagged `0.11.0` / `0.12.0` (the repos tag without a `v`), and the hosts installed
  from those tags (the `environment` block records the commit and tag of each checkout).
- An AWS Budgets alarm at the cap (recommended, not created by the lab).

## Launch

Same procedure as `rerun_scale.md`, with `LAB_RUN_TAG=v011-<profile>` so the results land next to the 2026-11-03
files (`results/<id>.v011-default.json`, `results/<id>.v011-tuned.json`) and `LAB_OUTAGES=60,600` for SC-03.
