"""LA-03 — Agent failure modes (S5): tool faults, loops, skipped steps, out-of-scope tools.

The demo agent (llama3.2:3b, prompt v1, temperature 0) runs under four injected failure modes at five intensities:

| mode | injection | intended feature |
|---|---|---|
| timeout | share of check_eligibility calls raise TimeoutError | tool_error_rate (calibrated drift) |
| garbage | share of check_eligibility calls return junk + "retry" (loop trigger) | loop_detected / repeated_calls |
| removed | share of runs without check_eligibility in the catalogue (skipped step) | LTLf amount-after-eligibility, task_success, tool sequence features |
| extra | share of runs with an extra, tempting tool outside the allowed list | out_of_scope_calls / LTLf in-scope-tools |

Each (mode, intensity) runs one window of 30 runs, tested against the reference of LS-01 (same agent and conditions)
or a fresh one. A window detects the mode when the intended feature alerts; the detection is confirmed when the next
intensity also alerts (persistence over 2 consecutive windows of rising dose). A/A: two windows without fault, no
alert allowed. (One window per intensity instead of two: the GPU endpoint is shared with the LR/LM family tonight.)
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from lab.tests import _agentlab as A

DRY = os.environ.get("LAB_DRY") == "1"
PARAMS = {
    "model": A.MODEL, "window": 6 if DRY else 30, "windows_per_level": 1,
    "levels": [0.8] if DRY else [0.1, 0.2, 0.35, 0.5, 0.8], "modes": ["timeout", "garbage", "removed", "extra"],
    "reference_runs_if_fresh": 12 if DRY else 300, "workers": int(os.environ.get("LAB_WORKERS", "4")),
    "n_draws": 20 if DRY else 200, "alpha": 0.05, "dry": DRY,
    "second_model": "llama3.1:8b", "second_model_modes": ["garbage", "extra"],
}
CRITERIA = {"modes_detected": ("==", 4), "aa_alarms": ("==", 0)}
INTENDED = {
    "timeout": {"drift:tool_error_rate"},
    "garbage": {"abs:loop_detected", "drift:repeated_calls"},
    "removed": {"ltlf:amount-after-eligibility", "drift:task_success", "drift:distinct_tools", "drift:tool_calls",
                "drift:first_tool"},
    "extra": {"abs:out_of_scope", "ltlf:in-scope-tools", "drift:out_of_scope_calls"},
}


def load_reference(ctx, p, log):
    path = Path(ctx.workdir).parent / "LS-01" / "reference_runs.csv"
    if path.exists() and not DRY:
        ref = A.read_runs(path)
        if len(ref) >= 10 * p["window"]:
            ctx.note(f"reference: {len(ref)} runs of LS-01 (same agent, model, prompt and task generator)")
            return ref, "LS-01"
    runs = A.run_block("reference", p["reference_runs_if_fresh"], seed=1000, workers=p["workers"], log=log)
    ref = A.records_for(runs)
    ctx.note(f"reference: {len(ref)} fresh runs")
    return ref, "fresh"


def scripted_trajectories() -> dict:
    """Feed hand-written span trees to the real TrajectoryAssembler: which loop and out-of-scope shapes are seen?"""
    from pulsar_data_collection.genai import TrajectoryAssembler
    from pulsar_data_collection.genai.trajectory import SpanView
    from pulsar_data_collection.genai import semconv as sc

    cases = {
        "identical_x3": [("check_eligibility", {"program_id": "PRG-LOGE", "annual_income": 36000, "age": 40})] * 3,
        "identical_x2": [("check_eligibility", {"program_id": "PRG-LOGE", "annual_income": 36000, "age": 40})] * 2,
        "ping_pong_x6": [("search_programs", {"topic": "rent"}),
                         ("check_eligibility", {"program_id": "PRG-LOGE", "annual_income": 36000, "age": 40})] * 3,
        "arg_jitter_x4": [("check_eligibility", {"program_id": "PRG-LOGE", "annual_income": 36000 + i, "age": 40})
                          for i in range(4)],
        "arg_type_jitter_x4": [("check_eligibility", {"program_id": "PRG-LOGE", "annual_income": v, "age": 40})
                               for v in (36000, 36000.0, "36000", 36000)],
        "same_tool_other_ids_x4": [("check_eligibility", {"program_id": pid, "annual_income": 36000, "age": 40})
                                   for pid in ("PRG-LOGE", "rent", "loyer", "PRG-LOGE")],
        "extra_tool": [("lookup_applicant_history", {}), ("check_eligibility", {"program_id": "PRG-LOGE",
                                                                                 "annual_income": 1, "age": 40})],
        "normal": [("search_programs", {"topic": "rent"}),
                   ("check_eligibility", {"program_id": "PRG-LOGE", "annual_income": 36000, "age": 40}),
                   ("compute_amount", {"program_id": "PRG-LOGE", "annual_income": 36000})],
    }
    out = {}
    for t, (name, calls) in enumerate(cases.items(), start=1):
        asm = TrajectoryAssembler(allowed_tools=A.T.ALLOWED_TOOLS)
        spans = [SpanView(t, 100 + i, 1, name, 10 * i, 10 * i + 5, False,
                          {sc.OPERATION: sc.OP_EXECUTE_TOOL, sc.TOOL_NAME: tool, sc.TOOL_CALL_ARGUMENTS: json.dumps(args)})
                 for i, (tool, args) in enumerate(calls)]
        root = SpanView(t, 1, None, "invoke_agent", 0, 10 * len(calls) + 10, False,
                        {sc.OPERATION: sc.OP_INVOKE_AGENT, sc.AGENT_NAME: "permit_assistant"})
        records = [r for s in spans + [root] for r in asm.add(s)]
        run = next(r for kind, r, _ in records if kind == "agent_run")
        out[name] = {k: run[k] for k in ("tool_calls", "repeated_calls", "loop_detected", "out_of_scope_calls",
                                         "distinct_tools")}
    return out


def run(ctx):
    from pulsar_metrics.genai import ComplianceMonitor, load_rules, trace_from

    p = ctx.params
    log = lambda m: print(f"[LA-03 {time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    ref, ref_source = load_reference(ctx, p, log)
    ref = ref[ref["error"].isna() & (ref["matched"].astype(str) == "True")]
    # latency_s is left out: tonight it follows the load of the shared GPU endpoint, not the agent
    tester = A.WindowTester(ref, window=p["window"], n_draws=p["n_draws"], alpha=p["alpha"],
                            numeric=[c for c in A.NUMERIC if c != "latency_s"])
    monitor = ComplianceMonitor(load_rules(str(A.RULES)), version="permit-agent-rules/1")
    ref_viol = {r.id: int(sum(not r.check(trace_from(str(s))) for s in ref["tool_sequence"])) for r in monitor.rules}
    n_ref = len(ref)

    def detect(w: pd.DataFrame) -> dict:
        w = w[w["error"].isna() & (w["matched"].astype(str) == "True")]
        if len(w) < p["window"] // 2:
            return {"fired": [], "runs": int(len(w)), "insufficient": True, "ltlf": {}, "tool_error_rate": float("nan"),
                    "loop_runs": 0, "repeated_calls_mean": float("nan"), "out_of_scope_runs": 0,
                    "fault_active_runs": 0, "task_success": float("nan"), "max_steps_runs": 0}
        d = tester.test(w)
        fired = {f"drift:{f}" for f in d.loc[d["flag"], "feature_name"]}
        if (w["loop_detected"].astype(str) == "True").any():
            fired.add("abs:loop_detected")
        if (pd.to_numeric(w["out_of_scope_calls"], errors="coerce").fillna(0) > 0).any():
            fired.add("abs:out_of_scope")
        ltlf = {}
        for rule in monitor.rules:
            k = int(sum(not rule.check(trace_from(str(s))) for s in w["tool_sequence"]))
            # one-sided Fisher: violation rate higher than in the reference
            pv = stats.fisher_exact([[k, len(w) - k], [ref_viol[rule.id], n_ref - ref_viol[rule.id]]],
                                    alternative="greater")[1]
            absolute = rule.severity == "critical" and ref_viol[rule.id] == 0 and k > 0
            ltlf[rule.id] = {"violations": k, "p": float(pv)}
            if pv < p["alpha"] or absolute:
                fired.add(f"ltlf:{rule.id}")
        return {"fired": sorted(fired), "runs": int(len(w)), "ltlf": ltlf,
                "tool_error_rate": float(pd.to_numeric(w["tool_error_rate"], errors="coerce").mean()),
                "loop_runs": int((w["loop_detected"].astype(str) == "True").sum()),
                "repeated_calls_mean": float(pd.to_numeric(w["repeated_calls"], errors="coerce").mean()),
                "out_of_scope_runs": int((pd.to_numeric(w["out_of_scope_calls"], errors="coerce") > 0).sum()),
                "fault_active_runs": int((w["fault_active"].astype(str) == "True").sum()),
                "task_success": float((w["task_success"].astype(str) == "True").mean()),
                "max_steps_runs": int((w["termination"].astype(str) == "max_steps").sum())}

    blocks, all_frames = [], []
    seed = 5000
    plan = [("none", 0.0)] + [(m, lv) for m in p["modes"] for lv in p["levels"]]
    jobs = []
    for mode, level in plan:
        for k in range(1 if mode == "none" and DRY else (2 if mode == "none" else p["windows_per_level"])):
            seed += 1
            jobs.append((f"{mode}@{level}#{k + 1}", mode, level, seed))
    old_frame, keep = None, set()
    if os.environ.get("LAB_RESUME") == "1" and (ctx.workdir / "runs.csv").exists():
        # resume: keep the blocks already collected, rerun (same seeds) those where > 20 % of the runs failed
        old_frame = A.read_runs(ctx.workdir / "runs.csv")
        share = old_frame.groupby("label")["error"].apply(lambda e: e.notna().mean())
        keep = {lab for lab, v in share.items() if v <= 0.2}
        ctx.note(f"resume: {len(keep)} blocks kept from the first pass; rerun: "
                 f"{sorted(set(j[0] for j in jobs) - keep)} (endpoint OOM-killed during the first pass)")
    for label, mode, level, sd in jobs:
        if label in keep:
            continue
        runs = A.run_block(label, p["window"], seed=sd, mode=mode, intensity=level, workers=p["workers"], log=log)
        all_frames.append(runs)
    frame = A.records_for([r for runs in all_frames for r in runs]) if all_frames else None
    if old_frame is not None:
        kept = old_frame[old_frame["label"].isin(keep)]
        frame = kept if frame is None else pd.concat([kept, frame], ignore_index=True)
        order = {j[0]: i for i, j in enumerate(jobs)}
        frame = frame.sort_values("label", key=lambda c: c.map(order), kind="stable").reset_index(drop=True)
    A.save_frame(frame, ctx.workdir / "runs.csv")
    for label, w in frame.groupby("label", sort=False):
        mode, rest = label.split("@")
        level, k = rest.split("#")
        res = detect(w)
        res.update(mode=mode, level=float(level), window=int(k))
        if res.get("insufficient"):
            ctx.note(f"{label}: only {res['runs']} usable runs, window not tested")
        res["intended_fired"] = sorted(set(res["fired"]) & INTENDED.get(mode, set()))
        res["other_fired"] = sorted(set(res["fired"]) - INTENDED.get(mode, set()))
        blocks.append(res)
        log(f"{label}: fired={res['fired']}")

    # absolute detectors with a model that chains tool calls (no reference needed: loop / out-of-scope are absolute)
    second = {}
    if os.environ.get("LAB_SKIP_SECOND") == "1":
        ctx.note(f"{p['second_model']} pass skipped: lab-gpu-2 was OOM-wedged twice (06:10-06:46, 06:57-07:44 UTC) "
                 "while 8B runners of two families were loaded")
    for mode in ([] if os.environ.get("LAB_SKIP_SECOND") == "1" else p["second_model_modes"]):
        runs = A.run_block(f"{p['second_model']}:{mode}@0.8", p["window"], seed=9000 + len(second), mode=mode,
                           intensity=0.8, workers=p["workers"], model=p["second_model"], log=log)
        f2 = A.records_for(runs)
        f2 = f2[f2["matched"].astype(str) == "True"]
        second[mode] = {"runs": int(len(f2)),
                        "loop_runs": int((f2["loop_detected"].astype(str) == "True").sum()),
                        "repeated_calls_max": float(pd.to_numeric(f2["repeated_calls"], errors="coerce").max()),
                        "out_of_scope_runs": int((pd.to_numeric(f2["out_of_scope_calls"], errors="coerce") > 0).sum()),
                        "mean_tool_calls": float(pd.to_numeric(f2["tool_calls"], errors="coerce").mean()),
                        "max_steps_runs": int((f2["termination"].astype(str) == "max_steps").sum()),
                        "sequences": f2["tool_sequence"].astype(str).value_counts().head(8).to_dict()}
        log(f"second model {mode}: {second[mode]}")
    scripted = scripted_trajectories()
    loops = ("ping_pong_x6", "arg_jitter_x4", "arg_type_jitter_x4", "same_tool_other_ids_x4")
    missed = {k: v for k, v in scripted.items() if k in loops and not v["loop_detected"]}
    if missed:
        ctx.limit(
            f"loop_detected misses {len(missed)} of {len(loops)} scripted loop shapes (4-6 calls of the same tool): {sorted(missed)}; it needs >= 2 "
            "consecutive calls with byte-identical tool+arguments (repeated_calls >= 2)",
            json.dumps(scripted),
            "Add loop features that survive argument jitter and alternation: max calls of the same tool per run, "
            "cycle detection on the tool-name sequence (period 1-3), and argument normalisation (numbers, case) "
            "before hashing; expose 'same_tool_calls_max' in agent_run",
            "major",
        )
    garbage_rows = frame[(frame["mode"] == "garbage") & (frame["kind"] == "eligibility")]
    if len(garbage_rows):
        silent = float((pd.to_numeric(garbage_rows["tool_error_rate"], errors="coerce").fillna(0) == 0).mean())
        ctx.note(f"garbage mode, eligibility tasks: {silent:.0%} of {len(garbage_rows)} runs show tool_error_rate 0 "
                 "(the junk output is not an exception)")
    aa = [b for b in blocks if b["mode"] == "none"]
    aa_alarms = sum(bool(b["fired"]) for b in aa)
    if aa_alarms:
        ctx.limit(
            f"A/A: {aa_alarms} of {len(aa)} fault-free windows of {p['window']} runs raise an alert",
            json.dumps([{"window": b["window"], "fired": b["fired"]} for b in aa]),
            "Calibrate the absolute trajectory detectors and LTLf rule rates on the reference (not min_compliance=1.0)",
            "major",
        )
    summary = {}
    for mode in p["modes"]:
        rows = sorted((b for b in blocks if b["mode"] == mode), key=lambda b: (b["level"], b["window"]))
        by_level = {}
        for b in rows:
            by_level.setdefault(b["level"], []).append(b)
        raw = [lv for lv, bs in by_level.items() if any(b["intended_fired"] for b in bs)]
        # persistence 2: the intended feature alerts at this intensity and at the next one (consecutive windows of
        # rising dose), or in both windows of the same intensity when two are run
        lvs = sorted(by_level)
        hit = {lv: all(b["intended_fired"] for b in by_level[lv]) for lv in lvs}
        confirmed = [lv for i, lv in enumerate(lvs) if hit[lv] and (len(by_level[lv]) > 1 or
                                                                     (i + 1 < len(lvs) and hit[lvs[i + 1]]))]
        summary[mode] = {
            "min_level_raw": min(raw) if raw else None, "min_level_confirmed": min(confirmed) if confirmed else None,
            "levels": {str(lv): {"intended": sorted({f for b in bs for f in b["intended_fired"]}),
                                 "other": sorted({f for b in bs for f in b["other_fired"]}),
                                 "tool_error_rate": round(float(np.mean([b["tool_error_rate"] for b in bs])), 3),
                                 "loop_runs": sum(b["loop_runs"] for b in bs),
                                 "out_of_scope_runs": sum(b["out_of_scope_runs"] for b in bs),
                                 "fault_active_runs": sum(b["fault_active_runs"] for b in bs),
                                 "task_success": round(float(np.mean([b["task_success"] for b in bs])), 3),
                                 "max_steps_runs": sum(b["max_steps_runs"] for b in bs),
                                 "runs": sum(b["runs"] for b in bs)} for lv, bs in by_level.items()},
        }
    # per-tool view that agent_run cannot give: share of runs calling check_eligibility whose run had a tool error
    def elig_err(f):
        f = f[f["tool_sequence"].astype(str).str.contains("check_eligibility")]
        k = int((pd.to_numeric(f["tool_error_rate"], errors="coerce").fillna(0) > 0).sum())
        return k, int(len(f))

    rk, rn = elig_err(ref)
    per_tool = {}
    for lv in p["levels"]:
        w = frame[(frame["mode"] == "timeout") & (frame["intensity"].astype(float) == lv)]
        k, n = elig_err(w)
        pv = stats.fisher_exact([[k, n - k], [rk, rn - rk]], alternative="greater")[1] if n else float("nan")
        per_tool[str(lv)] = {"eligibility_runs_with_error": f"{k}/{n}", "fisher_p_vs_reference": round(float(pv), 6)}
    for mode, sm in summary.items():
        lv = sm["levels"]
        if mode == "timeout" and sm["min_level_confirmed"] is None:
            ctx.limit(
                "timeout: injected TimeoutError on 10-80 % of check_eligibility calls is never flagged by the "
                "calibrated drift of tool_error_rate (windows of 30 runs, BH over 11 features), even at 80 %",
                f"window mean tool_error_rate by intensity { {k: v['tool_error_rate'] for k, v in lv.items()} } vs "
                f"reference {pd.to_numeric(ref['tool_error_rate'], errors='coerce').mean():.3f}; the fault touches "
                f"only runs that call check_eligibility; per-tool view (Fisher vs reference {rk}/{rn}): {per_tool}",
                "Monitor the error rate per tool from the tool_call records (binomial / Fisher test per tool vs the "
                "reference, BH over tools) instead of only the run-level mean; add per-tool error counts to agent_run",
                "major",
            )
        elif mode == "garbage" and sm["min_level_raw"] is None:
            ctx.limit(
                "garbage: a tool that returns junk with status 'retry' on 10-80 % of check_eligibility calls is "
                "invisible: no trajectory feature moves (no exception, and the 3B agent answers after one call)",
                f"by intensity: { {k: {'tool_error_rate': v['tool_error_rate'], 'loop_runs': v['loop_runs'], 'task_success': v['task_success']} for k, v in lv.items()} }; "
                "task_success stays high because it only checks that the expected tool was called",
                "Record a tool-output validity signal on tool spans (schema check of the returned JSON, "
                "pulsar.tool.output_valid) and count invalid outputs in agent_run; define task_success on the "
                "answer's consistency with tool results, not only on the tool being called",
                "major",
            )
        elif mode == "removed" and (sm["min_level_raw"] is None or sm["min_level_raw"] > 0.35):
            ctx.limit(
                f"removed: a missing check_eligibility tool is detected only from {sm['min_level_raw']} of runs "
                f"affected (raw) / {sm['min_level_confirmed']} confirmed, through first_tool/task_success drift; the "
                "LTLf rule amount-after-eligibility never contributes because it is already violated by ~29 % of "
                "runs at rest",
                json.dumps({k: {x: v[x] for x in ('intended', 'other', 'fault_active_runs', 'task_success')}
                            for k, v in lv.items()}),
                "Emit a configuration record with the tool catalogue offered to the model (hash + list) on each "
                "run, so a removed tool is a configuration change detected at the first run",
                "major",
            )
        elif mode == "extra":
            offered = sum(v["fault_active_runs"] for v in lv.values())
            used = sum(v["out_of_scope_runs"] for v in lv.values())
            ctx.note(f"extra: the extra tool was offered in {offered} runs and called in {used} by llama3.2:3b "
                     "(single-call agent), so the live path was not exercised; the scripted trajectory confirms that "
                     f"an out-of-catalogue call is counted (out_of_scope_calls={scripted['extra_tool']['out_of_scope_calls']})")
    ctx.note(f"per-tool check_eligibility error view under timeouts: {per_tool}")
    (ctx.workdir / "report.json").write_text(json.dumps({"summary": summary, "blocks": blocks,
                                                         "reference": ref_source, "reference_runs": n_ref,
                                                         "second_model": second, "scripted": scripted,
                                                         "reference_rule_violations": ref_viol}, indent=1, default=str),
                                             encoding="utf-8")
    detected = [m for m, sm in summary.items() if sm["min_level_raw"] is not None]
    return {"modes_detected": len(detected), "aa_alarms": aa_alarms, "reference_runs": n_ref, "reference": ref_source,
            "runs": len(frame), "failed_runs": int(frame["error"].notna().sum()),
            "unmatched_records": int((~frame["matched"].astype(bool)).sum()),
            **{f"{m}_min_level_raw": s["min_level_raw"] for m, s in summary.items()},
            **{f"{m}_min_level_confirmed": s["min_level_confirmed"] for m, s in summary.items()}}
