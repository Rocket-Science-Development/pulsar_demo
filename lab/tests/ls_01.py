"""LS-01 — LLM false-alarm soak (A/A, all detectors).

A reference of the demo agent (llama3.2:3b, prompt v1, temperature 0, default task mix) then 10 windows of 60 runs
under the same conditions (fresh synthetic tasks from the same generator). Every window is tested against the
reference by every test Pulsar.ml offers for agents at rest:

- calibrated drift of the agent_run features (KS / chi-square, AACalibrator thresholds, BH over the features) and its
  confirmation over 2 consecutive windows (DriftPersistence);
- MMD on the embeddings of the requests and of the answers (nomic-embed-text);
- LTLf compliance rules (rules/permit_agent.yaml), as absolute per-window alerts;
- absolute trajectory detectors: loop_detected, out_of_scope_calls;
- personal-information leaks in the answers (PiiSession, rules);
- InjectionDetector on the requests and FaithfulnessScorer on rule answers, when torch is installed.

A false alarm is any alert in a window (nothing changed). Pass: false alarm rate <= 5 % per test.
An excess of alarms is a limit only when it survives BH across all the tests of the soak (one-sided binomial test of
the alarm count against alpha, per test, then Benjamini-Hochberg). The calibrated per-test drift flags without BH are
reported, never turned into a limit on their own. The reference holds >= 1200 runs (>= 20 disjoint windows of 60, so
the A/A draws of the calibrator do not overlap as they did with 596 runs) and the soak has 20 windows.
Also checked: every run produces exactly one agent_run record that matches what the agent did (exporter integrity).
"""

import json
import os
import random
import time

import numpy as np
import pandas as pd

from lab.tests import _agentlab as A

DRY = os.environ.get("LAB_DRY") == "1"
PARAMS = {
    "model": A.MODEL, "reference_runs": 20 if DRY else 1260, "windows": 2 if DRY else 20,
    "window": 10 if DRY else 60, "workers": int(os.environ.get("LAB_WORKERS", "4")), "persistence": 2,
    "alpha": 0.05, "n_draws": 20 if DRY else 200, "mmd_permutations": 100 if DRY else 500, "dry": DRY,
}
MIN_REFERENCE_WINDOWS = 20  # disjoint reference windows needed for non-overlapping A/A draws
CRITERIA = {"max_false_alarm_rate": ("<=", 0.05), "max_false_alarm_rate_drift_mmd": ("<=", 0.05),
            "record_match_rate": (">=", 0.999)}


def wilson(k, n):
    from scipy import stats
    if n == 0:
        return [None, None]
    ci = stats.binomtest(k, n).proportion_ci(method="wilson")
    return [round(float(ci.low), 4), round(float(ci.high), 4)]


def bh_excess(summary: dict, alpha: float) -> dict:
    """{test: (one-sided binomial p-value of its alarm count vs alpha, BH-adjusted q across all tests)}"""
    from scipy import stats
    names = [t for t, s in summary.items() if s["windows"]]
    pv = np.array([stats.binomtest(summary[t]["alarms"], summary[t]["windows"], alpha, alternative="greater").pvalue
                   for t in names])
    if not len(pv):
        return {}
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    q = np.empty_like(pv)
    q[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return {t: (round(float(a), 6), round(float(b), 6)) for t, a, b in zip(names, pv, q)}


def detectors(ctx):
    """Torch detectors if available (shared venv); otherwise None, noted"""
    try:
        from pulsar_data_collection.genai import FaithfulnessScorer, InjectionDetector
        root = "/data/lab/models"
        device = 0 if __import__("torch").cuda.is_available() else None
        inj = InjectionDetector(model=f"{root}/protectai/deberta-v3-base-prompt-injection-v2", device=device)
        faith = FaithfulnessScorer(model=f"{root}/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", device=device)
        return inj, faith
    except Exception as e:  # noqa: BLE001
        ctx.note(f"torch detectors unavailable in this venv ({type(e).__name__}: {str(e)[:120]}); injection and "
                 "faithfulness detectors not soaked here")
        return None, None


def exporter_idle_split(run_seconds: float = 45.0, tool_end_s: float = 5.0) -> dict:
    """Replay the export pattern of one slow run through the real TrajectoryAssembler (default trace_timeout_s=30):
    an LLM span and a tool span end early and are exported, other traffic triggers flush_idle, the root ends late"""
    from pulsar_data_collection.genai import TrajectoryAssembler
    from pulsar_data_collection.genai import semconv as sc
    from pulsar_data_collection.genai.trajectory import SpanView

    s = int(1e9)
    asm = TrajectoryAssembler(allowed_tools=A.T.ALLOWED_TOOLS)
    llm = SpanView(7, 2, 1, "chat", 0, 2 * s, False, {sc.OPERATION: sc.OP_CHAT, sc.REQUEST_MODEL: "llama3.2:3b"})
    tool = SpanView(7, 3, 1, "check_eligibility", 3 * s, int(tool_end_s * s), False,
                    {sc.OPERATION: sc.OP_EXECUTE_TOOL, sc.TOOL_NAME: "check_eligibility", sc.TOOL_CALL_ARGUMENTS: "{}"})
    root = SpanView(7, 1, None, "invoke_agent", 0, int(run_seconds * s), False,
                    {sc.OPERATION: sc.OP_INVOKE_AGENT, sc.AGENT_NAME: "permit_assistant"})
    out = asm.add(llm, now_ns=2 * s) + asm.add(tool, now_ns=int(tool_end_s * s))
    out += asm.flush_idle(now_ns=int((tool_end_s + 30.5) * s))  # another run's batch export 30.5 s later
    out += asm.add(root, now_ns=int(run_seconds * s))
    runs = [r for kind, r, _ in out if kind == "agent_run"]
    return {"agent_runs": len(runs), "tool_calls": [r["tool_calls"] for r in runs],
            "steps": [r["steps"] for r in runs], "tool_sequence": [r["tool_sequence"] for r in runs],
            "expected": {"agent_runs": 1, "tool_calls": 1, "steps": 1, "tool_sequence": "check_eligibility"}}


def run(ctx):
    from pulsar_data_collection.privacy import PiiSession
    from pulsar_metrics.analyzers.persistence import DriftPersistence
    from pulsar_metrics.genai import ComplianceMonitor, embedding_drift, load_rules, trace_from

    p = ctx.params
    log = lambda m: print(f"[LS-01 {time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    reanalyse = os.environ.get("LAB_REANALYSE") == "1" and (ctx.workdir / "runs.csv").exists()
    previous = {}
    if reanalyse:
        # re-run the analysis on the runs already collected (no model call); MMD p-values come from the first report
        frame = A.read_runs(ctx.workdir / "runs.csv")
        raw = pd.read_csv(ctx.workdir / "runs.csv")
        if raw["model_version"].dtype != object:
            ctx.limit(
                "Analyzer.run rejects a reference read back from CSV: model_version '1.0' becomes the float 1.0 and the "
                "metadata filter (reference.model_version == '1.0') keeps no row",
                f"pd.read_csv gives model_version dtype {raw['model_version'].dtype}; error 'Wrong model metadata for "
                "reference dataset.' on the LS-01 re-analysis",
                "Compare model_id/model_version as strings in Analyzer.run (astype(str)) and say in the error how many "
                "rows each filter kept", "minor")
        previous = json.loads((ctx.workdir / "report.json").read_text(encoding="utf-8"))
        inj = "injection" in frame
        first = json.loads((ctx.workdir.parent / "LS-01.pass1.json").read_text(encoding="utf-8")) \
            if (ctx.workdir.parent / "LS-01.pass1.json").exists() else {"metrics": {}}
        runs = None
        ref_seconds = previous.get("reference_seconds", first["metrics"].get("reference_seconds", float("nan")))
        ctx.note("re-analysis of the collected runs (LAB_REANALYSE=1): agent runs and MMD p-values reused")
    else:
        inj, faith = detectors(ctx)
        pii = {}

        def on_finish(span, request, result):
            with PiiSession() as session:
                session.scan("input", request)
                leaks = session.scan("output", result.answer or "").leaks
            scores = {"pii_leaks": len(leaks)}
            if inj is not None:
                scores["injection"] = inj.score(request)
            if faith is not None and result.context and result.answer:
                scores["faithfulness"] = faith.score(result.answer, result.context)
            pii[format(span.get_span_context().trace_id, "032x")] = scores

        t0 = time.perf_counter()
        ref_runs = A.run_block("reference", p["reference_runs"], seed=1000, workers=p["workers"], on_finish=on_finish,
                               log=log)
        ref_seconds = time.perf_counter() - t0
        win_runs = []
        for k in range(p["windows"]):
            win_runs += A.run_block(f"w{k + 1:02d}", p["window"], seed=2000 + k, workers=p["workers"],
                                    on_finish=on_finish, extra={"window": k + 1}, log=log)
        runs = ref_runs + win_runs
        frame = A.records_for(runs)
        scores = pd.DataFrame([{**pii.get(t, {}), "trace_id": t} for t in frame["trace_id"]])
        frame = pd.concat([frame.reset_index(drop=True), scores.drop(columns="trace_id")], axis=1)
        A.save_frame(frame, ctx.workdir / "runs.csv")
        # reference reusable by LA-03 (same agent, same conditions)
        A.save_frame(frame[frame["label"] == "reference"], ctx.workdir / "reference_runs.csv")
        inj = inj is not None

    def as_num(series):
        return pd.to_numeric(series.replace({"True": 1, "False": 0, True: 1, False: 0}), errors="coerce").fillna(0)

    # ---------------------------------------------------------------- exporter integrity
    ok = frame["error"].isna()
    matched = frame.loc[ok, "matched"]
    agree = frame.loc[ok & frame["matched"]].apply(lambda r: r["agent_tools"] == r["tool_sequence"], axis=1)
    record_match_rate = float((matched & agree.reindex(matched.index, fill_value=False)).mean()) if len(matched) else 0.0
    failed_runs = int((~ok).sum())
    if record_match_rate < 0.999:
        bad = frame.loc[ok & (~frame["matched"] | (frame["agent_tools"] != frame["tool_sequence"]))]
        slow = bad["wall_s"].describe().to_dict() if len(bad) else {}
        ctx.limit(
            f"{len(bad)} of {int(ok.sum())} agent runs have no agent_run record or a record whose tool sequence differs "
            "from what the agent did",
            f"unmatched={int((~frame.loc[ok, 'matched']).sum())}, sequence mismatches={int((~agree).sum())}; wall time of "
            f"the affected runs {json.dumps({k: round(v, 1) for k, v in slow.items()})} s; trace_timeout_s=30 default",
            "Assemble a trace only when its root span arrives, or make the idle timeout relative to the longest open LLM "
            "call; count partial assemblies in exporter stats so the loss is visible",
            "major",
        )

    split = exporter_idle_split()
    if split["tool_calls"] != [1]:
        dry = ctx.workdir.parent / "LS-01.dry.json"
        seen = ""
        if dry.exists():
            d = json.loads(dry.read_text(encoding="utf-8"))
            seen = next((l["finding"] + " (" + l["evidence"][:200] + ")" for l in d.get("limits", [])
                         if "tool sequence differs" in l["finding"]), "")
        ctx.limit(
            "A run longer than trace_timeout_s (30 s) loses its LLM and tool calls: the child spans exported early are "
            "assembled as an incomplete trace by flush_idle and dropped, and the late root span becomes an agent_run "
            f"with tool_calls={split['tool_calls']}, steps={split['steps']}, tool_sequence={split['tool_sequence']} "
            "instead of 1 / 1 / check_eligibility",
            f"deterministic replay through TrajectoryAssembler (tool span ends at 5 s, another export at 35.5 s, root "
            f"ends at 45 s): {json.dumps(split)}; observed live in the LS-01 dry run: {seen or 'n/a'}; in the full run "
            f"(max_tokens cap, p50 run {frame['wall_s'].median():.1f} s) record_match_rate={record_match_rate:.4f}",
            "In TrajectoryAssembler, never assemble a trace whose spans all have a parent that is still unknown; keep "
            "the partial spans until the root arrives (bounded by max_pending_traces), or make the idle timeout "
            "start from the newest span start, not its end; count dropped partial traces in the exporter stats",
            "major",
        )

    good = ok & frame["matched"]
    # ---------------------------------------------------------------- truncated / runaway generations
    truncated = frame.loc[good & (frame["llm_length_stops"] > 0)]
    if len(truncated):
        terms = truncated["termination"].astype(str).value_counts().to_dict()
        ctx.limit(
            f"{len(truncated)} of {int(good.sum())} runs contain an LLM call cut at max_tokens={A.MAX_TOKENS} "
            "(finish_reason=length), yet their agent_run record carries no truncation field",
            f"termination of those runs: {terms}; task_success: "
            f"{truncated['task_success'].astype(str).value_counts().to_dict()}; the finish reason only exists on "
            "llm_call records",
            "Add truncated_calls (finish_reason=length count) to agent_run and a termination value 'truncated', so "
            "drift and absolute alerts can see degenerate generations",
            "minor",
        )
    attempt = ctx.workdir / "attempt1" / "runaway_generation_journal.txt"
    if attempt.exists():
        lines = attempt.read_text(encoding="utf-8").strip().splitlines()
        last = next((l for l in reversed(lines) if "n_gen" in l), "")
        ctx.limit(
            "A run that never ends is invisible: in the first attempt one llama3.2:3b call (temperature 0, no output "
            "cap, as in the demo agent) generated ~40k tokens for 9 min (04:49:40-04:58:40 UTC) with context shifts; "
            "no agent_run record exists until the root span ends, so Pulsar showed nothing while the endpoint "
            "(Ollama -np 1) was blocked for every caller",
            "ollama journal, task 125331: " + last.split("ollama[1730]:")[-1].strip()[:160] +
            f"; journal excerpt in {attempt.relative_to(ctx.workdir.parent)}; the client default (600 s x 3 attempts) "
            "would have repeated the same deterministic runaway",
            "Emit an 'agent_run_open' heartbeat (or export the root span start) so runs older than a threshold raise a "
            "stuck-run alert; document max_tokens and a client timeout without retry for instrumented agents; add "
            "the demo agent's max_tokens",
            "major",
        )

    ref = frame[(frame["label"] == "reference") & good]
    # a re-analysis may read runs collected with fewer windows than PARAMS
    n_win = min(p["windows"], int(pd.to_numeric(frame.get("window"), errors="coerce").max() or 0))
    windows = [frame[(frame.get("window") == k + 1) & good] for k in range(n_win)]
    min_ref = MIN_REFERENCE_WINDOWS * p["window"]
    if len(ref) < min_ref and not DRY:
        ctx.note(f"reference of {len(ref)} good runs < {min_ref} ({MIN_REFERENCE_WINDOWS} disjoint windows of "
                 f"{p['window']}): the calibrator's A/A draws overlap; drift false-alarm rates are not conclusive")

    # ---------------------------------------------------------------- calibrated drift
    tester = A.WindowTester(ref, window=p["window"], n_draws=p["n_draws"], alpha=p["alpha"])
    if tester.calibration_warnings:
        ctx.note("calibration warnings: " + " | ".join(tester.calibration_warnings))
    tracker = DriftPersistence(windows=p["persistence"])
    per_window, flags_by_feature, nominal_by_feature, confirmed_any = [], {}, {}, []
    for k, w in enumerate(windows, start=1):
        d = tester.test(w)
        confirmed = sorted(v.feature_name for v in tracker.update(d) if v.confirmed_drift)
        confirmed_any.append(bool(confirmed))
        for r in d.itertuples():
            flags_by_feature.setdefault(r.feature_name, []).append(bool(r.flag))
            nominal_by_feature.setdefault(r.feature_name, []).append(bool(r.nominal_flag))
        per_window.append({"window": k, "runs": len(w), "flagged": sorted(d.loc[d["flag"], "feature_name"]),
                           "nominal_flagged": sorted(d.loc[d["nominal_flag"], "feature_name"]), "confirmed": confirmed,
                           "min_p_adjusted": float(np.nanmin(d["p_adjusted"]))})
    tests = {}
    for f, v in flags_by_feature.items():
        tests[f"drift_bh|{f}"] = v
        tests[f"drift_calibrated_no_bh|{f}"] = nominal_by_feature[f]
    tests["drift_bh|any_feature"] = [bool(r["flagged"]) for r in per_window]
    tests["drift_confirmed|any_feature"] = confirmed_any

    # ---------------------------------------------------------------- MMD on embeddings
    if reanalyse:
        for name in ("request", "answer"):
            ps = [w.get(f"mmd_{name}_p") for w in previous.get("per_window", [])]
            if ps and all(x is not None for x in ps):
                tests[f"mmd|{name}"] = [x < p["alpha"] for x in ps]
                for k, x in enumerate(ps):
                    per_window[k][f"mmd_{name}_p"] = x
    texts = {} if reanalyse else {"request": [r.get("_request") for r in runs], "answer": [r.get("_answer") for r in runs]}
    labels = [] if reanalyse else [r["label"] for r in runs]
    for name, values in texts.items():
        try:
            emb = A.embed(values)
            ref_e = emb[[i for i, l in enumerate(labels) if l == "reference"]]
            res = []
            for k in range(n_win):
                cur = emb[[i for i, l in enumerate(labels) if l == f"w{k + 1:02d}"]]
                r = embedding_drift(ref_e, cur, feature_name=name, n_permutations=p["mmd_permutations"], seed=k)
                res.append(bool(r.drift_status))
                per_window[k][f"mmd_{name}_p"] = r.p_value
            tests[f"mmd|{name}"] = res
        except Exception as e:  # noqa: BLE001
            ctx.note(f"MMD on {name} failed: {type(e).__name__}: {str(e)[:200]}")

    # ---------------------------------------------------------------- compliance and absolute detectors
    monitor = ComplianceMonitor(load_rules(str(A.RULES)), version="permit-agent-rules/1")
    ref_traces = [trace_from(s) for s in ref["tool_sequence"].astype(str)]
    ref_compliance = {r.feature_name: r.metric_value for r in monitor.evaluate(ref_traces)}
    for rule in monitor.rules:
        tests[f"ltlf|{rule.id}"] = []
    for k, w in enumerate(windows):
        for r in monitor.evaluate([trace_from(s) for s in w["tool_sequence"].astype(str)]):
            tests[f"ltlf|{r.feature_name}"].append(bool(r.drift_status))
    for name, col, thr in (("loop_detected", "loop_detected", 0.5), ("out_of_scope", "out_of_scope_calls", 0.5),
                           ("pii_leak", "pii_leaks", 0.5), ("injection>0.5", "injection", 0.5)):
        if col in frame:
            tests[f"absolute|{name}"] = [bool((as_num(w[col]) > thr).any()) for w in windows]
    if "faithfulness" in frame:
        tests["absolute|faithfulness<0.5"] = [bool((pd.to_numeric(w["faithfulness"], errors="coerce") < 0.5).any())
                                              for w in windows]

    # ---------------------------------------------------------------- verdicts
    summary = {}
    for name, v in tests.items():
        k, n = int(sum(v)), len(v)
        summary[name] = {"alarms": k, "windows": n, "rate": k / n if n else None, "ci95": wilson(k, n)}
    # An excess is established only when it survives BH across every test of the soak: one-sided binomial p-value of
    # the alarm count against alpha per test, Benjamini-Hochberg over the tests. The calibrated per-test flags without
    # BH (drift_calibrated_no_bh) are reported but never make a limit on their own: they are the uncorrected version
    # of drift_bh (refuted at the night's run: 1 test of 36 without correction, 2/10, Wilson low 5.7 %).
    excess_p = bh_excess(summary, p["alpha"])
    for t, (pv, q) in excess_p.items():
        summary[t]["p_excess"], summary[t]["q_bh"] = pv, q
    over = {t: s for t, s in summary.items() if s.get("q_bh") is not None and s["q_bh"] <= p["alpha"]
            and not t.startswith("drift_calibrated_no_bh|")}
    noted = {t: s for t, s in summary.items() if s["rate"] is not None and s["rate"] > p["alpha"] and t not in over}
    if noted:
        ctx.note("alarm rate above 5 % but compatible with 5 % (Wilson lower bound <= 5 %): " + "; ".join(
            f"{t} {s['alarms']}/{s['windows']} CI {s['ci95']}" for t, s in noted.items()))
    per_run = {}
    for name, col in (("loop_detected", "loop_detected"), ("out_of_scope_calls>0", "out_of_scope_calls"),
                      ("pii_leaks>0", "pii_leaks")):
        if col in frame:
            vals = as_num(frame.loc[good, col]) > 0
            per_run[name] = {"runs": int(vals.sum()), "of": int(len(vals))}

    def viol(seqs, rule):
        seqs = list(seqs)
        return float(np.mean([not rule.check(trace_from(x)) for x in seqs])) if seqs else float("nan")

    rule_rate = {r.id: viol(frame.loc[good, "tool_sequence"].astype(str), r) for r in monitor.rules}
    window_rule_rate = {r.id: [round(viol(w["tool_sequence"].astype(str), r), 3) for w in windows] for r in monitor.rules}

    for t, s in over.items():
        family, feature = t.split("|", 1)
        head = (f"{t}: alerts in {s['alarms']}/{s['windows']} unchanged windows of {p['window']} runs "
                f"(Wilson 95 % CI {s['ci95']})")
        if family == "ltlf":
            ctx.limit(
                head + f"; these are real violations at a steady rate ({rule_rate[feature]:.1%} of runs; reference "
                f"{1 - ref_compliance[feature]:.1%}), but ComplianceMonitor has no baseline: with min_compliance=1.0 a "
                "steady state alerts in every window and a change of rate cannot be told from it",
                f"per-window violation rate {window_rule_rate[feature]}; reference compliance "
                f"{ref_compliance[feature]:.3f} over {len(ref)} runs",
                "Add a reference-aware mode to ComplianceMonitor.evaluate (one-sided two-proportion / Fisher test of the "
                "window violation rate vs the reference, BH over rules) and keep the absolute per-run alert for "
                "critical rules only; report the steady violation rate instead of re-alerting",
                "major",
            )
        elif family == "absolute" and feature.startswith("injection"):
            g = frame.loc[good].assign(hit=as_num(frame.loc[good, "injection"]) > 0.5)
            by = g.groupby(["language", "kind"])["hit"].agg(["sum", "count"])
            by = {f"{a}/{b}": f"{int(r['sum'])}/{int(r['count'])}" for (a, b), r in by.iterrows() if r["sum"] > 0}
            ctx.limit(
                head + f"; InjectionDetector scores > 0.5 on {int(g['hit'].sum())} of {len(g)} benign synthetic "
                "requests (the task generator has no injection)",
                f"flagged requests by language/kind: {by}; model protectai/deberta-v3-base-prompt-injection-v2, "
                "threshold 0.5",
                "Calibrate the injection threshold per language on an A/A reference of benign traffic (or alert on "
                "the drift of the score rather than an absolute 0.5); ship a French benign calibration set",
                "major",
            )
        elif family.startswith("drift"):
            mix = pd.crosstab(frame.loc[good, "label"], frame.loc[good, "kind"], normalize="index").round(3)
            flagged = [f"w{k + 1:02d}" for k, v in enumerate(tests[t]) if v]
            ctx.limit(
                head + f" after BH over the features and BH across the {len(summary)} tests of the soak (q "
                f"{s['q_bh']:.3g}); confirmed over 2 windows {summary['drift_confirmed|any_feature']['alarms']}/"
                f"{summary['drift_confirmed|any_feature']['windows']}",
                f"flagged windows {flagged}: request mix {mix.loc[flagged].to_dict('index')} vs reference "
                f"{mix.loc['reference'].to_dict()}; reference {len(ref)} runs; "
                f"calibration warning: {tester.calibration_warnings}",
                "When the request kind is logged, test first_tool conditionally on it (stratified chi-square) so mix "
                "noise is not read as behaviour drift; alert on BH + persistence",
                "minor",
            )
        else:
            ctx.limit(head + " exceeds 5 %", f"per-window detail in report.json; per-run absolute counts {per_run}",
                      "Calibrate this test on the reference (A/A) before alerting", "major")
    if not inj:
        ctx.note("InjectionDetector not soaked (torch unavailable)")
    if "faithfulness" not in frame or frame["faithfulness"].notna().sum() == 0:
        ctx.note("FaithfulnessScorer not soaked: the default task mix has no 'rules' request, so no run retrieved "
                 "passages (it is measured in LR-01/LR-02)")

    report = {"summary": summary, "per_window": per_window, "per_run_absolute": per_run, "rule_violation_rate": rule_rate,
              "window_rule_violation_rate": window_rule_rate, "reference_seconds": ref_seconds,
              "reference_compliance": ref_compliance, "calibration_warnings": tester.calibration_warnings}
    (ctx.workdir / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    rates = [s["rate"] for s in summary.values() if s["rate"] is not None]
    wall = frame["wall_s"].describe()
    return {
        "runs": len(frame), "failed_runs": failed_runs, "record_match_rate": record_match_rate,
        "reference_runs": int(len(ref)), "reference_seconds": round(ref_seconds, 1),
        "run_wall_s_p50": float(wall["50%"]), "run_wall_s_max": float(wall["max"]),
        "tests": len(summary), "tests_excess_established": sorted(over),
        "tests_above_5pct_not_established": sorted(noted), "max_false_alarm_rate": max(rates) if rates else None,
        "max_false_alarm_rate_drift_mmd": max(v["rate"] for t, v in summary.items() if t.startswith(("drift", "mmd"))),
        "any_feature_bh_rate": summary["drift_bh|any_feature"]["rate"],
        "confirmed_rate": summary["drift_confirmed|any_feature"]["rate"],
        "detectors_torch": bool(inj), "runs_with_length_stop": int(len(truncated)),
        "replayed_slow_run_tool_calls": split["tool_calls"],
        "failed_run_errors": frame.loc[~ok, "error"].astype(str).str[:80].value_counts().to_dict(),
    }
