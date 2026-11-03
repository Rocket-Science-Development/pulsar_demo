"""LA-01 — Function calling quality and trajectories (BFCL simple / multiple / parallel).

Dataset: gorilla-llm/Berkeley-Function-Calling-Leaderboard v3 (Apache 2.0), categories simple (400), multiple (200),
parallel (200), with their possible answers. Each case is one agent run of a local model through Ollama, traced with
the real Pulsar instrumentation (agent_span / llm_span / tool_span -> PulsarSpanExporter -> agent_run records). The
model may call tools for up to 3 steps; tools return a fixed stub result. Accuracy is the BFCL AST check of the calls
of the first step (a simplified re-implementation: name, no unexpected parameter, required parameters present, value
in the accepted list after BFCL string normalisation; int accepted for float). Rules (ComplianceMonitor / Rule):
"required tool called" (existence of every expected function) and "offered tools only" (allowed_tools of the case).
Drift llama3.2:3b -> llama3.1:8b: calibrated KS/chi2 windows (lab WindowTester), with an A/A control on the same
model; cases split in half A (reference) and half B (windows), stratified by category, so both models answer the
same B cases.

No real person in BFCL: synthetic questions. Only features, tool names and verdicts are written.
"""

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from lab.tests._gpulab import BASE_URL, DATA, DRY, Collector, wilson

BFCL = DATA / "gorilla-llm" / "bfcl"
CATEGORIES = {"simple": 200, "multiple": 200, "parallel": 200}
MODELS = ["llama3.2:3b", "llama3.1:8b"]
PARAMS = {"models": MODELS, "per_category": {k: (12 if DRY else v) for k, v in CATEGORIES.items()},
          "max_steps": 3, "max_tokens": 512, "window": 4 if DRY else 30, "workers": 2, "temperature": 0.0,
          "dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard v3 (Apache 2.0)", "dry": DRY}
CRITERIA = {"models_scored": (">=", 2), "malformed_records": ("==", 0), "aa_false_alarm_windows_share": ("<=", 0.2),
            "drift_windows_share_trajectory": (">=", 0.8)}

TRAJ_NUMERIC = ["steps", "tool_calls", "distinct_tools", "tool_error_rate", "repeated_calls", "output_tokens"]
TRAJ_CATEGORICAL = ["termination", "task_success"]
ALL_NUMERIC = TRAJ_NUMERIC + ["input_tokens", "latency_s"]
REQUIRED_FIELDS = ["steps", "tool_calls", "distinct_tools", "tool_sequence", "termination", "input_tokens",
                   "output_tokens", "latency_s", "task_success", "trace_id", "request_model"]


# ------------------------------------------------------------------------------------------------ data
def load(category, n):
    cases = [json.loads(x) for x in (BFCL / f"BFCL_v3_{category}.json").open(encoding="utf-8")]
    answers = {a["id"]: a["ground_truth"] for a in
               (json.loads(x) for x in (BFCL / "possible_answer" / f"BFCL_v3_{category}.json").open(encoding="utf-8"))}
    cases = cases[:n]
    for c in cases:
        c["category"], c["truth"] = category, answers[c["id"]]
    return cases


TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def to_schema(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = TYPES.get(v, v) if k == "type" and isinstance(v, str) else to_schema(v)
        return out
    if isinstance(obj, list):
        return [to_schema(x) for x in obj]
    return obj


def safe(name):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def tools_of(case):
    return [{"type": "function", "function": {"name": safe(f["name"]), "description": f.get("description", ""),
                                              "parameters": to_schema(f["parameters"])}} for f in case["function"]]


# ------------------------------------------------------------------------------------------------ AST check
def _norm(s):
    return re.sub(r"[ ,./\-_*^]", "", str(s)).lower()


def _match(v, a):
    if isinstance(a, bool) or isinstance(v, bool):
        return isinstance(a, bool) and isinstance(v, bool) and a == v
    if isinstance(a, (int, float)) and isinstance(v, (int, float)):
        return float(v) == float(a)
    if isinstance(a, str) and isinstance(v, str):
        return _norm(v) == _norm(a)
    if isinstance(a, list) and isinstance(v, list):
        return len(a) == len(v) and all(_match(x, y) for x, y in zip(v, a))
    if isinstance(a, dict) and isinstance(v, dict):
        if set(v) - set(a):
            return False
        for k, acc in a.items():
            acc = acc if isinstance(acc, list) else [acc]
            if k not in v:
                if "" not in acc:
                    return False
            elif not any(_match(v[k], x) for x in acc if x != ""):
                return False
        return True
    return False


def call_ok(name, args, truth_call):
    (gt_name, gt_params), = truth_call.items()
    if name != gt_name or not isinstance(args, dict):
        return False
    if set(args) - set(gt_params):
        return False
    for k, acc in gt_params.items():
        if k not in args:
            if "" not in acc:
                return False
        elif not any(_match(args[k], x) for x in acc if x != ""):
            return False
    return True


def coerce(calls, case):
    """Lenient variant: numeric strings turned into numbers where the schema says integer/float (BFCL is strict)"""
    props = {f["name"]: f["parameters"].get("properties", {}) for f in case["function"]}
    out, n = [], 0
    for name, args in calls:
        if isinstance(args, dict):
            args = dict(args)
            for k, v in args.items():
                t = props.get(name, {}).get(k, {}).get("type")
                if t in ("integer", "float", "number") and isinstance(v, str):
                    try:
                        args[k] = float(v) if t != "integer" else int(float(v))
                        n += 1
                    except ValueError:
                        pass
        out.append((name, args))
    return out, n


def ast_correct(calls, truth):
    """calls: [(name, args or None)]; parallel cases are matched without order"""
    if len(calls) != len(truth):
        return False
    left = list(range(len(calls)))
    for t in truth:
        hit = next((i for i in left if calls[i][1] is not None and call_ok(calls[i][0], calls[i][1], t)), None)
        if hit is None:
            return False
        left.remove(hit)
    return True


# ------------------------------------------------------------------------------------------------ runs
def run_model(model, cases, p, col, log):
    from openai import OpenAI
    from pulsar_data_collection.genai import agent_span, llm_span, record_llm_response, tool_span
    from pulsar_data_collection.genai import semconv as sc

    client = OpenAI(base_url=BASE_URL, api_key="local", max_retries=0, timeout=300)
    out = [None] * len(cases)
    done = [0]

    def one(i):
        case = cases[i]
        tools = tools_of(case)
        back = {safe(f["name"]): f["name"] for f in case["function"]}
        messages = [{"role": "user", "content": case["question"][0][0]["content"]}]
        row = {"model": model, "id": case["id"], "category": case["category"], "half": case["half"],
               "first_calls": None, "malformed_args": 0, "error": None}
        t0 = time.perf_counter()
        try:
            with agent_span(col.tracer, "bfcl_agent", prompt_version="bfcl-v3",
                            available_tools=[f["name"] for f in case["function"]]) as run:
                termination, tools_called = "max_steps", []
                for step in range(p["max_steps"]):
                    with llm_span(col.tracer, model=model, provider="ollama", temperature=p["temperature"]) as call:
                        r = client.chat.completions.create(model=model, messages=messages, tools=tools,
                                                           temperature=p["temperature"], max_tokens=p["max_tokens"])
                        ch = r.choices[0]
                        record_llm_response(call, response_model=r.model,
                                            input_tokens=getattr(r.usage, "prompt_tokens", None),
                                            output_tokens=getattr(r.usage, "completion_tokens", None),
                                            finish_reason=ch.finish_reason)
                    msg = ch.message
                    if not msg.tool_calls:
                        termination = "completed"
                        if step == 0:
                            row["first_calls"] = []
                            row["text_has_call"] = bool(re.search(r'"name"\s*:|\w+\(.*=.*\)', msg.content or ""))
                        break
                    messages.append(msg.model_dump(exclude_none=True))
                    calls = []
                    for tc in msg.tool_calls:
                        name = back.get(tc.function.name, tc.function.name)
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            args = None
                            row["malformed_args"] += 1
                        calls.append((name, args))
                        tools_called.append(name)
                        try:
                            with tool_span(col.tracer, name, arguments=tc.function.arguments or "{}", call_id=tc.id):
                                if name not in back.values():
                                    raise LookupError(f"unknown tool {name}")
                                if args is None:
                                    raise ValueError("arguments are not JSON")
                                result = {"result": "ok", "status": "done"}
                        except Exception as e:  # noqa: BLE001
                            result = {"error": type(e).__name__}
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
                    if step == 0:
                        row["first_calls"] = calls
                correct = ast_correct(row["first_calls"] or [], case["truth"])
                lenient, row["string_numbers"] = coerce(row["first_calls"] or [], case)
                row["ast_lenient"] = ast_correct(lenient, case["truth"])
                run.set_attribute(sc.TERMINATION, termination)
                run.set_attribute(sc.TASK_SUCCESS, bool(correct))
                row.update(trace_id=format(run.get_span_context().trace_id, "032x"), ast_correct=correct,
                           agent_tools=">".join(tools_called) or "none", agent_steps=step + 1,
                           agent_termination=termination)
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"[:300]
        row["wall_s"] = time.perf_counter() - t0
        out[i] = row
        with col.lock:
            done[0] += 1
            if done[0] % 50 == 0:
                log(f"{model}: {done[0]}/{len(cases)}")

    with ThreadPoolExecutor(max_workers=p["workers"]) as pool:
        list(pool.map(one, range(len(cases))))
    return out


def malformed(rec, run):
    """A record is malformed when a required field is missing/null or when it disagrees with what the agent did"""
    problems = [f for f in REQUIRED_FIELDS if rec.get(f) is None]
    if rec.get("tool_sequence") != run.get("agent_tools"):
        problems.append("tool_sequence")
    if rec.get("steps") != run.get("agent_steps"):
        problems.append("steps")
    if rec.get("termination") != run.get("agent_termination"):
        problems.append("termination")
    if rec.get("task_success") is not None and bool(rec["task_success"]) != bool(run.get("ast_correct")):
        problems.append("task_success")
    return problems


# ------------------------------------------------------------------------------------------------ test
def run(ctx):
    from scipy import stats
    from pulsar_metrics.genai.compliance import Rule, trace_from

    p = ctx.params
    log = lambda m: print(f"[LA-01 {time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    cases = []
    for cat, n in p["per_category"].items():
        part = load(cat, n)
        rng = random.Random({"simple": 1, "multiple": 2, "parallel": 3}[cat])
        idx = list(range(len(part)))
        rng.shuffle(idx)
        for j, k in enumerate(idx):
            part[k]["half"] = "A" if j % 2 == 0 else "B"
        cases += part
    random.Random(5).shuffle(cases)  # interleave categories so every window mixes them
    ctx.note(f"{len(cases)} BFCL cases: {dict(pd.Series([c['category'] for c in cases]).value_counts())}")

    col = Collector("bfcl-lab")  # allowed_tools is global to an exporter: per-case catalogues go in available_tools
    runs = []
    for model in p["models"]:
        t = time.perf_counter()
        runs += run_model(model, cases, p, col, log)
        log(f"{model}: {len(cases)} cases in {time.perf_counter() - t:.0f} s")
    records = col.records()
    by_trace = {r["trace_id"]: r for r in records}
    truth = {c["id"]: c for c in cases}

    # ---------------------------------------------------------------- records, malformed, rules
    rows, bad, field_keys = [], [], set()
    for r in records:
        field_keys |= set(r)
    for run_ in runs:
        rec = by_trace.get(run_.get("trace_id"))
        if run_.get("error"):
            rows.append({**run_, "matched": False})
            continue
        if rec is None:
            bad.append({"id": run_["id"], "model": run_["model"], "problems": ["no agent_run record"]})
            rows.append({**run_, "matched": False})
            continue
        prob = malformed(rec, run_)
        if prob:
            bad.append({"id": run_["id"], "model": run_["model"], "problems": prob})
        case = truth[run_["id"]]
        trace_ = trace_from(rec["tool_sequence"])
        required = sorted({list(t)[0] for t in case["truth"]})
        req_rule = Rule(id="required-tool", formula=" & ".join(f"F {fn}" for fn in required), severity="major")
        offered = Rule(id="offered-tools", template="allowed_tools", tools=[f["name"] for f in case["function"]])
        rows.append({**{k: v for k, v in run_.items() if k != "first_calls"},
                     "first_calls_json": json.dumps(run_.get("first_calls"), default=str)[:400], **{k: rec.get(k) for k in
                     ALL_NUMERIC + TRAJ_CATEGORICAL + ["first_tool", "tool_sequence", "tool_catalog_size", "unavailable_tool_calls"]},
                     "matched": True, "required_called": req_rule.check(trace_), "offered_only": offered.check(trace_),
                     "n_calls_first": len(run_["first_calls"] or []), "n_truth": len(case["truth"])})
    frame = pd.DataFrame(rows)
    frame.to_csv(ctx.workdir / "runs.csv", index=False)
    errors = int(frame["error"].notna().sum())

    per_model = {}
    for model, g in frame[frame["matched"]].groupby("model"):
        m = {"runs": int(len(g)), "errors": int(frame[(frame.model == model) & frame.error.notna()].shape[0])}
        for cat, gc in g.groupby("category"):
            k = int(gc["ast_correct"].sum())
            m[f"ast_{cat}"] = {"acc": round(k / len(gc), 4), "ci": wilson(k, len(gc)), "n": int(len(gc))}
        k = int(g["ast_lenient"].sum())
        m["ast_lenient_overall"] = {"acc": round(k / len(g), 4), "ci": wilson(k, len(g)), "n": int(len(g))}
        m["string_numbers"] = int(g["string_numbers"].sum())
        k = int(g["ast_correct"].sum())
        m["ast_overall"] = {"acc": round(k / len(g), 4), "ci": wilson(k, len(g)), "n": int(len(g))}
        for rule in ("required_called", "offered_only"):
            k = int(g[rule].sum())
            m[rule] = {"rate": round(k / len(g), 4), "ci": wilson(k, len(g))}
        m["no_call_first_step"] = int((g["n_calls_first"] == 0).sum())
        m["no_call_but_call_in_text"] = int(g.get("text_has_call", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        m["malformed_args"] = int(g["malformed_args"].sum())
        m["parallel_wrong_count"] = int(((g.category == "parallel") & (g.n_calls_first != g.n_truth)).sum())
        m["unavailable_tool_calls"] = int(g["unavailable_tool_calls"].fillna(0).sum())
        m["mean_steps"] = round(float(g["steps"].mean()), 3)
        m["mean_tool_calls"] = round(float(g["tool_calls"].mean()), 3)
        per_model[model] = m

    (ctx.workdir / "pre_drift.json").write_text(json.dumps({"malformed": bad, "field_keys": sorted(field_keys),
                                                           "records": len(records)}, default=str), encoding="utf-8")

    # ---------------------------------------------------------------- drift between models (calibrated windows)
    from lab.tests._agentlab import MODEL_ID, MODEL_VERSION, WindowTester

    drift = {}
    ok = frame[frame["matched"]].copy()
    ok["model_id"], ok["model_version"] = MODEL_ID, MODEL_VERSION  # the lab WindowTester's analyzer metadata
    ok["pred_timestamp"] = pd.Timestamp("2026-11-02") + pd.to_timedelta(np.arange(len(ok)), unit="s")
    ok["first_tool"] = ok["first_tool"].astype(str)
    for c in ("steps", "tool_calls", "distinct_tools", "tool_error_rate", "repeated_calls", "out_of_scope_calls",
              "input_tokens", "output_tokens", "latency_s", "unavailable_tool_calls"):
        if c not in ok:
            ok[c] = 0.0
    m0, m1 = p["models"][0], p["models"][1]
    ref = ok[(ok.model == m0) & (ok.half == "A")]
    w = p["window"]
    for label, numeric, categorical in (("trajectory", TRAJ_NUMERIC, TRAJ_CATEGORICAL),
                                        ("unlabeled", TRAJ_NUMERIC, ["termination"]),
                                        ("all", ALL_NUMERIC, TRAJ_CATEGORICAL)):
        try:
            tester = WindowTester(ref, window=w, numeric=numeric, categorical=categorical,
                                  allow_short=len(ref) < 10 * w)
        except Exception as e:  # noqa: BLE001
            drift[label] = {"error": f"{type(e).__name__}: {e}"}
            continue
        res = {}
        for name, model in (("aa", m0), ("model_change", m1)):
            cur = ok[(ok.model == model) & (ok.half == "B")].reset_index(drop=True)
            flags, feats = [], {}
            for s in range(0, len(cur) - w + 1, w):
                try:
                    d = tester.test(cur.iloc[s:s + w])
                except Exception as e:  # noqa: BLE001
                    res.setdefault("errors", []).append(f"{type(e).__name__}: {e}"[:300])
                    continue
                flags.append(bool(d["flag"].any()))
                for f in d.loc[d["flag"], "feature_name"]:
                    feats[f] = feats.get(f, 0) + 1
            res[name] = {"windows": len(flags), "flagged": int(sum(flags)),
                         "share": round(sum(flags) / len(flags), 3) if flags else None, "features_flagged": feats}
        res["calibration_warnings"] = tester.calibration_warnings
        drift[label] = res

    # paired comparison of the rule "required tool called" and of AST accuracy on the same cases
    paired = {}
    piv = ok.pivot_table(index="id", columns="model", values=["required_called", "ast_correct"], aggfunc="first")
    for v in ("required_called", "ast_correct"):
        if (v, m0) in piv and (v, m1) in piv:
            a, b = piv[(v, m0)].astype(bool), piv[(v, m1)].astype(bool)
            n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
            pv = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            paired[v] = {m0: round(float(a.mean()), 4), m1: round(float(b.mean()), 4), "only_8b": n01,
                         "only_3b": n10, "mcnemar_exact_p": float(pv)}

    # ---------------------------------------------------------------- limits
    per_tool_keys = sorted(k for k in field_keys if re.match(r"tool_.+_(calls|errors|timeouts|soft_failures)$", k))
    if len(per_tool_keys) > 200:
        ctx.limit(f"agent_run records of {len(cases)} BFCL cases carry {len(field_keys)} distinct field keys, "
                  f"{len(per_tool_keys)} of them per-tool counters (tool_<name>_calls/errors/timeouts/soft_failures "
                  "for every tool of each run's catalogue): the schema grows with the tool catalogue",
                  f"{len(records)} records; distinct keys {len(field_keys)}; per-tool keys {len(per_tool_keys)}; "
                  f"distinct tools offered {len({f['name'] for c in cases for f in c['function']})}; examples "
                  f"{per_tool_keys[:6]}",
                  "Make the per-tool counters opt-in or bounded (top-k tools, or only allowed_tools), and write the "
                  "per-tool detail as tool_call records (tag = tool) instead of agent_run fields, so an agent with a "
                  "large or per-request catalogue does not create unbounded InfluxDB fields", "major")
    ctx.note("allowed_tools is fixed per exporter: with a per-request catalogue (BFCL) out_of_scope_calls is always 0; "
             "the catalogue sent per run (available_tools) gives unavailable_tool_calls instead. The rules "
             "'required tool called' (per case) and 'offered tools only' are checked offline: ComplianceMonitor rules "
             "are global to an exporter and cannot depend on the request")
    bad_n = len(bad)
    if bad_n:
        ctx.limit(f"{bad_n} of {len(records)} agent_run records are malformed or disagree with the agent's own view",
                  json.dumps(bad[:20]), "Fix the assembly of the fields listed (see problems)", "critical")
    for model, m in per_model.items():
        if m["no_call_but_call_in_text"]:
            ctx.note(f"{model}: {m['no_call_but_call_in_text']} first steps with no parsed tool call but a call-like "
                     "text (the model wrote the call as text; Ollama did not parse it) — counted as no call")
    tr = drift.get("trajectory", {})
    if isinstance(tr.get("aa"), dict) and tr["aa"]["share"] is not None and tr["aa"]["share"] > 0.2:
        ctx.limit(f"A/A false alarms: {tr['aa']['flagged']} of {tr['aa']['windows']} windows of {w} runs of the same "
                  f"model ({m0}) flag drift on trajectory features",
                  json.dumps(tr["aa"]), "Check the calibration of the window tester on heterogeneous task mixes",
                  "major")
    if isinstance(tr.get("model_change"), dict) and tr["model_change"]["share"] is not None and \
            tr["model_change"]["share"] < 0.8:
        ctx.limit(f"Model change {m0} -> {m1} flagged in only {tr['model_change']['flagged']} of "
                  f"{tr['model_change']['windows']} windows of {w} runs on trajectory features",
                  json.dumps(tr), "Report the minimum detectable effect for the window size; add trajectory-level "
                  "features (e.g. calls per step, first-step call count)", "minor")

    un = drift.get("unlabeled", {}).get("model_change") if isinstance(drift.get("unlabeled"), dict) else None
    if un and un.get("share") is not None and un["share"] < 0.5:
        acc = {m: v["ast_overall"]["acc"] for m, v in per_model.items()}
        ctx.limit(f"Without the outcome label, the model change {m0} -> {m1} (AST accuracy {acc}) is flagged in only "
                  f"{un['flagged']} of {un['windows']} windows of {w} runs: the trajectory-shape features of agent_run "
                  "(steps, tool_calls, distinct_tools, tool_error_rate, repeated_calls, output_tokens, termination) "
                  "barely move; with task_success the change is flagged in "
                  f"{(tr.get('model_change') or {}).get('flagged')} of {(tr.get('model_change') or {}).get('windows')}; "
                  f"numbers sent as strings (first step): "
                  f"{ {m: v['string_numbers'] for m, v in per_model.items()} } over {len(cases)} cases each",
                  json.dumps({"unlabeled": drift.get("unlabeled"), "with_task_success": tr}, default=str)[:2500],
                  "Add label-free quality features to agent_run: argument validity against the tool schema "
                  "(type/enum/required checks, e.g. numbers sent as strings), call count vs the number of requested "
                  "actions, and the share of runs whose first step calls no tool; these moved between the two models "
                  "here while the current shape features did not", "major")

    report = {"per_model": per_model, "drift": drift, "paired": paired, "malformed": bad[:50],
              "field_keys": len(field_keys), "per_tool_keys": len(per_tool_keys), "records": len(records)}
    (ctx.workdir / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    tm = tr.get("model_change", {}) if isinstance(tr, dict) else {}
    return {"models_scored": len(per_model), "cases": len(cases), "runs": len(runs), "errors": errors,
            "records": len(records), "malformed_records": bad_n,
            **{f"ast_{m.split(':')[0].replace('.', '')}_{m.split(':')[1]}": v["ast_overall"]["acc"]
               for m, v in per_model.items()},
            "aa_false_alarm_windows_share": tr.get("aa", {}).get("share") if isinstance(tr.get("aa"), dict) else None,
            "drift_windows_share_trajectory": tm.get("share") if isinstance(tm, dict) else None,
            "drift_windows_share_all": (drift.get("all", {}).get("model_change") or {}).get("share"),
            "agent_run_field_keys": len(field_keys)}
