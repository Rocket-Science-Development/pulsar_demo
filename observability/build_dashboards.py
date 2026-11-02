"""Generate the Grafana dashboards of Pulsar.ml (JSON, provisioned from grafana/dashboards/).

Dashboards are code: one Python description, consistent names, colors, variables and links, reviewed like any change.
Queries are Flux (InfluxDB 2). Results come from the `pulsar_results` measurement (ResultsSink), captures from the
`<model>_<version>_input_data` measurements (Pulse), capture counters from `pulsar_capture_stats` (StatsReporter).

    python build_dashboards.py --out grafana/dashboards --bucket monitoring
"""

import argparse
import json
import re
from pathlib import Path

DS = {"type": "influxdb", "uid": "pulsar-influx"}
# Status palette (fixed) and categorical slots (dark mode) of the design system
GOOD, WARNING, SERIOUS, CRITICAL, NEUTRAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b", "#8a8980"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

STATUS_MAPPING = [{
    "type": "value",
    "options": {
        "1": {"text": "▲ Established", "color": CRITICAL, "index": 0},
        "0": {"text": "✓ Within zone", "color": GOOD, "index": 1},
        "-1": {"text": "? Inconclusive", "color": NEUTRAL, "index": 2},
    },
}]
DRIFT_MAPPING = [{
    "type": "value",
    "options": {
        "1": {"text": "▲ Drift", "color": CRITICAL, "index": 0},
        "0": {"text": "✓ Stable", "color": GOOD, "index": 1},
        "-1": {"text": "? n/a", "color": NEUTRAL, "index": 2},
    },
}]
CONFIRMED_MAPPING = [{
    "type": "value",
    "options": {
        "1": {"text": "▲ Confirmed", "color": CRITICAL, "index": 0},
        "0": {"text": "✓ Not confirmed", "color": GOOD, "index": 1},
    },
}]

RANGE = '|> range(start: v.timeRangeStart, stop: v.timeRangeStop)'
RESULTS = f'from(bucket: "${{bucket}}") {RANGE} |> filter(fn: (r) => r._measurement == "pulsar_results")'
CAPTURES = f'from(bucket: "${{bucket}}") {RANGE} |> filter(fn: (r) => r._measurement == "${{model}}_${{version}}_input_data")'
AGENT = (f'from(bucket: "${{bucket}}") {RANGE} |> filter(fn: (r) => r._measurement == "permit_assistant_1.0_input_data")')
STATS = f'from(bucket: "${{bucket}}") {RANGE} |> filter(fn: (r) => r._measurement == "pulsar_capture_stats")'


def fields(*names) -> str:
    return " or ".join(f'r._field == "{n}"' for n in names)


# ----------------------------------------------------------------------------------------------- panel builders
class Grid:
    """Lays panels left to right, 24 columns wide"""

    def __init__(self):
        self.x = self.y = self.row_h = 0

    def place(self, w: int, h: int) -> dict:
        if self.x + w > 24:
            self.x, self.y = 0, self.y + self.row_h
            self.row_h = 0
        pos = {"x": self.x, "y": self.y, "w": w, "h": h}
        self.x += w
        self.row_h = max(self.row_h, h)
        return pos

    def newline(self):
        if self.x:
            self.x, self.y, self.row_h = 0, self.y + self.row_h, 0


class Dashboard:
    def __init__(self, uid: str, title: str, description: str, variables: list, tags=("pulsar.ml",), time="now-7d"):
        self.uid, self.title, self.description, self.variables = uid, title, description, variables
        self.tags, self.time = list(tags), time
        self.panels, self.grid, self._id = [], Grid(), 0

    def add(self, panel: dict, w: int, h: int) -> dict:
        self._id += 1
        panel.update(id=self._id, gridPos=self.grid.place(w, h), datasource=panel.get("datasource", DS))
        self.panels.append(panel)
        return panel

    def row(self, title: str):
        self.grid.newline()
        self._id += 1
        self.panels.append({"type": "row", "title": title, "id": self._id, "collapsed": False, "panels": [],
                            "gridPos": {"x": 0, "y": self.grid.y, "w": 24, "h": 1}})
        self.grid.y += 1

    def text(self, content: str, w=24, h=3):
        return self.add({"type": "text", "title": "", "options": {"mode": "markdown", "content": content},
                         "datasource": None}, w, h)

    def json(self, links: list) -> dict:
        return {
            "uid": self.uid, "title": self.title, "description": self.description, "tags": self.tags,
            "timezone": "browser", "schemaVersion": 39, "version": 1, "editable": True, "graphTooltip": 1,
            "time": {"from": self.time, "to": "now"}, "refresh": "1m",
            "templating": {"list": self.variables}, "panels": self.panels,
            "links": [{"title": "Pulsar.ml", "type": "dashboards", "tags": ["pulsar.ml"], "asDropdown": True,
                       "includeVars": True, "keepTime": True}] + links,
            "annotations": {"list": [{"builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                                      "enable": True, "hide": True, "name": "Annotations & Alerts", "type": "dashboard"}]},
        }


def target(query: str, ref="A") -> dict:
    # a column added by set() is not part of the group key: group on it so that it names the series
    named = re.search(r'\|> set\(key: "(\w+)", value: "[^"]*"\)(?= \|> drop|\s*$)', query)
    if named:
        query = query.replace(named.group(0), named.group(0) + f' |> group(columns: ["{named.group(1)}"])', 1)
    return {"refId": ref, "datasource": DS, "query": query}


def stat(title, query, unit="none", decimals=None, thresholds=None, mappings=None, description="", link=None,
         color_mode="value", graph=False):
    defaults = {"unit": unit, "mappings": mappings or [],
                "thresholds": {"mode": "absolute", "steps": thresholds or [{"color": SERIES[0], "value": None}]},
                "color": {"mode": "thresholds"}}
    if decimals is not None:
        defaults["decimals"] = decimals
    if link:
        defaults["links"] = [{"title": f"Open {link[0]}", "url": f"/d/{link[1]}?${{__url_time_range}}&${{__all_variables}}"}]
    return {"type": "stat", "title": title, "description": description, "targets": [target(query)],
            "fieldConfig": {"defaults": defaults, "overrides": []},
            "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                        "colorMode": color_mode, "graphMode": "area" if graph else "none", "textMode": "auto",
                        "justifyMode": "center", "orientation": "auto"}}


def timeseries(title, queries, unit="none", description="", thresholds=None, display=None, overrides=None,
               stack=False, bars=False, min_=None, max_=None, log=False, show_thresholds=None, fill=10, legend="list"):
    defaults = {
        "unit": unit, "min": min_, "max": max_, "color": {"mode": "palette-classic"},
        "custom": {"lineWidth": 2, "fillOpacity": 70 if bars else fill, "pointSize": 6, "showPoints": "auto",
                   "drawStyle": "bars" if bars else "line", "spanNulls": True,
                   "stacking": {"mode": "normal" if stack else "none", "group": "A"},
                   "axisSoftMin": None, "barAlignment": 0,
                   "thresholdsStyle": {"mode": show_thresholds or "off"},
                   "scaleDistribution": {"type": "log", "log": 10} if log else {"type": "linear"}},
        "thresholds": {"mode": "absolute", "steps": thresholds or [{"color": "transparent", "value": None}]},
    }
    if display:
        defaults["displayName"] = display
    if isinstance(queries, str):
        queries = [queries]
    # without _start / _stop, Grafana takes _time as the time axis even when every tag left the group key
    queries = [q + ' |> drop(fn: (column) => column == "_start" or column == "_stop")' for q in queries]
    return {"type": "timeseries", "title": title, "description": description,
            "targets": [target(q, chr(65 + i)) for i, q in enumerate(queries)],
            "fieldConfig": {"defaults": defaults, "overrides": overrides or []},
            "options": {"legend": {"displayMode": legend, "placement": "bottom", "showLegend": True,
                                   "calcs": ["lastNotNull"] if legend == "table" else []},
                        "tooltip": {"mode": "multi", "sort": "desc"}}}


def status_history(title, query, mappings, description=""):
    return {"type": "status-history", "title": title, "description": description, "targets": [target(query)],
            "fieldConfig": {"defaults": {"mappings": mappings, "color": {"mode": "thresholds"},
                                         "thresholds": {"mode": "absolute", "steps": [{"color": NEUTRAL, "value": None}]},
                                         "custom": {"lineWidth": 1, "fillOpacity": 85}}, "overrides": []},
            "options": {"showValue": "never", "rowHeight": 0.85, "colWidth": 0.95, "legend": {"showLegend": False},
                        "tooltip": {"mode": "single"}}}


def table(title, query, description="", overrides=None, sort=None, unit="none", decimals=3, order=None):
    transformations = []
    if order:  # column order, by source name; unlisted columns go last
        transformations.append({"id": "organize", "options": {"indexByName": {name: i for i, name in enumerate(order)}}})
    return {"type": "table", "title": title, "description": description, "targets": [target(query)],
            "transformations": transformations,
            "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals, "custom": {"align": "auto",
                                                                                        "cellOptions": {"type": "auto"}}},
                            "overrides": overrides or []},
            "options": {"showHeader": True, "cellHeight": "sm", "footer": {"show": False},
                        "sortBy": [{"displayName": sort[0], "desc": sort[1]}] if sort else []}}


def barchart(title, query, unit="none", description="", horizontal=True, color=SERIES[0]):
    return {"type": "barchart", "title": title, "description": description, "targets": [target(query)],
            "fieldConfig": {"defaults": {"unit": unit, "color": {"mode": "fixed", "fixedColor": color},
                                         "custom": {"fillOpacity": 85, "lineWidth": 0}}, "overrides": []},
            "options": {"orientation": "horizontal" if horizontal else "vertical", "showValue": "auto",
                        "xTickLabelRotation": 0, "legend": {"showLegend": False}, "barWidth": 0.7,
                        "groupWidth": 0.7, "tooltip": {"mode": "single"}}}


def status_column(name: str, mapping=STATUS_MAPPING) -> dict:
    return {"matcher": {"id": "byName", "options": name},
            "properties": [{"id": "mappings", "value": mapping},
                           {"id": "custom.cellOptions", "value": {"type": "color-background", "mode": "basic"}},
                           {"id": "custom.width", "value": 150}]}


def rename(name: str, display: str) -> dict:
    return {"matcher": {"id": "byName", "options": name}, "properties": [{"id": "displayName", "value": display}]}


def color(name: str, hex_: str, **extra) -> dict:
    props = [{"id": "color", "value": {"mode": "fixed", "fixedColor": hex_}}]
    for key, value in extra.items():
        props.append({"id": key.replace("__", "."), "value": value})
    return {"matcher": {"id": "byName", "options": name}, "properties": props}


# ----------------------------------------------------------------------------------------------- variables
def var_bucket(default: str) -> dict:
    return {"name": "bucket", "label": "Bucket", "type": "textbox", "query": default,
            "current": {"text": default, "value": default}, "hide": 0}


def var_tag(name, label, tag, measurement_filter, default=None, include_all=False, hide=0):
    query = ('import "influxdata/influxdb/schema"\n'
             f'schema.tagValues(bucket: "${{bucket}}", tag: "{tag}", predicate: (r) => {measurement_filter}, start: -30d)')
    variable = {"name": name, "label": label, "type": "query", "datasource": DS, "query": query, "refresh": 2,
                "sort": 1, "includeAll": include_all, "multi": False, "hide": hide}
    if default:
        variable["current"] = {"text": default, "value": default}
    if include_all:
        variable["allValue"] = ".*"
    return variable


def model_vars(default_model="admissibilite"):
    return [
        var_tag("model", "Model", "model_id",
                'r._measurement == "pulsar_results" and r.model_id != "permit_assistant"', default_model),
        var_tag("version", "Version", "model_version",
                'r._measurement == "pulsar_results" and r.model_id == "${model}"', "1.0"),
    ]


# ----------------------------------------------------------------------------------------------- dashboards
def overview(bucket):
    d = Dashboard("pulsar-overview", "Pulsar.ml — Overview",
                  "State of every monitored model: captures, confirmed drift, fairness, personal information, capture health",
                  [var_bucket(bucket)] + model_vars(), time="now-24h")
    d.text("### Pulsar.ml — monitoring overview\nEvery tile shows the **latest verdict**, never a raw score: a drift counts only "
           "once confirmed over two windows, a disparity only when its confidence interval leaves the acceptable zone. "
           "Click a tile to open the detailed dashboard.", h=3)
    last = f'{RESULTS} |> filter(fn: (r) => r.model_id == "${{model}}" and r._field == "status")'
    d.add(stat("Predictions captured (period)",
               f'{CAPTURES} |> filter(fn: (r) => r._field == "y_pred") |> group() |> count()',
               link=("performance", "pulsar-performance"), graph=False), 4, 4)
    d.add(stat("Confirmed drift (latest window)",
               f'{RESULTS} |> filter(fn: (r) => r.model_id == "${{model}}" and r.metric_type == "drift" and r._field == "confirmed")'
               ' |> last() |> toInt() |> group() |> sum()',
               thresholds=[{"color": GOOD, "value": None}, {"color": CRITICAL, "value": 1}],
               link=("data drift", "pulsar-drift"),
               description="Features whose drift persisted over two consecutive windows (BH-adjusted, A/A-calibrated)"), 4, 4)
    d.add(stat("Disparities established (latest)",
               f'{last} |> filter(fn: (r) => r.metric_type == "fairness" and r.test_type == "fixed") |> last()'
               ' |> map(fn: (r) => ({r with _value: if r._value == 1 then 1 else 0})) |> group() |> sum()',
               thresholds=[{"color": GOOD, "value": None}, {"color": CRITICAL, "value": 1}],
               link=("fairness", "pulsar-fairness"),
               description="Groups whose confidence interval lies entirely outside the acceptable zone (latest analysis, "
                           "fixed test on a rolling window). Inconclusive groups are listed on the fairness dashboard."), 4, 4)
    d.add(stat("Explanation changes (latest)",
               f'{last} |> filter(fn: (r) => r.metric_type == "explainability" and r.metric_name == "importance_share_difference")'
               ' |> last() |> map(fn: (r) => ({r with _value: if r._value == 1 then 1 else 0})) |> group() |> sum()',
               thresholds=[{"color": GOOD, "value": None}, {"color": WARNING, "value": 1}],
               link=("explanations", "pulsar-explain")), 4, 4)
    d.add(stat("Personal information masked (period)",
               f'{CAPTURES} |> filter(fn: (r) => r._field == "pii_count") |> group() |> sum()',
               thresholds=[{"color": SERIES[0], "value": None}], link=("personal information", "pulsar-privacy"),
               description="Findings replaced by a label before writing (CaptureOptions(pii=PiiPolicy()))"), 4, 4)
    d.add(stat("Rows dropped by the capture",
               f'{STATS} |> filter(fn: (r) => r.model_id == "${{model}}" and r._field == "dropped_rows") |> last() |> group() |> sum()',
               thresholds=[{"color": GOOD, "value": None}, {"color": CRITICAL, "value": 1}],
               link=("capture health", "pulsar-capture")), 4, 4)
    d.add(timeseries("Alerts over time, by kind",
                     f'{last} |> filter(fn: (r) => r._value == 1 and (r.metric_type == "drift" or r.metric_type == "fairness"'
                     ' or r.metric_type == "explainability" or r.metric_type == "performance"))'
                     ' |> group(columns: ["metric_type"]) |> aggregateWindow(every: 6h, fn: count, createEmpty: false)',
                     bars=True, stack=True, display="${__field.labels.metric_type}",
                     overrides=[color("drift", SERIES[0]), color("fairness", SERIES[1]), color("explainability", SERIES[2]),
                                color("performance", SERIES[6])],
                     description="Number of results establishing a change in each 6-hour window"), 14, 9)
    d.add(table("Latest verdict by kind",
                f'{last} |> filter(fn: (r) => r.metric_type != "statistics") |> last()'
                ' |> group(columns: ["metric_type"])'
                ' |> reduce(identity: {established: 0, within: 0, inconclusive: 0}, fn: (r, accumulator) => ({'
                'established: accumulator.established + (if r._value == 1 then 1 else 0), '
                'within: accumulator.within + (if r._value == 0 then 1 else 0), '
                'inconclusive: accumulator.inconclusive + (if r._value == -1 then 1 else 0)}))'
                ' |> group()',
                overrides=[rename("metric_type", "Kind"), rename("established", "▲ established"), rename("within", "✓ within zone"),
                           rename("inconclusive", "? inconclusive")],
                decimals=0), 10, 9)
    d.row("Agents and LLM applications")
    d.add(stat("Agent runs (period)", f'{AGENT} |> filter(fn: (r) => r.record_type == "agent_run" and r._field == "steps")'
               ' |> group() |> count()', link=("agents", "pulsar-agents")), 6, 4)
    d.add(stat("Tool error rate", f'{AGENT} |> filter(fn: (r) => r.record_type == "agent_run" and r._field == "tool_error_rate")'
               ' |> group() |> mean()', unit="percentunit", decimals=1,
               thresholds=[{"color": GOOD, "value": None}, {"color": WARNING, "value": 0.1}, {"color": CRITICAL, "value": 0.25}],
               link=("agents", "pulsar-agents")), 6, 4)
    d.add(stat("Personal-information leaks", f'{AGENT} |> filter(fn: (r) => r.record_type == "agent_run" and r._field == "pii_leak_count")'
               ' |> group() |> sum()', thresholds=[{"color": GOOD, "value": None}, {"color": CRITICAL, "value": 1}],
               link=("personal information", "pulsar-privacy"),
               description="Personal information in an agent output that the user did not provide"), 6, 4)
    d.add(stat("Agent drift confirmed (latest window)",
               f'{RESULTS} |> filter(fn: (r) => r.model_id == "permit_assistant" and r.metric_type == "drift" and r._field == "confirmed")'
               ' |> group(columns: ["feature_name"]) |> sort(columns: ["_time"]) |> last() |> toInt() |> group() |> sum()',
               thresholds=[{"color": GOOD, "value": None}, {"color": CRITICAL, "value": 1}], link=("agents", "pulsar-agents")), 6, 4)
    return d


def drift(bucket):
    d = Dashboard("pulsar-drift", "Pulsar.ml — Data drift",
                  "Calibrated drift tests per feature and window: raw alerts, confirmation, adjusted p-values, distributions",
                  [var_bucket(bucket)] + model_vars() + [
                      var_tag("feature", "Feature", "feature_name",
                              'r._measurement == "pulsar_results" and r.model_id == "${model}" and r.metric_type == "drift"', "income")])
    d.text("**How to read**: each window is tested against the reference with thresholds calibrated by A/A resampling of the "
           "reference; p-values are adjusted for the number of features (Benjamini-Hochberg, 5 %). A raw alert becomes "
           "**confirmed** only when it persists over two consecutive windows.", h=3)
    base = f'{RESULTS} |> filter(fn: (r) => r.model_id == "${{model}}" and r.metric_type == "drift")'
    d.add(status_history("Raw alerts (BH-adjusted), by feature",
                         f'{base} |> filter(fn: (r) => r._field == "status") |> group()'
                         ' |> keep(columns: ["_time", "_value", "feature_name"])'
                         ' |> pivot(rowKey: ["_time"], columnKey: ["feature_name"], valueColumn: "_value")',
                         DRIFT_MAPPING), 12, 8)
    d.add(status_history("Confirmed drift (2 windows), by feature",
                         f'{base} |> filter(fn: (r) => r._field == "confirmed") |> toInt() |> group()'
                         ' |> keep(columns: ["_time", "_value", "feature_name"])'
                         ' |> pivot(rowKey: ["_time"], columnKey: ["feature_name"], valueColumn: "_value")',
                         CONFIRMED_MAPPING), 12, 8)
    d.add(timeseries("Adjusted p-value by feature (log scale)",
                     f'{base} |> filter(fn: (r) => r._field == "p_adjusted") |> group(columns: ["feature_name"])',
                     display="${__field.labels.feature_name}", log=True, min_=0.001, max_=1, fill=0,
                     thresholds=[{"color": CRITICAL, "value": None}, {"color": "transparent", "value": 0.05}],
                     show_thresholds="dashed",
                     description="Below the dashed red line (0.05) the window raises a raw alert for that feature"), 14, 9)
    d.add(barchart("Windows with a confirmed drift, by feature (period)",
                   f'{base} |> filter(fn: (r) => r._field == "confirmed") |> toInt() |> group(columns: ["feature_name"]) |> sum()'
                   ' |> group() |> keep(columns: ["feature_name", "_value"]) |> rename(columns: {_value: "windows"})'
                   ' |> sort(columns: ["windows"], desc: true)', color=SERIES[0]), 10, 9)
    d.row("Distribution of the selected feature: ${feature}")
    q = f'{CAPTURES} |> filter(fn: (r) => r._field == "${{feature}}") |> group()'
    d.add(timeseries("Hourly quantiles of ${feature}", [
        f'{q} |> aggregateWindow(every: 1h, fn: (column, tables=<-) => tables |> quantile(q: 0.9, column: column), createEmpty: false) |> set(key: "q", value: "p90")',
        f'{q} |> aggregateWindow(every: 1h, fn: median, createEmpty: false) |> set(key: "q", value: "median")',
        f'{q} |> aggregateWindow(every: 1h, fn: (column, tables=<-) => tables |> quantile(q: 0.1, column: column), createEmpty: false) |> set(key: "q", value: "p10")',
    ], display="${__field.labels.q}",
        overrides=[color("p90", SERIES[0], custom__fillBelowTo="p10", custom__lineWidth=1),
                   color("median", SERIES[1]), color("p10", SERIES[0], custom__lineWidth=1)],
        description="Numeric features: the band spans the 10th to 90th percentiles of each hour; a shift of the band is "
                    "the drift that the tests measure", fill=0), 16, 9)
    d.add(table("Latest window: tests by feature",
                f'{base} |> filter(fn: (r) => r._field == "metric_value" or r._field == "p_adjusted" or r._field == "status" or r._field == "confirmed")'
                ' |> last() |> toFloat() |> group() |> pivot(rowKey: ["feature_name", "metric_name"], columnKey: ["_field"], valueColumn: "_value")'
                ' |> keep(columns: ["feature_name", "metric_name", "metric_value", "p_adjusted", "status", "confirmed"])',
                overrides=[rename("feature_name", "Feature"), rename("metric_name", "Test"), rename("metric_value", "Statistic"),
                           rename("p_adjusted", "p (adjusted)"), status_column("status", DRIFT_MAPPING),
                           status_column("confirmed", CONFIRMED_MAPPING)]), 8, 9)
    return d


def performance(bucket):
    d = Dashboard("pulsar-performance", "Pulsar.ml — Model performance",
                  "Quality metrics once labels arrive, approval rate and scores", [var_bucket(bucket)] + model_vars())
    d.text("Labels arrive **late** (here 48 hours): the most recent windows have no performance yet, see the label coverage. "
           "Accuracy is flagged when it falls more than 5 points below the reference.", h=2)
    base = f'{RESULTS} |> filter(fn: (r) => r.model_id == "${{model}}" and r.metric_type == "performance")'
    d.add(stat("Accuracy (latest labelled window)", f'{base} |> filter(fn: (r) => r.metric_name == "accuracy" and r._field == "metric_value") |> last()',
               unit="percentunit", decimals=1, thresholds=[{"color": SERIES[0], "value": None}]), 6, 4)
    d.add(stat("Accuracy verdict", f'{base} |> filter(fn: (r) => r.metric_name == "accuracy" and r._field == "status") |> last()',
               mappings=[{"type": "value", "options": {"1": {"text": "▲ Below reference", "color": CRITICAL},
                                                         "0": {"text": "✓ Holds", "color": GOOD}}}], color_mode="background"), 6, 4)
    d.add(stat("Approval rate (period)", f'{CAPTURES} |> filter(fn: (r) => r._field == "y_pred") |> group() |> mean()',
               unit="percentunit", decimals=1), 6, 4)
    d.add(stat("Label coverage (latest window)", f'{base} |> filter(fn: (r) => r.metric_name == "label_coverage" and r._field == "metric_value") |> last()',
               unit="percentunit", decimals=0), 6, 4)
    d.add(timeseries("Quality by window",
                     [f'{base} |> filter(fn: (r) => r._field == "metric_value" and r.metric_name != "label_coverage") |> group(columns: ["metric_name"])',
                      f'{base} |> filter(fn: (r) => r.metric_name == "accuracy" and r._field == "threshold") |> group() |> set(key: "metric_name", value: "accuracy floor")'],
                     unit="percentunit", display="${__field.labels.metric_name}", min_=0.5, max_=1, fill=0,
                     overrides=[color("accuracy", SERIES[0]), color("precision", SERIES[1]), color("recall", SERIES[2]),
                                color("f1", SERIES[6]),
                                color("accuracy floor", CRITICAL, custom__lineStyle={"fill": "dash", "dash": [8, 6]},
                                      custom__lineWidth=1)]), 16, 9)
    d.add(timeseries("Label coverage by window",
                     f'{base} |> filter(fn: (r) => r.metric_name == "label_coverage" and r._field == "metric_value")',
                     unit="percentunit", bars=True, display="labelled share", min_=0, max_=1), 8, 9)
    d.add(timeseries("Hourly approval rate and mean score", [
        f'{CAPTURES} |> filter(fn: (r) => r._field == "y_pred") |> group() |> aggregateWindow(every: 1h, fn: mean, createEmpty: false) |> set(key: "s", value: "approval rate")',
        f'{CAPTURES} |> filter(fn: (r) => r._field == "y_pred_proba") |> group() |> aggregateWindow(every: 1h, fn: mean, createEmpty: false) |> set(key: "s", value: "mean score")',
    ], unit="percentunit", display="${__field.labels.s}", min_=0, max_=1, fill=0,
        overrides=[color("approval rate", SERIES[0]), color("mean score", SERIES[1])]), 24, 8)
    return d


def agents(bucket):
    d = Dashboard("pulsar-agents", "Pulsar.ml — Agents and LLM applications",
                  "Agent runs collected through OpenTelemetry GenAI: volume, latency, tools, tokens, terminations, drift",
                  [var_bucket(bucket),
                   var_tag("campaign", "Campaign", "campaign",
                           'r._measurement == "permit_assistant_1.0_input_data"', "s2-gpu-v2-20261101")], time="now-6h")
    runs = f'{AGENT} |> filter(fn: (r) => r.record_type == "agent_run" and r.campaign == "${{campaign}}")'
    tools = f'{AGENT} |> filter(fn: (r) => r.record_type == "tool_call" and r.campaign == "${{campaign}}")'
    d.add(stat("Runs", f'{runs} |> filter(fn: (r) => r._field == "steps") |> group() |> count()'), 4, 4)
    d.add(stat("Median latency", f'{runs} |> filter(fn: (r) => r._field == "latency_s") |> group() |> median()', unit="s", decimals=2), 4, 4)
    d.add(stat("p95 latency", f'{runs} |> filter(fn: (r) => r._field == "latency_s") |> group() |> quantile(q: 0.95)', unit="s", decimals=2,
               thresholds=[{"color": SERIES[0], "value": None}, {"color": WARNING, "value": 10}]), 4, 4)
    d.add(stat("Tool error rate", f'{runs} |> filter(fn: (r) => r._field == "tool_error_rate") |> group() |> mean()', unit="percentunit",
               decimals=1, thresholds=[{"color": GOOD, "value": None}, {"color": WARNING, "value": 0.1}, {"color": CRITICAL, "value": 0.25}]), 4, 4)
    d.add(stat("Task success", f'{runs} |> filter(fn: (r) => r._field == "task_success") |> toFloat() |> group() |> mean()', unit="percentunit",
               decimals=1, thresholds=[{"color": CRITICAL, "value": None}, {"color": WARNING, "value": 0.7}, {"color": GOOD, "value": 0.85}]), 4, 4)
    d.add(stat("Tokens (in + out)", f'{runs} |> filter(fn: (r) => r._field == "input_tokens" or r._field == "output_tokens") |> group() |> sum()',
               unit="short"), 4, 4)
    d.add(timeseries("Latency per run, by phase",
                     f'{runs} |> filter(fn: (r) => r._field == "latency_s") |> group(columns: ["scenario"])',
                     unit="s", display="${__field.labels.scenario}", fill=0,
                     description="Each point is one run; phases are the scenarios of the campaign (reference, holdout, injected drifts)"), 12, 9)
    d.add(timeseries("Tool error rate and task success (10-minute windows)", [
        f'{runs} |> filter(fn: (r) => r._field == "tool_error_rate") |> group() |> aggregateWindow(every: 5m, fn: mean, createEmpty: false) |> set(key: "s", value: "tool error rate")',
        f'{runs} |> filter(fn: (r) => r._field == "task_success") |> toFloat() |> group() |> aggregateWindow(every: 5m, fn: mean, createEmpty: false) |> set(key: "s", value: "task success")',
    ], unit="percentunit", display="${__field.labels.s}", min_=0, max_=1,
        overrides=[color("tool error rate", CRITICAL), color("task success", GOOD)]), 12, 9)
    d.add(barchart("First tool called",
                   f'{runs} |> filter(fn: (r) => r._field == "first_tool") |> group(columns: ["_value"]) |> count(column: "_field")'
                   ' |> group() |> rename(columns: {_value: "tool", _field: "runs"}) |> sort(columns: ["runs"], desc: true)'), 8, 8)
    d.add(barchart("Errors by tool",
                   f'{tools} |> filter(fn: (r) => r._field == "error" or r._field == "tool_name")'
                   ' |> map(fn: (r) => ({r with _value: string(v: r._value)})) |> group()'
                   ' |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'
                   ' |> map(fn: (r) => ({tool_name: r.tool_name, errors: if r.error == "true" then 1.0 else 0.0}))'
                   ' |> group(columns: ["tool_name"]) |> sum(column: "errors") |> group() |> sort(columns: ["errors"], desc: true)',
                   color=CRITICAL), 8, 8)
    d.add(barchart("Terminations",
                   f'{runs} |> filter(fn: (r) => r._field == "termination") |> group(columns: ["_value"]) |> count(column: "_field")'
                   ' |> group() |> rename(columns: {_value: "termination", _field: "runs"})', color=SERIES[2]), 8, 8)
    d.add(timeseries("Tokens per run", [
        f'{runs} |> filter(fn: (r) => r._field == "input_tokens") |> group() |> set(key: "s", value: "input")',
        f'{runs} |> filter(fn: (r) => r._field == "output_tokens") |> group() |> set(key: "s", value: "output")',
    ], unit="short", display="${__field.labels.s}", fill=0), 12, 8)
    d.add(timeseries("Steps and tool calls per run", [
        f'{runs} |> filter(fn: (r) => r._field == "steps") |> group() |> set(key: "s", value: "LLM calls")',
        f'{runs} |> filter(fn: (r) => r._field == "tool_calls") |> group() |> set(key: "s", value: "tool calls")',
    ], display="${__field.labels.s}", fill=0), 12, 8)
    d.row("Drift of the agent (windows of 30 runs, calibrated on the reference phase)")
    base = f'{RESULTS} |> filter(fn: (r) => r.model_id == "permit_assistant" and r.metric_type == "drift" and r.campaign == "${{campaign}}")'
    d.add(status_history("Confirmed drift by feature",
                         f'{base} |> filter(fn: (r) => r._field == "confirmed") |> toInt() |> group()'
                         ' |> keep(columns: ["_time", "_value", "feature_name"])'
                         ' |> pivot(rowKey: ["_time"], columnKey: ["feature_name"], valueColumn: "_value")', CONFIRMED_MAPPING), 16, 9)
    d.add(table("Windows and phases",
                f'{base} |> filter(fn: (r) => r._field == "confirmed") |> toInt() |> group(columns: ["_time", "phase"]) |> sum()'
                ' |> group() |> keep(columns: ["_time", "phase", "_value"]) |> rename(columns: {_value: "confirmed features"})',
                overrides=[rename("phase", "Phase"), rename("_time", "Window end")], decimals=0, sort=("Window end", True)), 8, 9)
    d.row("Detectors and behavioural rules (scores computed in-process, never the text)")
    d.add(timeseries("Injection score per run, by scenario",
                     f'{runs} |> filter(fn: (r) => r._field == "det_injection") |> group(columns: ["scenario"])',
                     display="${__field.labels.scenario}", min_=0, max_=1, fill=0,
                     thresholds=[{"color": "transparent", "value": None}, {"color": CRITICAL, "value": 0.5}],
                     show_thresholds="dashed",
                     description="Probability that the request is a prompt injection (protectai/deberta-v3-base-prompt-injection-v2)"), 12, 8)
    d.add(timeseries("Faithfulness of answers to retrieved rules, by scenario",
                     f'{runs} |> filter(fn: (r) => r._field == "det_faithfulness") |> group(columns: ["scenario"])',
                     display="${__field.labels.scenario}", min_=0, max_=1, fill=0,
                     description="Share of the answer entailed by the passages returned by search_documents (multilingual NLI)"), 12, 8)
    tests = f'{RESULTS} |> filter(fn: (r) => r.model_id == "permit_assistant" and r.metric_type == "agents" and r.campaign == "${{campaign}}")'
    d.add(table("Detector verdicts per window (S0 to S4) — p-value vs reference",
                f'{tests} |> filter(fn: (r) => r.metric_name != "rule_compliance" and r._field == "p_value") |> last()'
                ' |> group() |> pivot(rowKey: ["phase", "level"], columnKey: ["feature_name"], valueColumn: "_value")'
                ' |> sort(columns: ["phase", "level"])',
                overrides=[rename("phase", "Scenario"), rename("level", "Intensity"), rename("mmd", "MMD (inputs)"),
                           rename("faithfulness", "Faithfulness (RAG)"), rename("injection", "Injection rate"),
                           rename("compliance", "Rule violations")]
                + [{"matcher": {"id": "byName", "options": name},
                    "properties": [{"id": "custom.cellOptions", "value": {"type": "color-background", "mode": "basic"}},
                                   {"id": "thresholds", "value": {"mode": "absolute", "steps": [
                                       {"color": CRITICAL, "value": None}, {"color": "transparent", "value": 0.05}]}}]}
                   for name in ("mmd", "faithfulness", "injection", "compliance")],
                order=["phase", "level", "mmd", "faithfulness", "injection", "compliance"],
                description="Red: p < 0.05, the detector sees a change against the reference phase"), 14, 10)
    d.add(table("Compliance with the behavioural rules (LTLf), by scenario",
                f'{tests} |> filter(fn: (r) => r.metric_name == "rule_compliance" and r._field == "metric_value") |> last()'
                ' |> group() |> pivot(rowKey: ["phase"], columnKey: ["feature_name"], valueColumn: "_value")',
                unit="percentunit", decimals=1,
                overrides=[rename("phase", "Scenario")]
                + [{"matcher": {"id": "byType", "options": "number"},
                    "properties": [{"id": "custom.cellOptions", "value": {"type": "color-text"}},
                                   {"id": "thresholds", "value": {"mode": "absolute", "steps": [
                                       {"color": CRITICAL, "value": None}, {"color": WARNING, "value": 0.95},
                                       {"color": GOOD, "value": 1}]}}]}],
                description="Share of runs complying with each rule of genai_agent/rules/permit_agent.yaml"), 10, 10)
    return d


def fairness(bucket):
    d = Dashboard("pulsar-fairness", "Pulsar.ml — Fairness",
                  "Disparities between groups with confidence intervals, anytime-valid monitoring and counterfactual probes",
                  [var_bucket(bucket)] + model_vars() + [
                      var_tag("attribute", "Attribute", "feature_name",
                              'r._measurement == "pulsar_results" and r.model_id == "${model}" and r.metric_type == "fairness" and r.test_type == "fixed"',
                              "language"),
                      var_tag("group", "Group", "group",
                              'r._measurement == "pulsar_results" and r.model_id == "${model}" and r.feature_name == "${attribute}"',
                              "language=other")])
    d.text("**How to read**: a disparity is **established** only when its confidence interval lies entirely outside the acceptable "
           "zone (disparate impact ratio 0.8 to 1.25, the 4/5 rule; parity difference ±0.10). Small groups (< 30) are never shown. "
           "The anytime-valid monitor keeps its false-alarm rate below 5 % however often it is looked at.", h=3)
    fixed = f'{RESULTS} |> filter(fn: (r) => r.model_id == "${{model}}" and r.metric_type == "fairness" and r.test_type == "fixed" and r.feature_name == "${{attribute}}")'
    d.add(table("Latest window, by group",
                f'{fixed} |> filter(fn: (r) => r._field == "metric_value" or r._field == "ci_low" or r._field == "ci_high" or r._field == "n" or r._field == "status")'
                ' |> last() |> toFloat() |> group() |> pivot(rowKey: ["group", "metric_name", "reference_group"], columnKey: ["_field"], valueColumn: "_value")'
                ' |> keep(columns: ["group", "reference_group", "metric_name", "n", "metric_value", "ci_low", "ci_high", "status"])',
                overrides=[rename("group", "Group"), rename("reference_group", "Reference"), rename("metric_name", "Criterion"),
                           rename("metric_value", "Value"), rename("ci_low", "CI low"), rename("ci_high", "CI high"),
                           {"matcher": {"id": "byName", "options": "n"}, "properties": [{"id": "decimals", "value": 0}]},
                           status_column("status")], sort=("Group", False),
                order=["group", "reference_group", "metric_name", "n", "metric_value", "ci_low", "ci_high", "status"]), 24, 8)
    d.add(timeseries("Disparate impact ratio by group (zone 0.8 – 1.25)",
                     f'{fixed} |> filter(fn: (r) => r.metric_name == "disparate_impact_ratio" and r._field == "metric_value") |> group(columns: ["group"])',
                     display="${__field.labels.group}", min_=0.3, max_=1.5,
                     thresholds=[{"color": CRITICAL, "value": None}, {"color": "transparent", "value": 0.8}, {"color": CRITICAL, "value": 1.25}],
                     show_thresholds="area", fill=0), 12, 9)
    d.add(timeseries("Parity difference of ${group}, with its confidence interval", [
        f'{fixed} |> filter(fn: (r) => r.metric_name == "demographic_parity_difference" and r.group == "${{group}}" and r._field == "metric_value") |> group() |> set(key: "s", value: "difference")',
        f'{fixed} |> filter(fn: (r) => r.metric_name == "demographic_parity_difference" and r.group == "${{group}}" and r._field == "ci_high") |> group() |> set(key: "s", value: "CI high")',
        f'{fixed} |> filter(fn: (r) => r.metric_name == "demographic_parity_difference" and r.group == "${{group}}" and r._field == "ci_low") |> group() |> set(key: "s", value: "CI low")',
    ], display="${__field.labels.s}", min_=-0.4, max_=0.2,
        thresholds=[{"color": CRITICAL, "value": None}, {"color": "transparent", "value": -0.1}, {"color": CRITICAL, "value": 0.1}],
        show_thresholds="area", fill=0,
        overrides=[color("difference", SERIES[0]),
                   color("CI high", SERIES[0], custom__fillBelowTo="CI low", custom__lineWidth=0, custom__fillOpacity=20),
                   color("CI low", SERIES[0], custom__lineWidth=0)]), 12, 9)
    d.row("Anytime-valid monitoring (sequential test by betting)")
    seq = f'{RESULTS} |> filter(fn: (r) => r.model_id == "${{model}}" and r.test_type == "sequential")'
    d.add(timeseries("Evidence of a disparity (e-value, log scale) — alarm at 40",
                     f'{seq} |> filter(fn: (r) => r._field == "e_value") |> group(columns: ["group"])',
                     display="${__field.labels.group}", log=True, min_=0.01, max_=1000, fill=0,
                     thresholds=[{"color": "transparent", "value": None}, {"color": CRITICAL, "value": 40}],
                     show_thresholds="line+area",
                     description="1 / (alpha / number of groups) = 1 / (0.05 / 2) = 40. Once reached, the disparity is established for good."), 12, 9)
    d.add(timeseries("Cumulative selection-rate difference vs reference",
                     f'{seq} |> filter(fn: (r) => r._field == "metric_value") |> group(columns: ["group"])',
                     display="${__field.labels.group}",
                     thresholds=[{"color": CRITICAL, "value": None}, {"color": "transparent", "value": -0.05}, {"color": CRITICAL, "value": 0.05}],
                     show_thresholds="area", fill=0), 12, 9)
    d.row("Intersections (language × region), latest window")
    d.add(table("Disparate impact ratio of intersectional groups",
                f'{RESULTS} |> filter(fn: (r) => r.model_id == "${{model}}" and r.metric_name == "disparate_impact_ratio" and r.feature_name == "language|region")'
                ' |> filter(fn: (r) => r._field == "metric_value" or r._field == "n" or r._field == "status" or r._field == "suppressed")'
                ' |> last() |> toFloat() |> group() |> pivot(rowKey: ["group"], columnKey: ["_field"], valueColumn: "_value")',
                order=["group", "n", "metric_value", "status", "suppressed"],
                overrides=[rename("group", "Group"), rename("metric_value", "Ratio"), status_column("status"),
                           {"matcher": {"id": "byName", "options": "suppressed"},
                            "properties": [{"id": "mappings", "value": [{"type": "value", "options": {"1": {"text": "< 30, withheld"}, "0": {"text": ""}}}]}]},
                           {"matcher": {"id": "byName", "options": "n"}, "properties": [{"id": "decimals", "value": 0}]}],
                sort=("Ratio", False)), 12, 10)
    d.add(table("Counterfactual probe of the demo agent (first name)",
                f'{RESULTS} |> filter(fn: (r) => r.campaign == "probe-e4")'
                ' |> filter(fn: (r) => r._field == "metric_value" or r._field == "ci_low" or r._field == "ci_high" or r._field == "status")'
                ' |> last() |> toFloat() |> group() |> pivot(rowKey: ["scenario", "metric_name", "group"], columnKey: ["_field"], valueColumn: "_value")'
                ' |> filter(fn: (r) => r.metric_name == "counterfactual_action_rate_difference" or r.metric_name == "placebo_flip_rate" or r.metric_name == "position_effect" or r.metric_name == "counterfactual_excess_flip_rate")',
                overrides=[rename("scenario", "Prompt"), rename("metric_name", "Measure"), rename("group", "Group"),
                           rename("metric_value", "Value"), rename("ci_low", "CI low"), rename("ci_high", "CI high"),
                           status_column("status")], sort=("Prompt", True),
                order=["scenario", "metric_name", "group", "metric_value", "ci_low", "ci_high", "status"],
                description="Requests identical except for the first name; placebo sets change a file reference. "
                            "The biased prompt is a deliberate test of the instrument."), 12, 10)
    return d


def explain(bucket):
    d = Dashboard("pulsar-explain", "Pulsar.ml — Explanations",
                  "What the model relies on (TreeSHAP), ranking stability, changes of weight and importance by group",
                  [var_bucket(bucket)] + model_vars())
    d.text("Explanations are estimates too: every change carries a bootstrap interval, and no verdict is given when the ranking is "
           "not stable on the reference window. A rule applied **after** the model (such as a bias added in production) does not "
           "show here: explanations describe the model, not the decision pipeline.", h=3)
    base = f'{RESULTS} |> filter(fn: (r) => r.model_id == "${{model}}" and r.metric_type == "explainability")'
    d.add(timeseries("Share of the total importance, by feature",
                     f'{base} |> filter(fn: (r) => r.metric_name == "importance_share" and r._field == "metric_value") |> group(columns: ["feature_name"])',
                     unit="percentunit", stack=True, fill=80, display="${__field.labels.feature_name}", min_=0, max_=1), 12, 9)
    d.add(timeseries("Ranking stability (top 3): NDCG and overlap", [
        f'{base} |> filter(fn: (r) => r.metric_name == "attribution_ndcg_at_k" and r._field == "metric_value") |> set(key: "s", value: "NDCG@3 (threshold 0.9)")',
        f'{base} |> filter(fn: (r) => r.metric_name == "attribution_top_k_jaccard" and r._field == "metric_value") |> set(key: "s", value: "top-3 overlap (threshold 0.6)")',
    ], display="${__field.labels.s}", min_=0, max_=1.05,
        thresholds=[{"color": CRITICAL, "value": None}, {"color": "transparent", "value": 0.6}],
        show_thresholds="area", fill=0), 12, 9)
    d.add(table("Change of weight by feature, latest window",
                f'{base} |> filter(fn: (r) => r.metric_name == "importance_share_difference")'
                ' |> filter(fn: (r) => r._field == "metric_value" or r._field == "ci_low" or r._field == "ci_high" or r._field == "status")'
                ' |> last() |> toFloat() |> group() |> pivot(rowKey: ["feature_name"], columnKey: ["_field"], valueColumn: "_value")',
                overrides=[rename("feature_name", "Feature"), rename("metric_value", "Change (share)"),
                           rename("ci_low", "CI low"), rename("ci_high", "CI high"), status_column("status")],
                order=["feature_name", "metric_value", "ci_low", "ci_high", "status"]), 12, 8)
    d.add(table("Importance by group (vs fr), latest window",
                f'{base} |> filter(fn: (r) => r.metric_name == "importance_share_difference_by_group")'
                ' |> filter(fn: (r) => r._field == "metric_value" or r._field == "ci_low" or r._field == "ci_high" or r._field == "status" or r._field == "n")'
                ' |> last() |> toFloat() |> group() |> pivot(rowKey: ["group", "feature_name"], columnKey: ["_field"], valueColumn: "_value")',
                overrides=[rename("group", "Group"), rename("feature_name", "Feature"), rename("metric_value", "Difference of share"),
                           rename("ci_low", "CI low"), rename("ci_high", "CI high"),
                           status_column("status"), {"matcher": {"id": "byName", "options": "n"}, "properties": [{"id": "decimals", "value": 0}]}],
                order=["group", "feature_name", "n", "metric_value", "ci_low", "ci_high", "status"],
                description="A feature that weighs much more for one group can act as a proxy of the protected attribute"), 12, 8)
    d.add(status_history("Verdicts over time: change of weight by feature",
                         f'{base} |> filter(fn: (r) => r.metric_name == "importance_share_difference" and r._field == "status") |> group()'
                         ' |> keep(columns: ["_time", "_value", "feature_name"])'
                         ' |> pivot(rowKey: ["_time"], columnKey: ["feature_name"], valueColumn: "_value")', STATUS_MAPPING), 24, 7)
    return d


def privacy(bucket):
    d = Dashboard("pulsar-privacy", "Pulsar.ml — Personal information",
                  "Personal information detected and masked before writing; leaks of agents", [var_bucket(bucket)] + model_vars())
    d.text("Values are **never stored**: tabular captures keep the masked text and a `pii_count` per row; agent records keep counts and "
           "types by role. A **leak** is personal information in an agent output that the user did not provide.", h=2)
    pii = f'{CAPTURES} |> filter(fn: (r) => r._field == "pii_count")'
    d.add(stat("Findings masked (period)", f'{pii} |> group() |> sum()'), 6, 4)
    d.add(stat("Rows with personal information", f'{pii} |> map(fn: (r) => ({{r with _value: if r._value > 0 then 1.0 else 0.0}})) |> group() |> mean()',
               unit="percentunit", decimals=2), 6, 4)
    runs = f'{AGENT} |> filter(fn: (r) => r.record_type == "agent_run")'
    d.add(stat("Agent leaks", f'{runs} |> filter(fn: (r) => r._field == "pii_leak_count") |> group() |> sum()',
               thresholds=[{"color": GOOD, "value": None}, {"color": CRITICAL, "value": 1}], color_mode="background"), 6, 4)
    d.add(stat("Agent runs with a leak", f'{runs} |> filter(fn: (r) => r._field == "pii_leak_count") |> map(fn: (r) => ({{r with _value: if r._value > 0 then 1.0 else 0.0}})) |> group() |> mean()',
               unit="percentunit", decimals=1, thresholds=[{"color": GOOD, "value": None}, {"color": CRITICAL, "value": 0.0001}]), 6, 4)
    d.add(timeseries("Findings masked in tabular captures, per hour",
                     f'{pii} |> group() |> aggregateWindow(every: 1h, fn: sum, createEmpty: false)', bars=True, display="findings masked"), 12, 8)
    d.add(timeseries("Agent personal information by role", [
        f'{runs} |> filter(fn: (r) => r._field == "pii_{role}_count") |> group() |> aggregateWindow(every: 1m, fn: sum, createEmpty: false) |> set(key: "role", value: "{role}")'
        for role in ("input", "context", "output", "leak")
    ], bars=True, display="${__field.labels.role}",
        overrides=[color("input", SERIES[0]), color("context", SERIES[2]), color("output", SERIES[6]), color("leak", CRITICAL)]), 12, 8)
    d.add(barchart("Types leaked by agents",
                   f'{runs} |> filter(fn: (r) => r._field == "pii_leak_types" and r._value != "none") |> group(columns: ["_value"]) |> count(column: "_field")'
                   ' |> group() |> rename(columns: {_value: "type", _field: "leaks"})', color=CRITICAL), 12, 7)
    d.add(barchart("Types provided by users",
                   f'{runs} |> filter(fn: (r) => r._field == "pii_input_types" and r._value != "none") |> group(columns: ["_value"]) |> count(column: "_field")'
                   ' |> group() |> rename(columns: {_value: "type", _field: "runs"})', color=SERIES[0]), 12, 7)
    return d


def capture(bucket):
    d = Dashboard("pulsar-capture", "Pulsar.ml — Capture health",
                  "Completeness of the non-blocking capture: written, spooled, replayed and dropped rows, errors, storage availability",
                  [var_bucket(bucket)] + model_vars())
    d.text("The capture never slows down predictions: rows are queued and written by a background thread. When the database is "
           "unavailable they are **spooled** to disk and replayed; only without a spool directory, or when the queue is full, are "
           "they **dropped** and counted.", h=2)
    s = f'{STATS} |> filter(fn: (r) => r.model_id == "${{model}}")'
    for title, field, steps in (
        ("Captured rows", "captured_rows", None), ("Written rows", "written_rows", None),
        ("Spooled rows", "spooled_rows", [{"color": GOOD, "value": None}, {"color": WARNING, "value": 1}]),
        ("Dropped rows", "dropped_rows", [{"color": GOOD, "value": None}, {"color": CRITICAL, "value": 1}]),
        ("Write errors", "write_errors", [{"color": GOOD, "value": None}, {"color": WARNING, "value": 1}]),
    ):
        d.add(stat(title, f'{s} |> filter(fn: (r) => r._field == "{field}") |> last()', unit="short", thresholds=steps), 4 if field != "write_errors" else 4, 4)
    d.add(stat("Completeness", f'{s} |> filter(fn: (r) => r._field == "written_rows" or r._field == "captured_rows") |> last() |> group()'
               ' |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value") |> map(fn: (r) => ({_time: r._time, _value: r.written_rows / r.captured_rows}))',
               unit="percentunit", decimals=2, thresholds=[{"color": CRITICAL, "value": None}, {"color": WARNING, "value": 0.99}, {"color": GOOD, "value": 0.9999}]), 4, 4)
    d.add(timeseries("Rows per hour: captured and written", [
        f'{s} |> filter(fn: (r) => r._field == "captured_rows") |> derivative(unit: 1h, nonNegative: true) |> set(key: "s", value: "captured")',
        f'{s} |> filter(fn: (r) => r._field == "written_rows") |> derivative(unit: 1h, nonNegative: true) |> set(key: "s", value: "written")',
    ], unit="short", display="${__field.labels.s}", fill=0, overrides=[color("captured", SERIES[0]), color("written", GOOD)]), 12, 8)
    d.add(timeseries("Rows not yet written: queued, spooled, dropped", [
        f'{s} |> filter(fn: (r) => r._field == "{f}") |> set(key: "s", value: "{f}")' for f in ("queued_rows", "spooled_rows", "dropped_rows")
    ], unit="short", display="${__field.labels.s}", fill=0,
        overrides=[color("queued_rows", SERIES[0]), color("spooled_rows", WARNING), color("dropped_rows", CRITICAL)]), 12, 8)
    d.add(status_history("Storage available",
                         f'{s} |> filter(fn: (r) => r._field == "storage_available") |> toInt() |> group() |> set(key: "_field", value: "storage")'
                         ' |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")',
                         [{"type": "value", "options": {"1": {"text": "✓ Available", "color": GOOD}, "0": {"text": "▲ Unavailable", "color": CRITICAL}}}]), 24, 4)
    return d


BUILDERS = [overview, drift, performance, agents, fairness, explain, privacy, capture]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="grafana/dashboards")
    parser.add_argument("--bucket", default="monitoring")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for build in BUILDERS:
        dashboard = build(args.bucket)
        path = out / f"{dashboard.uid}.json"
        path.write_text(json.dumps(dashboard.json([]), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{path.name}: {len([p for p in dashboard.panels if p['type'] != 'row'])} panels")


if __name__ == "__main__":
    main()
