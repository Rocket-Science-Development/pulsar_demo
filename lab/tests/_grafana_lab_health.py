"""Create the InfluxDB (Flux) datasource and the 'Lab health' dashboard in the lab Grafana; reads the env file, prints no secret."""
import base64, json, urllib.request
env = dict(l.split("=", 1) for l in open("/etc/pulsar-lab/influx.env").read().split("\n") if "=" in l)
auth = base64.b64encode(f"admin:{env['GRAFANA_ADMIN_PASSWORD'].strip()}".encode()).decode()
def call(method, path, body=None):
    req = urllib.request.Request("http://127.0.0.1:3000" + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": "Basic " + auth, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
st, ds = call("GET", "/api/datasources/name/lab-influx")
if st != 200:
    st, ds = call("POST", "/api/datasources", {"name": "lab-influx", "type": "influxdb", "access": "proxy",
        "url": (env.get("LAB_INFLUX_URL") or env["INFLUX_URL"]).strip(), "jsonData": {"version": "Flux", "organization": env["INFLUX_ORG"].strip(),
        "defaultBucket": "lab_health"}, "secureJsonData": {"token": env["INFLUX_TOKEN"].strip()}})
    ds = ds.get("datasource", ds)
uid = ds["uid"]
def panel(i, title, flux, unit=None, thresholds=None):
    p = {"id": i, "type": "timeseries", "title": title, "datasource": {"type": "influxdb", "uid": uid},
         "gridPos": {"x": (i - 1) % 2 * 12, "y": (i - 1) // 2 * 8, "w": 12, "h": 8},
         "targets": [{"refId": "A", "query": flux, "datasource": {"type": "influxdb", "uid": uid}}],
         "fieldConfig": {"defaults": {"unit": unit or "short"}, "overrides": []}}
    if thresholds:
        p["fieldConfig"]["defaults"]["thresholds"] = {"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "red", "value": thresholds}]}
        p["fieldConfig"]["defaults"]["custom"] = {"thresholdsStyle": {"mode": "line"}}
    return p
host = lambda f: f'from(bucket: "lab_health") |> range(start: v.timeRangeStart, stop: v.timeRangeStop) |> filter(fn: (r) => r._measurement == "lab_host" and r._field == "{f}") |> aggregateWindow(every: v.windowPeriod, fn: mean)'
panels = [panel(1, "Host CPU %", host("cpu_pct"), "percent", 90), panel(2, "Host memory %", host("mem_pct"), "percent", 90),
          panel(3, "Host disk %", host("disk_pct"), "percent", 80), panel(4, "Network TX MB/s", host("net_tx_mb_s")),
          panel(5, "InfluxDB container memory (MB)", 'from(bucket: "lab_health") |> range(start: v.timeRangeStart, stop: v.timeRangeStop) |> filter(fn: (r) => r._measurement == "lab_host" and r._field =~ /^docker_influxdb.*_mem_mb$/)'),
          panel(6, "Pulse capture counters", 'from(bucket: "lab_health") |> range(start: v.timeRangeStart, stop: v.timeRangeStop) |> filter(fn: (r) => r._measurement == "pulsar_capture_stats" and (r._field == "queued_rows" or r._field == "dropped_rows"))')]
st, out = call("POST", "/api/dashboards/db", {"overwrite": True, "dashboard": {"uid": "lab-health", "title": "Lab health",
    "tags": ["pulsar-lab"], "time": {"from": "now-12h", "to": "now"}, "panels": panels, "schemaVersion": 39}})
print(json.dumps({"datasource_uid": uid, "dashboard_status": st, "url": out.get("url")}))
