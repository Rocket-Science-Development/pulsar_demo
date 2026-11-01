"""Draw the Pulsar.ml architecture diagram (images/pulsarml-architecture.png) with the pulsar.ml logo.

Usage: python images/draw_architecture.py images [fonts_dir]
Requires Pillow and the Segoe UI / Consolas fonts (Windows); pass another fonts directory as second argument.
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMAGES = Path(sys.argv[1])
OUT = IMAGES / "pulsarml-architecture.png"
S = 2  # supersampling factor, downscaled at the end for smooth edges
W, H = 2400, 1500

INDIGO = (30, 12, 110)
BLUE = (64, 160, 230)
LIGHT_BLUE = (232, 243, 253)
LAVENDER = (240, 237, 252)
GREY = (90, 90, 110)
LIGHT_GREY = (246, 246, 249)
GREEN = (226, 245, 236)
ORANGE = (253, 239, 224)
WHITE = (255, 255, 255)

FONTS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("C:/Windows/Fonts")


def font(name, size):
    return ImageFont.truetype(str(FONTS / name), size * S)


TITLE = font("segoeuib.ttf", 26)
BIG = font("segoeuib.ttf", 40)
BODY = font("segoeui.ttf", 20)
SMALL = font("segoeui.ttf", 17)
MONO = font("consola.ttf", 17)
LABEL = font("segoeuib.ttf", 18)

img = Image.new("RGB", (W * S, H * S), WHITE)
d = ImageDraw.Draw(img)


def p(*xy):
    return [v * S for v in xy]


def wrap(text, fnt, width):
    words, lines, line = text.split(), [], ""
    for w in words:
        test = f"{line} {w}".strip()
        if fnt.getlength(test) <= width * S or not line:
            line = test
        else:
            lines.append(line)
            line = w
    lines.append(line)
    return lines


def text_center(x, y, text, fnt, fill=INDIGO):
    d.text((x * S, y * S), text, font=fnt, fill=fill, anchor="mm")


def box(x0, y0, x1, y1, title, lines=(), fill=LAVENDER, outline=INDIGO, width=2, radius=18, dashed=False, mono=False):
    if dashed:
        dashed_rect(x0, y0, x1, y1, outline, radius)
    else:
        d.rounded_rectangle(p(x0, y0, x1, y1), radius=radius * S, fill=fill, outline=outline, width=width * S)
    cx = (x0 + x1) / 2
    items = []  # (text, font, color, advance)
    if title:
        items += [(t, TITLE, INDIGO, 32) for t in wrap(title, TITLE, x1 - x0 - 30)]
        items[-1] = (*items[-1][:3], 38)
    for line in lines:
        fnt = MONO if mono and line.startswith("`") else BODY
        items += [(t, fnt, GREY, 26) for t in wrap(line.strip("`"), fnt, x1 - x0 - 30)]
    total = sum(a for *_, a in items) - (items[-1][3] if items else 0)
    y = (y0 + y1) / 2 - total / 2
    for t, fnt, color, advance in items:
        text_center(cx, y, t, fnt, color)
        y += advance


def dashed_rect(x0, y0, x1, y1, color, radius=18, dash=14, gap=10):
    d.rounded_rectangle(p(x0, y0, x1, y1), radius=radius * S, fill=LIGHT_GREY, outline=None)
    for xa, ya, xb, yb in [(x0 + radius, y0, x1 - radius, y0), (x0 + radius, y1, x1 - radius, y1)]:
        x = xa
        while x < xb:
            d.line(p(x, ya, min(x + dash, xb), yb), fill=color, width=2 * S)
            x += dash + gap
    for xa, ya, xb, yb in [(x0, y0 + radius, x0, y1 - radius), (x1, y0 + radius, x1, y1 - radius)]:
        y = ya
        while y < yb:
            d.line(p(xa, y, xb, min(y + dash, yb)), fill=color, width=2 * S)
            y += dash + gap


def container(x0, y0, x1, y1, title, fill, subtitle=None):
    d.rounded_rectangle(p(x0, y0, x1, y1), radius=24 * S, fill=fill, outline=INDIGO, width=3 * S)
    d.rounded_rectangle(p(x0, y0, x1, y0 + 56), radius=24 * S, fill=INDIGO)
    d.rectangle(p(x0, y0 + 30, x1, y0 + 56), fill=INDIGO)
    text_center((x0 + x1) / 2, y0 + 28, title, TITLE, WHITE)
    if subtitle:
        text_center((x0 + x1) / 2, y0 + 78, subtitle, SMALL, GREY)


def arrow(x0, y0, x1, y1, label=None, color=INDIGO, dashed=False, label_dy=-16, both=False, head=True):
    if dashed:
        n = max(1, int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 // 18))
        for i in range(0, n, 2):
            a, b = i / n, min(1, (i + 1) / n)
            d.line(p(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, x0 + (x1 - x0) * b, y0 + (y1 - y0) * b), fill=color, width=3 * S)
    else:
        d.line(p(x0, y0, x1, y1), fill=color, width=3 * S)
    heads = ([(x0, y0, x1, y1)] if head else []) + ([(x1, y1, x0, y0)] if both else [])
    for ax, ay, bx, by in heads:
        import math

        ang = math.atan2(by - ay, bx - ax)
        size = 16
        pts = [
            (bx, by),
            (bx - size * math.cos(ang - 0.4), by - size * math.sin(ang - 0.4)),
            (bx - size * math.cos(ang + 0.4), by - size * math.sin(ang + 0.4)),
        ]
        d.polygon([(px * S, py * S) for px, py in pts], fill=color)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + label_dy
        tw = LABEL.getlength(label) / S
        d.rounded_rectangle(p(mx - tw / 2 - 8, my - 15, mx + tw / 2 + 8, my + 15), radius=8 * S, fill=WHITE)
        text_center(mx, my, label, LABEL, color)


def cylinder(cx, cy, w, h, title, lines):
    x0, x1 = cx - w / 2, cx + w / 2
    top, bottom, e = cy - h / 2, cy + h / 2, 26
    d.rectangle(p(x0, top + e / 2, x1, bottom - e / 2), fill=LIGHT_BLUE)
    d.ellipse(p(x0, bottom - e, x1, bottom), fill=LIGHT_BLUE, outline=INDIGO, width=2 * S)
    d.rectangle(p(x0 + 1, top + e / 2, x1 - 1, bottom - e / 2), fill=LIGHT_BLUE)
    d.line(p(x0, top + e / 2, x0, bottom - e / 2), fill=INDIGO, width=2 * S)
    d.line(p(x1, top + e / 2, x1, bottom - e / 2), fill=INDIGO, width=2 * S)
    d.ellipse(p(x0, top, x1, top + e), fill=WHITE, outline=INDIGO, width=2 * S)
    y = cy - 30
    text_center(cx, y, title, TITLE)
    for line in lines:
        y += 28
        text_center(cx, y, line, BODY, GREY)


# ------------------------------------------------------------------ header
logo = Image.open(IMAGES / "pulsarml-logo.png").convert("RGBA")
logo = logo.crop(logo.getbbox())
lw = 360 * S
logo = logo.resize((lw, int(logo.height * lw / logo.width)), Image.LANCZOS)
img.paste(logo, (50 * S, 38 * S), logo)
d.text((450 * S, 62 * S), "Architecture — model, LLM and agent monitoring", font=BIG, fill=INDIGO)
d.text((452 * S, 112 * S), "Open-source Python packages · Apache 2.0 · Python 3.12+", font=BODY, fill=GREY)
d.line(p(50, 160, W - 50, 160), fill=BLUE, width=4 * S)

# ------------------------------------------------------------------ sources
text_center(250, 200, "SOURCES", LABEL, BLUE)
box(60, 230, 440, 450, "Classic ML model", ["Inference service or notebook", "features · predictions", "ground truth when known"], fill=WHITE)
box(60, 560, 440, 900, "LLM application or agent", [
    "OpenAI-compatible model (vLLM, Ollama)", "tools · MCP · sub-agents", "OpenTelemetry GenAI spans",
    "`invoke_agent · chat · execute_tool`"], fill=WHITE, mono=True)

# ------------------------------------------------------------------ pulsar-data-collection
container(540, 190, 1080, 1060, "pulsar-data-collection", LAVENDER, "collect · featurize · store")
box(570, 290, 1050, 520, "Pulse — non-blocking capture", [
    "bounded queue + writer thread", "≈1 ms per capture, never blocks predictions", "circuit breaker · stats() completeness"])
box(570, 600, 1050, 830, "genai · PulsarSpanExporter", [
    "traces → agent_run · tool_call · llm_call", "steps, tools, loops, out-of-scope, tokens…", "features only: no prompts stored"])
box(570, 900, 1050, 1030, "Spool on disk", ["batches kept while the database is down,", "replayed when it is back"], fill=ORANGE)

arrow(440, 360, 570, 400, "capture_data()", label_dy=-34)
arrow(440, 730, 570, 715, "spans", label_dy=-20)
arrow(810, 600, 810, 520, "PulseSink", label_dy=0)
# Pulse <-> spool, routed in the container margin so that it does not cross the genai box
arrow(553, 470, 570, 470, None, color=GREY)
arrow(553, 470, 553, 965, None, color=GREY, dashed=True, head=False)
arrow(553, 965, 570, 965, None, color=GREY)

# ------------------------------------------------------------------ InfluxDB
cylinder(1200, 470, 190, 300, "InfluxDB", ["2.x", "time series", "tags · fields"])
arrow(1050, 405, 1105, 430, None)

# ------------------------------------------------------------------ pulsar-metrics
container(1320, 190, 1880, 1060, "pulsar-metrics", GREEN, "compare windows · decide")
box(1350, 290, 1850, 540, "Analyzer", [
    "drift: psi · kl · wasserstein · mmd", "tests: KS · t · Mann-Whitney · Levene · CvM · χ²",
    "performance · summary statistics", "reference window vs current window"], fill=WHITE)
box(1350, 580, 1850, 730, "Calibrated thresholds", ["A/A null distributions per feature", "Benjamini-Hochberg (FDR)"], fill=WHITE)
box(1350, 770, 1850, 900, "DriftPersistence", ["confirm a drift over N windows"], fill=WHITE)
box(1350, 940, 1850, 1035, None, ["custom metrics · explicit errors"], fill=WHITE)
arrow(1295, 450, 1350, 420, None)
arrow(1600, 540, 1600, 580, None)
arrow(1600, 730, 1600, 770, None)

# ------------------------------------------------------------------ outputs
text_center(2150, 200, "OUTPUTS", LABEL, BLUE)
box(1960, 240, 2340, 400, "Dashboards", ["Grafana: drift by family,", "performance, completeness"], fill=WHITE)
box(1960, 450, 2340, 640, "Alerts & re-evaluation", ["drift_status · confirmed changes", "webhooks · events"], fill=WHITE)
box(1960, 690, 2340, 850, "Reports", ["JSON / pandas results,", "reproducible evidence"], fill=WHITE)
arrow(1880, 380, 1960, 330)
arrow(1880, 600, 1960, 545)
arrow(1880, 800, 1960, 770)

# ------------------------------------------------------------------ demos
dashed_rect(60, 1120, 2340, 1420, INDIGO)
text_center(1200, 1152, "pulsar_demo  ·  Docker Compose", TITLE)
box(100, 1190, 1180, 1395, "genai_agent — tool-calling agent (recommended)", [
    "public-services assistant, 5 tools, French and English · llama3.2 / granite3.3",
    "drift scenarios: model change · tool fault · question shift · prompt change"], fill=WHITE)
box(1220, 1190, 2300, 1395, "Classic demo (legacy)", [
    "tabular model, synthetic data and drift simulation, cron metrics, Grafana",
    "frozen on the 2023 versions (Python 3.9, InfluxDB 1.5)"], fill=WHITE)

d.text((60 * S, 1450 * S), "Rocket Science Development · github.com/Rocket-Science-Development", font=SMALL, fill=GREY)

img = img.resize((W, H), Image.LANCZOS)
img.save(OUT, optimize=True)
print(OUT, img.size)
