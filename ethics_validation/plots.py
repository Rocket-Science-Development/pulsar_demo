"""Figures of the E5 validation: power curves by sample size, one line per injected bias, false alarms in gray.

    python plots.py --results results --sequential-json bench_synthetic.json bench_folktables.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#1f1f1e"
INK_MUTED = "#6b6a63"
GRID = "#e6e5e0"
NULL = "#8a8980"  # bias 0: false alarms
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # one hue, light -> dark = small -> large bias


def style(ax, title, xlabel, ylabel="Share of runs"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=24)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


CATEGORICAL = ["#2a78d6", "#eb6834"]  # two different measures (not ordered): slots 1 and 2


def _dodge(ys, gap=0.07):
    """End-label positions: same order as the line ends, at least `gap` apart"""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed = [0.0] * len(ys)
    last = -1.0
    for i in order:
        placed[i] = max(ys[i], last + gap)
        last = placed[i]
    return placed


def lines(ax, series, x_log=True, categorical=False):
    """series: list of (label, xs, ys, is_null); null series dashed gray, the others on the ramp (ordered biases) or on
    the categorical slots (different measures); each line is labelled at its end, labels never overlap"""
    biased = [s for s in series if not s[3]]
    colors = CATEGORICAL if categorical else (RAMP[-len(biased):] if len(biased) <= len(RAMP) else RAMP)
    k = 0
    ends = []
    for label, xs, ys, is_null in series:
        color = NULL if is_null else colors[k]
        k += 0 if is_null else 1
        ax.plot(xs, ys, color=color, linewidth=2, linestyle="--" if is_null else "-", marker="o", markersize=4,
                label=label, solid_capstyle="round")
        ends.append((label, xs[-1], ys[-1]))
    for (label, x, _), y in zip(ends, _dodge([e[2] for e in ends])):
        ax.annotate(label, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=INK,
                    annotation_clip=False)
    if x_log:
        ax.set_xscale("log")
        ax.set_xticks(series[0][1])
        ax.set_xticklabels([f"{x:,}".replace(",", " ") for x in series[0][1]])
        ax.minorticks_off()
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=len(series),
              handlelength=1.8, columnspacing=1.2)


def figure(path, panels, width=11):
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 4.2), facecolor=SURFACE)
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, draw in zip(axes, panels):
        draw(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fixed_figure(data, out):
    cells = data["cells"]
    sizes = sorted({c["size"] for c in cells})

    def get(delta, key):
        return [next(c[key] for c in cells if c["size"] == s and c["delta"] == delta) for s in sizes]

    def established(ax):
        series = [("no bias (false alarms)", sizes, get(0.0, "established"), True)]
        series += [(f"bias −{d:.2f}", sizes, get(d, "established"), False) for d in (0.15, 0.2, 0.3)]
        lines(ax, series)
        style(ax, "Disparity established", "Size of the minority group (reference: 2 000)")

    def within(ax):
        series = [("no bias", sizes, get(0.0, "within"), False), ("bias −0.05", sizes, get(0.05, "within"), False)]
        lines(ax, series, categorical=True)
        style(ax, "Within the zone ±0.10 certified", "Size of the minority group (reference: 2 000)")

    figure(out / "fig_fixed.png", [established, within])


def probe_figure(data, out):
    cells = data["cells"]
    sizes = sorted({c["sets"] for c in cells})

    def get(delta, measure, key):
        return [next(c[measure][key] for c in cells if c["sets"] == s and c["delta"] == delta) for s in sizes]

    def action(ax):
        series = [("no bias (false alarms)", sizes, get(0.0, "action", "established"), True)]
        series += [(f"bias +{d:.2f}", sizes, get(d, "action", "established"), False) for d in (0.2, 0.33)]
        lines(ax, series)
        style(ax, "Action-rate disparity established", "Number of sets per group")

    def absence(ax):
        series = [
            ("action rate within ±0.10", sizes, get(0.0, "action", "within"), False),
            ("excess flip ≤ 0.05", sizes, get(0.0, "excess_flip", "within"), False),
        ]
        lines(ax, series, categorical=True)
        style(ax, "No bias: absence certified", "Number of sets per group")

    figure(out / "fig_probe.png", [action, absence])


def explain_figure(data, out):
    cells = data["cells"]
    sizes = sorted({c["window"] for c in cells})
    shift = data["sex_share_shift_full_mix"]

    def get(mix):
        return [next(c["sex_share"]["established"] for c in cells if c["window"] == s and c["mix"] == mix) for s in sizes]

    def draw(ax):
        alarms = [next(c["any_alarm"] for c in cells if c["window"] == s and c["mix"] == 0.0) for s in sizes]
        series = [("no change (any alarm)", sizes, alarms, True)]
        series += [(f"+{100 * m * shift:.1f} pp", sizes, get(m), False) for m in (0.25, 0.5, 1.0)]
        lines(ax, series)
        style(ax, "Change of the weight of SEX established", "Window size (rows), ACSIncome")

    figure(out / "fig_explain.png", [draw], width=6.5)


def sequential_figure(paths, out):
    def panel(path, title):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cells = data["cells"]
        deltas = sorted({c["delta"] for c in cells})

        def draw(ax):
            shares = sorted({c["share"] for c in cells})
            series = []
            for share in shares:
                ys = [next(c["monitor"]["power"] for c in cells if c["share"] == share and c["delta"] == d) for d in deltas]
                series.append((f"group {share:.0%}", deltas, ys, False))
            lines(ax, series, x_log=False)
            style(ax, title, "Injected drop of the selection rate")

        return draw

    titles = ["Anytime-valid monitor, synthetic", "Anytime-valid monitor, ACSIncome"]
    figure(out / "fig_sequential.png", [panel(p, t) for p, t in zip(paths, titles)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    parser.add_argument("--sequential-json", nargs="*", default=[])
    args = parser.parse_args()
    out = Path(args.results)
    for name, draw in (("fixed", fixed_figure), ("probe", probe_figure), ("explain", explain_figure)):
        path = out / f"power_{name}.json"
        if path.exists():
            draw(json.loads(path.read_text(encoding="utf-8")), out)
            print(f"fig_{name}.png")
    if args.sequential_json:
        sequential_figure(args.sequential_json, out)
        print("fig_sequential.png")


if __name__ == "__main__":
    main()
