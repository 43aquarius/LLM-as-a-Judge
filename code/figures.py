#!/usr/bin/env python3
"""Generate publication figures (vector PDF) from results.json + judgments.

Follows skill guidance: clean academic style, low saturation, colorblind-safe-ish
family palette, no chartjunk, all text in English (paper language).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm

for fp in [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

RESEARCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures")
os.makedirs(FIGDIR, exist_ok=True)

JUDGE_ORDER = [
    "Qwen3.7-Flash", "Qwen3.8-Flash", "Qwen3.8-27B", "Qwen3.8-Max",
    "Qwen3.8-2.4T", "DeepSeek-V4-Flash", "DeepSeek-V4.1-Flash",
    "DeepSeek-V4-Pro", "GLM-5.3", "Kimi-K3",
    "Agnes-2.5-Flash", "Agnes-3.0-Flash",
]
SHORT = {
    "Qwen3.7-Flash": "Qw3.7-Fl", "Qwen3.8-Flash": "Qw3.8-Fl", "Qwen3.8-27B": "Qw3.8-27B",
    "Qwen3.8-Max": "Qw3.8-Max", "Qwen3.8-2.4T": "Qw3.8-2.4T",
    "DeepSeek-V4-Flash": "DS-V4-Fl", "DeepSeek-V4.1-Flash": "DS-V4.1-Fl",
    "DeepSeek-V4-Pro": "DS-V4-Pro", "GLM-5.3": "GLM-5.3", "Kimi-K3": "Kimi-K3",
    "Agnes-2.5-Flash": "Ag2.5-Fl", "Agnes-3.0-Flash": "Ag3.0-Fl",
}
FAMILY = {
    "Qwen3.7-Flash": "Qwen", "Qwen3.8-Flash": "Qwen", "Qwen3.8-27B": "Qwen",
    "Qwen3.8-Max": "Qwen", "Qwen3.8-2.4T": "Qwen",
    "DeepSeek-V4-Flash": "DeepSeek", "DeepSeek-V4.1-Flash": "DeepSeek",
    "DeepSeek-V4-Pro": "DeepSeek", "GLM-5.3": "GLM", "Kimi-K3": "Kimi",
    "Agnes-2.5-Flash": "Agnes", "Agnes-3.0-Flash": "Agnes",
}
FC = {"Qwen": "#5B8DB8", "DeepSeek": "#4C9A8A", "GLM": "#B3748A", "Kimi": "#C89442", "Agnes": "#8B7FC7"}
FAMILY_TIERS = {
    "Qwen": ["Qwen3.7-Flash", "Qwen3.8-Flash", "Qwen3.8-27B", "Qwen3.8-Max", "Qwen3.8-2.4T"],
    "DeepSeek": ["DeepSeek-V4-Flash", "DeepSeek-V4.1-Flash", "DeepSeek-V4-Pro"],
    "GLM": ["GLM-5.3"],
    "Kimi": ["Kimi-K3"],
    "Agnes": ["Agnes-2.5-Flash", "Agnes-3.0-Flash"],
}


def load_jsonl(p):
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def half_width(ci, point_on_ci_scale):
    """Half-CI widths from a [lo, hi] interval; clamped at 0 for degenerate cases."""
    lo, hi = ci[0], ci[1]
    return (max(0.0, point_on_ci_scale - lo), max(0.0, hi - point_on_ci_scale))


def errbar(ax, xs, ys, cis, colors, width=0.62):
    lo = [c[0] for c in cis]
    hi = [c[1] for c in cis]
    ax.bar(xs, ys, width=width, color=colors, edgecolor="black", linewidth=0.5, zorder=3)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=0.8,
                capsize=2.5, zorder=4)


def main():
    R = json.load(open(os.path.join(RESEARCH, "results.json")))
    judges = R["judges"]
    N = len(judges)
    colors = [FC[FAMILY[j]] for j in judges]
    labels = [SHORT[j] for j in judges]
    xs = np.arange(N)

    # ---------- Fig 1: kappa heatmap ----------
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    M = np.full((N, N), np.nan)
    for key, v in R["kappa_matrix"].items():
        a, b = key.split("||")
        if a in judges and b in judges:
            ia, ib = judges.index(a), judges.index(b)
            M[ia, ib] = v; M[ib, ia] = v
    np.fill_diagonal(M, 1.0)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels([SHORT[j] for j in judges], rotation=45, ha="right")
    ax.set_yticklabels([SHORT[j] for j in judges])
    fam_seq = [FAMILY[j] for j in judges]
    for i in range(1, N):
        if fam_seq[i] != fam_seq[i - 1]:
            ax.axhline(i - 0.5, color="black", lw=1.0)
            ax.axvline(i - 0.5, color="black", lw=1.0)
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("Cohen's $\\kappa$")
    ax.grid(False)
    fig.savefig(os.path.join(FIGDIR, "fig_agreement.pdf"))
    plt.close(fig)

    # ---------- Fig 2: bias trio (PB, flip, length) ----------
    # NOTE: position_bias/length_bias store CIs on the RATE scale while the
    # plotted value is bias = rate - 0.5; CI half-widths are shift-invariant.
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.2), sharex=True, constrained_layout=True)

    pb = [R["position_bias"][j]["bias"] for j in judges]
    pbci = [half_width(R["position_bias"][j]["ci"], R["position_bias"][j]["first_rate"])
            for j in judges]
    errbar(axes[0], xs, pb, pbci, colors)
    axes[0].axhline(0, color="black", lw=0.8, linestyle="--", zorder=2)
    axes[0].set_ylabel("Position bias\n(first-win rate $-$ 0.5)")
    axes[0].set_title("Position bias and order consistency", fontsize=10)

    fl = [R["flip_rate"][j]["flip_rate"] for j in judges]
    flci = [half_width(R["flip_rate"][j]["ci"], R["flip_rate"][j]["flip_rate"])
            for j in judges]
    errbar(axes[1], xs, fl, flci, colors)
    axes[1].set_ylabel("Order-swap\nflip rate")

    lb = [R["length_bias"][j]["bias"] for j in judges]
    lbci = [half_width(R["length_bias"][j]["ci"], R["length_bias"][j]["longer_win_rate"])
            for j in judges]
    errbar(axes[2], xs, lb, lbci, colors)
    axes[2].axhline(0, color="black", lw=0.8, linestyle="--", zorder=2)
    axes[2].set_ylabel("Length bias\n(longer-win rate $-$ 0.5)")
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(labels, rotation=45, ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=FC[f]) for f in ["Qwen", "DeepSeek", "GLM", "Kimi", "Agnes"]]
    axes[1].legend(handles, ["Qwen", "DeepSeek", "GLM", "Kimi", "Agnes (proprietary)"], ncol=5,
                   loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=7.5)
    fig.savefig(os.path.join(FIGDIR, "fig_bias.pdf"))
    plt.close(fig)

    # ---------- Fig 3: accuracy ----------
    fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    acc_judges = [j for j in judges if j in R["gt_accuracy"]]
    acc = [R["gt_accuracy"][j]["acc"] for j in acc_judges]
    accci = [half_width(R["gt_accuracy"][j]["ci"], R["gt_accuracy"][j]["acc"])
             for j in acc_judges]
    xs_acc = np.arange(len(acc_judges))
    colors_acc = [FC[FAMILY[j]] for j in acc_judges]
    labels_acc = [SHORT[j] for j in acc_judges]
    errbar(ax, xs_acc, acc, accci, colors_acc)
    ax.axhline(0.5, color="black", lw=0.8, linestyle="--", zorder=2)
    ax.set_xticks(xs_acc); ax.set_xticklabels(labels_acc, rotation=45, ha="right")
    ax.set_ylabel("Accuracy on GT-clear pairs")
    ax.set_ylim(0.0, 1.02)
    fig.savefig(os.path.join(FIGDIR, "fig_accuracy.pdf"))
    plt.close(fig)

    # ---------- Fig 4: kin preference (open judges only; Agnes has no kin generators) ----------
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    open_judges = [j for j in judges if FAMILY[j] != "Agnes"]
    xs_o = np.arange(len(open_judges))
    colors_o = [FC[FAMILY[j]] for j in open_judges]
    labels_o = [SHORT[j] for j in open_judges]
    kw, kwci, kwn = [], [], []
    for j in open_judges:
        k = R["family_preference"][j]
        v = k["kin_wrong_pick_rate"]
        if v is None or k["kin_wrong_ci"] is None:
            kw.append(0.0); kwci.append((0, 0)); kwn.append(0)
        else:
            kw.append(v)
            kwci.append(half_width(k["kin_wrong_ci"], v))
            kwn.append(k["kin_wrong_n"])
    errbar(axes[0], xs_o, kw, kwci, colors_o)
    axes[0].set_ylabel("Kin-wrong pick rate")
    axes[0].set_title("Judge picks its own family's answer\nwhen the reference proves it wrong", fontsize=9)
    axes[0].set_xticks(xs_o); axes[0].set_xticklabels(labels_o, rotation=45, ha="right")

    kwin = []
    for j in open_judges:
        v = R["family_preference"][j]["kin_win_rate"]
        kwin.append(v if v is not None else 0.0)
    axes[1].bar(xs_o, kwin, width=0.62, color=colors_o, edgecolor="black", linewidth=0.5, zorder=3)
    axes[1].axhline(0.5, color="black", lw=0.8, linestyle="--", zorder=2)
    axes[1].set_ylabel("Kin win rate (non-tie kin pairs)")
    axes[1].set_title("Overall preference for same-family\nanswers among non-tie verdicts", fontsize=9)
    axes[1].set_xticks(xs_o); axes[1].set_xticklabels(labels_o, rotation=45, ha="right")
    fig.savefig(os.path.join(FIGDIR, "fig_kin.pdf"))
    plt.close(fig)

    # ---------- Fig 5: within-family scale trends (open families with >=2 tiers) ----------
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), constrained_layout=True)
    for fam, tiers in FAMILY_TIERS.items():
        if len(tiers) < 2 or fam == "Agnes":
            continue
        present = [t for t in tiers if t in R["gt_accuracy"] and t in R["position_bias"] and t in R["flip_rate"]]
        if len(present) < 2:
            continue
        xs2 = range(len(present))
        accs = [R["gt_accuracy"][t]["acc"] for t in present]
        pbs = [abs(R["position_bias"][t]["bias"]) for t in present]
        fls = [R["flip_rate"][t]["flip_rate"] for t in present]
        axes[0].plot(xs2, accs, "o-", color=FC[fam], label=fam, lw=1.2, ms=4)
        axes[1].plot(xs2, pbs, "o-", color=FC[fam], label=fam, lw=1.2, ms=4)
        axes[2].plot(xs2, fls, "o-", color=FC[fam], label=fam, lw=1.2, ms=4)
    axes[0].set_ylabel("GT accuracy")
    axes[1].set_ylabel("|Position bias|")
    axes[2].set_ylabel("Flip rate")
    for a in axes:
        a.set_xlabel("Tier index within family")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(os.path.join(FIGDIR, "fig_scale.pdf"))
    plt.close(fig)

    # ---------- Fig 6: panels ----------
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), constrained_layout=True)
    ks = [1, 3, 5]
    acc_means = [R["single_judge_acc_mean"], R["panels"]["k3"]["panel_acc_mean"], R["panels"]["k5"]["panel_acc_mean"]]
    acc_los = [np.nan, R["panels"]["k3"]["panel_acc_ci"][0], R["panels"]["k5"]["panel_acc_ci"][0]]
    acc_his = [np.nan, R["panels"]["k3"]["panel_acc_ci"][1], R["panels"]["k5"]["panel_acc_ci"][1]]
    pb_means = [R["single_judge_abs_pb_mean"], R["panels"]["k3"]["panel_pb_mean"], R["panels"]["k5"]["panel_pb_mean"]]
    pb_los = [np.nan, R["panels"]["k3"]["panel_pb_ci"][0], R["panels"]["k5"]["panel_pb_ci"][0]]
    pb_his = [np.nan, R["panels"]["k3"]["panel_pb_ci"][1], R["panels"]["k5"]["panel_pb_ci"][1]]
    for ax, m, lo, hi, yl in [
        (axes[0], acc_means, acc_los, acc_his, "Panel accuracy (GT-clear)"),
        (axes[1], pb_means, pb_los, pb_his, "Mean |position bias|"),
    ]:
        ax.plot(ks, m, "o-", color="#444444", lw=1.4, ms=5)
        for k, v, l, h in zip(ks, m, lo, hi):
            if not np.isnan(l):
                ax.errorbar([k], [v], yerr=[[max(0.0, v - l)], [max(0.0, h - v)]], fmt="none", ecolor="black",
                            elinewidth=0.8, capsize=2.5)
        ax.set_xticks(ks)
        ax.set_xticklabels(["single", "panel of 3", "panel of 5"])
        ax.set_xlabel("Judging configuration")
        ax.set_ylabel(yl)
    fig.savefig(os.path.join(FIGDIR, "fig_panel.pdf"))
    plt.close(fig)

    print("figures saved to", FIGDIR)
    for f in sorted(os.listdir(FIGDIR)):
        print(" -", f)


if __name__ == "__main__":
    main()
