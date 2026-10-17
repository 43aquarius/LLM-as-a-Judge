#!/usr/bin/env python3
"""E7 analysis: platform replication + GLM gap-fill MCAR verification + cross-platform
retest + premium-anchor pilot. Self-contained; kappa/Wilson semantics identical to
reanalysis.py. Handles partial coverage honestly (n's and Wilson CIs everywhere).

Inputs (research/): judgments.jsonl, judgments_agnes.jsonl, retests.jsonl,
  e7_platform_check.jsonl, e7_glm_fill.jsonl, e7_glm_retest.jsonl, judgments_e7.jsonl,
  e7_determinism_v2.json, pairs.jsonl
Outputs: e7_results.json + e7_report.md
"""
import json, math, os
from collections import Counter, defaultdict

import numpy as np

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load(p):
    if not os.path.exists(p):
        return []
    out = []
    for l in open(p):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (None, None)
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def cohen_kappa(a, b, labels=("A", "B", "TIE")):
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    if n == 0:
        return None
    po = float((a == b).mean())
    pe = sum(float((a == l).mean()) * float((b == l).mean()) for l in labels)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def fmt(x, nd=3):
    if x is None:
        return "--"
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def fmt_ci(ci):
    if ci[0] is None:
        return "--"
    return f"[{ci[0]:.3f},{ci[1]:.3f}]"


def two_prop(k1, n1, k2, n2):
    """Normal-approx 2-sided p for difference of proportions (as in reanalysis)."""
    if n1 == 0 or n2 == 0:
        return None
    p1, p2 = k1 / n1, k2 / n2
    pbar = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0 if p1 == p2 else 0.0
    z = (p1 - p2) / se
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


pairs = load(f"{R}/pairs.jsonl")
pair_by_id = {p["pair_id"]: p for p in pairs}
gtclear_ids = {p["pair_id"] for p in pairs if p.get("gt_better") in ("A", "B")}

bail = load(f"{R}/judgments.jsonl")
agnes = load(f"{R}/judgments_agnes.jsonl")
retests = load(f"{R}/retests.jsonl")
check = load(f"{R}/e7_platform_check.jsonl")
fill = load(f"{R}/e7_glm_fill.jsonl")
glm_rt = load(f"{R}/e7_glm_retest.jsonl")
anchor = load(f"{R}/judgments_e7.jsonl")
det = {}
if os.path.exists(f"{R}/e7_determinism_v2.json"):
    det = json.load(open(f"{R}/e7_determinism_v2.json"))

res = {}
md = ["# E7 analysis (platform replication / GLM fill / retest / anchor pilot)",
      f"_generated {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}_", ""]

# ---------- 1. platform replication ----------
md.append("## 1. Cross-platform spot check (same model id, same prompts, temp 0)")
plat = {}
for jn in ("DeepSeek-V4-Flash", "DeepSeek-V4-Pro", "GLM-5.3", "Qwen3.8-27B"):
    rs = [r for r in check if r.get("judge") == jn and r.get("verdict") in ("A", "B", "TIE")]
    n = len(rs)
    agree = sum(1 for r in rs if r["verdict"] == r.get("bailian_verdict"))
    rate = agree / n if n else None
    kap = cohen_kappa([r["verdict"] for r in rs], [r.get("bailian_verdict") for r in rs])
    tie_ia = sum(1 for r in rs if r["verdict"] == "TIE")
    tie_bl = sum(1 for r in rs if r.get("bailian_verdict") == "TIE")
    lat = np.mean([r.get("latency") or 0 for r in rs]) if rs else None
    plat[jn] = {"n": n, "agree": agree, "raw_agreement": rate, "wilson": wilson_ci(agree, n),
                "kappa": kap, "tie_internai": tie_ia, "tie_bailian": tie_bl,
                "tie_rate_ia": tie_ia / n if n else None, "mean_latency": lat}
    md.append(f"- **{jn}**: raw agreement {agree}/{n} = {fmt(rate,3)} {fmt_ci(wilson_ci(agree,n))}, "
              f"kappa {fmt(kap,3)}, tie% IA {fmt(tie_ia/n if n else None,3)} vs Bailian {fmt(tie_bl/n if n else None,3)}, "
              f"latency {fmt(lat,1)}s")
res["platform_replication"] = plat

# ---------- 2. determinism ----------
det_sum = {"cells": len(det), "stable": sum(1 for d in det.values() if d.get("stable"))}
for k, d in det.items():
    det_sum.setdefault("detail", {})[k] = d
res["determinism"] = det_sum
md.append("")
md.append(f"## 2. Determinism (identical call x3, temp 0, high caps)")
md.append(f"- {det_sum['stable']}/{det_sum['cells']} probe cells produced identical verdicts; "
          "exceptions: GLM know-12 (truncation at 1200 cap), Qwen3.8-27B know-12 (B/C wobble, "
          "one length-cut generation).")

# ---------- 3. GLM gap fill + MCAR verification ----------
md.append("")
md.append("## 3. GLM-5.3 gap fill + MCAR verification")
glm_main = [r for r in bail if r.get("judge") == "GLM-5.3" and r.get("verdict") in ("A", "B", "TIE")]
by_cell = {(r["pair_id"], r["order"]): r for r in glm_main}
filled = {(r["pair_id"], r["order"]): r for r in fill if r.get("verdict") in ("A", "B", "TIE")}
missing_total = 126
mc = {}
for tag, recs in (("original", list(by_cell.values())), ("filled", list(filled.values()))):
    n = len(recs)
    tie = sum(1 for r in recs if r["verdict"] == "TIE")
    a_sh = sum(1 for r in recs if r["verdict"] == "A")
    gtc = [r for r in recs if r["pair_id"] in gtclear_ids]
    gtc_n = len(gtc)
    gtc_ok = sum(1 for r in gtc if r["verdict"] == pair_by_id[r["pair_id"]]["gt_better"])
    mc[tag] = {"n": n, "tie": tie, "tie_rate": tie / n if n else None,
               "a_share": a_sh / n if n else None,
               "gtclear_n": gtc_n, "gtclear_acc": gtc_ok / gtc_n if gtc_n else None}
# domain mix of filled
dom_mix = Counter(r["domain"] for r in filled.values())
p_tie = two_prop(mc["filled"]["tie"], mc["filled"]["n"], mc["original"]["tie"], mc["original"]["n"])
p_acc = two_prop(round(mc["filled"]["gtclear_acc"] * mc["filled"]["gtclear_n"]), mc["filled"]["gtclear_n"],
                 round(mc["original"]["gtclear_acc"] * mc["original"]["gtclear_n"]), mc["original"]["gtclear_n"]) \
    if mc["filled"]["gtclear_n"] and mc["original"]["gtclear_n"] else None
cov_before, cov_fill = len(by_cell), len(filled)
res["glm_fill"] = {"coverage_before": cov_before, "filled": cov_fill,
                   "coverage_after": cov_before + cov_fill, "target": 800,
                   "original": mc["original"], "filled": mc["filled"],
                   "p_tie_diff": p_tie, "p_gtclear_acc_diff": p_acc,
                   "filled_domain_mix": dict(dom_mix)}
md.append(f"- coverage: {cov_before}/800 original + {cov_fill} filled (of {missing_total} missing) "
          f"= {cov_before+cov_fill}/800 ({(cov_before+cov_fill)/800:.1%})")
md.append(f"- original cells: tie {mc['original']['tie']}/{mc['original']['n']} "
          f"({fmt(mc['original']['tie_rate'],3)}); filled cells: tie {mc['filled']['tie']}/{mc['filled']['n']} "
          f"({fmt(mc['filled']['tie_rate'],3)}); 2-prop p={fmt(p_tie)}")
md.append(f"- GT-clear acc: original {fmt(mc['original']['gtclear_acc'],3)} (n={mc['original']['gtclear_n']}), "
          f"filled {fmt(mc['filled']['gtclear_acc'],3)} (n={mc['filled']['gtclear_n']}); p={fmt(p_acc)}")
md.append(f"- A-share: original {fmt(mc['original']['a_share'],3)}, filled {fmt(mc['filled']['a_share'],3)}")
md.append(f"- filled domain mix: {dict(dom_mix)}")

# ---------- 3b. GLM revised (merged view: original + fill) ----------
md.append("")
md.append("## 3b. GLM-5.3 merged view (789 cells): revised headline metrics")
merged = {}
for r in list(glm_main) + [r for r in fill if r.get("verdict") in ("A", "B", "TIE")]:
    merged[(r["pair_id"], r["order"])] = r["verdict"]
mcells = [(k, v) for k, v in merged.items()]
n_m = len(mcells)
tie_m = sum(1 for _, v in mcells if v == "TIE")
by_po2 = defaultdict(dict)
for (pid, o), v in mcells:
    by_po2[pid][o] = v
dbl_m = {p: d for p, d in by_po2.items() if "AB" in d and "BA" in d}
flips_m = sum(1 for p, d in dbl_m.items() if d["AB"] != d["BA"])
# kappa vs all other judges on merged common cells
ms = {}
for src in (bail, agnes):
    for r in src:
        if r.get("verdict") in ("A", "B", "TIE"):
            ms.setdefault(r["judge"], {})[(r["pair_id"], r["order"])] = r["verdict"]
kap_rev = {}
for mj, mcells_j in ms.items():
    if mj == "GLM-5.3":
        continue
    common = [k for k in merged if k in mcells_j]
    if len(common) >= 30:
        kap_rev[mj] = {"n": len(common),
                       "kappa": cohen_kappa([merged[k] for k in common], [mcells_j[k] for k in common])}
kaps = sorted(v["kappa"] for v in kap_rev.values() if v["kappa"] is not None)
gtc_m = [k for k in merged if k[0] in gtclear_ids]
gtc_ok_m = sum(1 for k in gtc_m if merged[k] == pair_by_id[k[0]]["gt_better"])
res["glm_revised"] = {"cells": n_m, "double_pairs": len(dbl_m), "flips": flips_m,
                      "flip_rate": flips_m / len(dbl_m) if dbl_m else None,
                      "tie_rate": tie_m / n_m if n_m else None,
                      "gtclear_n": len(gtc_m), "gtclear_acc": gtc_ok_m / len(gtc_m) if gtc_m else None,
                      "kappa_range": (kaps[0], kaps[-1]) if kaps else None,
                      "kappa_median": float(np.median(kaps)) if kaps else None,
                      "kappa_detail": kap_rev}
md.append(f"- cells {n_m}, double-covered pairs {len(dbl_m)}, order-swap flips {flips_m}/{len(dbl_m)} "
          f"= {fmt(flips_m/len(dbl_m) if dbl_m else None,3)}")
md.append(f"- tie rate (3-caliber raw): {tie_m}/{n_m} = {fmt(tie_m/n_m if n_m else None,3)} "
          f"(frozen v1 value on 674: 0.491)")
md.append(f"- GT-clear acc: {fmt(gtc_ok_m/len(gtc_m) if gtc_m else None,3)} (n={len(gtc_m)})")
md.append(f"- kappa vs other judges: range {fmt(kaps[0] if kaps else None,3)}-{fmt(kaps[-1] if kaps else None,3)}, "
          f"median {fmt(float(np.median(kaps)) if kaps else None,3)} "
          f"(n per judge {min(v['n'] for v in kap_rev.values())}-{max(v['n'] for v in kap_rev.values())})")

# ---------- 4. GLM cross-platform retest ----------
md.append("")
md.append("## 4. GLM-5.3 retest on the third platform")
rt_by = defaultdict(dict)
for r in glm_rt:
    if r.get("verdict") in ("A", "B", "TIE"):
        rt_by[r["pair_id"]][r.get("rep", 0)] = r
both = {pid: d for pid, d in rt_by.items() if 0 in d and 1 in d}
ag_rep = sum(1 for pid, d in both.items() if d[0]["verdict"] == d[1]["verdict"])
# vs main canonical (GLM main on same pair, either order cell -> use AB cell)
main_by_pair = {}
for r in glm_main:
    if r["order"] == "AB":
        main_by_pair[r["pair_id"]] = r
    elif r["pair_id"] not in main_by_pair:
        main_by_pair[r["pair_id"]] = r
main_fill_by_pair = {}
for r in fill:
    if r.get("verdict") in ("A", "B", "TIE"):
        if r["order"] == "AB":
            main_fill_by_pair.setdefault(r["pair_id"], r)
        elif r["pair_id"] not in main_fill_by_pair:
            main_fill_by_pair.setdefault(r["pair_id"], r)
glm_main_all = {}
for p_ in list(main_by_pair.items()) + list(main_fill_by_pair.items()):
    glm_main_all.setdefault(p_[0], p_[1])
ag_main = n_main = 0
for pid, d in both.items():
    m = glm_main_all.get(pid)
    if m and m.get("verdict") in ("A", "B", "TIE"):
        n_main += 1
        ag_main += (d[0]["verdict"] == m["verdict"])
# context: other judges' retest agreement (rep0 vs rep1)
ctx = {}
for jn in sorted({r["judge"] for r in retests}):
    d2 = defaultdict(dict)
    for r in retests:
        if r.get("judge") == jn and r.get("verdict") in ("A", "B", "TIE"):
            d2[r["pair_id"]][r.get("rep", 0)] = r
    b2 = {p: d for p, d in d2.items() if 0 in d and 1 in d}
    if b2:
        a2 = sum(1 for p, d in b2.items() if d[0]["verdict"] == d[1]["verdict"])
        ctx[jn] = [a2, len(b2), a2 / len(b2)]
res["glm_retest"] = {"pairs_both_reps": len(both), "rep_agree": ag_rep,
                     "rep_agree_rate": ag_rep / len(both) if both else None,
                     "vs_main_n": n_main, "vs_main_agree": ag_main,
                     "vs_main_rate": ag_main / n_main if n_main else None,
                     "other_judges_retest": ctx}
md.append(f"- pairs with both reps parsed: {len(both)}; rep0-vs-rep1 agreement "
          f"{ag_rep}/{len(both)} = {fmt(ag_rep/len(both) if both else None,3)}")
md.append(f"- vs main canonical verdicts: {ag_main}/{n_main} = {fmt(ag_main/n_main if n_main else None,3)}")
md.append("- other judges' intra-session retest agreement (context): "
          + "; ".join(f"{j} {a}/{n}={r:.2f}" for j, (a, n, r) in ctx.items()))

# ---------- 5. anchor pilot ----------
md.append("")
md.append("## 5. Premium anchors (pilot; full grid blocked by daily token quota)")
anch = {}
for jn in ("Intern-S2", "MiniMax-M3"):
    rs = [r for r in anchor if r.get("judge") == jn]
    ok = [r for r in rs if r.get("verdict") in ("A", "B", "TIE")]
    by_po = defaultdict(dict)
    for r in ok:
        by_po[r["pair_id"]][r["order"]] = r
    dbl = {p: d for p, d in by_po.items() if "AB" in d and "BA" in d}
    flips = sum(1 for p, d in dbl.items() if d["AB"]["verdict"] != d["BA"]["verdict"])
    nties = sum(1 for p, d in dbl.items()
                if d["AB"]["verdict"] == "TIE" and d["BA"]["verdict"] == "TIE")
    gtc = [r for r in ok if r["pair_id"] in gtclear_ids]
    gtc_ok = sum(1 for r in gtc if r["verdict"] == pair_by_id[r["pair_id"]]["gt_better"])
    tie_all = sum(1 for r in ok if r["verdict"] == "TIE")
    # longer-win among non-tie, length-differ cells
    lw_n = lw_k = 0
    for r in ok:
        if r["verdict"] == "TIE":
            continue
        p = pair_by_id[r["pair_id"]]
        la, lb = p.get("len_a") or 0, p.get("len_b") or 0
        if la == lb:
            continue
        winner_long = (r["verdict"] == "A" and la > lb) or (r["verdict"] == "B" and lb > la)
        lw_n += 1
        lw_k += winner_long
    anch[jn] = {"cells": len(rs), "parsed": len(ok), "double_pairs": len(dbl),
                "flips": flips, "flip_rate": flips / len(dbl) if dbl else None,
                "both_tie_pairs": nties, "tie_rate": tie_all / len(ok) if ok else None,
                "gtclear_cells": len(gtc), "gtclear_acc": gtc_ok / len(gtc) if gtc else None,
                "longer_win": lw_k / lw_n if lw_n else None, "longer_win_n": lw_n,
                "compliance": len(ok) / len(rs) if rs else None}
res["anchors_pilot"] = anch
for jn, a in anch.items():
    md.append(f"- **{jn}**: {a['parsed']} parsed cells ({a['compliance']:.1%} compliance), "
              f"{a['double_pairs']} double-covered pairs, order-swap flips {a['flips']}/{a['double_pairs']} "
              f"= {fmt(a['flip_rate'],3)}, tie% {fmt(a['tie_rate'],3)}, "
              f"GT-clear acc {fmt(a['gtclear_acc'],3)} (n={a['gtclear_cells']}), "
              f"longer-wins {fmt(a['longer_win'],3)} (n={a['longer_win_n']})")

# kappa of anchors vs the 12 judges + between anchors
md.append("")
md.append("### 5a. Anchor vs judge kappa (common cells)")
ref_sets = {"Intern-S2": {}, "MiniMax-M3": {}}
for r in anchor:
    if r.get("verdict") in ("A", "B", "TIE"):
        ref_sets.setdefault(r["judge"], {})[(r["pair_id"], r["order"])] = r["verdict"]
main_sets = {}
for src, tag in ((bail, "bail"), (agnes, "agnes")):
    for r in src:
        if r.get("verdict") in ("A", "B", "TIE"):
            main_sets.setdefault(r["judge"], {})[(r["pair_id"], r["order"])] = r["verdict"]
kappa_tab = {}
for aj in ("Intern-S2", "MiniMax-M3"):
    kappa_tab[aj] = {}
    for mj, mcells in main_sets.items():
        common = [k for k in ref_sets[aj] if k in mcells]
        if len(common) >= 20:
            kap = cohen_kappa([ref_sets[aj][k] for k in common], [mcells[k] for k in common])
            raw = sum(ref_sets[aj][k] == mcells[k] for k in common) / len(common)
            kappa_tab[aj][mj] = {"n": len(common), "kappa": kap, "raw": raw}
            md.append(f"- {aj} vs {mj}: kappa {fmt(kap,3)} (raw {fmt(raw,3)}, n={len(common)})")
common2 = [k for k in ref_sets["Intern-S2"] if k in ref_sets["MiniMax-M3"]]
if len(common2) >= 20:
    kap2 = cohen_kappa([ref_sets["Intern-S2"][k] for k in common2],
                       [ref_sets["MiniMax-M3"][k] for k in common2])
    raw2 = sum(ref_sets["Intern-S2"][k] == ref_sets["MiniMax-M3"][k] for k in common2) / len(common2)
    kappa_tab["anchor_vs_anchor"] = {"n": len(common2), "kappa": kap2, "raw": raw2}
    md.append(f"- Intern-S2 vs MiniMax-M3: kappa {fmt(kap2,3)} (raw {fmt(raw2,3)}, n={len(common2)})")
res["anchor_kappa"] = kappa_tab

# ---------- 6. pending inventory ----------
pend_anchor = 1600 - len(anchor)
pend = {"anchor_cells": pend_anchor,
        "anchor_est_tokens": int(pend_anchor * 850),
        "glm_fill_cells": 126 - len(fill),
        "glm_retest_cells": 80 - len(glm_rt),
        "check_cells": 1}
res["pending"] = pend
md.append("")
md.append("## 6. Pending (blocked by ~1M/day platform token quota)")
md.append(f"- anchor cells {pend_anchor} (~{pend['anchor_est_tokens']} tokens), "
          f"GLM fill {pend['glm_fill_cells']}, GLM retest {pend['glm_retest_cells']}, check {pend['check_cells']}")

json.dump(res, open(f"{R}/e7_results.json", "w"), indent=1)
open(f"{R}/e7_report.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
print("\nsaved -> e7_results.json, e7_report.md")
