#!/usr/bin/env python3
"""Unified Stage C analysis (pre-registered H1/H2/H3/H4/H8 + E8 stability).

All metrics follow the frozen scoring conventions of reanalysis.py:
  picks-right / picks-wrong / tie, with ties-as-error (strict), ties-0.5
  (half), ties-excluded (coverage) reported where relevant.
Outputs: research/stageC_results.json + research/stageC_report.md
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reanalysis import wilson_ci, fisher_exact_2x2, holm_bh, cluster_boot

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_JSON = f"{R}/stageC_results.json"
OUT_MD = f"{R}/stageC_report.md"
RES = {}


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def v1_ab(pair_ids, judges):
    """v1 main-eval canonical AB verdicts for given pairs/judges -> {(pid,j): verdict}."""
    out = {}
    for r in jl(f"{R}/judgments.jsonl"):
        if r.get("order") == "AB" and r.get("pair_id") in pair_ids \
                and r.get("judge") in judges:
            out[(r["pair_id"], r["judge"])] = r.get("verdict")
    return out


# ================= H1: E1 instruction ablation =================
def h1_e1():
    e1 = jl(f"{R}/judgments_e1.jsonl")
    e1_ids = set(json.load(open(f"{R}/e1_pairs.json")))
    judges = ["Qwen3.7-Flash", "Qwen3.8-Max", "DeepSeek-V4-Flash", "DeepSeek-V4.1-Flash"]
    v1 = v1_ab(e1_ids, judges)
    pairs = {p["pair_id"]: p for p in jl(f"{R}/pairs.jsonl")}
    rows = []
    for r in e1 + [{"pair_id": pid, "judge": j, "prompt_version": "P1",
                    "verdict": v, "len_a": pairs[pid]["len_a"], "len_b": pairs[pid]["len_b"],
                    "domain": pairs[pid]["domain"], "qid": pairs[pid]["qid"]}
                   for (pid, j), v in v1.items() if v]:
        p = pairs[r["pair_id"]]
        ld = p["len_a"] - p["len_b"]
        longer = "A" if ld > 0 else ("B" if ld < 0 else None)
        v = r.get("verdict")
        if v not in ("A", "B", "TIE") or longer is None:
            continue
        rows.append({
            "pair": r["pair_id"], "judge": r["judge"], "arm": r["prompt_version"],
            "qid": r["qid"], "domain": r["domain"],
            "len_z": (ld - 118.4) / 350.0,  # z-score with v1 pooled stats (frozen)
            "y": 1 if v == longer else (0 if v != "TIE" else np.nan),
            "tie": 1 if v == "TIE" else 0,
            "picks_longer": 1 if v == longer else 0,
        })
    df = pd.DataFrame(rows)
    res = {"per_arm": {}, "per_judge_arm": {}}
    for arm in ("P0", "P1", "P2", "P3"):
        d = df[df.arm == arm]
        sub = d.dropna(subset=["y"])
        res["per_arm"][arm] = {
            "n": int(len(d)), "n_decided": int(len(sub)),
            "tie_rate": round(float(d.tie.mean()), 3),
            "picks_longer_rate": round(float(sub.y.mean()), 3),
            "picks_longer_ci": [round(x, 3) for x in wilson_ci(int(sub.y.sum()), len(sub))],
        }
        for j in judges:
            dj = sub[sub.judge == j]
            res["per_judge_arm"][f"{j}|{arm}"] = {
                "n": int(len(dj)),
                "picks_longer": round(float(dj.y.mean()), 3) if len(dj) else None,
            }
    # pooled interaction test: y ~ len_z * isP1 + judge FE, cluster by qid
    d2 = df[df.arm.isin(["P0", "P1"])].dropna(subset=["y"]).copy()
    d2["isP1"] = (d2.arm == "P1").astype(float)
    d2["inter"] = d2.len_z * d2.isP1
    X = pd.get_dummies(d2[["len_z", "isP1", "inter", "judge"]], drop_first=True).astype(float)
    X = sm.add_constant(X)
    fit = sm.GLM(d2.y.astype(float), X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": d2.qid})
    inter_col = [c for c in X.columns if c == "inter"][0]
    beta_inter = float(fit.params[inter_col])
    p_two = float(fit.pvalues[inter_col])
    p_one = p_two / 2 if beta_inter < 0 else 1 - p_two / 2  # H1: beta_P1 < beta_P0
    beta_len_P0 = float(fit.params[[c for c in X.columns if c == "len_z"][0]])
    res["interaction_test"] = {
        "beta_len_z_P0": round(beta_len_P0, 3),
        "beta_interaction_P1_minus_P0": round(beta_inter, 3),
        "se": round(float(fit.bse[inter_col]), 3),
        "p_one_sided": round(p_one, 4),
        "n_obs": int(len(d2)),
    }
    # per-arm len coefficients for reporting
    for arm in ("P0", "P1", "P2", "P3"):
        d = df[df.arm == arm].dropna(subset=["y"]).copy()
        Xa = pd.get_dummies(d[["len_z", "judge"]], drop_first=True).astype(float)
        Xa = sm.add_constant(Xa)
        fa = sm.GLM(d.y.astype(float), Xa, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": d.qid})
        res["per_arm"][arm]["beta_len_z"] = round(float(fa.params["len_z"]), 3)
    res["H1_decision"] = ("KEEP_HEADLINE" if p_one < 0.05 and beta_inter < 0
                          else "DOWNGRADE_TO_PROMPT_SENSITIVITY")
    RES["H1_E1"] = res
    return res


# ================= H2: E2 tie ablation =================
def h2_e2():
    e2 = jl(f"{R}/judgments_e2.jsonl")
    e2_ids = set(json.load(open(f"{R}/e2_pairs.json")))
    judges = ["Qwen3.7-Flash", "Qwen3.8-27B", "DeepSeek-V4-Flash", "DeepSeek-V4-Pro"]
    v1 = v1_ab(e2_ids, judges)
    pairs = {p["pair_id"]: p for p in jl(f"{R}/pairs.jsonl")}
    res = {"per_judge": {}, "acc_T": {}}
    pvals = []
    tests = []
    for j in judges:
        t1 = [r for r in e2 if r["judge"] == j and r["prompt_version"] == "T1"]
        t2 = [r for r in e2 if r["judge"] == j and r["prompt_version"] == "T2"]
        p1 = [(pid, v) for (pid, jj), v in v1.items() if jj == j and v]
        def tie_rate(rows):
            if not rows:
                return None, 0
            k = sum(1 for r in rows if r.get("verdict") == "TIE")
            return k / len(rows), len(rows)
        tr1, n1 = tie_rate(t1)
        tr2, n2 = tie_rate(t2)
        trp, np_ = tie_rate([{"verdict": v} for _, v in p1])
        # McNemar exact on discordant pairs (T1 vs P1; T2 vs P1)
        for label, new_rows in (("T1_vs_P1", t1), ("T2_vs_P1", t2)):
            disc01 = disc10 = 0
            for r in new_rows:
                old = v1.get((r["pair_id"], j))
                if old is None:
                    continue
                new_tie = r.get("verdict") == "TIE"
                old_tie = old == "TIE"
                if new_tie and not old_tie:
                    disc01 += 1
                elif old_tie and not new_tie:
                    disc10 += 1
            nd = disc01 + disc10
            if nd:
                # two-sided exact McNemar
                pv = stats.binomtest(min(disc01, disc10), nd, 0.5).pvalue
            else:
                pv = 1.0
            pvals.append(pv)
            tests.append((j, label, disc01, disc10, pv))
        res["per_judge"][j] = {
            "n_T1": n1, "n_T2": n2, "n_P1_v1": np_,
            "tie_T1": round(tr1, 3) if tr1 is not None else None,
            "tie_T1_ci": [round(x, 3) for x in wilson_ci(int((tr1 or 0) * n1), n1)] if n1 else None,
            "tie_T2": round(tr2, 3) if tr2 is not None else None,
            "tie_P1_v1": round(trp, 3) if trp is not None else None,
            "tie_P1_ci": [round(x, 3) for x in wilson_ci(int((trp or 0) * np_), np_)] if np_ else None,
        }
        # acc_T on GT-clear subset (ties=error / half / coverage)
        def acc_rows(rows, tag):
            out = {"n": 0, "right": 0, "tie": 0}
            for r in rows:
                p = pairs[r["pair_id"]]
                if not p.get("gt_better"):
                    continue
                v = r.get("verdict")
                if v not in ("A", "B", "TIE"):
                    continue
                out["n"] += 1
                right_side = p["gt_better"]
                if v == right_side:
                    out["right"] += 1
                elif v == "TIE":
                    out["tie"] += 1
            if out["n"]:
                half = (out["right"] + 0.5 * out["tie"]) / out["n"]
                cov = out["right"] / (out["n"] - out["tie"]) if out["n"] > out["tie"] else None
                return {"n": out["n"], "acc_T_ties_err": round(out["right"] / out["n"], 3),
                        "acc_T_ties_half": round(half, 3),
                        "acc_T_cov": round(cov, 3) if cov else None}
            return None
        res["acc_T"][j] = {
            "T1": acc_rows(t1, "T1"), "T2": acc_rows(t2, "T2"),
            "P1_v1": acc_rows([{"pair_id": pid, "verdict": v} for pid, v in p1], "P1"),
        }
    # Holm across the 8 paired tests
    holm, _bh = holm_bh(pvals)
    res["mcnemar_tests"] = [
        {"judge": j, "contrast": lab, "new_tie_added": d01, "old_tie_removed": d10,
         "p": round(pv, 4), "p_holm": round(float(a), 4)}
        for (j, lab, d01, d10, pv), a in zip(tests, holm)]
    res["H2_decision"] = "PROMPT_DRIVEN" if all(
        t[2] > 0 for t in tests if t[1] == "T1_vs_P1") else "MIXED"
    RES["H2_E2"] = res
    return res


# ================= H3 + E4: E3 kin analysis =================
def h3_e3():
    e3j = jl(f"{R}/judgments_e3.jsonl")
    pairs = {p["pair_id"]: p for p in jl(f"{R}/pairs_e3.jsonl")}
    res = {"families": {}, "layers": {}, "regression": {}}
    rows = []
    for r in e3j:
        p = pairs[r["pair_id"]]
        v = r.get("verdict")
        if v not in ("A", "B", "TIE"):
            continue
        right_side = p["gt_better"]
        wrong_fam = None
        # family of wrong side and right side
        wrong_fam = _fam(p["wrong_gen"])
        right_fam = _fam(p["right_gen"])
        jfam = r["judge_family"]
        layer = "unrelated"
        if jfam == wrong_fam:
            layer = "self" if r["judge"] == p["wrong_gen"] else "brother"
        rows.append({
            "pair": r["pair_id"], "tid": r["qid"], "judge": r["judge"], "jfam": jfam,
            "wrong_fam": wrong_fam, "right_fam": right_fam, "layer": layer,
            "wrong_gen": p["wrong_gen"], "right_gen": p["right_gen"],
            "domain": r["domain"],
            "picks_right": 1 if v == right_side else 0,
            "picks_wrong": 1 if (v != "TIE" and v != right_side) else 0,
            "tie": 1 if v == "TIE" else 0,
        })
    df = pd.DataFrame(rows)
    # per-family: kin-wrong error vs non-kin baseline (Fisher) + Wilson
    for fam in ("Qwen", "DeepSeek", "Kimi"):
        kin = df[(df.jfam == fam) & (df.wrong_fam == fam)]
        nonkin_pool = df[(df.jfam == fam) & (df.wrong_fam != fam) & (df.right_fam != fam)]
        def rates(d):
            if not len(d):
                return None
            n = len(d)
            pw = int(d.picks_wrong.sum())
            ti = int(d.tie.sum())
            pr = int(d.picks_right.sum())
            return {
                "n": n, "picks_wrong_rate": round(pw / n, 3),
                "picks_wrong_ci": [round(x, 3) for x in wilson_ci(pw, n)],
                "tie_rate": round(ti / n, 3),
                "picks_right_rate": round(pr / n, 3),
            }
        kr = rates(kin)
        bl = rates(nonkin_pool)
        fish = None
        if kr and bl:
            fish = round(fisher_exact_2x2(round(kr["picks_wrong_rate"] * kr["n"]),
                                          kr["n"], round(bl["picks_wrong_rate"] * bl["n"]),
                                          bl["n"]), 4)
        res["families"][fam] = {"kin_wrong": kr, "nonkin_baseline": bl, "fisher_p": fish}
        # E4 layers within kin pairs
        lay = {}
        for l in ("self", "brother"):
            d = kin[kin.layer == l]
            lay[l] = rates(d)
        res["layers"][fam] = lay
    # pooled regression: picks_wrong ~ kin(jfam==wrong_fam) + domain FE, cluster by tid
    df["kin"] = (df.jfam == df.wrong_fam).astype(float)
    X = pd.get_dummies(df[["kin", "domain"]], drop_first=True).astype(float)
    X = sm.add_constant(X)
    fit = sm.GLM(df.picks_wrong.astype(float), X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": df.tid})
    res["regression"] = {
        "beta_kin": round(float(fit.params["kin"]), 3),
        "se": round(float(fit.bse["kin"]), 3),
        "p": round(float(fit.pvalues["kin"]), 4),
        "or_kin": round(float(np.exp(fit.params["kin"])), 3),
        "n_obs": int(len(df)),
    }
    # right-side kin secondary: family judges on pairs where right side is theirs
    for fam in ("Qwen", "DeepSeek", "Kimi"):
        d = df[(df.jfam == fam) & (df.right_fam == fam)]
        if len(d):
            res["families"][fam]["kin_right_picks_right"] = round(float(d.picks_right.mean()), 3)
            res["families"][fam]["kin_right_n"] = int(len(d))
    RES["H3_E3"] = res
    return res


def _fam(name):
    from judge import GEN_FAMILY
    if name in GEN_FAMILY:
        return GEN_FAMILY[name]
    return {"Qwen3.7-Flash": "Qwen"}.get(name, "Qwen" if "Qwen" in name else
           "DeepSeek" if "DeepSeek" in name else "Kimi" if "Kimi" in name else "GLM")


# ================= H8: E5 no-reference =================
def h8_e5():
    e5 = jl(f"{R}/judgments_e5.jsonl")
    e3j = jl(f"{R}/judgments_e3.jsonl")
    with_ref = {(r["pair_id"], r["judge"]): r for r in e3j}
    # judges present in E5; paired analysis only for those with E3 with-ref arms
    judges_present = []
    for r in e5:
        if r["judge"] not in judges_present:
            judges_present.append(r["judge"])
    res = {}
    pairs = {p["pair_id"]: p for p in jl(f"{R}/pairs_e3.jsonl")}
    for j in judges_present:
        rows = []
        for r in e5:
            if r["judge"] != j:
                continue
            w = with_ref.get((r["pair_id"], j))
            if w:
                rows.append({
                    "pair": r["pair_id"],
                    "right_ref": w.get("verdict"), "right_noref": r.get("verdict"),
                })
        if not rows:
            # judge has no with-ref twin (not an E3 judge): unpaired summary only
            nr = [r for r in e5 if r["judge"] == j]
            n = len(nr)
            pr = sum(1 for r in nr if r.get("verdict") == pairs[r["pair_id"]]["gt_better"])
            ti = sum(1 for r in nr if r.get("verdict") == "TIE")
            res[j] = {
                "n_pairs": n, "mode": "UNPAIRED (no with-ref twin; not an E3 judge)",
                "no_ref": {"picks_right": pr, "tie": ti,
                           "acc_half": round((pr + 0.5 * ti) / n, 3) if n else None},
            }
            continue
        n = len(rows)
        def score(key):
            r_, t_, w_ = 0, 0, 0
            for x in rows:
                side = pairs[x["pair"]]["gt_better"]
                v = x[key]
                if v == side:
                    r_ += 1
                elif v == "TIE":
                    t_ += 1
                else:
                    w_ += 1
            return {"picks_right": r_, "tie": t_, "picks_wrong": w_}
        s_ref, s_noref = score("right_ref"), score("right_noref")
        diffs = []
        for x in rows:
            side = pairs[x["pair"]]["gt_better"]
            f = lambda v: 1 if v == side else (0.5 if v == "TIE" else 0)
            diffs.append(f(x["right_noref"]) - f(x["right_ref"]))
        ci = cluster_boot(diffs, [x["pair"] for x in rows], stat=np.mean)
        res[j] = {
            "n_pairs": n,
            "with_ref": {"picks_right": s_ref["picks_right"], "tie": s_ref["tie"],
                         "picks_wrong": s_ref["picks_wrong"],
                         "acc_half": round((s_ref["picks_right"] + 0.5 * s_ref["tie"]) / n, 3)},
            "no_ref": {"picks_right": s_noref["picks_right"], "tie": s_noref["tie"],
                       "picks_wrong": s_noref["picks_wrong"],
                       "acc_half": round((s_noref["picks_right"] + 0.5 * s_noref["tie"]) / n, 3)},
            "delta_acc_half": round(float(np.mean(diffs)), 3),
            "delta_ci": [round(x, 3) for x in ci[:2]],
            "delta_tie": round((s_noref["tie"] - s_ref["tie"]) / n, 3),
        }
    RES["H8_E5"] = res
    return res


# ================= E8: cross-session stability =================
def e8_stab():
    e8 = jl(f"{R}/judgments_e8.jsonl")
    e8_ids = set(json.load(open(f"{R}/e8_pairs.json")))
    v1 = v1_ab(e8_ids, ["Qwen3.8-27B", "DeepSeek-V4-Flash", "DeepSeek-V4.1-Flash"])
    # v1 retest intra-agreement for the same judges
    rt = jl(f"{R}/retests.jsonl")
    res = {"per_judge": {}}
    for j in ("Qwen3.8-27B", "DeepSeek-V4-Flash", "DeepSeek-V4.1-Flash"):
        same = tot = 0
        for r in e8:
            if r["judge"] != j:
                continue
            old = v1.get((r["pair_id"], j))
            if old is None or r.get("verdict") not in ("A", "B", "TIE"):
                continue
            tot += 1
            if old == r["verdict"]:
                same += 1
        # v1 retest intra (rep 0 vs rep 1; v1 retest reps are 0/1) for judge j
        rt = jl(f"{R}/retests.jsonl")
        by = {}
        for r in rt:
            if r.get("judge") == j and r.get("order") == "AB":
                by.setdefault(r["pair_id"], {})[r.get("rep")] = r.get("verdict")
        it = itot = 0
        for pid, d in by.items():
            if 0 in d and 1 in d and d[0] and d[1]:
                itot += 1
                if d[0] == d[1]:
                    it += 1
        res["per_judge"][j] = {
            "n": tot,
            "e8_vs_v1_agreement": round(same / tot, 3) if tot else None,
            "ci": [round(x, 3) for x in wilson_ci(same, tot)] if tot else None,
            "v1_retest_intra_agreement": round(it / itot, 3) if itot else None,
            "v1_retest_n": itot,
        }
    RES["E8_stability"] = res
    return res


# ================= version table =================
def versions():
    vers = {}
    for f in ("judgments_e0.jsonl", "judgments_e1.jsonl", "judgments_e2.jsonl",
              "judgments_e3.jsonl", "judgments_e5.jsonl", "judgments_e8.jsonl",
              "answers_e3.jsonl"):
        for r in jl(f"{R}/{f}"):
            key = r.get("judge_model") or r.get("generator_model")
            snap = r.get("snapshot")
            if key:
                vers.setdefault(key, set())
                if snap:
                    vers[key].add(snap)
    RES["snapshots"] = {k: sorted(v) for k, v in vers.items()}
    return RES["snapshots"]


def main():
    print("== H1 (E1 instruction ablation) ==")
    h1 = h1_e1()
    print(json.dumps(h1["interaction_test"], indent=1))
    print("per-arm picks-longer:", {a: d["picks_longer_rate"] for a, d in h1["per_arm"].items()})
    print("per-arm beta_len_z:", {a: d.get("beta_len_z") for a, d in h1["per_arm"].items()})
    print("H1 decision:", h1["H1_decision"])
    print("\n== H2 (E2 tie ablation) ==")
    h2 = h2_e2()
    for j, d in h2["per_judge"].items():
        print(f"  {j}: T1={d['tie_T1']} T2={d['tie_T2']} P1(v1)={d['tie_P1_v1']}")
    print("  acc_T (ties=err):", {j: {k: (v or {}).get("acc_T_ties_err") for k, v in d.items()}
                                   for j, d in h2["acc_T"].items()})
    print("\n== H3/E4 (E3 kin) ==")
    h3 = h3_e3()
    for fam, d in h3["families"].items():
        kw = d["kin_wrong"]
        bl = d["nonkin_baseline"]
        print(f"  {fam}: kin-wrong n={kw and kw['n']} picks_wrong={kw and kw['picks_wrong_rate']} "
              f"| baseline n={bl and bl['n']} picks_wrong={bl and bl['picks_wrong_rate']} "
              f"| fisher_p={d['fisher_p']}")
    for fam, lay in h3["layers"].items():
        print(f"  {fam} layers: self={lay['self'] and lay['self']['picks_wrong_rate']} (n={lay['self'] and lay['self']['n']}), "
              f"brother={lay['brother'] and lay['brother']['picks_wrong_rate']} (n={lay['brother'] and lay['brother']['n']})")
    print("  regression:", h3["regression"])
    print("\n== H8 (E5 no-reference) ==")
    h8 = h8_e5()
    print(json.dumps(h8, indent=1))
    print("\n== E8 stability ==")
    e8 = e8_stab()
    print(json.dumps(e8, indent=1))
    print("\n== snapshots ==")
    print(json.dumps(versions(), indent=1))
    json.dump(RES, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    with open(OUT_MD, "w") as f:
        f.write("# Stage C results (E1-E8)\n\n```json\n")
        f.write(json.dumps(RES, indent=1, ensure_ascii=False))
        f.write("\n```\n")
    print(f"\nwrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
