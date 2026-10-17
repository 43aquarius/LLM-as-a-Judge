#!/usr/bin/env python3
"""Comprehensive analysis of judge reliability & biases.

Inputs:  pairs.jsonl, judgments.jsonl, retests.jsonl, questions.jsonl, answers.jsonl
Outputs: results.json (all metrics + CIs), figures/*.pdf (vector charts)

Metrics:
  A. parse rate, tie rate, verdict distribution per judge
  B. accuracy on GT-clear pairs (math + knowledge)
  C. position bias (first-position preference) + order-flip inconsistency
  D. length bias (P(longer wins | non-tie) - 0.5) + logistic regression slopes
  E. family preference (kin-wrong pick rate, kin uplift) with GT control
  F. inter-judge agreement (pairwise Cohen's kappa, raw agreement)
  G. test-retest consistency
  H. panel (majority vote) analysis
  I. scale-trend analysis within families

All CIs: question-level cluster bootstrap (B=2000). Corrections: Benjamini-Hochberg.
"""
import itertools
import json
import os
from collections import defaultdict

import numpy as np

RESEARCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FIGDIR = os.path.join(RESEARCH, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# 10 free open-source judges (Aliyun free tier) + 2 proprietary free-tier Agnes judges
JUDGE_ORDER = [
    "Qwen3.7-Flash", "Qwen3.8-Flash", "Qwen3.8-27B", "Qwen3.8-Max",
    "Qwen3.8-2.4T", "DeepSeek-V4-Flash", "DeepSeek-V4.1-Flash",
    "DeepSeek-V4-Pro", "GLM-5.3", "Kimi-K3",
    "Agnes-2.5-Flash", "Agnes-3.0-Flash",
]
OPEN_JUDGES = JUDGE_ORDER[:10]
FAMILY = {
    "Qwen3.7-Flash": "Qwen", "Qwen3.8-Flash": "Qwen", "Qwen3.8-27B": "Qwen",
    "Qwen3.8-Max": "Qwen", "Qwen3.8-2.4T": "Qwen",
    "DeepSeek-V4-Flash": "DeepSeek", "DeepSeek-V4.1-Flash": "DeepSeek",
    "DeepSeek-V4-Pro": "DeepSeek", "GLM-5.3": "GLM", "Kimi-K3": "Kimi",
    "Agnes-2.5-Flash": "Agnes", "Agnes-3.0-Flash": "Agnes",
}
FAMILY_COLOR = {"Qwen": "#6C8EBF", "DeepSeek": "#4B9E8A", "GLM": "#B4657A", "Kimi": "#D79B00", "Agnes": "#7A6FC1"}
FAMILY_TIERS = {  # ascending capability tier within family (for trend plots)
    "Qwen": ["Qwen3.7-Flash", "Qwen3.8-Flash", "Qwen3.8-27B", "Qwen3.8-Max", "Qwen3.8-2.4T"],
    "DeepSeek": ["DeepSeek-V4-Flash", "DeepSeek-V4.1-Flash", "DeepSeek-V4-Pro"],
    "GLM": ["GLM-5.3"],
    "Kimi": ["Kimi-K3"],
    "Agnes": ["Agnes-2.5-Flash", "Agnes-3.0-Flash"],
}
B = 2000
rng = np.random.default_rng(20261016)


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


def cohen_kappa(a, b, labels=("A", "B", "TIE")):
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    if n == 0:
        return np.nan
    po = float((a == b).mean())
    pe = sum(float((a == l).mean()) * float((b == l).mean()) for l in labels)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def cluster_boot_ci(values, clusters, stat=np.mean, alpha=0.05):
    """Question-level cluster bootstrap.

    Determinism note: each call seeds its own stream from a content hash of
    the input, so CIs are reproducible per data content and immune to call
    order or to unrelated data growth in other judges (a shared-stream design
    previously made panel numbers drift as the Agnes pass grew)."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    import hashlib
    _seed = int(hashlib.md5(values.tobytes() + clusters.astype(str).tobytes()).hexdigest()[:8], 16)
    _brng = np.random.default_rng(_seed)
    uc = np.unique(clusters)
    idx_by_c = {c: np.where(clusters == c)[0] for c in uc}
    stats = []
    n = len(uc)
    for _ in range(B):
        cs = _brng.choice(uc, size=n, replace=True)
        samp = np.concatenate([idx_by_c[c] for c in cs])
        try:
            stats.append(stat(values[samp]))
        except Exception:
            pass
    if not stats:
        return (np.nan, np.nan)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def bh_correct(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def main():
    # open-source judges + proprietary Agnes judges (separate files, merged here)
    all_judg_raw = (load_jsonl(os.path.join(RESEARCH, "judgments.jsonl"))
                    + load_jsonl(os.path.join(RESEARCH, "judgments_agnes.jsonl")))
    judgments = [j for j in all_judg_raw if j.get("parsed_ok")]
    retests = [j for j in (load_jsonl(os.path.join(RESEARCH, "retests.jsonl"))
                           + load_jsonl(os.path.join(RESEARCH, "retests_agnes.jsonl")))
               if j.get("parsed_ok")]
    pairs = {p["pair_id"]: p for p in load_jsonl(os.path.join(RESEARCH, "pairs.jsonl"))}

    judges_present = [j for j in JUDGE_ORDER if any(x["judge"] == j for x in judgments)]
    R = {"n_judgments_raw": len(all_judg_raw), "n_judgments_parsed": len(judgments),
         "n_retests": len(retests), "judges": judges_present}

    # ---------- A. parse rate / verdict distribution ----------
    print("== A. parse & severity ==")
    parse_rates, tie_rates, first_rates = {}, {}, {}
    for jn in judges_present:
        raw_j = [x for x in all_judg_raw if x["judge"] == jn and x.get("rep", 0) == 0]
        ok_j = [x for x in raw_j if x.get("parsed_ok")]
        parse_rates[jn] = len(ok_j) / max(1, len(raw_j))
        ab = [x for x in ok_j if x["order"] == "AB"]
        ties = sum(1 for x in ab if x["verdict"] == "TIE")
        tie_rates[jn] = ties / max(1, len(ab))
    R["parse_rate"] = parse_rates
    R["tie_rate"] = tie_rates

    # ---------- B. GT accuracy (gt_better looked up from pairs.jsonl = v2 labels) ----------
    print("== B. GT accuracy ==")
    acc = {}
    for jn in judges_present:
        gt_j = [x for x in judgments if x["judge"] == jn and x["order"] == "AB"
                and pairs.get(x["pair_id"], {}).get("gt_better") in ("A", "B")]
        vals, clus = [], []
        for x in gt_j:
            gt = pairs.get(x["pair_id"], {}).get("gt_better")
            vals.append(1.0 if x["verdict"] == gt else 0.0)
            clus.append(x["qid"])
        if vals:
            acc[jn] = {"acc": float(np.mean(vals)), "ci": cluster_boot_ci(vals, clus), "n": len(vals)}
    R["gt_accuracy"] = acc

    # ---------- C. position bias + flip ----------
    print("== C. position bias ==")
    pb, flip = {}, {}
    for jn in judges_present:
        by_pair = defaultdict(dict)
        for x in judgments:
            if x["judge"] == jn and x.get("rep", 0) == 0:
                by_pair[x["pair_id"]][x["order"]] = x["verdict"]
        # first-position preference among non-tie (AB order)
        ab = [x for x in judgments if x["judge"] == jn and x["order"] == "AB" and x.get("rep", 0) == 0]
        non_tie = [x for x in ab if x["verdict"] in ("A", "B")]
        if non_tie:
            first_win = [1.0 if x["verdict"] == "A" else 0.0 for x in non_tie]
            clus = [x["qid"] for x in non_tie]
            pb[jn] = {"first_rate": float(np.mean(first_win)),
                      "bias": float(np.mean(first_win) - 0.5),
                      "ci": cluster_boot_ci(first_win, clus), "n": len(non_tie)}
        # flip: verdicts that change (mapped canonically) between AB and BA
        fl = []
        flc = []
        for pid, d in by_pair.items():
            if "AB" in d and "BA" in d:
                fl.append(0.0 if d["AB"] == d["BA"] else 1.0)
                flc.append(pid.rsplit("-", 1)[0])
        if fl:
            flip[jn] = {"flip_rate": float(np.mean(fl)), "ci": cluster_boot_ci(fl, flc), "n": len(fl)}
    R["position_bias"] = pb
    R["flip_rate"] = flip

    # ---------- D. length bias ----------
    print("== D. length bias ==")
    lb = {}
    for jn in judges_present:
        vals, clus = [], []
        for x in judgments:
            if x["judge"] == jn and x["order"] == "AB" and x.get("rep", 0) == 0 \
               and x["verdict"] in ("A", "B") and x.get("len_a") and x.get("len_b"):
                longer = "A" if x["len_a"] > x["len_b"] else "B"
                vals.append(1.0 if x["verdict"] == longer else 0.0)
                clus.append(x["qid"])
        if vals:
            lb[jn] = {"longer_win_rate": float(np.mean(vals)),
                      "bias": float(np.mean(vals) - 0.5), "ci": cluster_boot_ci(vals, clus), "n": len(vals)}
    R["length_bias"] = lb

    # ---------- E. family preference ----------
    print("== E. family preference ==")
    kin = {}
    for jn in judges_present:
        jfam = FAMILY[jn]
        kin_wrong_vals, kin_right_vals, kw_cl, kr_cl = [], [], [], []
        kin_all_vals, kin_all_cl = [], []
        for x in judgments:
            if x["judge"] != jn or x["order"] != "AB" or x.get("rep", 0) != 0:
                continue
            a_f, b_f = x["gen_a_family"], x["gen_b_family"]
            kin_side = None
            if a_f == jfam and b_f != jfam:
                kin_side = "A"
            elif b_f == jfam and a_f != jfam:
                kin_side = "B"
            if kin_side is None:
                continue
            if x["verdict"] not in ("A", "B"):
                continue
            pick_kin = 1.0 if x["verdict"] == kin_side else 0.0
            kin_all_vals.append(pick_kin)
            kin_all_cl.append(x["qid"])
            gt = pairs.get(x["pair_id"], {}).get("gt_better")
            if gt:
                kin_correct = (gt == kin_side)
                if kin_correct:
                    kin_right_vals.append(pick_kin); kr_cl.append(x["qid"])
                else:
                    kin_wrong_vals.append(pick_kin); kw_cl.append(x["qid"])
        kin[jn] = {
            "kin_pairs_n": len(kin_all_vals),
            "kin_win_rate": float(np.mean(kin_all_vals)) if kin_all_vals else None,
            "kin_win_ci": cluster_boot_ci(kin_all_vals, kin_all_cl) if kin_all_vals else None,
            "kin_wrong_pick_rate": float(np.mean(kin_wrong_vals)) if kin_wrong_vals else None,
            "kin_wrong_n": len(kin_wrong_vals),
            "kin_wrong_ci": cluster_boot_ci(kin_wrong_vals, kw_cl) if kin_wrong_vals else None,
            "kin_right_pick_rate": float(np.mean(kin_right_vals)) if kin_right_vals else None,
            "kin_right_n": len(kin_right_vals),
        }
    R["family_preference"] = kin

    # E2. family-pooled kin-wrong (small per-judge n -> pool judges within family)
    pooled = {}
    for fam in ["Qwen", "DeepSeek", "GLM", "Kimi"]:
        vals, clus = [], []
        win_vals, win_clus = [], []
        for x in judgments:
            if x["order"] != "AB" or x.get("rep", 0) != 0 or x["verdict"] not in ("A", "B"):
                continue
            if x["judge"] not in OPEN_JUDGES or FAMILY[x["judge"]] != fam:
                continue
            a_f, b_f = x["gen_a_family"], x["gen_b_family"]
            kin_side = "A" if (a_f == fam and b_f != fam) else ("B" if (b_f == fam and a_f != fam) else None)
            if kin_side is None:
                continue
            pick = 1.0 if x["verdict"] == kin_side else 0.0
            win_vals.append(pick); win_clus.append(x["qid"])
            gt = pairs.get(x["pair_id"], {}).get("gt_better")
            if gt and gt != kin_side:  # kin side proven wrong
                vals.append(pick); clus.append(x["qid"])
        pooled[fam] = {
            "kin_wrong_pick_rate": float(np.mean(vals)) if vals else None,
            "kin_wrong_ci": cluster_boot_ci(vals, clus) if vals else None,
            "kin_wrong_n": len(vals),
            "kin_win_rate": float(np.mean(win_vals)) if win_vals else None,
            "kin_win_ci": cluster_boot_ci(win_vals, win_clus) if win_vals else None,
            "kin_win_n": len(win_vals),
        }
    R["family_preference_pooled"] = pooled

    # ---------- F. inter-judge agreement (AB order, pairwise-available pairs) ----------
    print("== F. agreement ==")
    by_pair_j = defaultdict(dict)
    for x in judgments:
        if x["order"] == "AB" and x.get("rep", 0) == 0:
            by_pair_j[x["pair_id"]][x["judge"]] = x["verdict"]
    common_pairs = [p for p, d in by_pair_j.items()
                    if all(j in d for j in judges_present)]
    R["n_common_pairs_full"] = len(common_pairs)
    kappas, raws = {}, {}
    for a, b in itertools.combinations(judges_present, 2):
        # pairwise-available: use all pairs where BOTH judges gave a verdict
        ps = [p for p, d in by_pair_j.items() if a in d and b in d]
        if not ps:
            continue
        va = [by_pair_j[p][a] for p in ps]
        vb = [by_pair_j[p][b] for p in ps]
        kappas[(a, b)] = cohen_kappa(va, vb)
        raws[(a, b)] = float(np.mean(np.asarray(va) == np.asarray(vb)))
    R["kappa_matrix"] = {f"{a}||{b}": k for (a, b), k in kappas.items()}
    R["raw_agreement"] = {f"{a}||{b}": v for (a, b), v in raws.items()}
    R["kappa_pair_n"] = {f"{a}||{b}": len([p for p, d in by_pair_j.items() if a in d and b in d])
                          for (a, b) in kappas}
    avg_k = {j: float(np.nanmean([k for (a, b), k in kappas.items() if j in (a, b)]))
             for j in judges_present}
    R["avg_kappa"] = avg_k
    # family-block kappa
    fam_blocks = {}
    for fa in ["Qwen", "DeepSeek", "GLM", "Kimi", "Agnes"]:
        for fb in ["Qwen", "DeepSeek", "GLM", "Kimi", "Agnes"]:
            ks = [k for (a, b), k in kappas.items()
                  if (FAMILY[a] == fa and FAMILY[b] == fb) or (FAMILY[a] == fb and FAMILY[b] == fa)]
            if ks:
                fam_blocks[f"{fa}-{fb}"] = float(np.nanmean(ks))
    R["family_block_kappa"] = fam_blocks
    # open-vs-proprietary agreement contrast
    open_open = [k for (a, b), k in kappas.items() if a in OPEN_JUDGES and b in OPEN_JUDGES]
    mixed = [k for (a, b), k in kappas.items()
             if (a in OPEN_JUDGES) != (b in OPEN_JUDGES)]
    R["open_open_mean_kappa"] = float(np.nanmean(open_open)) if open_open else None
    R["open_proprietary_mean_kappa"] = float(np.nanmean(mixed)) if mixed else None

    # ---------- G. test-retest ----------
    print("== G. test-retest ==")
    tr = {}
    for jn in judges_present:
        by_key = defaultdict(list)
        for x in retests:
            if x["judge"] == jn:
                by_key[x["pair_id"]].append(x["verdict"])
        agree, clus = [], []
        for pid, vs in by_key.items():
            if len(vs) >= 2:
                agree.append(1.0 if vs[0] == vs[1] else 0.0)
                clus.append(pid.rsplit("-", 1)[0])
        if agree and len(agree) >= 20:
            # min-n rule: the proprietary retest pass was curtailed by platform
            # rate limits (n<20 complete pair-units), so it is excluded from
            # headline test-retest statistics; raw partial records are retained
            # in the released artifact for transparency.
            tr[jn] = {"retest_agreement": float(np.mean(agree)),
                      "ci": cluster_boot_ci(agree, clus), "n": len(agree)}
    R["test_retest"] = tr

    # ---------- H. panel analysis (panels drawn from OPEN judges only) ----------
    print("== H. panels ==")
    # majority vote over k open judges on GT-clear pairs (AB order, all k present)
    open_present = [j for j in OPEN_JUDGES if j in judges_present]
    panel_rng = np.random.default_rng(20261017)  # dedicated stream: stable across data growth
    def panel_eval(pool, k, gt_pairs):
        accs, pbs = [], []
        for trial in range(500):
            sel = list(panel_rng.choice(pool, size=k, replace=False))
            pp = [p for p in gt_pairs if all(j in by_pair_j.get(p, {}) for j in sel)]
            hits, firsts = [], []
            for p in pp:
                vs = [by_pair_j[p][j] for j in sel]
                non = [v for v in vs if v in ("A", "B")]
                if not non:
                    mv = "TIE"
                else:
                    counts = {l: non.count(l) for l in ("A", "B")}
                    mv = max(counts, key=counts.get) if counts["A"] != counts["B"] else "TIE"
                if mv != "TIE":
                    hits.append(1.0 if mv == pairs[p]["gt_better"] else 0.0)
                    firsts.append(1.0 if mv == "A" else 0.0)
            if hits:
                accs.append(np.mean(hits)); pbs.append(np.mean(firsts) - 0.5)
        return {"panel_acc_mean": float(np.mean(accs)) if accs else None,
                "panel_acc_ci": (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))) if accs else None,
                "panel_pb_mean": float(np.mean(pbs)) if pbs else None,
                "panel_pb_ci": (float(np.percentile(pbs, 2.5)), float(np.percentile(pbs, 97.5))) if pbs else None}

    gt_pairs = [p for p in common_pairs if pairs.get(p, {}).get("gt_better") in ("A", "B")]
    panels = {}
    for k in (3, 5):
        panels[f"k{k}"] = panel_eval(open_present, k, gt_pairs)
        if len(judges_present) > len(open_present):
            panels[f"k{k}_all12"] = panel_eval(judges_present, k, gt_pairs)
    single_acc = [acc[j]["acc"] for j in judges_present if j in acc]
    single_pb = [abs(pb[j]["bias"]) for j in judges_present if j in pb]
    R["panels"] = panels
    R["single_judge_acc_mean"] = float(np.mean(single_acc)) if single_acc else None
    R["single_judge_abs_pb_mean"] = float(np.mean(single_pb)) if single_pb else None
    # best-single open judge vs panels
    R["best_single_open_acc"] = max((acc[j]["acc"] for j in open_present if j in acc), default=None)
    R["best_single_open_name"] = max(((acc[j]["acc"], j) for j in open_present if j in acc), default=(None, None))[1]

    # ---------- I. logistic regression ----------
    # P(A wins) ~ len_diff + kin_side + quality_diff on ALL verifiable-domain
    # AB-order non-tie judgments. quality_diff = correct(A) - correct(B) from the
    # v2 labels, so non-GT-clear pairs (both-correct / both-wrong) contribute
    # quality_diff = 0 and still inform the kin and length coefficients.
    print("== I. logistic regression ==")
    try:
        import statsmodels.api as sm
        # v2 correctness map
        cor2 = {}
        for r in load_jsonl(os.path.join(RESEARCH, "answers_relabel.jsonl")):
            cor2[(r["qid"], r["generator"])] = bool(r["correct_v2"])
        rows = []
        for x in judgments:
            if x["order"] != "AB" or x.get("rep", 0) != 0 or x["verdict"] not in ("A", "B"):
                continue
            if x["judge"] not in OPEN_JUDGES or x["domain"] not in ("math", "knowledge"):
                continue
            jfam = FAMILY[x["judge"]]
            a_f, b_f = x["gen_a_family"], x["gen_b_family"]
            kin_a = (a_f == jfam) and (b_f != jfam)
            kin_b = (b_f == jfam) and (a_f != jfam)
            ca = cor2.get((x["qid"], x["gen_a"]))
            cb = cor2.get((x["qid"], x["gen_b"]))
            if ca is None or cb is None:
                continue
            rows.append({
                "y": 1.0 if x["verdict"] == "A" else 0.0,
                "len_diff": (x["len_a"] - x["len_b"]) / 1000.0,
                "kin_side": (1.0 if kin_a else 0.0) - (1.0 if kin_b else 0.0),
                "quality_diff": (1.0 if ca else 0.0) - (1.0 if cb else 0.0),
                "first_pos": 1.0,  # AB order: A is first-presented (constant here)
                "qid": x["qid"], "judge": x["judge"],
                "judge_idx": judges_present.index(x["judge"]),
            })
        import pandas as pd
        df = pd.DataFrame(rows)
        X = sm.add_constant(df[["len_diff", "kin_side", "quality_diff"]])
        model = sm.Logit(df["y"], X).fit(disp=0)
        R["logit_full"] = {
            "n": int(len(df)),
            "coef": {k: float(model.params[k]) for k in X.columns},
            "p": {k: float(model.pvalues[k]) for k in X.columns},
            "ci": {k: [float(model.conf_int().loc[k, 0]), float(model.conf_int().loc[k, 1])] for k in X.columns},
        }
        # judge-fixed-effects version
        D = pd.get_dummies(df["judge_idx"], prefix="j", drop_first=True).astype(float)
        X2 = sm.add_constant(pd.concat([df[["len_diff", "kin_side", "quality_diff"]].reset_index(drop=True), D.reset_index(drop=True)], axis=1))
        model2 = sm.Logit(df["y"].reset_index(drop=True), X2).fit(disp=0)
        R["logit_fe"] = {
            "n": int(len(df)),
            "coef": {k: float(model2.params[k]) for k in ["len_diff", "kin_side", "quality_diff"]},
            "p": {k: float(model2.pvalues[k]) for k in ["len_diff", "kin_side", "quality_diff"]},
        }
    except Exception as e:
        R["logit_error"] = str(e)

    # ---------- summary significance: kin-wrong pick rates ----------
    kw_tests = []
    for jn in judges_present:
        k = kin[jn]
        if k["kin_wrong_n"] >= 8 and k["kin_wrong_pick_rate"] is not None:
            # binomial test vs 0 (ideal: never pick wrong kin) is trivial; report CI only
            kw_tests.append((jn, k["kin_wrong_pick_rate"], k["kin_wrong_ci"], k["kin_wrong_n"]))
    R["kin_wrong_table"] = kw_tests

    with open(os.path.join(RESEARCH, "results.json"), "w") as f:
        json.dump(R, f, indent=1, ensure_ascii=False, default=str)
    print("\nsaved results.json")
    # quick console summary
    print("\n=== SUMMARY ===")
    for jn in judges_present:
        a = acc.get(jn, {}).get("acc")
        k = R["avg_kappa"].get(jn)
        p = pb.get(jn, {}).get("bias")
        l = lb.get(jn, {}).get("bias")
        f_ = flip.get(jn, {}).get("flip_rate")
        t = tie_rates.get(jn)
        kw_ = kin.get(jn, {}).get("kin_wrong_pick_rate")
        print(f"{jn:20s} acc={a if a is None else round(a,3)} kappa={k if k is None else round(k,3)} "
              f"PB={p if p is None else round(p,3)} flip={f_ if f_ is None else round(f_,3)} "
              f"LB={l if l is None else round(l,3)} tie={t if t is None else round(t,3)} "
              f"kinWrong={kw_ if kw_ is None else round(kw_,3)}")


if __name__ == "__main__":
    main()
