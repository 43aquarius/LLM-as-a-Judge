#!/usr/bin/env python3
"""Phase B/D revision reanalysis (v2 metrics) — implements review items 5, 7, 9-15, 25, 26.

Differences from analyze.py (the frozen v1 pipeline behind the submitted paper):
  S1  [item 7]  truncation audit + sensitivity (seen-length LB, drop-truncated reanalysis)
  S2  [item 9]  three-tier accuracy (tie=wrong / tie-excluded+coverage / tie=0.5)
                + GT-clear tie rates + math/knowledge split [item 5 circularity]
  S3  [item 10] bidirectional position bias (both orders, pair-averaged)
  S4  [item 11] flip decomposition: hard (A<->B) vs soft (win<->TIE)
  S5  [item 12] cluster-robust + two-way (question x judge) SEs for the logit
                models; seen-length vs full-length length covariate [item 7]
  S6  [item 13] common-coverage reanalysis (all 10 open judges present)
  S7  [item 14] agreement robustness: Gwet AC1 + non-tie-subset kappa
  S8  [item 15] multiple-testing correction (Holm + BH) over per-judge tests
  S9  [item 25] per-family kinship: Wilson/Clopper-Pearson CIs + non-kin
                baseline error + regression-adjusted kinship + domain split
  S10 [item 26] panels: exact enumeration C(10,3)=120, C(10,5)=252, abstention
                rates, cross- vs same-family contrast, order-swap aggregation,
                question-cluster bootstrap CIs, signed single-judge PB baseline
  S12 [item 6]  GT-clear audit worksheet (CSV) for human verification

Outputs: research/reanalysis_results.json, research/reanalysis_report.md,
         research/gtclear_audit.csv
"""
import itertools
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

RESEARCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

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
GEN_FAMILY = {
    "Qwen3.8-Max": "Qwen", "Qwen3.8-Flash": "Qwen",
    "DeepSeek-V4-Pro": "DeepSeek", "GLM-5.3": "GLM", "Kimi-K3": "Kimi",
}

MAX_ANSWER_CHARS = 900   # judge.py clip threshold
B_BOOT = 2000
B_TEST = 1000


# ---------------------------------------------------------------- helpers
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


def clip(t, n):
    t = t.strip()
    return t if len(t) <= n else t[:n] + " ...[truncated]"


def math_final(content):
    m = re.findall(r"[Aa]nswer[:\s]*\$?([\d,.\-]+)", content)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?[\d,]*\d\.?\d*", content)
    return nums[-1].replace(",", "") if nums else None


def num_eq(a, b):
    if a is None or b is None:
        return False
    try:
        fa = float(str(a).replace(",", "").replace("$", "").strip())
        fb = float(str(b).replace(",", "").replace("$", "").strip())
    except ValueError:
        return False
    return abs(fa - fb) < 1e-6


def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (None, None)
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (None, None)
    from scipy import stats as st
    lo = 0.0 if k == 0 else st.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else st.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def cluster_boot(values, clusters, stat=np.mean, alpha=0.05, B=B_BOOT, seed_extra=""):
    """Question-level cluster bootstrap with content-hashed seed (deterministic)."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters).astype(str)
    if len(values) == 0:
        return (None, None, None)
    import hashlib
    seed = int(hashlib.md5(values.tobytes() + clusters.tobytes()
                           + seed_extra.encode()).hexdigest()[:8], 16)
    brng = np.random.default_rng(seed)
    uc = np.unique(clusters)
    idx_by_c = {c: np.where(clusters == c)[0] for c in uc}
    stats = []
    for _ in range(B):
        cs = brng.choice(uc, size=len(uc), replace=True)
        samp = np.concatenate([idx_by_c[c] for c in cs])
        try:
            stats.append(stat(values[samp]))
        except Exception:
            pass
    if not stats:
        return (None, None, None)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    se = float(np.std(stats, ddof=1)) if len(stats) > 1 else None
    return (float(lo), float(hi), se)


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


def gwet_ac1(a, b, labels=("A", "B", "TIE")):
    """Gwet AC1 for two raters, q categories (tie-prevalence robust)."""
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    if n == 0:
        return np.nan
    po = float((a == b).mean())
    q = len(labels)
    if q <= 1:
        return np.nan
    e = 0.0
    for l in labels:
        pi = (float((a == l).mean()) + float((b == l).mean())) / 2
        e += pi * (1 - pi)
    e = e / (q - 1)
    if e >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - e) / (1 - e)


def holm_bh(pvals):
    """Return (holm adjusted, bh adjusted) arrays."""
    from scipy import stats as st
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p, p
    order = np.argsort(p)
    # Holm (family-wise)
    holm_sorted = p[order] * (n - np.arange(n))
    holm_sorted = np.maximum.accumulate(holm_sorted)
    holm = np.empty(n); holm[order] = np.clip(holm_sorted, 0, 1)
    # BH (FDR)
    bh_sorted = p[order] * n / (np.arange(n) + 1)
    bh_sorted = np.minimum.accumulate(bh_sorted[::-1])[::-1]
    bh = np.empty(n); bh[order] = np.clip(bh_sorted, 0, 1)
    return holm, bh


def two_prop_test(k1, n1, k2, n2):
    """Two-proportion z-test (pooled). Returns (diff, z, p)."""
    if n1 == 0 or n2 == 0:
        return (None, None, None)
    from scipy import stats as st
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return (p1 - p2, None, None)
    z = (p1 - p2) / se
    return (p1 - p2, z, float(2 * (1 - st.norm.cdf(abs(z)))))


def fisher_exact_2x2(k1, n1, k2, n2):
    from scipy import stats as st
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    _, p = st.fisher_exact(table)
    return float(p)


# ---------------------------------------------------------------- main
def main():
    R = {}

    pairs = {p["pair_id"]: p for p in load_jsonl(os.path.join(RESEARCH, "pairs.jsonl"))}
    questions = {q["qid"]: q for q in load_jsonl(os.path.join(RESEARCH, "questions.jsonl"))}
    answers = {(a["qid"], a["generator"]): a
               for a in load_jsonl(os.path.join(RESEARCH, "answers.jsonl"))}
    relabel = {(r["qid"], r["generator"]): r
               for r in load_jsonl(os.path.join(RESEARCH, "answers_relabel.jsonl"))}
    all_judg_raw = (load_jsonl(os.path.join(RESEARCH, "judgments.jsonl"))
                    + load_jsonl(os.path.join(RESEARCH, "judgments_agnes.jsonl")))
    judgments = [j for j in all_judg_raw if j.get("parsed_ok") and j.get("rep", 0) == 0]
    judges_present = [j for j in JUDGE_ORDER if any(x["judge"] == j for x in judgments)]

    # derived: full lengths recomputed from text; seen lengths after judge.py clip
    for p in pairs.values():
        p["full_a"] = len(p["text_a"])
        p["full_b"] = len(p["text_b"])
        p["seen_a"] = min(p["full_a"], MAX_ANSWER_CHARS)
        p["seen_b"] = min(p["full_b"], MAX_ANSWER_CHARS)
        p["trunc_a"] = p["full_a"] > MAX_ANSWER_CHARS
        p["trunc_b"] = p["full_b"] > MAX_ANSWER_CHARS

    R["meta"] = {
        "n_pairs": len(pairs), "n_judgments_parsed": len(judgments),
        "judges": judges_present,
        "gt_clear_pairs": sum(1 for p in pairs.values() if p.get("gt_better") in ("A", "B")),
    }

    # ================================================================ S1 [7]
    print("== S1. truncation audit ==")
    trunc = {"by_domain": {}, "by_generator": {}, "math_final_hidden": {},
             "generation_side": {}, "judge_output": {}, "ref_trunc": {}}
    for dom in ["math", "knowledge", "creative", "summ"]:
        ans = [a for (qid, g), a in answers.items() if a["domain"] == dom]
        trunc["by_domain"][dom] = {
            "n_answers": len(ans),
            "pct_over_900chars": round(100 * sum(1 for a in ans if len(a["content"]) > MAX_ANSWER_CHARS) / max(1, len(ans)), 1),
        }
    for g in GEN_FAMILY:
        ans = [a for (qid, gg), a in answers.items() if gg == g]
        trunc["by_generator"][g] = round(100 * sum(1 for a in ans if len(a["content"]) > MAX_ANSWER_CHARS) / max(1, len(ans)), 1)
    # math: is the extracted final answer still visible in the clipped text?
    hidden = tot = 0
    hidden_examples = []
    for (qid, g), a in answers.items():
        if a["domain"] != "math":
            continue
        tot += 1
        f_full = math_final(a["content"])
        f_seen = math_final(clip(a["content"], MAX_ANSWER_CHARS))
        if (f_full is not None) and (f_seen is None or not num_eq(f_seen, f_full)):
            hidden += 1
            if len(hidden_examples) < 8:
                hidden_examples.append({"qid": qid, "gen": g, "full": f_full, "seen": f_seen})
    trunc["math_final_hidden"] = {"n_math": tot, "n_final_hidden": hidden,
                                  "pct": round(100 * hidden / max(1, tot), 1),
                                  "examples": hidden_examples}
    # generation-side: completion tokens at the cap => likely truncated generation
    gen_cap = 0
    gen_tot = 0
    for a in answers.values():
        u = a.get("usage") or {}
        ct = u.get("completion_tokens")
        if ct is None:
            continue
        gen_tot += 1
        cap = 1600 if a.get("generator_model") == "glm-5.3" else 900
        if ct >= cap - 2:
            gen_cap += 1
    trunc["generation_side"] = {"n_with_usage": gen_tot, "n_at_cap": gen_cap,
                                "pct": round(100 * gen_cap / max(1, gen_tot), 2)}
    # judge output side
    jo = {}
    for jn in judges_present:
        cap = 900 if jn in ("GLM-5.3", "Qwen3.8-2.4T") else 400
        js = [x for x in all_judg_raw if x["judge"] == jn and x.get("rep", 0) == 0]
        at_cap = sum(1 for x in js if (x.get("usage") or {}).get("completion_tokens", 0) >= cap - 2)
        jo[jn] = {"n": len(js), "at_cap": at_cap, "pct": round(100 * at_cap / max(1, len(js)), 2)}
    trunc["judge_output"] = jo
    # reference answers clipped at 400 chars in judge prompt
    refs_kn = [q for q in questions.values() if q["domain"] == "knowledge"]
    trunc["ref_trunc"] = {"n_knowledge_refs": len(refs_kn),
                          "n_over_400": sum(1 for q in refs_kn if len(str(q.get("reference") or "")) > 400)}
    # pair-level truncation
    trunc["pairs_any_truncated"] = sum(1 for p in pairs.values() if p["trunc_a"] or p["trunc_b"])
    trunc["gtclear_any_truncated"] = sum(1 for p in pairs.values()
                                         if p.get("gt_better") in ("A", "B") and (p["trunc_a"] or p["trunc_b"]))
    R["S1_truncation"] = trunc

    # ================================================================ S2 [9] + S11 [5]
    print("== S2. three-tier accuracy + GT-clear tie rates ==")
    acc3 = {}
    for jn in judges_present:
        gt_j = [x for x in judgments if x["judge"] == jn and x["order"] == "AB"
                and pairs.get(x["pair_id"], {}).get("gt_better") in ("A", "B")]
        rec = {}
        for dom_key, doms in [("all", ("math", "knowledge")),
                              ("math", ("math",)), ("knowledge", ("knowledge",))]:
            sub = [x for x in gt_j if x["domain"] in doms]
            n = len(sub)
            if n == 0:
                continue
            correct = sum(1 for x in sub if x["verdict"] == pairs[x["pair_id"]]["gt_better"])
            ties = sum(1 for x in sub if x["verdict"] == "TIE")
            decided = n - ties
            vals = [1.0 if x["verdict"] == pairs[x["pair_id"]]["gt_better"] else 0.0 for x in sub]
            lo, hi, _ = cluster_boot(vals, [x["qid"] for x in sub], seed_extra=f"acc3-{jn}-{dom_key}")
            rec[dom_key] = {
                "n": n,
                "acc_tie_wrong": round(correct / n, 3),
                "acc_tie_excluded": round(correct / decided, 3) if decided else None,
                "coverage_decided": round(decided / n, 3),
                "acc_tie_half": round((correct + 0.5 * ties) / n, 3),
                "tie_rate_gtclear": round(ties / n, 3),
                "acc_ci_boot": [round(lo, 3) if lo is not None else None,
                                round(hi, 3) if hi is not None else None],
            }
        acc3[jn] = rec
    R["S2_accuracy3"] = acc3

    # ================================================================ S3 [10]
    print("== S3. bidirectional position bias ==")
    pb2 = {}
    for jn in judges_present:
        by_pair = defaultdict(dict)
        for x in judgments:
            if x["judge"] == jn:
                by_pair[x["pair_id"]][x["order"]] = x["verdict"]
        vals, clus = [], []
        for pid, d in by_pair.items():
            inds = []
            if d.get("AB") in ("A", "B"):
                inds.append(1.0 if d["AB"] == "A" else 0.0)   # A presented first in AB
            if d.get("BA") in ("A", "B"):
                inds.append(1.0 if d["BA"] == "B" else 0.0)   # B presented first in BA
            if inds:
                vals.append(float(np.mean(inds)))
                clus.append(pid.rsplit("-", 1)[0])
        if vals:
            lo, hi, se = cluster_boot(vals, clus, seed_extra=f"pb2-{jn}")
            mean = float(np.mean(vals))
            from scipy import stats as st
            z = (mean - 0.5) / se if se else None
            p = float(2 * (1 - st.norm.cdf(abs(z)))) if z is not None else None
            pb2[jn] = {"first_win_rate_bd": round(mean, 3), "bias_bd": round(mean - 0.5, 3),
                       "ci": [round(lo, 3), round(hi, 3)], "n_pairs": len(vals),
                       "z": round(z, 2) if z is not None else None,
                       "p": p}
    R["S3_pb_bidirectional"] = pb2

    # ================================================================ S4 [11]
    print("== S4. flip decomposition ==")
    flipd = {}
    for jn in judges_present:
        by_pair = defaultdict(dict)
        for x in judgments:
            if x["judge"] == jn:
                by_pair[x["pair_id"]][x["order"]] = x["verdict"]
        hard, soft, cons, clus = [], [], [], []
        for pid, d in by_pair.items():
            if "AB" not in d or "BA" not in d:
                continue
            a, b = d["AB"], d["BA"]
            q = pid.rsplit("-", 1)[0]
            if a == b:
                cons.append(0.0)
            elif a in ("A", "B") and b in ("A", "B"):
                cons.append(1.0)   # hard flip
            else:
                cons.append(0.0)
            hard.append(1.0 if (a != b and a in ("A", "B") and b in ("A", "B")) else 0.0)
            soft.append(1.0 if (a != b and (a == "TIE" or b == "TIE")) else 0.0)
            clus.append(q)
        n = len(hard)
        if n:
            h = float(np.mean(hard)); s = float(np.mean(soft))
            hlo, hhi, _ = cluster_boot(hard, clus, seed_extra=f"fh-{jn}")
            slo, shi, _ = cluster_boot(soft, clus, seed_extra=f"fs-{jn}")
            flipd[jn] = {"n": n, "hard_flip": round(h, 3), "soft_flip": round(s, 3),
                         "total_flip": round(h + s, 3),
                         "hard_ci": [round(hlo, 3), round(hhi, 3)],
                         "soft_ci": [round(slo, 3), round(shi, 3)]}
    R["S4_flip_decomposition"] = flipd

    # ================================================================ S5 [12]
    print("== S5. cluster-robust regressions ==")
    reg = {}
    try:
        import pandas as pd
        import statsmodels.api as sm
        cor2 = {(r["qid"], r["generator"]): bool(r["correct_v2"])
                for r in load_jsonl(os.path.join(RESEARCH, "answers_relabel.jsonl"))}
        rows = []
        for x in judgments:
            if x["order"] != "AB" or x["verdict"] not in ("A", "B"):
                continue
            if x["judge"] not in OPEN_JUDGES or x["domain"] not in ("math", "knowledge"):
                continue
            p = pairs[x["pair_id"]]
            jfam = FAMILY[x["judge"]]
            a_f, b_f = GEN_FAMILY[p["gen_a"]], GEN_FAMILY[p["gen_b"]]
            kin_a = (a_f == jfam) and (b_f != jfam)
            kin_b = (b_f == jfam) and (a_f != jfam)
            ca = cor2.get((x["qid"], p["gen_a"]))
            cb = cor2.get((x["qid"], p["gen_b"]))
            if ca is None or cb is None:
                continue
            rows.append({
                "y": 1.0 if x["verdict"] == "A" else 0.0,
                "len_diff_seen": (p["seen_a"] - p["seen_b"]) / 1000.0,
                "len_diff_full": (p["full_a"] - p["full_b"]) / 1000.0,
                "kin_side": (1.0 if kin_a else 0.0) - (1.0 if kin_b else 0.0),
                "quality_diff": (1.0 if ca else 0.0) - (1.0 if cb else 0.0),
                "judge": x["judge"], "judge_family": jfam, "qid": x["qid"],
            })
        df = pd.DataFrame(rows)
        reg["n_obs"] = int(len(df))

        def fit_robust(frame, covars, cluster1, cluster2=None, label=""):
            X = sm.add_constant(frame[covars])
            y = frame["y"]
            m0 = sm.Logit(y, X).fit(disp=0)
            out = {"label": label, "n": int(len(df)), "specs": {}}
            # naive
            out["specs"]["naive"] = {
                c: {"coef": float(m0.params[c]), "p": float(m0.pvalues[c]),
                    "ci": [float(m0.conf_int().loc[c, 0]), float(m0.conf_int().loc[c, 1])]}
                for c in covars}
            # cluster by question
            m1 = sm.Logit(y, X).fit(disp=0, cov_type="cluster",
                                    cov_kwds={"groups": np.asarray(frame[cluster1])})
            out["specs"][f"cluster_{cluster1}"] = {
                c: {"coef": float(m1.params[c]), "se": float(m1.bse[c]),
                    "z": float(m1.tvalues[c]), "p": float(m1.pvalues[c]),
                    "ci": [float(m1.conf_int().loc[c, 0]), float(m1.conf_int().loc[c, 1])]}
                for c in covars}
            if cluster2 is not None:
                m2 = sm.Logit(y, X).fit(disp=0, cov_type="cluster",
                                        cov_kwds={"groups": np.asarray(frame[cluster2])})
                # two-way CGM: V = V_c1 + V_c2 - V_c1xc2 (obs-level cluster ~ robust)
                inter = np.asarray([f"{a}|{b}" for a, b in zip(frame[cluster1], frame[cluster2])])
                m3 = sm.Logit(y, X).fit(disp=0, cov_type="cluster",
                                        cov_kwds={"groups": inter})
                V = (np.asarray(m1.cov_params()) + np.asarray(m2.cov_params())
                     - np.asarray(m3.cov_params()))
                from scipy import stats as st
                spec = {}
                for c in covars:
                    i = list(X.columns).index(c)
                    se = math.sqrt(max(V[i, i], 0.0))
                    z = float(m1.params[c]) / se if se > 0 else None
                    spec[c] = {"coef": float(m1.params[c]), "se": se,
                               "z": z if z is None else round(z, 2),
                               "p": float(2 * (1 - st.norm.cdf(abs(z)))) if z is not None else None,
                               "ci": [float(m1.params[c] - 1.96 * se), float(m1.params[c] + 1.96 * se)]}
                out["specs"]["two_way_qid_judge"] = spec
            return out

        # main spec: seen length + kin + quality (+ judge FE via within transform not needed;
        # judge FE added as dummies for the cluster fits)
        D = pd.get_dummies(df["judge"], prefix="j", drop_first=True).astype(float)
        dfx = pd.concat([df.reset_index(drop=True), D.reset_index(drop=True)], axis=1)
        fe_cols = list(D.columns)
        reg["main_seen_len"] = fit_robust(dfx, ["len_diff_seen", "kin_side", "quality_diff"] + fe_cols,
                                          "qid", "judge", "main: seen length + FE")
        reg["main_full_len"] = fit_robust(dfx, ["len_diff_full", "kin_side", "quality_diff"] + fe_cols,
                                          "qid", "judge", "main: full length + FE")

        # per-family kinship interaction (item 25): kin x judge-family
        dfi = df.copy()
        for fam in ["Qwen", "DeepSeek", "GLM", "Kimi"]:
            dfi[f"kin_{fam}"] = dfi["kin_side"] * (dfi["judge_family"] == fam).astype(float)
        D2 = pd.get_dummies(dfi["judge"], prefix="j", drop_first=True).astype(float)
        dfi = pd.concat([dfi.reset_index(drop=True), D2.reset_index(drop=True)], axis=1)
        kin_fam_cols = ["kin_Qwen", "kin_DeepSeek", "kin_GLM", "kin_Kimi"]
        # NOTE: kin_side main effect is the base; per-family slopes = kin_side + kin_<fam>
        # cleaner: drop the main kin_side and use family-specific kin columns directly
        X_cols = ["len_diff_seen", "quality_diff"] + kin_fam_cols + list(D2.columns)
        X = sm.add_constant(dfi[X_cols])
        mf = sm.Logit(dfi["y"], X).fit(disp=0, cov_type="cluster",
                                       cov_kwds={"groups": np.asarray(dfi["qid"])})
        reg["kin_by_family_cluster_qid"] = {
            c: {"coef": float(mf.params[c]), "se": float(mf.bse[c]),
                "z": float(mf.tvalues[c]), "p": float(mf.pvalues[c]),
                "ci": [float(mf.conf_int().loc[c, 0]), float(mf.conf_int().loc[c, 1])]}
            for c in ["len_diff_seen", "quality_diff"] + kin_fam_cols}
        reg["kin_by_family_n"] = int(len(dfi))
    except Exception as e:
        import traceback
        reg["error"] = str(e) + " | " + traceback.format_exc()[-400:]
    R["S5_regressions"] = reg

    # ================================================================ S6 [13]
    print("== S6. common-coverage reanalysis ==")
    by_pair_j = defaultdict(dict)
    for x in judgments:
        if x["order"] == "AB":
            by_pair_j[x["pair_id"]][x["judge"]] = x["verdict"]
    by_pair_both = defaultdict(dict)   # pair -> judge -> (AB, BA)
    for x in judgments:
        by_pair_both[x["pair_id"]].setdefault(x["judge"], {})[x["order"]] = x["verdict"]
    open_present = [j for j in OPEN_JUDGES if j in judges_present]
    common_ab = [pid for pid, d in by_pair_j.items()
                 if all(j in d for j in open_present)]
    common_abba = [pid for pid, d in by_pair_both.items()
                   if all(d.get(j, {}).get("AB") is not None and d.get(j, {}).get("BA") is not None
                          for j in open_present)]
    # domain composition: common vs full (selection-bias check for GLM/2.4T exhaustion)
    dom_common = defaultdict(int)
    for pid in common_ab:
        dom_common[pairs[pid]["domain"]] += 1
    dom_full = defaultdict(int)
    for pid in by_pair_j:
        dom_full[pairs[pid]["domain"]] += 1
    # per-judge metrics: full vs common
    cc = {"n_common_AB": len(common_ab), "n_common_ABBA": len(common_abba),
          "domain_common": dict(dom_common), "domain_full": dict(dom_full),
          "per_judge": {}}
    for jn in open_present:
        fullrec, comrec = {}, {}
        # accuracy (GT-clear)
        for tag, pset in [("full", [pid for pid in by_pair_j if jn in by_pair_j[pid]]),
                          ("common", common_ab)]:
            gt = [pid for pid in pset if pairs[pid].get("gt_better") in ("A", "B")
                  and jn in by_pair_j.get(pid, {})]
            cor = sum(1 for pid in gt if by_pair_j[pid][jn] == pairs[pid]["gt_better"])
            ties_all = sum(1 for pid in pset if by_pair_j[pid][jn] == "TIE")
            nt = [pid for pid in pset if by_pair_j[pid][jn] in ("A", "B")]
            longer = 0
            for pid in nt:
                p = pairs[pid]
                longer += int(by_pair_j[pid][jn] == ("A" if p["full_a"] > p["full_b"] else "B"))
            rec = {"acc": round(cor / len(gt), 3) if gt else None,
                   "tie": round(ties_all / len(pset), 3),
                   "lb": round(longer / len(nt) - 0.5, 3) if nt else None}
            # bidirectional PB on this set
            vals = []
            for pid in pset:
                d = by_pair_both.get(pid, {}).get(jn, {})
                inds = []
                if d.get("AB") in ("A", "B"):
                    inds.append(1.0 if d["AB"] == "A" else 0.0)
                if d.get("BA") in ("A", "B"):
                    inds.append(1.0 if d["BA"] == "B" else 0.0)
                if inds:
                    vals.append(float(np.mean(inds)))
            rec["pb_bd"] = round(float(np.mean(vals)) - 0.5, 3) if vals else None
            (fullrec if tag == "full" else comrec).update(rec)
        cc["per_judge"][jn] = {"full": fullrec, "common": comrec}
    # agreement on the common set
    ks_common = []
    for a, b in itertools.combinations(open_present, 2):
        va = [by_pair_j[p][a] for p in common_ab]
        vb = [by_pair_j[p][b] for p in common_ab]
        ks_common.append(cohen_kappa(va, vb))
    cc["mean_kappa_common"] = round(float(np.nanmean(ks_common)), 3) if ks_common else None
    R["S6_common_coverage"] = cc

    # ================================================================ S7 [14]
    print("== S7. agreement robustness ==")
    agr = {"per_pair": {}, "mean_per_judge": {}}
    kappas, ac1s, kappas_nt = defaultdict(list), defaultdict(list), defaultdict(list)
    for a, b in itertools.combinations(judges_present, 2):
        ps = [p for p, d in by_pair_j.items() if a in d and b in d]
        if not ps:
            continue
        va = [by_pair_j[p][a] for p in ps]
        vb = [by_pair_j[p][b] for p in ps]
        k = cohen_kappa(va, vb)
        g = gwet_ac1(va, vb)
        # non-tie subset kappa (both decided)
        sub = [(x, y) for x, y in zip(va, vb) if x != "TIE" and y != "TIE"]
        knt = cohen_kappa([x for x, _ in sub], [y for _, y in sub], labels=("A", "B")) if sub else np.nan
        agr["per_pair"][f"{a}||{b}"] = {"kappa": round(float(k), 3), "ac1": round(float(g), 3),
                                        "kappa_nontie": round(float(knt), 3) if not np.isnan(knt) else None,
                                        "n": len(ps), "n_nontie": len(sub)}
        for j in (a, b):
            kappas[j].append(k); ac1s[j].append(g); kappas_nt[j].append(knt)
    for j in judges_present:
        agr["mean_per_judge"][j] = {
            "kappa": round(float(np.nanmean(kappas[j])), 3),
            "ac1": round(float(np.nanmean(ac1s[j])), 3),
            "kappa_nontie": round(float(np.nanmean(kappas_nt[j])), 3),
        }
    allk = [v["kappa"] for v in agr["per_pair"].values()]
    allg = [v["ac1"] for v in agr["per_pair"].values()]
    agr["overall"] = {"kappa_range": [round(min(allk), 3), round(max(allk), 3)],
                      "ac1_range": [round(min(allg), 3), round(max(allg), 3)],
                      "corr_kappa_ac1": round(float(np.corrcoef(allk, allg)[0, 1]), 3)}
    R["S7_agreement"] = agr

    # ================================================================ S8 [15]
    print("== S8. multiple-testing corrections ==")
    tests = []
    for jn in open_present:
        if jn in pb2 and pb2[jn].get("p") is not None:
            tests.append({"judge": jn, "metric": "PB_bd", "est": pb2[jn]["bias_bd"],
                          "p": pb2[jn]["p"]})
    # LB with cluster SE
    for jn in open_present:
        vals, clus = [], []
        for x in judgments:
            if x["judge"] == jn and x["order"] == "AB" and x["verdict"] in ("A", "B"):
                p = pairs[x["pair_id"]]
                longer = "A" if p["full_a"] > p["full_b"] else "B"
                vals.append(1.0 if x["verdict"] == longer else 0.0)
                clus.append(x["qid"])
        if vals:
            lo, hi, se = cluster_boot(vals, clus, seed_extra=f"lb-{jn}", B=B_TEST)
            mean = float(np.mean(vals))
            from scipy import stats as st
            z = (mean - 0.5) / se if se else None
            p = float(2 * (1 - st.norm.cdf(abs(z)))) if z is not None else None
            tests.append({"judge": jn, "metric": "LB", "est": round(mean - 0.5, 3), "p": p})
    # kin-win vs 0.5 (non-tie kin pairs)
    for jn in open_present:
        jfam = FAMILY[jn]
        vals, clus = [], []
        for x in judgments:
            if x["judge"] != jn or x["order"] != "AB" or x["verdict"] not in ("A", "B"):
                continue
            p = pairs[x["pair_id"]]
            a_f, b_f = GEN_FAMILY[p["gen_a"]], GEN_FAMILY[p["gen_b"]]
            kin_side = "A" if (a_f == jfam and b_f != jfam) else ("B" if (b_f == jfam and a_f != jfam) else None)
            if kin_side is None:
                continue
            vals.append(1.0 if x["verdict"] == kin_side else 0.0)
            clus.append(x["qid"])
        if vals:
            lo, hi, se = cluster_boot(vals, clus, seed_extra=f"kw-{jn}", B=B_TEST)
            mean = float(np.mean(vals))
            from scipy import stats as st
            z = (mean - 0.5) / se if se else None
            p = float(2 * (1 - st.norm.cdf(abs(z)))) if z is not None else None
            tests.append({"judge": jn, "metric": "kin_win", "est": round(mean, 3), "p": p})
    ps = np.array([t["p"] for t in tests], dtype=float)
    holm, bh = holm_bh(ps)
    for t, h, b_ in zip(tests, holm, bh):
        t["p_holm"] = float(h)
        t["p_bh"] = float(b_)
        t["sig_raw"] = bool(t["p"] < 0.05)
        t["sig_holm"] = bool(h < 0.05)
        t["sig_bh"] = bool(b_ < 0.05)
    R["S8_multiple_testing"] = tests

    # ================================================================ S9 [25]
    print("== S9. kinship redo (per family) ==")
    kin = {"by_family": {}, "by_judge": {}, "domain_split": {}}
    for fam in ["Qwen", "DeepSeek", "GLM", "Kimi"]:
        fam_judges = [j for j in open_present if FAMILY[j] == fam]
        kw_k = kw_n = 0          # kin-wrong picks
        kr_k = kr_n = 0          # kin-right picks (sanity)
        nk_k = nk_n = 0          # non-kin baseline errors (GT-clear, no kin side)
        win_k = win_n = 0        # kin-win overall
        math_k = math_n = kn_math = 0
        know_k = know_n = kn_know = 0
        for x in judgments:
            if x["order"] != "AB" or x["verdict"] not in ("A", "B"):
                continue
            if x["judge"] not in fam_judges:
                continue
            p = pairs[x["pair_id"]]
            a_f, b_f = GEN_FAMILY[p["gen_a"]], GEN_FAMILY[p["gen_b"]]
            kin_side = "A" if (a_f == fam and b_f != fam) else ("B" if (b_f == fam and a_f != fam) else None)
            pick = x["verdict"]
            win_n += 1
            if kin_side is None:
                # non-kin GT-clear pair: baseline error = picking the wrong side
                gt = p.get("gt_better")
                if gt in ("A", "B"):
                    nk_n += 1
                    nk_k += int(pick != gt)
                continue
            win_k += int(pick == kin_side)
            gt = p.get("gt_better")
            if gt in ("A", "B"):
                kin_correct = (gt == kin_side)
                if kin_correct:
                    kr_n += 1
                    kr_k += int(pick == kin_side)
                else:
                    kw_n += 1
                    kw_k += int(pick == kin_side)
            # domain split of kin-win (circularity check)
            if x["domain"] == "math":
                math_n += 1
                math_k += int(pick == kin_side)
            elif x["domain"] == "knowledge":
                know_n += 1
                kn_know += int(pick == kin_side)
        wlo, whi = wilson_ci(kw_k, kw_n)
        wloE, whiE = clopper_pearson(kw_k, kw_n)
        nlo, nhi = wilson_ci(nk_k, nk_n)
        diff, z, pz = two_prop_test(kw_k, kw_n, nk_k, nk_n)
        pf = fisher_exact_2x2(kw_k, kw_n, nk_k, nk_n) if kw_n and nk_n else None
        winlo, winhi = wilson_ci(win_k, win_n)
        kin["by_family"][fam] = {
            "kin_wrong": {"k": kw_k, "n": kw_n,
                          "rate": round(kw_k / kw_n, 3) if kw_n else None,
                          "wilson": [round(wlo, 3), round(whi, 3)] if kw_n else None,
                          "exact_cp": [round(wloE, 3), round(whiE, 3)] if kw_n else None},
            "nonkin_baseline_error": {"k": nk_k, "n": nk_n,
                                      "rate": round(nk_k / nk_n, 3) if nk_n else None,
                                      "wilson": [round(nlo, 3), round(nhi, 3)] if nk_n else None},
            "kin_wrong_vs_baseline": {"diff": round(diff, 3) if diff is not None else None,
                                      "z": round(z, 2) if z is not None else None,
                                      "p_z": pz, "p_fisher": pf},
            "kin_right": {"k": kr_k, "n": kr_n,
                          "rate": round(kr_k / kr_n, 3) if kr_n else None},
            "kin_win": {"k": win_k, "n": win_n,
                        "rate": round(win_k / win_n, 3) if win_n else None,
                        "wilson": [round(winlo, 3), round(winhi, 3)] if win_n else None},
            "kin_win_math_only": {"k": math_k, "n": math_n,
                                  "rate": round(math_k / math_n, 3) if math_n else None},
            "kin_win_knowledge_only": {"k": kn_know, "n": know_n,
                                       "rate": round(kn_know / know_n, 3) if know_n else None},
        }
    # per-judge kin-wrong with Wilson CI (small n)
    for jn in open_present:
        jfam = FAMILY[jn]
        kw_k = kw_n = 0
        for x in judgments:
            if x["judge"] != jn or x["order"] != "AB" or x["verdict"] not in ("A", "B"):
                continue
            p = pairs[x["pair_id"]]
            a_f, b_f = GEN_FAMILY[p["gen_a"]], GEN_FAMILY[p["gen_b"]]
            kin_side = "A" if (a_f == jfam and b_f != jfam) else ("B" if (b_f == jfam and a_f != jfam) else None)
            if kin_side is None:
                continue
            gt = p.get("gt_better")
            if gt in ("A", "B") and gt != kin_side:
                kw_n += 1
                kw_k += int(x["verdict"] == kin_side)
        lo, hi = wilson_ci(kw_k, kw_n)
        kin["by_judge"][jn] = {"k": kw_k, "n": kw_n,
                               "rate": round(kw_k / kw_n, 3) if kw_n else None,
                               "wilson": [round(lo, 3), round(hi, 3)] if kw_n else None}
    R["S9_kinship"] = kin

    # ================================================================ S10 [26]
    print("== S10. panels redo (exact enumeration) ==")
    panels = {}
    gt_pairs = [pid for pid in by_pair_j
                if pairs[pid].get("gt_better") in ("A", "B")]
    # verdict matrices: rows = gt pairs, cols = open judges; +1 A, -1 B, 0 TIE, NaN missing
    idx = {pid: i for i, pid in enumerate(gt_pairs)}
    jidx = {j: i for i, j in enumerate(open_present)}
    Vmat = np.full((len(gt_pairs), len(open_present)), np.nan)
    PBmat = np.full_like(Vmat, np.nan)   # first-win indicator (A first in AB) among decided
    for pid in gt_pairs:
        for j in open_present:
            v = by_pair_j.get(pid, {}).get(j)
            if v is None:
                continue
            Vmat[idx[pid], jidx[j]] = {"A": 1.0, "B": -1.0, "TIE": 0.0}[v]
    gt_sign = np.array([1.0 if pairs[pid]["gt_better"] == "A" else -1.0 for pid in gt_pairs])

    def eval_panels(combos, k):
        """combos: array (C, k) of judge column indices. Returns dict of metrics."""
        sub = Vmat[:, combos]                      # (P, C, k)
        valid = ~np.isnan(sub).any(axis=2)         # panel usable if all k present
        a = (sub == 1.0).sum(axis=2).astype(float)
        b = (sub == -1.0).sum(axis=2).astype(float)
        dec_a = a > b
        dec_b = b > a
        abstain = ~(dec_a | dec_b)
        correct = ((dec_a & (gt_sign[:, None] > 0)) | (dec_b & (gt_sign[:, None] < 0)))
        with np.errstate(invalid="ignore"):
            usable = valid
        acc_T = np.where(usable, correct, np.nan)              # abstain counts as wrong
        cov = np.where(usable, (dec_a | dec_b), np.nan)        # decided fraction
        acc_D = np.where(usable & (dec_a | dec_b), correct, np.nan)
        # panel PB: first position = A (canonical order) among decided
        first_win = dec_a
        pb = np.where(usable & (dec_a | dec_b), first_win.astype(float) - 0.5, np.nan)
        out = {
            "n_combos": int(combos.shape[0]),
            "acc_tie_wrong": {"mean": float(np.nanmean(acc_T)),
                              "min": float(np.nanmin(acc_T)), "max": float(np.nanmax(acc_T)),
                              "pct_ge_best_single": float(np.mean(np.nanmean(acc_T, axis=0) >= 0.714))},
            "abstain_rate": float(np.nanmean(np.where(usable, abstain, np.nan))),
            "acc_decided": float(np.nanmean(acc_D)),
            "pb_signed": float(np.nanmean(pb)),
        }
        return out, acc_T, usable

    results_enum = {}
    accT_by_k = {}
    for k in (3, 5):
        combos = np.array(list(itertools.combinations(range(len(open_present)), k)))
        res, accT, usable = eval_panels(combos, k)
        # question-cluster bootstrap of the mean-over-panels acc (resample questions)
        qids = [pid.rsplit("-", 1)[0] for pid in gt_pairs]
        uq = np.unique(qids)
        qmap = defaultdict(list)
        for i, q in enumerate(qids):
            qmap[q].append(i)
        brng = np.random.default_rng(20261017)
        boots = []
        for _ in range(B_BOOT):
            qs = brng.choice(uq, size=len(uq), replace=True)
            rows = np.concatenate([qmap[q] for q in qs])
            subA = accT[rows]
            m = np.nanmean(np.where(np.isnan(subA), np.nan, subA))
            if not np.isnan(m):
                boots.append(m)
        res["acc_tie_wrong"]["boot_ci"] = [round(float(np.percentile(boots, 2.5)), 3),
                                           round(float(np.percentile(boots, 97.5)), 3)]
        results_enum[f"k{k}"] = res
        accT_by_k[k] = accT
    panels["enumerated_open"] = results_enum

    # cross-family vs same-family (k=3)
    combos3 = list(itertools.combinations(range(len(open_present)), 3))
    fams_of = [FAMILY[open_present[i]] for i in range(len(open_present))]
    cross_idx, same_idx = [], []
    for ci, cb in enumerate(combos3):
        fs = {fams_of[i] for i in cb}
        (cross_idx if len(fs) >= 3 else same_idx).append(ci)
        # note: "cross" here = 3 distinct families (maximally diverse)
    def combo_group_metrics(idxs, k):
        arr = np.array(list(itertools.combinations(range(len(open_present)), k)))
        sub = arr[idxs]
        _, accT, usable = eval_panels(sub, k)
        return {"n_panels": len(idxs), "acc_mean": round(float(np.nanmean(accT)), 3)}
    panels["k3_cross3fam"] = combo_group_metrics(cross_idx, 3)
    panels["k3_other"] = combo_group_metrics(same_idx, 3)
    # finer: pure same-family panels (all judges one family) vs 2-family vs 3-family
    pure, two, three = [], [], []
    for ci, cb in enumerate(combos3):
        fs = {fams_of[i] for i in cb}
        if len(fs) == 1:
            pure.append(ci)
        elif len(fs) == 2:
            two.append(ci)
        else:
            three.append(ci)
    panels["k3_same1fam"] = combo_group_metrics(pure, 3)
    panels["k3_two2fam"] = combo_group_metrics(two, 3)
    panels["k3_three3fam"] = combo_group_metrics(three, 3)

    # order-swap aggregated panels (item 26): judge-level both-order aggregation first
    W = np.full((len(gt_pairs), len(open_present)), np.nan)  # +1 both A, -1 both B, 0 both TIE, NaN inconsistent/mixed
    for pid in gt_pairs:
        for j in open_present:
            d = by_pair_both.get(pid, {}).get(j, {})
            ab, ba = d.get("AB"), d.get("BA")
            if ab is None or ba is None:
                continue
            if ab == ba == "A":
                W[idx[pid], jidx[j]] = 1.0
            elif ab == ba == "B":
                W[idx[pid], jidx[j]] = -1.0
            elif ab == ba == "TIE":
                W[idx[pid], jidx[j]] = 0.0
            else:
                W[idx[pid], jidx[j]] = np.nan   # inconsistent -> judge abstains
    def eval_swap_panels(combos, k):
        sub = W[:, combos]
        valid = ~np.isnan(sub).any(axis=2)
        a = (sub == 1.0).sum(axis=2).astype(float)
        b = (sub == -1.0).sum(axis=2).astype(float)
        dec = (a > b) | (b > a)
        dec_a = a > b
        correct = dec_a == (gt_sign[:, None] > 0)
        accT = np.where(valid, np.where(dec, correct, 0.0), np.nan)
        abst = np.where(valid, ~dec, np.nan)
        return {"n_combos": int(combos.shape[0]),
                "acc_tie_wrong": round(float(np.nanmean(accT)), 3),
                "abstain_rate": round(float(np.nanmean(abst)), 3)}
    swapres = {}
    for k in (3, 5):
        combos = np.array(list(itertools.combinations(range(len(open_present)), k)))
        swapres[f"k{k}"] = eval_swap_panels(combos, k)
    panels["order_swap_aggregated"] = swapres

    # single-judge baseline on the SAME gt-pair set, SIGNED pb (fixes Fig.7 inconsistency)
    single_acc_tw = []     # tie=wrong (unified with panel acc_tie_wrong)
    single_acc_dec = []    # decided-only
    single_pb_signed = []
    for j in open_present:
        col = Vmat[:, jidx[j]]
        m = ~np.isnan(col)
        if m.sum() == 0:
            continue
        dec = col[m] != 0
        correct_all = (col[m] == gt_sign[m])          # tie -> False (tie=wrong)
        single_acc_tw.append(float(np.mean(correct_all)))
        single_acc_dec.append(float(np.mean(correct_all[dec])) if dec.any() else np.nan)
        single_pb_signed.append(float(np.mean(col[m][dec] == 1.0)) - 0.5 if dec.any() else np.nan)
    panels["single_judge_same_set"] = {
        "acc_tie_wrong_mean": round(float(np.nanmean(single_acc_tw)), 3),
        "acc_tie_wrong_best": round(float(np.nanmax(single_acc_tw)), 3),
        "acc_decided_mean": round(float(np.nanmean(single_acc_dec)), 3),
        "acc_decided_best": round(float(np.nanmax(single_acc_dec)), 3),
        "pb_signed_mean": round(float(np.nanmean(single_pb_signed)), 3),
        "pb_signed_range": [round(float(np.nanmin(single_pb_signed)), 3),
                            round(float(np.nanmax(single_pb_signed)), 3)],
        "pb_abs_mean": round(float(np.nanmean(np.abs(single_pb_signed))), 3),
    }
    # all-12 reference
    allp = [j for j in judges_present]
    aidx = {j: i for i, j in enumerate(allp)}
    V12 = np.full((len(gt_pairs), len(allp)), np.nan)
    for pid in gt_pairs:
        for j in allp:
            v = by_pair_j.get(pid, {}).get(j)
            if v is not None:
                V12[idx[pid], aidx[j]] = {"A": 1.0, "B": -1.0, "TIE": 0.0}[v]
    Vmat_save, jcols = Vmat, open_present
    Vmat = V12
    res12 = {}
    for k in (3, 5):
        combos = np.array(list(itertools.combinations(range(len(allp)), k)))
        r, _, _ = eval_panels(combos, k)
        res12[f"k{k}"] = {"acc_tie_wrong": r["acc_tie_wrong"]["mean"],
                          "abstain_rate": r["abstain_rate"]}
    panels["all12_reference"] = res12
    Vmat = Vmat_save
    R["S10_panels"] = panels

    # ================================================================ S12 [6]
    print("== S12. GT-clear audit worksheet ==")
    import csv
    audit_path = os.path.join(RESEARCH, "gtclear_audit.csv")
    cols = (["pair_id", "domain", "qid", "gt_better", "gt_v1",
             "gen_a", "label_a_v1", "label_a_v2", "evidence_a",
             "gen_b", "label_b_v1", "label_b_v2", "evidence_b",
             "trunc_a", "trunc_b", "math_final_hidden", "review_flag"]
            + [f"v:{j}" for j in judges_present] + ["majority"])
    nflag = 0
    with open(audit_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for pid in sorted(pairs):
            p = pairs[pid]
            if p.get("gt_better") not in ("A", "B"):
                continue
            q = questions.get(p["qid"], {})
            ra = relabel.get((p["qid"], p["gen_a"]), {})
            rb = relabel.get((p["qid"], p["gen_b"]), {})
            aa = answers.get((p["qid"], p["gen_a"]), {})
            ab_ = answers.get((p["qid"], p["gen_b"]), {})
            ev = lambda r, a: (r.get("final_answer_v2") or "") if r.get("labeler") == "numeric" else str(r.get("evidence", ""))[:90]
            hidden_a = (p["domain"] == "math" and math_final(clip(str(aa.get("content", "")), MAX_ANSWER_CHARS)) is None)
            hidden_b = (p["domain"] == "math" and math_final(clip(str(ab_.get("content", "")), MAX_ANSWER_CHARS)) is None)
            flags = []
            if p["domain"] == "knowledge":
                flags.append("LLM-label")
            if p.get("gt_better_v1") != p.get("gt_better"):
                flags.append("v1v2-flip")
            if p["trunc_a"] or p["trunc_b"]:
                flags.append("truncated")
            if hidden_a or hidden_b:
                flags.append("math-final-hidden")
            if ra.get("labeler") == "numeric" and ra.get("final_answer_v2") is None:
                flags.append("no-extract-a")
            if rb.get("labeler") == "numeric" and rb.get("final_answer_v2") is None:
                flags.append("no-extract-b")
            nflag += int(bool(flags))
            verdicts = [by_pair_j.get(pid, {}).get(j, "") for j in judges_present]
            nonempty = [v for v in verdicts if v in ("A", "B")]
            majority = "TIE" if not nonempty else ("A" if nonempty.count("A") > nonempty.count("B")
                                                   else ("B" if nonempty.count("B") > nonempty.count("A") else "TIE"))
            w.writerow([pid, p["domain"], p["qid"], p["gt_better"], p.get("gt_better_v1"),
                        p["gen_a"], aa.get("correct_v1", ""), bool(ra.get("correct_v2")), ev(ra, aa),
                        p["gen_b"], ab_.get("correct_v1", ""), bool(rb.get("correct_v2")), ev(rb, ab_),
                        p["trunc_a"], p["trunc_b"], hidden_a or hidden_b, "|".join(flags)]
                       + verdicts + [majority])
    R["S12_audit"] = {"path": audit_path, "n_rows": sum(1 for p in pairs.values()
                                                        if p.get("gt_better") in ("A", "B")),
                      "n_flagged": nflag}

    # ================================================================ save
    with open(os.path.join(RESEARCH, "reanalysis_results.json"), "w") as f:
        json.dump(R, f, indent=1, ensure_ascii=False, default=str)
    print("saved reanalysis_results.json")

    # ---------------- markdown report ----------------
    rep = []
    rep.append("# Phase B/D 重分析报告 (reanalysis v2)\n")
    rep.append(f"数据: {R['meta']['n_pairs']} pairs, {R['meta']['n_judgments_parsed']} parsed verdicts, "
               f"{R['meta']['gt_clear_pairs']} GT-clear pairs\n")
    rep.append("\n## S1 截断审计 [item 7]\n")
    rep.append("| domain | n | >900字比例 |")
    rep.append("|---|---|---|")
    for d, v in trunc["by_domain"].items():
        rep.append(f"| {d} | {v['n_answers']} | {v['pct_over_900chars']}% |")
    rep.append(f"\n数学答案最终答案被截掉: {trunc['math_final_hidden']['n_final_hidden']}/{trunc['math_final_hidden']['n_math']} "
               f"({trunc['math_final_hidden']['pct']}%)")
    rep.append(f"生成侧到达 max_tokens 上限: {trunc['generation_side']['n_at_cap']}/{trunc['generation_side']['n_with_usage']} "
               f"({trunc['generation_side']['pct']}%)")
    rep.append(f"知识域参考答案>400字被截: {trunc['ref_trunc']['n_over_400']}/{trunc['ref_trunc']['n_knowledge_refs']}")
    rep.append(f"被截断答案的 pair: {trunc['pairs_any_truncated']}/400; GT-clear 中: {trunc['gtclear_any_truncated']}/66\n")
    rep.append("### 裁判输出触顶(可能截断)\n")
    rep.append("| judge | at_cap | % |")
    rep.append("|---|---|---|")
    for jn, v in trunc["judge_output"].items():
        rep.append(f"| {jn} | {v['at_cap']}/{v['n']} | {v['pct']}% |")

    rep.append("\n## S2 三口径准确率 + GT-clear 平局率 [item 9]\n")
    rep.append("| judge | n | acc(平局错) | acc(剔平局) | 覆盖率 | acc(平局0.5) | GT-clear平局率 |")
    rep.append("|---|---|---|---|---|---|---|")
    for jn in judges_present:
        a = acc3[jn].get("all", {})
        rep.append(f"| {jn} | {a.get('n')} | {a.get('acc_tie_wrong')} | {a.get('acc_tie_excluded')} | "
                   f"{a.get('coverage_decided')} | {a.get('acc_tie_half')} | {a.get('tie_rate_gtclear')} |")
    rep.append("\n### 数学(确定性标签) vs 知识(LLM标签) 拆分 [item 5 标注循环性检查]\n")
    rep.append("| judge | math acc | know acc | math平局 | know平局 |")
    rep.append("|---|---|---|---|---|")
    for jn in judges_present:
        m = acc3[jn].get("math", {}); k = acc3[jn].get("knowledge", {})
        rep.append(f"| {jn} | {m.get('acc_tie_wrong')} (n={m.get('n')}) | {k.get('acc_tie_wrong')} (n={k.get('n')}) | "
                   f"{m.get('tie_rate_gtclear')} | {k.get('tie_rate_gtclear')} |")

    rep.append("\n## S3 双向位置偏差 [item 10]\n")
    rep.append("| judge | bias_bd | CI | n_pairs | p |")
    rep.append("|---|---|---|---|---|")
    for jn in judges_present:
        d = pb2.get(jn, {})
        if d:
            rep.append(f"| {jn} | {d['bias_bd']:+.3f} | {d['ci']} | {d['n_pairs']} | "
                       f"{d['p']:.4f} |")

    rep.append("\n## S4 翻转分解 [item 11]\n")
    rep.append("| judge | hard(A<->B) | soft(win<->TIE) | total |")
    rep.append("|---|---|---|---|")
    for jn in judges_present:
        d = flipd.get(jn, {})
        if d:
            rep.append(f"| {jn} | {d['hard_flip']} | {d['soft_flip']} | {d['total_flip']} |")

    if "error" not in reg:
        rep.append("\n## S5 聚类稳健回归 [item 12]\n")
        main_spec = reg["main_seen_len"]["specs"]
        rep.append("主设定: y=A胜 ~ len_diff(裁判所见长度,千字符) + kin_side + quality_diff + 裁判FE\n")
        rep.append("| covariate | coef | naive p | cluster(qid) SE | p | two-way(qid×judge) SE | p |")
        rep.append("|---|---|---|---|---|---|---|")
        for c in ["len_diff_seen", "kin_side", "quality_diff"]:
            nv = main_spec["naive"][c]
            cl = main_spec["cluster_qid"][c]
            tw = main_spec["two_way_qid_judge"][c]
            rep.append(f"| {c} | {cl['coef']:.3f} | {nv['p']:.2e} | {cl['se']:.3f} | {cl['p']:.2e} | "
                       f"{tw['se']:.3f} | {tw['p']:.2e} |")
        fl = reg["main_full_len"]["specs"]["cluster_qid"]["len_diff_full"]
        rep.append(f"\n对照: 全长度 len_diff_full 系数 {fl['coef']:.3f} (cluster p={fl['p']:.2e}) — "
                   f"所见长度 {main_spec['cluster_qid']['len_diff_seen']['coef']:.3f}")
        rep.append("\n### 按裁判家族的亲缘系数(回归调整) [item 25]\n")
        kf = reg["kin_by_family_cluster_qid"]
        rep.append("| family | kin coef | SE | p | 95% CI |")
        rep.append("|---|---|---|---|---|")
        for fam in ["Qwen", "DeepSeek", "GLM", "Kimi"]:
            d = kf[f"kin_{fam}"]
            rep.append(f"| {fam} | {d['coef']:.3f} | {d['se']:.3f} | {d['p']:.3g} | [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}] |")

    rep.append("\n## S6 共同覆盖重分析 [item 13]\n")
    rep.append(f"共同覆盖: AB 全到齐 {cc['n_common_AB']} 对; AB+BA 全到齐 {cc['n_common_ABBA']} 对\n")
    rep.append("| judge | acc full | acc common | tie full | tie common | PB_bd full | PB_bd common | LB full | LB common |")
    rep.append("|---|---|---|---|---|---|---|---|---|")
    for jn, d in cc["per_judge"].items():
        fu, co = d["full"], d["common"]
        rep.append(f"| {jn} | {fu['acc']} | {co['acc']} | {fu['tie']} | {co['tie']} | "
                   f"{fu['pb_bd']} | {co['pb_bd']} | {fu['lb']} | {co['lb']} |")
    rep.append(f"\n共同覆盖集 kappa 均值: {cc['mean_kappa_common']}")
    rep.append(f"域构成 common: {dict(dom_common)} vs full: {dict(dom_full)}")

    rep.append("\n## S7 一致性稳健指标 [item 14]\n")
    rep.append("| judge | kappa | Gwet AC1 | kappa(非平局子集) |")
    rep.append("|---|---|---|---|")
    for jn in judges_present:
        d = agr["mean_per_judge"][jn]
        rep.append(f"| {jn} | {d['kappa']} | {d['ac1']} | {d['kappa_nontie']} |")
    rep.append(f"\n整体: kappa {agr['overall']['kappa_range']}, AC1 {agr['overall']['ac1_range']}, "
               f"相关 {agr['overall']['corr_kappa_ac1']}")

    rep.append("\n## S8 多重比较校正 [item 15]\n")
    rep.append("| judge | metric | est | raw p | Holm p | BH q | raw显著 | Holm显著 | BH显著 |")
    rep.append("|---|---|---|---|---|---|---|---|---|")
    for t in tests:
        rep.append(f"| {t['judge']} | {t['metric']} | {t['est']} | {t['p']:.4f} | {t['p_holm']:.4f} | "
                   f"{t['p_bh']:.4f} | {t['sig_raw']} | {t['sig_holm']} | {t['sig_bh']} |")

    rep.append("\n## S9 亲缘分析重做 [item 25]\n")
    rep.append("| family | kin-wrong率 [Wilson] | 非亲缘基线错误率 | 差异检验 | kin-right | kin-win率 [Wilson] |")
    rep.append("|---|---|---|---|---|---|")
    for fam in ["Qwen", "DeepSeek", "GLM", "Kimi"]:
        d = kin["by_family"][fam]
        kw = d["kin_wrong"]; nk = d["nonkin_baseline_error"]; tst = d["kin_wrong_vs_baseline"]
        rep.append(f"| {fam} | {kw['rate']} [{kw['wilson']}] (n={kw['n']}) | "
                   f"{nk['rate']} [{nk['wilson']}] (n={nk['n']}) | "
                   f"z={tst['z']}, p={tst['p_fisher'] if tst['p_fisher'] is not None else tst['p_z']:.4f} | "
                   f"{d['kin_right']['rate']} (n={d['kin_right']['n']}) | "
                   f"{d['kin_win']['rate']} [{d['kin_win']['wilson']}] |")
    rep.append("\n### 域拆分(标签循环性检查 [item 5])\n")
    rep.append("| family | kin-win math-only | kin-win knowledge-only |")
    rep.append("|---|---|---|")
    for fam in ["Qwen", "DeepSeek", "GLM", "Kimi"]:
        d = kin["by_family"][fam]
        m = d["kin_win_math_only"]; k = d["kin_win_knowledge_only"]
        rep.append(f"| {fam} | {m['rate']} (n={m['n']}) | {k['rate']} (n={k['n']}) |")

    rep.append("\n## S10 面板分析重做 [item 26]\n")
    e = panels["enumerated_open"]
    s = panels["single_judge_same_set"]
    rep.append(f"单裁判同集基线: acc(平局错)均值 {s['acc_tie_wrong_mean']}, 最佳 {s['acc_tie_wrong_best']}; "
               f"acc(剔平局)均值 {s['acc_decided_mean']}, 最佳 {s['acc_decided_best']}; "
               f"有符号PB均值 {s['pb_signed_mean']} (范围 {s['pb_signed_range']}), |PB|均值 {s['pb_abs_mean']}\n")
    for k in ("k3", "k5"):
        d = e[k]
        rep.append(f"- **{k} 精确枚举 ({d['n_combos']}组合)**: acc(平局错)={d['acc_tie_wrong']['mean']:.3f} "
                   f"[boot CI {d['acc_tie_wrong']['boot_ci']}] (min {d['acc_tie_wrong']['min']:.3f} / max {d['acc_tie_wrong']['max']:.3f}), "
                   f"弃权率={d['abstain_rate']:.3f}, 决策集准确率={d['acc_decided']:.3f}, 有符号PB={d['pb_signed']:.3f}")
    rep.append(f"\n家族构成对比(k=3): 同家族 {panels['k3_same1fam']['acc_mean']} (n={panels['k3_same1fam']['n_panels']}) | "
               f"两家族 {panels['k3_two2fam']['acc_mean']} (n={panels['k3_two2fam']['n_panels']}) | "
               f"三家族 {panels['k3_three3fam']['acc_mean']} (n={panels['k3_three3fam']['n_panels']})")
    rep.append(f"\n换序聚合面板: k3 {panels['order_swap_aggregated']['k3']} | k5 {panels['order_swap_aggregated']['k5']}")
    rep.append(f"\n全12裁判参照: {panels['all12_reference']}")

    rep.append(f"\n## S12 GT-clear 审核包 [item 6]\n")
    rep.append(f"已生成 {R['S12_audit']['path']} ({R['S12_audit']['n_rows']} 对, 其中 {R['S12_audit']['n_flagged']} 对带复核标记)")

    with open(os.path.join(RESEARCH, "reanalysis_report.md"), "w") as f:
        f.write("\n".join(rep))
    print("saved reanalysis_report.md")


if __name__ == "__main__":
    main()
