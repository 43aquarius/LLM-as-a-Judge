#!/usr/bin/env python3
"""Stage C orchestrator (review plan items 16-24, scheme Yi / near-zero cost).

Design (converged 2026-10-17 against live console quotas; see --plan):
  E0  snapshot check   : 20 pairs x {27B, Kimi} x AB/BA              80 calls
  E1  instruction abl. : 60 pairs x {3.7F, Max, V4F, V4.1F} x P0/P2/P3 (AB)
                         P1 arm reuses v1 AB rows (requires E0 pass)   720 calls
  E2  tie ablation     : 45 pairs x {3.7F, 27B, V4F, V4-Pro} x T1/T2 (AB)
                         P1 arm reuses v1 AB rows                      360 calls
  E3  GT-clear expand  : 100 tasks x 4 generators + ~115 pairs x
                         {3.7F, Max, V4-Pro, Kimi} judges (AB)        ~880 calls
  E5  no-reference     : 60 E3 pairs x {3.7F, V4.1F} (AB)             120 calls
  E8  retest expand    : 60 pairs x {27B, V4F, V4.1F, 3.7F} (AB)      240 calls
  E7  commercial anchor: DEFERRED - needs account recharge (~CNY 30-50);
                         run via e7.py separately once funded.
All records carry prompt_version + platform snapshot id + ts. Resumable.
"""
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llmclient
from judge import parse_verdict, clip, MAX_ANSWER_CHARS, MAX_QUESTION_CHARS, \
    JUDGE_FAMILY, GEN_FAMILY

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SEED = 20261017

# ---- model ids ----
M = {
    "37f": "qwen3.7-flash-2026-07-15", "38f": "qwen3.8-flash",
    "27b": "qwen3.8-27b", "max": "qwen3.8-max-0902",
    "v4f": "deepseek-v4-flash-0731", "v41f": "deepseek-v4.1-flash",
    "v4p": "deepseek-v4-pro-0813", "kimi": "kimi-k3",
    "37fa": "qwen3.7-flash",  # alias model-code, own free grant (batch-2 Qwen generator)
}
NAME = {  # display names matching v1 records
    "37f": "Qwen3.7-Flash", "38f": "Qwen3.8-Flash", "27b": "Qwen3.8-27B",
    "max": "Qwen3.8-Max", "v4f": "DeepSeek-V4-Flash", "v41f": "DeepSeek-V4.1-Flash",
    "v4p": "DeepSeek-V4-Pro", "kimi": "Kimi-K3", "37fa": "Qwen3.7-Flash",
}
GF = dict(GEN_FAMILY)  # generator families incl. new members
GF["Qwen3.7-Flash"] = "Qwen"

# ---- prompt variants (frozen; P1 = v1 SYSTEM) ----
P1 = ("You are an impartial evaluator. You will be shown a question and two answers "
      "from different AI assistants. Evaluate which answer is better according to "
      "correctness, completeness, clarity, and how well it follows the question. "
      "Do not reward an answer merely because it is longer or more elaborate. "
      "Judge only on quality. Reply with a brief justification of one or two "
      "sentences, and end your reply with a final verdict line containing exactly "
      "one of: [[A]] if Assistant A is better, [[B]] if Assistant B is better, "
      "or [[C]] if the two answers are similarly good.")
P0 = ("You are an impartial evaluator. You will be shown a question and two answers "
      "from different AI assistants. Evaluate which answer is better according to "
      "correctness, completeness, clarity, and how well it follows the question. "
      "Judge only on quality. Reply with a brief justification of one or two "
      "sentences, and end your reply with a final verdict line containing exactly "
      "one of: [[A]] if Assistant A is better, [[B]] if Assistant B is better, "
      "or [[C]] if the two answers are similarly good.")
P2 = ("You are an impartial evaluator. You will be shown a question and two answers "
      "from different AI assistants. Evaluate which answer is better according to "
      "correctness, completeness, clarity, and how well it follows the question. "
      "Answer length alone does not determine quality: a longer answer is not better "
      "merely because it is longer, and a concise answer is not worse merely because "
      "it is concise. Judge only on quality. Reply with a brief justification of one "
      "or two sentences, and end your reply with a final verdict line containing "
      "exactly one of: [[A]] if Assistant A is better, [[B]] if Assistant B is better, "
      "or [[C]] if the two answers are similarly good.")
P3 = ("You are an impartial evaluator. You will be shown a question and two answers "
      "from different AI assistants. Evaluate which answer is better according to "
      "correctness, completeness, clarity, and how well it follows the question. "
      "When one answer is much longer than the other, evaluate whether the extra "
      "length adds correctness or substance; do not reward length itself, and do "
      "not penalize brevity when the shorter answer fully answers the question. "
      "Judge only on quality. Reply with a brief justification of one or two "
      "sentences, and end your reply with a final verdict line containing exactly "
      "one of: [[A]] if Assistant A is better, [[B]] if Assistant B is better, "
      "or [[C]] if the two answers are similarly good.")
T1 = ("You are an impartial evaluator. You will be shown a question and two answers "
      "from different AI assistants. Evaluate which answer is better according to "
      "correctness, completeness, clarity, and how well it follows the question. "
      "Do not reward an answer merely because it is longer or more elaborate. "
      "Judge only on quality. You must choose exactly one of the two answers; "
      "a tie is not an available verdict. Reply with a brief justification of one "
      "or two sentences, and end your reply with a final verdict line containing "
      "exactly one of: [[A]] if Assistant A is better, or [[B]] if Assistant B is better.")
T2 = ("You are an impartial evaluator. You will be shown a question and two answers "
      "from different AI assistants. Evaluate which answer is better according to "
      "correctness, completeness, clarity, and how well it follows the question. "
      "Do not reward an answer merely because it is longer or more elaborate. "
      "Judge only on quality. Reply with a brief justification of one or two "
      "sentences, and end your reply with a final verdict line containing exactly "
      "one of: [[A]] if Assistant A is better, [[B]] if Assistant B is better, "
      "or [[C]] only if the two answers are substantively equivalent in quality — "
      "minor differences should lead you to choose the better answer instead.")
PROMPTS = {"P0": P0, "P1": P1, "P2": P2, "P3": P3, "T1": T1, "T2": T2}

E1_JUDGES = ["37f", "max", "v4f", "v41f"]
E2_JUDGES = ["37f", "27b", "v4f", "v4p"]
E3_GENS_B1 = ["38f", "max", "v4p", "kimi"]           # batch-1 roster (v1 generator set)
E3_GENS_B2 = ["37fa", "max", "v4p", "kimi"]          # batch-2: 3.8-Flash quota-low,
                                                       # qwen3.7-flash alias model-code
                                                       # (own free grant) takes the Qwen slot
E3_JUDGES = ["37f", "max", "v4p", "kimi"]
E5_JUDGES = ["37f", "max"]  # must be subset of E3_JUDGES for paired with/without-ref
E8_JUDGES = ["27b", "v4f", "v41f", "37f"]


def jl(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def load_v1():
    pairs = {p["pair_id"]: p for p in jl(f"{R}/pairs.jsonl")}
    qs = {q["qid"]: q for q in jl(f"{R}/questions.jsonl")}
    return pairs, qs


def v1_verdicts():
    """(pair_id, judge_name, order) -> canonical verdict, from v1 main eval."""
    v = {}
    for r in jl(f"{R}/judgments.jsonl"):
        v[(r["pair_id"], r["judge"], r["order"])] = r.get("verdict")
    return v


def user_msg(question, a, b, ref, include_ref=True):
    q = clip(question, MAX_QUESTION_CHARS)
    ref_block = ""
    if include_ref and ref:
        ref_block = ("\n### Reference answer (ground truth; use it to check "
                     "correctness only):\n" + clip(str(ref), 400))
    return ("### Question:\n" + q + ref_block
            + "\n\n### Assistant A's answer:\n" + clip(a, MAX_ANSWER_CHARS)
            + "\n\n### Assistant B's answer:\n" + clip(b, MAX_ANSWER_CHARS)
            + "\n\n### Your evaluation (brief justification, then final verdict line):")


def judge_variant(jkey, system, umsg, task_id, mt=400):
    r = llmclient.chat(M[jkey], [{"role": "system", "content": system},
                                 {"role": "user", "content": umsg}],
                       max_tokens=mt, temperature=0.0, task_id=task_id)
    v = parse_verdict(r["content"])
    ok = v is not None
    if not ok:
        r2 = llmclient.chat(M[jkey], [{"role": "system", "content": system},
                                      {"role": "user", "content": umsg},
                                      {"role": "assistant", "content": clip(r["content"], 2000)},
                                      {"role": "user", "content":
                                       "Your reply did not contain a valid verdict line. Reply again and end with exactly [[A]], [[B]], or [[C]]."}],
                            max_tokens=mt, temperature=0.0, task_id=task_id + ":retry")
        v = parse_verdict(r2["content"])
        ok = v is not None
        if ok:
            r = r2
    return v, ok, r


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def done_keys(path):
    keys = set()
    for r in jl(path):
        keys.add((r.get("pair_id") or r.get("tid"), r.get("judge"), r.get("order"),
                  r.get("prompt_version"), r.get("rep", 0), r.get("generator")))
    return keys


# ---------------- E0: snapshot stability ----------------
def cmd_e0():
    pairs, qs = load_v1()
    rng = random.Random(SEED)
    by_dom = {}
    for p in pairs.values():
        by_dom.setdefault(p["domain"], []).append(p)
    subset = []
    for dom in sorted(by_dom):
        lst = sorted(by_dom[dom], key=lambda x: x["pair_id"])
        subset.extend(rng.sample(lst, 5))
    out = f"{R}/judgments_e0.jsonl"
    dk = done_keys(out)
    tasks = []
    for p in subset:
        q = qs[p["qid"]]
        for order in ("AB", "BA"):
            for jk in ("27b", "kimi"):
                if (p["pair_id"], NAME[jk], order, "P1", 0, None) in dk:
                    continue
                ta, tb = (p["text_a"], p["text_b"]) if order == "AB" else (p["text_b"], p["text_a"])
                um = user_msg(q["question"], ta, tb, p.get("ref"))
                tasks.append({"p": p, "q": q, "jk": jk, "order": order, "um": um})
    random.Random(SEED + 1).shuffle(tasks)
    print(f"E0 tasks: {len(tasks)} (skip done {len(dk)})")

    def worker(t):
        task_id = f"e0:{t['p']['pair_id']}:{NAME[t['jk']]}:{t['order']}"
        try:
            v, ok, r = judge_variant(t["jk"], P1, t["um"], task_id)
        except llmclient.QuotaExceeded:
            return None
        canon = {"A": "A" if t["order"] == "AB" else "B",
                 "B": "B" if t["order"] == "AB" else "A"}.get(v, "TIE" if v == "C" else None)
        append(out, {
            "exp": "E0", "pair_id": t["p"]["pair_id"], "qid": t["p"]["qid"],
            "domain": t["p"]["domain"], "judge": NAME[t["jk"]],
            "judge_model": M[t["jk"]], "judge_family": JUDGE_FAMILY[NAME[t["jk"]]],
            "order": t["order"], "prompt_version": "P1", "rep": 0,
            "verdict_raw": v, "verdict": canon, "parsed_ok": ok,
            "snapshot": r.get("snapshot"), "usage": r["usage"], "latency": r["latency"],
        })
        return 1

    res, err = llmclient.run_parallel(tasks, worker, n_workers=6, desc="E0")
    # comparison vs v1
    v1 = v1_verdicts()
    rows = jl(out)
    stat = {}
    lat = []
    for r in rows:
        if not r.get("parsed_ok"):
            continue
        old = v1.get((r["pair_id"], r["judge"], r["order"]))
        if old is None:
            continue
        s = stat.setdefault(r["judge"], {"n": 0, "same": 0})
        s["n"] += 1
        if old == r["verdict"]:
            s["same"] += 1
        lat.append(r["latency"])
    report = {"per_judge": {}, "decision": ""}
    ok_all = True
    for j, s in stat.items():
        agree = s["same"] / s["n"] if s["n"] else 0
        report["per_judge"][j] = {"n": s["n"], "agreement": round(agree, 3)}
        print(f"  {j}: agreement {agree:.3f} ({s['same']}/{s['n']})")
        if agree < 0.85:
            ok_all = False
    if lat:
        report["median_latency_s"] = sorted(lat)[len(lat) // 2]
    report["decision"] = "PASS" if ok_all else "DRIFT"
    print(f"E0 decision: {report['decision']} (threshold 0.85 per judge)")
    json.dump(report, open(f"{R}/e0_report.json", "w"), indent=1)


# ---------------- subset selectors ----------------
def cmd_subsets():
    pairs, qs = load_v1()
    rng = random.Random(SEED)
    lst = sorted(pairs.values(), key=lambda x: x["pair_id"])
    # E1: 60 pairs, within-domain len-diff quartiles
    e1 = []
    for dom in ("math", "knowledge", "creative", "summ"):
        dp = [p for p in lst if p["domain"] == dom]
        dp.sort(key=lambda p: abs(p["len_a"] - p["len_b"]))
        n = 15 if dom in ("math", "knowledge") else 15  # 15*4=60
        per_q = n // 4 + 1
        for qi in range(4):
            seg = dp[qi * len(dp) // 4: (qi + 1) * len(dp) // 4] or dp
            e1.extend(rng.sample(seg, min(per_q, len(seg))))
    e1 = e1[:60]
    json.dump([p["pair_id"] for p in e1], open(f"{R}/e1_pairs.json", "w"), indent=1)
    # E2: 45 pairs, tie-prone (v1 AB-order tie counts), domain-stratified
    tiecount = {}
    for r in jl(f"{R}/judgments.jsonl"):
        if r.get("order") == "AB" and r.get("verdict") == "TIE":
            tiecount[r["pair_id"]] = tiecount.get(r["pair_id"], 0) + 1
    e2 = []
    for dom in ("math", "knowledge", "creative", "summ"):
        dp = [p for p in lst if p["domain"] == dom]
        dp.sort(key=lambda p: (-tiecount.get(p["pair_id"], 0), p["pair_id"]))
        e2.extend(dp[:11])
    e2 = e2[:45]
    json.dump([p["pair_id"] for p in e2], open(f"{R}/e2_pairs.json", "w"), indent=1)
    # E8: 60 pairs, prefer overlap with E1/E2, domain-stratified
    prefer = {p["pair_id"] for p in e1} | {p["pair_id"] for p in e2}
    e8 = []
    for dom in ("math", "knowledge", "creative", "summ"):
        dp = [p for p in lst if p["domain"] == dom]
        in_s = [p for p in dp if p["pair_id"] in prefer]
        out_s = [p for p in dp if p["pair_id"] not in prefer]
        take_in = min(len(in_s), 10)
        e8.extend(in_s[:take_in])
        e8.extend(rng.sample(out_s, min(len(out_s), 15 - take_in)))
    e8 = e8[:60]
    json.dump([p["pair_id"] for p in e8], open(f"{R}/e8_pairs.json", "w"), indent=1)
    from collections import Counter
    print(f"E1 subset {len(e1)}: {Counter(p['domain'] for p in e1)}")
    print(f"E2 subset {len(e2)}: {Counter(p['domain'] for p in e2)} "
          f"(tie-prone: {sum(tiecount.get(p['pair_id'],0) for p in e2)} ties)")
    print(f"E8 subset {len(e8)}: {Counter(p['domain'] for p in e8)} "
          f"overlap with E1/E2: {sum(1 for p in e8 if p['pair_id'] in prefer)}/{len(e8)}")


# ---------------- E1 / E2 runners ----------------
def _run_prompt_ablation(exp, pairfile, judges, versions, outfile):
    pairs, qs = load_v1()
    ids = json.load(open(pairfile))
    dk = done_keys(outfile)
    tasks = []
    for pid in ids:
        p = pairs[pid]
        q = qs[p["qid"]]
        for jk in judges:
            for pv in versions:
                if (pid, NAME[jk], "AB", pv, 0, None) in dk:
                    continue
                um = user_msg(q["question"], p["text_a"], p["text_b"], p.get("ref"))
                tasks.append({"p": p, "q": q, "jk": jk, "pv": pv, "um": um})
    random.Random(SEED + 2).shuffle(tasks)
    print(f"{exp} tasks: {len(tasks)} (skip done {len(dk)})")

    def worker(t):
        task_id = f"{exp.lower()}:{t['p']['pair_id']}:{NAME[t['jk']]}:{t['pv']}"
        try:
            v, ok, r = judge_variant(t["jk"], PROMPTS[t["pv"]], t["um"], task_id)
        except llmclient.QuotaExceeded:
            return None
        p = t["p"]
        append(outfile, {
            "exp": exp, "pair_id": p["pair_id"], "qid": p["qid"], "domain": p["domain"],
            "judge": NAME[t["jk"]], "judge_model": M[t["jk"]],
            "judge_family": JUDGE_FAMILY[NAME[t["jk"]]],
            "order": "AB", "prompt_version": t["pv"], "rep": 0,
            "gen_a": p["gen_a"], "gen_b": p["gen_b"],
            "gen_a_family": GF[p["gen_a"]], "gen_b_family": GF[p["gen_b"]],
            "verdict_raw": v, "verdict": v, "parsed_ok": ok,
            "len_a": p["len_a"], "len_b": p["len_b"], "gt_better": p.get("gt_better"),
            "snapshot": r.get("snapshot"), "usage": r["usage"], "latency": r["latency"],
        })
        return 1

    res, err = llmclient.run_parallel(tasks, worker, n_workers=6, desc=exp)
    print(f"{exp}: {len(res)} records, {len(err)} errors")


def cmd_e1():
    _run_prompt_ablation("E1", f"{R}/e1_pairs.json", E1_JUDGES,
                          ["P0", "P2", "P3"], f"{R}/judgments_e1.jsonl")


def cmd_e2():
    _run_prompt_ablation("E2", f"{R}/e2_pairs.json", E2_JUDGES,
                          ["T1", "T2"], f"{R}/judgments_e2.jsonl")


# ---------------- E3: generation, pairing, judging ----------------
def cmd_e3gen():
    from e3_build_sources import parse_math_out, parse_mcq_out, parse_code_out, \
        check_code, ans_eq
    tasks = jl(f"{R}/e3_tasks.jsonl")
    out = f"{R}/answers_e3.jsonl"
    dk = done_keys(out)
    gens = []
    for t in tasks:
        roster = E3_GENS_B2 if t.get("meta", {}).get("batch") == 2 else E3_GENS_B1
        for gk in roster:
            if (t["tid"], None, None, None, 0, NAME[gk]) in dk:
                continue
            gens.append({"t": t, "gk": gk})
    random.Random(SEED + 3).shuffle(gens)
    print(f"E3 gen tasks: {len(gens)} (skip done {len(dk)})")

    def worker(w):
        t, gk = w["t"], w["gk"]
        task_id = f"e3gen:{t['tid']}:{NAME[gk]}"
        temp = 1.0 if M[gk] == "kimi-k3" else 0.7
        try:
            r = llmclient.chat(M[gk], [{"role": "system", "content":
                                        "You are a helpful assistant. Answer the user's question or follow the instruction."},
                                       {"role": "user", "content": t["gen_prompt"]}],
                               max_tokens=900, temperature=temp, task_id=task_id)
        except llmclient.QuotaExceeded:
            return None
        content = r["content"] or "[no direct answer]"
        correct = None
        kind = t["payload"]["kind"]
        if kind == "math":
            correct = ans_eq(parse_math_out(content), t["payload"]["ref_answer"])
        elif kind == "mcq":
            want = "ABCD"[t["payload"]["answer_idx"]]
            correct = parse_mcq_out(content) == want
        elif kind == "code":
            code = parse_code_out(content, t["payload"]["entry_point"])
            correct = check_code(code, t["payload"]["entry_point"], t["payload"]["test"])
        append(out, {
            "exp": "E3gen", "tid": t["tid"], "source": t["source"], "domain": t["domain"],
            "batch": t.get("meta", {}).get("batch", 1),
            "generator": NAME[gk], "generator_model": M[gk],
            "gen_family": GF[NAME[gk]],
            "content": content, "reference": t["reference"],
            "correct": correct, "snapshot": r.get("snapshot"),
            "usage": r["usage"], "latency": r["latency"],
        })
        return 1

    res, err = llmclient.run_parallel(gens, worker, n_workers=5, desc="E3gen")
    # inventory
    rows = jl(out)
    from collections import Counter, defaultdict
    by_t = defaultdict(dict)
    for r in rows:
        by_t[r["tid"]][r["generator"]] = r
    corr = Counter()
    for tid, d in by_t.items():
        for g, r in d.items():
            corr[(r["source"], r["generator"], r["correct"])] += 1
    print("correctness inventory (source, generator, correct):")
    for k in sorted(corr, key=str):
        print("  ", k, corr[k])
    json.dump({str(k): v for k, v in corr.items()},
              open(f"{R}/e3_gen_inventory.json", "w"), indent=1)


def cmd_e3pair():
    rng = random.Random(SEED + 4)
    tasks = {t["tid"]: t for t in jl(f"{R}/e3_tasks.jsonl")}
    rows = jl(f"{R}/answers_e3.jsonl")
    by_t = {}
    for r in rows:
        by_t.setdefault(r["tid"], {})[r["generator"]] = r
    # candidates: (task, wrongGen, correctGen)
    cands = []
    for tid, d in by_t.items():
        gens = list(d.values())
        for i in range(len(gens)):
            for j in range(len(gens)):
                if i == j:
                    continue
                a, b = gens[i], gens[j]
                if a.get("correct") is True and b.get("correct") is False:
                    cands.append((tasks[tid], b, a))  # (task, wrong, correct)
    print(f"GT-clear candidates: {len(cands)}")
    # stratified quota: wrong-family targets, spread across tasks and domains
    targets = {"Qwen": 50, "DeepSeek": 40, "Kimi": 40}
    caps = {"Qwen": 2, "DeepSeek": 3, "Kimi": 3}  # per-task cap: looser for thin pools
    chosen = []
    used_per_task = {}
    for fam, quota in targets.items():
        pool = [c for c in cands if GF[c[1]["generator"]] == fam]
        # round-robin across domains so no domain is starved
        by_dom = {}
        for c in sorted(pool, key=lambda c: (c[0]["tid"], c[1]["generator"])):
            by_dom.setdefault(c[0]["domain"], []).append(c)
        for lst in by_dom.values():
            rng.shuffle(lst)
        doms = [d for d in by_dom if by_dom[d]]
        queue = []
        while any(by_dom[d] for d in doms):
            for d in doms:
                if by_dom[d]:
                    queue.append(by_dom[d].pop(0))
        got = 0
        for c in queue:
            if got >= quota:
                break
            if used_per_task.get(c[0]["tid"], 0) >= caps[fam]:
                continue
            chosen.append(c)
            used_per_task[c[0]["tid"]] = used_per_task.get(c[0]["tid"], 0) + 1
            got += 1
        print(f"  {fam}-wrong pairs: {got}/{quota} (pool {len(pool)})")
    out = f"{R}/pairs_e3.jsonl"
    with open(out, "w") as f:
        for k, (t, wrong, right) in enumerate(chosen):
            ga, gb = wrong["generator"], right["generator"]
            ta, tb = wrong["content"], right["content"]
            gt = "A"
            if rng.random() < 0.5:  # seeded orientation randomization
                ga, gb = gb, ga
                ta, tb = tb, ta
                gt = "B"
            f.write(json.dumps({
                "pair_id": f"e3-{k:03d}", "tid": t["tid"], "qid": t["tid"],
                "domain": t["domain"], "source": t["source"],
                "gen_a": ga, "gen_b": gb,
                "text_a": ta, "text_b": tb,
                "len_a": len(ta), "len_b": len(tb),
                "ref": t["reference"], "gt_better": gt,
                "wrong_gen": wrong["generator"], "right_gen": right["generator"],
            }, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"E3 pairs: {len(chosen)} -> {out}")
    print("  by domain:", Counter(c[0]["domain"] for c in chosen))
    print("  by wrong-family:", Counter(GF[c[1]["generator"]] for c in chosen))
    print("  by source:", Counter(c[0]["source"] for c in chosen))


def cmd_e3judge():
    pairs = jl(f"{R}/pairs_e3.jsonl")
    out = f"{R}/judgments_e3.jsonl"
    dk = done_keys(out)
    tmap = {t["tid"]: t for t in jl(f"{R}/e3_tasks.jsonl")}
    tasks = []
    for p in pairs:
        t = tmap[p["tid"]]
        for jk in E3_JUDGES:
            if (p["pair_id"], NAME[jk], "AB", "P1", 0, None) in dk:
                continue
            um = user_msg(_e3_display_question(t), p["text_a"], p["text_b"], p["ref"])
            tasks.append({"p": p, "jk": jk, "um": um})
    random.Random(SEED + 5).shuffle(tasks)
    print(f"E3 judge tasks: {len(tasks)} (skip done {len(dk)})")

    def worker(w):
        p, jk, um = w["p"], w["jk"], w["um"]
        task_id = f"e3j:{p['pair_id']}:{NAME[jk]}"
        try:
            v, ok, r = judge_variant(jk, P1, um, task_id)
        except llmclient.QuotaExceeded:
            return None
        except Exception as e:
            with open(out + ".errors", "a") as f:
                f.write(json.dumps({"task": task_id, "error": str(e)[:300]}) + "\n")
            return None
        append(out, {
            "exp": "E3", "pair_id": p["pair_id"], "qid": p["tid"], "domain": p["domain"],
            "source": p["source"], "judge": NAME[jk], "judge_model": M[jk],
            "judge_family": JUDGE_FAMILY[NAME[jk]],
            "order": "AB", "prompt_version": "P1", "rep": 0,
            "gen_a": p["gen_a"], "gen_b": p["gen_b"],
            "gen_a_family": GF[p["gen_a"]], "gen_b_family": GF[p["gen_b"]],
            "wrong_gen": p["wrong_gen"], "right_gen": p["right_gen"],
            "gt_better": p["gt_better"],
            "verdict_raw": v, "verdict": v, "parsed_ok": ok,
            "len_a": p["len_a"], "len_b": p["len_b"],
            "snapshot": r.get("snapshot"), "usage": r["usage"], "latency": r["latency"],
        })
        return 1

    res, err = llmclient.run_parallel(tasks, worker, n_workers=6, desc="E3j")
    print(f"E3 judge: {len(res)} records, {len(err)} errors")
    for e in err[:8]:
        print("  ERR", e.get("task", ""), str(e.get("error"))[:220])


def _e3_display_question(t):
    return t["question"]  # MATH problem / HumanEval prompt / MCQ with options


def cmd_e5():
    pairs = jl(f"{R}/pairs_e3.jsonl")
    rng = random.Random(SEED + 6)
    pairs = rng.sample(pairs, min(60, len(pairs)))
    out = f"{R}/judgments_e5.jsonl"
    dk = done_keys(out)
    tmap = {t["tid"]: t for t in jl(f"{R}/e3_tasks.jsonl")}
    tasks = []
    for p in pairs:
        t = tmap[p["tid"]]
        for jk in E5_JUDGES:
            if (p["pair_id"], NAME[jk], "AB", "NOREF", 0, None) in dk:
                continue
            um = user_msg(_e3_display_question(t), p["text_a"], p["text_b"], None,
                          include_ref=False)
            tasks.append({"p": p, "jk": jk, "um": um})
    random.Random(SEED + 7).shuffle(tasks)
    print(f"E5 tasks: {len(tasks)} (skip done {len(dk)})")

    def worker(w):
        p, jk, um = w["p"], w["jk"], w["um"]
        task_id = f"e5:{p['pair_id']}:{NAME[jk]}"
        try:
            v, ok, r = judge_variant(jk, P1, um, task_id)
        except llmclient.QuotaExceeded:
            return None
        append(out, {
            "exp": "E5", "pair_id": p["pair_id"], "qid": p["tid"], "domain": p["domain"],
            "source": p["source"], "judge": NAME[jk], "judge_model": M[jk],
            "judge_family": JUDGE_FAMILY[NAME[jk]],
            "order": "AB", "prompt_version": "NOREF", "rep": 0,
            "gen_a": p["gen_a"], "gen_b": p["gen_b"],
            "gen_a_family": GF[p["gen_a"]], "gen_b_family": GF[p["gen_b"]],
            "wrong_gen": p["wrong_gen"], "right_gen": p["right_gen"],
            "gt_better": p["gt_better"], "verdict_raw": v, "verdict": v, "parsed_ok": ok,
            "len_a": p["len_a"], "len_b": p["len_b"],
            "snapshot": r.get("snapshot"), "usage": r["usage"], "latency": r["latency"],
        })
        return 1

    res, err = llmclient.run_parallel(tasks, worker, n_workers=6, desc="E5")
    print(f"E5: {len(res)} records, {len(err)} errors")


def cmd_e8():
    pairs, qs = load_v1()
    ids = json.load(open(f"{R}/e8_pairs.json"))
    out = f"{R}/judgments_e8.jsonl"
    dk = done_keys(out)
    tasks = []
    for pid in ids:
        p = pairs[pid]
        q = qs[p["qid"]]
        for jk in E8_JUDGES:
            if (pid, NAME[jk], "AB", "P1", 1, None) in dk:
                continue
            um = user_msg(q["question"], p["text_a"], p["text_b"], p.get("ref"))
            tasks.append({"p": p, "q": q, "jk": jk, "um": um})
    random.Random(SEED + 8).shuffle(tasks)
    print(f"E8 tasks: {len(tasks)} (skip done {len(dk)})")

    def worker(t):
        p, jk, um = t["p"], t["jk"], t["um"]
        task_id = f"e8:{p['pair_id']}:{NAME[jk]}"
        try:
            v, ok, r = judge_variant(jk, P1, um, task_id)
        except llmclient.QuotaExceeded:
            return None
        append(out, {
            "exp": "E8", "pair_id": p["pair_id"], "qid": p["qid"], "domain": p["domain"],
            "judge": NAME[jk], "judge_model": M[jk], "judge_family": JUDGE_FAMILY[NAME[jk]],
            "order": "AB", "prompt_version": "P1", "rep": 1,
            "gen_a": p["gen_a"], "gen_b": p["gen_b"],
            "gen_a_family": GF[p["gen_a"]], "gen_b_family": GF[p["gen_b"]],
            "verdict_raw": v, "verdict": v, "parsed_ok": ok,
            "len_a": p["len_a"], "len_b": p["len_b"], "gt_better": p.get("gt_better"),
            "snapshot": r.get("snapshot"), "usage": r["usage"], "latency": r["latency"],
        })
        return 1

    res, err = llmclient.run_parallel(tasks, worker, n_workers=6, desc="E8")
    print(f"E8: {len(res)} records, {len(err)} errors")


def cmd_plan():
    usage = json.load(open(f"{R}/usage.json"))
    ceilings = {  # console free remaining 2026-10-17 (tokens)
        "qwen3.7-flash-2026-07-15": 412060, "qwen3.8-flash": 105330,
        "qwen3.8-27b": 148540, "qwen3.8-max-0902": 361960,
        "deepseek-v4-flash-0731": 285520, "deepseek-v4.1-flash": 262810,
        "deepseek-v4-pro-0813": 264310, "kimi-k3": 217440,
    }
    plan_calls = {}
    def add(jk, exp, n, tok_per=720):
        key = M[jk]
        plan_calls[key] = plan_calls.get(key, 0) + n * tok_per
    add("27b", "E0", 40); add("kimi", "E0", 40)
    for jk in E1_JUDGES:
        add(jk, "E1", 180)
    for jk in E2_JUDGES:
        add(jk, "E2", 90)
    for gk in E3_GENS:
        add(gk, "E3gen", 100, 800)
    for jk in E3_JUDGES:
        add(jk, "E3j", 115)
    for jk in E5_JUDGES:
        add(jk, "E5", 60)
    for jk in E8_JUDGES:
        add(jk, "E8", 60)
    print(f"{'model':34s} {'planned':>9s} {'remaining':>10s} {'util%':>7s}")
    total_p = total_r = 0
    for key in sorted(plan_calls):
        rem = ceilings.get(key, 0)
        p = plan_calls[key]
        total_p += p; total_r += rem
        print(f"{key:34s} {p/1000:8.1f}K {rem/1000:9.1f}K {p/rem*100:6.1f}%")
    print(f"{'TOTAL':34s} {total_p/1000:8.1f}K {total_r/1000:9.1f}K {total_p/total_r*100:6.1f}%")


CMDS = {
    "e0": cmd_e0, "subsets": cmd_subsets, "e1": cmd_e1, "e2": cmd_e2,
    "e3gen": cmd_e3gen, "e3pair": cmd_e3pair, "e3judge": cmd_e3judge,
    "e5": cmd_e5, "e8": cmd_e8, "plan": cmd_plan,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(__doc__)
        print("commands:", " ".join(CMDS))
        sys.exit(1)
    CMDS[sys.argv[1]]()
