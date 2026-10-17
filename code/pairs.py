#!/usr/bin/env python3
"""Construct evaluation pairs (v2: quality-stratified in GT domains).

Strategy:
  - math/knowledge (GT domains): 3 pairs per question. On questions where
    generators disagree in correctness, pairs prefer one-correct-vs-one-wrong
    (GT-clear) combinations, rotating wrong answers; remaining slots filled
    round-robin. This maximizes the number of pairs with objective quality
    difference for accuracy / kin-preference analyses.
  - creative/summ (no GT): 1 pair per question, round-robin over the 10
    generator pairings for balance.

Total: 100*3 + 100*1 = 400 pairs.
"""
import itertools
import json
import os
import random
from collections import defaultdict

RESEARCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ANSWERS = os.path.join(RESEARCH, "answers.jsonl")
OUT = os.path.join(RESEARCH, "pairs.jsonl")

SEED = 20261016
GEN_NAMES = ["Qwen3.8-Max", "Qwen3.8-Flash", "DeepSeek-V4-Pro", "GLM-5.3", "Kimi-K3"]
PAIR_TYPES = list(itertools.combinations(GEN_NAMES, 2))


def main():
    rng = random.Random(SEED)
    answers = [json.loads(l) for l in open(ANSWERS)]
    by_q = defaultdict(dict)
    for a in answers:
        by_q[a["qid"]][a["generator"]] = a

    out = []
    for domain, n_pairs in [("math", 3), ("knowledge", 3), ("creative", 1), ("summ", 1)]:
        qids = sorted([qid for qid, d in by_q.items()
                       if next(iter(d.values()))["domain"] == domain])
        rng.shuffle(qids)
        rr = itertools.cycle(PAIR_TYPES)
        for qid in qids:
            d = by_q[qid]
            chosen = []
            if domain in ("math", "knowledge"):
                corr = [g for g in GEN_NAMES if d[g].get("correct_v2", d[g].get("correct")) is True]
                wrong = [g for g in GEN_NAMES if d[g].get("correct_v2", d[g].get("correct")) is False]
                if corr and wrong:
                    # GT-clear first: rotate wrong answers against correct ones
                    wi = 0
                    for k in range(n_pairs):
                        wa = wrong[wi % len(wrong)]
                        wb = rng.choice([g for g in corr if g != wa])
                        chosen.append((wa, wb))
                        wi += 1
                    # ensure the three pairs are distinct pairs
                    if len(set(map(frozenset, chosen))) < len(chosen):
                        seen, uniq = set(), []
                        for p in chosen:
                            key = frozenset(p)
                            if key not in seen:
                                seen.add(key); uniq.append(p)
                        gi = 0
                        while len(uniq) < n_pairs:
                            g1, g2 = corr[gi % len(corr)], wrong[(gi // len(corr)) % len(wrong)]
                            if frozenset((g1, g2)) not in seen:
                                seen.add(frozenset((g1, g2))); uniq.append((g1, g2))
                            gi += 1
                        chosen = uniq[:n_pairs]
                else:
                    for k in range(n_pairs):
                        chosen.append(next(rr))
            else:
                chosen.append(next(rr))

            for k, (ga0, gb0) in enumerate(chosen):
                ga, gb = ga0, gb0
                if rng.random() < 0.5:
                    ga, gb = gb, ga
                aa, ab = d[ga], d[gb]
                gt = None
                if domain in ("math", "knowledge"):
                    ca, cb = aa.get("correct_v2", aa.get("correct")), ab.get("correct_v2", ab.get("correct"))
                    if ca is True and cb is False:
                        gt = "A"
                    elif cb is True and ca is False:
                        gt = "B"
                out.append({
                    "pair_id": f"{qid}-p{k}", "qid": qid, "domain": domain,
                    "gen_a": ga, "gen_b": gb,
                    "text_a": aa["content"], "text_b": ab["content"],
                    "len_a": len(aa["content"]), "len_b": len(ab["content"]),
                    "ref": aa.get("reference"),
                    "gt_better": gt,
                })

    with open(OUT, "w") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"total pairs: {len(out)}")
    print("per domain:", Counter(p["domain"] for p in out))
    print("gt-clear pairs:", Counter(p["domain"] for p in out if p["gt_better"]))
    pt = Counter(tuple(sorted([p["gen_a"], p["gen_b"]])) for p in out)
    for k, v in sorted(pt.items()):
        print(f"  {k[0]} vs {k[1]}: {v}")
    # kin-wrong availability per family
    FAM = {"Qwen3.8-Max": "Qwen", "Qwen3.8-Flash": "Qwen", "DeepSeek-V4-Pro": "DeepSeek",
           "GLM-5.3": "GLM", "Kimi-K3": "Kimi"}
    for fam in ["Qwen", "DeepSeek", "GLM", "Kimi"]:
        n_kin = n_kw = n_kr = 0
        for p in out:
            fa, fb = FAM[p["gen_a"]], FAM[p["gen_b"]]
            kin_side = None
            if fa == fam and fb != fam: kin_side = "A"
            elif fb == fam and fa != fam: kin_side = "B"
            if kin_side is None: continue
            n_kin += 1
            if p["gt_better"]:
                if p["gt_better"] == kin_side: n_kr += 1
                else: n_kw += 1
        print(f"{fam}: kin pairs={n_kin}, kin-right={n_kr}, kin-wrong={n_kw}")


if __name__ == "__main__":
    main()
