#!/usr/bin/env python3
"""Apply v2 correctness labels to pairs and all judgment files.

Steps:
  1. Build correct_v2 map from answers_relabel.jsonl (math numeric, knowledge LLM).
  2. Recompute gt_better for every pair: 'A'/'B' iff exactly one side correct_v2.
  3. Rewrite pairs.jsonl (adds gt_better_v2 -> replaces gt_better; archives v1).
  4. Retroactively update gt_better inside judgments.jsonl, judgments_agnes.jsonl,
     retests.jsonl, retests_agnes.jsonl (match by pair_id).
  5. Print before/after GT-clear statistics per domain.

Run AFTER all judging processes have exited.
"""
import json
import os
import shutil
import sys

RESEARCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SKIP_AGNES = "--skip-agnes" in sys.argv  # don't touch agnes files (a writer may still be appending)


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


def main():
    # 1. v2 correctness
    cor2 = {}
    for r in load_jsonl(os.path.join(RESEARCH, "answers_relabel.jsonl")):
        cor2[(r["qid"], r["generator"])] = bool(r["correct_v2"])
    print(f"v2 labels loaded: {len(cor2)}")

    # 1b. stamp correct_v2 back into answers.jsonl (keeps the release pipeline consistent)
    ans_path = os.path.join(RESEARCH, "answers.jsonl")
    answers = load_jsonl(ans_path)
    n_stamp = 0
    for a in answers:
        v2 = cor2.get((a["qid"], a["generator"]))
        if v2 is not None:
            a["correct_v1"] = a.get("correct")
            a["correct"] = v2  # final label is authoritative
            n_stamp += 1
    shutil.copy(ans_path, ans_path + ".v1backup")
    with open(ans_path, "w") as f:
        for a in answers:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"answers.jsonl stamped with v2 labels: {n_stamp}/{len(answers)}")

    # 2-3. pairs
    pairs = load_jsonl(os.path.join(RESEARCH, "pairs.jsonl"))
    n_clear_v1 = n_clear_v2 = 0
    dom_stats = {}
    for p in pairs:
        v1 = p.get("gt_better")
        ca = cor2.get((p["qid"], p["gen_a"]))
        cb = cor2.get((p["qid"], p["gen_b"]))
        if ca is True and cb is False:
            v2 = "A"
        elif cb is True and ca is False:
            v2 = "B"
        else:
            v2 = None
        p["gt_better_v1"] = v1
        p["gt_better"] = v2
        d = dom_stats.setdefault(p["domain"], {"n": 0, "c1": 0, "c2": 0, "flip": 0})
        d["n"] += 1
        d["c1"] += int(v1 in ("A", "B"))
        d["c2"] += int(v2 in ("A", "B"))
        d["flip"] += int((v1 in ("A", "B")) != (v2 in ("A", "B")) or
                         (v1 in ("A", "B") and v2 in ("A", "B") and v1 != v2))
        n_clear_v1 += int(v1 in ("A", "B"))
        n_clear_v2 += int(v2 in ("A", "B"))
    shutil.copy(os.path.join(RESEARCH, "pairs.jsonl"),
                os.path.join(RESEARCH, "pairs_v1_backup.jsonl"))
    with open(os.path.join(RESEARCH, "pairs.jsonl"), "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"GT-clear pairs: v1={n_clear_v1} -> v2={n_clear_v2}")
    for d, s in sorted(dom_stats.items()):
        print(f"  {d:10s} n={s['n']:3d} clear_v1={s['c1']:3d} clear_v2={s['c2']:3d} changed={s['flip']}")

    # 4. retro-update judgment files
    gt = {p["pair_id"]: p["gt_better"] for p in pairs}
    targets = ["judgments.jsonl", "retests.jsonl"]
    if not SKIP_AGNES:
        targets += ["judgments_agnes.jsonl", "retests_agnes.jsonl"]
    for fname in targets:
        path = os.path.join(RESEARCH, fname)
        if not os.path.exists(path):
            continue
        rows = load_jsonl(path)
        seen = set()
        out = []
        for r in rows:
            key = (r["pair_id"], r["judge"], r["order"], r.get("rep", 0))
            if key in seen:
                continue
            seen.add(key)
            if r["pair_id"] in gt:
                r["gt_better"] = gt[r["pair_id"]]
            out.append(r)
        shutil.copy(path, path + ".v1gt")
        with open(path, "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"updated {fname}: {len(out)} rows")

    print("done.")


if __name__ == "__main__":
    main()
