#!/usr/bin/env python3
"""Relabel answer correctness with refined criteria (v2 labels).

Math (500 answers): numeric comparison of the extracted final answer against
the reference (v1 used string equality, so '162.00' != '162').
Knowledge (250 answers): independent LLM labeling with qwen3.7-flash
(a free-tier model that is NOT part of the judge panel, avoiding circularity).

Outputs:
  - research/answers_relabel.jsonl  (qid, generator, domain, correct_v2, evidence)
  - prints a summary + heuristic-vs-LLM disagreement stats for knowledge

The pairs' gt_better fields are updated separately (relabel_pairs.py) after
all judging finishes.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llmclient

RESEARCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
QUESTIONS = os.path.join(RESEARCH, "questions.jsonl")
ANSWERS = os.path.join(RESEARCH, "answers.jsonl")
OUT = os.path.join(RESEARCH, "answers_relabel.jsonl")

LABELER_MODEL = "qwen3.7-flash"  # free tier, NOT in the judge panel

# isolate this process's ledger/log so it cannot interfere with the running judge
llmclient.USAGE_FILE = os.path.join(RESEARCH, "label_usage.json")
llmclient.CALL_LOG = os.path.join(RESEARCH, "label_calls.jsonl")
llmclient.MODEL_BUDGETS[LABELER_MODEL] = 950_000


def num_eq(a: str | None, b: str | None) -> bool:
    """Numeric equality with normalization; False on parse failure."""
    if a is None or b is None:
        return False
    try:
        fa = float(str(a).replace(",", "").replace("$", "").strip())
        fb = float(str(b).replace(",", "").replace("$", "").strip())
    except ValueError:
        return False
    return abs(fa - fb) < 1e-6


def math_final(content: str):
    m = re.findall(r"[Aa]nswer[:\s]*\$?([\d,.\-]+)", content)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?[\d,]*\d\.?\d*", content)
    return nums[-1].replace(",", "") if nums else None


def know_label_prompt(question, ref, cand):
    return (
        "Question: " + question.strip()[:1200]
        + "\n\nReference (factually correct) answer: " + str(ref).strip()[:600]
        + "\n\nCandidate answer: " + cand.strip()[:1500]
        + "\n\nTask: Does the candidate answer endorse the same fact as the "
          "reference answer? Ignore style, length, and formatting. Reply with "
          "exactly one word on the first line: CORRECT if the candidate's core "
          "factual claim matches the reference, or INCORRECT if it endorses a "
          "different (e.g. misconception-based) claim, hedges away from the "
          "reference, or does not answer. Then give a one-sentence reason."
    )


def main():
    questions = {json.loads(l)["qid"]: json.loads(l) for l in open(QUESTIONS)}
    answers = [json.loads(l) for l in open(ANSWERS)]

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                r = json.loads(l)
                done.add((r["qid"], r["generator"]))
            except Exception:
                pass

    out_f = open(OUT, "a")

    # ---------- math: deterministic numeric relabel ----------
    n_math = n_flip = 0
    for a in answers:
        if a["domain"] != "math" or (a["qid"], a["generator"]) in done:
            continue
        fa = math_final(a["content"])
        v2 = num_eq(fa, a.get("reference"))
        rec = {"qid": a["qid"], "generator": a["generator"], "domain": "math",
               "correct_v2": v2, "final_answer_v2": fa,
               "correct_v1": a.get("correct"),
               "labeler": "numeric"}
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_math += 1
        n_flip += int(v2 != bool(a.get("correct")))
    out_f.flush()
    print(f"math relabeled: {n_math} (flips vs v1: {n_flip})", flush=True)

    # ---------- knowledge: LLM labeling ----------
    from concurrent.futures import ThreadPoolExecutor, as_completed
    sys.path.insert(0, RESEARCH)
    from llmclient import run_parallel

    tasks = []
    for a in answers:
        if a["domain"] != "knowledge" or (a["qid"], a["generator"]) in done:
            continue
        tasks.append(a)
    print(f"knowledge to label: {len(tasks)}", flush=True)

    def worker(a):
        q = questions[a["qid"]]
        user = know_label_prompt(q["question"], a.get("reference") or q.get("reference"), a["content"])
        try:
            r = llmclient.chat(
                LABELER_MODEL,
                [{"role": "user", "content": user}],
                max_tokens=200, temperature=0.0,
                task_id=f'label:{a["qid"]}:{a["generator"]}',
            )
        except Exception as e:
            return {"error": str(e)[:120]}
        first = r["content"].strip().split("\n")[0].upper()
        v2 = first.startswith("CORRECT") and not first.startswith("INCORRECT")
        return {"qid": a["qid"], "generator": a["generator"], "domain": "knowledge",
                "correct_v2": v2, "correct_v1": a.get("correct"),
                "labeler": LABELER_MODEL, "evidence": r["content"][:160]}

    def write_rec(r):
        if "error" not in r:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return r

    results, errors = run_parallel(tasks, lambda t: write_rec(worker(t)), n_workers=4, desc="label")
    out_f.close()

    ok = [r for r in results if "error" not in r]
    print(f"knowledge labeled: {len(ok)}, errors: {len(errors)}")
    flips = sum(1 for r in ok if r["correct_v2"] != bool(r["correct_v1"]))
    print(f"knowledge flips vs heuristic: {flips}/{len(ok)}")


if __name__ == "__main__":
    main()
