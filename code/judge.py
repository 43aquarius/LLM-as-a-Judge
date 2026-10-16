#!/usr/bin/env python3
"""Run pairwise judging: 13 judges x pairs x 2 orders (+ test-retest subset).

Judging prompt: MT-Bench-style rubric with reference answer for GT domains.
Verdict: [[A]] / [[B]] / [[C]] on the final line; strict parser with one retry.
Resumable via judgments.jsonl.

Usage:
  python3 judge.py            # main judging pass
  python3 judge.py --retest   # test-retest subset (2 repeats per selected pair)
"""
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llmclient

RESEARCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
QUESTIONS = os.path.join(RESEARCH, "questions.jsonl")
PAIRS = os.path.join(RESEARCH, "pairs.jsonl")
OUT = os.path.join(RESEARCH, "judgments.jsonl")

JUDGES = [
    ("qwen3.7-flash-2026-07-15", "Qwen3.7-Flash"),
    ("qwen3.8-flash", "Qwen3.8-Flash"),
    ("qwen3.8-27b", "Qwen3.8-27B"),
    ("qwen3.8-max-0902", "Qwen3.8-Max"),
    ("qwen3.8-2.4t-a95b", "Qwen3.8-2.4T"),
    ("deepseek-v4-flash-0731", "DeepSeek-V4-Flash"),
    ("deepseek-v4.1-flash", "DeepSeek-V4.1-Flash"),
    ("deepseek-v4-pro-0813", "DeepSeek-V4-Pro"),
    ("glm-5.3", "GLM-5.3"),
    ("kimi-k3", "Kimi-K3"),
]

JUDGE_FAMILY = {
    "Qwen3.7-Flash": "Qwen", "Qwen3.8-Flash": "Qwen", "Qwen3.8-27B": "Qwen",
    "Qwen3.8-Max": "Qwen", "Qwen3.8-2.4T": "Qwen",
    "DeepSeek-V4-Flash": "DeepSeek", "DeepSeek-V4.1-Flash": "DeepSeek",
    "DeepSeek-V4-Pro": "DeepSeek", "GLM-5.3": "GLM", "Kimi-K3": "Kimi",
}

GEN_FAMILY = {
    "Qwen3.8-Max": "Qwen", "Qwen3.8-Flash": "Qwen",
    "DeepSeek-V4-Pro": "DeepSeek", "GLM-5.3": "GLM", "Kimi-K3": "Kimi",
}

MAX_ANSWER_CHARS = 900
MAX_QUESTION_CHARS = 1500

SYSTEM = (
    "You are an impartial evaluator. You will be shown a question and two answers "
    "from different AI assistants. Evaluate which answer is better according to "
    "correctness, completeness, clarity, and how well it follows the question. "
    "Do not reward an answer merely because it is longer or more elaborate. "
    "Judge only on quality. Reply with a brief justification of one or two "
    "sentences, and end your reply with a final verdict line containing exactly "
    "one of: [[A]] if Assistant A is better, [[B]] if Assistant B is better, "
    "or [[C]] if the two answers are similarly good."
)

VERDICT_PAT = re.compile(r"\[\[\s*([ABCabc])\s*\]\]")


def parse_verdict(text: str) -> str | None:
    ms = VERDICT_PAT.findall(text.upper().replace("[[A]]", "[[A]]"))
    if not ms:
        # tolerate formats like "Verdict: A" or final "A." on last line
        last = text.strip().split("\n")[-1].upper()
        m = re.search(r"\b([ABC])\b", last)
        if m:
            return m.group(1)
        return None
    return ms[-1].upper()


def clip(t: str, n: int) -> str:
    t = t.strip()
    return t if len(t) <= n else t[:n] + " ...[truncated]"


def make_user_msg(question, ans_a, ans_b, ref):
    q = clip(question, MAX_QUESTION_CHARS)
    ref_block = ""
    if ref:
        ref_block = "\n### Reference answer (ground truth; use it to check correctness only):\n" + clip(str(ref), 400)
    return (
        "### Question:\n" + q + ref_block
        + "\n\n### Assistant A's answer:\n" + clip(ans_a, MAX_ANSWER_CHARS)
        + "\n\n### Assistant B's answer:\n" + clip(ans_b, MAX_ANSWER_CHARS)
        + "\n\n### Your evaluation (brief justification, then final verdict line):"
    )


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def load_done_keys(outpath):
    keys = set()
    if os.path.exists(outpath):
        with open(outpath) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    keys.add((r["pair_id"], r["judge"], r["order"], r.get("rep", 0)))
                except Exception:
                    pass
    return keys


def judge_once(jmodel, jname, user_msg, task_id, max_tokens):
    r = llmclient.chat(
        jmodel,
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}],
        max_tokens=max_tokens, temperature=0.0, task_id=task_id,
    )
    v = parse_verdict(r["content"])
    parsed_ok = v is not None
    if not parsed_ok:
        # one reminder retry
        r2 = llmclient.chat(
            jmodel,
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg},
             {"role": "assistant", "content": clip(r["content"], 2000)},
             {"role": "user", "content": "Your reply did not contain a valid verdict line. Reply again and end with exactly [[A]], [[B]], or [[C]]."}],
            max_tokens=max_tokens, temperature=0.0, task_id=task_id + ":retry",
        )
        v = parse_verdict(r2["content"])
        parsed_ok = v is not None
        r = r2 if parsed_ok else r
    return v, parsed_ok, r


def main(retest: bool = False):
    questions = {q["qid"]: q for q in load_jsonl(QUESTIONS)}
    pairs = load_jsonl(PAIRS)
    if retest:
        rng = random.Random(20261017)
        by_dom = {}
        for p in pairs:
            by_dom.setdefault(p["domain"], []).append(p)
        selected = []
        for dom, lst in by_dom.items():
            lst2 = sorted(lst, key=lambda x: x["pair_id"])
            selected.extend(rng.sample(lst2, min(10, len(lst2))))
        pairs = selected
        out_path = OUT.replace("judgments", "retests")
        reps = 2
        print(f"retest pairs: {len(pairs)} x {len(JUDGES)} judges x order AB x {reps} reps")
    else:
        out_path = OUT
        reps = 1

    done = load_done_keys(out_path)
    tasks = []
    for p in pairs:
        q = questions[p["qid"]]
        orders = ["AB"] if retest else ["AB", "BA"]
        for rep in range(1 if not retest else reps):
            for order in orders:
                for jmodel, jname in JUDGES:
                    key = (p["pair_id"], jname, order, rep if retest else 0)
                    if key in done:
                        continue
                    tasks.append({"p": p, "q": q, "jmodel": jmodel, "jname": jname,
                                  "order": order, "rep": rep if retest else 0})
    print(f"judging tasks: {len(tasks)} (skipped done: {len(done)})", flush=True)

    # Seeded shuffle: if a model's free quota runs out mid-run, the skipped
    # subset is spread uniformly over pairs/orders instead of truncating the
    # tail of the pair list (which is domain-ordered).
    random.Random(20261016).shuffle(tasks)

    forced = {m for m, mode in llmclient.THINKING_MODE.items() if mode == "forced"}

    def worker(t):
        p, q = t["p"], t["q"]
        if t["order"] == "AB":
            ga, gb, ta, tb = p["gen_a"], p["gen_b"], p["text_a"], p["text_b"]
        else:
            ga, gb, ta, tb = p["gen_b"], p["gen_a"], p["text_b"], p["text_a"]
        user_msg = make_user_msg(q["question"], ta, tb, p.get("ref"))
        mt = 900 if t["jmodel"] in forced else 400
        task_id = f'judge:{p["pair_id"]}:{t["jname"]}:{t["order"]}:r{t["rep"]}'
        try:
            v, ok, r = judge_once(t["jmodel"], t["jname"], user_msg, task_id, mt)
        except llmclient.QuotaExceeded:
            return None
        except Exception as e:
            with open(out_path + ".errors", "a") as f:
                f.write(json.dumps({"task": task_id, "error": str(e)[:200]}) + "\n")
            return None
        # normalize verdict to canonical side (A side is p['gen_a'] in AB order)
        canon = None
        if v == "A":
            canon = "A" if t["order"] == "AB" else "B"
        elif v == "B":
            canon = "B" if t["order"] == "AB" else "A"
        elif v == "C":
            canon = "TIE"
        rec = {
            "pair_id": p["pair_id"], "qid": p["qid"], "domain": p["domain"],
            "judge": t["jname"], "judge_model": t["jmodel"],
            "judge_family": JUDGE_FAMILY[t["jname"]],
            "order": t["order"], "rep": t["rep"],
            "gen_a": p["gen_a"], "gen_b": p["gen_b"],
            "gen_a_family": GEN_FAMILY[p["gen_a"]], "gen_b_family": GEN_FAMILY[p["gen_b"]],
            "verdict_raw": v, "verdict": canon, "parsed_ok": ok,
            "len_a": p["len_a"], "len_b": p["len_b"], "gt_better": p.get("gt_better"),
            "content_chars": len(r["content"]), "usage": r["usage"], "latency": r["latency"],
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    results, errors = llmclient.run_parallel(tasks, worker, n_workers=6, desc="judge")
    print(f"judged {len(results)}, errors {len(errors)}")


if __name__ == "__main__":
    main(retest="--retest" in sys.argv)
