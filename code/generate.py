#!/usr/bin/env python3
"""Generate candidate answers: 5 generators x 200 questions.

Resumable via answers.jsonl (skip existing task keys).
Output record: {qid, domain, generator, content, correct(optional), usage, latency}
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llmclient

RESEARCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
QUESTIONS = os.path.join(RESEARCH, "questions.jsonl")
OUT = os.path.join(RESEARCH, "answers.jsonl")

GENERATORS = [
    ("qwen3.8-max-0902", "Qwen3.8-Max"),
    ("qwen3.8-flash", "Qwen3.8-Flash"),
    ("deepseek-v4-pro-0813", "DeepSeek-V4-Pro"),
    ("glm-5.3", "GLM-5.3"),
    ("kimi-k3", "Kimi-K3"),
]

GEN_FAMILY = {
    "Qwen3.8-Max": "Qwen", "Qwen3.8-Flash": "Qwen",
    "DeepSeek-V4-Pro": "DeepSeek", "GLM-5.3": "GLM", "Kimi-K3": "Kimi",
}

SYSTEM = "You are a helpful assistant. Answer the user's question or follow the instruction."

MAX_TOKENS = {"glm-5.3": 1600}  # forced-thinking model needs headroom
DEFAULT_MAX = 900
# kimi-k3 rejects fractional temperatures (0.7); use 1.0 for sampling diversity
TEMP_OVERRIDE = {"kimi-k3": 1.0}
DEFAULT_TEMP = 0.7


def num_eq(a, b) -> bool:
    """Numeric equality after normalization ('162.00' == '162')."""
    if a is None or b is None:
        return False
    try:
        fa = float(str(a).replace(",", "").replace("$", "").strip())
        fb = float(str(b).replace(",", "").replace("$", "").strip())
    except ValueError:
        return False
    return abs(fa - fb) < 1e-6


def math_final(content: str) -> str | None:
    m = re.findall(r"[Aa]nswer[:\s]*\$?([\d,.\-]+)", content)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?[\d,]*\d\.?\d*", content)
    return nums[-1].replace(",", "") if nums else None


def know_match(content: str, ref: str) -> bool:
    c = content.lower()
    r = ref.lower().strip()
    # tolerant containment check on key tokens
    toks = [t for t in re.findall(r"[a-z]{4,}", r)]
    if not toks:
        return r in c
    hit = sum(1 for t in toks if t in c)
    return hit / len(toks) >= 0.6


def load_done():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(f'{r["qid"]}|{r["generator"]}')
                except Exception:
                    pass
    return done


def main():
    questions = [json.loads(l) for l in open(QUESTIONS)]
    done = load_done()
    tasks = []
    for q in questions:
        for gmodel, gname in GENERATORS:
            key = f'{q["qid"]}|{gname}'
            if key in done:
                continue
            tasks.append({"qid": q["qid"], "domain": q["domain"],
                          "question": q["question"], "reference": q.get("reference"),
                          "gmodel": gmodel, "gname": gname})
    print(f"generation tasks: {len(tasks)} (done previously: {len(done)})", flush=True)

    def worker(t):
        try:
            r = llmclient.chat(
                t["gmodel"],
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": t["question"]}],
                max_tokens=MAX_TOKENS.get(t["gmodel"], DEFAULT_MAX),
                temperature=TEMP_OVERRIDE.get(t["gmodel"], DEFAULT_TEMP),
                task_id=f'gen:{t["qid"]}:{t["gname"]}',
            )
        except llmclient.QuotaExceeded:
            return None
        content = r["content"]
        if not content and r.get("reasoning"):
            content = "[no direct answer]"
        rec = {
            "qid": t["qid"], "domain": t["domain"], "generator": t["gname"],
            "generator_model": t["gmodel"], "family": GEN_FAMILY[t["gname"]],
            "content": content, "reference": t["reference"],
            "usage": r["usage"], "latency": r["latency"],
        }
        if t["domain"] == "math":
            rec["final_answer"] = math_final(content)
            rec["correct"] = num_eq(rec["final_answer"], t["reference"])
        elif t["domain"] == "knowledge":
            rec["correct"] = know_match(content, t["reference"] or "")
        with open(OUT, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    results, errors = llmclient.run_parallel(tasks, worker, n_workers=5, desc="gen")
    print(f"generated {len(results)}, errors {len(errors)}")
    for e in errors[:10]:
        print("ERR", e)


if __name__ == "__main__":
    main()
