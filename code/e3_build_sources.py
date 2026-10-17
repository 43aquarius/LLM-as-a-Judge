#!/usr/bin/env python3
"""Build E3 task sources (review item 18): 50 MATH + 25 HumanEval code + 25 MMLU MCQ.

All selection is deterministic (seed 20261017) and logged. Correctness of
generator answers is decided by frozen deterministic parsers (no LLM labeler):
  math -> last \boxed{...} (fallback 'answer is X'), numeric/fraction compare
  code  -> subprocess: generated code + unit tests, 10s timeout, pass/fail
  mcq   -> last 'Answer: <letter>' (fallback standalone letter in tail)

Reference answers for MATH are filtered to plain numbers / simple fractions so
the deterministic parser has no LaTeX ambiguity.
Output: research/e3_tasks.jsonl
"""
import gzip
import json
import os
import random
import re
import subprocess
import sys
import tempfile

RESEARCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SRC = os.path.join(RESEARCH, "cache", "e3src")
OUT = os.path.join(RESEARCH, "e3_tasks.jsonl")
SEED = 20261017

MATH_SUBJECTS = ["intermediate_algebra", "number_theory", "precalculus",
                 "counting_and_probability"]
MMLU_HARD = ["abstract_algebra", "college_mathematics", "college_physics",
             "formal_logic", "high_school_mathematics", "machine_learning",
             "college_chemistry", "high_school_statistics",
             "electrical_engineering", "high_school_computer_science"]


def _parquet_ok(path):
    try:
        import pyarrow.parquet as pq
        pq.read_table(path)
        return True
    except Exception:
        return False


def dl(url, path):
    """curl-based download with validation + retry (urlretrieve truncated once)."""
    import subprocess
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        if path.endswith(".parquet"):
            if _parquet_ok(path):
                print(f"  cached {os.path.basename(path)} ({os.path.getsize(path)}B)")
                return
            print(f"  corrupted {os.path.basename(path)}, redownloading")
        else:
            print(f"  cached {os.path.basename(path)}")
            return
    for attempt in range(3):
        r = subprocess.run(["curl", "-sL", "--max-time", "180", "-o", path, url],
                           capture_output=True)
        ok = os.path.exists(path) and os.path.getsize(path) > 1000
        if ok and path.endswith(".parquet"):
            ok = _parquet_ok(path)
        if ok:
            print(f"  downloaded {os.path.basename(path)} {os.path.getsize(path)}B")
            return
        print(f"  retry {attempt + 1} for {os.path.basename(path)} "
              f"(rc={r.returncode}, size={os.path.getsize(path) if os.path.exists(path) else 0})")
    raise RuntimeError(f"download failed: {url}")


def load_math():
    import pyarrow.parquet as pq
    rows = []
    for s in MATH_SUBJECTS:
        p = os.path.join(SRC, f"math_{s}.parquet")
        dl(f"https://hf-mirror.com/datasets/EleutherAI/hendrycks_math/resolve/main/"
           f"{s}/test-00000-of-00001.parquet", p)
        t = pq.read_table(p).to_pylist()
        for r in t:
            r["subject"] = s
            rows.append(r)
    return rows


def boxed_last(text):
    ms = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return ms[-1] if ms else None


def norm_ans(a):
    if a is None:
        return None
    a = a.strip().strip("$").replace(",", "").replace(" ", "")
    a = a.replace("\\dfrac", "\\frac")
    a = re.sub(r"\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", r"\1/\2", a)
    a = re.sub(r"\\left|\\right|\\!|\\,", "", a)
    return a


def is_simple_numeric(a):
    if a is None:
        return False
    return re.fullmatch(r"-?\d+(\.\d+)?(/\d+)?", a) is not None


def ans_eq(a, b):
    """Frozen deterministic answer comparison (numbers / fractions)."""
    if a is None or b is None:
        return False
    try:
        if "/" in a or "/" in b:
            fa = eval_frac(a); fb = eval_frac(b)
            return abs(fa - fb) < 1e-9
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return a == b


def eval_frac(a):
    if "/" in str(a):
        n, d = str(a).split("/")
        return float(n) / float(d)
    return float(a)


# ---------- frozen output parsers (shared by correctness checks) ----------
def parse_math_out(content):
    ms = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", content)
    if ms:
        return norm_ans(ms[-1])
    m = re.findall(r"[Aa]nswer[:\s]*\$?(-?[\d,.\-/]+)", content)
    if m:
        return norm_ans(m[-1])
    nums = re.findall(r"-?[\d,]*\d\.?\d*", content)
    return norm_ans(nums[-1]) if nums else None


def parse_mcq_out(content):
    ms = re.findall(r"[Aa]nswer[:\s]*\(?([A-D])\)?", content)
    if ms:
        return ms[-1].upper()
    tail = [l for l in content.strip().splitlines() if l.strip()][-3:]
    for line in reversed(tail):
        m = re.search(r"^\(?([A-D])\)?[).:]?\s*$", line.strip())
        if m:
            return m.group(1)
    return None


def parse_code_out(content, entry_point):
    code = content.strip()
    if "```" in code:
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", code, re.S)
        if blocks:
            code = blocks[0].strip()
    # keep from first def/import line
    lines = code.splitlines()
    start = 0
    for i, l in enumerate(lines):
        if re.match(r"\s*(def |import |from |class )", l):
            start = i
            break
    code = "\n".join(lines[start:])
    return code


def check_code(code, entry_point, test_code, timeout=10):
    """Run generated code against unit tests in an isolated subprocess."""
    harness = (
        f"{code}\n\ntry:\n    candidate = {entry_point}\n"
        f"except Exception:\n    raise SystemExit(3)\n\n{test_code}\n"
        "print('ALL_TESTS_OK')\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(harness)
        path = f.name
    try:
        r = subprocess.run([sys.executable, "-I", path], capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode == 0 and "ALL_TESTS_OK" in r.stdout
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


# ---------------- task construction ----------------
def build():
    rng = random.Random(SEED)
    tasks = []

    # ---- MATH: Level 4-5, numeric/fraction boxed answers ----
    math_rows = load_math()
    pool = []
    for r in math_rows:
        if r["level"] not in ("Level 4", "Level 5"):
            continue
        ref = norm_ans(boxed_last(r["solution"]))
        if not is_simple_numeric(ref):
            continue
        pool.append({"subject": r["subject"], "level": r["level"],
                     "problem": r["problem"], "ref": ref})
    by_lvl = {"Level 4": [p for p in pool if p["level"] == "Level 4"],
              "Level 5": [p for p in pool if p["level"] == "Level 5"]}
    print(f"MATH pool: level4={len(by_lvl['Level 4'])} level5={len(by_lvl['Level 5'])}")
    picked = []
    for lvl, quota in [("Level 4", 25), ("Level 5", 25)]:
        lst = sorted(by_lvl[lvl], key=lambda x: (x["subject"], x["problem"]))
        picked.extend(rng.sample(lst, min(quota, len(lst))))
    for i, p in enumerate(picked):
        tasks.append({
            "tid": f"e3m-{i:02d}", "source": "math", "domain": "math",
            "question": p["problem"],
            "gen_prompt": ("Solve the following math problem. Show your reasoning "
                            "briefly, then end with the final answer inside \\boxed{...} "
                            "on the last line.\n\n" + p["problem"]),
            "reference": f"Final answer: {p['ref']}",
            "payload": {"kind": "math", "ref_answer": p["ref"]},
            "meta": {"subject": p["subject"], "level": p["level"]},
        })

    # ---- HumanEval code: 25 seeded over 164 ----
    he = [json.loads(l) for l in gzip.open(
        os.path.join(SRC, "HumanEval.jsonl.gz"), "rt", encoding="utf-8")]
    idx = sorted(rng.sample(range(len(he)), 25))
    for n, i in enumerate(idx):
        r = he[i]
        q = r["prompt"]
        tasks.append({
            "tid": f"e3c-{n:02d}", "source": "code", "domain": "code",
            "question": q,
            "gen_prompt": ("Complete the following Python function. Reply with only "
                            "the Python code implementing the function; no test code, "
                            "no usage examples, no markdown fences.\n\n" + q),
            "reference": "Reference (canonical) solution:\n" + r["canonical_solution"],
            "payload": {"kind": "code", "entry_point": r["entry_point"],
                        "test": r["test"], "canonical": r["canonical_solution"]},
            "meta": {"human_eval_idx": i, "task_id": r["task_id"]},
        })

    # ---- MMLU MCQ: 25 from hard subjects ----
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(SRC, "mmlu_test.parquet")).to_pylist()
    hard = [r for r in t if r["subject"] in MMLU_HARD]
    print(f"MMLU hard-subject pool: {len(hard)}")
    sel = rng.sample(hard, 25)
    for n, r in enumerate(sorted(sel, key=lambda x: x["subject"])):
        letters = "ABCD"
        opts = "\n".join(f"({letters[k]}) {c}" for k, c in enumerate(r["choices"]))
        q = r["question"] + "\n" + opts
        ai = int(r["answer"])
        tasks.append({
            "tid": f"e3q-{n:02d}", "source": "mcq", "domain": "mcq",
            "question": q,
            "gen_prompt": ("Answer the following multiple-choice question. End your "
                            "reply with 'Answer: <letter>' on the final line.\n\n" + q),
            "reference": f"Correct choice: ({letters[ai]}) {r['choices'][ai]}",
            "payload": {"kind": "mcq", "answer_idx": ai},
            "meta": {"subject": r["subject"]},
        })

    with open(OUT, "w") as f:
        for t_ in tasks:
            f.write(json.dumps(t_, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"e3 tasks: {len(tasks)} -> {OUT}")
    print("by source:", Counter(t_["source"] for t_ in tasks))
    print("math levels:", Counter(t_["meta"].get("level") for t_ in tasks if t_["source"] == "math"))
    print("code idx range:", min(t_["meta"]["human_eval_idx"] for t_ in tasks if t_["source"] == "code"),
          "-", max(t_["meta"]["human_eval_idx"] for t_ in tasks if t_["source"] == "code"))


def batch2():
    """Append 40 more Level-5 MATH tasks (disjoint from batch-1) to e3_tasks.jsonl."""
    import pyarrow.parquet as pq
    rng = random.Random(SEED + 10)
    tasks = jl_like = [json.loads(l) for l in open(OUT)] if os.path.exists(OUT) else []
    seen = {t["question"] for t in tasks if t["source"] == "math"}
    n_math = sum(1 for t in tasks if t["source"] == "math")
    pool = []
    for s in MATH_SUBJECTS:
        p = os.path.join(SRC, f"math_{s}.parquet")
        t = pq.read_table(p).to_pylist()
        for r in t:
            if r["level"] != "Level 5":
                continue
            ref = norm_ans(boxed_last(r["solution"]))
            if not is_simple_numeric(ref) or r["problem"] in seen:
                continue
            pool.append({"subject": s, "problem": r["problem"], "ref": ref})
    print(f"batch2 L5 pool (disjoint): {len(pool)}")
    picked = rng.sample(pool, 40)
    with open(OUT, "a") as f:
        for i, p_ in enumerate(picked):
            f.write(json.dumps({
                "tid": f"e3m-{n_math + i:02d}", "source": "math", "domain": "math",
                "batch": 2,
                "question": p_["problem"],
                "gen_prompt": ("Solve the following math problem. Show your reasoning "
                                "briefly, then end with the final answer inside \\boxed{...} "
                                "on the last line.\n\n" + p_["problem"]),
                "reference": f"Final answer: {p_['ref']}",
                "payload": {"kind": "math", "ref_answer": p_["ref"]},
                "meta": {"subject": p_["subject"], "level": "Level 5", "batch": 2},
            }, ensure_ascii=False) + "\n")
    print(f"batch2 appended 40 -> {OUT} (total {len(tasks) + 40})")


def selftest():
    """Frozen-parser self-tests (must all pass before any generation call)."""
    assert parse_math_out("so \\boxed{\\frac{1}{2}} done") == "1/2", parse_math_out("x \\boxed{\\frac{1}{2}}")
    assert ans_eq("1/2", "0.5")
    assert ans_eq("17/3", "17/3")
    assert not ans_eq("1/2", "0.6")
    assert parse_mcq_out("blah\nAnswer: (C)") == "C"
    assert parse_mcq_out("...\nfinal\nB.") == "B"
    code = "def add(a, b):\n    return a + b\n"
    ok = check_code(code, "add", "assert candidate(1,2)==3\nassert candidate(0,0)==0")
    bad = check_code("def add(a, b):\n    return a - b\n", "add",
                     "assert candidate(1,2)==3")
    assert ok and not bad, (ok, bad)
    print("parser self-tests passed")


if __name__ == "__main__":
    selftest()
    if "--build" in sys.argv:
        build()
    if "--batch2" in sys.argv:
        batch2()
