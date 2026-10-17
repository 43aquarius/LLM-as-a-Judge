#!/usr/bin/env python3
"""Agnes proprietary free-tier judges (agnes-2.5-flash, agnes-3.0-flash, both $0-priced).

Same protocol as judge.py (MT-Bench-style rubric, [[A]]/[[B]]/[[C]] verdict).
Writes to judgments_agnes.jsonl / retests_agnes.jsonl (kept separate from the
open-source judges' files; analyze.py merges both).

Usage:
  python3 agnes_judge.py            # main pass: 400 pairs x 2 orders x 2 models
  python3 agnes_judge.py --retest   # same 40-pair subset x 2 reps x order AB
"""
import json
import os
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge import SYSTEM, make_user_msg, parse_verdict, load_jsonl, load_done_keys, JUDGE_FAMILY, GEN_FAMILY

RESEARCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
QUESTIONS = os.path.join(RESEARCH, "questions.jsonl")
PAIRS = os.path.join(RESEARCH, "pairs.jsonl")
OUT = os.path.join(RESEARCH, "judgments_agnes.jsonl")

AGNES_JUDGES = [
    ("agnes-2.5-flash", "Agnes-2.5-Flash"),
    ("agnes-3.0-flash", "Agnes-3.0-Flash"),
]
AGNES_FAMILY = {"Agnes-2.5-Flash": "Agnes", "Agnes-3.0-Flash": "Agnes"}

_log_lock = threading.Lock()
CALL_LOG = os.path.join(RESEARCH, "agnes_calls.jsonl")


def get_key():
    k = os.environ.get("AGNES_API_KEY")
    if k:
        return k
    raise RuntimeError("export AGNES_API_KEY first")


KEY = get_key()
_pace_lock = threading.Lock()
_last_call_ts = [0.0]


def agnes_chat(model, messages, max_tokens=800, temperature=0.0, max_retries=10):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature}
    last_err = None
    for attempt in range(max_retries):
        # gentle pacing: single worker, 8.5s spacing (~7 RPM, under the
        # documented 10 RPM free-user limit; avoids penalty escalation)
        with _pace_lock:
            wait = 8.5 - (time.time() - _last_call_ts[0])
            if wait > 0:
                time.sleep(wait)
            _last_call_ts[0] = time.time()
        req = urllib.request.Request(
            "https://apihub.agnes-ai.com/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                d = json.loads(r.read())
            msg = (d.get("choices") or [{}])[0].get("message", {})
            usage = d.get("usage", {})
            with _log_lock:
                with open(CALL_LOG, "a") as f:
                    f.write(json.dumps({
                        "ts": time.time(), "model": model,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "latency": round(time.time() - t0, 2),
                    }) + "\n")
            return {
                "content": (msg.get("content") or "").strip(),
                "usage": usage,
                "latency": round(time.time() - t0, 2),
                "model": model,
            }
        except urllib.error.HTTPError as e:
            try:
                err = e.read().decode()[:300]
            except Exception:
                err = str(e)
            last_err = f"HTTP {e.code}: {err}"
            if e.code == 429:
                # free-user rate limit: be patient, wait out the window
                time.sleep(min(30 + 30 * attempt + random.random() * 10, 150))
                continue
            if e.code in (500, 502, 503, 504):
                time.sleep(min(2 ** attempt * 2, 30))
                continue
            raise RuntimeError(last_err)
        except Exception as e:
            last_err = str(e)[:200]
            time.sleep(min(2 ** attempt * 2, 20))
    raise RuntimeError(f"{model} failed after {max_retries} retries: {last_err}")


def judge_once(model, user_msg, task_id, max_tokens=800):
    r = agnes_chat(model, [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": user_msg}], max_tokens=max_tokens)
    v = parse_verdict(r["content"])
    ok = v is not None
    if not ok:
        r2 = agnes_chat(model, [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg},
            {"role": "assistant", "content": r["content"][:2000]},
            {"role": "user", "content": "Your reply did not contain a valid verdict line. Reply again and end with exactly [[A]], [[B]], or [[C]]."},
        ], max_tokens=max_tokens)
        v2 = parse_verdict(r2["content"])
        if v2 is not None:
            return v2, True, r2
        return None, False, r
    return v, ok, r


def clip(t, n):
    t = t.strip()
    return t if len(t) <= n else t[:n] + " ...[truncated]"


def main(retest: bool = False, time_budget: float = 0):
    t_start = time.time()
    questions = {q["qid"]: q for q in load_jsonl(QUESTIONS)}
    pairs = load_jsonl(PAIRS)
    if retest:
        rng = random.Random(20261017)  # SAME seed/subset as the open-judge retest
        by_dom = {}
        for p in pairs:
            by_dom.setdefault(p["domain"], []).append(p)
        selected = []
        for dom, lst in by_dom.items():
            lst2 = sorted(lst, key=lambda x: x["pair_id"])
            selected.extend(rng.sample(lst2, min(10, len(lst2))))
        pairs = selected
        out_path = OUT.replace("judgments_agnes", "retests_agnes")
        reps = 2
        print(f"retest pairs: {len(pairs)} x {len(AGNES_JUDGES)} judges x order AB x {reps} reps")
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
                for amodel, aname in AGNES_JUDGES:
                    key = (p["pair_id"], aname, order, rep if retest else 0)
                    if key in done:
                        continue
                    tasks.append({"p": p, "q": q, "amodel": amodel, "aname": aname,
                                  "order": order, "rep": rep if retest else 0})
    print(f"agnes judging tasks: {len(tasks)} (skipped done: {len(done)})", flush=True)

    def worker(t):
        p, q = t["p"], t["q"]
        if t["order"] == "AB":
            ga, gb, ta, tb = p["gen_a"], p["gen_b"], p["text_a"], p["text_b"]
        else:
            ga, gb, ta, tb = p["gen_b"], p["gen_a"], p["text_b"], p["text_a"]
        user_msg = make_user_msg(q["question"], ta, tb, p.get("ref"))
        task_id = f'agnes:{p["pair_id"]}:{t["aname"]}:{t["order"]}:r{t["rep"]}'
        try:
            v, ok, r = judge_once(t["amodel"], t["aname"] and user_msg, task_id)
        except Exception as e:
            with open(out_path + ".errors", "a") as f:
                f.write(json.dumps({"task": task_id, "error": str(e)[:200]}) + "\n")
            return None
        canon = None
        if v == "A":
            canon = "A" if t["order"] == "AB" else "B"
        elif v == "B":
            canon = "B" if t["order"] == "AB" else "A"
        elif v == "C":
            canon = "TIE"
        rec = {
            "pair_id": p["pair_id"], "qid": p["qid"], "domain": p["domain"],
            "judge": t["aname"], "judge_model": t["amodel"],
            "judge_family": AGNES_FAMILY[t["aname"]],
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

    results, errors = [], []
    done_n = 0
    total = len(tasks)
    stop_submit = [False]
    with ThreadPoolExecutor(max_workers=1) as ex:
        futs = set()
        for t in tasks:
            if stop_submit[0]:
                break
            if time_budget and time.time() - t_start > time_budget:
                print(f"time budget ({time_budget}s) reached; remaining tasks will resume next run", flush=True)
                break
            futs.add(ex.submit(worker, t))
        for fut in as_completed(futs):
            try:
                rr = fut.result()
                if rr is not None:
                    results.append(rr)
            except Exception as e:
                errors.append(str(e)[:100])
            done_n += 1
            if done_n % 25 == 0 or done_n == total:
                print(f"  [agnes] {done_n}/{total} done, {len(errors)} errors", flush=True)
    print(f"agnes judged {len(results)}, errors {len(errors)}")


if __name__ == "__main__":
    tb = 0.0
    for a in sys.argv:
        if a.startswith("--time-budget="):
            tb = float(a.split("=", 1)[1])
    main(retest="--retest" in sys.argv, time_budget=tb)
