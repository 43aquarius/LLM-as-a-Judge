#!/usr/bin/env python3
"""E7: external premium anchors + platform replication, on Intern AI Discovery API.

Phases (sequential, resumable):
  1. check   : cross-platform spot check, 4 overlap models x 48 stratified cells
               (for GLM-5.3 this doubles as the snapshot-compatibility gate)
  2. fill    : GLM-5.3 gap fill (126 missing main cells) -- only if gate passes
               + GLM-5.3 retest (40 pairs x AB x 2 reps, same seeded selection
               as the original retest battery) -- run regardless of gate
  3. anchor  : two new-family premium anchors, 400 pairs x 2 orders each:
               intern-s2 (family Intern) + minimax-m3 (family MiniMax)

Prompts are imported verbatim from judge.py (SYSTEM / make_user_msg / parse_verdict),
so anchor and fill conditions are byte-identical to the main evaluation.

Zero-cost guard: any 402/403-payment/quota response trips a global stop file and
aborts all workers (never bills, never retries a quota signal).
"""
import json, os, random, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge import parse_verdict, make_user_msg, SYSTEM, load_jsonl  # noqa: E402

BASE = os.environ["INTERNAI_BASE_URL"]
KEY = os.environ["INTERNAI_API_KEY"]
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

USAGE = os.path.join(R, "internai_usage.json")
STOP_FILE = os.path.join(R, "internai_STOP")
OUT_CHECK = os.path.join(R, "e7_platform_check.jsonl")
OUT_FILL = os.path.join(R, "e7_glm_fill.jsonl")
OUT_RETEST = os.path.join(R, "e7_glm_retest.jsonl")
OUT_ANCHOR = os.path.join(R, "judgments_e7.jsonl")

MAXTOK = {"intern-s2": 1500, "minimax-m3": 800, "glm-5.3": 3000,
          "deepseek-v4-flash-0731": 1200, "deepseek-v4-pro-0813": 1200, "qwen3.8-27b": 2400}

# overlap models: platform model-id -> (study display name, family)
OVERLAP = {
    "deepseek-v4-flash-0731": ("DeepSeek-V4-Flash", "DeepSeek"),
    "deepseek-v4-pro-0813": ("DeepSeek-V4-Pro", "DeepSeek"),
    "glm-5.3": ("GLM-5.3", "GLM"),
    "qwen3.8-27b": ("Qwen3.8-27B", "Qwen"),
}
ANCHORS = {
    "intern-s2": ("Intern-S2", "Intern"),
    "minimax-m3": ("MiniMax-M3", "MiniMax"),
}

GEN_FAMILY = {"Qwen3.8-Max": "Qwen", "Qwen3.8-Flash": "Qwen", "DeepSeek-V4-Pro": "DeepSeek",
              "GLM-5.3": "GLM", "Kimi-K3": "Kimi"}

STOP = threading.Event()
_usage_lock = threading.Lock()
_pace_lock = threading.Lock()
_last_call_ts = [0.0]
_cooldown_until = {}  # per-model 429 cooldown ts

def _pace(model):
    """Global request pacer: >=0.7s between call starts, respects per-model cooldown."""
    while True:
        with _pace_lock:
            now = time.time()
            cd = _cooldown_until.get(model, 0.0)
            if cd > now:
                wait_cd = cd - now
            else:
                wait_cd = 0.0
            ready = _last_call_ts[0] + 0.7
            wait_iv = max(0.0, ready - now)
            if wait_cd <= 0 and wait_iv <= 0:
                _last_call_ts[0] = now
                return
            w = max(wait_cd, wait_iv)
        time.sleep(min(w, 5.0))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _record_usage(model, usage):
    with _usage_lock:
        u = json.load(open(USAGE)) if os.path.exists(USAGE) else {}
        e = u.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        e["calls"] += 1
        e["prompt_tokens"] += (usage or {}).get("prompt_tokens") or 0
        e["completion_tokens"] += (usage or {}).get("completion_tokens") or 0
        json.dump(u, open(USAGE + ".tmp", "w"), indent=1)
        os.replace(USAGE + ".tmp", USAGE)


class QuotaStop(Exception):
    pass


def ia_chat(model, messages, max_tokens, temperature=0.0, timeout=180):
    """One chat completion with retry/backoff. Raises QuotaStop on payment signals."""
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": temperature, "stream": False}).encode()
    delays = [2, 5, 12, 30, 60, 120]
    q429 = [0]  # 429s seen in this call
    for attempt, delay in enumerate([0] + delays):
        if delay:
            time.sleep(delay)
        if STOP.is_set():
            raise QuotaStop("stop file")
        _pace(model)
        req = urllib.request.Request(BASE + "/chat/completions", data=body, method="POST",
                                     headers={"Authorization": "Bearer " + KEY,
                                              "Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body_b = resp.read()
            try:
                r = json.loads(body_b)
            except (json.JSONDecodeError, ValueError):
                # 200 with empty/non-JSON body: treat as transient, retry
                log(f"  {model}: empty/non-JSON 200 body (len={len(body_b)}) attempt {attempt}")
                if attempt < len(delays) - 1:
                    continue
                raise RuntimeError("empty 200 body after retries")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode()[:300]
            except Exception:
                pass
            if e.code == 429:
                q429[0] += 1
                with _pace_lock:
                    _cooldown_until[model] = max(_cooldown_until.get(model, 0), time.time() + 90)
                if q429[0] % 3 == 1:
                    log(f"  {model}: 429 #{q429[0]} cooldown90s {detail[:80]}")
                if q429[0] > 3:
                    raise RuntimeError("429 persist x4")
                continue
            if e.code in (402,) or (e.code == 403 and "payment" in detail.lower()) or \
               (e.code == 403 and "quota" in detail.lower()):
                with open(STOP_FILE, "w") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + detail[:200])
                STOP.set()
                raise QuotaStop(detail[:120])
            if 500 <= e.code < 600 and attempt < len(delays) - 1:
                log(f"  {model}: HTTP{e.code} attempt {attempt} {detail[:120]}")
                continue
            if e.code == 400 and attempt < 3 and ("upstream_request_rejected" in detail
                                                  or "invalid character" in detail):
                log(f"  {model}: transient 400 attempt {attempt} {detail[:100]}")
                time.sleep(3)
                continue
            raise RuntimeError(f"HTTP{e.code}: {detail[:200]}")
        except urllib.error.URLError as e:
            if attempt < len(delays) - 1:
                continue
            raise RuntimeError(f"URLError: {e}")
        msg = (r.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content")
        if content is None:
            content = msg.get("reasoning_content") or ""
        usage = r.get("usage") or {}
        _record_usage(model, usage)
        finish = (r.get("choices") or [{}])[0].get("finish_reason")
        return {"content": content, "usage": usage, "latency": round(time.time() - t0, 2),
                "finish_reason": finish}


def judge_once_ia(model, user_msg, task_id):
    """Identical protocol to judge.judge_once: one parse, one reminder retry."""
    mt = MAXTOK[model]
    r = ia_chat(model, [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user_msg}], mt)
    v = parse_verdict(r["content"])
    if v is None:
        r2 = ia_chat(model, [{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": user_msg},
                              {"role": "assistant", "content": r["content"][-2000:]},
                              {"role": "user", "content":
                               "Your reply did not contain a valid verdict line. Reply again "
                               "and end with exactly [[A]], [[B]], or [[C]]."}], mt)
        v = parse_verdict(r2["content"])
        r = r2
    return v, r


def canon_verdict(v, order):
    if v == "A":
        return "A" if order == "AB" else "B"
    if v == "B":
        return "B" if order == "AB" else "A"
    if v == "C":
        return "TIE"
    return None


def build_rec(p, q, model, jname, jfam, order, rep, v, r, phase, extra=None):
    if order == "AB":
        ta, tb = p["text_a"], p["text_b"]
    else:
        ta, tb = p["text_b"], p["text_a"]
    rec = {
        "pair_id": p["pair_id"], "qid": p["qid"], "domain": p["domain"],
        "judge": jname, "judge_model": model, "judge_family": jfam,
        "order": order, "rep": rep, "platform": "internai", "e7_phase": phase,
        "gen_a": p["gen_a"], "gen_b": p["gen_b"],
        "gen_a_family": GEN_FAMILY[p["gen_a"]], "gen_b_family": GEN_FAMILY[p["gen_b"]],
        "verdict_raw": v, "verdict": canon_verdict(v, order),
        "parsed_ok": v is not None,
        "len_a": p.get("len_a"), "len_b": p.get("len_b"), "gt_better": p.get("gt_better"),
        "content_chars": len(r["content"]), "usage": r["usage"], "latency": r["latency"],
        "finish_reason": r.get("finish_reason"),
    }
    if extra:
        rec.update(extra)
    return rec


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                done.add((r["pair_id"], r["judge"], r["order"], r.get("rep", 0)))
            except Exception:
                pass
    return done


def run_tasks(tasks, worker, n_workers=6, desc=""):
    n_done, n_err = 0, 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(worker, t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                res = fu.result()
            except QuotaStop:
                log(f"QUOTA/STOP signal -- aborting {desc} ({i}/{len(tasks)})")
                for f in futs:
                    f.cancel()
                raise
            except Exception as e:
                n_err += 1
                log(f"  task error: {str(e)[:160]}")
                continue
            if res:
                n_done += 1
            if i % 100 == 0:
                log(f"  {desc}: {i}/{len(tasks)} done={n_done} err={n_err}")
    return n_done, n_err


def phase_check(pairs_by_id, questions, bailian):
    """4 overlap models x 48 stratified existing cells; returns GLM gate stats."""
    if os.path.exists(STOP_FILE):
        raise QuotaStop("stop file exists")
    done = load_done(OUT_CHECK)
    # frame: cells with valid bailian verdict, stratified 12/domain
    frame = {}
    by_model = {}
    for line in open(os.path.join(R, "judgments.jsonl")):
        rec = json.loads(line)
        if rec.get("verdict") not in ("A", "B", "TIE"):
            continue
        by_model.setdefault(rec["judge"], {})[(rec["pair_id"], rec["order"])] = rec
    tasks = []
    for k, (model, (jname, jfam)) in enumerate(OVERLAP.items()):
        cells = by_model.get(jname, {})
        rng = random.Random(20261017 + k)
        per_domain = {}
        for (pid, order), rec in cells.items():
            per_domain.setdefault(rec["domain"], []).append((pid, order, rec))
        picked = []
        for dom in sorted(per_domain):
            lst = sorted(per_domain[dom], key=lambda x: x[0])
            picked.extend(rng.sample(lst, min(12, len(lst))))
        log(f"check {jname}: {len(picked)} cells sampled from {len(cells)}")
        for pid, order, brec in picked:
            if (pid, jname, order, 0) in done:
                continue
            tasks.append({"p": pairs_by_id[pid], "q": questions[brec["qid"]], "model": model,
                          "jname": jname, "jfam": jfam, "order": order,
                          "bailian_verdict": brec["verdict"], "rep": 0})

    def worker(t):
        if STOP.is_set():
            raise QuotaStop("stop")
        p, q = t["p"], t["q"]
        ta, tb = (p["text_a"], p["text_b"]) if t["order"] == "AB" else (p["text_b"], p["text_a"])
        user_msg = make_user_msg(q["question"], ta, tb, p.get("ref"))
        v, r = judge_once_ia(t["model"], user_msg, f"check:{t['jname']}:{p['pair_id']}:{t['order']}")
        rec = build_rec(p, q, t["model"], t["jname"], t["jfam"], t["order"], 0, v, r,
                        "check", {"bailian_verdict": t["bailian_verdict"]})
        append(OUT_CHECK, rec)
        return rec

    run_tasks(tasks, worker, desc="check")
    # gate stats for GLM from whatever is now on disk (resumable-safe)
    glm_recs = [json.loads(l) for l in open(OUT_CHECK)
                if json.loads(l).get("judge") == "GLM-5.3"] if os.path.exists(OUT_CHECK) else []
    n = len(glm_recs)
    agree = sum(1 for g in glm_recs if g.get("verdict") == g.get("bailian_verdict"))
    gate = (n >= 40 and agree / n >= 0.85) if n else False
    log(f"GLM gate: {agree}/{n} raw agreement -> {'PASS' if gate else 'FAIL'}")
    return {"glm_check_n": n, "glm_agree": agree, "gate": gate}


def phase_fill(pairs, questions, pairs_by_id, gate):
    if os.path.exists(STOP_FILE):
        raise QuotaStop("stop file exists")
    # --- 126 missing cells ---
    filled = 0
    if gate:
        done = load_done(OUT_FILL)
        missing = json.load(open(os.path.join(R, "glm_missing.json")))
        tasks = [{"p": pairs_by_id[m["pair_id"]], "q": questions[pairs_by_id[m["pair_id"]]["qid"]],
                  "order": m["order"]} for m in missing
                 if (m["pair_id"], "GLM-5.3", m["order"], 0) not in done]
        log(f"fill GLM-5.3: {len(tasks)} missing cells")

        def worker_fill(t):
            if STOP.is_set():
                raise QuotaStop("stop")
            p, q, order = t["p"], t["q"], t["order"]
            ta, tb = (p["text_a"], p["text_b"]) if order == "AB" else (p["text_b"], p["text_a"])
            user_msg = make_user_msg(q["question"], ta, tb, p.get("ref"))
            v, r = judge_once_ia("glm-5.3", user_msg, f"fill:{p['pair_id']}:{order}")
            rec = build_rec(p, q, "glm-5.3", "GLM-5.3", "GLM", order, 0, v, r, "fill")
            append(OUT_FILL, rec)
            return rec

        filled, _ = run_tasks(tasks, worker_fill, desc="fill")
    else:
        log("fill skipped (gate FAIL)")
    # --- GLM retest: exact replication of judge.py --retest selection ---
    done_rt = load_done(OUT_RETEST)
    rng = random.Random(20261017)
    by_dom = {}
    for p in pairs:  # pairs.jsonl file order -- identical to original run
        by_dom.setdefault(p["domain"], []).append(p)
    selected = []
    for dom, lst in by_dom.items():
        lst2 = sorted(lst, key=lambda x: x["pair_id"])
        selected.extend(rng.sample(lst2, min(10, len(lst2))))
    tasks = []
    for p in selected:
        for rep in (0, 1):
            if (p["pair_id"], "GLM-5.3", "AB", rep) in done_rt:
                continue
            tasks.append({"p": p, "q": questions[p["qid"]], "rep": rep})
    log(f"GLM retest: {len(tasks)} cells (40 pairs x AB x 2 reps)")

    def worker_rt(t):
        if STOP.is_set():
            raise QuotaStop("stop")
        p, q = t["p"], t["q"]
        user_msg = make_user_msg(q["question"], p["text_a"], p["text_b"], p.get("ref"))
        v, r = judge_once_ia("glm-5.3", user_msg, f"retest:{p['pair_id']}:r{t['rep']}")
        rec = build_rec(p, q, "glm-5.3", "GLM-5.3", "GLM", "AB", t["rep"], v, r, "retest_glm")
        append(OUT_RETEST, rec)
        return rec

    n_rt, _ = run_tasks(tasks, worker_rt, desc="retest_glm")
    return {"filled": filled, "retest": n_rt}


def phase_anchor(pairs, questions):
    if os.path.exists(STOP_FILE):
        raise QuotaStop("stop file exists")
    done = load_done(OUT_ANCHOR)
    tasks = []
    for p in pairs:
        q = questions[p["qid"]]
        for order in ("AB", "BA"):
            for model, (jname, jfam) in ANCHORS.items():
                if (p["pair_id"], jname, order, 0) in done:
                    continue
                tasks.append({"p": p, "q": q, "model": model, "jname": jname,
                              "jfam": jfam, "order": order})
    random.Random(20261016).shuffle(tasks)  # same seeded shuffle as main run
    log(f"anchor tasks: {len(tasks)} (2 anchors x 400 pairs x 2 orders)")

    def worker(t):
        if STOP.is_set():
            raise QuotaStop("stop")
        p, q, order = t["p"], t["q"], t["order"]
        ta, tb = (p["text_a"], p["text_b"]) if order == "AB" else (p["text_b"], p["text_a"])
        user_msg = make_user_msg(q["question"], ta, tb, p.get("ref"))
        v, r = judge_once_ia(t["model"], user_msg, f"anchor:{t['jname']}:{p['pair_id']}:{order}")
        rec = build_rec(p, q, t["model"], t["jname"], t["jfam"], order, 0, v, r, "anchor")
        append(OUT_ANCHOR, rec)
        return rec

    n, _ = run_tasks(tasks, worker, n_workers=4, desc="anchor")
    return {"anchor_records": n}


def main():
    phase = "all"
    if "--phase" in sys.argv:
        phase = sys.argv[sys.argv.index("--phase") + 1]
    pairs = load_jsonl(os.path.join(R, "pairs.jsonl"))
    pairs_by_id = {p["pair_id"]: p for p in pairs}
    questions = {q["qid"]: q for q in load_jsonl(os.path.join(R, "questions.jsonl"))}
    report = {"phase": phase}
    if phase in ("all", "check"):
        log("=== phase 1: cross-platform spot check (gate for GLM fill) ===")
        report["check"] = phase_check(pairs_by_id, questions, None)
    if phase in ("all", "fill"):
        log("=== phase 2: GLM-5.3 gap fill + cross-platform retest ===")
        gate = report.get("check", {}).get("gate")
        if gate is None and os.path.exists(OUT_CHECK):
            n = agree = 0
            for l in open(OUT_CHECK):
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                if r.get("judge") == "GLM-5.3" and r.get("verdict") in ("A", "B", "TIE"):
                    n += 1
                    agree += (r["verdict"] == r.get("bailian_verdict"))
            gate = (n >= 40 and agree / n >= 0.85) if n else False
            report["check"] = {"glm_check_n": n, "glm_agree": agree, "gate": gate}
            log(f"gate recomputed from disk: {agree}/{n} -> {'PASS' if gate else 'FAIL'}")
        report["fill"] = phase_fill(pairs, questions, pairs_by_id, gate)
    if phase in ("all", "anchor"):
        log("=== phase 3: premium anchors (intern-s2, minimax-m3) ===")
        report["anchor"] = phase_anchor(pairs, questions)
    json.dump(report, open(os.path.join(R, "e7_run_report.json"), "w"), indent=1)
    log("report -> e7_run_report.json")
    log(json.dumps(report))


if __name__ == "__main__":
    main()
