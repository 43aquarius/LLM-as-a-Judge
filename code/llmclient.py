#!/usr/bin/env python3
"""Unified LLM client for the judge-reliability experiment.

Features:
- OpenAI-compatible calls to Aliyun Bailian free-tier endpoint
- Per-model quota ledger (prompt+completion tokens), persisted to JSON
- Per-model config: thinking mode (direct / forced-thinking)
- Threaded worker pool, exponential backoff on 429/5xx, JSONL call log
- Resumable: every successful call appended to a JSONL file; caller skips done keys
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.environ.get("JUDGE_BASE_URL")
KEY = os.environ.get("JUDGE_API_KEY")

RESEARCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
USAGE_FILE = os.path.join(RESEARCH, "usage.json")
CALL_LOG = os.path.join(RESEARCH, "calls.jsonl")

# thinking mode: "direct" (we pass enable_thinking=False), "forced" (platform requires thinking)
THINKING_MODE = {
    "qwen3.8-flash": "direct",
    "qwen3.7-flash-2026-07-15": "direct",
    "qwen3.8-27b": "direct",
    "qwen3.8-max-0902": "direct",
    "qwen3.8-2.4t-a95b": "forced",
    "deepseek-v4-flash-0731": "direct",
    "deepseek-v4.1-flash": "direct",
    "deepseek-v4-pro-0813": "direct",
    "glm-5.3": "forced",
    "kimi-k3": "direct",
    "qwen3.7-flash": "direct",  # ALIAS model-code, own 1M free grant (console 2026-10-17:
    #                                508.21K remaining, expires 2026-10-22); same underlying
    #                                qwen3.7-flash snapshot family as the dated judge id.
    # --- FORBIDDEN models (NO free quota on Aliyun; any call BILLS the account) ---
    # "qwen3.7-max": ..., "glm-5.2": ..., "glm-5.3-prime": ...  (removed from judges)
}

# Hard per-model cumulative token budgets (circuit breaker).
# 2026-10-17 refresh: budgets = ledger usage (usage.json) + 95% of console-remaining
# free quota scraped live from the bailian console free-quota page on 2026-10-17.
# A model whose ledger usage reaches its budget is never scheduled again.
# glm-5.3 and qwen3.8-2.4t-a95b confirmed 403 FreeTierOnly by live probe -> budget 0.
# Paid-only models (qwen3.7-max/glm-5.2/glm-5.3-prime) stay absent -> budget 0;
# E7 must use its own client with an explicit paid cap, never this ledger.
MODEL_BUDGETS = {
    "qwen3.7-flash-2026-07-15": 981485,   # 590028 + 0.95*412060; free quota expires 2026-10-22
    "qwen3.8-flash": 761907,              # 661843 + 0.95*105330
    "qwen3.8-27b": 760844,                # 619731 + 0.95*148540
    "qwen3.8-max-0902": 983984,           # 640122 + 0.95*361960
    "qwen3.8-2.4t-a95b": 0,               # exhausted (403 FreeTierOnly 2026-10-17 probe)
    "deepseek-v4-flash-0731": 801743,     # 530499 + 0.95*285520; expires 2026-10-30
    "deepseek-v4.1-flash": 783600,        # 533930 + 0.95*262810
    "deepseek-v4-pro-0813": 838837,       # 587742 + 0.95*264310
    "glm-5.3": 0,                         # exhausted (403 FreeTierOnly)
    "kimi-k3": 839509,                    # 632941 + 0.95*217440
    "qwen3.7-flash": 482800,               # alias model-code: 0 used + 0.95*508210 console grant
}

_usage_lock = threading.Lock()
_log_lock = threading.Lock()


def _load_usage() -> dict:
    if os.path.exists(USAGE_FILE):
        with open(USAGE_FILE) as f:
            return json.load(f)
    return {}


def get_usage(model: str) -> int:
    with _usage_lock:
        return _load_usage().get(model, 0)


def over_cap(model: str, cap: int | None = None) -> bool:
    budget = MODEL_BUDGETS.get(model, 0)  # unknown model -> budget 0 -> blocked
    return get_usage(model) >= budget


def _record_usage(model: str, tokens: int):
    with _usage_lock:
        u = _load_usage()
        u[model] = u.get(model, 0) + tokens
        tmp = USAGE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(u, f, indent=1)
        os.replace(tmp, USAGE_FILE)


def _log_call(record: dict):
    with _log_lock:
        with open(CALL_LOG, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class QuotaExceeded(Exception):
    pass


def _kill_all_budgets(reason: str):
    """Cost-safety circuit breaker: zero every model budget and persist it.
    Used when the platform signals billing trouble (arrears / unpurchased)."""
    with _usage_lock:
        u = _load_usage()
        for m in list(MODEL_BUDGETS) + [k for k in u if k not in MODEL_BUDGETS]:
            MODEL_BUDGETS[m] = 0
        with open(USAGE_FILE + ".budgets", "w") as f:
            json.dump({"killed": True, "reason": reason, "ts": time.time()}, f)


def chat(
    model: str,
    messages: list,
    max_tokens: int = 512,
    temperature: float = 0.0,
    task_id: str = "",
    max_retries: int = 4,
    cap: int | None = None,
) -> dict:
    """One chat completion. Returns {content, usage:{...}, latency, model}."""
    if over_cap(model, cap):
        raise QuotaExceeded(
            f"{model} over hard budget {MODEL_BUDGETS.get(model, 0)} "
            f"(usage {get_usage(model)}; non-free models are budget 0)"
        )
    mode = THINKING_MODE.get(model, "direct")
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if mode == "direct":
        body["enable_thinking"] = False
    last_err = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            f"{BASE}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                d = json.loads(resp.read())
            msg = (d.get("choices") or [{}])[0].get("message", {})
            content = (msg.get("content") or "").strip()
            usage = d.get("usage", {})
            snapshot = d.get("model") or model  # platform-reported snapshot id, per-call version record
            total = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
            _record_usage(model, total)
            _log_call({
                "ts": time.time(), "task_id": task_id, "model": model, "snapshot": snapshot,
                "n_messages": len(messages),
                "content_chars": len(content),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "latency": round(time.time() - t0, 2),
            })
            return {
                "content": content,
                "reasoning": (msg.get("reasoning_content") or "")[:4000],
                "usage": usage,
                "latency": round(time.time() - t0, 2),
                "model": model,
                "snapshot": snapshot,
            }
        except urllib.error.HTTPError as e:
            try:
                err = e.read().decode()[:300]
            except Exception:
                err = str(e)
            last_err = f"HTTP {e.code}: {err}"
            if e.code == 403 and ("Free quota exhausted" in err or "AllocationQuota.FreeTierOnly" in err):
                # free tier exhausted for this model -> stop scheduling it (no billing risk)
                with _usage_lock:
                    u = _load_usage()
                    u[model] = max(u.get(model, 0), MODEL_BUDGETS.get(model, 0) + 1)
                    tmp = USAGE_FILE + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(u, f, indent=1)
                    os.replace(tmp, USAGE_FILE)
                MODEL_BUDGETS[model] = 0
                raise QuotaExceeded(f"{model}: free quota exhausted")
            if e.code == 403 and ("AccessDenied.Unpurchased" in err or "Arrears" in err):
                # account in arrears / model unpurchased -> FATAL: kill all budgets, stop everything
                _kill_all_budgets(last_err)
                raise QuotaExceeded(f"FATAL account issue, stopping all calls: {last_err[:120]}")
            if e.code in (429, 500, 502, 503, 504):
                sleep_s = min(2 ** attempt * 2 + random.random(), 30)
                time.sleep(sleep_s)
                continue
            if e.code == 400:
                # retry once without thinking control param (compat fallback)
                if "enable_thinking" in body:
                    body.pop("enable_thinking")
                    continue
                raise RuntimeError(last_err)
            raise RuntimeError(last_err)
        except Exception as e:  # network hiccup
            last_err = str(e)[:200]
            time.sleep(min(2 ** attempt * 2, 20))
    raise RuntimeError(f"{model} failed after {max_retries} retries: {last_err}")


def run_parallel(tasks, worker_fn, n_workers=5, desc=""):
    """tasks: list of dicts; worker_fn(task) -> result dict. Skips task if raises QuotaExceeded.
    Returns (results, errors)."""
    results, errors = [], []
    done = 0
    total = len(tasks)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(worker_fn, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r = fut.result()
                if r is not None:
                    results.append(r)
            except QuotaExceeded:
                errors.append({"task": t, "error": "quota"})
            except Exception as e:
                errors.append({"task": t, "error": str(e)[:200]})
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  [{desc}] {done}/{total} done, {len(errors)} errors", flush=True)
    return results, errors
