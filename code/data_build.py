#!/usr/bin/env python3
"""Build the 200-question evaluation set across 4 domains.

Domains:
  math       - GSM8K test problems (GT available)   [raw.githubusercontent.com]
  knowledge  - TruthfulQA questions (GT available)  [via ghfast.top proxy]
  creative   - 50 fixed open-ended writing prompts  [authored, shipped with code]
  summ       - 50 arXiv abstract explanation tasks  [export.arxiv.org API]

Output: ../data/questions.jsonl
"""
import json
import os
import random
import re
import time
import urllib.request

RESEARCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT = os.path.join(RESEARCH, "questions.jsonl")
CACHE = os.path.join(RESEARCH, "cache")
os.makedirs(CACHE, exist_ok=True)

SEED = 20261016
N_PER_DOMAIN = 50
UA = {"User-Agent": "Mozilla/5.0 (research; judge-eval-study)"}


def fetch(url, cache_name, timeout=60, retries=3):
    path = os.path.join(CACHE, cache_name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if len(data) < 50:
                raise ValueError("empty")
            with open(path, "wb") as f:
                f.write(data)
            return data
        except Exception as e:
            last = e
            time.sleep(2 * (a + 1))
    raise RuntimeError(f"fetch failed {url}: {last}")


def build_math(rng):
    raw = fetch(
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        "gsm8k_test.jsonl",
    ).decode()
    items = [json.loads(l) for l in raw.strip().split("\n") if l.strip()]
    # prefer problems with a clean numeric final answer, moderate length
    cands = [it for it in items if 80 < len(it["question"]) < 500]
    sel = rng.sample(cands, N_PER_DOMAIN)
    out = []
    for i, it in enumerate(sel):
        ans = it["answer"]
        final = ans.split("####")[-1].strip().replace(",", "")
        rationale = re.sub(r"<<[^>]*>>", "", ans.split("####")[0]).strip()
        out.append({
            "domain": "math", "qid": f"math-{i:02d}", "question": it["question"].strip(),
            "reference": final, "reference_rationale": rationale,
        })
    return out


def build_knowledge(rng):
    import csv, io
    raw = fetch(
        "https://ghfast.top/https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
        "truthfulqa.csv",
    ).decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(raw))
    rows = list(reader)
    # keep short factual questions with a clean best answer
    cands = []
    for r in rows:
        q = (r.get("Question") or "").strip()
        best = (r.get("Best Answer") or "").strip()
        if 25 < len(q) < 200 and 2 < len(best) < 120:
            cands.append({"q": q, "best": best})
    sel = rng.sample(cands, N_PER_DOMAIN)
    out = []
    for i, it in enumerate(sel):
        out.append({
            "domain": "knowledge", "qid": f"know-{i:02d}", "question": it["q"],
            "reference": it["best"],
        })
    return out


CREATIVE_PROMPTS = [
    "Write a six-sentence science fiction story about a lighthouse keeper on a comet mining station who discovers a signal in the static.",
    "Write a short poem (8-12 lines) about the sounds of a city at 4 a.m.",
    "Draft a polite but firm email to a neighbor whose dog barks loudly every night, proposing a solution.",
    "Write a fun 100-word origin story for why the octopus has eight arms.",
    "Compose a persuasive paragraph (120 words) arguing that libraries are more important than ever in the digital age.",
    "Write the opening two paragraphs of a mystery novel set in a small coastal town during a power outage.",
    "Create a catchy slogan and a one-paragraph pitch for a startup that rents houseplants to offices.",
    "Write a letter from a 90-year-old grandmother to her 20-year-old self, sharing three life lessons.",
    "Describe a bustling night market so vividly that a reader can smell it, in exactly five sentences.",
    "Write a short fable about a smartphone that becomes jealous of a paper notebook.",
    "Draft a toast (about 120 words) for a friend's 30th birthday dinner that is funny but heartfelt.",
    "Write a haiku about debugging code at midnight, then a short paragraph explaining what the haiku means to you.",
    "Imagine you are a travel guide robot in the year 2200. Write a 150-word pitch for a tour of Earth's abandoned cities.",
    "Write a dialogue (10-14 lines) between a raindrop and a leaf just before a storm.",
    "Compose a product review (100 words) for an imaginary umbrella that changes color with the weather.",
    "Write a motivational speech of about 150 words for a team that has lost every game this season.",
    "Pen a break-up letter to procrastination, written by a student during finals week.",
    "Write a 150-word scene where two strangers are stuck in an elevator and discover they share an unusual hobby.",
    "Craft a bedtime story about a firefly who is afraid of the dark, for a 5-year-old audience.",
    "Write a eulogy (150 words) for a beloved old bicycle.",
    "Create a recipe with a whimsical backstory for 'storm soup', a comfort dish for rainy days.",
    "Write a journal entry from the perspective of a museum night guard who suspects the exhibits rearrange themselves.",
    "Draft a short press release announcing that a cat has been elected mayor of a small town.",
    "Write a persuasive 150-word paragraph on why schools should teach cooking as a core subject.",
    "Compose song lyrics (two verses and a chorus) about moving to a new city and feeling at home by the end.",
    "Write a short story in exactly six sentences where the last sentence changes the meaning of everything before it.",
    "Write a witty warning label for a time machine, plus one paragraph of safety instructions.",
    "Describe the four seasons as if they were four siblings at a family reunion.",
    "Write a monologue for a Shakespearean character who has just discovered the internet.",
    "Draft a 150-word apology letter from a coffee machine to its office, for all the bad coffee.",
    "Write an encouraging note from a mountain to a nervous first-time hiker.",
    "Create a short myth explaining why the moon changes shape, featuring a shy seamstress.",
    "Write a paragraph (120 words) of pure sensory description of baking bread, without naming the bread.",
    "Compose an out-of-office reply from a wizard taking a sabbatical to find a missing dragon.",
    "Write a five-sentence story where the hero is a stapler.",
    "Draft the text for a museum plaque describing an ordinary pencil as a 3,000-year-old artifact, with deadpan humor.",
    "Write a conversation between Sunday and Monday about who has the harder job.",
    "Compose a limerick about a coder who only writes bugs, then a two-line apology from the coder.",
    "Write a 150-word advertisement for a hotel at the bottom of the ocean.",
    "Pen a short thank-you note to gravity, from someone who nearly floated away.",
    "Write the final paragraph of a novel about the last library on Earth, leaving hope for a sequel.",
    "Describe colors to someone who has never seen them, using only sound, taste, and touch metaphors.",
    "Write a humorous list of five rules for surviving a family road trip, each with a one-sentence explanation.",
    "Craft a short legend about a bridge that only appears to people who are lost.",
    "Write a 150-word pitch to a city council for a park designed entirely for stargazing.",
    "Compose a telegram-style message (under 40 words) from a Mars colonist to Earth on the colony's first anniversary.",
    "Write a soliloquy for a traffic light contemplating its purpose at 3 a.m.",
    "Draft an acceptance speech (120 words) for the 'World's Okayest Gardener' award.",
    "Write a short ghost story set in a laundromat, suitable for a middle-grade audience.",
    "Write a reflective 150-word piece about what a river might say if it could keep a diary for one year.",
]

ARXIV_QUERIES = [
    ("cs.AI", 0), ("cs.CL", 0), ("cs.LG", 0), ("cs.CV", 0), ("cs.RO", 0),
    ("cs.NE", 0), ("cs.IR", 0), ("cs.HC", 0), ("cs.SY", 0), ("stat.ML", 0),
    ("cs.AI", 50), ("cs.CL", 50), ("cs.LG", 50), ("cs.CV", 50), ("stat.ML", 50),
    ("cs.AI", 100), ("cs.CL", 100), ("cs.LG", 100), ("cs.CV", 100), ("cs.RO", 50),
]


def build_summ(rng):
    out = []
    per_query = {}
    for i, (cat, start) in enumerate(ARXIV_QUERIES):
        url = (
            f"https://export.arxiv.org/api/query?search_query=cat:{cat}"
            f"&start={start}&max_results=25&sortBy=relevance"
        )
        try:
            xml = fetch(url, f"arxiv_{cat}_{start}.xml").decode("utf-8", errors="ignore")
        except Exception:
            continue
        # parse entries
        entries = re.findall(r"<entry>(.*?)</entry>", xml, re.S)
        for e in entries:
            m_abs = re.search(r"<summary>(.*?)</summary>", e, re.S)
            m_title = re.search(r"<title>(.*?)</title>", e, re.S)
            if not m_abs or not m_title:
                continue
            abstract = re.sub(r"\s+", " ", m_abs.group(1)).strip()
            title = re.sub(r"\s+", " ", m_title.group(1)).strip()
            if len(abstract) < 450 or len(abstract) > 2200:
                continue
            # up to 4 per query (variety of fields)
            key = cat + str(start)
            lst = per_query.setdefault(key, [])
            if len(lst) < 4:
                lst.append({"title": title, "abstract": abstract})
        time.sleep(0.2)
    picked = [v for vs in per_query.values() for v in vs]
    rng.shuffle(picked)
    for i, it in enumerate(picked[:N_PER_DOMAIN]):
        q = (
            "Explain the following research paper abstract to a general audience "
            "in about three plain-language sentences.\n\nTitle: " + it["title"]
            + "\n\nAbstract:\n" + it["abstract"]
        )
        out.append({
            "domain": "summ", "qid": f"summ-{i:02d}", "question": q,
            "reference": None,
        })
    return out


if __name__ == "__main__":
    rng = random.Random(SEED)
    questions = []

    math_q = build_math(rng)
    print(f"math: {len(math_q)}", flush=True)
    know_q = build_knowledge(rng)
    print(f"knowledge: {len(know_q)}", flush=True)
    creat_q = [{"domain": "creative", "qid": f"creat-{i:02d}", "question": p, "reference": None}
               for i, p in enumerate(CREATIVE_PROMPTS)]
    print(f"creative: {len(creat_q)}", flush=True)
    summ_q = build_summ(rng)
    print(f"summ: {len(summ_q)}", flush=True)

    questions = math_q + know_q + creat_q + summ_q
    with open(OUT, "w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"total {len(questions)} -> {OUT}")
    print(Counter(q["domain"] for q in questions))
