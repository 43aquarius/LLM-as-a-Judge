# Free and Fair? A Systematic Study of Reliability and Bias in Open-Source LLM Judges across Families and Scales

This repository contains the complete experimental artifact for a systematic
audit of **LLM-as-a-Judge reliability and bias** on freely accessible
(free-tier) large language models.

**Study design in one paragraph.** Ten open-source judges from four model
families (Qwen x5, DeepSeek x3, GLM x1, Kimi x1) plus two proprietary
free-tier reference judges (Agnes-2.5-Flash, Agnes-3.0-Flash) evaluate 400
stratified answer pairs built from 200 questions in four domains
(GSM8K mathematics, TruthfulQA knowledge, creative writing, arXiv abstract
explanation). Every judge scores every pair in **both presentation orders**
at temperature zero, plus a 40-pair test-retest subset. We measure verdict
compliance, ground-truth accuracy, position bias, length bias, family
(kin) preference with ground-truth control, inter-judge agreement (pairwise
Cohen's kappa), test-retest stability, and majority-panel behaviour, all with
question-level cluster bootstrap confidence intervals.

**Zero-cost guarantee.** The entire experiment runs on free API quotas
(1M tokens per model on the open judges' provider; $0-priced promotional
flash tier for the proprietary judges) with hard per-model token budgets
enforced in code (`code/llmclient.py`), so the full study can be reproduced
at zero monetary cost.

## Repository layout

```
data/                      # all experimental data (JSONL + JSON)
  questions.jsonl            200 questions x 4 domains
  answers.jsonl              1,000 candidate answers (5 generators x 200)
  pairs.jsonl                400 answer pairs (66 ground-truth-clear)
  judgments.jsonl            open-judge verdicts (10 judges x 400 pairs x 2 orders)
  judgments_agnes.jsonl      proprietary-judge verdicts
  retests.jsonl              test-retest subset, open judges
  retests_agnes.jsonl        test-retest subset, proprietary judges
  results.json               every reported metric + bootstrap CIs

code/                      # full pipeline (Python 3.10+)
  data_build.py              dataset construction (questions, prompts)
  generate.py                candidate-answer generation
  pairs.py                   stratified pair construction
  judge.py                   open-judge evaluation (resumable, budget-capped)
  agnes_judge.py             proprietary-judge evaluation
  llmclient.py               quota-managed OpenAI-compatible client
  analyze.py                 all statistics (kappa, bootstrap, regression, panels)
  figures.py                 publication figures (vector PDF)

paper/                     # LaTeX source (Elsevier elsarticle) + figures
```

## Reproducing

1. Set up two free API endpoints (one OpenAI-compatible provider for the
   open judges, one for the proprietary judges) and export their keys.
2. Run the pipeline in order:

```bash
export JUDGE_BASE_URL=...      JUDGE_API_KEY=...   # any OpenAI-compatible endpoint
export AGNES_API_KEY=...           # proprietary endpoint

python3 code/data_build.py         # builds data/questions.jsonl
python3 code/generate.py           # 5 generators x 200 questions
python3 code/pairs.py              # 400 stratified pairs
python3 code/judge.py              # main judging pass (~8,000 calls)
python3 code/judge.py --retest     # test-retest subset
python3 code/agnes_judge.py        # proprietary judges
python3 code/agnes_judge.py --retest
python3 code/analyze.py            # -> data/results.json
python3 code/figures.py            # -> paper/figures/*.pdf
```

Every stage is **resumable**: completed work is deduplicated from the output
JSONL files, so interrupted runs simply continue where they stopped. The
token-budget ledger in `llmclient.py` hard-stops any model that reaches its
free quota, which keeps the experiment inside the zero-cost envelope.

## Key findings

See `paper/main.pdf` for the full write-up. Headline numbers are also
available in machine-readable form in `data/results.json`.

## Citation

If you use this artifact, please cite the accompanying paper:

```bibtex
@article{tan2026freefair,
  title   = {Free and Fair? A Systematic Study of Reliability and Bias in
             Open-Source LLM Judges across Families and Scales},
  author  = {Tan, Chang},
  journal = {Engineering Applications of Artificial Intelligence
             (submission)},
  year    = {2026}
}
```

## License

Code: MIT. Data and prompts: CC-BY-4.0. See `LICENSE` and `DATA_LICENSE`.


## v1.1 (2026-10-17)
- Robustness battery E0-E8: prompt ablations, harder verifiable tasks (MATH/HumanEval/MMLU with deterministic parsers), no-reference deployment, cross-session retests -> data/judgments_e*.jsonl, data/e3_*.jsonl, data/stageC_results.json
- Unified reanalysis with three-accuracy conventions -> code/reanalysis.py, data/reanalysis_results.json, data/gtclear_audit.csv
- De-identified human-annotation kit (110 pairs + 10 calibration; unblind key withheld) -> annotation/


## v1.2 (2026-10-17, E7 third-platform supplement)
- Same-code cross-platform deployment check (4 overlap judges, 47-48 cells each): raw agreement 0.58-0.88 -> model identity does not port across platforms
- GLM-5.3 coverage restoration: 115/126 missing cells recovered on a compatible deployment (gate 42/48); tie rate revised 0.49->0.46; domain-skew MCAR correction measured directly -> data/e7_glm_fill.jsonl
- Cross-platform retest for GLM-5.3: 20/20 on double-repeat pairs -> data/e7_glm_retest.jsonl
- Premium external anchors Intern-S2 + MiniMax-M3 (families outside the roster): 194-cell pilot within the platform's daily free quota; resumable runner included (scripts/finish_e7.sh pattern; code/e7_run.py completes the grid automatically across quota windows) -> data/judgments_e7.jsonl, data/e7_results.json
