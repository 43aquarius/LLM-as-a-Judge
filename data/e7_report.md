# E7 analysis (platform replication / GLM fill / retest / anchor pilot)
_generated 2026-10-17 13:53:06_

## 1. Cross-platform spot check (same model id, same prompts, temp 0)
- **DeepSeek-V4-Flash**: raw agreement 28/48 = 0.583 [0.443,0.712], kappa 0.379, tie% IA 0.333 vs Bailian 0.396, latency 6.8s
- **DeepSeek-V4-Pro**: raw agreement 34/48 = 0.708 [0.568,0.818], kappa 0.565, tie% IA 0.375 vs Bailian 0.417, latency 5.9s
- **GLM-5.3**: raw agreement 42/48 = 0.875 [0.753,0.941], kappa 0.811, tie% IA 0.333 vs Bailian 0.312, latency 8.1s
- **Qwen3.8-27B**: raw agreement 34/47 = 0.723 [0.582,0.831], kappa 0.565, tie% IA 0.191 vs Bailian 0.191, latency 4.8s

## 2. Determinism (identical call x3, temp 0, high caps)
- 13/15 probe cells produced identical verdicts; exceptions: GLM know-12 (truncation at 1200 cap), Qwen3.8-27B know-12 (B/C wobble, one length-cut generation).

## 3. GLM-5.3 gap fill + MCAR verification
- coverage: 674/800 original + 115 filled (of 126 missing) = 789/800 (98.6%)
- original cells: tie 331/674 (0.491); filled cells: tie 34/115 (0.296); 2-prop p=0.000
- GT-clear acc: original 0.608 (n=130), filled 0.500 (n=2); p=0.757
- A-share: original 0.255, filled 0.330
- filled domain mix: {'math': 6, 'knowledge': 10, 'creative': 37, 'summ': 62}

## 3b. GLM-5.3 merged view (789 cells): revised headline metrics
- cells 789, double-covered pairs 392, order-swap flips 62/392 = 0.158
- tie rate (3-caliber raw): 365/789 = 0.463 (frozen v1 value on 674: 0.491)
- GT-clear acc: 0.606 (n=132)
- kappa vs other judges: range 0.425-0.656, median 0.590 (n per judge 773-789)

## 4. GLM-5.3 retest on the third platform
- pairs with both reps parsed: 20; rep0-vs-rep1 agreement 20/20 = 1.000
- vs main canonical verdicts: 20/20 = 1.000
- other judges' intra-session retest agreement (context): DeepSeek-V4-Flash 36/40=0.90; DeepSeek-V4-Pro 30/40=0.75; DeepSeek-V4.1-Flash 33/40=0.82; Kimi-K3 36/40=0.90; Qwen3.7-Flash 34/40=0.85; Qwen3.8-27B 39/40=0.97; Qwen3.8-Flash 34/40=0.85; Qwen3.8-Max 38/40=0.95

## 5. Premium anchors (pilot; full grid blocked by daily token quota)
- **Intern-S2**: 91 parsed cells (100.0% compliance), 6 double-covered pairs, order-swap flips 0/6 = 0.000, tie% 0.484, GT-clear acc 0.636 (n=11), longer-wins 0.191 (n=47)
- **MiniMax-M3**: 103 parsed cells (100.0% compliance), 11 double-covered pairs, order-swap flips 2/11 = 0.182, tie% 0.544, GT-clear acc 0.467 (n=15), longer-wins 0.447 (n=47)

### 5a. Anchor vs judge kappa (common cells)
- Intern-S2 vs Qwen3.7-Flash: kappa 0.611 (raw 0.736, n=91)
- Intern-S2 vs Qwen3.8-27B: kappa 0.688 (raw 0.802, n=91)
- Intern-S2 vs Qwen3.8-Max: kappa 0.677 (raw 0.791, n=91)
- Intern-S2 vs Qwen3.8-Flash: kappa 0.809 (raw 0.879, n=91)
- Intern-S2 vs DeepSeek-V4.1-Flash: kappa 0.726 (raw 0.835, n=91)
- Intern-S2 vs DeepSeek-V4-Flash: kappa 0.465 (raw 0.681, n=91)
- Intern-S2 vs DeepSeek-V4-Pro: kappa 0.486 (raw 0.703, n=91)
- Intern-S2 vs Qwen3.8-2.4T: kappa 0.836 (raw 0.899, n=89)
- Intern-S2 vs GLM-5.3: kappa 0.781 (raw 0.871, n=85)
- Intern-S2 vs Kimi-K3: kappa 0.771 (raw 0.857, n=91)
- Intern-S2 vs Agnes-2.5-Flash: kappa 0.767 (raw 0.857, n=91)
- Intern-S2 vs Agnes-3.0-Flash: kappa 0.697 (raw 0.802, n=91)
- MiniMax-M3 vs Qwen3.7-Flash: kappa 0.368 (raw 0.544, n=103)
- MiniMax-M3 vs Qwen3.8-27B: kappa 0.482 (raw 0.641, n=103)
- MiniMax-M3 vs Qwen3.8-Max: kappa 0.558 (raw 0.699, n=103)
- MiniMax-M3 vs Qwen3.8-Flash: kappa 0.517 (raw 0.670, n=103)
- MiniMax-M3 vs DeepSeek-V4.1-Flash: kappa 0.599 (raw 0.757, n=103)
- MiniMax-M3 vs DeepSeek-V4-Flash: kappa 0.571 (raw 0.748, n=103)
- MiniMax-M3 vs DeepSeek-V4-Pro: kappa 0.514 (raw 0.728, n=103)
- MiniMax-M3 vs Qwen3.8-2.4T: kappa 0.513 (raw 0.670, n=100)
- MiniMax-M3 vs GLM-5.3: kappa 0.691 (raw 0.818, n=88)
- MiniMax-M3 vs Kimi-K3: kappa 0.565 (raw 0.728, n=103)
- MiniMax-M3 vs Agnes-2.5-Flash: kappa 0.653 (raw 0.777, n=103)
- MiniMax-M3 vs Agnes-3.0-Flash: kappa 0.470 (raw 0.631, n=103)

## 6. Pending (blocked by ~1M/day platform token quota)
- anchor cells 1406 (~1195100 tokens), GLM fill 10, GLM retest 37, check 1
