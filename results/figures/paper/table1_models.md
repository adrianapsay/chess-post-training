| Model | Change | Acc. | 95% CI | Legal | Best move (%) | ACPL (cp) | Blunders (%) | Tokens (M) |
|---|---|---|---|---|---|---|---|---|
| random legal move | reference | 5.6 | – | 100 | 5.3 | 391 | 56.3 | – |
| base | no training | 2.8 | 1.9–4.0 | 41 | 2.7 | 344* | 47.0* | 0.0 |
| base_legal_moves† | no training | 6.0 | 4.7–7.6 | 84 | 8.2 | 381 | 55.7 | 0.0 |
| sft_v1 | 20k, player's moves | 7.8 | 6.3–9.6 | 72 | 6.5 | 392 | 54.9 | 1.9 |
| sft_stockfish | 20k, Stockfish's moves | 7.3 | 5.8–9.1 | 72 | 6.9 | 414 | 57.8 | 1.9 |
| sft_100k | 100k, player's moves | 9.4 | 7.7–11.4 | 79 | 10.1 | 348 | 49.8 | 9.5 |
| sft_100k_stockfish | 100k, Stockfish's moves | 9.2 | 7.6–11.2 | 81 | 9.6 | 370 | 51.9 | 9.5 |
| sft_legal_moves† | 20k, player's moves | 13.2 | 11.2–15.4 | 100 | 10.8 | 389 | 56.1 | 4.5 |
| sft_100k_legal_moves† | 100k, player's moves | 17.5 | 15.3–20.0 | 100 | 15.1 | 344 | 49.5 | 22.6 |
| dpo_v1 | random blunders rejected | 5.9 | 4.6–7.5 | 63 | 5.2 | 397 | 55.0 | 2.8 |
| dpo_v2 | own mistakes rejected | 4.4 | 3.3–5.9 | 62 | 4.2 | 346 | 51.7 | 2.8 |
| dpo_v3 | own mistakes + SFT loss | 6.7 | 5.3–8.4 | 73 | 6.1 | 389 | 54.6 | 2.8 |
| grpo_v1 | β = 0 | 3.3 | 2.4–4.6 | 82 | 2.5 | 322 | 47.5 | 3.4 |
| grpo_v2 | β = 0.04 | 4.9 | 3.7–6.4 | 81 | 3.2 | 336 | 49.1 | 3.4 |
| grpo_v3 | β = 0.04, new reward | 4.2 | 3.1–5.6 | 80 | 3.5 | 335 | 46.8 | 3.4 |
| grpo_v4 | β = 0.1 | 4.8 | 3.6–6.3 | 80 | 4.1 | 342 | 49.2 | 3.4 |
| grpo_v5 | β = 0.2 | 5.0 | 3.8–6.5 | 78 | 5.1 | 335 | 49.7 | 3.4 |
| sft_legal_dpo† | DPO as dpo_v3 | 12.8 | 10.9–15.0 | 100 | 8.9 | 413 | 58.1 | 6.7 |
| sft_legal_rl† | RL as grpo_v5 | 10.8 | 9.0–12.9 | 100 | 12.0 | 314 | 45.3 | 8.1 |
| sft_legal_dpo_rl† | DPO, then RL | 10.9 | 9.1–13.0 | 100 | 12.6 | 301 | 43.9 | 10.3 |

Table 1 of the paper (the LaTeX version is table1_models.tex). Puzzles: accuracy (Acc.), its 95% interval and legal answers, in %. Held-out positions: same move as Stockfish (Best move), and ACPL and blunders of the legal answers.
