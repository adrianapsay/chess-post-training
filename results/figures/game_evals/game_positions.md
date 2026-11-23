| Model | Legal (%) | ACPL (cp) | Blunders (%) | Same as Stockfish (%) | Same as the player (%) | Illegal (%) | Unparseable (%) |
|---|---|---|---|---|---|---|---|
| base | 40.4 | 344.0 | 19.0 | 2.7 | 2.6 | 47.9 | 11.7 |
| sft_v1 | 64.7 | 392.0 | 35.5 | 6.5 | 6.8 | 35.3 | 0.0 |
| dpo_v1 | 55.4 | 397.0 | 30.5 | 5.2 | 5.4 | 44.5 | 0.1 |
| grpo_v1 | 81.5 | 322.0 | 38.7 | 2.5 | 3.7 | 18.5 | 0.0 |
| sft_stockfish | 65.2 | 414.0 | 37.7 | 6.9 | 5.9 | 34.8 | 0.0 |
| dpo_v2 | 57.8 | 346.0 | 29.9 | 4.2 | 4.1 | 42.2 | 0.0 |
| dpo_v3 | 65.9 | 389.0 | 36.0 | 6.1 | 6.6 | 34.1 | 0.0 |
| grpo_v2 | 78.2 | 336.0 | 38.4 | 3.2 | 4.5 | 21.8 | 0.0 |
| grpo_v3 | 78.6 | 335.0 | 36.8 | 3.5 | 4.6 | 21.4 | 0.0 |
| random legal move | 100.0 | 391.0 | 56.3 | 5.3 | 6.5 | 0.0 | 0.0 |

1,000 held-out positions from games between 2000+ players. Blunder = a legal answer losing 200+ centipawns; its rate is over all answers. ACPL is over legal answers only.
