| Model | Wins | Draws | Losses | Score (%) | Invalid moves (%) | Average length (plies) |
|---|---|---|---|---|---|---|
| base | 2.0 | 46.0 | 2.0 | 50.0 | 76.7 | 188.0 |
| sft_v1 | 2.0 | 47.0 | 1.0 | 51.0 | 39.6 | 156.0 |
| dpo_v1 | 2.0 | 48.0 | 0.0 | 52.0 | 46.5 | 166.0 |
| grpo_v1 | 1.0 | 49.0 | 0.0 | 51.0 | 50.5 | 170.0 |
| sft_stockfish | 2.0 | 48.0 | 0.0 | 52.0 | 36.9 | 156.0 |
| dpo_v2 | 4.0 | 43.0 | 3.0 | 51.0 | 57.7 | 173.0 |
| dpo_v3 | 4.0 | 43.0 | 3.0 | 51.0 | 38.3 | 161.0 |
| grpo_v2 | 3.0 | 44.0 | 3.0 | 50.0 | 43.8 | 164.0 |
| grpo_v3 | 2.0 | 46.0 | 2.0 | 50.0 | 43.1 | 154.0 |
| sft_100k | 5.0 | 45.0 | 0.0 | 55.0 | 23.2 | 138.0 |
| sft_100k_stockfish | 0.0 | 50.0 | 0.0 | 50.0 | 19.7 | 139.0 |
| base_legal_moves | 4.0 | 45.0 | 1.0 | 53.0 | 22.7 | 162.0 |
| sft_legal_moves | 3.0 | 47.0 | 0.0 | 53.0 | 0.0 | 131.0 |
| grpo_v4 | 3.0 | 45.0 | 2.0 | 51.0 | 38.1 | 157.0 |
| grpo_v5 | 4.0 | 44.0 | 2.0 | 52.0 | 38.1 | 170.0 |
| sft_100k_legal_moves | 4.0 | 46.0 | 0.0 | 54.0 | 0.0 | 93.0 |
| sft_legal_dpo | 1.0 | 49.0 | 0.0 | 51.0 | 0.0 | 138.0 |
| sft_legal_rl | 10.0 | 40.0 | 0.0 | 60.0 | 0.0 | 110.0 |
| sft_legal_dpo_rl | 8.0 | 42.0 | 0.0 | 58.0 | 0.0 | 120.0 |

50 games per model against a random mover, each opening once with each color. Invalid moves were replaced by a random legal move. Score = (wins + draws/2) / games.
