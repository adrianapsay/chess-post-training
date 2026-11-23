| Model | Wins | Draws | Losses | Score (%) | Invalid moves (%) | Average length (plies) |
|---|---|---|---|---|---|---|
| base | 0.0 | 4.0 | 46.0 | 4.0 | 78.3 | 44.0 |
| sft_v1 | 0.0 | 5.0 | 45.0 | 5.0 | 60.5 | 47.0 |
| dpo_v1 | 0.0 | 2.0 | 48.0 | 2.0 | 71.4 | 50.0 |
| grpo_v1 | 0.0 | 4.0 | 46.0 | 4.0 | 60.3 | 42.0 |
| sft_stockfish | 0.0 | 5.0 | 45.0 | 5.0 | 58.3 | 42.0 |
| dpo_v2 | 0.0 | 5.0 | 45.0 | 5.0 | 65.4 | 46.0 |
| dpo_v3 | 0.0 | 4.0 | 46.0 | 4.0 | 61.4 | 49.0 |
| grpo_v2 | 0.0 | 5.0 | 45.0 | 5.0 | 62.4 | 46.0 |
| grpo_v3 | 0.0 | 2.0 | 48.0 | 2.0 | 60.3 | 40.0 |
| sft_100k | 0.0 | 7.0 | 43.0 | 7.0 | 42.4 | 49.0 |
| sft_100k_stockfish | 0.0 | 7.0 | 43.0 | 7.0 | 41.9 | 49.0 |
| base_legal_moves | 0.0 | 13.0 | 37.0 | 13.0 | 20.6 | 35.0 |
| sft_legal_moves | 0.0 | 11.0 | 39.0 | 11.0 | 0.0 | 41.0 |
| grpo_v4 | 0.0 | 3.0 | 47.0 | 3.0 | 58.2 | 44.0 |
| grpo_v5 | 0.0 | 6.0 | 44.0 | 6.0 | 59.7 | 59.0 |
| sft_100k_legal_moves | 0.0 | 19.0 | 31.0 | 19.0 | 0.0 | 48.0 |
| sft_legal_dpo | 0.0 | 16.0 | 34.0 | 16.0 | 0.0 | 41.0 |
| sft_legal_rl | 0.0 | 9.0 | 41.0 | 9.0 | 0.0 | 51.0 |
| sft_legal_dpo_rl | 0.0 | 14.0 | 36.0 | 14.0 | 0.0 | 53.0 |

50 games per model against Stockfish (depth 1), each opening once with each color. Invalid moves were replaced by a random legal move. Score = (wins + draws/2) / games.
