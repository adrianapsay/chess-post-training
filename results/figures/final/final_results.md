| Model | Method | What changed | Puzzle accuracy (%) | 95% interval | Legal (%) | ACPL (cp) | Positions: same as Stockfish (%) | Positions: blunders among legal (%) | Score vs. random (%) | Score vs. Stockfish (%) | Invalid moves in games (%) | Training tokens (M) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | base | no training | 2.8 | 1.9-4.0 | 41.2 | 828 | 2.7 | 47.0 | 50.0 | 4.0 | 77.0 | 0.0 |
| base_legal_moves | base | legal moves listed in the prompt | 6.0 | 4.7-7.6 | 84.3 | 791 | 8.2 | 55.7 | 53.0 | 13.0 | 22.3 | 0.0 |
| sft_v1 | SFT | 20k positions, the player's move | 7.8 | 6.3-9.6 | 72.2 | 793 | 6.5 | 54.9 | 51.0 | 5.0 | 44.4 | 1.9 |
| sft_stockfish | SFT | 20k positions, Stockfish's move | 7.3 | 5.8-9.1 | 71.5 | 799 | 6.9 | 57.8 | 52.0 | 5.0 | 41.4 | 1.9 |
| sft_100k | SFT | 100k positions, the player's move | 9.4 | 7.7-11.4 | 78.6 | 777 | 10.1 | 49.8 | 55.0 | 7.0 | 28.2 | 9.5 |
| sft_100k_stockfish | SFT | 100k positions, Stockfish's move | 9.2 | 7.6-11.2 | 80.9 | 766 | 9.6 | 51.9 | 50.0 | 7.0 | 25.5 | 9.5 |
| sft_legal_moves | SFT | 20k positions, legal moves listed in the prompt | 13.2 | 11.2-15.4 | 100.0 | 761 | 10.8 | 56.1 | 53.0 | 11.0 | 0.0 | 4.53 |
| dpo_v1 | DPO from sft_v1 | rejected = a random move losing 200+ cp | 5.9 | 4.6-7.5 | 63.3 | 820 | 5.2 | 55.0 | 52.0 | 2.0 | 52.2 | 2.77 |
| dpo_v2 | DPO from sft_v1 | rejected = the model's own mistakes | 4.4 | 3.3-5.9 | 61.6 | 793 | 4.2 | 51.7 | 51.0 | 5.0 | 59.3 | 2.77 |
| dpo_v3 | DPO from sft_v1 | own mistakes + SFT loss | 6.7 | 5.3-8.4 | 72.9 | 806 | 6.1 | 54.6 | 51.0 | 4.0 | 43.6 | 2.77 |
| grpo_v1 | RL from sft_v1 | no KL penalty | 3.3 | 2.4-4.6 | 81.6 | 836 | 2.5 | 47.5 | 51.0 | 4.0 | 52.4 | 3.41 |
| grpo_v2 | RL from sft_v1 | KL penalty 0.04 | 4.9 | 3.7-6.4 | 80.6 | 818 | 3.2 | 49.1 | 50.0 | 5.0 | 47.9 | 3.41 |
| grpo_v3 | RL from sft_v1 | KL penalty 0.04, reward without legality bonus | 4.2 | 3.1-5.6 | 79.7 | 828 | 3.5 | 46.8 | 50.0 | 2.0 | 46.6 | 3.41 |
| grpo_v4 | RL from sft_v1 | KL penalty 0.1 | 4.8 | 3.6-6.3 | 80.2 | 825 | 4.1 | 49.2 | 51.0 | 3.0 | 42.5 | 3.41 |
| grpo_v5 | RL from sft_v1 | KL penalty 0.2 | 5.0 | 3.8-6.5 | 78.3 | 822 | 5.1 | 49.7 | 52.0 | 6.0 | 43.7 | 3.41 |

Puzzles: 1,000 test puzzles (a random legal move solves 5.6%); ACPL over legal answers. Positions: 1,000 held-out game positions. Games: 50 per opponent; score = (wins + draws/2) / games. Training tokens: summed over every training stage behind the model. A dash = not evaluated yet.
