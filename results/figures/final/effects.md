| What changed | Model | Compared with | Puzzle accuracy (points) | Solved only by the model / by the other | p-value | Adjusted p-value (Holm) | Positions: same as Stockfish (points) | Positions: centipawns saved per legal move |
|---|---|---|---|---|---|---|---|---|
| more data (5x), player's moves | sft_100k | sft_v1 | +1.6 [-0.4, +3.6] | 60 / 44 | 0.14 | 1.0 | +3.6 [+1.5, +5.7] | +38 [+12, +64] |
| more data (5x), Stockfish's moves | sft_100k_stockfish | sft_stockfish | +1.9 [+0.0, +3.8] | 58 / 39 | 0.067 | 0.67 | +2.7 [+0.8, +4.7] | +43 [+16, +70] |
| more data (5x), legal moves listed | sft_100k_legal_moves | sft_legal_moves | +4.3 [+2.1, +6.6] | 86 / 43 | 0.00019 | 0.003 | +4.3 [+2.0, +6.6] | +45 [+24, +66] |
| Stockfish's moves instead of the player's, 20k | sft_stockfish | sft_v1 | -0.5 [-2.0, +1.0] | 27 / 32 | 0.6 | 1.0 | +0.4 [-1.1, +1.9] | -16 [-37, +5] |
| Stockfish's moves instead of the player's, 100k | sft_100k_stockfish | sft_100k | -0.2 [-1.8, +1.4] | 31 / 33 | 0.9 | 1.0 | -0.5 [-2.4, +1.3] | -22 [-40, -3] |
| legal moves listed, base model (no training) | base_legal_moves | base | +3.2 [+1.5, +5.0] | 58 / 26 | 0.00063 | 0.0093 | +5.5 [+3.7, +7.4] | -60 [-102, -18] |
| legal moves listed, SFT on 20k | sft_legal_moves | sft_v1 | +5.4 [+2.9, +8.0] | 109 / 55 | 3e-05 | 0.00051 | +4.3 [+2.0, +6.6] | -13 [-42, +15] |
| legal moves listed, SFT on 100k | sft_100k_legal_moves | sft_100k | +8.1 [+5.3, +10.9] | 145 / 64 | 2.1e-08 | 3.8e-07 | +5.0 [+2.4, +7.7] | -9 [-35, +18] |
| DPO after SFT (DPO + SFT loss) | dpo_v3 | sft_v1 | -1.1 [-2.4, +0.1] | 15 / 26 | 0.12 | 1.0 | -0.4 [-1.8, +1.0] | +11 [-6, +28] |
| DPO pairs: own mistakes instead of random blunders | dpo_v2 | dpo_v1 | -1.5 [-3.3, +0.3] | 35 / 50 | 0.13 | 1.0 | -1.0 [-2.8, +0.8] | +27 [-9, +63] |
| DPO: SFT loss added | dpo_v3 | dpo_v2 | +2.3 [+0.6, +4.0] | 50 / 27 | 0.012 | 0.14 | +1.9 [+0.1, +3.7] | -17 [-47, +13] |
| DPO after SFT, legal moves listed | sft_legal_dpo | sft_legal_moves | -0.4 [-2.4, +1.5] | 46 / 50 | 0.76 | 1.0 | -1.9 [-3.6, -0.2] | -24 [-42, -6] |
| RL after SFT (KL penalty 0.2) | grpo_v5 | sft_v1 | -2.8 [-4.4, -1.3] | 18 / 46 | 0.00062 | 0.0093 | -1.4 [-3.2, +0.4] | +46 [+22, +71] |
| RL: KL penalty 0.04 instead of none | grpo_v2 | grpo_v1 | +1.6 [+0.6, +2.6] | 21 / 5 | 0.0025 | 0.033 | +0.7 [-0.3, +1.8] | -14 [-25, -2] |
| RL: reward without the legality bonus | grpo_v3 | grpo_v2 | -0.7 [-1.5, +0.0] | 4 / 11 | 0.12 | 1.0 | +0.3 [-0.7, +1.3] | +6 [-2, +14] |
| KL penalty 0.2 instead of 0.04 | grpo_v5 | grpo_v2 | +0.1 [-1.1, +1.3] | 18 / 17 | 1.0 | 1.0 | +1.9 [+0.5, +3.3] | -1 [-16, +13] |
| RL after SFT, legal moves listed | sft_legal_rl | sft_legal_moves | -2.4 [-4.4, -0.3] | 43 / 67 | 0.028 | 0.31 | +1.2 [-0.8, +3.2] | +75 [+55, +96] |
| RL after DPO, legal moves listed | sft_legal_dpo_rl | sft_legal_dpo | -1.9 [-4.3, +0.5] | 67 / 86 | 0.15 | 1.0 | +3.7 [+1.6, +5.9] | +112 [+90, +134] |

Each model minus the model it changes one thing in, on the same 1,000 test puzzles and 1,000 held-out game positions, with 95% bootstrap intervals. Centipawns saved: on positions both models answered legally. Adjusted p-value: Holm's correction for the 18 changes compared here. – = not measured yet.
