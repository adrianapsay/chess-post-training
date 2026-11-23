| Model | compared with | puzzles | solved only by this model | solved only by the other | p-value (sign test) | adjusted p-value (Holm) | Accuracy (%) | vs. sft_v1 |
|---|---|---|---|---|---|---|---|---|
| base | sft_v1 | 1000 | 21 | 71 | 1.6e-07 | 2.7e-06 | 2.8 | worse |
| base_legal_moves | sft_v1 | 1000 | 55 | 73 | 0.13 | 0.6 | 6.0 | no significant difference |
| sft_stockfish | sft_v1 | 1000 | 27 | 32 | 0.6 | 0.6 | 7.3 | no significant difference |
| sft_100k | sft_v1 | 1000 | 60 | 44 | 0.14 | 0.6 | 9.4 | no significant difference |
| sft_100k_stockfish | sft_v1 | 1000 | 58 | 44 | 0.2 | 0.6 | 9.2 | no significant difference |
| sft_legal_moves | sft_v1 | 1000 | 109 | 55 | 3e-05 | 0.00045 | 13.2 | better |
| sft_100k_legal_moves | sft_v1 | 1000 | 148 | 51 | 3.7e-12 | 6.7e-11 | 17.5 | better |
| dpo_v1 | sft_v1 | 1000 | 22 | 41 | 0.023 | 0.16 | 5.9 | no significant difference |
| dpo_v2 | sft_v1 | 1000 | 29 | 63 | 0.00051 | 0.0056 | 4.4 | worse |
| dpo_v3 | sft_v1 | 1000 | 15 | 26 | 0.12 | 0.6 | 6.7 | no significant difference |
| grpo_v1 | sft_v1 | 1000 | 17 | 62 | 3.6e-07 | 5.8e-06 | 3.3 | worse |
| grpo_v2 | sft_v1 | 1000 | 25 | 54 | 0.0015 | 0.013 | 4.9 | worse |
| grpo_v3 | sft_v1 | 1000 | 21 | 57 | 5.6e-05 | 0.00078 | 4.2 | worse |
| grpo_v4 | sft_v1 | 1000 | 20 | 50 | 0.00044 | 0.0053 | 4.8 | worse |
| grpo_v5 | sft_v1 | 1000 | 18 | 46 | 0.00062 | 0.0062 | 5.0 | worse |
| sft_legal_dpo | sft_v1 | 1000 | 103 | 53 | 7.7e-05 | 0.001 | 12.8 | better |
| sft_legal_rl | sft_v1 | 1000 | 97 | 67 | 0.023 | 0.16 | 10.8 | no significant difference |
| sft_legal_dpo_rl | sft_v1 | 1000 | 95 | 64 | 0.017 | 0.14 | 10.9 | no significant difference |

Each model vs. sft_v1 on the same 1,000 test puzzles. Adjusted p-value: Holm's correction for 18 comparisons; below 0.05 = a real difference.
