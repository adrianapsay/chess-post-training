| model | compared with | puzzles | solved only by this model | solved only by the other | p-value (sign test) |
|---|---|---|---|---|---|
| sft_legal_dpo | sft_legal_moves | 1000 | 46 | 50 | 0.76 |
| sft_legal_rl | sft_legal_moves | 1000 | 43 | 67 | 0.028 |
| sft_legal_dpo_rl | sft_legal_moves | 1000 | 50 | 73 | 0.047 |
| sft_legal_dpo_rl | sft_legal_dpo | 1000 | 67 | 86 | 0.15 |
| sft_legal_dpo_rl | sft_legal_rl | 1000 | 29 | 28 | 1.0 |
| sft_legal_moves | sft_v1 | 1000 | 109 | 55 | 3e-05 |

Puzzles solved by only one of the two models, and the sign-test p-value (below 0.05 = unlikely to be chance).
