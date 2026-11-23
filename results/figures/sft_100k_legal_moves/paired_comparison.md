| model | compared with | puzzles | solved only by this model | solved only by the other | p-value (sign test) |
|---|---|---|---|---|---|
| sft_100k_legal_moves | sft_legal_moves | 1000 | 86 | 43 | 0.00019 |
| sft_100k_legal_moves | sft_100k | 1000 | 145 | 64 | 2.1e-08 |
| sft_100k_legal_moves | sft_v1 | 1000 | 148 | 51 | 3.7e-12 |
| sft_100k_legal_moves_at_20k | sft_legal_moves | 1000 | 42 | 74 | 0.0038 |

Puzzles solved by only one of the two models, and the sign-test p-value (below 0.05 = unlikely to be chance).
