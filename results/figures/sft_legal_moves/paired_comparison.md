| model | compared with | puzzles | solved only by this model | solved only by the other | p-value (sign test) |
|---|---|---|---|---|---|
| base_legal_moves | base | 1000 | 58 | 26 | 0.00063 |
| sft_legal_moves | sft_v1 | 1000 | 109 | 55 | 3e-05 |
| sft_legal_moves | base_legal_moves | 1000 | 115 | 43 | 9e-09 |

Puzzles solved by only one of the two models, and the sign-test p-value (below 0.05 = unlikely to be chance).
