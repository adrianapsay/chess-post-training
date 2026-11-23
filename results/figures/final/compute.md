| Model | Training stages | Tokens, last stage (M) | Tokens, all stages (M) | GPU-hours, last stage | GPU-hours, all stages | GPU, last stage |
|---|---|---|---|---|---|---|
| sft_v1 | sft_v1 | 1.9 | 1.9 | 6.6 | 6.6 | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| sft_stockfish | sft_stockfish | 1.9 | 1.9 | 0.94 | 0.94 | NVIDIA A10G |
| sft_100k | sft_100k | 9.5 | 9.5 | 4.45 | 4.45 | NVIDIA A10G |
| sft_100k_stockfish | sft_100k_stockfish | 9.5 | 9.5 | 4.42 | 4.42 | NVIDIA A10G |
| sft_legal_moves | sft_legal_moves | 4.53 | 4.53 | 1.14 | 1.14 | NVIDIA A10G |
| dpo_v1 | sft_v1 -> dpo_v1 | 0.87 | 2.77 | 0.33 | 6.93 | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| dpo_v2 | sft_v1 -> dpo_v2 | 0.87 | 2.77 | 0.21 | 6.82 | NVIDIA A10G |
| dpo_v3 | sft_v1 -> dpo_v3 | 0.87 | 2.77 | 0.21 | 6.82 | NVIDIA A10G |
| grpo_v1 | sft_v1 -> grpo_v1 | 1.5 | 3.41 | 0.54 | 7.15 | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| grpo_v2 | sft_v1 -> grpo_v2 | 1.5 | 3.41 | 0.5 | 7.1 | NVIDIA A10G |
| grpo_v3 | sft_v1 -> grpo_v3 | 1.5 | 3.41 | 0.5 | 7.1 | NVIDIA A10G |
| grpo_v4 | sft_v1 -> grpo_v4 | 1.5 | 3.41 | 0.5 | 7.1 | NVIDIA A10G |
| grpo_v5 | sft_v1 -> grpo_v5 | 1.5 | 3.41 | 0.5 | 7.1 | NVIDIA A10G |

Training compute of each model's stages. Compare by tokens; GPU-hours depend on the GPU, and sft_v1's are inflated (laptop GPU, before the memory fix). DPO pair collection and grading are not counted.
