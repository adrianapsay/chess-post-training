# Project notes

What we are doing and what the results show, readable without the code. Exact settings are in `configs/`, details are in each notebook, and setup instructions are in the [README](../README.md).

As of Nov 5, all four models (base, SFT, DPO, and RL) are trained and graded. After the progress report we ran five new training runs (notebooks 08–10) and added two new evaluations (notebook 11), and designed three more experiments (notebooks 12–14). Their design, results so far, and where to find each figure are in [section 8](#8-after-the-progress-report-follow-up-experiments).

1. [The project](#1-the-project)
2. [How the methods work](#2-how-the-methods-work)
3. [What each notebook does](#3-what-each-notebook-does)
4. [Results, and why](#4-results-and-why)
5. [Limitations and next steps](#5-limitations-and-next-steps)
6. [Key facts for the report](#6-key-facts-for-the-report)
7. [Glossary](#7-glossary)
8. [After the progress report: follow-up experiments](#8-after-the-progress-report-follow-up-experiments)

---

## 1. The project

Language models can describe chess but often play it poorly, frequently proposing illegal moves. We study which post-training methods improve a small open model (Google's Gemma 4 E2B), and why. We compare supervised fine-tuning (SFT), direct preference optimization (DPO), and reinforcement learning (RL), first separately and then in combination.

```
Lichess games (2000+ players) ──> training positions ──> SFT ──┬──> DPO ──┐
                                                                └──> RL ───┤
Lichess puzzles (never trained on) ─────────────────────────> grade every model the same way ──> compare
```

We train on positions from games between strong players, and test on puzzles. Games supply many varied, realistic positions to learn from. Puzzles each have one verified answer: in a typical game position several moves are about equally good, but in a puzzle one move wins and almost every other move loses, so an answer is clearly right or wrong. No test puzzle appears in the training data, so test scores measure how well a model handles positions it has never seen, not whether it remembers answers.

We measure three things. Our main metric is **puzzle accuracy**, the percentage of puzzles solved. The **legal-move rate** is the percentage of answers that are legal moves; the rest are *illegal* (not allowed in the position) or *unparseable* (no move found). **ACPL** (average centipawn loss) measures how much worse the model's legal moves are than the best move according to Stockfish, a strong chess engine, so lower is better.

---

## 2. How the methods work

It helps to start with how the model produces a move. The model reads the prompt, which describes the board as a line of text (a FEN), and writes its answer one token at a time: `e2e4` is four tokens (`e`, `2`, `e`, `4`). At each step it assigns a probability to every possible next token, and we always take the most likely one. Nothing checks the move against the rules of chess before it is written. All three training methods change these probabilities; they differ in which answers they make more or less likely, and based on what.

**The base model** is Gemma 4 E2B as Google released it. It was trained on a very large amount of text to predict the next token, then tuned to follow instructions. That text includes chess, so it knows what moves look like, but it was never trained to choose a move for a given position.

**SFT** (supervised fine-tuning) continues the same next-token training on our own examples: a position, followed by the move a strong player made there. Training makes that move more likely. The model only ever sees correct examples; it is never shown an illegal move or told that one of its own answers was wrong.

**DPO** (direct preference optimization) trains on pairs of answers to the same position: a *chosen* move (Stockfish's best) and a *rejected* move (one that loses a lot). It makes the chosen move more likely *relative to* the rejected one, compared with where the model started. It only cares about this difference. It does not require the chosen move itself to become more likely, and it says nothing about any other move.

**RL** (reinforcement learning) shows the model no answers at all. The model generates its own answers, a *reward* scores each one, and training makes high-reward answers more likely. Our reward comes from Stockfish: −1 for an illegal or unparseable answer, and for a legal move, more the closer it is to the best move. RL is the only method that directly penalizes the model's own illegal moves. The specific algorithm we use is **GRPO**. For each position, the model samples 8 answers, and each one is compared with the group's average reward: answers above the average become more likely, and answers below it less likely. In our settings, this is the classic *policy gradient with a baseline*, where the baseline is the group average. We chose GRPO because it judges each answer simply by comparing it with the model's other answers to the same position.

| | SFT | DPO | RL (GRPO) |
|---|---|---|---|
| Starts from | base model | SFT model | SFT model |
| Learns from | 20,000 positions + the move a strong player made | 4,549 pairs of a better and a worse move | its own answers on 2,000 positions, scored by Stockfish |
| Makes more likely | the human move | the better move, relative to the worse one | answers that score above the group average |
| Compute | 6.6 GPU-hours | 0.33 GPU-hours | 0.54 GPU-hours |

All three train a small add-on to the model (a LoRA adapter) while the original model stays frozen, which lets training fit on one 12 GB GPU.

---

## 3. What each notebook does

| Notebook | What it does | Needs a GPU? |
|---|---|---|
| [00](../notebooks/00_setup_check.ipynb) Setup check | checks the libraries, GPU, Stockfish, and tokenizer | no |
| [01](../notebooks/01_data_pipeline.ipynb) Data pipeline | builds the training positions and the puzzle sets | no |
| [02](../notebooks/02_eval_base_model.ipynb) Base model | grades the untrained model | yes |
| [03](../notebooks/03_sft_first_run.ipynb) SFT | trains with SFT and grades the result | yes |
| [04](../notebooks/04_dpo_rl_inputs.ipynb) DPO and RL inputs | builds the DPO pairs and the RL reward | no |
| [05](../notebooks/05_dpo_first_run.ipynb) DPO | trains with DPO and grades the result | yes |
| [06](../notebooks/06_grpo_first_run.ipynb) RL | trains with GRPO, grades the result, and compares all four models | yes |
| [07](../notebooks/07_understanding_results.ipynb) Understanding the results | explains *why* the models score as they do | no |

Exact settings for every step are in `configs/`.

### 00 · Setup

This notebook checks that the libraries load, shows which GPU is available, tests Stockfish, and downloads the model's tokenizer. Because Stockfish grades every move in the project, we test it on two positions with known answers. The starting position scores about 0 (+32 centipawns), and the same position without Black's queen scores about a queen more (+886), which confirms the sign and the scale of its scores. The notebook also shows how the model splits moves into tokens.

### 01 · Data pipeline

This notebook builds all the data. The test and dev sets come from the Lichess puzzle database (6.2 million puzzles): 1,000 test puzzles and 200 dev puzzles, with equal numbers in four rating bands (<1200, 1200–1800, 1800–2400, 2400+). Both sets are fixed and committed in `data/eval/`.

The training data comes from one file of Lichess games from September 2025, filtered as follows ([numbers](../results/data_stats.json)):

| Filter step | Games remaining |
|---|---|
| Games read | 1,404,031 |
| Both players rated 2000+ | 245,522 |
| No bot accounts | 242,573 |
| Not bullet (at least 3 minutes per player) | 92,501 |
| Ended normally (not on time or abandoned) | 70,904 |
| At least 20 moves (first 50,000 kept) | 50,000 |

From each game we take three positions, skipping the first five moves since openings are mostly memorized, and pair each with the move that was played. This gives 142,500 positions for training and 7,500 for validation, split by game so that no game contributes to both. We exclude bot accounts because a cluster of players rated near 3000 turned out to be almost all bots, and we want the model to learn from human play. The notebook checks that no training position appears among the puzzles, and dataset versions are pinned so the data can be rebuilt exactly.

One detail of the puzzle format matters: a Lichess puzzle starts one move *before* the position to solve. The first listed move is the opponent's, so we apply it, and the second move is the answer.

### 02 · Base model evaluation

This notebook asks the base model every puzzle and grades its answers, giving the baseline every trained model is compared with. Every model, in training and testing, gets the same prompt:

```
Chess position (FEN): <the board>
White to move.
Reply with the best move in UCI notation (for example e2e4) and nothing else.
```

We grade the first move found in the reply. It is checked against the board (legal, illegal, or unparseable), compared with the puzzle's answer (in mate-in-1 puzzles, any checkmating move counts), and scored by Stockfish at a fixed search depth, with centipawn loss capped at 1000. Both UCI and standard notation are accepted, so a correct move is never marked wrong for its format. Text around the move is ignored as well: bold markers, brackets, move numbers, and HTML tags (`<h2>Rd7</h2>` reads as Rd7). The model always gives its single most likely answer (greedy decoding), so grading is reproducible. We load only Gemma's language model and skip its image and audio components, which leaves 4.63B parameters (about 9.3 GB). Each model is graded on the 200 dev puzzles first as a quick check, then on the 1,000 test puzzles.

**Same results on a different GPU.** From mid-November we also run on the course's Databricks GPUs (NVIDIA A10G), so we re-graded the base model there and compared it with the laptop run (RTX 5070 Ti) puzzle by puzzle. 768 of the 1,000 replies were identical. Greedy decoding is reproducible on one machine, but two GPUs round numbers slightly differently, and when the model's top two choices for a token are nearly tied, the rounding can decide between them. For about 10% of puzzles this tipped the base model into wrapping its move in HTML tags (`<h2>Rd7</h2>`), which our move reader did not yet ignore. With tags ignored, the two runs agree closely: accuracy 2.9% vs. 2.8%, legal moves 41.6% vs. 41.2%, unparseable 12.3% vs. 13.0%, ACPL 823 vs. 828. Ignoring tags changes no earlier result, since no laptop reply from any model contained one, and the trained models never use tags.

### 03 · SFT

This notebook trains the SFT model on 20,000 of the training positions, then grades it exactly as in notebook 02 and makes the base vs. SFT comparison. The LoRA adapter has rank 16 and covers the attention and MLP layers, and the frozen model is stored in 4 bits (QLoRA) so training fits in 12 GB of GPU memory. The loss counts only the answer, not the prompt. A 500-example trial run comes first to catch problems early. The full run took 6.6 GPU-hours.

### 04 · DPO and RL inputs

This notebook builds what DPO and RL train on, using training positions only, never puzzles. For DPO, it takes 5,000 positions. In each, Stockfish's best move is the chosen move, and a random legal move that loses at least 200 centipawns is the rejected one. In 212 positions no move lost that much, leaving 4,788 pairs. The rejected moves lose 620 centipawns at the median.

It also defines the RL reward. An illegal or unparseable answer gets −1. In the default *dense* mode, a legal move gets 0.1 plus up to 1 more, depending on how close it is to Stockfish's best move. In the *sparse* mode, only Stockfish's best move gets a reward (1). Stockfish searches less deeply here than in grading, to keep things fast.

### 05 · DPO

This notebook trains the DPO model on 4,549 pairs, starting from the SFT model, and holds out 239 pairs to monitor training. The SFT model is also the reference that DPO compares against, with β = 0.1. One large table in Gemma (4.7 GB) is kept on the CPU during training so it fits on the GPU; this was added after the SFT run, which likely ran slower without it. Training took 0.33 GPU-hours, and the model is then graded like the others.

### 06 · RL

This notebook trains the RL model with GRPO, starting from the SFT model. It uses 2,000 training positions, samples 8 answers for each (16,000 answers in all, each scored by Stockfish with the dense reward), and has no KL penalty holding it close to the SFT model. Training took 0.54 GPU-hours. The notebook then grades the model and makes the comparison table and figures for all four models.

### 07 · Understanding the results

This notebook analyzes the saved results without a GPU or the model. It measures how certain the results are, compares them with random guessing, classifies the illegal moves, measures move quality, and looks at what happened during SFT and DPO training. Every graded model is included automatically, so we rerun it after RL. Section 4 summarizes it.

---

## 4. Results, and why

All results are on the 1,000 test puzzles. The figures and calculations are in [notebook 07](../notebooks/07_understanding_results.ipynb).

| Model | Accuracy | Legal answers | Accuracy when the answer is legal |
|---|---|---|---|
| Random legal move | 5.6% | 100% | 5.6% |
| Base | 2.8% | 41% | 6.8% |
| SFT | 7.8% | 72% | 10.8% |
| DPO | 5.9% | 63% | 9.3% |
| RL (GRPO) | 3.3% | 82% | 4.0% |

The first row is a reference point: picking a random legal move solves 5.6% of the puzzles. The last column separates two questions: can the model produce a valid move, and does it choose well among valid moves?

**The base model** scores below random guessing. Only 41% of its answers are legal, and most are written in standard notation even though we ask for UCI. This fits how it was trained: it writes what chess text usually looks like, without working the move out from the board. When its answer is legal, it is about as good as random.

**SFT** is better than the base model on every measure, and the difference is statistically clear ([figure](../results/figures/accuracy_intervals.png)). It always answers in UCI, because every training example ends with a UCI move. When its answer is legal, it is right about twice as often as random, because it has learned which moves strong players make in which kinds of positions. But 28% of its answers are still illegal, because SFT makes the human move more likely without ever making illegal moves less likely. The mistakes show what the model has not learned: it almost always moves one of its own pieces, so it reads *where* the pieces are, but it often moves them where they cannot go or leaves its king in check ([figure](../results/figures/illegal_reasons.png)). This is worst when the king is under attack and few moves are legal. Puzzle accuracy also stays low because imitating typical moves in ordinary positions is a different skill from finding the one winning move in a puzzle.

**DPO** made the SFT model worse (7.8% → 5.9%), and this drop is unlikely to be chance. The training logs show why. DPO only asks for the better move to become more likely *than* the worse one, and the model achieved this by making *both* less likely: the better move ended about 20 times less likely than under SFT, the worse one about 500 times less likely ([figure](../results/figures/dpo_diagnostics.png)). Since the probabilities of all possible answers add up to 1, the removed probability went to other answers, including illegal ones: 155 answers that were legal under SFT became illegal. So DPO taught the model to prefer Stockfish's move over a bad one, but not to avoid illegal moves. This is one run with one setting, so it does not show that DPO cannot help.

**RL** fixed the weakness it targets and lost accuracy doing so. Its reward penalizes illegal answers, and RL has the most legal answers of any model (82%, up from SFT's 72%). But accuracy fell from 7.8% to 3.3%, a drop that is statistically clear, and when its answer is legal it is right less often than random (4.0% vs. 5.6%). The kind of moves it plays shows why: 41% of its legal answers are pawn moves (SFT: 2%), while captures fell from 25% to 6% and checks from 18% to 8%. A quiet pawn move is almost always legal, so it earns the legal bonus and avoids the −1 penalty, but puzzles are solved by captures, checks, and other forcing moves. The reward invites this trade: an illegal answer costs −1 against at least +0.1 for any legal one, while the best and worst legal moves differ by only 1. The reward rose steadily during training while the answers became much less varied, so the model learned what the reward asked for, not what the puzzles test. Training on game positions and testing on puzzles may add to this. As with DPO, this is one run with one setting.

Across all models, accuracy stays within a few points of random guessing. The model answers in a single move with no reasoning steps, and puzzles are hard. ACPL also turns out to say little on puzzles: almost every move in a puzzle is either about the best or a large mistake, so it mostly repeats what accuracy says ([figure](../results/figures/move_quality.png)).

Our proposal asks whether a method helps by reducing blunders or by improving tactics. On puzzles, these turn out to be almost the same thing: missing the tactic usually *is* the blunder (88% of SFT's legal answers lose 200+ centipawns, and nearly all the rest are correct). Accuracy per puzzle theme (fork, pin, mate in 1, ...) shows no theme standing out clearly, with about 100 puzzles per theme. Separating the two questions needs ordinary game positions, where most moves are reasonable and a blunder is a distinct mistake.

---

## 5. Limitations and next steps

**Limitations**
- Each method has one run, so small differences may be noise. Notebook 07 shows which differences are clear.
- Results per difficulty level are noisy (250 puzzles each, about ±3.4 points).
- SFT learns from ordinary game positions, which may transfer only partly to puzzles.
- Many DPO pairs are easy to tell apart, since the rejected moves are often very bad.
- Note to self: this is currently a WIP

**Next steps**

Before the progress report (Nov 6):
- Done: RL (notebook 06) has run, and notebook 07 compares all four models.

Mid November, before the first draft (Nov 24), in this order:

| Step | What to try | Why at this point |
|---|---|---|
| 1. Fix DPO and RL on their own | RL: a KL penalty to the SFT model (`beta` in `rl.yaml`), a smaller gap between illegal and legal answers, the sparse reward, or puzzle-like training positions. DPO: a lower learning rate, adding the SFT loss back in (TRL can combine the two) so the better move stops losing probability, or changing how pairs are built, as our TA suggested (the SFT model's own wrong or illegal answers as the rejected moves, or a different 200-centipawn threshold). | Both currently make SFT worse, so we cannot yet say whether either method helps, and that is the core of our research question. |
| 2. SFT on Stockfish's moves | Keep the same game positions, but use Stockfish's best move as the answer instead of the move the player made. | If it clearly beats `sft_v1`, it becomes the starting point for everything after it, so it comes before the combinations. |
| 3. Combine methods | At least SFT → DPO → RL, from the best SFT model, with the DPO and RL settings that worked in step 1. | Our research question asks which combination works best, and combining methods that each make SFT worse would teach us little. |

Step 2 targets a mismatch in our current SFT data. SFT copies whatever answer it is shown, so it now learns to play like a 2000+ player, who usually picks a reasonable move but not always the best one. A puzzle accepts only the best move. Training on Stockfish's choice teaches the model to find that move directly, which a higher rating threshold only approximates. It also means all three methods learn from the same source, since DPO and RL already use Stockfish.

Alongside these, if time allows:
- Other SFT data changes our TA suggested: train on puzzles (not the test ones) instead of games, or change the rating threshold. Each needs a new SFT run, so one or two at most.

Promised in our proposal, still to decide as a team:
- Move quality on held-out game positions (legal-move rate, centipawn loss, blunder rate). It is the cheapest of these, and the only way to separate "fewer blunders" from "better tactics", which puzzles cannot do.
- Win-loss rate in games against Stockfish (simple to run).
- ChessQA, and a comparison with ChessLLM (more work).
- RL on puzzle positions as well as games.
- A similar data and compute budget for every method. So far SFT used 6.6 GPU-hours, DPO 0.33, and RL 0.54, so at the least, we report compute for every run. SFT's time is inflated: it ran before the memory fix described under notebook 05, when its 500-example trial ran about 5 times faster per step, so tokens seen (1.9M for SFT, 0.87M for DPO, 1.5M for RL) compare the methods better than hours.

Each ablation changes one factor, retrains, and grades the result the same way. Most factors are a setting in a config file:

| Ablation | Setting (file) | Default | Alternatives |
|---|---|---|---|
| DPO rejected moves | to be added | a random move losing 200+ cp | the SFT model's own wrong answers; illegal moves |
| DPO rejection threshold | `rejected_min_cp_loss` (`dpo.yaml`) | 200 | e.g. 100, 400 |
| SFT training data | to be added | games | puzzles (not the test ones); a mix |
| SFT answer for each position | to be added | the move the player made | Stockfish's best move |
| Player rating threshold | `min_elo` (`data.yaml`) | 2000 | e.g. 1600, 2200 |
| RL reward | `mode` (`rl.yaml`) | dense | sparse (following Hwang et al., cited in our draft) |
| Legal moves in the prompt (optional) | `show_legal_moves` (`prompt.yaml`) | no | yes |

---

## 6. Key facts for the report

Key facts for the Data and Methods sections. More detail is in the sections above, exact numbers are in `results/data_stats.json`, and all settings are in `configs/`.

**Data**
- **Training games:** Lichess game database, September 2025 (one file of 1.4M games), filtered to 50,000 games:
  - both players rated 2000+
  - no bot accounts
  - at least 3 minutes per player
  - a normal ending
  - at least 20 moves
- **Training examples:** 3 positions per game, skipping the first 5 moves. Each example is a board position and the move played. 142,500 for training and 7,500 for validation, split by game.
- **Test puzzles:** from the Lichess puzzle database (6.2M puzzles). 1,000 test and 200 dev puzzles, with equal numbers in four rating bands (<1200, 1200–1800, 1800–2400, 2400+). Each has one correct move.
- **No leakage:** no training position appears among the test or dev puzzles.

**Method**
- **Model:** Gemma 4 E2B, instruction-tuned. Only the language part is used: 4.63B parameters, about 2.3B "effective".
- **Prompt:** the board as a FEN, whose turn it is, and "reply with the best move". It is the same for every model, in training and testing.
- **SFT:** LoRA (rank 16) on 20,000 examples, one epoch. The model learns to produce the move that was played.
- **DPO:** 4,788 pairs. The preferred move is Stockfish's best; the rejected move is a legal move losing 200+ centipawns (our metric that grades how worse the model's move is than Stockfish's best). Training continues the SFT adapter for one epoch over 95% of the pairs (5% held out), with β = 0.1 and the SFT model as the reference.
- **RL (GRPO):** the model's own answers are scored by Stockfish. Illegal answers get −1; legal moves score higher the closer they are to Stockfish's best. Training continues the SFT adapter on 2,000 training positions, with 8 sampled answers per position and no KL penalty.
- **Evaluation:**
  - puzzle accuracy (primary; overall and per rating band)
  - legal-move rate (illegal and unparseable answers counted separately)
  - average centipawn loss (ACPL) against Stockfish
- **Fair comparison:** every model gets the same prompts, puzzles, and grading. Decoding is deterministic, seeds are fixed, data and model versions are pinned, and compute is logged per run.

---

## 7. Glossary

- **Token:** the unit of text a model reads and writes.
- **FEN:** a one-line text encoding of a full board position.
- **UCI / SAN:** two ways to write a move: `e2e4` (from-square, to-square) and `e4` or `Nf3` (standard chess notation).
- **Stockfish:** a free chess engine, far stronger than any human; our reference for move quality.
- **Centipawn:** one hundredth of a pawn. +100 means White is about one pawn ahead.
- **LoRA / adapter:** fine-tuning that freezes the original model and trains a small add-on (the adapter). QLoRA does the same with the frozen model stored in 4 bits, to save GPU memory.
- **Cross-entropy loss:** the training loss of SFT. Lower means the model gives higher probability to the correct move.
- **Reference model / β:** in DPO, the frozen starting model that the trained model is compared with, and how strongly the trained model is held close to it.
- **Policy gradient / baseline:** the basic RL update: make each sampled answer more or less likely according to how much its reward exceeds a baseline, the reward the model typically gets.
- **Greedy decoding:** always choosing the most likely next token, so results are reproducible.
- **Test set / dev set:** 1,000 puzzles for final results, never used for training or for choosing settings, and 200 separate puzzles for checks and for choosing settings.
- **Data leakage:** overlap between training and test data, which inflates test scores.
- **Statistically clear:** unlikely to be explained by chance (p < 0.05 in a paired comparison on the same puzzles).
- **Ablation:** an experiment that changes one factor and keeps everything else fixed.
- **GPU-hour:** one hour of one GPU's time; our measure of compute.

---

## 8. After the progress report: follow-up experiments

The first results raised three questions: would SFT learn more from Stockfish's moves than from human moves, why did DPO and RL make the SFT model worse, and do the models differ in ways puzzles cannot show? Each experiment below changes **one thing** compared with a model we already have, so a difference in results can be attributed to that change. The earlier models, results, and figures are left exactly as they were.

All new models are graded with the same puzzles, prompt, greedy decoding, and grader as before. Each experiment saves its tables and figures in its own folder, `results/figures/<model>/`; the [README](../README.md#where-the-results-are) has a map of every folder. All new runs used one NVIDIA A10G GPU on Databricks.

**Status (Nov 22):** 08, 09 and 10 are finished (results below). 11 is running for the nine models trained so far. 12, 13 and 14 are designed and ready to run. 15 and 16 follow once those results are in.

### Results at a glance (notebooks 08–10)

All numbers are on the 1,000 test puzzles. "Only this / only sft_v1" counts the puzzles solved by one of the two models but not the other; p is the sign-test p-value (below 0.05 = unlikely to be chance).

| Model | What changed | Accuracy | Legal | ACPL (cp) | vs. `sft_v1`: only this / only sft_v1, p | Training compute |
|---|---|---|---|---|---|---|
| `sft_v1` (reference) | – | 7.8% | 72.2% | 793 | – | 1.9M tokens |
| `sft_stockfish` | SFT answer = Stockfish's move | 7.3% | 71.5% | 799 | 27 / 32, p = 0.6 | 0.94 GPU-h, 1.9M tokens |
| `dpo_v1` (reference) | – | 5.9% | 63.3% | 820 | 22 / 41, p = 0.023 | 0.87M tokens |
| `dpo_v2` | rejected = the model's own mistakes | 4.4% | 61.6% | 793 | 29 / 63, p = 0.0005 | 0.22 GPU-h, 0.87M tokens |
| `dpo_v3` | `dpo_v2` + SFT loss | 6.7% | 72.9% | 806 | 15 / 26, p = 0.12 | 0.21 GPU-h, 0.87M tokens |
| `grpo_v1` (reference) | – | 3.3% | 81.6% | 836 | 17 / 62 | 1.5M tokens |
| `grpo_v2` | KL penalty β = 0.04 | 4.9% | 80.6% | 818 | 25 / 54, p = 0.0015 | 0.50 GPU-h, 1.5M tokens |
| `grpo_v3` | `grpo_v2` + new reward | 4.2% | 79.7% | 828 | 21 / 57, p < 0.001 | 0.50 GPU-h, 1.5M tokens |

A random legal move solves 5.6%. With 1,000 puzzles, the 95% margin of error at these accuracies is about ±1.6 percentage points, so the paired comparisons, not the raw percentages, show which differences are real.

**In one sentence:** no change beat `sft_v1` on puzzles; the changes to DPO (an added SFT loss) and RL (a KL penalty) significantly reduced the damage that post-training did, without improving on SFT.

### 08 · SFT on Stockfish's moves (`sft_stockfish`)

- **What changes:** the answer the model learns. `sft_v1` learned the move the 2000+ player made; `sft_stockfish` learns Stockfish's best move (search depth 10) in exactly the same 20,000 positions, in the same order, with the same settings.
- **Why:** a puzzle accepts only the best move, and strong players often choose a different, reasonable move. Training on the engine's choice targets puzzle-solving directly. It also means all three methods learn from the same source, since DPO's preferred moves and RL's reward already come from Stockfish. Our TA suggested this ablation.
- **Compared with:** `sft_v1` (the effect of the answer source) and the base model.
- **Also reported:** how often the human move and Stockfish's move agree in the training positions, which shows how much the training data actually changes.
- **Results:** the player's move equals Stockfish's in only 42.7% of the 20,000 training positions, so the experiment changed 57.3% of the answers. Yet `sft_stockfish` scores about the same as `sft_v1`: 7.3% vs. 7.8%, legal 71.5% vs. 72.2%; 27 puzzles are solved only by `sft_stockfish` and 32 only by `sft_v1` (p = 0.6, no significant difference). It is far better than the base model (68 vs. 23, p = 2.5 × 10⁻⁶).
- **What it suggests:** with 20,000 examples, the model does not learn *which* move is best from either kind of answer; it learns the answer format and plausible-looking moves. Notebook 11's "same as Stockfish" and "same as the player" columns will show whether its moves changed in ordinary positions.
- **Figures and tables** (`results/figures/sft_stockfish/`): `training_data` (how often the two answers agree), `training_loss.png`, `puzzle_results`, `paired_comparison`, `puzzle_metrics.png`, `accuracy_by_band.png`, `answer_outcomes.png`.

### 09 · DPO on the SFT model's own mistakes (`dpo_v2`, `dpo_v3`)

- **What changes in the data:** the *rejected* answer of each pair. In `dpo_v1` it was a random legal move losing at least 200 centipawns: mostly severe blunders the model would rarely play, so avoiding them taught little. Now we run `sft_v1` on training positions it was not trained on and keep its own mistakes: answers that are illegal, unparseable, or legal but losing at least 200 centipawns. Each mistake is paired with Stockfish's best move as the *chosen* answer. Building pairs from the model's own outputs is sometimes called *on-policy* preference data. Our TA suggested changing how pairs are built.
- **Two runs, both starting from `sft_v1`, with the same 4,788 pairs (the same number as `dpo_v1`):**
  - `dpo_v2`: the standard DPO loss. Compared with `dpo_v1`, only the rejected answers differ.
  - `dpo_v3`: the DPO loss plus the SFT loss on the chosen answer. In `dpo_v1`, DPO lowered the probability of both the chosen and the rejected move, and the freed probability partly went to illegal moves. The added SFT term rewards the chosen move directly, so its probability should not fall. Compared with `dpo_v2`, only the loss differs.
- **Also reported:** how often `sft_v1` makes a mistake on new game positions, and what share of the mistakes are illegal moves vs. blunders.
- **The pairs:** `sft_v1` answered 8,000 new training positions and made a mistake in 59.9% of them. Of the 4,788 rejected answers, 50.2% are illegal moves and 49.8% legal moves losing 200+ centipawns (median loss 680 cp). `dpo_v1`'s rejected moves were all legal blunders (median 620 cp).
- **Results:** `dpo_v2` scores 4.4%, significantly below `sft_v1` (29 vs. 63, p = 0.0005) and not significantly different from `dpo_v1` (35 vs. 50, p = 0.13). `dpo_v3` scores 6.7% with 72.9% legal answers: significantly better than `dpo_v2` (50 vs. 27, p = 0.012) and not significantly different from `sft_v1` (15 vs. 26, p = 0.12).
- **What the training logs show:** the model learned to separate the pairs in both runs (it preferred the chosen answer in 83% of training pairs for `dpo_v2` and 91% for `dpo_v3` by the end). In `dpo_v2`, however, the probability of Stockfish's move *fell* (log-probability −4.19 → −5.51), the same failure as `dpo_v1`: DPO pushed the rejected answer down faster than the chosen one, and lowered both. In `dpo_v3`, the SFT loss held the chosen move's probability steady (−4.19 → −4.22) while the model's own mistakes fell to about half their SFT probability. This is why `dpo_v3` keeps SFT's accuracy and legality, but nothing in it pushes the best move *up*, so it does not improve on SFT.
- **So:** changing the rejected answers alone did not help; the SFT loss is what prevented the damage.
- **Figures and tables:** `results/figures/dpo_mistake_pairs/` (`pairs`: what the mistakes are; `rejected_move_loss.png`: how much the legal mistakes lose), and in `results/figures/dpo_v2/` and `dpo_v3/`: `dpo_diagnostics.png` (left: how often the chosen answer is preferred; right: the probability of the chosen and rejected answers relative to SFT), `training_loss.png`, `puzzle_results`, `paired_comparison`, `puzzle_metrics.png`, `accuracy_by_band.png`, `answer_outcomes.png`.

### 10 · RL with a KL penalty and an adjusted reward (`grpo_v2`, `grpo_v3`)

- **Background:** `grpo_v1` raised the legal-answer rate to 82% but fell to 3.3% puzzle accuracy by switching to quiet pawn moves. Two features of its setup favor this: nothing held it close to the SFT model, and the reward valued legality (−1 for illegal, at least +0.1 for any legal move) as much as the whole range of move quality (best vs. worst legal move: a difference of 1).
- **The KL penalty:** the *KL divergence* measures how far the model's probabilities over answers have moved from those of the SFT model. GRPO subtracts β × KL from what it maximizes, so drifting away from SFT has a cost that β controls. The model can still change where the reward gain outweighs that cost.
- **Two runs, both starting from `sft_v1`, on the same 2,000 positions as `grpo_v1`:**
  - `grpo_v2`: adds the KL penalty (β = 0.04), with the original reward. Compared with `grpo_v1`, only β differs.
  - `grpo_v3`: the KL penalty and a new reward: −0.5 for an illegal or unparseable answer, and 1 − (centipawn loss / 1000) for a legal move, with no bonus for legality alone. An illegal answer is still worse than any legal move, but the gap is now half the range of move quality. Compared with `grpo_v2`, only the reward differs.
- **Also reported:** the KL divergence during training, and the share of pawn moves, captures, and checks among each model's legal answers.
- **Results:** the KL penalty helped. `grpo_v2` solves 4.9%, significantly more than `grpo_v1`'s 3.3% (21 vs. 5, p = 0.0025), and its share of pawn moves among legal answers fell from 41% to 25% (captures 9%, checks 10%; `sft_v1`: 2% pawn moves, 25% captures, 18% checks). It is still significantly below `sft_v1` (25 vs. 54, p = 0.0015). The reward change in `grpo_v3` made no significant difference compared with `grpo_v2` (4.2%; 4 vs. 11, p = 0.12; pawn moves 30%).
- **What the training logs show:** the average reward rose in every run (`grpo_v2`: −0.14 → 0.30), while puzzle accuracy fell. The KL divergence from SFT kept growing throughout training (0 → 0.29 for `grpo_v2`, 0 → 0.32 for `grpo_v3`): β = 0.04 slowed the drift away from SFT but did not stop it. Notebook 14 tests larger values.
- **Figures and tables** (`results/figures/grpo_v2/` and `grpo_v3/`, the same comparison in both): `reward.png`, `kl.png`, `move_kinds` (table and figure), `puzzle_results`, `paired_comparison`, `puzzle_metrics.png`, `accuracy_by_band.png`, `answer_outcomes.png`.

### Why post-training has not beaten SFT so far

Three observations from the results and training logs above:

1. **The SFT model gives the best move little probability, so there is little for DPO and RL to sharpen.** DPO and RL do not teach new moves; they shift probability between answers the model already produces. At the start of DPO training, `sft_v1` gave Stockfish's move a log-probability of about −4.2, a probability of roughly 1.5%. `sft_stockfish` points the same way: changing 57% of the training answers did not change puzzle accuracy, so 20,000 examples do not seem to be enough for the model to learn which move is best.
2. **The training signal and the test measure different things.** All training positions come from ordinary games, and the DPO pairs and RL reward favor moves that avoid large centipawn losses. Puzzles require one forcing tactical move. In ordinary positions a quiet move is often good enough, which is what RL learned: its reward rose while its puzzle accuracy fell. Notebook 11 measures move quality in ordinary positions, which is closer to what DPO and RL were trained for.
3. **The fixes worked only partly.** The SFT loss kept DPO from lowering the best move's probability but did not raise it; the KL penalty slowed RL's drift away from SFT but β = 0.04 did not stop it.

These motivate notebooks 12 (a stronger SFT starting point), 13 (separating "knowing which moves are legal" from "choosing a good one"), and 14 (a larger KL penalty).

### 11 · Held-out game positions and full games (every model)

These are the two evaluations our progress report committed to. They run on every model, old and new, and add new result files without changing earlier ones. The notebook skips models that already have results, so it is rerun after each round of new models and only evaluates what is new.

- **Held-out game positions:** 1,000 ordinary positions from validation games (games no model trained on), frozen in `data/eval/game_positions.csv`. There is no single right answer in most of them, so we measure the legal-move rate, ACPL, the **blunder rate** (answers losing 200+ centipawns), and how often the answer equals Stockfish's move and the player's move. A random legal move is included as a reference. This separates "makes fewer blunders in ordinary positions" from "finds tactics in puzzles", which puzzles alone cannot.
- **Full games:** 50 games per model against each of two opponents: a player that moves at random (the weakest possible opponent) and Stockfish searching to depth 1 (shallow, but far stronger than our models). Games start after the first three moves of 25 different held-out games, and each opening is played once with each color. An illegal or unparseable answer is replaced by a random legal move and counted, so every game finishes. A game still going after 100 moves counts as a draw. We report wins, draws, losses, the score ((wins + ½ draws) / games), and the share of invalid moves over whole games.
- **Why it matters:** DPO and RL were trained on ordinary game positions to avoid bad moves, not to solve puzzles. These two evaluations measure exactly that, so they are the fairer test of what DPO and RL changed. It is possible that a model with lower puzzle accuracy makes fewer blunders in ordinary positions; this is what the comparison will show.
- **Outputs:** `results/positions_summary.csv` and `results/games_summary.csv` (one row per model, and per opponent), `results/<model>_positions.json` and `results/<model>_games.json` (every answer and every game, as PGN that any chess site can replay), and in `results/figures/game_evals/`: `game_positions` (table and figure: legal rate, ACPL, blunders, same move as Stockfish), `games_vs_random` and `games_vs_stockfish` (tables and figures: wins, draws, losses).
- **Results:** pending (running as of Nov 22).

### 12 · SFT with more training data (`sft_100k`)

- **What changes:** the number of training examples, from 20,000 to 100,000 (of the 142,500 available). The first 20,000 are exactly `sft_v1`'s positions, in the same order; everything else (answers, settings, prompt) is the same.
- **Why:** every later method builds on SFT, and the observations above suggest the SFT model is too weak a starting point. A stronger SFT model should also give DPO and RL more to work with.
- **The scaling curve:** instead of training five separate models, one run saves a copy of the model after 20k, 40k, 60k and 80k examples, and every copy is graded on the puzzles. This gives puzzle accuracy as a function of training data. Caveat: the learning rate decreases over the whole run (a cosine schedule), so a copy saved partway has not finished that schedule. The copies are close to, but not exactly, models trained on fewer examples. Comparing the 20k copy with `sft_v1` (the same data, schedule finished) shows how large this effect is.
- **Compared with:** `sft_v1` (the effect of 5 times more data).
- **Figures and tables** (`results/figures/sft_100k/`): `scaling_curve.png` and `scaling` (accuracy and legal rate against training examples), `training_loss.png`, `puzzle_results`, `paired_comparison`, `puzzle_metrics.png`, `accuracy_by_band.png`, `answer_outcomes.png`.
- **Compute:** about 5 hours on one A10.

### 13 · Listing the legal moves in the prompt (`sft_legal_moves`, `base_legal_moves`)

- **What changes:** the prompt gains one line listing every legal move in the position (`show_legal_moves` in `configs/prompt.yaml`, which existed but was off). The model must still choose a move, but no longer has to work out from the FEN which moves are possible. Training uses `sft_v1`'s 20,000 positions and answers; training and grading both use the new prompt.
- **Why:** illegal answers are the largest single failure (28% of `sft_v1`'s answers). This experiment separates two reasons a model can fail: naming an impossible move, and choosing a poor move among the possible ones.
- **Two models:** `base_legal_moves` (the untrained base model with the new prompt; compared with `base`, the effect of the list alone) and `sft_legal_moves` (SFT with the new prompt; compared with `sft_v1`, the effect with training).
- **Also reported:** *accuracy when legal* = puzzle accuracy among the legal answers only. If legality rises but this does not, the list removed impossible moves without improving the choice.
- **Caveat:** the list gives the model information the other models must work out themselves, so this is an easier version of the task, reported as an ablation rather than a replacement for the original setting.
- **Figures and tables** (`results/figures/sft_legal_moves/`): `prompt_length`, `training_loss.png`, `puzzle_results`, `accuracy_when_legal` (table and figure), `paired_comparison`, `answer_outcomes.png`, `accuracy_by_band.png`.
- **Compute:** about 1.5–2 hours on one A10 (the prompts are about twice as long).

### 14 · RL with a larger KL penalty (`grpo_v4`, `grpo_v5`)

- **What changes:** the KL penalty β, from 0.04 (`grpo_v2`) to 0.1 (`grpo_v4`) and 0.2 (`grpo_v5`). Everything else is as in `grpo_v2`: the original reward (`grpo_v3`'s reward change made no significant difference), the same 2,000 positions, starting from `sft_v1`.
- **Why:** with β = 0.04 the KL divergence from SFT kept rising and the model still drifted towards pawn moves. Together with `grpo_v1` (β = 0) and `grpo_v2`, this gives four values of β.
- **Also reported:** the KL divergence and reward of all runs on one figure, and the share of pawn moves, captures, and checks.
- **Figures and tables** (`results/figures/grpo_v4/` and `grpo_v5/`, the same comparison in both): `kl_by_beta.png`, `reward_by_beta.png`, `puzzle_results` (with β), `paired_comparison`, `move_kinds` (table and figure), `puzzle_metrics.png`, `accuracy_by_band.png`, `answer_outcomes.png`.
- **Compute:** about 1.5–2 hours on one A10 for both runs.

### 15–16 · Combinations and the final comparison

Once 12–14 are in, notebook 15 combines the best SFT model with the best DPO and RL settings (for example SFT → DPO, SFT → RL, and SFT → DPO → RL); based on 09 and 10 so far, those settings are `dpo_v3`'s (DPO + SFT loss) and the best KL penalty from 10 and 14. If both more data and the legal-move list help, the SFT model for the combinations can use both. Notebook 16 puts every model and every evaluation in one final table and set of figures. These are planned for the revised paper (Dec 11).
