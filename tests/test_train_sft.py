import chess

from chesslm.train_sft import to_sft_example

ROW = {"fen": chess.STARTING_FEN, "history_uci": "", "history_san": "", "move_uci": "g1f3", "move_san": "Nf3"}


def test_sft_example_uses_the_evaluation_prompt_and_the_move_played():
    example = to_sft_example(ROW, {"notation": "uci", "history_moves": 0, "show_legal_moves": False})
    assert example["prompt"][0]["role"] == "user"
    assert chess.STARTING_FEN in example["prompt"][0]["content"]
    assert example["completion"] == [{"role": "assistant", "content": "g1f3"}]


def test_sft_example_answer_follows_the_notation_setting():
    example = to_sft_example(ROW, {"notation": "san", "history_moves": 0, "show_legal_moves": False})
    assert example["completion"][0]["content"] == "Nf3"


def test_sft_example_can_learn_stockfish_move_instead():
    row = {**ROW, "best_uci": "e2e4", "best_san": "e4"}
    uci = to_sft_example(row, {"notation": "uci", "history_moves": 0, "show_legal_moves": False}, answer="stockfish")
    san = to_sft_example(row, {"notation": "san", "history_moves": 0, "show_legal_moves": False}, answer="stockfish")
    assert uci["completion"][0]["content"] == "e2e4"
    assert san["completion"][0]["content"] == "e4"


def test_same_seed_gives_same_positions_in_same_order():
    import pandas as pd

    from chesslm.train_sft import make_dataset, sample_examples, to_dataset

    positions = pd.DataFrame([{**ROW, "fen": chess.STARTING_FEN, "ply": i} for i in range(50)])
    prompt = {"notation": "uci", "history_moves": 0, "show_legal_moves": False}
    sample = sample_examples(positions, 10, seed=42)
    assert list(sample.ply) == list(positions.sample(n=10, random_state=42).ply)  # what sft_v1 used
    assert to_dataset(sample, prompt).to_list() == make_dataset(positions, 10, prompt, seed=42).to_list()


def test_copies_are_saved_after_the_requested_numbers_of_examples(tmp_path):
    from types import SimpleNamespace

    from chesslm.train_sft import SaveCopies, copy_dir

    saved = []
    model = SimpleNamespace(save_pretrained=saved.append)
    callback = SaveCopies(tmp_path / "sft_100k", save_at=[20000, 40000], examples_per_step=32)
    for step in range(1, 1300):
        callback.on_step_end(None, SimpleNamespace(global_step=step), None, model=model)
    assert saved == [str(tmp_path / "sft_100k_at_20k"), str(tmp_path / "sft_100k_at_40k")]
    assert copy_dir(tmp_path / "sft_100k", 20000) == tmp_path / "sft_100k_at_20k"
