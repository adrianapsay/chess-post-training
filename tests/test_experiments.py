import json

from chesslm import experiments


def write_record(folder, name, start, tokens, hours):
    record = {"config": {}, "start_adapter": start, "log": [],
              "compute": {"device": "NVIDIA A10G", "gpu_hours": hours, "tokens_seen": tokens}}
    (folder / f"{name}_training.json").write_text(json.dumps(record))


def test_training_stages_follow_the_start_adapters(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "RESULTS", tmp_path)
    monkeypatch.setattr(experiments, "REPO_RESULTS", tmp_path)
    write_record(tmp_path, "sft_v1", None, 1_900_000, 6.6)
    write_record(tmp_path, "dpo_v3", "/Workspace/Users/someone/repo/checkpoints/sft_v1", 870_000, 0.2)
    write_record(tmp_path, "sft_v1_dpo_rl", "D:\repo\checkpoints\dpo_v3\\", 1_500_000, 0.5)

    stages = experiments.training_stages("sft_v1_dpo_rl")
    assert [s["stage"] for s in stages] == ["sft_v1", "dpo_v3", "sft_v1_dpo_rl"]
    assert sum(s["tokens_seen"] for s in stages) == 4_270_000
    assert experiments.training_stages("base") == []  # not trained: no stages


def test_change_effect_measures_puzzles_and_positions(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "RESULTS", tmp_path)
    monkeypatch.setattr(experiments, "REPO_RESULTS", tmp_path)
    for name, correct in [("new", [1, 1, 0, 1]), ("old", [0, 1, 0, 0])]:
        rows = [{"puzzle_id": str(i), "correct": bool(c)} for i, c in enumerate(correct)]
        (tmp_path / f"{name}.json").write_text(json.dumps({"rows": rows}))
    positions = {"new": [("legal", 100, True), ("legal", 0, False)], "old": [("legal", 300, False), ("illegal", None, False)]}
    for name, answers in positions.items():
        rows = [{"game_id": "g", "ply": i, "outcome": o, "cp_loss": cp, "matches_stockfish": sf}
                for i, (o, cp, sf) in enumerate(answers)]
        (tmp_path / f"{name}_positions.json").write_text(json.dumps({"rows": rows}))
    effect = experiments.change_effect("new", "old")
    assert effect["solved only by this model"] == 2 and effect["solved only by the other"] == 0
    assert effect["puzzles"][0] == 50.0
    assert effect["stockfish"][0] == 50.0
    assert effect["cp_saved"][0] == 200.0 and effect["positions both legal"] == 1
    (tmp_path / "old_positions.json").unlink()  # a model not evaluated on positions yet
    assert experiments.change_effect("new", "old")["cp_saved"] == (None, None, None)
