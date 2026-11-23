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
