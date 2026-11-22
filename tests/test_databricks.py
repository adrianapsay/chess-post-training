from chesslm.databricks import databricks_requirements, on_databricks, setup_databricks


def test_databricks_keeps_its_own_torch_and_notebook_packages():
    names = [line.split("==")[0] for line in databricks_requirements()]
    for kept in ["torch", "pandas", "matplotlib", "jupyter", "ipywidgets", "pytest"]:
        assert kept not in names
    assert not any(line.startswith("-") for line in databricks_requirements())  # no `-e .`


def test_training_and_grading_libraries_use_our_pins():
    lines = databricks_requirements()
    for pinned in ["transformers==", "trl==", "peft==", "datasets==", "accelerate==", "chess=="]:
        assert any(line.startswith(pinned) for line in lines)
    assert any(line.startswith("bitsandbytes==") and "sys_platform" in line for line in lines)


def test_setup_does_nothing_off_databricks():
    if not on_databricks():
        assert setup_databricks() is None


def test_stockfish_found_where_the_databricks_setup_puts_it(tmp_path, monkeypatch):
    import chesslm.engine as engine

    binary = tmp_path / "stockfish"
    binary.write_text("")
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    monkeypatch.setattr(engine.shutil, "which", lambda name: None)
    monkeypatch.setattr(engine, "STOCKFISH_BINARY", binary)
    assert engine.find_stockfish() == str(binary)


def test_gpu_check_explains_how_to_fix_missing_gpu(monkeypatch):
    import pytest
    import torch

    from chesslm.databricks import check_gpu

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Serverless GPU"):
        check_gpu()
