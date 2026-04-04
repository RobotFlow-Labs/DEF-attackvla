from anima_def_attackvla.config import load_config


def test_load_default_config() -> None:
    cfg = load_config("configs/default.toml")
    assert cfg.project_name == "DEF-attackvla"
    assert cfg.poisoning_rate == 0.04
    assert cfg.benchmark_trials == 200
