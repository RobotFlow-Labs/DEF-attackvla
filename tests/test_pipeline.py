import numpy as np

from anima_def_attackvla.config import load_config
from anima_def_attackvla.pipelines.inference import DefenseAwareInferencePipeline, InferenceRequest


def test_pipeline_allows_clean_request() -> None:
    pipeline = DefenseAwareInferencePipeline(load_config("configs/debug.toml"))
    image = np.zeros((224, 224, 3), dtype=np.float32)
    out = pipeline.predict(InferenceRequest("pick up the cup", image))
    assert out.allowed is True


def test_pipeline_blocks_trigger_request() -> None:
    pipeline = DefenseAwareInferencePipeline(load_config("configs/debug.toml"))
    image = np.zeros((224, 224, 3), dtype=np.float32)
    out = pipeline.predict(InferenceRequest("pick up *magic* cup", image))
    assert out.allowed is False
