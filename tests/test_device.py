from anima_def_attackvla.device import build_runtime_context


def test_runtime_context_mlx() -> None:
    ctx = build_runtime_context("mlx")
    assert ctx.backend == "mlx"


def test_runtime_context_cuda() -> None:
    ctx = build_runtime_context("cuda")
    assert ctx.backend == "cuda"
