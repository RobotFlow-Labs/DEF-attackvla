from anima_def_attackvla.pipelines.benchmark import AttackCounts, compute_metrics


def test_metric_computation() -> None:
    bundle = compute_metrics(
        AttackCounts(total=10, task_success=4, static_failures=3, targeted_success=5, clean_success=6)
    )
    assert abs(bundle.asr_u - 0.6) < 1e-9
    assert abs(bundle.asr_s - 0.3) < 1e-9
    assert abs(bundle.asr_t - 0.5) < 1e-9
    assert abs(bundle.cp - 0.6) < 1e-9
