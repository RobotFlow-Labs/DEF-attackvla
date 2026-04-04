from anima_def_attackvla.ros2_bridge import build_bridge_config


def test_bridge_topics() -> None:
    cfg = build_bridge_config()
    assert cfg.command_topic.startswith("/anima/")
    assert cfg.decision_topic.startswith("/anima/")
