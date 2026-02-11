from predictive_analytics_system.hello import hello


def test_hello() -> None:
    assert hello() == "hello"
