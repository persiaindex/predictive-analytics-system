from predictive_analytics_system.data import load_and_validate_raw_data

df = load_and_validate_raw_data()

# print(df.shape)


def test_dataLoad() -> None:
    assert df.shape == (7043, 50)
