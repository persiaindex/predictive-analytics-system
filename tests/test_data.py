from predictive_analytics_system.data import load_and_validate_raw_data

df = load_and_validate_raw_data()
print(df.shape)
print(df["Churn"].value_counts()[0:2])


def test_dataLoad() -> None:
    assert df.shape == (7043, 21), "Unexpected shape of the dataset"
    assert df["Churn"].value_counts().to_dict() == {
        "No": 5174,
        "Yes": 1869,
    }, "Unexpected distribution of Churn values"
