from fault_prediction.statistics import run_statistical_tests


def test_statistical_tests_return_p_values(balanced_sample):
    results = run_statistical_tests(balanced_sample)

    assert {"feature", "test", "p_value", "significant"}.issubset(results.columns)
    assert {"welch_t_test", "mann_whitney_u", "chi_square"}.issubset(set(results["test"]))
    assert results["p_value"].notna().any()
