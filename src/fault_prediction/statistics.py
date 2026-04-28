"""Statistical tests for feature relevance and target association."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind

from fault_prediction.config import QUALITY_COLUMN, RAW_SENSOR_COLUMNS, TARGET_COLUMN
from fault_prediction.data import validate_schema
from fault_prediction.features import engineer_features


def _interpret_p_value(p_value: float, alpha: float) -> str:
    if np.isnan(p_value):
        return "not enough data"
    if p_value < alpha:
        return "statistically significant association"
    return "no statistically significant association"


def numeric_feature_tests(
    df: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run Welch t-tests and Mann-Whitney U tests for numeric features by target."""

    validate_schema(df, require_target=True)
    engineered = engineer_features(
        df,
        drop_source_features=False,
        drop_identifier_columns=True,
        drop_target=False,
    )
    numeric_columns = [
        column
        for column in (*RAW_SENSOR_COLUMNS, "Temperature Gradient", "Power Indicator")
        if column in engineered.columns
    ]

    rows: list[dict[str, object]] = []
    for column in numeric_columns:
        clean = engineered[[column, TARGET_COLUMN]].dropna()
        normal = clean.loc[clean[TARGET_COLUMN] == 0, column]
        fault = clean.loc[clean[TARGET_COLUMN] == 1, column]

        if len(normal) < 2 or len(fault) < 2:
            test_results = [("welch_t_test", np.nan, np.nan), ("mann_whitney_u", np.nan, np.nan)]
        else:
            t_stat, t_p = ttest_ind(normal, fault, equal_var=False, nan_policy="omit")
            u_stat, u_p = mannwhitneyu(normal, fault, alternative="two-sided")
            test_results = [
                ("welch_t_test", float(t_stat), float(t_p)),
                ("mann_whitney_u", float(u_stat), float(u_p)),
            ]

        for test_name, statistic, p_value in test_results:
            rows.append(
                {
                    "feature": column,
                    "test": test_name,
                    "statistic": statistic,
                    "p_value": p_value,
                    "alpha": alpha,
                    "significant": bool(False if np.isnan(p_value) else p_value < alpha),
                    "interpretation": _interpret_p_value(float(p_value), alpha),
                }
            )

    return pd.DataFrame(rows)


def categorical_feature_tests(
    df: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run chi-square tests for categorical features by target."""

    validate_schema(df, require_target=True)
    rows: list[dict[str, object]] = []
    for column in [QUALITY_COLUMN]:
        clean = df[[column, TARGET_COLUMN]].dropna()
        contingency_table = pd.crosstab(clean[column], clean[TARGET_COLUMN])

        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            statistic = np.nan
            p_value = np.nan
            dof = np.nan
        else:
            statistic, p_value, dof, _ = chi2_contingency(contingency_table)

        rows.append(
            {
                "feature": column,
                "test": "chi_square",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "degrees_of_freedom": float(dof),
                "alpha": alpha,
                "significant": bool(False if np.isnan(p_value) else p_value < alpha),
                "interpretation": _interpret_p_value(float(p_value), alpha),
            }
        )

    return pd.DataFrame(rows)


def run_statistical_tests(df: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    """Run all configured p-value tests and sort by p-value."""

    results = pd.concat(
        [
            numeric_feature_tests(df, alpha=alpha),
            categorical_feature_tests(df, alpha=alpha),
        ],
        ignore_index=True,
        sort=False,
    )
    return results.sort_values(["p_value", "feature", "test"], na_position="last").reset_index(
        drop=True
    )
