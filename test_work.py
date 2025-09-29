import math
import numpy as np
import pandas as pd
import polars as pl
import work as W


# Pandas Testing
class TestPandas:
    def test_pandas_loaded(self):
        assert isinstance(W.df_pandas, pd.DataFrame)
        assert len(W.df_pandas) > 0
        assert "Purchase_Amount" in W.df_pandas.columns
        assert "Gender" in W.df_pandas.columns

    def test_purchase_amount_cleaned(self):
        purchase_series = W.df_pandas["Purchase_Amount"]
        assert pd.api.types.is_numeric_dtype(purchase_series)
        for value in purchase_series.dropna():
            assert value >= 0

    def test_case1_groupby(self):
        gender_edu_table = W.gender_edu_purchase
        assert "mean" in gender_edu_table.columns
        assert isinstance(gender_edu_table.index, pd.MultiIndex)

    def test_case2_groupby(self):
        smartphone_stats = W.smartphone_stats_pd
        assert "avg_satisfaction" in smartphone_stats.columns
        assert "purchase_count" in smartphone_stats.columns
        assert (smartphone_stats["purchase_count"] > 0).all()


# Polars Testing
class TestPolars:
    def test_polars_loaded(self):
        assert isinstance(W.df_polars, pl.DataFrame)
        assert W.df_polars.height > 0
        assert "Purchase_Amount" in W.df_polars.columns

    def test_polars_purchase_amount(self):
        purchase_column = W.df_polars.get_column("Purchase_Amount")
        if purchase_column.null_count() < purchase_column.len():
            assert purchase_column.drop_nulls().min() >= 0

    def test_polars_case1(self):
        gender_edu_table_pl = W.gender_edu_purchase_pl
        assert "mean" in gender_edu_table_pl.columns


# System Testing
class TestSystem:
    def test_case1_pd_vs_pl(self):
        pd_group = W.gender_edu_purchase.reset_index()
        pd_group["key"] = (
            pd_group["Gender"].astype(str)
            + "|"
            + pd_group["Education_Level"].astype(str)
        )
        pd_dict = dict(zip(pd_group["key"], pd_group["mean"]))

        pl_group = W.gender_edu_purchase_pl.with_columns(
            (
                pl.col("Gender").cast(pl.Utf8)
                + pl.lit("|")
                + pl.col("Education_Level").cast(pl.Utf8)
            ).alias("key")
        ).select(["key", "mean"])
        pl_dict = dict(zip(pl_group["key"].to_list(), pl_group["mean"].to_list()))

        assert set(pd_dict.keys()) == set(pl_dict.keys())
        for k in pd_dict:
            assert np.isclose(pd_dict[k], pl_dict[k])

    def test_ml_table(self):
        model_data = W.ml_dataset
        for col in W.features:
            assert pd.api.types.is_numeric_dtype(
                model_data[col]
            ) or pd.api.types.is_bool_dtype(model_data[col])
        assert not model_data[W.features + [W.target]].isna().any().any()

    def test_ml_split(self):
        total_rows = len(W.ml_dataset)
        expected_test = math.ceil(total_rows * 0.2)
        assert len(W.X_test) == expected_test
        assert len(W.X_train) == total_rows - expected_test

    def test_rf_seed(self):
        assert W.rf.get_params().get("random_state") == 17
