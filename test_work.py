import numpy as np
import pandas as pd
import polars as pl
import work as W
import math


# Pandas Unit tests
class TestPandasUnit:
    def test_pd_loaded(self):
        """Pandas: dataset is loaded (non-empty) and required columns exist."""
        need = {
            "Purchase_Amount",
            "Gender",
            "Education_Level",
            "Device_Used_for_Shopping",
            "Payment_Method",
            "Customer_Satisfaction",
            "Customer_ID",
        }
        assert isinstance(W.df_pandas, pd.DataFrame) and len(W.df_pandas) > 0
        assert need.issubset(W.df_pandas.columns)

    def test_pd_purchase_amount_numeric_and_nonneg(self):
        """Pandas: Purchase_Amount is numeric and non-negative after cleaning."""
        s = W.df_pandas["Purchase_Amount"]
        assert pd.api.types.is_numeric_dtype(s)
        assert (s.dropna() >= 0).all()

    def test_pd_case1_shape_and_rows(self):
        """Pandas Case1: result has 'mean' column, MultiIndex, and row count matches unique (Gender, Education)."""
        g = W.gender_edu_purchase
        assert list(g.columns) == ["mean"]
        assert isinstance(g.index, pd.MultiIndex)
        assert pd.api.types.is_numeric_dtype(g["mean"])
        valid = W.df_pandas.dropna(subset=["Purchase_Amount"])
        uniq = valid[["Gender", "Education_Level"]].drop_duplicates()
        assert len(g) == len(uniq)

    def test_pd_case2_cols_and_ranges(self):
        """Pandas Case2: has expected columns and values are in valid ranges (count>0, 0<=satisfaction<=10)."""
        a = W.smartphone_stats_pd
        assert {"avg_satisfaction", "purchase_count"}.issubset(a.columns)
        assert (a["purchase_count"] > 0).all()
        assert (a["avg_satisfaction"].between(0, 10)).all()

    def test_pd_edge_empty_group_ok(self):
        """Pandas Edge: grouping an empty selection returns an empty DataFrame (no crash)."""
        empty = W.df_pandas[
            W.df_pandas["Device_Used_for_Shopping"] == "__NO_SUCH_DEVICE__"
        ]
        out = (
            empty.groupby("Payment_Method")
            .agg({"Customer_Satisfaction": "mean", "Customer_ID": "count"})
            .rename(
                columns={
                    "Customer_Satisfaction": "avg_satisfaction",
                    "Customer_ID": "purchase_count",
                }
            )
        )
        assert isinstance(out, pd.DataFrame) and out.empty


# Polar Unit tests
class TestPolarsUnit:
    def test_pl_loaded(self):
        """Polars: dataset is loaded (non-empty) and required columns exist."""
        need = {
            "Purchase_Amount",
            "Gender",
            "Education_Level",
            "Device_Used_for_Shopping",
            "Payment_Method",
            "Customer_Satisfaction",
            "Customer_ID",
        }
        assert isinstance(W.df_polars, pl.DataFrame) and W.df_polars.height > 0
        assert need.issubset(set(W.df_polars.columns))

    def test_pl_purchase_amount_float_and_nonneg(self):
        """Polars: Purchase_Amount is float dtype and non-negative when present."""
        assert W.df_polars.schema["Purchase_Amount"] in (pl.Float64, pl.Float32)
        s = W.df_polars.get_column("Purchase_Amount")
        if s.null_count() < s.len():
            assert s.drop_nulls().min() >= 0

    def test_pl_case1_shape(self):
        """Polars Case1: has grouping keys and 'mean' column."""
        gp = W.gender_edu_purchase_pl
        assert {"Gender", "Education_Level", "mean"}.issubset(set(gp.columns))

    def test_pl_case2_cols_and_types(self):
        """Polars Case2: expected columns; purchase_count is integer type and >=1."""
        b = W.smartphone_stats_pl
        assert {"Payment_Method", "avg_satisfaction", "purchase_count"}.issubset(
            set(b.columns)
        )
        int_ok = (
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        )
        assert b.schema["purchase_count"] in int_ok
        assert b.select(pl.col("purchase_count").min()).item() >= 1

    def test_pl_edge_empty_group_ok(self):
        """Polars Edge: grouping an empty selection returns an empty DataFrame (no crash)."""
        empty = W.df_polars.filter(
            pl.col("Device_Used_for_Shopping") == "__NO_SUCH_DEVICE__"
        )
        out = empty.group_by("Payment_Method").agg(
            [
                pl.col("Customer_Satisfaction").mean().alias("avg_satisfaction"),
                pl.count("Customer_ID").alias("purchase_count"),
            ]
        )
        assert isinstance(out, pl.DataFrame) and out.height == 0


# System Tests
class TestSystem:
    def test_case1_pd_vs_pl_agree(self):
        """System: Case1 means match between Pandas and Polars for every (Gender|Education) key."""
        pd_df = W.gender_edu_purchase.reset_index()
        pd_df["key"] = (
            pd_df["Gender"].astype(str) + "|" + pd_df["Education_Level"].astype(str)
        )
        a = dict(zip(pd_df["key"], pd_df["mean"]))
        pl_df = W.gender_edu_purchase_pl.with_columns(
            (
                pl.col("Gender").cast(pl.Utf8) + pl.lit("|") + pl.col("Education_Level").cast(pl.Utf8)
            ).alias("key")
        ).select(["key", "mean"])
        b = dict(
            zip(pl_df.get_column("key").to_list(), pl_df.get_column("mean").to_list())
        )
        assert set(a) == set(b)
        for k in a:
            assert np.isclose(a[k], b[k], atol=1e-8)

    def test_case2_pd_vs_pl_counts_agree(self):
        """System: Case2 purchase_count per payment method matches between Pandas and Polars."""
        a = dict(zip(W.smartphone_stats_pd.index.tolist(), W.smartphone_stats_pd["purchase_count"].tolist()))
        pl_keys = W.smartphone_stats_pl.get_column("Payment_Method").to_list()
        pl_vals = W.smartphone_stats_pl.get_column("purchase_count").to_list()
        b = dict(zip(pl_keys, pl_vals))
        assert set(a) == set(b)
        for k in a:
            assert a[k] == b[k]

    def test_case2_index_matches_unique_methods(self):
        """System: Pandas Case2 index equals unique Payment_Method among Smartphone shoppers."""
        subset = W.df_pandas[W.df_pandas["Device_Used_for_Shopping"] == "Smartphone"]
        assert set(W.smartphone_stats_pd.index) == set(subset["Payment_Method"].unique())


    def test_ml_table_numeric_and_no_nan(self):
        """System/ML: ML table has no NaNs and each feature is numeric or boolean."""
        cols = W.features + [W.target]
        assert not W.ml_dataset[cols].isna().any().any()
        for c in W.features:
            assert pd.api.types.is_numeric_dtype(
                W.ml_dataset[c]
            ) or pd.api.types.is_bool_dtype(W.ml_dataset[c])

    def test_ml_split_ratio_80_20(self):
        """System/ML: train/test split ratio is 80/20 (sklearn ceil behavior)."""
        n = len(W.ml_dataset)
        expected_test = math.ceil(n * 0.2)
        assert len(W.X_test) == expected_test
        assert len(W.X_train) == n - expected_test

    def test_ml_importances_and_metrics(self):
        """System/ML: RF importances length & ~sum==1; R² in [-1,1]."""
        assert len(W.rf.feature_importances_) == len(W.features)
        assert np.isclose(W.rf.feature_importances_.sum(), 1.0, atol=1e-6)
        assert -1.0 <= W.r2 <= 1.0

    def test_ml_seed_fixed(self):
        """System/ML: RF uses fixed random_state for reproducibility."""
        assert W.rf.get_params().get("random_state", None) == 17
