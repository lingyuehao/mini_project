import time
import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt


def timed_execution(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"{func.__name__} completed in {end - start:.6f} seconds")
    return result


def check_missing(df):
    if isinstance(df, pd.DataFrame):
        print("\n--- Missing Values per Column ---")
        print(df.isnull().sum())
        print("\n--- Total Missing Values ---")
        print(df.isnull().sum().sum())
    elif isinstance(df, pl.DataFrame):
        print("\n--- Missing Values per Column ---")
        print(df.null_count())
        print("\n--- Total Missing Values ---")
        print(df.null_count().to_numpy().sum())


def evaluate_model(model, X_test, y_test, features):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("\n--- Model Evaluation ---")
    print("RMSE:", round(rmse, 3))
    print("R²:", round(r2, 3))
    importance = pd.DataFrame(
        {"Feature": features, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)
    print("\nFeature Importances:")
    print(importance)


# Import the Dataset using pandas and polars
input_file = "ecb.csv"
df_pandas = pd.read_csv(input_file)
df_polars = pl.read_csv(input_file)

df_pandas = timed_execution(pd.read_csv, input_file)
df_polars = timed_execution(pl.read_csv, input_file)

# Inspect the Dataset using pandas
print("\n--- First 6 rows using pandas---")
print(df_pandas.head(6))
print("\n--- Info ---")
print(df_pandas.info())
print("\n--- Summary Statistics ---")
print(df_pandas.describe())
check_missing(df_pandas)

# Inspect the Dataset using polars
print("\n--- First 6 rows using polars ---")
print(df_polars.head(6))
print("\n--- Schema ---")
print(df_polars.schema)
print("\n--- Summary Statistics ---")
print(df_polars.describe())
check_missing(df_polars)

# Case 1: Average Purchase Amount (by Gender and Education Level)
df_pandas["Purchase_Amount"] = pd.to_numeric(
    df_pandas["Purchase_Amount"].replace(r"[^\d.]", "", regex=True), errors="coerce"
)
gender_edu_purchase = timed_execution(
    lambda d: d.groupby(["Gender", "Education_Level"])["Purchase_Amount"].agg(["mean"]),
    df_pandas,
)
print(gender_edu_purchase)

df_polars = df_polars.with_columns(
    pl.col("Purchase_Amount")
    .str.replace_all(r"[^\d.]", "")
    .cast(pl.Float64)
    .alias("Purchase_Amount")
)
gender_edu_purchase_pl = timed_execution(
    lambda d: d.group_by(["Gender", "Education_Level"])
    .agg([pl.col("Purchase_Amount").mean().alias("mean")])
    .sort(["Gender", "Education_Level"]),
    df_polars,
)
print(gender_edu_purchase_pl)

# Case 2: Smartphone shoppers and Group by Payment_Method
subset_pd = df_pandas[df_pandas["Device_Used_for_Shopping"] == "Smartphone"]
smartphone_stats_pd = timed_execution(
    lambda s: s.groupby("Payment_Method")
    .agg({"Customer_Satisfaction": "mean", "Customer_ID": "count"})
    .rename(
        columns={
            "Customer_Satisfaction": "avg_satisfaction",
            "Customer_ID": "purchase_count",
        }
    ),
    subset_pd,
)
print(smartphone_stats_pd)

subset_pl = df_polars.filter(pl.col("Device_Used_for_Shopping") == "Smartphone")
smartphone_stats_pl = timed_execution(
    lambda s: s.group_by("Payment_Method").agg(
        [
            pl.col("Customer_Satisfaction").mean().alias("avg_satisfaction"),
            pl.count("Customer_ID").alias("purchase_count"),
        ]
    ),
    subset_pl,
)
print(smartphone_stats_pl)

# Explore a Machine Learning Algorithm (Random Forest)
features = [
    "Product_Rating",
    "Return_Rate",
    "Purchase_Channel",
    "Discount_Sensitivity",
    "Brand_Loyalty",
    "Customer_Loyalty_Program_Member",
    "Social_Media_Influence",
    "Purchase_Intent",
    "Age",
    "Income_Level",
]
target = "Purchase_Amount"

ml_dataset = df_pandas.copy()
ml_dataset = ml_dataset.dropna(subset=features + [target])

for col in features:
    if ml_dataset[col].dtype == "object" or ml_dataset[col].dtype.name == "category":
        ml_dataset[col] = LabelEncoder().fit_transform(ml_dataset[col])

X = ml_dataset[features]
y = ml_dataset[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=17
)

rf = RandomForestRegressor(n_estimators=300, random_state=17)
rf.fit(X_train, y_train)

evaluate_model(rf, X_test, y_test, features)

# Visualization
# Plot 1
df_pandas["Purchase_Amount"] = pd.to_numeric(
    df_pandas["Purchase_Amount"].replace(r"[^\d.]", "", regex=True), errors="coerce"
)
df_subset = df_pandas[df_pandas["Gender"].isin(["Male", "Female"])]

edu_order = ["High School", "Bachelor's", "Master's"]
heatmap_data = (
    df_subset.groupby(["Gender", "Education_Level"])["Purchase_Amount"]
    .mean()
    .unstack("Education_Level")[edu_order]
)

plt.figure(figsize=(8, 6))
ax = sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="Blues")
plt.title("Mean Purchase Amount by Gender and Education Level")
plt.xlabel("Education Level")
plt.ylabel("Gender")
cbar = ax.collections[0].colorbar
cbar.set_label("Purchase Amount ($)")
plt.tight_layout()
plt.show()


# Plot 2
avg_purchase = (
    df_pandas.groupby("Purchase_Category")["Purchase_Amount"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=avg_purchase,
    y="Purchase_Category",
    x="Purchase_Amount",
    hue="Purchase_Category",
    palette=sns.color_palette("Blues", n_colors=len(avg_purchase))[::-1],
    orient="h",
    legend=False,
)
plt.title("Average Purchase Amount by Category")
plt.xlabel("Average Purchase Amount ($)")
plt.ylabel("Purchase Category")
plt.tight_layout()
plt.show()
