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

# Import the Dataset using pandas and polars
csv_file = "ecb.csv"
 
start = time.time()
df_pd = pd.read_csv(csv_file)
end = time.time()
print(f"Pandas Import completed in {end - start:.6f} seconds")

start = time.time()
df_pl = pl.read_csv(csv_file)
end = time.time()
print(f"Polars Import completed in {end - start:.6f} seconds")


# Inspect the Dataset using pandas and polars
start = time.time()
print("\n--- First 6 rows using pandas---")
print(df_pd.head(6))

print("\n--- Info ---")
print(df_pd.info())

print("\n--- Summary Statistics ---")
print(df_pd.describe())

print("\n--- Missing Values per Column ---")
print(df_pd.isnull().sum())

print("\n--- Total Missing Values ---")
print(df_pd.isnull().sum().sum())
end = time.time()
print(f"\nPandas Inspection completed in {end - start:.6f} seconds")


start = time.time()
print("\n--- First 6 rows using polars ---")
print(df_pl.head(6))

print("\n--- Schema ---")
print(df_pl.schema)

print("\n--- Summary Statistics ---")
print(df_pl.describe())

print("\n--- Missing Values per Column ---")
print(df_pl.null_count())

print("\n--- Total Missing Values ---")
print(df_pl.null_count().to_numpy().sum())
end = time.time()
print(f"\nPolars Inspection completed in {end - start:.6f} seconds")


# Basic Filtering and Grouping using pandas and polars

# Case 1: Average Purchase Amount (by Gender and Education Level)
start = time.time()
df_pd["Purchase_Amount"] = pd.to_numeric(
    df_pd["Purchase_Amount"].replace(r"[^\d.]", "", regex=True),
    errors="coerce"
)
gender_edu_purchase = df_pd.groupby(["Gender", "Education_Level"])["Purchase_Amount"].agg(["mean"])
print(gender_edu_purchase)
end = time.time()
print(f"\nPandas completed case 1 in {end - start:.6f} seconds")


start = time.time()
df_pl = df_pl.with_columns(
    pl.col("Purchase_Amount")
      .str.replace_all(r"[^\d.]", "")    
      .cast(pl.Float64)
      .alias("Purchase_Amount")
)

gender_edu_purchase_pl = (
    df_pl.group_by(["Gender", "Education_Level"])
         .agg([
             pl.col("Purchase_Amount").mean().alias("mean")
         ])
         .sort(["Gender", "Education_Level"])    
)
print(gender_edu_purchase_pl)
end = time.time()
print(f"\nPolars completed case 1in {end - start:.6f} seconds")

# Case 2: Smartphone shoppers and Group by Payment_Metho (using pandas and polars)
start = time.time()
subset_pd = df_pd[df_pd["Device_Used_for_Shopping"] == "Smartphone"]
case2_pd = (
    subset_pd.groupby("Payment_Method")
             .agg({"Customer_Satisfaction": "mean", "Customer_ID": "count"})
             .rename(columns={"Customer_Satisfaction": "avg_satisfaction", 
                              "Customer_ID": "purchase_count"})
)
print(case2_pd)
end = time.time()
print(f"Pandas completed Case 2 in {end - start:.6f} seconds")


start = time.time()
subset_pl = df_pl.filter(pl.col("Device_Used_for_Shopping") == "Smartphone")
case2_pl = (
    subset_pl.group_by("Payment_Method")
             .agg([
                 pl.col("Customer_Satisfaction").mean().alias("avg_satisfaction"),
                 pl.count("Customer_ID").alias("purchase_count")
             ])
)
print(case2_pl)
end = time.time()
print(f"Polars completed case 2 in {end - start:.6f} seconds")


# Explore a Machine Learning Algorithm (Random Forest)
features = [
    "Product_Rating", "Return_Rate",
    "Purchase_Channel", "Discount_Sensitivity",
    "Brand_Loyalty", "Customer_Loyalty_Program_Member",
    "Social_Media_Influence", "Purchase_Intent",
    "Age", "Income_Level"
]
target = "Purchase_Amount"

 
df_ml = df_pd.copy()
df_ml = df_ml.dropna(subset=features + [target])

for col in features:
    if df_ml[col].dtype == "object" or df_ml[col].dtype.name == "category":
        df_ml[col] = LabelEncoder().fit_transform(df_ml[col])

X = df_ml[features]
y = df_ml[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=17
)

rf = RandomForestRegressor(n_estimators=300, random_state=17)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Random Forest Results---")
print("RMSE:", round(rmse, 3))
print("R²:", round(r2, 3))

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importances:")
print(importance)


# Visualization

#Graph 1
device_counts = df_pd["Device_Used_for_Shopping"].value_counts()
plt.figure(figsize=(10,6))
plt.pie(device_counts, labels=device_counts.index)
plt.title("Shopping Device Distribution")
plt.show()

#Graph 2
plt.figure(figsize=(10,6))
sns.barplot(data=df_pd, x="Income_Level", y="Customer_Satisfaction", ci=None, palette="pastel")
plt.title("Average Customer Satisfaction by Income Level")
plt.xlabel("Income Level")
plt.ylabel("Average Satisfaction")
plt.ylim(0, 10)   
plt.show()