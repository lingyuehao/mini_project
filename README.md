[![main](https://github.com/lingyuehao/mini_project/actions/workflows/main.yml/badge.svg)](https://github.com/lingyuehao/mini_project/actions/workflows/main.yml)

# mini_project

## Setup Instructions：

1. Clone or download the repository to your local machine.
```python
git clone https://github.com/lingyuehao/mini_project.git
```
2. Install required dependencies:
```python
pip install pandas polars numpy scikit-learn seaborn matplotlib
```
3. Run the project
```python
python work.py
```

## Project goal
The purpose of this project is to analyze consumer behavior data using both Pandas and Polars. Specifically, the project aims to:
- Compare the performance of Pandas and Polars in data import, inspection, and grouping.
- Explore consumer insights by filtering and grouping data (e.g., purchase amount by demographics).
- Apply a machine learning model (Random Forest Regressor) to predict purchase amounts and evaluate model performance.
- Visualize findings with a pie chart of shopping device distribution and a bar chart of average customer satisfaction by income level.


## Data source

The project uses the Ecommerce Consumer Behavior Analysis Data, a comprehensive dataset designed for analyzing shopping trends and customer preferences. Think of this dataset as a profile of online shoppers. Each row is like a customer’s “digital footprint” — from who they are, how they shop, to why they make decisions.

Link to dataset: https://www.kaggle.com/datasets/salahuddinahmedshuvo/ecommerce-consumer-behavior-analysis-data/data

#### Who they are

Demographics: age, gender, income level, marital status, education, occupation, and location.

Purpose: helps us segment customers into meaningful groups

#### How they shop

Purchase activity: product category, purchase amount, frequency, and channel (online, in-store, or mixed).

Devices & payments: which device they used (smartphone, desktop, tablet) and payment method (credit card, PayPal, cash, etc.).

Shipping choices and payment frequency give a sense of convenience vs. cost priorities.

#### Why they buy (or don’t)

Loyalty: brand loyalty score, whether they’re in a loyalty program, and engagement with ads.

Product feedback: customer satisfaction (1–10), product ratings, and return rate.

Decision journey: time spent researching products, time taken to make a purchase decision, and whether it was impulsive or planned.

External influence: impact of social media, discount sensitivity, and preference for promotions.


## Data analysis steps
#### Step 1: Data Import & Performance Check

- Loaded the dataset ecb.csv using both Pandas and Polars.
- Compared execution times to evaluate performance differences between the two libraries.

#### Step 2: Data Inspection

- With Pandas: checked first rows, column info, summary statistics, and missing values.
- With Polars: examined schema, descriptive statistics, and null counts.
- Compared the inspection speed of Pandas vs. Polars.

#### Step 3: Filtering & Grouping

Case 1: Calculated average purchase amount grouped by Gender and Education Level.

Case 2: Focused on smartphone shoppers, grouped by Payment Method, and computed: 
        -  Average customer satisfaction
        -  Purchase count per payment method
Implemented in both Pandas and Polars, with timing comparisons.

#### Step 4: Machine Learning- Random Forest Regressor

- Selected features: product rating, return rate, purchase channel, discount sensitivity, brand loyalty, loyalty program membership, social media influence, purchase intent, age, and income level.
- Target variable: purchase amount.
- Encoded categorical features using LabelEncoder.
- Split data into training and testing sets (80/20 split).
- Trained a RandomForestRegressor with 300 estimators.
- Evaluated the model using RMSE and R², and extracted feature importance rankings.

#### Step 5: Visualization

Graph 1: Pie chart of shopping device distribution.

Graph 2: Bar chart of average customer satisfaction by income level.


## Outcomes
1. Performance: Pandas ran faster than Polars for both reading and grouping operations.
    
2. Data inspection: The dataset contained missing values mainly in Engagement_with_Ads and Social_Media_Influence. Pandas reported 503 nulls, while Polars reported none, showing differences in how the libraries treat nulls.

3. Grouping analysis:
- Average purchase amounts showed large variation by gender and education level, with Polygender + High School averaging close to 400, the highest among groups.
  
- Among smartphone shoppers, PayPal users had the highest satisfaction (~5.79), while cash users scored the lowest (~5.00).

4. Machine learning attempt:

Random Forest struggled to predict purchase amounts (RMSE ~143, R² < 0). Still, feature importances highlighted which factors mattered most: Age, Product Rating, Brand Loyalty, and Purchase Intent ranked highest, while loyalty program membership had little impact.

5. Visualizations:
- Device usage was fairly balanced across desktop, tablet, and smartphone.
- Middle-income customers reported slightly higher satisfaction than high-income ones, suggesting income is not the only driver of satisfaction.
