[![main](https://github.com/lingyuehao/mini_project/actions/workflows/main.yml/badge.svg)](https://github.com/lingyuehao/mini_project/actions/workflows/main.yml)

# mini_project
## Project goal
I examine a dataset on consumer behavior to identify which product categories and demographic pairings influence purchase amount (AOV) and which dataframe engine provides the quickest analysis. I showcase the main trends using a gender×education heatmap and a ranked category bar chart, while also incorporating a small Random-Forest baseline to assess predictability.

## Repo Structure
```python
mini_project/
├── work.py                  # main analysis script
├── test_work.py             # pytest suite (unit + system tests)
├── ecb.csv                  # dataset
├── requirements.txt       
├── Dockerfile               
├── Makefile                  
├── .github/
│   └── workflows/
│       └── main.yml         # CI: run pytest on push
├── .devcontainer/
│   └── devcontainer.json    
├── .vscode/                 
├── .gitignore             
├── .dockerignore           
├── Plot_1.png               # figure: Gender × Education heatmap
├── Plot_2.png               # figure: Avg purchase by category
├── test_pass.png            # screenshot of passing tests  
├── workflows.png            # CI workflow screenshot
├── commit_difference_1.png  # commit illustration  
├── commit_difference_2.png  # commit illustration  
└── README.md                # project overview and instructions
```

## Setup Instructions and Use Guide：
You can run this project in three ways. Pick one.
### A) Local Python
1. Clone or download the repository to your local machine.
```python
git clone https://github.com/lingyuehao/mini_project.git
```
2. Install required dependencies:
```python
python -m pip install --upgrade pip
pip install -r requirements.txt
```
3. Run the project
```python
pytest -q (run tests)          
python work.py (run analysis)
```

### B) VS Code Dev Container
1. Install Docker Desktop and VS Code with the Dev Containers extension.

2. Open the repo folder in VS Code → bottom-right prompt “Reopen in Container”.
   
3. Wait for the build to finish (it installs everything from requirements.txt).

4. In the integrated terminal (inside the container):
```python
pytest -q (run tests)          
python work.py (run analysis)
```

### C）Docker
The Docker image is set to run tests by default.
```python
# build
docker build -t mini-project .

# run tests  
docker run --rm mini-project

# run the analysis instead of tests
docker run --rm mini-project python work.py
```


## Test Description and how to run tests
### Unit tests for Pandas

- Loading: df_pandas holds data.

- Cleaning: Purchase_Amount must be a numerical value and cannot be less than zero.

- Scenario 1 (Gender & Education): gender_edu_purchase includes an average column and has a MultiIndex structure.

- Case 2 (Smartphone & Payment_Method): smartphone_stats_pd contains avg_satisfaction and purchase_count, with all purchase_count > 0.

### Polars Unit Tests

- Loading: df_polars is a non-empty pl.DataFrame that includes Purchase_Amount.

- Cleaning: Values in Purchase_Amount that are not null are ≥ 0.

- Scenario 1 (Gender & Education): gender_edu_purchase_pl contains an average column.

### Testing of the system

- Cross-library equivalence (Case 1): Generate Gender|Education_Level keys and verify that both Pandas and Polars have identical key sets and similar np.isclose means.

- ML table integrity: In ml_dataset, each feature is either numeric or boolean, and both features and the target do not contain any NaN values.

- Division of train/test: With test_size=0.2, the count of test rows equals ceil(0.2 * N); the training rows consist of the others.

- RF reproducibility: RandomForestRegressor is configured with a random_state value of 17.

### How to run tests?
```bash
# Dev Container / Local
pip install -r requirements.txt
pytest -q test_work.py

# Docker:
docker build -t mini_project
docker run --rm mini_project
```

## Data source

The dataset utilized for this project focuses on Ecommerce Consumer Behavior Analysis and is built for carrying out trend and preference analysis in Infomediary Ecommerce. The dataset provides a comprehensive profile of online shoppers. Each record describes a customers’ “digital footprint”; detailing information on their demographics, shopping habits, and the rationale behind their purchasing decisions.

Link to dataset: https://www.kaggle.com/datasets/salahuddinahmedshuvo/ecommerce-consumer-behavior-analysis-data/data

#### Who they are

Demographics: age, gender, income level, marital status, education, occupation, and location.

Purpose: helps us segment customers into meaningful groups

#### How they shop

Purchase activity: product category, purchase amount, frequency, and channel (online, in-store, or mixed).

Devices & payments: which device they used (smartphone, desktop, tablet) and payment method (credit card, PayPal, cash, etc.).
 
#### Why they buy (or don’t)

Loyalty: brand loyalty score, whether they’re in a loyalty program, and engagement with ads.

Product feedback: customer satisfaction (1–10), product ratings, and return rate.

External influence: impact of social media, discount sensitivity, and preference for promotions.


## Data analysis steps
### Step 1 – Importing Data & Checking Performance

I compared data ingestion speeds by timing each read_csv call wrapped with a timer using both Pandas and Polars on ecb.csv. Each timed dataframe is used for everything that follows.

### Step 2 – Data Inspection.

I quickly inspected the data using Pandas by calling head, info, describe, and checking for missing values and using Polars by calling the schema, describe, and null counts. These inspection calls were not benchmarked; only the reads were.

### Step 3 – Filtering and Grouping.

I converted Purchase_Amount to numeric and across the Gender × Education grid, in each library and appropriate Polars and Pandas, compared group-by timings on summarizing average spend. I then focused specifically on smartphone shoppers and used average satisfaction and payment method to calculate satisfaction, number of purchases, and timed in Pandas and Polars for both these operations.

### Step 4 – Machine Learning Model.

I trained a Random Forest to predict purchase amount, using the data set’s behavioral signals and basic demographics. I dropped rows with missing values, label-encoded the categorical variables, and then split the data 80/20 with a fixed seed. In the end, I reported the RMSE and R² and feature importances to evaluate the model and determine the most influential factors.

### Step 5: Visualization

Graph 1: Heatmap — mean Purchase_Amount by Gender and Education_Level .

Graph 2: Ranked horizontal bar chart — Average Purchase Amount by Purchase Category.


## Outcomes
In general, Pandas outperformed Polars in my read and group-by operations, making it the more convenient engine for iteration in this scenario. The data verification revealed inconsistent null handling (Pandas identified around 503 missing values in several engagement fields, whereas Polars did not), which I will normalize prior to more intensive modeling. Aggregated findings indicate a genuine gender × education interplay—Female-High School ranks highest (~292), followed by Male-Bachelor’s (~287)—and among smartphone buyers, PayPal users indicate the highest satisfaction while cash users report the lowest. The Random Forest baseline poorly accounted for AOV (low R²), yet it still identified age, product rating, brand loyalty, and purchase intent as the most significant signals, while category spending is evidently focused (Software & Apps, Jewelry, Books). Next task: synchronize NA parsing and incorporate memory profiling, formally assess the interaction, apply log-AOV and more robust models (e.g., XGBoost with CV), and delve into payment × category to suggest minor, testable segments.
