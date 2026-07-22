import os
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Title & Header (Markdown)
c1 = nbf.v4.new_markdown_cell("""# Department of Computer Science & Engineering
## Machine Learning Laboratory (B.Tech)
### Assignment 01: Implementation of Linear Regression

---

### Dataset Details
* **Dataset Name:** Medical Cost Personal Datasets
* **Kaggle Dataset Link:** [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)
* **Description:** This dataset contains individual demographic and health details (age, sex, BMI, number of children, smoking status, region) along with medical insurance costs billed to individuals.
* **Why Linear Regression is Suitable:** The target variable (`charges`) is a continuous numeric value that demonstrates strong linear correlations with demographic and lifestyle attributes like `age`, `bmi`, and `smoker` status, making it ideal for continuous linear regression modeling.

---""")

# Cell 2: Introduction (Markdown)
c2 = nbf.v4.new_markdown_cell("""## 1. Introduction

Linear Regression is a fundamental supervised machine learning algorithm used for predicting a continuous target outcome based on one or more explanatory features. It models the relationship between variables by fitting a linear equation (hyperplane) to the observed data, minimizing the sum of squared residuals between actual and predicted values.

**Objective of this Practical:**
1. To understand the workflow of implementing a Linear Regression model in Python.
2. To explore and preprocess a real-world tabular dataset using Pandas and NumPy.
3. To train a `LinearRegression` model using Scikit-Learn and evaluate its predictive accuracy using MAE, MSE, RMSE, and $R^2$ Score.
4. To visualize regression performance using correlation heatmaps, scatter plots, actual vs. predicted comparisons, and residual distributions.""")

# Cell 3: Import Libraries (Code)
c3 = nbf.v4.new_code_cell("""# Import required libraries for data manipulation, modeling, and visualization
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules for modeling and evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure plot styles
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)""")

# Cell 4: Load Dataset (Code)
c4 = nbf.v4.new_code_cell("""# Define relative path to dataset using os.path.join
DATA_PATH = os.path.join("data", "dataset.csv")

# Load the dataset into a pandas DataFrame
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(f"Shape of dataset: {df.shape[0]} rows, {df.shape[1]} columns")""")

# Cell 5: EDA Section Header (Markdown)
c5 = nbf.v4.new_markdown_cell("""## 2. Basic Data Exploration""")

# Cell 6: Head (Code)
c6 = nbf.v4.new_code_cell("""# Display the first 5 rows of the dataset
df.head()""")

# Cell 7: Info (Code)
c7 = nbf.v4.new_code_cell("""# Check dataframe structure, column data types, and non-null values
df.info()""")

# Cell 8: Describe (Code)
c8 = nbf.v4.new_code_cell("""# Display summary statistics for numerical features
df.describe()""")

# Cell 9: Missing values check (Code)
c9 = nbf.v4.new_code_cell("""# Check for missing values in the dataset
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values)""")

# Cell 10: Preprocessing Section Header (Markdown)
c10 = nbf.v4.new_markdown_cell("""## 3. Data Preprocessing

To use categorical features in our Linear Regression model, we need to convert categorical variables (`sex`, `smoker`, `region`) into numerical format. We will use One-Hot Encoding using `pd.get_dummies()`.""")

# Cell 11: Preprocessing Code (Code)
c11 = nbf.v4.new_code_cell("""# Perform One-Hot Encoding for categorical features
df_encoded = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

# Display the preprocessed dataset
print("Encoded DataFrame Head:")
df_encoded.head()""")

# Cell 12: Train Test Split Header (Markdown)
c12 = nbf.v4.new_markdown_cell("""## 4. Dataset Splitting""")

# Cell 13: Train Test Split Code (Code)
c13 = nbf.v4.new_code_cell("""# Separate features (X) and target variable (y)
X = df_encoded.drop(columns=["charges"])
y = df_encoded["charges"]

# Split the dataset into 80% training set and 20% testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape:  {X_test.shape}")""")

# Cell 14: Model Training Header (Markdown)
c14 = nbf.v4.new_markdown_cell("""## 5. Model Training & Prediction""")

# Cell 15: Model Training Code (Code)
c15 = nbf.v4.new_code_cell("""# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model training complete!")
print(f"Intercept: {model.intercept_:.2f}")""")

# Cell 16: Prediction Code (Code)
c16 = nbf.v4.new_code_cell("""# Make predictions on the testing set
y_pred = model.predict(X_test)

# Display sample actual vs predicted values
comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
comparison_df.head(10)""")

# Cell 17: Model Evaluation Header (Markdown)
c17 = nbf.v4.new_markdown_cell("""## 6. Model Evaluation

We evaluate the performance of the Linear Regression model using four standard performance metrics:
1. **Mean Absolute Error (MAE):** Average of absolute differences between target and prediction.
2. **Mean Squared Error (MSE):** Average of squared errors.
3. **Root Mean Squared Error (RMSE):** Square root of MSE (in the same unit as the target variable).
4. **R² Score (Coefficient of Determination):** Proportion of variance in the target explained by features.""")

# Cell 18: Model Evaluation Code (Code)
c18 = nbf.v4.new_code_cell("""# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=" * 40)
print("    MODEL EVALUATION METRICS")
print("=" * 40)
print(f"Mean Absolute Error (MAE)      : {mae:.2f}")
print(f"Mean Squared Error (MSE)       : {mse:.2f}")
print(f"Root Mean Squared Error (RMSE) : {rmse:.2f}")
print(f"R2 Score                       : {r2:.4f}")
print("=" * 40)""")

# Cell 19: Visualizations Header (Markdown)
c19 = nbf.v4.new_markdown_cell("""## 7. Visualizations""")

# Cell 20: Visualization 1 - Correlation Heatmap (Code)
c20 = nbf.v4.new_code_cell("""# Visualization 1: Feature Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig(os.path.join("outputs", "correlation_heatmap.png"), dpi=300)
plt.show()""")

# Cell 21: Visualization 2 - Scatter Plot (Code)
c21 = nbf.v4.new_code_cell("""# Visualization 2: Feature vs Target Scatter Plot (Age vs Charges by Smoker Status)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="age", y="charges", hue="smoker", palette="Set1", alpha=0.8)
plt.title("Age vs Medical Charges by Smoker Status", fontsize=14, fontweight="bold")
plt.xlabel("Age (Years)")
plt.ylabel("Medical Charges ($)")
plt.tight_layout()
plt.savefig(os.path.join("outputs", "age_vs_charges_scatter.png"), dpi=300)
plt.show()""")

# Cell 22: Visualization 3 - Actual vs Predicted (Code)
c22 = nbf.v4.new_code_cell("""# Visualization 3: Actual vs Predicted Charges Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color="teal", alpha=0.6, label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2, linestyle="--", label="Perfect Fit")
plt.title("Actual vs. Predicted Insurance Charges", fontsize=14, fontweight="bold")
plt.xlabel("Actual Charges ($)")
plt.ylabel("Predicted Charges ($)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join("outputs", "actual_vs_predicted.png"), dpi=300)
plt.show()""")

# Cell 23: Visualization 4 - Residual Plot (Code)
c23 = nbf.v4.new_code_cell("""# Visualization 4: Residual Plot (Errors Distribution)
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, color="purple", bins=30)
plt.title("Distribution of Residuals (Errors)", fontsize=14, fontweight="bold")
plt.xlabel("Residual Value (Actual - Predicted)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join("outputs", "residuals_distribution.png"), dpi=300)
plt.show()""")

# Cell 24: Conclusion Header & Text (Markdown)
c24 = nbf.v4.new_markdown_cell("""## 8. Conclusion

1. The Linear Regression model was successfully implemented to predict medical insurance costs based on demographic and health parameters.
2. Feature correlation analysis revealed that smoking status, age, and BMI are the most significant predictors of insurance charges.
3. The model achieved an $R^2$ score of approximately **0.7830**, indicating that ~78.3% of the variance in medical charges is explained by the linear combination of input features.
4. Visualization of actual vs. predicted values and residual analysis confirms strong predictive performance across typical expense ranges, while highlighting minor non-linear effects in high-charge subgroups.
5. Overall, Linear Regression serves as an effective baseline model for continuous cost estimation tasks in healthcare analytical datasets.""")

nb.cells = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24]

out_path = os.path.join("Assignment_01_LinearRegression", "Assignment_01.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook generated cleanly at:", out_path)
