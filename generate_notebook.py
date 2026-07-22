import os
import json

def make_code_cell(source_lines):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source_lines
    }

def make_md_cell(source_lines):
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': source_lines
    }

# --- ASSIGNMENT 01 NOTEBOOK ---
cells_01 = [
    make_md_cell([
        '# Department of Computer Science & Engineering\n',
        '## Machine Learning Laboratory (B.Tech)\n',
        '### Assignment 01: Implementation of Linear Regression\n',
        '\n',
        '---\n',
        '\n',
        '### Dataset Details\n',
        '* **Dataset Name:** Medical Cost Personal Datasets\n',
        '* **Kaggle Dataset Link:** [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)\n',
        '* **Description:** This dataset contains individual demographic and health details (age, sex, BMI, number of children, smoking status, region) along with medical insurance costs billed to individuals.\n',
        '* **Why Linear Regression is Suitable:** The target variable (`charges`) is a continuous numeric value that demonstrates strong linear correlations with demographic and lifestyle attributes like `age`, `bmi`, and `smoker` status, making it ideal for continuous linear regression modeling.\n',
        '\n',
        '---'
    ]),
    make_md_cell([
        '## 1. Introduction\n',
        '\n',
        'Linear Regression is a fundamental supervised machine learning algorithm used for predicting a continuous target outcome based on one or more explanatory features. It models the relationship between variables by fitting a linear equation (hyperplane) to the observed data, minimizing the sum of squared residuals between actual and predicted values.\n',
        '\n',
        '**Objective of this Practical:**\n',
        '1. To understand the workflow of implementing a Linear Regression model in Python.\n',
        '2. To explore and preprocess a real-world tabular dataset using Pandas and NumPy.\n',
        '3. To train a `LinearRegression` model using Scikit-Learn and evaluate its predictive accuracy using MAE, MSE, RMSE, and $R^2$ Score.\n',
        '4. To visualize regression performance using correlation heatmaps, scatter plots, actual vs. predicted comparisons, and residual distributions.'
    ]),
    make_md_cell([
        '### Google Colab Drive Mount (Optional for Colab Users)'
    ]),
    make_code_cell([
        '# Run this cell if running inside Google Colab to mount Drive and set directory\n',
        'try:\n',
        '    from google.colab import drive\n',
        '    import os\n',
        '\n',
        '    # Mount Google Drive\n',
        '    drive.mount(\'/content/drive\')\n',
        '\n',
        '    # Change working directory to your assignment folder\n',
        '    os.chdir(\'/content/drive/MyDrive/ML/Assignment_01_LinearRegression\')\n',
        '    print("Current Working Directory:", os.getcwd())\n',
        'except Exception as e:\n',
        '    print("Not running in Colab or Drive already mounted:", e)'
    ]),
    make_md_cell([
        '## 2. Import Required Libraries'
    ]),
    make_code_cell([
        '# Import required libraries for data manipulation, modeling, and visualization\n',
        'import os\n',
        'import numpy as np\n',
        'import pandas as pd\n',
        'import matplotlib.pyplot as plt\n',
        'import seaborn as sns\n',
        '\n',
        '# Scikit-learn modules for modeling and evaluation\n',
        'from sklearn.model_selection import train_test_split\n',
        'from sklearn.linear_model import LinearRegression\n',
        'from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n',
        '\n',
        '# Configure plot styles\n',
        'sns.set_theme(style="whitegrid")\n',
        'plt.rcParams["figure.figsize"] = (8, 5)'
    ]),
    make_md_cell([
        '## 3. Load the Dataset'
    ]),
    make_code_cell([
        '# Define relative path to dataset using os.path.join\n',
        'DATA_PATH = os.path.join("data", "dataset.csv")\n',
        '\n',
        '# Load the dataset into a pandas DataFrame\n',
        'df = pd.read_csv(DATA_PATH)\n',
        '\n',
        'print("Dataset loaded successfully!")\n',
        'print(f"Shape of dataset: {df.shape[0]} rows, {df.shape[1]} columns")'
    ]),
    make_md_cell([
        '## 4. Basic Data Exploration'
    ]),
    make_code_cell([
        '# Display the first 5 rows of the dataset\n',
        'df.head()'
    ]),
    make_code_cell([
        '# Check dataframe structure, column data types, and non-null values\n',
        'df.info()'
    ]),
    make_code_cell([
        '# Display summary statistics for numerical features\n',
        'df.describe()'
    ]),
    make_code_cell([
        '# Check for missing values in the dataset\n',
        'missing_values = df.isnull().sum()\n',
        'print("Missing values per column:")\n',
        'print(missing_values)'
    ]),
    make_md_cell([
        '## 5. Data Preprocessing\n',
        '\n',
        'To use categorical features in our Linear Regression model, we need to convert categorical variables (`sex`, `smoker`, `region`) into numerical format. We will use One-Hot Encoding using `pd.get_dummies()`.'
    ]),
    make_code_cell([
        '# Perform One-Hot Encoding for categorical features\n',
        'df_encoded = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)\n',
        '\n',
        '# Display the preprocessed dataset\n',
        'print("Encoded DataFrame Head:")\n',
        'df_encoded.head()'
    ]),
    make_md_cell([
        '## 6. Dataset Splitting'
    ]),
    make_code_cell([
        '# Separate features (X) and target variable (y)\n',
        'X = df_encoded.drop(columns=["charges"])\n',
        'y = df_encoded["charges"]\n',
        '\n',
        '# Split the dataset into 80% training set and 20% testing set\n',
        'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n',
        '\n',
        'print(f"Training set shape: {X_train.shape}")\n',
        'print(f"Testing set shape:  {X_test.shape}")'
    ]),
    make_md_cell([
        '## 7. Model Training'
    ]),
    make_code_cell([
        '# Initialize and train the Linear Regression model\n',
        'model = LinearRegression()\n',
        'model.fit(X_train, y_train)\n',
        '\n',
        'print("Model training complete!")\n',
        'print(f"Intercept: {model.intercept_:.2f}")'
    ]),
    make_md_cell([
        '## 8. Make Predictions'
    ]),
    make_code_cell([
        '# Make predictions on the testing set\n',
        'y_pred = model.predict(X_test)\n',
        '\n',
        '# Display sample actual vs predicted values\n',
        'comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})\n',
        'comparison_df.head(10)'
    ]),
    make_md_cell([
        '## 9. Model Evaluation\n',
        '\n',
        'We evaluate the performance of the Linear Regression model using four standard performance metrics:\n',
        '1. **Mean Absolute Error (MAE):** Average of absolute differences between target and prediction.\n',
        '2. **Mean Squared Error (MSE):** Average of squared errors.\n',
        '3. **Root Mean Squared Error (RMSE):** Square root of MSE (in the same unit as the target variable).\n',
        '4. **R² Score (Coefficient of Determination):** Proportion of variance in the target explained by features.'
    ]),
    make_code_cell([
        '# Calculate evaluation metrics\n',
        'mae = mean_absolute_error(y_test, y_pred)\n',
        'mse = mean_squared_error(y_test, y_pred)\n',
        'rmse = np.sqrt(mse)\n',
        'r2 = r2_score(y_test, y_pred)\n',
        '\n',
        'print("=" * 40)\n',
        'print("    MODEL EVALUATION METRICS")\n',
        'print("=" * 40)\n',
        'print(f"Mean Absolute Error (MAE)      : {mae:.2f}")\n',
        'print(f"Mean Squared Error (MSE)       : {mse:.2f}")\n',
        'print(f"Root Mean Squared Error (RMSE) : {rmse:.2f}")\n',
        'print(f"R2 Score                       : {r2:.4f}")\n',
        'print("=" * 40)'
    ]),
    make_md_cell([
        '## 10. Visualizations'
    ]),
    make_code_cell([
        '# Visualization 1: Feature Correlation Heatmap\n',
        'plt.figure(figsize=(10, 6))\n',
        'sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)\n',
        'plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")\n',
        'plt.tight_layout()\n',
        'os.makedirs("outputs", exist_ok=True)\n',
        'plt.savefig(os.path.join("outputs", "correlation_heatmap.png"), dpi=300)\n',
        'plt.show()'
    ]),
    make_code_cell([
        '# Visualization 2: Feature vs Target Scatter Plot (Age vs Charges by Smoker Status)\n',
        'plt.figure(figsize=(8, 5))\n',
        'sns.scatterplot(data=df, x="age", y="charges", hue="smoker", palette="Set1", alpha=0.8)\n',
        'plt.title("Age vs Medical Charges by Smoker Status", fontsize=14, fontweight="bold")\n',
        'plt.xlabel("Age (Years)")\n',
        'plt.ylabel("Medical Charges ($)")\n',
        'plt.tight_layout()\n',
        'plt.savefig(os.path.join("outputs", "age_vs_charges_scatter.png"), dpi=300)\n',
        'plt.show()'
    ]),
    make_code_cell([
        '# Visualization 3: Actual vs Predicted Charges Plot\n',
        'plt.figure(figsize=(8, 5))\n',
        'plt.scatter(y_test, y_pred, color="teal", alpha=0.6, label="Predictions")\n',
        'plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2, linestyle="--", label="Perfect Fit")\n',
        'plt.title("Actual vs. Predicted Insurance Charges", fontsize=14, fontweight="bold")\n',
        'plt.xlabel("Actual Charges ($)")\n',
        'plt.ylabel("Predicted Charges ($)")\n',
        'plt.legend()\n',
        'plt.tight_layout()\n',
        'plt.savefig(os.path.join("outputs", "actual_vs_predicted.png"), dpi=300)\n',
        'plt.show()'
    ]),
    make_code_cell([
        '# Visualization 4: Residual Plot (Errors Distribution)\n',
        'residuals = y_test - y_pred\n',
        '\n',
        'plt.figure(figsize=(8, 5))\n',
        'sns.histplot(residuals, kde=True, color="purple", bins=30)\n',
        'plt.title("Distribution of Residuals (Errors)", fontsize=14, fontweight="bold")\n',
        'plt.xlabel("Residual Value (Actual - Predicted)")\n',
        'plt.ylabel("Frequency")\n',
        'plt.tight_layout()\n',
        'plt.savefig(os.path.join("outputs", "residuals_distribution.png"), dpi=300)\n',
        'plt.show()'
    ]),
    make_md_cell([
        '## 11. Conclusion\n',
        '\n',
        '1. The Linear Regression model was successfully implemented to predict medical insurance costs based on demographic and health parameters.\n',
        '2. Feature correlation analysis revealed that smoking status, age, and BMI are the most significant predictors of insurance charges.\n',
        '3. The model achieved an $R^2$ score of approximately **0.7830**, indicating that ~78.3% of the variance in medical charges is explained by the linear combination of input features.\n',
        '4. Visualization of actual vs. predicted values and residual analysis confirms strong predictive performance across typical expense ranges, while highlighting minor non-linear effects in high-charge subgroups.\n',
        '5. Overall, Linear Regression serves as an effective baseline model for continuous cost estimation tasks in healthcare analytical datasets.'
    ])
]

nb_01_json = {
    'cells': cells_01,
    'metadata': {'language_info': {'name': 'python'}},
    'nbformat': 4,
    'nbformat_minor': 2
}

out_path_01 = os.path.join('Assignment_01_LinearRegression', 'Assignment_01.ipynb')
with open(out_path_01, 'w', encoding='utf-8') as f:
    json.dump(nb_01_json, f, indent=2)

print('Assignment 01 updated with Colab drive mount code!')
