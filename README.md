# Machine Learning Laboratory Practicals (B.Tech CSE)

This repository contains Machine Learning laboratory practical assignments implemented in Python using Jupyter Notebooks / Google Colab.

---

## Repository Structure

```text
.
├── Assignment_01_LinearRegression/
│   ├── Assignment_01.ipynb
│   ├── data/
│   │   └── dataset.csv          # Medical Cost Personal Datasets (Insurance)
│   └── outputs/                 # Plots & correlation heatmap
│
├── Assignment_02_LogisticRegression/
│   ├── Assignment_02.ipynb
│   ├── data/
│   │   ├── binary_dataset.csv     # Diabetes Prediction Dataset
│   │   └── multiclass_dataset.csv # Mobile Price Classification Dataset
│   └── outputs/                 # Confusion matrices, ROC curves & boxplots
│
├── generate_notebook.py
├── generate_assignment_02.py
└── README.md
```

---

## Assignment Overview

### Assignment 01: Linear Regression
* **Objective:** Predict continuous medical insurance costs (`charges`) based on demographic and lifestyle attributes (`age`, `bmi`, `smoker`, etc.).
* **Dataset:** [Medical Cost Personal Datasets (Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance)
* **Metrics:** MAE, MSE, RMSE, $R^2$ Score.
* **Visualizations:** Feature Correlation Heatmap, Age vs. Charges Scatter Plot, Actual vs. Predicted Plot, Residual Distribution.

### Assignment 02: Logistic Regression (Binary & Multinomial)
* **Part A (Binary Logistic Regression):**
  * **Objective:** Predict patient diabetes status (`0` vs `1`).
  * **Dataset:** [Diabetes Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
  * **Metrics:** Accuracy, Precision, Recall, F1-Score, Classification Report, Confusion Matrix, ROC-AUC Curve.
* **Part B (Multinomial Logistic Regression):**
  * **Objective:** Classify mobile phones into 4 distinct price ranges (`0`: Low, `1`: Medium, `2`: High, `3`: Very High).
  * **Dataset:** [Mobile Price Classification (Kaggle)](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
  * **Metrics:** Accuracy, Weighted Precision, Recall, F1-Score, Classification Report, Multiclass Confusion Matrix.
* **Part C:** Binary vs. Multinomial Comparison Table.

---

## How to Run in Google Colab

1. Upload the project folder to Google Drive under `MyDrive/ML/`.
2. Open `Assignment_01.ipynb` or `Assignment_02.ipynb` in Google Colab.
3. Execute the Colab Drive mount cell right after the introduction:

```python
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Change working directory to assignment folder
os.chdir('/content/drive/MyDrive/ML/Assignment_01_LinearRegression')  # or Assignment_02_LogisticRegression
print("Current Working Directory:", os.getcwd())
```

4. Select **Runtime > Run all**.

---

## Requirements

* Python 3.8+
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`
