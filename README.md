# 🏦 AI-Powered Loan Approval Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📄 Overview
This project is a Machine Learning pipeline designed to automate the loan eligibility process. It analyzes applicant data to predict whether a loan should be **Approved** or **Rejected**. 

The system implements multiple classification algorithms to determine the most accurate model, performing comprehensive data preprocessing, feature scaling, and visualization of performance metrics.

## 📊 Features
* **Automated Preprocessing:** Handles missing values and encodes categorical variables.
* **Exploratory Data Analysis (EDA):** Visualizes class distribution and feature correlations.
* **Multi-Model Comparison:** Trains and evaluates four distinct algorithms:
    * K-Nearest Neighbors (KNN)
    * Logistic Regression
    * Decision Tree Classifier
    * Neural Network (MLPClassifier)
* **Adaptive Scaling:** Applies `MinMaxScaler` for distance-based algorithms (KNN) and `StandardScaler` for others.
* **Performance Visualization:** Generates Confusion Matrices, ROC Curves, and Accuracy comparisons.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (sklearn)

## 📂 Dataset
The project relies on `Loan Approval Dataset.csv`. 
* **Input Features:** Applicant demographics, credit history, income, etc.
* **Target Variable:** `loan_status` (Approved/Rejected).

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Danial-Dirar/CreditSense-AI.git
    cd CreditSense-AI
    ```

2.  **(Recommended) Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place your dataset:**
    Ensure `Loan Approval Dataset.csv` is in the root directory (it is already included in this repo).

5.  **Run the script:**
    ```bash
    python main.py
    ```

## 🧠 Methodology

### 1. Data Preprocessing
* **Missing Values:** Imputed using the `most_frequent` strategy.
* **Encoding:** `LabelEncoder` is used to convert categorical text data into numerical format.
* **Splitting:** Data is split 70% for training and 30% for testing using **stratified sampling** to maintain class balance.

### 2. Model Training
The system compares four models. Logic is applied to use specific scalers for specific models:
* **KNN:** Uses `MinMaxScaler` (sensitive to data magnitude).
* **Logistic Regression, Decision Tree, Neural Network:** Use `StandardScaler`.

### 3. Evaluation Metrics
The models are ranked based on:
* **Accuracy:** Overall correctness.
* **Precision & Recall:** To understand False Positives vs False Negatives.
* **ROC AUC Score:** To measure the model's ability to distinguish between classes.

## 📈 Results & Visualizations

Running the script automatically creates a **`figures/`** folder and saves every plot there as a `.png` file (the plots are also displayed on screen). The generated figures are:

| File | Description |
| --- | --- |
| `01_class_distribution.png` | **Class Distribution** — checks for dataset imbalance. |
| `02_correlation_heatmap.png` | **Correlation Heatmap** — identifies relationships between features. |
| `03_model_accuracy.png` | **Model Accuracy Comparison** — accuracy of all four models side-by-side. |
| `04_precision_recall.png` | **Precision vs Recall** — precision and recall for each model. |
| `05_confusion_matrix_<model>.png` | **Confusion Matrices** — one per model (KNN, Logistic Regression, Decision Tree, Neural Network). |
| `06_roc_curve.png` | **ROC Curve Comparison** — True Positive Rate vs False Positive Rate for all models. |

> The `figures/` folder is created automatically on each run, so you don't need to make it manually.

## 🔮 Future Improvements
* Implement Hyperparameter Tuning (GridSearchCV) to optimize model performance.
* Deploy the best model using Streamlit or Flask as a web app.
* Add Feature Importance analysis (SHAP values) to explain *why* a loan was rejected.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## 🎓 Credits
This project was originally built as a course submission for the Artificial Intelligence course, Section 5, by student IDs **24241314** and **22101931**. The main script was later renamed from `5_24241314_22101931.py` to `main.py` for clarity.

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
