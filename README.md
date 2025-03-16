
# Telco Customer Churn Prediction

## Overview
This project implements an end-to-end machine learning pipeline to predict customer churn for a telecommunications company. By analyzing historical customer data, the solution identifies individuals with a high likelihood of churning, enabling proactive retention strategies and reducing customer attrition.

## Features
- **Data Cleaning & Preprocessing:**  
  - Removal of non-informative identifiers (e.g., customerID).  
  - Conversion of data types (e.g., transforming 'TotalCharges' from string to numeric).  
  - Handling missing values and duplicates to maintain a robust dataset of 7,032 records.
  
- **Exploratory Data Analysis (EDA):**  
  - Visualization of feature distributions using pie charts, histograms, and pairplots.  
  - Correlation analysis among numerical variables with heatmaps.

- **Feature Engineering & Transformation:**  
  - Scaling of numerical features using StandardScaler.  
  - Categorical variable encoding via OneHotEncoder through a ColumnTransformer.

- **Handling Class Imbalance:**  
  - Applied Random Under Sampling to balance the churn classes (Yes/No).

- **Model Training & Hyperparameter Tuning:**  
  - Evaluated multiple classifiers (Logistic Regression, Random Forest, SVM, XGBoost, KNN, Decision Tree, Gradient Boosting, AdaBoost) using GridSearchCV.
  - Selected the AdaBoost Classifier based on its balanced performance.

- **Model Evaluation:**  
  - Achieved a test accuracy of 76.60% with the AdaBoost model.
  - Performance metrics include Precision (~74.1%), Recall (~80.9%), and F1 Score (~77.2%).

- **Project Functionality:**  
  - Utilizes historical customer data to predict churn probabilities, enabling targeted retention strategies.

## Technologies Used
- **Programming Language:** Python
- **Libraries:**  
  - Data Manipulation: Pandas, NumPy  
  - Visualization: Matplotlib, Seaborn  
  - Machine Learning: Scikit-Learn, XGBoost  
  - Imbalanced Data Handling: Imbalanced-learn  
  - Model Serialization: Pickle

## Dataset
The project uses a Telco Customer Churn dataset consisting of over 7,000 records. Key attributes include:
- **Demographics:** Gender, SeniorCitizen, Partner, Dependents.
- **Service Information:** Tenure, PhoneService, MultipleLines, InternetService.
- **Additional Services:** OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies.
- **Billing Details:** MonthlyCharges, TotalCharges.
- **Target Variable:** Churn (Yes/No).

## Project Structure
```
├── README.md
├── telco-customer-churn.csv
├── churn_prediction_pipeline.ipynb
├── models/
│   ├── lr_model.pkl
│   ├── ab_model.pkl
│   ├── column_transformations.pkl
│   └── label_encoder.pkl
```
- **README.md:** Project documentation.
- **telco-customer-churn.csv:** The dataset file.
- **churn_prediction_pipeline.ipynb:** Jupyter Notebook containing the complete pipeline.
- **models/:** Directory with serialized models and transformation objects for deployment.

## Installation
1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. **Install Dependencies:**
   - Create a virtual environment (optional):
     ```bash
     python -m venv env
     source env/bin/activate  # On Windows: env\Scripts\activate
     ```
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```
   - Alternatively, install libraries manually:
     ```bash
     pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
     ```

## Usage
1. **Data Preprocessing & EDA:**
   - Open the `churn_prediction_pipeline.ipynb` notebook.
   - Execute the cells to load and preprocess the dataset.
   - Review the EDA visualizations to understand feature distributions and relationships.

2. **Model Training & Evaluation:**
   - Split the data into training and testing sets.
   - Apply feature transformations using StandardScaler and OneHotEncoder.
   - Address class imbalance with Random Under Sampling.
   - Train multiple models with hyperparameter tuning via GridSearchCV.
   - Evaluate performance using accuracy, precision, recall, and F1 score.
   - Final model selection: AdaBoost Classifier with a test accuracy of 76.60%.

3. **Running the Pipeline:**
   - Execute the notebook sequentially to reproduce the pipeline.
   - The final models and transformers are saved in the `models/` directory for further use.

## Model Evaluation Metrics
- **AdaBoost Classifier:**  
  - **Test Accuracy:** 76.60%  
  - **Precision:** ~74.1%  
  - **Recall:** ~80.9%  
  - **F1 Score:** ~77.2%
