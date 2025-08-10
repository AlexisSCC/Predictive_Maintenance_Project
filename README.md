# Predictive_Maintenance_Project
# Predicting Electric Vehicle Failures Using Machine Learning
This repository contains the MSc research project "Predicting Electric Vehicle Failures Using Machine Learning: A Comparative Study on Imbalanced Sensor Data" by Alexis R. Santa Cruz Crespo. The project evaluates and compares four machine learning models — Logistic Regression, Decision Tree, Random Forest, and XGBoost — to predict Diagnostic Trouble Codes (DTC) from EV sensor data. It also examines the impact of SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance.

# Repository: https://github.com/AlexisSCC/Predictive_Maintenance_Project

# Project Structure
Predictive_Maintenance_Project/
├── Data/ - Raw and processed datasets
├── Models/ - Saved trained models
├── BalancedData.ipynb - Model training & evaluation with SMOTE
├── ImbalancedData.ipynb - Model training & evaluation without SMOTE
├── ComparationModels.ipynb - Metrics and results comparison
├── README.md - Project documentation

# Abstract
Electric Vehicles (EVs) are increasingly popular but can suffer from unexpected technical failures affecting safety, reliability, and operational costs. Predictive maintenance using machine learning offers a proactive approach to detect failures early. This project uses EV sensor datasets from three user profiles (Rare, Moderate, Heavy) and compares the performance of multiple ML models on imbalanced and balanced datasets.

# Key findings 
Logistic Regression struggled on imbalanced data, missing most failures. Tree-based models (Decision Tree, Random Forest, XGBoost) performed better but risked overfitting. Applying SMOTE improved recall and overall performance. Random Forest and XGBoost achieved the most balanced results.

# Requirements
Install dependencies with:
pip install pandas numpy matplotlib.pyplot seaborn scikit-learn imbalanced-learn xgboost

# How to Run
Clone the repository:
git clone https://github.com/AlexisSCC/Predictive_Maintenance_Project.git

# Navigate into the folder:
cd Predictive_Maintenance_Project

# Open Jupyter Notebook:
jupyter notebook

# Run the notebooks in this order:

ImbalancedData.ipynb → Train models without SMOTE

BalancedData.ipynb → Train models with SMOTE

ComparationModels.ipynb → Compare model performances

# Results Summary
Model - Dataset - Best Metrics Highlights
Logistic Regression - SMOTE - Recall increased but many false positives
Decision Tree - SMOTE - Good balance precision/recall
Random Forest - SMOTE - High precision & recall
XGBoost - SMOTE - Stable performance

Future Work
Extend to multi-class prediction for specific failure types. Apply deep learning (e.g., LSTMs for time-series patterns). Use real-time EV fleet data for testing. Integrate predictions with maintenance scheduling tools.
