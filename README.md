# Customer Churn Prediction for a Power and Gas Utility
## 🎯Project Goal
The primary objective of this project is to develop a machine learning model that can accurately predict customer churn for a power and gas utility company. By identifying customers who are likely to switch to a competitor, the company can implement targeted retention strategies to reduce revenue loss and improve customer loyalty.

## 📂Project Structure
The project is organized into three main stages, each documented in a separate Jupyter Notebook:

1. Model.ipynb - Exploratory Data Analysis (EDA): This notebook is the starting point of our analysis. It involves loading the raw client and price datasets, performing descriptive statistical analysis, and creating various visualizations. The goal of the EDA is to gain a deep understanding of the data, identify patterns, uncover anomalies, and form initial hypotheses about the drivers of churn.

2. Feature Engineering.ipynb - Feature Creation and Transformation: Raw data is often not in the ideal format for machine learning models. In this notebook, we engineer new features from the existing data to better capture the underlying signals related to churn. Key activities include:

- Calculating price sensitivity features.

- Creating a tenure feature to measure the duration of a customer's relationship with the company.
  
- Transforming date-related columns into more informative formats.

- Encoding categorical variables into numerical representations.

- Applying data transformations to handle skewed distributions.

3. PREDICT CHURN.ipynb - Model Training and Evaluation: This is the final and most critical stage. We take the engineered features and build a predictive model. The process includes:

- Splitting the dataset into training and testing sets to ensure a robust evaluation.

- Training a Random Forest Classifier, a powerful ensemble learning method suitable for this type of classification problem.

- Evaluating the model's performance using key metrics like accuracy, precision, and recall.

- Analyzing the feature importances provided by the model to understand which factors are the most significant predictors of churn.

## 🚀How to Run This Project
### Prerequisites
To run these notebooks, you will need a Python environment with the following libraries installed:

- pandas

- numpy

- seaborn

- matplotlib

- scikit-learn

You can install these dependencies using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Execution Steps
- Explore the Data: Begin by running the cells in Model.ipynb to see the initial data analysis and visualizations.

- Engineer Features: Execute the Feature Engineering.ipynb notebook. This will process the raw data and generate a new file, data_for_predictions.csv, which contains the features needed for modeling.

- Predict Churn: Run the PREDICT CHURN.ipynb notebook. This will load the engineered data, train the Random Forest model, and output the performance evaluation and feature importance results.

## 📊Key Findings
The final Random Forest model demonstrated a strong ability to predict customer churn. The analysis of feature importances highlighted that customer tenure, net margin, and energy consumption levels were the most influential factors in determining whether a customer would churn. While price sensitivity was a contributing factor, it was not as dominant as the others.

## 💡Potential Next Steps
- Hyperparameter Tuning: Further optimize the Random Forest model by tuning its hyperparameters to potentially improve precision and recall.

- Explore Other Models: Experiment with other classification algorithms (e.g., Gradient Boosting, Logistic Regression, or Neural Networks) to see if they can achieve better performance.

- In-depth Customer Segmentation: Use the model's predictions and feature importances to segment customers and design tailored retention campaigns.

- Deployment: Package the final model into an API or a dashboard to enable real-time churn predictions for business users.
