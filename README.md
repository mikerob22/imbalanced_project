# imbalanced_project

This project is a project to learn how to to go through a binary classification task using
a multiple logistic regression model, handle imbalanced data, and learn how to use poetry and a cookiecutter datascience template.

### ABOUT YOUR DATA

**Where did you get it?**
- The data used was downloaded from Kaggle using a dataset made to classify turnover within a company.
- https://www.kaggle.com/datasets/lnvardanyan/hr-analytics

**Do you think you have enough?**
- The raw data used had 15,000 records and 10 features. This is a sufficient amount of data for this task. 

**How much data did you start with (rows, columns, size) and how much did you use to model?**
- Raw data (15,000, 10)
- Training data (12000, 10)
- Testing data (3000, 10)

**How could your dataset be improved?**
- The dataset could be improved with more data along with better descriptions of the features in the dataset.


### FEATURE ENGINEERING AND DIMENSIONALITY STRATEGY

**Are you sure you have the right target variable? How did you confirm?**
- The classification task is to classify whether or not an employee left the company or not.
- The binary feature in the dataset that classified this was called 'left'

**What features are key? How do you know?**
- A key numeric feature identified in the correlation matrix was 'satisfaction_level' as it correlated with the target variable by -0.4.

**Did you reduce dimensionality? How did you decide to alter the size?**
- Dimensionality was actually increased due to applying OneHotEncoding to the categorical variables ```sales``` and ```salary```

### DATA PREP/PIPELINE

How do you know your data is appropriate and complete prior to modelling?
- The raw data was split into training (80% of raw data) and testing sets (20%)
- There were no missing values to remove or impute
- Two engineered features were added:
    - ```monthly_hours_per_proj``` : ```average_monthly_hours``` / ```number_project```
    - ```satis_eval_interaction```: ```satisfaction_level``` * ```last_evaluation```
- Categorical features ```sales``` and ```salary``` were encoded using OneHotEncoder()
- Numerical features were scaled using StandardScaler()
- After training a baseline model, this model was compared to a model trained by applied SMOTE to the training data

### MODELLING

**What was your model selection process?**
- Multiple Logistic Regression was selected because it is known to be a powerful model for binary classification tasks.
- Two models were compared:
    - Baseline model with Multiple Logisitic Regression
    - SMOTE model with Multiple Logistic Regression

**How did you go about the train / test split?**
- The raw data was split before any preprocessing was done 

**Are you sure there’s no data leakage? How do you know?**
- There is no data leakage because preprocessing was done after splitting data. The first time I went throught this project, my approach was incorrect because I preprocessed the data before splitting, do not do that.

**What are your thoughts on cross validation?**
- Cross validation was not applied in this project but I may apply it in the future.

**Was any hyperparameter tuning required? How did you tune?**
- No hyperparameter tuning was performed as it was outside the scope of what I went through this project for.

**What final model did you decide on? Why?**
- The best performing model was the Multiple Linear Regression with SMOTE applied to the training data.



