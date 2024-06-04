import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load cleaned data
df_turkey = pd.read_csv('turkey_cleaned_data.csv')
df_argentina = pd.read_csv('argentina_cleaned_data.csv')

# Ensure the event_date is in datetime format
df_turkey['event_date'] = pd.to_datetime(df_turkey['event_date'], errors='coerce')
df_argentina['event_date'] = pd.to_datetime(df_argentina['event_date'], errors='coerce')

# Extract year from event_date for regression analysis
df_turkey['year'] = df_turkey['event_date'].dt.year
df_argentina['year'] = df_argentina['event_date'].dt.year

# Ensure 'fatalities' column is numeric
df_turkey['fatalities'] = pd.to_numeric(df_turkey['fatalities'], errors='coerce')
df_argentina['fatalities'] = pd.to_numeric(df_argentina['fatalities'], errors='coerce')

# Log transform 'fatalities' to handle skewness
df_turkey['log_fatalities'] = np.log1p(df_turkey['fatalities'])
df_argentina['log_fatalities'] = np.log1p(df_argentina['fatalities'])

# Include additional predictors and feature engineering
df_turkey['interaction_year'] = df_turkey['interaction'] * df_turkey['year']
df_argentina['interaction_year'] = df_argentina['interaction'] * df_argentina['year']

# Example function for linear regression analysis with additional variables
def perform_regression(df, dependent_var, independent_vars):
    # Ensure all data is numeric
    df = df.dropna(subset=[dependent_var] + independent_vars)
    
    if len(df) == 0:
        print("No data available for regression analysis after cleaning.")
        return None

    # Check for multicollinearity
    corr_matrix = df[independent_vars].corr()
    print("Correlation Matrix:")
    print(corr_matrix)
    
    X = df[independent_vars]
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    y = df[dependent_var]

    model = sm.OLS(y, X).fit()
    return model.summary()

# Define dependent and independent variables
dependent_var = 'log_fatalities'
independent_vars = ['year', 'time_precision', 'interaction_year']

# Perform regression analysis for Turkey
summary_turkey = perform_regression(df_turkey, dependent_var, independent_vars)
if summary_turkey:
    print("Regression Summary for Turkey:")
    print(summary_turkey)

# Perform regression analysis for Argentina
summary_argentina = perform_regression(df_argentina, dependent_var, independent_vars)
if summary_argentina:
    print("Regression Summary for Argentina:")
    print(summary_argentina)

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load cleaned data
df_turkey = pd.read_csv('turkey_cleaned_data.csv')
df_argentina = pd.read_csv('argentina_cleaned_data.csv')

# Ensure the event_date is in datetime format
df_turkey['event_date'] = pd.to_datetime(df_turkey['event_date'], errors='coerce')
df_argentina['event_date'] = pd.to_datetime(df_argentina['event_date'], errors='coerce')

# Extract year from event_date for regression analysis
df_turkey['year'] = df_turkey['event_date'].dt.year
df_argentina['year'] = df_argentina['event_date'].dt.year

# Ensure 'fatalities' column is numeric
df_turkey['fatalities'] = pd.to_numeric(df_turkey['fatalities'], errors='coerce')
df_argentina['fatalities'] = pd.to_numeric(df_argentina['fatalities'], errors='coerce')

# Log transform 'fatalities' to handle skewness
df_turkey['log_fatalities'] = np.log1p(df_turkey['fatalities'])
df_argentina['log_fatalities'] = np.log1p(df_argentina['fatalities'])

# Include additional predictors and feature engineering
df_turkey['interaction_year'] = df_turkey['interaction'] * df_turkey['year']
df_argentina['interaction_year'] = df_argentina['interaction'] * df_argentina['year']

# Define dependent and independent variables
dependent_var = 'log_fatalities'
independent_vars = ['year', 'time_precision', 'interaction', 'interaction_year']

# Function for Ridge regression analysis
def perform_ridge_regression(df, dependent_var, independent_vars):
    df = df.dropna(subset=[dependent_var] + independent_vars)
    if len(df) == 0:
        print("No data available for regression analysis after cleaning.")
        return None
    
    X = df[independent_vars]
    y = df[dependent_var]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create Ridge regression model
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = ridge_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

# Perform Ridge regression for Turkey
print("Ridge Regression for Turkey:")
perform_ridge_regression(df_turkey, dependent_var, independent_vars)

# Perform Ridge regression for Argentina
print("Ridge Regression for Argentina:")
perform_ridge_regression(df_argentina, dependent_var, independent_vars)

