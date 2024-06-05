import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.seasonal import seasonal_decompose
import os

output_dir = "output_HRI_StockMarket"
os.makedirs(output_dir, exist_ok=True)

# Load the data
stock_data = pd.read_csv('data/BIST_Merval_Stock_Data_Yearly.csv')
human_rights_data = pd.read_csv('data/distribution-human-rights-index-vdem.csv')

# Filter Human Rights Index data for Turkey and Argentina
human_rights_turkey = human_rights_data[human_rights_data["Entity"].isin(["Turkey", "Turkiye"])]
human_rights_argentina = human_rights_data[human_rights_data["Entity"] == "Argentina"]

# Merge stock market data with Human Rights Index data
turkey_merged_data = pd.merge(human_rights_turkey, stock_data, left_on='Year', right_on='Year', how='inner')
argentina_merged_data = pd.merge(human_rights_argentina, stock_data, left_on='Year', right_on='Year', how='inner')

# Rename columns for clarity
turkey_merged_data.rename(columns={"Civil liberties index (best estimate, aggregate: average)": "Human_Rights_Index"}, inplace=True)
argentina_merged_data.rename(columns={"Civil liberties index (best estimate, aggregate: average)": "Human_Rights_Index"}, inplace=True)

# Remove rows with missing values in Human Rights Index and Stock Market Data
turkey_merged_data = turkey_merged_data.dropna(subset=['Human_Rights_Index', 'BIST_30'])
argentina_merged_data = argentina_merged_data.dropna(subset=['Human_Rights_Index', 'Merval_Index'])

# Check the merged data
print("Turkey Merged Data:")
print(turkey_merged_data.head())
print(turkey_merged_data.info())

print("\nArgentina Merged Data:")
print(argentina_merged_data.head())
print(argentina_merged_data.info())

# Perform correlation analysis
def correlation_analysis(data, x_col, y_col, country_name):
    if data.empty:
        print(f"No data available for correlation analysis for {country_name}")
        return None
    
    corr, p_value = pearsonr(data[x_col], data[y_col])
    
    result = f"Correlation between {x_col} and {y_col} for {country_name}: {corr:.4f}"
    print(result)
    with open(os.path.join(output_dir, f"{country_name}_correlation.txt"), 'w') as f:
        f.write(result)
    return corr, p_value

print("Correlation Analysis for Turkey:")
corr_turkey, p_value_turkey = correlation_analysis(turkey_merged_data, 'Human_Rights_Index', 'BIST_30', "Turkey")

print("Correlation Analysis for Argentina:")
corr_argentina, p_value_argentina = correlation_analysis(argentina_merged_data, 'Human_Rights_Index', 'Merval_Index', "Argentina")

# Perform regression analysis
def regression_analysis(data, x_col, y_col, country_name):
    if data.empty:
        print(f"No data available for regression analysis for {country_name}")
        return None
    
    X = data[x_col]
    y = data[y_col]
    
    if X.empty or y.empty:
        print(f"No valid data points for regression analysis for {country_name}")
        return None
    
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    model = sm.OLS(y, X).fit()
    
    result = model.summary()
    print(result)
    with open(os.path.join(output_dir, f"{country_name}_regression.txt"), 'w') as f:
        f.write(result.as_text())   

    return model

print("Linear Regression for Turkey:")
model_turkey = regression_analysis(turkey_merged_data, 'Human_Rights_Index', 'BIST_30', "Turkey")

print("Linear Regression for Argentina:")
model_argentina = regression_analysis(argentina_merged_data, 'Human_Rights_Index', 'Merval_Index', "Argentina")

# Plotting the data and the regression line
def plot_data(data, x_col, y_col, model, country_name):
    if data.empty or model is None:
        print(f"Cannot plot data for {country_name} due to insufficient data or model")
        return
    
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x_col], data[y_col], label="Data points")
    plt.plot(data[x_col], model.predict(sm.add_constant(data[x_col])), color='red', label="Regression line")
    plt.xlabel("Human Rights Index")
    plt.ylabel(f"{y_col} (Stock Market Index)")
    plt.title(f"{y_col} vs Human Rights Index for {country_name}")
    plt.savefig(os.path.join(output_dir, f"{country_name}_plot.png"))
    plt.close()


# Plot data for Turkey
plot_data(turkey_merged_data, 'Human_Rights_Index', 'BIST_30', model_turkey, "Turkey")

# Plot data for Argentina
plot_data(argentina_merged_data, 'Human_Rights_Index', 'Merval_Index', model_argentina, "Argentina")
