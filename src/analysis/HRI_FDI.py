import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.seasonal import seasonal_decompose
import os

output_dir = "output_HRI_FDI"
os.makedirs(output_dir, exist_ok=True)

# Load the data
fdi_data = pd.read_csv('data/fdi_data.csv')
human_rights_data = pd.read_csv('data/distribution-human-rights-index-vdem.csv')

# Identify and remove non-year columns
non_year_columns = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
year_columns = [col for col in fdi_data.columns if col not in non_year_columns]

# Reshape the FDI data from wide to long format
fdi_data_long = fdi_data.melt(id_vars=non_year_columns, value_vars=year_columns, var_name="Year", value_name="FDI")

# Convert the 'Year' column to integer
fdi_data_long['Year'] = pd.to_numeric(fdi_data_long['Year'], errors='coerce').astype('Int64')

# Filter FDI data for Turkey and Argentina
fdi_turkey = fdi_data_long[fdi_data_long["Country Name"].isin(["Turkey", "Turkiye"])]
fdi_argentina = fdi_data_long[fdi_data_long["Country Name"] == "Argentina"]

# Filter Human Rights Index data for Turkey and Argentina
human_rights_turkey = human_rights_data[human_rights_data["Entity"].isin(["Turkey", "Turkiye"])]
human_rights_argentina = human_rights_data[human_rights_data["Entity"] == "Argentina"]

# Merge FDI data with Human Rights Index data
turkey_merged_data = pd.merge(fdi_turkey, human_rights_turkey, left_on='Year', right_on='Year', how='inner')
argentina_merged_data = pd.merge(fdi_argentina, human_rights_argentina, left_on='Year', right_on='Year', how='inner')

# Rename columns for clarity
turkey_merged_data.rename(columns={"Civil liberties index (best estimate, aggregate: average)": "Human_Rights_Index"}, inplace=True)
argentina_merged_data.rename(columns={"Civil liberties index (best estimate, aggregate: average)": "Human_Rights_Index"}, inplace=True)

# Remove rows with missing values in FDI and Human Rights Index
turkey_merged_data = turkey_merged_data.dropna(subset=['FDI', 'Human_Rights_Index'])
argentina_merged_data = argentina_merged_data.dropna(subset=['FDI', 'Human_Rights_Index'])

# Check the merged data
print("Turkey Merged Data:")
print(turkey_merged_data.head())
print(turkey_merged_data.info())

print("\nArgentina Merged Data:")
print(argentina_merged_data.head())
print(argentina_merged_data.info())

# Perform correlation analysis
def correlation_analysis(data, country_name):
    if data.empty:
        print(f"No data available for correlation analysis for {country_name}")
        return None
    
    corr, p_value = pearsonr(data['Human_Rights_Index'], data['FDI'])
    
    result = f"Correlation between Human Rights Index and FDI for {country_name}: {corr:.4f}"
    print(result)
    with open(os.path.join(output_dir, f"{country_name}_correlation.txt"), 'w') as f:
        f.write(result) 
    return corr, p_value

print("Correlation Analysis for Turkey:")
corr_turkey, p_value_turkey = correlation_analysis(turkey_merged_data, "Turkey")

print("Correlation Analysis for Argentina:")
corr_argentina, p_value_argentina = correlation_analysis(argentina_merged_data, "Argentina")

# Perform regression analysis
def regression_analysis(data, country_name):
    if data.empty:
        print(f"No data available for regression analysis for {country_name}")
        return None
    
    X = data["Human_Rights_Index"]
    y = data["FDI"]
    
    if X.empty or y.empty:
        print(f"No valid data points for regression analysis for {country_name}")
        return None
    
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    model = sm.OLS(y, X).fit()
    
    result = model.summary()
    with open(os.path.join(output_dir, f"{country_name}_regression_analysis.txt"), 'w') as f:
        f.write(result.as_text())
    return model

print("Linear Regression for Turkey:")
model_turkey = regression_analysis(turkey_merged_data, "Turkey")

print("Linear Regression for Argentina:")
model_argentina = regression_analysis(argentina_merged_data, "Argentina")

# Plotting the data and the regression line
def plot_data(data, model, country_name):
    if data.empty or model is None:
        print(f"Cannot plot data for {country_name} due to insufficient data or model")
        return
    
    plt.figure(figsize=(10, 6))
    plt.scatter(data["Human_Rights_Index"], data["FDI"], label="Data points")
    plt.plot(data["Human_Rights_Index"], model.predict(sm.add_constant(data["Human_Rights_Index"])), color='red', label="Regression line")
    plt.xlabel("Human Rights Index")
    plt.ylabel("FDI (% of GDP)")
    plt.title(f"FDI vs Human Rights Index for {country_name}")    
    plt.savefig(os.path.join(output_dir, f"{country_name}_regression_plot.png"))
    plt.close()

# Plot data for Turkey
plot_data(turkey_merged_data, model_turkey, "Turkey")

# Plot data for Argentina
plot_data(argentina_merged_data, model_argentina, "Argentina")
