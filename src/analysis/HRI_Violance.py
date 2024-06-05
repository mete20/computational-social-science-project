import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.seasonal import seasonal_decompose
import os

output_dir = "output_HRI_Violence"
os.makedirs(output_dir, exist_ok=True)

# Load the data
human_rights_data = pd.read_csv('data/distribution-human-rights-index-vdem.csv')
violence_data = pd.read_csv('data/all_violence_data.csv')

# Filter violence data for 'Political violence' disorder_type
violence_data = violence_data[violence_data['disorder_type'] == 'Political violence']

# Ensure 'Year' is the correct type in violence data
violence_data['Year'] = pd.to_numeric(violence_data['year'], errors='coerce').astype('Int64')

# Drop rows with NaN values in 'Year' in violence data
violence_data = violence_data.dropna(subset=['Year'])

# Aggregate state violence data by year and country
violence_agg = violence_data.groupby(['country', 'Year']).size().reset_index(name='State_Violence')

# Filter Human Rights Index data for Turkey and Argentina
human_rights_turkey = human_rights_data[human_rights_data["Entity"].isin(["Turkey", "Turkiye"])]
human_rights_argentina = human_rights_data[human_rights_data["Entity"] == "Argentina"]

# Filter violence data for Turkey and Argentina
violence_turkey = violence_agg[violence_agg["country"] == "Turkey"]
violence_argentina = violence_agg[violence_agg["country"] == "Argentina"]

# Merge Human Rights Index data with violence data
turkey_merged_data = pd.merge(human_rights_turkey, violence_turkey, left_on='Year', right_on='Year', how='inner')
argentina_merged_data = pd.merge(human_rights_argentina, violence_argentina, left_on='Year', right_on='Year', how='inner')

# Rename columns for clarity
turkey_merged_data.rename(columns={"Civil liberties index (best estimate, aggregate: average)": "Human_Rights_Index"}, inplace=True)
argentina_merged_data.rename(columns={"Civil liberties index (best estimate, aggregate: average)": "Human_Rights_Index"}, inplace=True)

# Remove rows with missing values in Human Rights Index and State Violence
turkey_merged_data = turkey_merged_data.dropna(subset=['Human_Rights_Index', 'State_Violence'])
argentina_merged_data = argentina_merged_data.dropna(subset=['Human_Rights_Index', 'State_Violence'])

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
    
    corr, p_value = pearsonr(data['Human_Rights_Index'], data['State_Violence'])
    result = f"{country_name} Correlation Analysis\nPearson correlation coefficient: {corr}, P-value: {p_value}"
    print(result)
    
    with open(os.path.join(output_dir, f"{country_name}_correlation_analysis.txt"), "w") as f:
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
    y = data["State_Violence"]
    
    if X.empty or y.empty:
        print(f"No valid data points for regression analysis for {country_name}")
        return None
    
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    model = sm.OLS(y, X).fit()
   
    with open(os.path.join(output_dir, f"{country_name}_regression_analysis.txt"), "w") as f:
        f.write(model.summary().as_text())
    
    print(f"{country_name} Regression Analysis:")
    print(model.summary())
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
    plt.scatter(data["Human_Rights_Index"], data["State_Violence"], label="Data points")
    plt.plot(data["Human_Rights_Index"], model.predict(sm.add_constant(data["Human_Rights_Index"])), color='red', label="Regression line")
    plt.xlabel("Human Rights Index")
    plt.ylabel("State Violence")
    plt.title(f"State Violence vs Human Rights Index for {country_name}")
    plt.savefig(os.path.join(output_dir, f"{country_name}_state_violence_vs_human_rights_index.png"))
    plt.close()

# Plot data for Turkey
plot_data(turkey_merged_data, model_turkey, "Turkey")

# Plot data for Argentina
plot_data(argentina_merged_data, model_argentina, "Argentina")
