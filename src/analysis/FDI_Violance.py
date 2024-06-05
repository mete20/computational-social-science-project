import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.seasonal import seasonal_decompose
import os

output_dir = "output_FDI_Violence"
os.makedirs(output_dir, exist_ok=True)

# Load the data
fdi_data = pd.read_csv('data/fdi_data.csv')
violence_data = pd.read_csv('data/all_violence_data.csv')

# Standardize country names in FDI data
fdi_data['Country Name'] = fdi_data['Country Name'].replace({
    'Turkiye': 'Turkey'
})

# Identify and remove non-year columns (these are the first 4 columns)
non_year_columns = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
year_columns = [col for col in fdi_data.columns if col not in non_year_columns and col.isdigit()]

# Reshape the FDI data from wide to long format
fdi_data_long = fdi_data.melt(id_vars=non_year_columns, value_vars=year_columns, 
                              var_name="Year", value_name="FDI")

# Convert the 'Year' column to integer
fdi_data_long['Year'] = pd.to_numeric(fdi_data_long['Year'], errors='coerce').astype('Int64')

# Drop rows with NaN values in 'Year'
fdi_data_long = fdi_data_long.dropna(subset=['Year'])

# Filter FDI data for the years 2016 to 2020
fdi_data_long = fdi_data_long[fdi_data_long['Year'].isin(range(2016, 2021))]

# Filter violence data for 'Political violence' disorder_type
violence_data = violence_data[violence_data['disorder_type'] == 'Political violence']

# Ensure 'Year' is the correct type in violence data
violence_data['Year'] = pd.to_numeric(violence_data['year'], errors='coerce').astype('Int64')

# Drop rows with NaN values in 'Year' in violence data
violence_data = violence_data.dropna(subset=['Year'])

# Aggregate state violence data by year and country
violence_agg = violence_data.groupby(['country', 'Year']).size().reset_index(name='State_Violence')

# Filter FDI data for Turkey and Argentina
fdi_turkey = fdi_data_long[fdi_data_long["Country Name"] == "Turkey"]
fdi_argentina = fdi_data_long[fdi_data_long["Country Name"] == "Argentina"]

# Filter violence data for Turkey and Argentina
violence_turkey = violence_agg[violence_agg["country"] == "Turkey"]
violence_argentina = violence_agg[violence_agg["country"] == "Argentina"]

# Merge FDI data with violence data
turkey_merged_data = pd.merge(fdi_turkey, violence_turkey, left_on='Year', right_on='Year', how='inner')
argentina_merged_data = pd.merge(fdi_argentina, violence_argentina, left_on='Year', right_on='Year', how='inner')

# Ensure there are no NaN values in FDI and State_Violence columns
turkey_merged_data = turkey_merged_data.dropna(subset=['FDI', 'State_Violence'])
argentina_merged_data = argentina_merged_data.dropna(subset=['FDI', 'State_Violence'])

# Perform correlation analysis
def correlation_analysis(data, country_name):
    if data.empty:
        print(f"No data available for correlation analysis for {country_name}")
        return None
    
    corr, p_value = pearsonr(data['State_Violence'], data['FDI'])
    
    result = f"Correlation between Human Rights Index and FDI for {country_name}: {corr:.4f}"
    print(result)
    with open(os.path.join(output_dir, f"{country_name}_correlation.txt"), 'w') as f:
        f.write(result) 
    return corr, p_value

print("Correlation Analysis for Turkey:")
corr_turkey, p_value_turkey = correlation_analysis(turkey_merged_data, "Turkey")

print("Correlation Analysis for Argentina:")
corr_argentina, p_value_argentina = correlation_analysis(argentina_merged_data, "Argentina")


# Linear Regression Analysis
def linear_regression_analysis(data, x_col, y_col, country_name):
    X = sm.add_constant(data[x_col])
    y = data[y_col]
    model = sm.OLS(y, X).fit()
    
    result = f"{country_name} Regression Analysis\n{model.summary().as_text()}"
    print(result)
    with open(os.path.join(output_dir, f"{data['Country Name'].iloc[0]}_regression.txt"), 'w') as f:
        f.write(result)
    return model

print("Linear Regression for Turkey:")
linear_regression_analysis(turkey_merged_data, 'FDI', 'State_Violence', 'Turkey')

print("Linear Regression for Argentina:")
linear_regression_analysis(argentina_merged_data, 'FDI', 'State_Violence', 'Argentina')

# Function to plot FDI and State Violence with correlation
def plot_correlation(data, country, corr):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='FDI', y='State_Violence', data=data)
    sns.regplot(x='FDI', y='State_Violence', data=data, scatter=False, color='red')
    plt.title(f'FDI and State Violence in {country}\nCorrelation: {corr:.2f}')
    plt.xlabel('FDI (% of GDP)')
    plt.ylabel('State Violence')
    plt.savefig(os.path.join(output_dir, f"{country}_correlation_plot.png"))
    plt.close()

# Visualization for Turkey
if not turkey_merged_data.empty:
    plot_correlation(turkey_merged_data, 'Turkey', corr_turkey)
else:
    print("Not enough data for Turkey to create visualization")

# Visualization for Argentina
if not argentina_merged_data.empty:
    plot_correlation(argentina_merged_data, 'Argentina', corr_argentina)
else:
    print("Not enough data for Argentina to create visualization")
