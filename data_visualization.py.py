import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
df_turkey = pd.read_csv('turkey_cleaned_data.csv')
df_argentina = pd.read_csv('argentina_cleaned_data.csv')

# Plotting function
def plot_data(df, country):
    df.groupby(df['event_date'].dt.year).size().plot(kind='bar', figsize=(10, 5))
    plt.title(f'Number of Violent Events Over Time in {country}')
    plt.xlabel('Year')
    plt.ylabel('Number of Events')
    plt.show()

# Plot data for Turkey
plot_data(df_turkey, 'Turkey')

# Plot data for Argentina
plot_data(df_argentina, 'Argentina')
