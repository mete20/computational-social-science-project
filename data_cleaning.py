import pandas as pd

# Load data
df_turkey = pd.read_csv('turkey_state_violence_data.csv')
df_argentina = pd.read_csv('argentina_state_violence_data.csv')

# Data cleaning function
def clean_data(df):
    # Initial data preview
    print("Initial data preview:")
    print(df.head())
    print(f"Initial number of rows: {len(df)}")

    # Drop rows with critical missing values
    df_cleaned = df.dropna(subset=['event_date', 'event_type', 'country'])
    print("After dropping rows with critical missing values:")
    print(df_cleaned.head())
    print(f"Number of rows after dropping critical missing values: {len(df_cleaned)}")

    # Convert 'event_date' to datetime format
    df_cleaned['event_date'] = pd.to_datetime(df_cleaned['event_date'], errors='coerce')
    print("After converting 'event_date' to datetime:")
    print(df_cleaned[['event_date']].head())
    print(f"Number of rows after converting 'event_date': {len(df_cleaned)}")

    # Filter for relevant event types
    relevant_event_types = ['Violence against civilians', 'Protests', 'Battles']
    df_filtered = df_cleaned[df_cleaned['event_type'].isin(relevant_event_types)]
    print("After filtering for relevant event types:")
    print(df_filtered.head())
    print(f"Number of rows after filtering: {len(df_filtered)}")

    return df_filtered

# Clean data
df_turkey_cleaned = clean_data(df_turkey)
df_argentina_cleaned = clean_data(df_argentina)

# Save cleaned data
df_turkey_cleaned.to_csv('turkey_cleaned_data.csv', index=False)
df_argentina_cleaned.to_csv('argentina_cleaned_data.csv', index=False)

# Display the first few rows of the cleaned dataframe
print("Final cleaned data preview:")
print(df_turkey_cleaned.head())
print(df_argentina_cleaned.head())
print(f"Final number of rows: {len(df_turkey_cleaned)}")
print(f"Final number of rows: {len(df_argentina_cleaned)}")


