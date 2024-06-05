import yfinance as yf
import pandas as pd

# Define the tickers for BIST 30 and Merval Index
tickers = ["XU030.IS", "^MERV"]

# Download historical market data for both indices
data = yf.download(tickers, start="2000-01-01", end="2020-12-31", interval="1mo")

# Only keep the 'Adj Close' columns
data = data['Adj Close']

# Rename columns for clarity
data.columns = ["BIST_30", "Merval_Index"]

# Reset the index to have the date as a column
data.reset_index(inplace=True)

# Convert the date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Aggregate to yearly data
data_yearly = data.resample('Y', on='Date').mean()

# Reset the index to have the date as a column
data_yearly.reset_index(inplace=True)

# Extract year from Date and replace Date with Year
data_yearly['Year'] = data_yearly['Date'].dt.year
data_yearly.drop(columns=['Date'], inplace=True)

# Save to CSV
data_yearly.to_csv("BIST_Merval_Stock_Data_Yearly.csv", index=False)

print(data_yearly.head())
