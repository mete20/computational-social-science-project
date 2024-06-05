import requests
import os
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Your API key and email
api_key = os.getenv('API_KEY') 
email = os.getenv('EMAIL')
# Base URL for the ACLED API
base_url = 'https://api.acleddata.com/acled/read'

# Function to fetch data
def fetch_data(country, start_year, end_year):
    params = {
        'key': api_key,
        'email': email,
        'country': country,
        'year': f'{start_year}|{end_year}',
        'limit': 0  # Fetch all data
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json().get('data', [])
        return pd.DataFrame(data)
    else:
        print(f'Error: {response.status_code}')
        return pd.DataFrame()

# Fetch data for Turkey and Argentina

years = range(2000, 2021)  # From 2000 to 2020
df_turkey = fetch_data('Turkey', 2000, 2020)
df_argentina = fetch_data('Argentina', 2000, 2020)

# Save to CSV
df_turkey.to_csv('turkey_state_violence_data.csv', index=False)
df_argentina.to_csv('argentina_state_violence_data.csv', index=False)

