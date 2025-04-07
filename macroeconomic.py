import os
import pandas as pd
import matplotlib.pyplot as plt

file_paths = {
    'GDP': 'HW6/GDP.csv',
    'GDP per Capita': 'HW6/gdp_pc.csv',
    'Inflation Rate': 'HW6/inflation_rates.csv',
    'Interest Rate': 'HW6/interest_rates.csv',
    'Public Debt': 'HW6/public_debt.csv',
    'S&P 500': 'HW6/sp_500.csv',
    'Unemployment Rate': 'HW6/unemp_rate.csv'
}

def visualize_last_two_years(file_path, title, date_col='observation_date', value_col=None):
    df = pd.read_csv(file_path)
    df[date_col] = pd.to_datetime(df[date_col])

    if not value_col:
        value_col = df.columns[1]

    max_date = df[date_col].max()
    two_years_ago = max_date - pd.DateOffset(years=2)
    recent_df = df[df[date_col] >= two_years_ago]

    plt.figure(figsize=(10, 5))
    plt.plot(recent_df[date_col], recent_df[value_col], marker='o')
    plt.title(f'{title} Over the Last 2 Years')
    plt.xlabel('Date')
    plt.ylabel(value_col)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Visualize each dataset
for title, path in file_paths.items():
    visualize_last_two_years(path, title)