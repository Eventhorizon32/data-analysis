import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
file_path = r'Oil Consumption by Country 1965 to 2023.csv'
dataframe = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(dataframe.head())

# Display the columns of the dataframe
print(dataframe.columns)

# Display the shape of the dataframe
print(dataframe.shape)

# Display information about the dataframe
print(dataframe.info())

# Display the count of missing values in each column
print(dataframe.isnull().sum())

# Fill missing values with 0
dataframe.fillna(0, inplace=True)

# Identify the top 10 oil-consuming countries in 2020
top_10_countries = dataframe[['Entity', '2020']].sort_values(by='2020', ascending=False).head(10)['Entity']

# Plot oil consumption trends for the top 10 countries
plt.figure(figsize=(14, 8))
for country in top_10_countries:
    plt.plot(dataframe.columns[1:], dataframe[dataframe['Entity'] == country].iloc[:, 1:].values.flatten(), label=country)

plt.xlabel('Year')
plt.ylabel('Oil Consumption')
plt.title('Oil Consumption Trends for Top 10 Countries (1965-2023)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate year-over-year growth rates for the top 10 countries
growth_rates = dataframe[dataframe['Entity'].isin(top_10_countries)].copy()
growth_rates.set_index('Entity', inplace=True)

# Calculate growth rates
growth_rates = growth_rates.pct_change(axis=1).dropna(axis=1) * 100

# Plot year-over-year growth rates for the top 10 countries
plt.figure(figsize=(14, 8))
for country in top_10_countries:
    plt.plot(growth_rates.columns, growth_rates.loc[country], label=country)

plt.xlabel('Year')
plt.ylabel('Year-over-Year Growth Rate (%)')
plt.title('Year-over-Year Growth Rates for Top 10 Countries (1966-2023)')
plt.legend()
plt.grid(True)
plt.show()

# Correlation matrix
correlation_matrix = dataframe.drop(columns=['Entity']).corr()

# Plot heatmap of the correlation matrix
plt.figure(figsize=(14, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Oil Consumption Over the Years')
plt.show()

# Select data for a specific country for forecasting
country_data = dataframe[dataframe['Entity'] == 'United States'].drop(columns=['Entity']).T
country_data.columns = ['Oil_Consumption']
country_data.index = pd.to_datetime(country_data.index, format='%Y')
country_data.index.freq = 'YS'

# Fit the ARIMA model
model = ARIMA(country_data, order=(5, 1, 0))
model_fit = model.fit()

# Forecast the next 10 years
forecast = model_fit.get_forecast(steps=10)
forecast_index = pd.date_range(start=country_data.index[-1] + pd.DateOffset(years=1), periods=10, freq='YS')
forecast_values = forecast.predicted_mean
conf_int = forecast.conf_int()

# Plot the forecast
plt.figure(figsize=(14, 8))
plt.plot(country_data, label='Observed')
plt.plot(forecast_index, forecast_values, label='Forecast')
plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='k', alpha=0.1)
plt.xlabel('Year')
plt.ylabel('Oil Consumption')
plt.title('Oil Consumption Forecast for the United States')
plt.legend()
plt.grid(True)
plt.show()

# Function to fit ARIMA model and forecast for a country
def forecast_oil_consumption(country, steps=10):
    country_data = dataframe[dataframe['Entity'] == country].drop(columns=['Entity']).T
    country_data.columns = ['Oil_Consumption']
    country_data.index = pd.to_datetime(country_data.index, format='%Y')
    country_data.index.freq = 'YS'

    model = ARIMA(country_data, order=(5, 1, 0))
    model_fit = model.fit()

    forecast = model_fit.get_forecast(steps=steps)
    forecast_index = pd.date_range(start=country_data.index[-1] + pd.DateOffset(years=1), periods=steps, freq='YS')
    forecast_values = forecast.predicted_mean
    conf_int = forecast.conf_int()

    return country_data, forecast_index, forecast_values, conf_int

# Forecasting for all top 10 countries
for country in top_10_countries:
    country_data, forecast_index, forecast_values, conf_int = forecast_oil_consumption(country)

    plt.figure(figsize=(14, 8))
    plt.plot(country_data, label='Observed')
    plt.plot(forecast_index, forecast_values, label='Forecast')
    plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='k', alpha=0.1)
    plt.xlabel('Year')
    plt.ylabel('Oil Consumption')
    plt.title(f'Oil Consumption Forecast for {country}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Define regions and their countries
regions = {
    'North America': ['United States', 'Canada', 'Mexico'],
    'Europe': ['Germany', 'United Kingdom', 'France', 'Italy'],
    'Asia': ['China', 'India', 'Japan', 'South Korea'],
    'Middle East': ['Saudi Arabia', 'Iran', 'Iraq', 'United Arab Emirates'],
    'South America': ['Brazil', 'Argentina', 'Venezuela']
}

# Plot oil consumption trends for different regions
plt.figure(figsize=(14, 8))
for region, countries in regions.items():
    region_data = dataframe[dataframe['Entity'].isin(countries)].drop(columns=['Entity']).sum()
    plt.plot(region_data.index[1:], region_data.values[1:], label=region)

plt.xlabel('Year')
plt.ylabel('Oil Consumption')
plt.title('Oil Consumption Trends by Region (1965-2023)')
plt.legend()
plt.grid(True)
plt.show()
