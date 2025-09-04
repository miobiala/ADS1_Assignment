# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 00:19:17 2024

@author: Michael
"""


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit


def extractUrbanPop_df(file_path):
    '''Function for reading, slicing and transposing specific \
        series code from the general dataframe'''
    # Read the CSV file into a DataFrame
    urban_pop = pd.read_csv(file_path)

    # Define columns to extract
    columns_extract = ['Country Name', 'Country Code', 'Series Name',
                       'Series Code'] + [str(year) for year in
                                         range(2000, 2016)]

    # Extract relevant columns
    urban_pop = urban_pop[urban_pop['Series Code']
                          == 'SP.URB.TOTL.IN.ZS'][columns_extract]

    # Define countries to filter
    countries = ['BRA', 'CAN', 'CHL', 'COL', 'FRA', 'DEU', 'MEX', 'USA', 'GBR',
                 'ARG', 'AUS', 'BEL', 'CHN', 'GHA', 'IND', 'JPN', 'KEN', 'NGA']

    # Filter rows based on countries
    urban_pop = urban_pop[urban_pop['Country Code'].isin(countries)]

    # Select specific columns
    urban_pop = urban_pop.drop(
        ['Series Name', 'Series Code', 'Country Code'], axis=1)

    # Set 'Country Name' as the index
    urban_pop.set_index('Country Name', inplace=True)
    urban_pop = urban_pop.astype('float64')

    # Transpose the DataFrame
    urban_pop_T = urban_pop.transpose()

    return urban_pop, urban_pop_T


file_path = 'sdg df.csv'

# Call the function with the file path
urban_pop, urban_pop_T = extractUrbanPop_df(file_path)

# Display the DataFrames
print('Original Urban Population Percentage DataFrame:')
print(urban_pop)
print('\nTransposed Urban Population Percentage DataFrame:')
print(urban_pop_T)


def extractRenewableCountries_df(file_path):
    '''Function for reading, slicing and transposing specific \
        series code from the general dataframe'''
    # Read the CSV file into a DataFrame
    renew_energy_city_df = pd.read_csv(file_path)

    # Define columns to extract
    columns_extract = ['Country Name', 'Country Code', 'Series Name',
                       'Series Code'] + [str(year) for year in
                                         range(2000, 2016)]

    # Extract relevant columns
    renew_energy_city_df =\
        renew_energy_city_df[renew_energy_city_df['Series Code']
                             == 'EG.ELC.RNEW.ZS'][columns_extract]
    countries = ['BRA', 'CAN', 'CHL', 'COL', 'FRA', 'DEU', 'MEX', 'USA', 'GBR',
                 'ARG', 'AUS', 'BEL', 'CHN', 'GHA', 'IND', 'JPN', 'KEN', 'NGA']
    renew_energy_city_df =\
        renew_energy_city_df[renew_energy_city_df['Country Code'].isin(
            countries)]

    # Select specific columns
    renew_energy_city_df = renew_energy_city_df.drop(
        ['Series Name', 'Series Code', 'Country Code'], axis=1)

    # Set 'Country Name' as the index
    renew_energy_city_df.set_index('Country Name', inplace=True)
    renew_energy_city_df = renew_energy_city_df.astype('float64')

    # Transpose the DataFrame
    renew_energy_city_df_T = renew_energy_city_df.transpose()

    return renew_energy_city_df, renew_energy_city_df_T


file_path = 'sdg df.csv'

# Call the function with the file path
renew_energy_city_df, renew_energy_city_df_T = extractRenewableCountries_df(
    file_path)

# Display the DataFrames
print('Original Renewable Energy DataFrame:')
print(renew_energy_city_df)
print('\nTransposed Renewable Energy DataFrame:')
print(renew_energy_city_df_T)


def extractRenewableUK_df(file_path):
    '''Function for reading, slicing and transposing specific \
        series code from the general dataframe'''
    # Read the CSV file into a DataFrame
    uk_renew = pd.read_csv(file_path)

    # Define columns to extract
    columns_extract = ['Country Name', 'Country Code', 'Series Name',
                       'Series Code'] + [str(year) for year in
                                         range(2000, 2016)]

    # Extract relevant columns
    uk_renew =\
        uk_renew[uk_renew['Series Code']
                 == 'EG.ELC.RNEW.ZS'][columns_extract]
    countries = ['GBR']
    uk_renew =\
        uk_renew[uk_renew['Country Code'].isin(
            countries)]

    # Select specific columns
    uk_renew = uk_renew.drop(
        ['Series Name', 'Series Code', 'Country Code'], axis=1)

    # Set 'Country Name' as the index
    uk_renew.set_index('Country Name', inplace=True)
    uk_renew = uk_renew.astype('float64')

    # Transpose the DataFrame
    uk_renew_T = uk_renew.transpose()

    return uk_renew, uk_renew_T


file_path = 'sdg df.csv'

# Call the function with the file path
uk_renew, uk_renew_T = extractRenewableUK_df(
    file_path)

# Display the DataFrames
print('Original UK Renewable Energy DataFrame:')
print(uk_renew)
print('\nTransposed UK Renewable Energy DataFrame:')
print(uk_renew_T)


energy = renew_energy_city_df_T.mean()
urban = urban_pop_T.mean()

# Merge the two DataFrames based on 'Country Name'
merged_df = pd.merge(urban_pop, renew_energy_city_df, left_index=True,
                     right_index=True, suffixes=('_forest', '_renewable'))


print(merged_df)


# K-means clustering
n_clusters = 3  # Specify the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
merged_df['cluster'] = kmeans.fit_predict(merged_df)

# Scatter plot showing the relationship between renewable energy and Urban Pop
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.scatterplot(x=urban, y=energy, hue='cluster', data=merged_df,
                palette='viridis')
plt.title('Cluster Scatterplot: Renewable Energy vs. Urban Population')
plt.xlabel('Mean Urban Population')
plt.ylabel('Mean Renewable Energy')
plt.xticks(rotation=45, ha='right')
plt.show()

# Scatter plot showing the relationship between renewable energy and Urban Pop
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.scatterplot(x=urban, y=energy, hue='Country Name', data=merged_df)
plt.title('Cluster Scatterplot: Renewable Energy vs. Urban Population')
plt.xlabel('Mean Urban Population')
plt.ylabel('Mean Renewable Energy')
plt.xticks(rotation=45, ha='right')
plt.show()

# Plotting Box Plot to show relation between clusters and Urban Pop
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.boxplot(x='cluster', y=urban, data=merged_df, palette='viridis')
plt.title('Box Plot: Urban Population Distribution Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Urban Population')
plt.show()

# Plotting Box Plot to show relation between clusters and renewable energy
plt.figure(figsize=(12, 8))
sns.boxplot(x='cluster', y=energy, data=merged_df, palette='viridis')
plt.title('Box Plot: Renewable Energy Distribution Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Renewable Energy')
plt.show()


def extractRenewableUK_df(file_path):
    '''Function for reading, slicing and transposing specific \
        series code from the general dataframe'''
    # Read the CSV file into a DataFrame
    uk_renew = pd.read_csv(file_path)

    # Define columns to extract
    columns_extract = ['Country Name', 'Country Code', 'Series Name',
                       'Series Code'] + [str(year) for year in
                                         range(2000, 2016)]

    # Extract relevant columns
    uk_renew =\
        uk_renew[uk_renew['Series Code']
                 == 'EG.ELC.RNEW.ZS'][columns_extract]
    countries = ['GBR']
    uk_renew =\
        uk_renew[uk_renew['Country Code'].isin(
            countries)]

    # Select specific columns
    uk_renew = uk_renew.drop(
        ['Series Name', 'Series Code', 'Country Code'], axis=1)

    # Set 'Country Name' as the index
    uk_renew.set_index('Country Name', inplace=True)
    uk_renew = uk_renew.astype('float64')

    # Transpose the DataFrame
    uk_renew_T = uk_renew.transpose()

    return uk_renew, uk_renew_T


file_path = 'sdg df.csv'

# Call the function with the file path
uk_renew, uk_renew_T = extractRenewableUK_df(
    file_path)

# Display the DataFrames
print('Original UK Renewable Energy DataFrame:')
print(uk_renew)
print('\nTransposed UK Renewable Energy DataFrame:')
print(uk_renew_T)


def exponential_function(x, a, b, c):
    return a * np.exp(b * (x - 2000)) + c


def fit_curve(x, y, func):
    params, covariance = curve_fit(func, x, y)
    return params


def plot_curve_fit(x, y, params, func, title):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Original Data', color='blue')

    x_fit = np.arange(min(x), max(x) + 1, 1)
    y_fit = func(x_fit, *params)

    plt.plot(x_fit, y_fit, label='Curve Fit', color='red')

    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Renewable Energy (% of Total Electricity)')
    plt.legend()
    plt.show()


# Make years as columns
years = uk_renew.columns.astype(int)
renewable_data = uk_renew.values.flatten()


# Fit the curve using curve_fit
params = fit_curve(years, renewable_data, exponential_function)

# Plotting the original data and the fitted curve
plot_curve_fit(years, renewable_data, params, exponential_function,
               'Renewable Energy Forecast for the UK')
