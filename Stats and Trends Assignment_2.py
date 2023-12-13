# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:43:10 2023

@author: Michael
"""

import pandas as pd
import matplotlib.pyplot as plt


def extractUrbanPop_df(file_path2):
    '''Function for reading, slicing and transposing specific \
        series code from the general dataframe'''
    # Read the CSV file into a DataFrame
    urban_pop = pd.read_csv(file_path)

    # Define columns to extract
    columns_extract = ['Country Name', 'Country Code', 'Series Name',
                       'Series Code'] + [str(year) for year in
                                         range(2000, 2020)]

    # Extract relevant columns
    urban_pop = urban_pop[urban_pop['Series Code']
                          == 'SP.URB.TOTL.IN.ZS'][columns_extract]

    # Define countries to filter
    countries = ['CEB', 'EAS', 'ECS', 'LCN', 'MEA', 'NAC', 'SAS', 'SSF']

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


def extractFemaleInIndustrydf(file_path):
    '''Function for reading, slicing and transposing specific \
        series code from the general dataframe'''
    # Read the CSV file into a DataFrame
    fem_industry = pd.read_csv(file_path)

    # Define columns to extract
    columns_extract = ['Country Name', 'Country Code', 'Series Name',
                       'Series Code'] + [str(year) for year
                                         in range(2000, 2020)]

    # Extract relevant columns
    fem_industry = fem_industry[fem_industry['Series Code']
                                == 'SL.IND.EMPL.FE.ZS'][columns_extract]

    # Define countries to filter
    countries = ['CEB', 'EAS', 'ECS', 'LCN', 'MEA', 'NAC', 'SAS', 'SSF']

    # Filter rows based on countries
    fem_industry = fem_industry[fem_industry['Country Code'].isin(countries)]

    # Select specific columns
    fem_industry = fem_industry.drop(
        ['Series Name', 'Series Code', 'Country Code'], axis=1)

    # Set 'Country Name' as the index
    fem_industry.set_index('Country Name', inplace=True)
    fem_industry = fem_industry.astype('float64')

    # Transpose the DataFrame
    fem_industry_T = fem_industry.transpose()

    return fem_industry, fem_industry_T


# Provide the correct file path to your Excel file
file_path = 'sdg df.csv'

# Call the function with the file path
fem_industry, fem_industry_T = extractFemaleInIndustrydf(file_path)

# Display the DataFrames
print('Original Percentage Female in Industry DataFrame:')
print(fem_industry)
print('\nTransposed Percentage Female in Industry DataFrame:')
print(fem_industry_T)


def extractMaleInIndustrydf(file_path):
    '''Function for reading, slicing and transposing specific\
        series code from the general dataframe'''
    # Read the CSV file into a DataFrame
    male_industry = pd.read_csv(file_path)

    # Define columns to extract
    columns_extract = ['Country Name', 'Country Code', 'Series Name',
                       'Series Code'] + [str(year) for year
                                         in range(2000, 2020)]

    # Extract relevant columns
    male_industry = male_industry[male_industry['Series Code']
                                  == 'SL.IND.EMPL.MA.ZS'][columns_extract]

    # Define countries to filter
    countries = ['CEB', 'EAS', 'ECS', 'LCN', 'MEA', 'NAC', 'SAS', 'SSF']

    # Filter rows based on countries
    male_industry = male_industry[male_industry['Country Code'].isin(
        countries)]

    # Select specific columns
    male_industry = male_industry.drop(
        ['Series Name', 'Series Code', 'Country Code'], axis=1)

    # Set 'Country Name' as the index
    male_industry.set_index('Country Name', inplace=True)
    male_industry = male_industry.astype('float64')

    # Transpose the DataFrame
    male_industry_T = male_industry.transpose()

    return male_industry, male_industry_T


# Provide the correct file path to your Excel file
file_path = 'sdg df.csv'

# Call the function with the file path
male_industry, male_industry_T = extractMaleInIndustrydf(file_path)

# Display the DataFrames
print('Original Percentage Male in Industry DataFrame:')
print(male_industry)
print('\nTransposed Percentage Male in Industry DataFrame:')
print(male_industry_T)

# Set the figure size
plt.figure(figsize=(18, 6))

# Create subplots
plt.subplot(1, 2, 1)
plt.plot(fem_industry_T.index, fem_industry_T[fem_industry_T.columns],
         marker='o', markersize=8, linewidth=2, label=fem_industry_T.columns)
plt.xticks(rotation=90)
plt.title('Female Employment in Industry')
plt.xlabel('Year')
plt.ylabel('Percentage of Employment')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(male_industry_T.index, male_industry_T[male_industry_T.columns],
         marker='o', markersize=8, linewidth=2, label=male_industry_T.columns)
plt.xticks(rotation=90)
plt.title('Male Employment in Industry')
plt.xlabel('Year')
plt.ylabel('Percentage of Employment')
plt.grid(True)

# Adding Legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjusting Layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


# Plotting Bar Plot to show mean Urban Population growth over 20years

urban_pop_mean = urban_pop_T.mean()
plt.figure(4)
plt.bar(urban_pop.index, urban_pop_mean, width=0.8,
        label=urban_pop.index)
plt.xticks(rotation=90)
plt.title('Urban Population Growth over 20years')
plt.xlabel('Regions')
plt.ylabel('Growth Percentage')

# Stats for all DataFrames
print('Stats for Urban Population DataFrame:')
print(urban_pop_T.describe())

print('Stats for Females in industry DataFrame:')
print(fem_industry_T.describe())

print('Stats for Males in industry DataFrame:')
print(male_industry_T.describe())

print('Correlation between Female in Industry and Urban Pop DataFrames:')
correlation = fem_industry.corrwith(urban_pop)
print(correlation)

print('Mean for Urban Population DataFrame:')
print(urban_pop_T.mean())
