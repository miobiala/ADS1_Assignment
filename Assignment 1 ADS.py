# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 22:55:13 2023

@author: Michael
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Graph 1
# Defining my lineplot function for my Pandas DataFrame on Substance Use


def myline_plot(substance_use, x_column='age',
                y_columns=['alcohol_use', 'marijuana_use', 'cocaine_use',
                           'hallucinogen_use'],
                labels=['Alcohol', 'Marijuana', 'Cocaine', 'Hallucinogens'],
                title='Substance Use by Age Group', xlabel='Age Group',
                ylabel='Percentage of Users in Past 12 Months', rotation=90):
    '''

    Function to plot a multi-column line graph for a Pandas DataFrame

    Parameters;
    - Substance_use: Pandas DataFrame
        Where the Data is being plotted from. 
    - x_column: str
        Name of the column in which x-axis gets it values
    - y_columns(List of str)
        This is the list of columns where yaxis gets its values
    - Labels(list of str):
        List of Labels for the legend
    - Title(str):
        Title of the Plot
    - xlabel(str)
        This is the xaxis label
    - ylabel(str)
        This is the yaxis label
    - rotation(int)
        Angle in which the x ticks will be positioned

    # it returns a lineplot showing multiple columns on a specified x axis
    '''

    plt.figure(1)

    # Applying a for loop to iterate over my y_columns and list of labels
    for y_col, label in zip(y_columns, labels):
        plt.plot(substance_use[x_column], substance_use[y_col],
                 label=label)

    # Adding defining features to my line graph for better understanding
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.grid(True)

    # save plot as .png file in directory
    plt.savefig('2018 Sub_use Lineplot.png')

    plt.show()

    return


# Reading my data as a Pandas DataFrame
substance_use = pd.read_csv('drug-use-by-age.csv')

# Calling line plot function to plot from my dataframe
myline_plot(substance_use)


# Graph 2
# Writing a function for ploting a Pie Chart for my  Pandas DataFrame

def mypie_plot(fifa_conf, lab_col='confederation', val_col='tv_audience_share',
               title='Distribution of World Cup TV audience across FIFA Confederations(%)'):
    '''

    Function to plot a Pie Chart for my Pandas DataFrame

    Parameters;
    - fifa_conf: Pandas DataFrame
        Where the Data is being plotted from. 
    - lab_col: str
        Name of the column where Pie chart Labels are.
    - val_col: str
        Name of column where Pie chart values are.
    - title: str
        Title of the Pie chart.
    # The Function returns a Pie Chart
    '''

    plt.figure(2, figsize=(8, 8))

    # Extract data as variables
    values = fifa_conf[val_col]
    labels = fifa_conf[lab_col]

    # create my piechart with matplotlib
    plt.pie(values, labels=labels, autopct='%1.1f%%',
            startangle=60, counterclock=True)

    # setting aspect ratio to give a perfect circle
    # setting a title to graph
    plt.axis('equal')
    plt.title(title)

    # save file as .png to directory
    plt.savefig('World Cup Audience across FIFA Confeds.png')

    plt.show()

    return


# Read my data as a Pandas Dataframe
countries_Fifa = pd.read_csv('fifa_countries_audience.csv')

# grouping countries by their FIFA confederation organization and
# summing the corresponding columns
fifa_confed = pd.DataFrame(countries_Fifa.groupby('confederation').sum())

# Reset confederation column as the index to create dataframe to be used
fifa_conf = fifa_confed.reset_index()

# Calling Pie Chart function to plot from my dataframe
mypie_plot(fifa_conf)


# Graph 3
# Writing Function to plot scatter plots with mathplotlib

def myscatter_plot(substance_use, x_column='age',
                   y_columns=['alcohol_frequency', 'marijuana_frequency'],
                   labels=['Alcohol', 'Marijuana'], color=['red', 'green'],
                   title='Frequency of Use for Alcohol & Marijuana, 2018',
                   xlabel='Age Group',
                   ylabel='Median No. of time per User', rotation=90):
    '''

    Function to plot a multi-column line graph for a Pandas DataFrame

    Parameters;
    - Substance_use: Pandas DataFrame
        Where the Data is being plotted from. 
    - x_column: str
        Name of the column in which x-axis gets it values
    - y_columns(List of str)
        This is the list of columns where yaxis gets its values
    - Labels(list of str):
        List of Labels for the legend
    - color(list of str)
        List of color for different column
    - Title(str):
        Title of the Plot
    - xlabel(str)
        This is the xaxis label
    - ylabel(str)
        This is the yaxis label
    - rotation(int)
        Angle in which the x ticks will be positioned

    # it returns a scatter plot showing multiple columns on a specified x axis
    '''
    plt.figure(9, figsize=(10, 10))

    # Applying a for loop to iterate over my y_columns and list of labels
    # and colors
    for y_col, label, c in zip(y_columns, labels, color):
        plt.scatter(substance_use[x_column], substance_use[y_col],
                    s=300, c=c, label=label)

    # Adding defining features to my scatter graph for better understanding
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.grid(True)

    # save plot as .png file
    plt.savefig('Frequency of Sub_Use.png')

    plt.show()

    return


# Reading my data as a Pandas DataFrame
substance_use = pd.read_csv('drug-use-by-age.csv')

# Calling line plot function to plot from my dataframe
myscatter_plot(substance_use)
