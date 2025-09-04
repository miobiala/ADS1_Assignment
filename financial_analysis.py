# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:05:50 2023

@author: Michael
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Read Returns Data to DataFrames
returns_BP = pd.read_csv('BP_ann.csv')
print(returns_BP)

returns_BCS = pd.read_csv('BCS_ann.csv')
print(returns_BCS)

returns_TSCO = pd.read_csv('TSCO_ann.csv')
print(returns_TSCO)

returns_VOD = pd.read_csv('VOD_ann.csv')
print(returns_VOD)




# Plot subplots for Histogram of Annual Returns
plt.figure(1)
plt.subplot(2, 2, 1)
plt.hist(returns_BP['ann_return'], label='BP', density=True)

plt.legend()
plt.subplot(2, 2, 2)
plt.hist(returns_BCS['ann_return'], label='BCS', density=True)

plt.legend()
plt.subplot(2, 2, 3)
plt.hist(returns_TSCO['ann_return'], label='TSCO', density=True)

plt.legend()
plt.subplot(2, 2, 4)
plt.hist(returns_VOD['ann_return'], label='VOD', density=True)

plt.legend()
plt.savefig('SubPlot of % Annual Returns.png')
plt.show()




# Pick two stocks and create a combined histogram of both

plt.figure(2)

plt.hist(returns_BCS['ann_return'], bins=10, label='BCS', density=True)
plt.hist(returns_TSCO['ann_return'], bins=10, label='TSCO', density=True, alpha=0.7)

plt.legend()
plt.savefig('Histogram comparison of BCS and TSCO.png')
plt.show()



# Produce a Box Plot of Returns distribution of the four companies

plt.figure(3)
plt.boxplot([returns_BP['ann_return'], returns_BCS['ann_return'],
             returns_TSCO['ann_return'],returns_VOD['ann_return']],
             labels=['BP', 'Barclays', 'Tesco', 'Vodafone'])

plt.ylabel('Annual Returns %')
plt.savefig('Box Plot of % Annual Returns.png')
plt.show()


# Create a table for plot of 4 companies and their Market Capitalization

mcap = np.array([33367, 68785, 20979, 29741])
company = ['Barclays', 'BP', 'Tesco', 'Vodaphone']



# Plot Pie Chart for four companies.
plt.figure(4)
plt.pie(mcap, labels=company)

plt.title('Pie Chart figure for Market Capitalization')
plt.savefig('Pie Chart figure for Market Capitalization.png')
plt.show()



# Plot Pie Chart for % of the total market cap
# create % total market cap

mcap_p = mcap / 1814000

plt.figure(5)
plt.pie(mcap_p, labels=company)
plt.title('Pie Chart Figure for % MC/TM')

plt.savefig('Plot of Market Cap against Total Market.png')
plt.show()

print(mcap)
print(mcap_p)

# Create Bar Chart of Market Capital

plt.figure(6)
plt.bar(company, mcap)

plt.title('Market Capital')
plt.xlabel('Company')
plt.ylabel('Marke Cap per Mill. Pounds')
plt.savefig('Market CapitaLization Bar Chart')
plt.show()
