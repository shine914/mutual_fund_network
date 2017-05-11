# -*- coding: utf-8 -*-
import glob,os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1')

#filename = 'predecessors_lower_rate'
#filename = 'successors_sus_lower_rate'
#filename = 'predecessors_sus_lower_rate'
#filename = 'successors_lower_rate'


#filename = 'predecessors_sus_lower_rate_weight'
#filename = 'successors_sus_lower_rate_weight'
#filename = 'predecessors_lower_rate_weight'
# filename = 'successors_lower_rate_weight'

# xlab = list(df.columns)
# ylab = list(df.index)



os.chdir('E:/financial_network/data/threshold/stock_alpha/0.95_2015-1/investment_behavior')

filename = '0all_company_5group_number'
df = pd.read_excel(filename+'.xlsx', sheet_name='Sheet1')
df = pd.DataFrame(df)

df = df.set_index('company')


# xlab = list(df.columns)
# ylab = list(df.index)
# print xlab,ylab

fig, ax = plt.subplots()
heatmap = ax.pcolor(df, cmap=plt.cm.Blues, alpha=0.8)
ax.set_xticklabels(list(df.columns), minor=True)
ax.set_yticklabels(list(df.index), minor=True)
# plt.xlabel('date')
# plt.ylabel('delta_t')
ax.set_title(filename)
plt.savefig(filename+'_heat.png')
plt.clf()



