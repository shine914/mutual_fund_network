# -*- coding: utf-8 -*-

#avoid chinese mess
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import glob,os
import xlrd
import networkx as nx
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from my_dataframes import drop_duplicate, drop_emptycell, stock_status

os.chdir('/Users/shine/work_hard/financial_network/data/2956_sec/successor_decline_10min')
df = pd.read_excel('one_successor_decline_manual.xlsx', sheet_name='Sheet1')


# df['decline_abs'] = np.absolute(df['decline'])


plt.figure()
sns.lmplot('low', 'successor_decline', data=df, hue='node', fit_reg=False, legend=False)
#sns.lmplot('decline_abs', 'successor_lower_rate', data=df, hue='date', fit_reg=False)
slope, intercept = np.polyfit(df['low'], df['successor_decline'], 1)
print(slope)
plt.legend(loc='upper left')
# plt.ylim([-0.11,0])
# plt.xlim([-0.11,0.02])
plt.savefig("one_successor_decline_manual.png")
plt.clf()
