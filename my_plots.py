# -*- coding: utf-8 -*-

# avoid chinese mess
import sys

reload(sys)
sys.setdefaultencoding('utf8')

import glob, os

import xlrd

import networkx as nx
from networkx.algorithms import bipartite

import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
import glob, os
import math
from math import log10

import collections
from collections import Counter
import matplotlib.pyplot as plt
import scipy
from scipy import stats, integrate  # draw density line
import seaborn as sns

"""
Draw denstiy with seaborn.
"""


def density(filename, sequence):
    ax = plt.axes()
    sns.kdeplot(sequence, ax=ax)
    ax.set_title(filename + '_density')
    plt.savefig(filename + '_density.png')
    plt.clf()


"""
Draw hist with plt
"""


def hist_plt(x, bin, filename, filename2):  # filename2是横轴的变量名称，比如出强度;filename是日期
    #plt.hist(x, bins=bin, normed=True, facecolor='green', alpha=0.75)  # normed=1则是频率；log = True,
    plt.hist(x, bins=bin, facecolor='green', alpha=0.75)
    #plt.hist(x, facecolor='green', alpha=0.75)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(filename2)
    plt.ylabel('Frequency')
    plt.title(filename + '_density_hist')
    plt.savefig(filename + '_' + filename2 + '_density_hist.png')
    plt.clf()


"""
Draw hist in scatter.
"""


def hist_to_scatter(x, bin, filename, filename2):
    n, bins = np.histogram(x, bins=bin)
    bins_mean = [0.5 * (bins[i] + bins[i + 1]) for i in range(len(n))]
    plt.scatter(bins_mean, n, marker='x', s=30)
    plt.xlabel(filename2)
    plt.ylabel('Frequency')
    plt.savefig(filename + '_' + filename2 + '_frequency_scatter.png')
    plt.clf()


"""
Draw hist in scatter-log_bin.
"""


def hist_log_bin(x, filename, MIN, MAX, bin_count):
    #The most direct way is to just compute the log10 of the limits, compute linearly spaced bins, and then convert back by raising to the power of 10,
    #http: // stackoverflow.com / questions / 6855710 / how - to - have - logarithmic - bins - in -a - python - histogram
    plt.figure()
    plt.hist(x, bins=10 ** np.linspace(np.log10(MIN), np.log10(MAX), bin_count)) #normed=True
    plt.gca().set_xscale("log")
    plt.title(filename + '_log-bin_hist')
    plt.savefig(filename + "_log-bin_hist.png")
    plt.clf()

"""
Draw zhexian.
"""

def zhexian(x, filename):
    plt.figure()
    x.plot()
    plt.title(filename)
    plt.savefig(filename + ".png")
    plt.clf()