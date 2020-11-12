"""
This document stores all the nice functions about plots
"""

# Easy way to plot % of values
font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16}

x, y, hue = "column1", "Percentage", "column2"

prop_df = (df[hue]
           .groupby(df[x])
           .value_counts(normalize=True)
           .rename(y)
           .reset_index())

splot = sns.barplot(x=x, y=y, hue=hue, data=prop_df, palette="Blues_r")
sns.set()
sns.set_style('white')
sns.despine(offset=10, trim=True, bottom=True)
plt.legend().remove()
plt.xlabel('X axis title', fontdict=font)
plt.yticks([])
plt.ylabel('')
plt.show()
