import pandas as pd

years = ['2009', '2010', '2011', '2012', '2013']

name = '2009_2013'
all_data = []
# for test and training
for year in years:
    print year
    data = pd.read_csv('../processed_data/pre/diff/%s_60_explosion.csv' % year)
    all_data.append(data)

data = pd.concat(all_data)
print data.iloc[0]
data = data.sample(frac=1).reset_index(drop=True)
print data.iloc[0]

print len(data)
data.to_csv('../processed_data/pre/diff/%s_60_explosion.csv' % name, index=None)
