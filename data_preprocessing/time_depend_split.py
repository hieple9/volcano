import pandas as pd

# test set: 25% each year 2013, 2014, 2015, 2016
# test and validation: the rest
first_half = ['2009_101_60_explosion', '2010_101_60_explosion', '2011_101_60_explosion', '2012_101_60_explosion']
second_half = ['2013_101_60_explosion', '2014_101_60_explosion', '2015_101_60_explosion', '2016_101_60_explosion']

test = 'validation_101_60_explosion_old'
training = 'training_101_60_explosion_old'

list_test_training = []
# for test and training
for year in second_half:
    print year
    data = pd.read_csv('../processed_data/pre/diff/%s.csv' % year)
    list_test_training.append(data)

all_data_test_training = pd.concat(list_test_training)
# print all_data_test_training.iloc[0]
# all_data_test_training = all_data_test_training.sample(frac=1).reset_index(drop=True)
# print all_data_test_training.iloc[0]
data_test_training_size = len(all_data_test_training)
print data_test_training_size

# sample test
test_size = 50000
print "test_size", test_size
all_data_test_training[:test_size].to_csv('../processed_data/pre/diff/%s.csv' % test, index=None)

# for test only
list_training = [all_data_test_training[test_size:]]
for year in first_half:
    print year
    data = pd.read_csv('../processed_data/pre/diff/%s.csv' % year)
    list_training.append(data)

all_data_training = pd.concat(list_training)
print all_data_training.iloc[0]
all_data_training = all_data_training.sample(frac=1).reset_index(drop=True)
print all_data_training.iloc[0]

data_training_size = len(all_data_training)
print "training_size", data_training_size
all_data_training.to_csv('../processed_data/pre/diff/%s.csv' % training, index=None)
