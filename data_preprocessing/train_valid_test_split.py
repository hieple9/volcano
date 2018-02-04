import pandas as pd

# Valid: 50K
# Test: 50K
# Train: Rest
# years = ['2009__101_60_sw50_explosion', '2010__101_60_sw50_explosion', '2011__101_60_sw50_explosion',
#          '2012__101_60_sw50_explosion', '2013__101_60_sw50_explosion', '2014__101_60_sw50_explosion',
#          '2015__101_60_sw50_explosion', '2016__101_60_sw50_explosion']

years = ['2009_101_60_explosion', '2010_101_60_explosion', '2011_101_60_explosion',
         "2012_101_60_explosion", "2013_101_60_explosion", "2014_101_60_explosion"]

test = 'test_101_60_explosion'
validation = 'validation_101_60_explosion'
training = 'training_101_60_explosion'

list_data = []
for year in years:
    print year
    data = pd.read_csv('../processed_data/pre/diff/%s.csv' % year)
    print year, sum(data.pre_labels.values)
    list_data.append(data)

all_data = pd.concat(list_data)
print "all_data", sum(all_data.pre_labels.values)
print "len", len(all_data)

test_size = 0
validation_size = 50000
print validation_size
training_size = len(all_data) - test_size - validation_size

training_data = all_data[:training_size]
validation_data = all_data[training_size:training_size + validation_size]
test_data = all_data[training_size + validation_size:]

training_data.to_csv('../processed_data/pre/diff/%s.csv' % training, index=None)
validation_data.to_csv('../processed_data/pre/diff/%s.csv' % validation, index=None)
# test_data.to_csv('../processed_data/pre/diff/%s.csv' % test, index=None)

# Test to make sure that the data is not overlapped
# Check sum of current and pre labels

training_current = sum(training_data.pre_labels.values)
print training_current
validation_current = sum(validation_data.pre_labels.values)
print validation_current
test_current = sum(test_data.pre_labels.values)
print test_current
sum_all = sum([validation_current, training_current, test_current])
print sum_all
