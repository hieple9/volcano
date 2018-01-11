import sys

sys.path.append("../")
from utils.utils import *

explosion_expert_reader = pd.read_csv(get_diff_path("explosion_expert_raw"))
explosion_expert_reader = explosion_expert_reader.sample(frac=1).reset_index(drop=True)
explosion_expert_reader['label'] = [1]*len(explosion_expert_reader)

train_reader = pd.read_csv(get_raw_path("2009_2016"))
train_reader = train_reader[train_reader.label == 0]

all_data = pd.concat([explosion_expert_reader, train_reader])
all_data = all_data.sample(frac=1).reset_index(drop=True)

test_size = int(0.125 * len(all_data))

test_set = all_data[:test_size]
print len(test_set)
print sum(test_set.label.values) / float(len(test_set))
test_set.to_csv('../processed_data/all/exact/raw/test.csv', index=None)

training_set = all_data[test_size:]
print len(training_set)
print sum(training_set.label.values) / float(len(training_set))
training_set.to_csv('../processed_data/all/exact/raw/training.csv', index=None)
