import sys

sys.path.append("../")
from utils.utils import *
import time
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

sensors = ['energy', 'maximum_amplitude', 'radial_strain', 'tangential_strain']
training = "training"
test = 'test'
train_sample = 1000
test_sample = 200


def dtw(x, y):
    distance, path = fastdtw(x, y, dist=euclidean)
    return distance

start_time = time.time()
for sensor in sensors:
    print sensor
    training_reader = pd.read_csv(get_diff_path(training))
    training_reader = training_reader.sample(train_sample)
    train_data, train_labels = read_data(training_reader, sensor)

    print "test"
    test_reader = pd.read_csv(get_diff_path(test))
    test_reader = test_reader.sample(test_sample)
    test_data, test_labels = read_data(test_reader, sensor)

    classified = []
    for test_example in test_data:
        min_distance = dtw(train_data[0], test_example)
        label = test_labels[0]
        for training_example, example_label in zip(train_data[1:], train_labels[1:]):
            dist = dtw(training_example, test_example)
            if dist < min_distance:
                min_distance = dist
                label = example_label
        classified.append(label)
    get_score_and_confusion_matrix(test_labels, classified)
    print("--- time %s seconds ---" % (time.time() - start_time))

print("--- testing time %s seconds ---" % (time.time() - start_time))
