import sys

sys.path.append("../")
from utils.utils import *
import time
from scipy.spatial import distance

sensors = ['energy', 'maximum_amplitude', 'radial_strain', 'tangential_strain']
training = "training"
test = 'test'
train_sample = 30000
test_sample = 10000

start_time = time.time()
for sensor in sensors:
    print sensor
    training_reader = pd.read_csv(get_diff_path(training))
    training_reader = training_reader.sample(train_sample)
    train_data, train_labels = read_data(training_reader, sensor)
    train_data = np.array([[x.mean(), x.var()] for x in train_data])

    print "test"
    test_reader = pd.read_csv(get_diff_path(test))
    test_reader = test_reader.sample(test_sample)
    test_data, test_labels = read_data(test_reader, sensor)
    test_data = np.array([[x.mean(), x.var()] for x in test_data])

    classified = []
    for test_example in test_data:
        min_distance = distance.euclidean(train_data[0], test_example)
        label = test_labels[0]
        for training_example, example_label in zip(train_data[1:], train_labels[1:]):
            dist = distance.euclidean(training_example, test_example)
            if dist < min_distance:
                min_distance = dist
                label = example_label
        classified.append(label)
    get_score_and_confusion_matrix(test_labels, classified)
    print("--- time %s seconds ---" % (time.time() - start_time))

print("--- testing time %s seconds ---" % (time.time() - start_time))
