from pyts.classification import SAXVSMClassifier
import time
import sys

sys.path.append("../")
from utils.utils import *

np.random.seed(7)
run_test = True
training_set = 'training'
test_sets = ['test']
sensors = ['energy', 'maximum_amplitude', 'radial_strain', 'tangential_strain']
train_sample = 40000
test_sample = 10000
chunk = 2500

start_time = time.time()
for sensor in sensors:
    print sensor
    training_reader = pd.read_csv(get_diff_path(training_set))
    training_reader = training_reader.sample(train_sample)
    train_data, train_labels = read_data(training_reader, sensor)
    clf = SAXVSMClassifier()
    clf.fit(train_data, train_labels)
    print("--- train %s seconds ---" % (time.time() - start_time))

    if run_test:
        print 'Test'
        for test in test_sets:
            test_reader = pd.read_csv(get_diff_path(test))
            test_reader = test_reader.sample(test_sample)
            test_data, test_labels = read_data(test_reader, sensor)
            prediction = np.array([])
            for i in range(test_sample/chunk):
                temp = clf.predict(test_data[i*chunk:(i+1)*chunk])
                prediction = np.concatenate((prediction, temp))
            get_score_and_confusion_matrix(test_labels, prediction)
    print("--- time %s seconds ---" % (time.time() - start_time))

print("--- %s seconds ---" % (time.time() - start_time))
