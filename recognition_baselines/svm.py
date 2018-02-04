from sklearn import svm
import sys

sys.path.append("../")
from utils.utils import *
import time

np.random.seed(7)
training_set = 'training'
test = 'test'
sensors = ['radial_strain', 'tangential_strain']
train_sample = 100000
test_sample = 20000
label_zero_ratio = 2

start_time = time.time()
for sensor in sensors:
    print sensor
    training_reader = pd.read_csv(get_diff_path(training_set))
    training_reader = training_reader.sample(train_sample)
    train_data, train_labels = read_data(training_reader, sensor)
    train_data = np.array([x.flatten() for x in train_data])

    kernels = ['rbf']
    for kernel in kernels:
        print kernel
        classifier = svm.SVC(kernel=kernel, class_weight={1: label_zero_ratio})
        classifier.fit(train_data, train_labels)
        print("--- training %s seconds ---" % (time.time() - start_time))

        print "test"
        test_reader = pd.read_csv(get_diff_path(test))
        test_reader = test_reader.sample(test_sample)
        test_data, test_labels = read_data(test_reader, sensor)
        test_data = np.array([x.flatten() for x in test_data])

        y_pred = classifier.predict(test_data)
        get_score_and_confusion_matrix(test_labels, y_pred)

    print("--- testing %s seconds ---" % (time.time() - start_time))

print("--- training and testing %s seconds ---" % (time.time() - start_time))
