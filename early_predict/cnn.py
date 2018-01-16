from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, MaxPooling1D
from keras.layers.convolutional import Conv1D
from collections import Counter
import time
import sys
sys.path.append("../")
from utils.utils import *

np.random.seed(7)

start_time = time.time()
training_set = 'training_60_explosion'
test_sets = ['test_60_explosion']
run_test = True
# sensors = ['tangential_strain', 'maximum_amplitude', 'radial_strain', 'energy']
sensors = ['tangential_strain']
batch_size = 1000
filters = 128
kernel_size = 5
epochs = 5
class_weight = 15

for sensor in sensors:
    print sensor
    training_reader = pd.read_csv(get_early_diff_path(training_set))
    print len(training_reader)
    training_reader = training_reader[training_reader.current_labels == 0]
    print len(training_reader)
    train_data, train_labels = read_data(training_reader, sensor, pre=True)
    dim = train_data[0].shape

    # Create the model
    model = Sequential()

    model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=dim))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(2, 2))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(2, 2))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
    print model.summary()

    print 'Training..................................................'
    model.fit(train_data, train_labels, epochs=epochs, verbose=1, batch_size=batch_size, shuffle=True,
              validation_split=0.2, class_weight={1: class_weight, 0: 1})

    print("--- train %s seconds ---" % (time.time() - start_time))

    original_classified = model.predict(np.array(train_data), batch_size=batch_size, verbose=0)
    best_threshold = find_best_threshold(train_labels, original_classified)
    classified = [get_class(x, best_threshold) for x in original_classified]

    # Check current label
    current_check = []
    current_labels = training_reader.current_labels.values
    for index, value in enumerate(classified):
        if value + train_labels[index] == 2:
            current = current_labels[index]
            if current >= 1:
                current_check.append(1)
            else:
                current_check.append(0)
    print "current_check", len(current_check)
    print "sum", sum(current_check)
    print "len", len([x for x in current_check if x != 0])

    def time_show(dtime):
        arr = []
        for i in range(len(dtime)):
            arr.append(dtime[i+1])
        return arr

    train_time_positions = training_reader.pre_time_positions.values
    train_deflation = training_reader.deflation.values
    true_time_distribution = Counter()
    false_time_distribution = Counter()
    true_deflation = []
    false_deflation = []
    for i in range(len(train_labels)):
        if train_labels[i] == 1:
            time_pos = train_time_positions[i]
            if classified[i] == 1:
                true_time_distribution[time_pos] += 1
                true_deflation.append(train_deflation[i])
            else:
                false_time_distribution[time_pos] += 1
                false_deflation.append(train_deflation[i])

    print "true"
    print time_show(true_time_distribution)
    get_statistics(true_deflation)
    print "false"
    print time_show(false_time_distribution)
    get_statistics(false_deflation)

    if run_test:
        print 'Test.......................................................'
        for test in test_sets:
            print test
            test_reader = pd.read_csv(get_early_diff_path(test))
            print len(test_reader)
            test_reader = test_reader[test_reader.current_labels == 0]
            print len(test_reader)
            test_data, test_labels = read_data(test_reader, sensor, pre=True)

            time_positions = test_reader.pre_time_positions.values
            deflations = test_reader.deflation.values
            print len(time_positions)
            print len(deflations)
            print len(test_data)
            print len(test_labels)
            print "sum", sum(test_labels)

            # flip label of some data which are under the conditions
            for i in range(len(test_labels)):
                time_position = time_positions[i]
                deflation = deflations[i]
                if 1 <= time_position <= 2 and deflation < -5:
                    test_labels[i] = 1
                else:
                    test_labels[i] = 0
            print len(test_labels)
            print len(test_data)
            print "sum", sum(test_labels)

            original_classified = model.predict(test_data, batch_size=batch_size, verbose=0)
            get_statistics(original_classified)

            print "0.5 threshold"
            classified = [np.around(x[0]) for x in original_classified]
            get_score_and_confusion_matrix(test_labels, classified)

            test_time_positions = test_reader.pre_time_positions.values
            test_deflation = test_reader.deflation.values
            true_time_distribution = Counter()
            false_time_distribution = Counter()
            true_deflation = []
            false_deflation = []
            for i in range(len(test_labels)):
                if test_labels[i] == 1:
                    time_pos = test_time_positions[i]
                    if classified[i] == 1:
                        true_time_distribution[time_pos] += 1
                        true_deflation.append(test_deflation[i])
                    else:
                        false_time_distribution[time_pos] += 1
                        false_deflation.append(test_deflation[i])

            print "true"
            print time_show(true_time_distribution)
            get_statistics(true_deflation)
            print "false"
            print time_show(false_time_distribution)
            get_statistics(false_deflation)

            print "best threshold"
            classified = [get_class(x, best_threshold) for x in original_classified]
            get_score_and_confusion_matrix(test_labels, classified)

            test_time_positions = test_reader.pre_time_positions.values
            test_deflation = test_reader.deflation.values
            true_time_distribution = Counter()
            false_time_distribution = Counter()
            true_deflation = []
            false_deflation = []
            for i in range(len(test_labels)):
                if test_labels[i] == 1:
                    time_pos = test_time_positions[i]
                    if classified[i] == 1:
                        true_time_distribution[time_pos] += 1
                        true_deflation.append(test_deflation[i])
                    else:
                        false_time_distribution[time_pos] += 1
                        false_deflation.append(test_deflation[i])

            print "true"
            print time_show(true_time_distribution)
            get_statistics(true_deflation)
            print "false"
            print time_show(false_time_distribution)
            get_statistics(false_deflation)

            print "0.8 threshold"
            classified = [get_class(x, 0.8) for x in original_classified]
            get_score_and_confusion_matrix(test_labels, classified)

            print "0.9 threshold"
            classified = [get_class(x, 0.9) for x in original_classified]
            get_score_and_confusion_matrix(test_labels, classified)

            high_possible_index = [i for i, x in enumerate(original_classified) if x >= 0.9]
            # high_possible_values = [x for x in original_classified if x >= 0.9]
            print "high_possible_index", high_possible_index
            # print "high_possible_values", high_possible_values
            print [test_time_positions[i] for i in high_possible_index]
            print [round(test_deflation[i], 2) for i in high_possible_index]
            # visual = [test_data[i] for i in high_possible_index]

print("--- %s seconds ---" % (time.time() - start_time))
