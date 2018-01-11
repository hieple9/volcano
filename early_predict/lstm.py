from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv1D
from collections import Counter
from keras.layers.recurrent import LSTM
import time
import sys
sys.path.append("../")
from utils.utils import *

np.random.seed(7)

start_time = time.time()
training_set = 'training_60_explosion'
test_sets = ['test_60_explosion']
run_test = False
# sensors = ['tangential_strain', 'maximum_amplitude', 'radial_strain', 'energy']
sensors = ['tangential_strain']
batch_size = 128
filters = 128
kernel_size = 5
epochs = 10
class_weight = 6

for sensor in sensors:
    print sensor
    training_reader = pd.read_csv(get_early_diff_path(training_set))
    train_data, train_labels = read_data(training_reader, sensor, pre=True)
    dim = train_data[0].shape

    # Create the model
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=dim))
    # # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=dim))
    # # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=dim))
    # # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=dim))
    # # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=dim))
    # # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=dim))
    # # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=dim))
    # # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=dim))
    # # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(LSTM(32))
    # model.add(Dense(num_class, activation='softmax'))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
    print model.summary()

    print 'Training..................................................'
    model.fit(train_data, train_labels, epochs=epochs, verbose=1, batch_size=batch_size, shuffle=True,
              validation_split=0.2, class_weight={1: class_weight, 0: 1})

    print("--- train %s seconds ---" % (time.time() - start_time))

    original_classified = model.predict(np.array(train_data), batch_size=batch_size, verbose=0)
    best_threshold = find_best_threshold(train_labels, original_classified)
    classified = [get_class(x, best_threshold) for x in original_classified]


    def time_show(dtime):
        arr = []
        for i in range(len(dtime)):
            arr.append(dtime[i + 1])
        return arr


    train_time_positions = training_reader.pre_time_positions.values
    true_time_distribution = Counter()
    false_time_distribution = Counter()
    for i in range(len(train_labels)):
        if train_labels[i] == 1:
            time_pos = train_time_positions[i]
            if classified[i] == 1:
                true_time_distribution[time_pos] += 1
            else:
                false_time_distribution[time_pos] += 1
    print "true", time_show(true_time_distribution)
    print "false", time_show(false_time_distribution)

    # classified = model.predict_classes(train_data, batch_size=batch_size, verbose=0)
    # get_score_and_confusion_matrix(encoded_labels, classified)

    if run_test:
        print 'Test.......................................................'
        for test in test_sets:
            print test
            test_reader = pd.read_csv(get_early_diff_path(test))
            test_data, test_labels = read_data(test_reader, sensor, pre=True)

            # test_labels = encoder.transform(test_labels)
            # classified = model.predict_classes(test_data, batch_size=batch_size, verbose=0)
            # score, confusion = get_score_and_confusion_matrix(test_labels, classified)

            # original_classified = model.predict(test_data, batch_size=batch_size, verbose=0)
            # print "0.5 threshold"
            # classified = [np.around(x[0]) for x in original_classified]
            # get_score_and_confusion_matrix(test_labels, classified)

            # print "best threshold"
            # classified = [get_class(x, best_threshold) for x in original_classified]
            # get_score_and_confusion_matrix(test_labels, classified)

print("--- %s seconds ---" % (time.time() - start_time))
