from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, MaxPooling1D
from keras.layers.convolutional import Conv1D

import time
import sys

sys.path.append("../")
from utils.utils import *

np.random.seed(7)

start_time = time.time()
training_set = 'training'
test_sets = ['test']
run_test = False
sensors = ['energy']
batch_size = 1000
num_filters = 128
kernel_size = 5
epochs = 50

for sensor in sensors:
    print "_____________________________________________________________________"
    print sensor

    training_reader = pd.read_csv(get_diff_path(training_set))
    train_data, train_labels = read_data(training_reader, sensor)
    dim = train_data[0].shape

    model = Sequential()

    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, input_shape=dim))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
    print model.summary()

    # Train the model
    print 'Training.....................................................................'
    model.fit(train_data, train_labels, epochs=epochs, verbose=1, batch_size=batch_size, shuffle=True, validation_split=0.2)

    print("--- train %s seconds ---" % (time.time() - start_time))

    original_classified = model.predict(train_data, batch_size=batch_size, verbose=0)
    best_threshold = find_best_threshold(train_labels, original_classified)

    if run_test:
        print 'Test......................................................................'
        for test in test_sets:
            print test
            test_reader = pd.read_csv(get_diff_path(test))
            test_data, test_labels = read_data(test_reader, sensor)

            original_classified = model.predict(test_data, batch_size=batch_size, verbose=0)
            print "0.5 threshold"
            classified = [np.around(x[0]) for x in original_classified]
            get_score_and_confusion_matrix(test_labels, classified)

            print "best threshold"
            classified = [get_class(x, best_threshold) for x in original_classified]
            get_score_and_confusion_matrix(test_labels, classified)

    # from pyts.transformation import GASF
    # gasf = GASF(image_size=100, overlapping=False, scale='0')
    # X_gasf = gasf.transform
    concerned_layers = ["conv1d_1", "conv1d_2", "conv1d_3", "conv1d_4", "conv1d_5", "conv1d_6", "conv1d_7", "conv1d_8"]
    for concerned_layer in concerned_layers:
        print concerned_layer
        layer = model.get_layer(concerned_layer)
        all_filters = layer.get_weights()[0]
        reorganized_filters = []
        for i in range(num_filters):
            print i
            temp_filter = []
            for filter in all_filters:
                temp_filter.append(filter[0][i])
            reorganized_filters.append(np.array(temp_filter))

        from pyts.visualization import plot_gasf
        filter_index = 0
        for reorganized_filter in reorganized_filters:
            plot_gasf(reorganized_filter, image_size=kernel_size, overlapping=False, scale='0',
                      output_file="layer_%s_filter%s" % (concerned_layer, filter_index))
            filter_index += 1

print("--- train %s seconds ---" % (time.time() - start_time))
