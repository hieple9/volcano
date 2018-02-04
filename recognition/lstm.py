from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.recurrent import LSTM
import time
import sys

sys.path.append("../")
from utils.utils import *

np.random.seed(7)

start_time = time.time()
training_set = 'training'
test_sets = ['test']
run_test = True
sensors = ['energy', 'maximum_amplitude', 'radial_strain', 'tangential_strain']
# sensors = ['energy']
batch_size = 1000
filters = 64
kernel_size = 5
epochs = 30
label_zero_ratio = 2

for sensor in sensors:
    print "_____________________________________________________________________"
    print sensor

    training_reader = pd.read_csv(get_diff_path(training_set))
    train_data, train_labels = read_data(training_reader, sensor)
    print train_data.shape
    dim = train_data[0].shape

    # Create the model
    model = Sequential()
    # model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=dim))
    # # model.add(BatchNormalization(axis=1))
    # model.add(Activation('relu'))
    model.add(LSTM(32, input_shape=dim))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
    print model.summary()

    # Train the model
    print 'Training.....................................................................'
    model.fit(train_data, train_labels, epochs=epochs, verbose=1, batch_size=batch_size,
              class_weight={1: label_zero_ratio, 0: 1}, shuffle=True, validation_split=0.2)

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

print("--- %s seconds ---" % (time.time() - start_time))
