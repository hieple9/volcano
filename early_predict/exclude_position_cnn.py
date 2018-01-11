from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
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
filters = 128
kernel_size = 5
epochs = 10
label_zero_ratio = 2
exclude = [1, 2, 3, 4, 5, 6, 7, 8]
len_ts = 80

for sensor in sensors:
    print "_____________________________________________________________________"
    print sensor
    training_reader = pd.read_csv(get_diff_path(training_set))
    training_reader = training_reader[~training_reader["time_positions"].isin(exclude)]

    train_data = np.array(
        [np.fromstring(e, dtype=float, sep=',')[:len_ts].reshape(len_ts, 1) for e in training_reader[sensor]])
    train_labels = training_reader.label.values
    print "Total training", len(training_reader)
    print "Class 1 in training", sum(train_labels)
    dim = train_data[0].shape

    # Create the model
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=dim))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
    print model.summary()

    # Train the model
    print 'Training...'
    model.fit(train_data, train_labels, epochs=epochs, verbose=1, batch_size=batch_size,
              class_weight={1: label_zero_ratio, 0: 1}, shuffle=True, validation_split=0.2)

    print("--- train %s seconds ---" % (time.time() - start_time))

    original_classified = model.predict(np.array(train_data), batch_size=batch_size, verbose=0)
    best_threshold = find_best_threshold(train_labels, original_classified)

    if run_test:
        print 'Test...'
        for test in test_sets:
            print test
            test_reader = pd.read_csv(get_diff_path(test))
            test_reader = test_reader[~test_reader["time_positions"].isin(exclude)]

            test_data = np.array(
                [np.fromstring(x, dtype=float, sep=',')[:len_ts].reshape(len_ts, 1) for x in test_reader[sensor]])
            validation_labels = test_reader.label.values
            print "Total test", len(test_reader)
            print "Class 1 in test", sum(validation_labels)
            original_classified = model.predict(np.array(test_data), batch_size=batch_size, verbose=0)

            classified = [np.around(x[0]) for x in original_classified]
            f_score = f1_score(validation_labels, classified, average=None)
            print "f_score", f_score
            print "f_score AVG", np.average(f_score)
            confusion = confusion_matrix(validation_labels, classified)
            print "confusion matrix"
            print confusion

            print "best threshold"
            classified = [get_class(x, best_threshold) for x in original_classified]
            f_score = f1_score(validation_labels, classified, average=None)
            print "f_score", f_score
            print "f_score AVG", np.average(f_score)
            confusion = confusion_matrix(validation_labels, classified)
            print "confusion matrix"
            print confusion

print("--- %s seconds ---" % (time.time() - start_time))
