from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.recurrent import LSTM
from collections import Counter
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
batch_size = 1000
filters = 32
kernel_size = 5
epochs = 10
class_weight = 3
pre = 59

for sensor in sensors:
    print sensor, training_set
    training_reader = pd.read_csv(get_early_diff_path(training_set))
    train_data, train_labels, pre_train_data = read_data2(training_reader, sensor, pre=True)
    dim = train_data[0].shape

    # Create the model
    predictor = Sequential()
    predictor.add(LSTM(pre, input_shape=dim))
    predictor.compile(loss='mean_squared_error', optimizer="adam", metrics=['mse'])
    print predictor.summary()
    print 'Training..................................................'
    predictor.fit(train_data, pre_train_data, epochs=epochs, verbose=1, batch_size=batch_size, shuffle=True,
                  validation_split=0.2)

    # Negative sampling
    # model.fit_generator(generator=sample_imbalanced_data(train_data, train_labels, batch_size),
    #                     steps_per_epoch=len(train_data)/batch_size, epochs=epochs)

    print("--- train %s seconds ---" % (time.time() - start_time))

    predicted_sensor = predictor.predict(np.array(train_data), batch_size=batch_size, verbose=0)
    predicted_sensor = np.array([x.reshape(59, 1) for x in predicted_sensor])

    classifier = Sequential()

    classifier.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=(pre, 1)))
    # model.add(BatchNormalization(axis=1))
    classifier.add(Activation('relu'))
    classifier.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    classifier.add(Activation('relu'))
    classifier.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    classifier.add(Activation('relu'))

    classifier.add(Flatten())
    classifier.add(Dense(32, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(1, activation='sigmoid'))

    classifier.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
    print classifier.summary()

    print 'Training..................................................'
    classifier.fit(predicted_sensor, train_labels, epochs=epochs, verbose=1, batch_size=batch_size, shuffle=True,
              validation_split=0.2, class_weight={1: class_weight, 0: 1})

    pre_predicted = predictor.predict(np.array(train_data), batch_size=batch_size, verbose=0)
    original_classified = classifier.predict(np.array([x.reshape(59, 1) for x in pre_predicted]), batch_size=batch_size, verbose=0)
    best_threshold = find_best_threshold(train_labels, original_classified)
    classified = [get_class(x, best_threshold) for x in original_classified]

    def time_show(dtime):
        arr = []
        for i in range(len(dtime)):
            arr.append(dtime[i+1])
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

    # list_index = []
    # for val in vals:
    #     print "................................................................."
    #     print val
    #     validation = pd.read_csv('processed_data/all/%s/%s/new/%s.csv' % (type_data, z, val))
    #     # validation = validation[~validation["time_positions"].isin(exclude)]
    #     # validation = validation[validation["year"].isin([2013, 2014, 2015, 2016])]
    #
    #     validation_data = np.array([np.fromstring(red, dtype=float, sep=',')[:len_ts].reshape(len_ts, 1)
    #     for red in validation[sensor]])
    #     print validation_data.shape
    #
    #     validation_labels = validation.label.values
    #     print "validation", len(validation)
    #     print "1 in validation", sum(validation_labels)
    #     original_classified = model.predict(np.array(validation_data), batch_size=batch_size, verbose=0)
    #
    #     classified = [np.around(x[0]) for x in original_classified]
    #     f_score = f1_score(validation_labels, classified, average=None)
    #     print "f_score", f_score
    #     print "f_score AVG", np.average(f_score)
    #     confusion = confusion_matrix(validation_labels, classified)
    #     print "confusion matrix"
    #     print confusion
    #
    #     print "best threshold"
    #     classified = [get_class(x, best_threshold) for x in original_classified]
    #     f_score = f1_score(validation_labels, classified, average=None)
    #     print "f_score", f_score
    #     print "f_score AVG", np.average(f_score)
    #     confusion = confusion_matrix(validation_labels, classified)
    #     print "confusion matrix"
    #     print confusion
    #
    #     # label = 1
    #     # for i in range(len(classified)):
    #     #     c = classified[i]
    #     #     v = validation_labels[i]
    #     #     if c == label and v == label:
    #     #         list_index.append(i)
    #
    #     # validation_time_positions = validation.time_positions.values
    #     # true_time_distribution = Counter()
    #     # false_time_distribution = Counter()
    #     # for i in range(len(validation_labels)):
    #     #     if validation_labels[i] == 1:
    #     #         time_pos = validation_time_positions[i]
    #     #         if classified[i] == 1:
    #     #             true_time_distribution[time_pos] += 1
    #     #         else:
    #     #             false_time_distribution[time_pos] += 1
    #     # print "true", time_show(true_time_distribution)
    #     # print "false", time_show(false_time_distribution)
    #
    # np.savetxt("processed_data/visual/%s/es_pattern.txt" % type_data, list_index,
    #      fmt='%s', delimiter=',', newline='\n')
    # print "len list_index", len(list_index)

print("--- %s seconds ---" % (time.time() - start_time))
