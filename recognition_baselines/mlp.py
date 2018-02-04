from keras.layers import Dense, Input
from keras.models import Model
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
batch_size = 1000
epochs = 30
label_zero_ratio = 2

for sensor in sensors:
    print "_____________________________________________________________________"
    print sensor

    training_reader = pd.read_csv(get_diff_path(training_set))
    train_data, train_labels = read_data(training_reader, sensor)
    train_data = np.array([x.flatten() for x in train_data])
    dim = len(train_data[0])
    print dim

    # Create the model
    input_layer = Input(shape=(dim,))
    hidden = Dense(50, activation='relu')(input_layer)
    hidden = Dense(10, activation='relu')(hidden)
    output_layer = Dense(1, activation='sigmoid')(hidden)
    model = Model(inputs=input_layer, outputs=output_layer)

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
            test_data = np.array([x.flatten() for x in test_data])

            original_classified = model.predict(test_data, batch_size=batch_size, verbose=0)
            print "0.5 threshold"
            classified = [np.around(x[0]) for x in original_classified]
            get_score_and_confusion_matrix(test_labels, classified)

            print "best threshold"
            classified = [get_class(x, best_threshold) for x in original_classified]
            get_score_and_confusion_matrix(test_labels, classified)

print("--- %s seconds ---" % (time.time() - start_time))
