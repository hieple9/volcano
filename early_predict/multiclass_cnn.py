from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv1D
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
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
filters = 128
kernel_size = 5
epochs = 10

for sensor in sensors:
    print sensor
    training_reader = pd.read_csv(get_early_diff_path(training_set))
    # include multiple class retrieve
    train_data, train_labels = read_data(training_reader, sensor, pre=True)
    dim = train_data[0].shape

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(train_labels)
    encoded_labels = encoder.transform(train_labels)
    # convert integers to dummy variables (i.e. one hot encoded)
    num_class = len(set(encoded_labels))
    train_labels = np_utils.to_categorical(encoded_labels)

    # Create the model
    model = Sequential()

    model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=dim))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_class, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
    print model.summary()

    print 'Training..................................................'
    model.fit(train_data, train_labels, epochs=epochs, verbose=1, batch_size=batch_size, shuffle=True,
              validation_split=0.2)

    print("--- train %s seconds ---" % (time.time() - start_time))

    classified = model.predict_classes(train_data, batch_size=batch_size, verbose=0)
    get_score_and_confusion_matrix(encoded_labels, classified)

    if run_test:
        print 'Test.......................................................'
        for test in test_sets:
            print test
            test_reader = pd.read_csv(get_early_diff_path(test))
            test_data, test_labels = read_data(test_reader, sensor, pre=True)

            test_labels = encoder.transform(test_labels)
            classified = model.predict_classes(test_data, batch_size=batch_size, verbose=0)
            score, confusion = get_score_and_confusion_matrix(test_labels, classified)

print("--- %s seconds ---" % (time.time() - start_time))
