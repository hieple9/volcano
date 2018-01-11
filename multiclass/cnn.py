from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv1D
import time
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import matplotlib as mpl
mpl.use('Agg')
import sys
sys.path.append("../")
from utils.utils import *

np.random.seed(7)

start_time = time.time()
training_set = 'training'
test_sets = ['test']
run_test = True
# sensors = ['energy', 'maximum_amplitude', 'radial_strain', 'tangential_strain']
sensors = ['energy']
batch_size = 1000
filters = 128
kernel_size = 5
epochs = 100
exclude = []

for sensor in sensors:
    print sensor
    training_reader = pd.read_csv(get_diff_path(training_set))
    train_data, train_labels = read_data(training_reader, sensor, True, exclude)
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

    print "Training....................................................."
    model.fit(train_data, train_labels, epochs=epochs, verbose=1, batch_size=batch_size,
              validation_split=0.2, shuffle=True)

    print("--- train %s seconds ---" % (time.time() - start_time))

    if run_test:
        print 'Testing....................................'
        for test in test_sets:
            print test
            test_reader = pd.read_csv(get_diff_path(test))
            test_data, test_labels = read_data(test_reader, sensor, True, exclude)

            test_labels = encoder.transform(test_labels)
            classified = model.predict_classes(test_data, batch_size=batch_size, verbose=0)
            score, confusion = get_score_and_confusion_matrix(test_labels, classified)

            if num_class == 11:
                class_range = range(0, num_class)
            else:
                class_range = range(1, num_class)
            # # Plot non-normalized confusion matrix
            # plt.figure()
            # plot_confusion_matrix(confusion, classes=class_range,
            #                       title='Confusion matrix, without normalization %s' % test)
            # plt.savefig("not_normalized_%s_%sclass" % (test, num_class))

            # Plot normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(confusion, classes=class_range, normalize=True,
                                  title='')
            plt.savefig("normalized_cnn_%sclass" % num_class)

print("--- %s seconds ---" % (time.time() - start_time))
