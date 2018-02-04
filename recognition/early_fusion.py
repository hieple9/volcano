from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D
import time
import sys

sys.path.append("../")
from utils.utils import *

np.random.seed(7)

start_time = time.time()
training_set = 'training'
test_sets = ['test']
run_test = True
batch_size = 1000
filters = 128
kernel_size = 5
epochs = 50

training_reader = pd.read_csv(get_diff_path(training_set))
energies, maximum_amplitudes, radial_strains, tangential_strains, train_labels = read_data(training_reader)
train_data = np.array(
    [np.array([energy, maximum, radial, tangential]).reshape(100, 4) for energy, maximum, radial, tangential in
     zip(energies, maximum_amplitudes, radial_strains, tangential_strains, )])

dim = train_data[0].shape
print dim

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
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
print model.summary()

print "Training....................................................."
model.fit(train_data, train_labels, epochs=epochs, verbose=1, batch_size=batch_size,
          shuffle=True, validation_split=0.2)

original_classified = model.predict(train_data, batch_size=batch_size, verbose=0)
best_threshold = find_best_threshold(train_labels, original_classified)

if run_test:
    print 'test'
    for test in test_sets:
        print "................................................................."
        print test
        test_reader = pd.read_csv(get_diff_path(test))
        energy_test, maximum_amplitude_test, radial_strain_test, tangential_strain_test, test_labels = read_data(
            test_reader)
        test_data = np.array(
            [np.array([energy, maximum, radial, tangential]).reshape(100, 4) for energy, maximum, radial, tangential in
             zip(energy_test, maximum_amplitude_test, radial_strain_test, tangential_strain_test)])

        original_classified = model.predict(np.array(test_data), batch_size=batch_size, verbose=0)
        print "0.5 threshold"
        classified = [np.around(x[0]) for x in original_classified]
        get_score_and_confusion_matrix(test_labels, classified)

        print "best threshold"
        classified = [get_class(x, best_threshold) for x in original_classified]
        get_score_and_confusion_matrix(test_labels, classified)

print("--- %s seconds ---" % (time.time() - start_time))
