from keras.models import Model
from keras.layers import Dropout, Flatten, Activation, Dense, Input, concatenate, MaxPooling1D
from keras.layers.convolutional import Conv1D
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
filters = 32
kernel_size = 16
epochs = 30
label_zero_ratio = 2

training_reader = pd.read_csv(get_diff_path(training_set))
energies, maximum_amplitudes, radial_strains, tangential_strains, train_labels = read_data(training_reader)
dim = radial_strains[0].shape

# Create the model
energy_input = Input(shape=dim, name="ei")
energy_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(energy_input)
energy_layer = Activation('relu')(energy_layer)
energy_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(energy_layer)
energy_layer = Activation('relu')(energy_layer)
energy_layer = (MaxPooling1D(pool_size=2, strides=2))(energy_layer)
energy_layer = Flatten()(energy_layer)
energy_layer = (Dense(32, activation='relu'))(energy_layer)
energy_layer = Dropout(0.5)(energy_layer)
energy_layer = (Dense(1, activation='sigmoid'))(energy_layer)

maximum_amplitude_input = Input(shape=dim, name='di')
maximum_amplitude_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(maximum_amplitude_input)
maximum_amplitude_layer = Activation('relu')(maximum_amplitude_layer)
maximum_amplitude_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(maximum_amplitude_layer)
maximum_amplitude_layer = Activation('relu')(maximum_amplitude_layer)
maximum_amplitude_layer = (MaxPooling1D(pool_size=2, strides=2))(maximum_amplitude_layer)
maximum_amplitude_layer = Flatten()(maximum_amplitude_layer)
maximum_amplitude_layer = (Dense(32, activation='relu'))(maximum_amplitude_layer)
maximum_amplitude_layer = Dropout(0.5)(maximum_amplitude_layer)
maximum_amplitude_layer = (Dense(1, activation='sigmoid'))(maximum_amplitude_layer)

radial_strain_input = Input(shape=dim, name='ri')
radial_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(radial_strain_input)
radial_strain_layer = Activation('relu')(radial_strain_layer)
radial_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(radial_strain_layer)
radial_strain_layer = Activation('relu')(radial_strain_layer)
radial_strain_layer = (MaxPooling1D(pool_size=2, strides=2))(radial_strain_layer)
radial_strain_layer = Flatten()(radial_strain_layer)
radial_strain_layer = (Dense(32, activation='relu'))(radial_strain_layer)
radial_strain_layer = Dropout(0.5)(radial_strain_layer)
radial_strain_layer = (Dense(1, activation='sigmoid'))(radial_strain_layer)

tangential_strain_input = Input(shape=dim, name='bi')
tangential_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(tangential_strain_input)
tangential_strain_layer = Activation('relu')(tangential_strain_layer)
tangential_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(tangential_strain_layer)
tangential_strain_layer = Activation('relu')(tangential_strain_layer)
tangential_strain_layer = (MaxPooling1D(pool_size=2, strides=2))(tangential_strain_layer)
tangential_strain_layer = Flatten()(tangential_strain_layer)
tangential_strain_layer = (Dense(32, activation='relu'))(tangential_strain_layer)
tangential_strain_layer = Dropout(0.5)(tangential_strain_layer)
tangential_strain_layer = (Dense(1, activation='sigmoid'))(tangential_strain_layer)

merge = concatenate([energy_layer, maximum_amplitude_layer, radial_strain_layer, tangential_strain_layer])
model = (Dense(1, activation='sigmoid'))(merge)

model = Model(inputs=[energy_input, maximum_amplitude_input, radial_strain_input, tangential_strain_input],
              outputs=model)
model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
print model.summary()

print "Training....................................................."
model.fit([energies, maximum_amplitudes, radial_strains, tangential_strains], train_labels, epochs=epochs, verbose=1,
          batch_size=batch_size,
          class_weight={1: label_zero_ratio, 0: 1}, shuffle=True, validation_split=0.2)

print model.get_weights()
print("--- train %s seconds ---" % (time.time() - start_time))

original_classified = model.predict([energies, maximum_amplitudes, radial_strains, tangential_strains],
                                    batch_size=batch_size, verbose=0)
best_threshold = find_best_threshold(train_labels, original_classified)

if run_test:
    print 'Testing..................................................'
    for test in test_sets:
        print test
        test_reader = pd.read_csv(get_diff_path(test))
        energy_test, maximum_amplitude_test, radial_strain_test, tangential_strain_test, test_labels = read_data(
            test_reader)

        original_classified = model.predict(
            [energy_test, maximum_amplitude_test, radial_strain_test, tangential_strain_test], batch_size=batch_size,
            verbose=0)

        print "0.5 threshold"
        classified = [np.around(x[0]) for x in original_classified]
        get_score_and_confusion_matrix(test_labels, classified)

        print "best threshold"
        classified = [get_class(x, best_threshold) for x in original_classified]
        get_score_and_confusion_matrix(test_labels, classified)

print("--- %s seconds ---" % (time.time() - start_time))
