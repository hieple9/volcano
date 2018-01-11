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

print "Training....................................................."
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

model_energy = Model(inputs=energy_input, outputs=energy_layer)
model_energy.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
model_energy.fit(energies, train_labels, epochs=epochs, verbose=1, batch_size=batch_size,
                 class_weight={1: label_zero_ratio, 0: 1}, shuffle=True)
classified_energy = model_energy.predict(energies, batch_size=batch_size, verbose=0)

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

model_maximum_amplitude = Model(inputs=maximum_amplitude_input, outputs=maximum_amplitude_layer)
model_maximum_amplitude.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
model_maximum_amplitude.fit(maximum_amplitudes, train_labels, epochs=epochs, verbose=1, batch_size=batch_size,
                            class_weight={1: label_zero_ratio, 0: 1}, shuffle=True)
classified_maximum_amplitude = model_energy.predict(maximum_amplitudes, batch_size=batch_size, verbose=0)

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

model_radial_strain = Model(inputs=radial_strain_input, outputs=radial_strain_layer)
model_radial_strain.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
model_radial_strain.fit(radial_strains, train_labels, epochs=epochs, verbose=1, batch_size=batch_size,
                        class_weight={1: label_zero_ratio, 0: 1}, shuffle=True)
classified_radial_strain = model_radial_strain.predict(radial_strains, batch_size=batch_size, verbose=0)

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

model_tangential_strain = Model(inputs=tangential_strain_input, outputs=tangential_strain_layer)
model_tangential_strain.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
model_tangential_strain.fit(tangential_strains, train_labels, epochs=epochs, verbose=1, batch_size=batch_size,
                            class_weight={1: label_zero_ratio, 0: 1}, shuffle=True)
classified_tangential_strain = model_tangential_strain.predict(tangential_strains, batch_size=batch_size, verbose=0)

train_input = np.array([np.array([energy, maximum_amplitude, radial_strain, tangential_strain]).reshape(4) for
                        energy, maximum_amplitude, radial_strain, tangential_strain
                        in zip(classified_energy, classified_maximum_amplitude, classified_radial_strain,
                               classified_maximum_amplitude)])

final_dim = len(train_input[0])
final_input = Input(shape=(final_dim,))
final_layer = Dense(1, activation='sigmoid')(final_input)
final_model = Model(inputs=final_input, outputs=final_layer)
print final_model.summary()

final_model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
final_model.fit(train_input, train_labels, epochs=epochs, verbose=1,
                batch_size=batch_size, class_weight={1: label_zero_ratio, 0: 1}, shuffle=True)
print final_model.get_weights()

print("--- train %s seconds ---" % (time.time() - start_time))

original_classified = final_model.predict(train_input, batch_size=batch_size, verbose=0)
best_threshold = find_best_threshold(train_labels, original_classified)

if run_test:
    print 'Testing..................................................'
    for test in test_sets:
        print test
        test_reader = pd.read_csv(get_diff_path(test))
        energy_test, maximum_amplitude_test, radial_strain_test, tangential_strain_test, test_labels = read_data(
            test_reader)

        classified_energy_test = model_energy.predict(energy_test, batch_size=batch_size, verbose=0)
        classified_maximum_amplitude_test = model_maximum_amplitude.predict(maximum_amplitude_test,
                                                                            batch_size=batch_size, verbose=0)
        classified_radial_strain_test = model_radial_strain.predict(radial_strain_test, batch_size=batch_size,
                                                                    verbose=0)
        classified_tangential_strain_test = model_tangential_strain.predict(tangential_strain_test,
                                                                            batch_size=batch_size, verbose=0)

        test_input = np.array([np.array([energy, maximum_amplitude, radial_strain, tangential_strain]).reshape(4) for
                               energy, maximum_amplitude, radial_strain, tangential_strain
                               in zip(classified_energy_test, classified_maximum_amplitude_test,
                                      classified_radial_strain_test, classified_tangential_strain_test)])

        original_classified = final_model.predict(test_input, batch_size=batch_size, verbose=0)

        print "0.5 threshold"
        classified = [np.around(x[0]) for x in original_classified]
        get_score_and_confusion_matrix(test_labels, classified)

        print "best threshold"
        classified = [get_class(x, best_threshold) for x in original_classified]
        get_score_and_confusion_matrix(test_labels, classified)

print("--- %s seconds ---" % (time.time() - start_time))
