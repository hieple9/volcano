from keras.models import Model
from keras.layers import Dropout, Flatten, Activation, Dense, Input, concatenate, MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import time
import sys

sys.path.append("../")
from utils.utils import *
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(7)

start_time = time.time()
tests = ["test"]
training_set = 'training'
run_test = True
batch_size = 1000
filters = 128
kernel_size = 5
epochs = 100
exclude = [0]

training_reader = pd.read_csv(get_diff_path(training_set))
energies, maximum_amplitudes, radial_strains, tangential_strains, train_labels = read_data(training_reader, None, True, exclude)
dim = radial_strains[0].shape

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(train_labels)
encoded_labels = encoder.transform(train_labels)
# convert integers to dummy variables (i.e. one hot encoded)
num_class = len(set(encoded_labels))
train_labels = np_utils.to_categorical(encoded_labels)

# Create the model
energy_input = Input(shape=dim, name="ei")
energy_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(energy_input)
energy_layer = Activation('relu')(energy_layer)
energy_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(energy_layer)
energy_layer = Activation('relu')(energy_layer)
energy_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(energy_layer)
energy_layer = Activation('relu')(energy_layer)
energy_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(energy_layer)
energy_layer = Activation('relu')(energy_layer)
energy_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(energy_layer)
energy_layer = Activation('relu')(energy_layer)
energy_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(energy_layer)
energy_layer = Activation('relu')(energy_layer)
energy_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(energy_layer)
energy_layer = Activation('relu')(energy_layer)
energy_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(energy_layer)
energy_layer = Activation('relu')(energy_layer)
energy_layer = (Dense(32, activation='relu'))(energy_layer)
energy_layer = Dropout(0.2)(energy_layer)
energy_layer = Flatten()(energy_layer)

maximum_amplitude_input = Input(shape=dim, name='di')
maximum_amplitude_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(maximum_amplitude_input)
maximum_amplitude_layer = Activation('relu')(maximum_amplitude_layer)
maximum_amplitude_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(maximum_amplitude_layer)
maximum_amplitude_layer = Activation('relu')(maximum_amplitude_layer)
maximum_amplitude_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(maximum_amplitude_layer)
maximum_amplitude_layer = Activation('relu')(maximum_amplitude_layer)
maximum_amplitude_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(maximum_amplitude_layer)
maximum_amplitude_layer = Activation('relu')(maximum_amplitude_layer)
maximum_amplitude_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(maximum_amplitude_layer)
maximum_amplitude_layer = Activation('relu')(maximum_amplitude_layer)
maximum_amplitude_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(maximum_amplitude_layer)
maximum_amplitude_layer = Activation('relu')(maximum_amplitude_layer)
maximum_amplitude_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(maximum_amplitude_layer)
maximum_amplitude_layer = Activation('relu')(maximum_amplitude_layer)
maximum_amplitude_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(maximum_amplitude_layer)
maximum_amplitude_layer = Activation('relu')(maximum_amplitude_layer)
maximum_amplitude_layer = (Dense(32, activation='relu'))(maximum_amplitude_layer)
maximum_amplitude_layer = Dropout(0.5)(maximum_amplitude_layer)
maximum_amplitude_layer = Flatten()(maximum_amplitude_layer)

radial_strain_input = Input(shape=dim, name='ri')
radial_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(radial_strain_input)
radial_strain_layer = Activation('relu')(radial_strain_layer)
radial_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(radial_strain_layer)
radial_strain_layer = Activation('relu')(radial_strain_layer)
radial_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(radial_strain_layer)
radial_strain_layer = Activation('relu')(radial_strain_layer)
radial_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(radial_strain_layer)
radial_strain_layer = Activation('relu')(radial_strain_layer)
radial_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(radial_strain_layer)
radial_strain_layer = Activation('relu')(radial_strain_layer)
radial_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(radial_strain_layer)
radial_strain_layer = Activation('relu')(radial_strain_layer)
radial_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(radial_strain_layer)
radial_strain_layer = Activation('relu')(radial_strain_layer)
radial_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(radial_strain_layer)
radial_strain_layer = Activation('relu')(radial_strain_layer)
radial_strain_layer = (Dense(32, activation='relu'))(radial_strain_layer)
radial_strain_layer = Dropout(0.5)(radial_strain_layer)
radial_strain_layer = Flatten()(radial_strain_layer)

tangential_strain_input = Input(shape=dim, name='bi')
tangential_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(tangential_strain_input)
tangential_strain_layer = Activation('relu')(tangential_strain_layer)
tangential_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(tangential_strain_layer)
tangential_strain_layer = Activation('relu')(tangential_strain_layer)
tangential_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(tangential_strain_layer)
tangential_strain_layer = Activation('relu')(tangential_strain_layer)
tangential_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(tangential_strain_layer)
tangential_strain_layer = Activation('relu')(tangential_strain_layer)
tangential_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(tangential_strain_layer)
tangential_strain_layer = Activation('relu')(tangential_strain_layer)
tangential_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(tangential_strain_layer)
tangential_strain_layer = Activation('relu')(tangential_strain_layer)
tangential_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(tangential_strain_layer)
tangential_strain_layer = Activation('relu')(tangential_strain_layer)
tangential_strain_layer = (Conv1D(filters=filters, kernel_size=kernel_size))(tangential_strain_layer)
tangential_strain_layer = Activation('relu')(tangential_strain_layer)
tangential_strain_layer = (Dense(32, activation='relu'))(tangential_strain_layer)
tangential_strain_layer = Dropout(0.5)(tangential_strain_layer)
tangential_strain_layer = Flatten()(tangential_strain_layer)

seismic_merge = concatenate([energy_layer, maximum_amplitude_layer])
# seismic_merge = (Conv1D(filters=filters, kernel_size=kernel_size))(seismic_merge)
# seismic_merge = Activation('relu')(seismic_merge)
# seismic_merge = Flatten()(seismic_merge)
seismic_model = (Dense(32, activation='relu'))(seismic_merge)
seismic_model = Dropout(0.2)(seismic_model)

strain_merge = concatenate([radial_strain_layer, tangential_strain_layer])
# strain_merge = (Conv1D(filters=filters, kernel_size=kernel_size))(strain_merge)
# strain_merge = Activation('relu')(strain_merge)
# strain_merge = Flatten()(strain_merge)
strain_model = (Dense(32, activation='relu'))(strain_merge)
strain_model = Dropout(0.2)(strain_model)

merge = concatenate([seismic_model, strain_model])
merge = (Dense(32, activation='relu'))(merge)
merge = Dropout(0.5)(merge)
model = (Dense(num_class, activation='softmax'))(merge)

model = Model(inputs=[energy_input, maximum_amplitude_input, radial_strain_input, tangential_strain_input],
              outputs=model)
model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
print model.summary()

print "Training....................................................."
model.fit([energies, maximum_amplitudes, radial_strains, tangential_strains], train_labels, epochs=epochs, verbose=1,
          batch_size=batch_size, shuffle=True, validation_split=0.2)

print("--- train %s seconds ---" % (time.time() - start_time))

if run_test:
    print 'Testing..................................................'
    for test in tests:
        print test
        test_data = pd.read_csv(get_diff_path(test))
        energy_test, maximum_amplitude_test, radial_strain_test, tangential_strain_test, test_labels = read_data(test_data,
                                                                                                   None, True, exclude)

        test_labels = encoder.transform(test_labels)
        classified = model.predict([energy_test, maximum_amplitude_test, radial_strain_test, tangential_strain_test],
                                   batch_size=batch_size, verbose=0)
        classified = np.argmax(classified, axis=1)
        f_score = f1_score(test_labels, classified, average=None)

        print "f_score", f_score
        print "f_score AVG", np.average(f_score)
        confusion = confusion_matrix(test_labels, classified)
        print "confusion matrix"
        print confusion

        if num_class == 11:
            class_range = range(0, num_class)
        else:
            class_range = range(1, num_class)
        # # Plot non-normalized confusion matrix
        # plt.figure()
        # plot_confusion_matrix(confusion, classes=class_range,
        #                       title='')
        # plt.savefig("not_normalized_dmmf_%sclass" % num_class)

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(confusion, classes=class_range, normalize=True,
                              title='')
        plt.savefig("normalized_dmmf_%sclass" % num_class)

print("--- %s seconds ---" % (time.time() - start_time))
