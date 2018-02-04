from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, MaxPooling1D
from keras.layers.convolutional import Conv1D
import sys

sys.path.append("../")
from utils.utils import *

np.random.seed(7)

sensor = 'tangential_strain'
run_test = False
batch_size = 1000
filters = 128
kernel_size = 5
epochs = 50

training_reader = pd.read_csv(get_diff_path("training"))
train_data, train_labels = read_data(training_reader, sensor)
dim = train_data[0].shape

# train_reader = pd.read_csv(get_raw_path("training"))
# train_data, train_labels = read_data(train_reader, sensor, dim=101)
# train_data = np.array([diff(x) for x in train_data])
# print train_data.shape
# dim = train_data[0].shape

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
          validation_split=0.2, shuffle=True)

original_classified = model.predict(train_data, batch_size=batch_size, verbose=0)
best_threshold = find_best_threshold(train_labels, original_classified)

validation_reader = pd.read_csv(get_raw_path("test"))
validation_data, validation_labels = read_data(validation_reader, sensor, dim=101)
validation_data = np.array([diff(x) for x in validation_data])

original_classified = model.predict(validation_data, batch_size=batch_size, verbose=0)
print "0.5 threshold"
classified = [np.around(x[0]) for x in original_classified]
get_score_and_confusion_matrix(validation_labels, classified)

print "best threshold"
classified = [get_class(x, best_threshold) for x in original_classified]
get_score_and_confusion_matrix(validation_labels, classified)
