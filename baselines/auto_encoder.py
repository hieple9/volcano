from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import f1_score, confusion_matrix
from utils.utils import *
import time


start_time = time.time()
# type_datas = ['exact', 'pre30', 'pre60']
type_datas = ['pre120']
zs = ['diff']

for type_data in type_datas:
    for z in zs:
        print type_data, z
        training = pd.read_csv('processed_data/%s/%s/training.csv' % (type_data, z))
        # training = training[0:20000]

        train_blues = training.blue.values
        train_blues = [blue.split(",") for blue in train_blues]
        train_labels = training.label.values
        print "training", len(training)
        print "1 in training", sum(train_labels)

        label_zero_ratio = 1

        validation = pd.read_csv('processed_data/%s/%s/validation.csv' % (type_data, z))
        validation_blues = validation.blue.values
        validation_blues = [validation_blues.split(",") for validation_blues in validation_blues]
        validation_labels = validation.label.values
        print "validation", len(validation)
        print "1 in validation", sum(validation_labels)

        vector_length = len(train_blues[0])
        encoding_dimension = 50
        input = Input(shape=(vector_length, ))
        encoded = Dense(encoding_dimension, activation='relu')(input)
        decoded = Dense(vector_length, activation='sigmoid')(encoded)

        auto_encoder = Model(inputs=input, outputs=decoded)
        encoder = Model(input, encoded)
        auto_encoder.compile(optimizer='adadelta', loss='mean_squared_error')
        auto_encoder.fit(train_blues, train_blues,
                         epochs=30, verbose=0,
                         batch_size=256)

        input = Input(shape=(encoding_dimension, ))
        hidden1 = Dense(25, activation='relu')(input)
        hidden2 = Dense(25, activation='relu')(hidden1)
        output = Dense(1, activation='sigmoid')(hidden2)
        model = Model(inputs=input, outputs=output)
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # Train the model, iterating on the data in batches of 32 samples
        encoded_blues = encoder.predict(train_blues)
        model.fit(encoded_blues, train_labels, epochs=35, verbose=0, batch_size=64, class_weight={1: label_zero_ratio, 0: 1})

        # Validation
        validation_blues = encoder.predict(validation_blues)
        classified = model.predict(validation_blues, batch_size=128, verbose=0)
        classified = [np.around(x[0]) for x in classified]
        f_score = f1_score(validation_labels, classified, average=None)
        print "f_score", f_score
        print "f_score AVG", np.average(f_score)
        confusion = confusion_matrix(validation_labels, classified)
        print "confusion matrix"
        print confusion

print("--- %s seconds ---" % (time.time() - start_time))
