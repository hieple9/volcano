import tensorflow as tf
import pandas as pd
import random
import numpy as np

import sys
sys.path.append("../")
from utils import utils


def generate_one_epoch(batch_size, num_steps):
    num_batches = int(len(train_data)) // batch_size
    # if batch_size * num_batches < len(train_data):
    #     num_batches += 1

    batch_indices = range(num_batches)
    random.shuffle(batch_indices)
    for j in batch_indices:
        batch_X = train_data[j * batch_size: (j + 1) * batch_size]
        batch_y = train_labels[j * batch_size: (j + 1) * batch_size].reshape(batch_size, config.output_size)
        assert set(map(len, batch_X)) == {num_steps}
        yield batch_X, batch_y


class RNNConfig():
    def __init__(self):
        pass
    input_size = 20
    num_steps = 5
    output_size = 1
    lstm_size = 128
    num_layers = 2
    keep_prob = 0.5
    batch_size = 64
    init_learning_rate = 0.001
    learing_rate_decay = 0.99
    init_epoch = 5
    max_epoch = 20

config = RNNConfig()
tf.reset_default_graph()
lstm_graph = tf.Graph()

with lstm_graph.as_default():
    # Data to feed
    inputs = tf.placeholder(tf.float32, [None, config.num_steps, config.input_size], name="input")
    targets = tf.placeholder(tf.float32, [None, config.output_size], name="target")
    learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
    keep_prob = tf.placeholder_with_default(1.0, None, name="keep_prob")

    def _create_one_cell():
        lstm_cell = tf.contrib.rnn.LSTMCell(config.lstm_size, state_is_tuple=True)
        # if keep_prob < 1.0:
        return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)

    # LSTM part
    # cell = _create_one_cell()
    cell = tf.contrib.rnn.MultiRNNCell(
        [_create_one_cell() for _ in range(config.num_layers)],
        state_is_tuple=True
    ) if config.num_layers > 1 else _create_one_cell()
    val, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    val = tf.transpose(val, [1, 0, 2])
    last_output = tf.gather(val, int(val.get_shape()[0])-1, name="last_lstm_output")

    # Fully connected network
    weight = tf.Variable(tf.truncated_normal([config.lstm_size, config.output_size]))
    bias = tf.Variable(tf.constant(0.1, shape=[config.output_size]))
    logits = tf.matmul(last_output, weight) + bias

    # Loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))

    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    # Prediction
    prediction = tf.nn.sigmoid(logits)

"""
Data loading
"""
training_set = "training_60_explosion"
validation_set = "test_60_explosion"
sensor = "tangential_strain"

print "Data loading"
print sensor
training_reader = pd.read_csv(utils.get_early_diff_path(training_set))
print len(training_reader)
training_reader = training_reader[training_reader.current_labels == 0]
print len(training_reader)
train_data, train_labels = utils.read_data(training_reader, sensor,  num_steps=5, dim=20, pre=True)

validation_reader = pd.read_csv(utils.get_early_diff_path(validation_set))
print len(validation_reader)
validation_reader = validation_reader[validation_reader.current_labels == 0]
print len(validation_reader)
validation_data, validation_labels = utils.read_data(validation_reader, sensor,  num_steps=5, dim=20, pre=True)

with tf.Session(graph=lstm_graph) as sess:
    tf.global_variables_initializer().run()
    for epoch in range(config.max_epoch):
        print("epoch", epoch)
        lr_decay = config.learing_rate_decay ** max(float(epoch + 1 - config.max_epoch), 0.0)
        current_lr = config.init_learning_rate * lr_decay

        batch_index = 0
        for batch_X, batch_Y in generate_one_epoch(config.batch_size, config.num_steps):
            train_data_feed = {
                inputs: batch_X,
                targets: batch_Y,
                learning_rate: current_lr,
                keep_prob: config.keep_prob
            }
            train_loss, _ = sess.run([loss, optimizer], train_data_feed)
            batch_index += 1
            # print accuracy of train model for each 50 batch
            if batch_index % 1000 == 0:
                print("train_loss: ", train_loss)
        # output validation score for each epoch
        print("Training score")
        validation_data_feed = {
            inputs: train_data,
            targets: train_labels.reshape(len(train_labels), config.output_size),
            learning_rate: 1.0,
            keep_prob: 1.0
        }
        validation_prediction = sess.run([prediction], validation_data_feed)
        validation_prediction = np.array(validation_prediction).reshape(len(train_labels), 1)
        # print validation_prediction.shape
        # print validation_labels.shape
        print "0.5 threshold"
        classified = [utils.get_class(x, 0.5) for x in validation_prediction]
        utils.get_score_and_confusion_matrix(train_labels, classified)
        print "Best threshold"
        best_threshold = utils.find_best_threshold(train_labels, validation_prediction)

