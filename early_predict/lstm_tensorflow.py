import tensorflow as tf
from utils import utils
import pandas as pd
import random

tf.reset_default_graph()
lstm_graph = tf.Graph()


class RNNConfig():
    def __init__(self):
        pass
    input_size = 20
    num_steps = 5
    output_size = 1
    lstm_size = 128
    num_layers = 1
    keep_prob = 0.5
    batch_size = 64
    init_learning_rate = 0.001
    learing_rate_decay = 0.99
    init_epoch = 5
    max_epoch = 10

config = RNNConfig()

with lstm_graph.as_default():
    inputs = tf.placeholder(tf.float32, [None, config.num_steps, config.input_size])
    targets = tf.placeholder(tf.float32, [None, config.output_size])
    learning_rate = tf.placeholder(tf.float32, None)

    def _create_one_cell():
        lstm_cell = tf.contrib.rnn.LSTMCell(config.lstm_size, state_is_tuple=True)
        if config.keep_prob < 1.0:
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)

    cell = _create_one_cell()
    val, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    val = tf.transpose(val, [1, 0, 2])
    last_output = tf.gather(val, int(val.get_shape()[0])-1, name="last_lstm_output")

    weight = tf.Variable(tf.truncated_normal([config.lstm_size, config.output_size]))
    bias = tf.Variable(tf.constant(0.1, shape=[config.lstm_size]))
    logits = tf.matmul(last_output, weight) + bias

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    minimize = optimizer.minimize(loss)


"""
Data loading
"""
training_set = "training_60_explosion"
# validation_set = "validation_60_explosion"
sensor = "tangential_strain"

print "Data loading"
print sensor
training_reader = pd.read_csv(utils.get_early_diff_path(training_set))
train_data, train_labels = utils.read_data(training_reader, sensor,  num_steps=5, dim=20, pre=True)


def generate_one_epoch(batch_size, num_steps):
    num_batches = int(len(train_data)) // batch_size
    if batch_size * num_batches < len(train_data):
        num_batches += 1

    batch_indices = range(num_batches)
    random.shuffle(batch_indices)
    for j in batch_indices:
        batch_X = train_data[j * batch_size: (j + 1) * batch_size]
        batch_y = train_labels[j * batch_size: (j + 1) * batch_size]
        assert set(map(len, batch_X)) == {num_steps}
        yield batch_X, batch_y


with tf.Session(graph=lstm_graph) as sess:
    tf.global_variables_initializer().run()
    for epoch in range(config.max_epoch):
        lr_decay = config.learing_rate_decay ** max(epoch + 1 - config.max_epoch, 0.0)
        current_lr = config.init_learning_rate * lr_decay

        for batch_X, batch_Y in generate_one_epoch(config.batch_size, config.num_steps):
            train_data_feed = {
                inputs: batch_X,
                targets: batch_Y,
                learning_rate: current_lr
            }
            train_loss, _ = sess.run([loss, minimize], train_data_feed)
            print("train_loss: ", train_loss)
