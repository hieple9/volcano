import tensorflow as tf
import pandas as pd
import numpy as np
from collections import Counter
import sys
import scipy as sp

sys.path.append("../")
from utils import utils


class RNNConfig:
    def __init__(self):
        self.input_size = 10
        self.num_steps = 10
        self.output_size = 1
        self.lstm_size = 128
        self.num_layers = 1
        self.keep_prob = 0.5
        self.batch_size = 64
        self.learning_rate = 0.001
        self.max_epoch = 10
        self.pos_weight = 3


for kk in [10]:
    config = RNNConfig()
    # config.input_size = input_size
    # config.num_steps = 100 / input_size
    # config.pos_weight = pos_weight
    tf.reset_default_graph()
    lstm_graph = tf.Graph()

    with lstm_graph.as_default():
        # Data to feed
        inputs = tf.placeholder(tf.float32, [None, config.num_steps, config.input_size], name="input")
        targets = tf.placeholder(tf.float32, [None, config.output_size], name="target")
        learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")


        def _create_one_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(config.lstm_size, state_is_tuple=True)
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)


        # LSTM part
        cell_b = tf.contrib.rnn.MultiRNNCell(
            [_create_one_cell() for _ in range(config.num_layers)],
            state_is_tuple=True
        ) if config.num_layers > 1 else _create_one_cell()

        cell_f = tf.contrib.rnn.MultiRNNCell(
            [_create_one_cell() for _ in range(config.num_layers)],
            state_is_tuple=True
        ) if config.num_layers > 1 else _create_one_cell()

        val, _ = tf.nn.bidirectional_dynamic_rnn(cell_b, cell_f, inputs, dtype=tf.float32, time_major=False)
        forward, backward = val
        val = tf.concat([forward, backward], axis=2)
        val = tf.transpose(val, [1, 0, 2])  # num_steps, batch, lstm_size
        # get the last element of num_steps tensor (batch_size, lstm_size)
        last_output = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")

        # concatenate everything ???
        # last_output = tf.reshape(val, [tf.shape(val)[0], config.num_steps*config.lstm_size])

        # Fully connected network
        # weight = tf.Variable(tf.truncated_normal([config.lstm_size, config.output_size]))
        # bias = tf.Variable(tf.constant(0.1, shape=[config.output_size]))
        # logits = tf.matmul(last_output, weight) + bias

        weight1 = tf.Variable(tf.truncated_normal([config.lstm_size*2, 32]))
        bias1 = tf.Variable(tf.constant(0.1, shape=[32]))
        logits1 = tf.matmul(last_output, weight1) + bias1
        logits1 = tf.nn.relu(logits1)
        logits1 = tf.nn.dropout(logits1, keep_prob)

        weight2 = tf.Variable(tf.truncated_normal([32, config.output_size]))
        bias2 = tf.Variable(tf.constant(0.1, shape=[config.output_size]))
        logits = tf.matmul(logits1, weight2) + bias2

        # Prediction
        prediction = tf.nn.sigmoid(logits)

        # Loss function
        loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(targets=targets, logits=logits, pos_weight=config.pos_weight))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    """
    Data loading
    """
    sensor = "tangential_strain"
    training_set = "training_101_60_explosion"
    validation_set = "validation_101_60_explosion"

    print("Data loading")
    print(sensor)
    training_reader = pd.read_csv(utils.get_early_diff_path(training_set))
    # print len(training_reader)
    # training_reader = training_reader[training_reader.current_labels == 0]
    # print len(training_reader)
    train_data, train_labels = utils.read_data(training_reader, sensor, num_steps=config.num_steps,
                                               dim=config.input_size, pre=True)

    validation_reader = pd.read_csv(utils.get_early_diff_path(validation_set))
    # print len(validation_reader)
    # validatfinal_prediction_reader = validation_reader[validation_reader.current_labels == 0]
    # print len(validation_reader)
    validation_data, validation_labels = utils.read_data(validation_reader, sensor, num_steps=config.num_steps,
                                                         dim=config.input_size, pre=True)

    with tf.Session(graph=lstm_graph) as sess:
        tf.global_variables_initializer().run()
        train_score = []
        validation_score = []
        for epoch in range(config.max_epoch):
            print("epoch ****************************************************************************", epoch)
            for batch_X, batch_Y in utils.generate_one_epoch(train_data, train_labels, config.batch_size):
                train_data_feed = {
                    inputs: batch_X,
                    targets: batch_Y,
                    learning_rate: config.learning_rate,
                    keep_prob: config.keep_prob
                }
                sess.run([loss, optimizer], train_data_feed)

            # output training score and validation score for each epoch
            for each in [[train_data, train_labels, "training", training_reader],
                         [validation_data, validation_labels, "validation", validation_reader]]:
                data = each[0]
                labels = each[1]
                which_data = each[2]
                reader = each[3]
                final_predict = np.array([])
                print("Data.....................................................................", which_data)
                for batch_X, batch_Y in utils.generate_one_epoch(data, labels, config.batch_size, shuffle=False):
                    data_feed = {
                        inputs: batch_X,
                        keep_prob: 1.0
                    }
                    predict = sess.run([prediction], data_feed)
                    predict = predict[0]
                    final_predict = np.append(final_predict, predict)
                final_predict = np.array(final_predict).reshape(len(final_predict), 1)
                stages = [utils.get_state(x) for x in final_predict]
                non_eruption_stage = [x for x, y in zip(labels, stages) if y == utils.State.non_eruption]
                pre_eruption_stage = [x for x, y in zip(labels, stages) if y == utils.State.pre_eruption]
                warning_stage = [x for x, y in zip(labels, stages) if y == utils.State.warning]
                critical_stage = [x for x, y in zip(labels, stages) if y == utils.State.critical]
                assert len(stages) == sum([len(non_eruption_stage), len(pre_eruption_stage),
                                           len(warning_stage), len(critical_stage)])
                print "non_eruption_stage", len(non_eruption_stage) / float(len(labels)), sum(non_eruption_stage) / (
                float(len(non_eruption_stage)) + 0.001)
                print "pre_eruption_stage", len(pre_eruption_stage) / float(len(labels)), sum(pre_eruption_stage) / (
                float(len(pre_eruption_stage)) + 0.001)
                print "warning_stage", len(warning_stage) / float(len(labels)), sum(warning_stage) / (
                float(len(warning_stage)) + 0.001)
                print "critical_stage", len(critical_stage) / float(len(labels)), sum(critical_stage) / (
                float(len(critical_stage)) + 0.001)

                print("0.5 threshold")
                classified = [utils.get_class(x, 0.5) for x in final_predict]
                f, _ = utils.get_score_and_confusion_matrix(labels, classified)
                print("Best threshold")
                best_threshold = utils.find_best_threshold(labels, final_predict)
                best_classified = [utils.get_class(x, best_threshold) for x in final_predict]
                best_f, _ = utils.get_score_and_confusion_matrix(labels, best_classified)

                if which_data == "training":
                    train_score.append(round(sp.average(best_f) * 100, 2))
                else:
                    validation_score.append(round(sp.average(best_f) * 100, 2))

                classified = [utils.get_class(x, best_threshold) for x in final_predict]
                # Check in all the cases we correctly predict label is explosive 1, how many have the current labels 1
                current_check = []
                current_labels = reader.current_labels.values
                for index, value in enumerate(classified):
                    if value + labels[index] == 2:  # correctly recognition 1
                        current = current_labels[index]
                        if current >= 1:
                            current_check.append(1)
                        else:
                            current_check.append(0)
                print("current_check", len(current_check))
                print("sum", sum(current_check))
                print("len", len([x for x in current_check if x != 0]))


                def time_show(dtime):
                    arr = []
                    for i in range(len(dtime)):
                        arr.append(dtime[i + 1])
                    return arr


                train_time_positions = reader.pre_time_positions.values
                train_deflation = reader.deflation.values
                true_time_distribution = Counter()
                false_time_distribution = Counter()
                true_deflation = []
                false_deflation = []
                for i in range(len(labels)):
                    if labels[i] == 1:
                        time_pos = train_time_positions[i]
                        if classified[i] == 1:
                            true_time_distribution[time_pos] += 1
                            true_deflation.append(train_deflation[i])
                        else:
                            false_time_distribution[time_pos] += 1
                            false_deflation.append(train_deflation[i])

                print("true")
                print(time_show(true_time_distribution))
                utils.get_statistics(true_deflation)
                print("false")
                print(time_show(false_time_distribution))
                utils.get_statistics(false_deflation)

        print validation_set
        # print("pos_weight >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", pos_weight)
        print(train_score)
        print(validation_score)
