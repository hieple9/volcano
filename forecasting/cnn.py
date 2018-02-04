import tensorflow as tf
import pandas as pd
import numpy as np
import random
from collections import Counter
import sys
import scipy as sp

sys.path.append("../")
from utils import utils


class CNNConfig:
    def __init__(self):
        self.input_size = 1
        self.num_steps = 100
        self.output_size = 1
        self.keep_prob = 0.5
        self.batch_size = 64
        self.learning_rate = 0.01
        self.max_epoch = 10
        self.pos_weight = 3


for training_set, validation_set in [("training_101_60_explosion", "validation_101_60_explosion")]:
    config = CNNConfig()
    # config.learning_rate = learning_rate
    tf.reset_default_graph()
    cnn_graph = tf.Graph()

    with cnn_graph.as_default():
        # Data to feed
        inputs = tf.placeholder(tf.float32, [None, config.num_steps, config.input_size], name="input")
        targets = tf.placeholder(tf.float32, [None, config.output_size], name="target")
        learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        keep_prob = tf.placeholder_with_default(1.0, None, name="keep_prob")

        # CNN
        filters1 = tf.Variable(tf.truncated_normal([5, 1, 128]))
        cnn_output = tf.nn.conv1d(inputs, filters=filters1, stride=1, padding="SAME")
        cnn_output = tf.nn.relu(cnn_output)

        filters2 = tf.Variable(tf.truncated_normal([5, 128, 128]))
        cnn_output = tf.nn.conv1d(cnn_output, filters=filters2, stride=1, padding="SAME")
        cnn_output = tf.nn.relu(cnn_output)

        filters3 = tf.Variable(tf.truncated_normal([5, 128, 128]))
        cnn_output = tf.nn.conv1d(cnn_output, filters=filters3, stride=1, padding="SAME")
        cnn_output = tf.nn.relu(cnn_output)

        filters4 = tf.Variable(tf.truncated_normal([5, 128, 128]))
        cnn_output = tf.nn.conv1d(cnn_output, filters=filters4, stride=1, padding="SAME")
        cnn_output = tf.nn.relu(cnn_output)

        cnn_output = tf.contrib.layers.flatten(cnn_output)

        # Fully connected network
        weight1 = tf.Variable(tf.truncated_normal([100*128, 32]))
        bias1 = tf.Variable(tf.constant(0.1, shape=[32]))
        logits1 = tf.matmul(cnn_output, weight1) + bias1
        logits1 = tf.nn.relu(logits1)
        logits1 = tf.nn.dropout(logits1, keep_prob)

        weight2 = tf.Variable(tf.truncated_normal([32, config.output_size]))
        bias2 = tf.Variable(tf.constant(0.1, shape=[config.output_size]))
        logits = tf.matmul(logits1, weight2) + bias2

        # Loss function
        loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(targets=targets, logits=logits, pos_weight=config.pos_weight))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # Prediction
        prediction = tf.nn.sigmoid(logits)

    """
    Data loading
    """
    # training_set = "training_60_explosion_half"
    # validation_set = "validation_60_explosion_half"
    sensor = "tangential_strain"

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
    # validation_reader = validation_reader[validation_reader.current_labels == 0]
    # print len(validation_reader)
    validation_data, validation_labels = utils.read_data(validation_reader, sensor, num_steps=config.num_steps,
                                                         dim=config.input_size, pre=True)

    with tf.Session(graph=cnn_graph) as sess:
        tf.global_variables_initializer().run()
        train_score = []
        validation_score = []
        for epoch in range(config.max_epoch):
            print("epoch ****************************************************************************", epoch)
            # lr_decay = config.learing_rate_decay ** max(float(epoch + 1 - config.init_epoch), 0.0)
            # current_lr = config.init_learning_rate * lr_decay
            # print("learning_rate............................", current_lr)

            # batch_index = 0
            for batch_X, batch_Y in utils.sample_imbalanced_data(train_data, train_labels, config.batch_size):
                train_data_feed = {
                    inputs: batch_X,
                    targets: batch_Y,
                    learning_rate: config.learning_rate,
                    keep_prob: config.keep_prob
                }
                train_loss, _ = sess.run([loss, optimizer], train_data_feed)
                # batch_index += 1
                # print accuracy of train model for each 50 batch
                # if batch_index % 1000 == 0:
                #     print("train_loss: ", train_loss)

            # output training score and validation score for each epoch

            for each in [[train_data, train_labels, "training", training_reader],
                         [validation_data, validation_labels, "validation", validation_reader]]:
                data = each[0]
                labels = each[1]
                which_data = each[2]
                reader = each[3]
                final_predict = np.array([])
                print("Data.....................................................................", which_data)
                for batch_X, batch_Y in utils.generate_one_epoch(data, labels, config.batch_size, config.num_steps,
                                                                 shuffle=False):
                    data_feed = {
                        inputs: batch_X,
                        keep_prob: 1.0
                    }
                    predict = sess.run([prediction], data_feed)
                    predict = predict[0]
                    final_predict = np.append(final_predict, predict)
                final_predict = np.array(final_predict).reshape(len(final_predict), 1)
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

                    # classified = [utils.get_class(x, best_threshold) for x in final_predict]
                    # # Check in all the cases we correctly predict label is explosive 1, how many have the current labels 1
                    # current_check = []
                    # current_labels = reader.current_labels.values
                    # for index, value in enumerate(classified):
                    #     if value + labels[index] == 2:  # correctly recognition 1
                    #         current = current_labels[index]
                    #         if current >= 1:
                    #             current_check.append(1)
                    #         else:
                    #             current_check.append(0)
                    # print("current_check", len(current_check))
                    # print("sum", sum(current_check))
                    # print("len", len([x for x in current_check if x != 0]))
                    #
                    #
                    # def time_show(dtime):
                    #     arr = []
                    #     for i in range(len(dtime)):
                    #         arr.append(dtime[i + 1])
                    #     return arr
                    #
                    #
                    # train_time_positions = reader.pre_time_positions.values
                    # train_deflation = reader.deflation.values
                    # true_time_distribution = Counter()
                    # false_time_distribution = Counter()
                    # true_deflation = []
                    # false_deflation = []
                    # for i in range(len(labels)):
                    #     if labels[i] == 1:
                    #         time_pos = train_time_positions[i]
                    #         if classified[i] == 1:
                    #             true_time_distribution[time_pos] += 1
                    #             true_deflation.append(train_deflation[i])
                    #         else:
                    #             false_time_distribution[time_pos] += 1
                    #             false_deflation.append(train_deflation[i])
                    #
                    # print("true")
                    # print(time_show(true_time_distribution))
                    # utils.get_statistics(true_deflation)
                    # print("false")
                    # print(time_show(false_time_distribution))
                    # utils.get_statistics(false_deflation)

        print validation_set
        # print("learning_rate >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>...", learning_rate)
        print(train_score)
        print(validation_score)
