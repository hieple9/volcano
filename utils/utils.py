import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from random import random, choice
import itertools
import scipy as sp
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

"""
Functions for reading file
"""


def get_raw_path(file_name):
    return "../processed_data/exact/raw/%s.csv" % file_name


def get_diff_path(file_name):
    return "../processed_data/exact/diff/%s.csv" % file_name


def get_early_diff_path(file_name):
    return "../processed_data/pre/diff/%s.csv" % file_name


def combine_labels(pre, current):
    if pre == current:
        return pre
    else:
        if pre == 0:
            return 2
        else:
            return 3


def read_data2(file_reader, sensor=None, multiclass=False, exclude=None, dim=100, pre=False):
    """
    Return the data of a specific sensor, if sensor is None, return all sensors
    """
    # file_reader = f0ile_reader.sample(frac=1).reset_index(drop=True)

    if pre:
        labels = file_reader.pre_labels.values
        # pre_labels = file_reader.pre_labels.values
        # current_labels = file_reader.current_labels.values
        # labels = [combine_labels(x, y) for x, y in zip(pre_labels, current_labels)]
    else:
        if multiclass:
            if exclude is not None:
                file_reader = file_reader[~file_reader["time_positions"].isin(exclude)]
            labels = file_reader.time_positions.values
        else:
            labels = file_reader.label.values

    print "Total", len(labels)
    print "Class 1", sum(labels)
    if sensor is None:
        energies = np.array([np.fromstring(x, dtype=float, sep=',').reshape(dim, 1) for x in file_reader.energy])
        maximum_amplitudes = np.array(
            [np.fromstring(x, dtype=float, sep=',').reshape(dim, 1) for x in file_reader.maximum_amplitude])
        radial_strains = np.array(
            [np.fromstring(x, dtype=float, sep=',').reshape(dim, 1) for x in file_reader.radial_strain])
        tangential_strains = np.array(
            [np.fromstring(x, dtype=float, sep=',').reshape(dim, 1) for x in file_reader.tangential_strain])

        pre_energies = np.array([np.fromstring(x, dtype=float, sep=',').reshape(dim, 1) for x in file_reader.pre_energy])
        pre_maximum_amplitudes = np.array(
            [np.fromstring(x, dtype=float, sep=',').reshape(dim, 1) for x in file_reader.pre_maximum_amplitude])
        pre_radial_strains = np.array(
            [np.fromstring(x, dtype=float, sep=',').reshape(dim, 1) for x in file_reader.pre_radial_strain])
        pre_tangential_strains = np.array(
            [np.fromstring(x, dtype=float, sep=',').reshape(dim, 1) for x in file_reader.pre_tangential_strain])

        return energies, maximum_amplitudes, radial_strains, tangential_strains, labels, \
               pre_energies, pre_maximum_amplitudes, pre_radial_strains, pre_tangential_strains
    else:
        new_dim100 = []
        new_dim59 = []
        new_labels = []
        dim100 = np.array(
            [np.fromstring(e, dtype=float, sep=',').reshape(dim, 1) for e in file_reader[sensor]])
        dim59 = np.array(
            [np.fromstring(e, dtype=float, sep=',') for e in file_reader["pre_" + sensor]])
        for a, b, c in zip(dim100, labels, dim59):
            if len(c) == 59:
                new_dim100.append(a)
                new_labels.append(b)
                new_dim59.append(c)
        return np.array(new_dim100), np.array(new_labels), np.array(new_dim59)


def read_data(file_reader, sensor=None, multiclass=False, exclude=None, num_steps=1, dim=100, pre=False):
    """
    Return the data of a specific sensor, if sensor is None, return all sensors
    """
    # file_reader = f0ile_reader.sample(frac=1).reset_index(drop=True)

    if pre:
        labels = file_reader.pre_labels.values
        """
        Modify
        """
        # deflations = file_reader.deflation.values
        # labels = [0 if x > -5 else 1 for x in deflations]

        # pre_labels = file_reader.pre_labels.values
        # current_labels = file_reader.current_labels.values
        # labels = [combine_labels(x, y) for x, y in zip(pre_labels, current_labels)]
    else:
        if multiclass:
            if exclude is not None:
                file_reader = file_reader[~file_reader["time_positions"].isin(exclude)]
            labels = file_reader.time_positions.values
        else:
            labels = file_reader.label.values

    print "Total", len(labels)
    print "Class 1", sum(labels)
    if sensor is None:
        energies = np.array([np.fromstring(x, dtype=float, sep=',').reshape(num_steps, dim) for x in file_reader.energy])
        maximum_amplitudes = np.array(
            [np.fromstring(x, dtype=float, sep=',').reshape(num_steps, dim) for x in file_reader.maximum_amplitude])
        radial_strains = np.array(
            [np.fromstring(x, dtype=float, sep=',').reshape(num_steps, dim) for x in file_reader.radial_strain])
        tangential_strains = np.array(
            [np.fromstring(x, dtype=float, sep=',').reshape(num_steps, dim) for x in file_reader.tangential_strain])

        return energies, maximum_amplitudes, radial_strains, tangential_strains, labels
    else:
        return np.array(
            [np.fromstring(e, dtype=float, sep=',').reshape(num_steps, dim) for e in file_reader[sensor]]), labels


"""
Functions for parsing time
"""


def date_parse(date):
    """
    Four formats
    '%m/%d/%Y %H:%M'
    '%Y-%m-%d %H:%M:%S'
    '%Y/%m/%d %H:%M'
    '%y/%m/%d %H:%M'
    """
    if "-" in date:
        return pd.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    else:
        last_part_format = date.split(" ")[0].split("/")[-1]
        if len(last_part_format) == 4:
            return pd.datetime.strptime(date, '%m/%d/%Y %H:%M')
        elif len(last_part_format) < 4:
            year_format = date.split("/")[0]
            if len(year_format) == 4:
                return pd.datetime.strptime(date, '%Y/%m/%d %H:%M')
            elif len(year_format) < 4:
                return pd.datetime.strptime(date, '%y/%m/%d %H:%M')
    print "ERROR in converting time!"
    print date
    return None


"""
Functions for explosion extraction
"""


# Check if an explosion exists between the start and the end of a sequence
# Return 1 if exist, otherwise return 0
def check_eruption(start, end, eruptions):
    for eruption in eruptions:
        if eruption > end:
            return 0
        if start <= eruption <= end:
            return 1
    return 0


# Check how many explosion in each sequence
def check_number_eruption(start, end, eruptions):
    count = 0
    for eruption in eruptions:
        if eruption > end:
            return count
        if start <= eruption <= end:
            count += 1
    return count


# Return the position of the explosion (minutes/10) counting from the start of the time series
# Return 0 if no explosion
def get_eruption_time(start, end, eruptions, interval):
    second_to_minute = 60
    ten_minutes_interval = interval * second_to_minute
    for eruption in eruptions:
        if eruption > end:
            return 0, 0
        if start <= eruption <= end:
            return ((eruption - start).seconds - 1) / ten_minutes_interval + 1, eruption
    return 0, 0


# Parameter "pre" is the number of minutes of pre30-pattern
def get_pre_eruption_time(start, end, eruptions, interval, pre):
    eruption_location, eruption_time = get_eruption_time(start, end, eruptions, interval)
    pre_eruption_location, pre_eruption_time = get_eruption_time(end, end + np.timedelta64(pre, 'm'), eruptions, interval)
    return eruption_location, eruption_time, pre_eruption_location, pre_eruption_time

"""
Pre process the sequence
"""


# First order difference
def diff(data):
    diff_data = []
    for i in range(len(data) - 1):
        diff_data.append(data[i + 1] - data[i])
    return diff_data


# Check a strain sequence if the values are noise
def is_strain_noise(data):
    data = [float(x) for x in data]
    min_data = np.min(data)
    max_data = np.max(data)
    noise_threshold = 300
    if max_data - min_data > noise_threshold:
        return True
    else:
        return False


# Check if a sequence lacks of data point
def is_missing_point(times):
    for i in range(len(times) - 1):
        if times[i + 1] - times[i] != np.timedelta64(1, 'm'):
            return True
    return False


# Check if a sequence contains invalid data
def is_invalid_data(values):
    for value in values:
        if np.isnan(value):
            return True
    return False


"""
Class processing
"""


# Return a binary class given a threshold
def get_class(x, threshold):
    if x > threshold:
        return 1
    else:
        return 0


def find_best_threshold(ground_truth, prediction):
    # Set 0.5 is initial threshold
    best_threshold = 0.5
    obtained_class = [np.around(x[0]) for x in prediction]
    f_score = f1_score(ground_truth, obtained_class, average=None)
    class_details = obtained_class
    best_score = np.average(f_score)
    print "normal score threshold 0.5", best_score
    # get_score_and_confusion_matrix(ground_truth, obtained_class)

    # find the threshold
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for threshold in thresholds:
        # print threshold
        obtained_class = [get_class(x, threshold) for x in prediction]
        # get_score_and_confusion_matrix(ground_truth, obtained_class)
        f_score = f1_score(ground_truth, obtained_class, average=None)
        f_avg = np.average(f_score)
        if best_score < f_avg:
            class_details = obtained_class
            best_threshold = threshold
    print "best_threshold", best_threshold
    get_score_and_confusion_matrix(ground_truth, class_details)
    return best_threshold


def get_score_and_confusion_matrix(test_labels, prediction):
    f_score = f1_score(test_labels, prediction, average=None)
    print "f_score", f_score
    print "f_score AVG", sp.average(f_score)
    confusion = confusion_matrix(test_labels, prediction)
    print "confusion matrix"
    print confusion
    return f_score, confusion


"""
Negative sampling training
"""


def sample_imbalanced_data(data, labels, batch_size):
    # create a sample with ratio 1:3
    batch_data = []
    batch_label = []
    while 1:
        for i in range(batch_size):
            sample_label = 0 if random() > 0.75 else 1
            chosen_index = choice([index for index, value in enumerate(labels) if value == sample_label])
            batch_data.append(data[chosen_index])
            batch_label.append(labels[chosen_index])
        yield np.array(batch_data), np.array(batch_label)


"""
PLot confusion matrix
"""


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # plt.text(j, i, format(cm[i, j], fmt),
        plt.text(j, i, "",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

"""
Statistics
"""


def get_statistics(data):
    # print "min", sp.amin(data)
    # print "max", sp.amax(data)
    median = sp.median(data)
    print "median", median
    mean = sp.mean(data)
    print "mean", mean
    std = sp.std(data)
    print "std", std
    # print "var", sp.var(data)
    return median, mean, std
