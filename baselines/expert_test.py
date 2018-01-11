import sys

sys.path.append("../")
from utils.utils import *
from pyts.transformation import PAA

sensor = 'tangential_strain'
explosive_point = 80

window_sizes = [4]
for window_size in window_sizes:
    print "_______________________________________________________________________________"
    print "window_size", window_size
    paa = PAA(window_size=window_size, overlapping=True)

    explosion_expert_reader = pd.read_csv(get_raw_path("training"))
    explosion_expert_reader = explosion_expert_reader[explosion_expert_reader.label == 1]
    explosive_train = np.array([np.fromstring(e, dtype=float, sep=',')
                                for e in explosion_expert_reader[sensor]])

    print explosive_train.shape
    explosive_train = paa.transform(explosive_train)
    print explosive_train.shape
    explosive_train = np.array([diff(x) for x in explosive_train])
    print explosive_train.shape

    inflations = []
    deflations = []
    for each in explosive_train:
        inflation = each[:explosive_point/window_size]
        deflation = each[explosive_point/window_size:]
        inflations.append(sum(inflation) / len(inflation))
        deflations.append(sum(deflation) / len(deflation))

    inflation_median, inflation_mean, inflation_std = get_statistics(inflations)
    deflation_median, deflation_mean, deflation_std = get_statistics(deflations)

    train_reader = pd.read_csv(get_raw_path("training"))
    train_reader = train_reader[train_reader.label == 0]
    not_explosive_strain, not_explosive_labels = read_data(train_reader, sensor, dim=101)
    not_explosive_strain = np.array([x.flatten() for x in not_explosive_strain])

    # print not_explosive_strain.shape
    not_explosive_strain = paa.transform(not_explosive_strain)
    # print not_explosive_strain.shape
    not_explosive_strain = np.array([diff(x) for x in not_explosive_strain])
    # print not_explosive_strain.shape

    inflations = []
    deflations = []
    for each in not_explosive_strain:
        inflation = each[:explosive_point/window_size]
        deflation = each[explosive_point/window_size:]
        inflations.append(sum(inflation) / len(inflation))
        deflations.append(sum(deflation) / len(deflation))

    normal_inflation_median, normal_inflation_mean, normal_inflation_std = get_statistics(inflations)
    normal_deflation_median, normal_deflation_mean, normal_deflation_std = get_statistics(deflations)

    print "Validation"
    validation_reader = pd.read_csv(get_raw_path("test"))

    validation_data, validation_label = read_data(validation_reader, sensor, dim=101)
    validation_data = np.array([x.flatten() for x in validation_data])
    validation_data = paa.transform(validation_data)
    validation_data = np.array([diff(x) for x in validation_data])
    # print validation_data.shape

    prediction = []
    for each in validation_data:

        inflation = each[:explosive_point/window_size]
        inflation = sum(inflation) / len(inflation)
        deflation = each[explosive_point/window_size:]
        deflation = sum(deflation) / len(deflation)

        inflation_explosion_difference = abs(inflation - inflation_mean)
        inflation_normal_difference = abs(inflation - normal_inflation_mean)
        deflation_explosion_difference = abs(deflation - deflation_mean)
        deflation_normal_difference = abs(deflation - normal_deflation_mean)

        if inflation_explosion_difference < inflation_normal_difference and deflation_explosion_difference < deflation_normal_difference:
            prediction.append(1)
        else:
            prediction.append(0)

    get_score_and_confusion_matrix(validation_label, prediction)
