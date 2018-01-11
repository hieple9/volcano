import sys
sys.path.append("../")
from utils.utils import get_early_diff_path, read_data, pd
import matplotlib.pyplot as plt
from scipy.stats.mstats import zscore

sensors = ['tangential_strain']
for sensor in sensors:
    print sensor

    test_reader = pd.read_csv(get_early_diff_path("test_60_explosion"))
    test_data, test_labels = read_data(test_reader, sensor, pre=True)

    label = [0, 1, 1, 2, 1, 0, 0, 0, 2, 1, 1, 2, 0, 4, 0, 2, 2, 1, 1, 1, 3, 2, 0, 3, 2, 0, 4, 1]

    high_possible_index = [623, 637, 1335, 3892, 4475, 6916, 9754, 13150, 13548, 13801, 18367, 19010, 20799, 22174,
                           22564, 22927, 24272, 28318, 28360, 30189, 32744, 32861, 33364, 34895, 39462, 41735, 44862, 47834]

    deflation = [0.0, -29.03, -32.7, -34.23, -21.6, 0.0, 0.0, 0.0, -12.19, -24.12, -36.21, -18.16, 0.0, -24.12, 0.0,
                 -33.67, -15.56, -27.93, -32.96, -14.45, -28.35, -60.14, 0.0, -50.28, -29.79, 0.0, -50.28, -29.22]

    new_label = []
    new_high_possible_index = []
    new_deflation = []
    for l, h, d in zip(label, high_possible_index, deflation):
        if l == 0:
            new_label.append(l)
            new_high_possible_index.append(h)
            new_deflation.append(d)

            visual = [x[0] for x in test_data[h]]
            visual = zscore(visual)
            plt.plot(range(1, 101), visual, color='red')
            plt.title("Index " + str(h))
            plt.ylabel('Tangential Strain Change')
            plt.xlabel('Time')
            plt.savefig("fail_high_possible_%s" % h)
            plt.clf()

    print new_label
    print new_high_possible_index
    print new_deflation
