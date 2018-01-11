from collections import Counter
import sys

sys.path.append("../")
from utils.utils import *

explosion_counter_per_sequence = Counter()
years = ["2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"]
volcano_status = "explosion"
for year in years:

    data = pd.read_csv("../data/eruption/%s_%s.csv" % (volcano_status, year))
    explosion_temp = []
    for x in data.time:
        explosion_temp.append(date_parse(str(x)))
    explosion = pd.Series(data=np.asarray(explosion_temp), index=data.index)
    deflation_db = pd.Series(data=np.asarray(data.deflation.values), index=explosion_temp)

    strain_data = pd.read_csv('../data/strain/data_%s.csv' % year)
    energy_data = pd.read_csv('../data/energy/data_%s.csv' % year)

    strain_data.time = pd.Series(data=np.asarray([date_parse(str(x)) for x in strain_data.time]),
                                 index=strain_data.index)
    energy_data.time = pd.Series(data=np.asarray([date_parse(str(x)) for x in energy_data.time]),
                                 index=energy_data.index)

    strain_len = len(strain_data)
    print 'strain_len', strain_len
    energy_len = len(energy_data)
    print 'energy_len', energy_len

    strain_starting_time = strain_data.time[0]
    strain_ending_time = strain_data.time[strain_len - 1]
    print strain_starting_time
    print strain_ending_time

    energy_starting_time = energy_data.time[0]
    energy_ending_time = energy_data.time[energy_len - 1]
    print energy_starting_time
    print energy_ending_time

    starting_time = strain_starting_time if strain_starting_time > energy_starting_time else energy_starting_time
    print 'starting_time', starting_time

    ending_time = strain_ending_time if strain_ending_time < energy_ending_time else energy_ending_time
    print 'ending_time', ending_time

    num_of_data_point = 101
    sliding_window = 10
    label_interval = 10
    pre = 60

    radial_strains = []
    tangential_strains = []
    energies = []
    maximum_amplitudes = []
    pre_radial_strains = []
    pre_tangential_strains = []
    pre_energies = []
    pre_maximum_amplitudes = []
    current_labels = []
    pre_labels = []
    time_intervals = []
    current_time_positions = []
    pre_time_positions = []
    deflation = []
    head_time = starting_time
    index = 0
    while head_time + np.timedelta64(num_of_data_point + pre, 'm') < ending_time:
        if index % 1000 == 0:
            print "index", index
        index += 1

        strain_index = strain_data[strain_data['time'] == head_time].index.tolist()
        energy_index = energy_data[energy_data['time'] == head_time].index.tolist()
        # check if there are no record or more than one record with the head time
        if len(strain_index) != 1 or len(energy_index) != 1:
            print "There are no record or more than one record with the head time!!!!!!!!!!!!!!!!!!!!"
            print "strain_index", strain_index
            print "energy_index", energy_index
            print head_time
        strain_index = strain_index[0]
        energy_index = energy_index[0]
        strain = strain_data.iloc[strain_index:strain_index + num_of_data_point]
        energy = energy_data.iloc[energy_index:energy_index + num_of_data_point]
        pre_strain = strain_data.iloc[strain_index + num_of_data_point:strain_index + num_of_data_point + pre]
        pre_energy = energy_data.iloc[energy_index + num_of_data_point:energy_index + num_of_data_point + pre]

        if not is_missing_point(strain.time.values) and not is_missing_point(
                energy.time.values) and not is_invalid_data(strain['tangential_strain'].values) and not is_invalid_data(
            strain['radial_strain'].values) and not is_invalid_data(
            energy['maximum_amplitude'].values) and not is_invalid_data(
            energy['energy'].values) and not is_strain_noise(
            strain['tangential_strain'].values) and not is_strain_noise(strain['radial_strain'].values)\
                and not is_missing_point(pre_strain.time.values) and not is_missing_point(
            pre_energy.time.values) and not is_invalid_data(pre_strain['tangential_strain'].values) and not is_invalid_data(
            pre_strain['radial_strain'].values) and not is_invalid_data(
            pre_energy['maximum_amplitude'].values) and not is_invalid_data(
            pre_energy['energy'].values) and not is_strain_noise(
            pre_strain['tangential_strain'].values) and not is_strain_noise(pre_strain['radial_strain'].values):

            tail_time = head_time + np.timedelta64(num_of_data_point - 1, 'm')
            time_intervals.append(str(head_time) + "," + str(tail_time))
            explosion_position, explosion_time, pre_explosion_position, pre_explosion_time =\
                get_pre_eruption_time(head_time, tail_time, explosion, (num_of_data_point - 1) / label_interval, pre)
            number_explosion = check_number_eruption(head_time, tail_time, explosion)
            explosion_counter_per_sequence[number_explosion] += 1

            if explosion_position == 0:
                current_labels.append(0)
                current_time_positions.append(0)
            else:
                current_labels.append(1)
                current_time_positions.append(explosion_position)

            if pre_explosion_position == 0:
                pre_labels.append(0)
                pre_time_positions.append(0)
            else:
                pre_labels.append(1)
                pre_time_positions.append(pre_explosion_position)

            radial_strains.append(','.join([str(x) for x in diff(strain['radial_strain'].values)]))
            tangential_strains.append(','.join([str(x) for x in diff(strain['tangential_strain'].values)]))
            energies.append(','.join([str(x) for x in diff(energy['energy'].values)]))
            maximum_amplitudes.append(','.join([str(x) for x in diff(energy['maximum_amplitude'].values)]))

            pre_radial_strains.append(','.join([str(x) for x in diff(pre_strain['radial_strain'].values)]))
            pre_tangential_strains.append(','.join([str(x) for x in diff(pre_strain['tangential_strain'].values)]))
            pre_energies.append(','.join([str(x) for x in diff(pre_energy['energy'].values)]))
            pre_maximum_amplitudes.append(','.join([str(x) for x in diff(pre_energy['maximum_amplitude'].values)]))

            deflation.append(0 if pre_explosion_time == 0 else deflation_db.get(pre_explosion_time))

        head_time += np.timedelta64(sliding_window, 'm')

        # check to make sure that the with the head time exists
        strain_index = strain_data[strain_data['time'] == head_time].index.tolist()
        energy_index = energy_data[energy_data['time'] == head_time].index.tolist()
        while len(strain_index) == 0 or len(energy_index) == 0:
            head_time += np.timedelta64(1, 'm')
            strain_index = strain_data[strain_data['time'] == head_time].index.tolist()
            energy_index = energy_data[energy_data['time'] == head_time].index.tolist()

    df = pd.DataFrame(
        {"time_interval": time_intervals, "tangential_strain": tangential_strains, "radial_strain": radial_strains,
         "energy": energies, "maximum_amplitude": maximum_amplitudes,
         "pre_tangential_strain": pre_tangential_strains, "pre_radial_strain": pre_radial_strains,
         "pre_energy": pre_energies, "pre_maximum_amplitude": pre_maximum_amplitudes,
         "current_labels": current_labels,
         "current_time_positions": current_time_positions, "pre_labels": pre_labels,
         "pre_time_positions": pre_time_positions, "deflation": deflation})
    df.to_csv("../processed_data/pre/diff/%s_%s_%s.csv" % (year, pre, volcano_status), index=None)
    print "total", len(time_intervals)
    print "explosion_counter_per_sequence", explosion_counter_per_sequence

# explosion_counter_per_sequence Counter({0: 366627, 1: 37952, 2: 6839, 3: 1228, 4: 212, 5: 37, 6: 4, 7: 1})
