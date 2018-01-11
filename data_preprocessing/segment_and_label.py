from collections import Counter
import sys
sys.path.append("../")
from utils.utils import *

eruption_counter_per_sequence = Counter()
years = ["2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"]
# years = ["2009"]
num_of_data_point = 101
sliding_window = 10
label_numbers = 10
print "num_of_data_point", num_of_data_point
print "sliding_window", sliding_window

for year in years:

    eruption = pd.read_csv("../data/eruption/explosion_%s.csv" % year)
    eruption_temp = []
    for x in eruption.explosion:
        eruption_temp.append(date_parse(str(x)))
    eruption = pd.Series(data=np.asarray(eruption_temp), index=eruption.index)

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

    radial_strains = []
    tangential_strains = []
    energies = []
    maximum_amplitudes = []
    labels = []
    time_intervals = []
    time_positions = []
    head_time = starting_time
    index = 0
    while head_time + np.timedelta64(num_of_data_point, 'm') < ending_time:
        if index % 10000 == 0:
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

        if not is_missing_point(strain.time.values) and not is_missing_point(
                energy.time.values) and not is_invalid_data(strain['tangential_strain'].values) and not is_invalid_data(
            strain['radial_strain'].values) and not is_invalid_data(
            energy['maximum_amplitude'].values) and not is_invalid_data(
            energy['energy'].values) and not is_strain_noise(
            strain['tangential_strain'].values) and not is_strain_noise(strain['radial_strain'].values):

            tail_time = head_time + np.timedelta64(num_of_data_point - 1, 'm')
            time_intervals.append(str(head_time) + "," + str(tail_time))
            eruption_position = get_eruption_time(head_time, tail_time, eruption,
                                                  (num_of_data_point - 1) / label_numbers)
            number_eruption = check_number_eruption(head_time, tail_time, eruption)
            eruption_counter_per_sequence[number_eruption] += 1

            if eruption_position == 0:
                labels.append(0)
                time_positions.append(0)
            else:
                labels.append(1)
                time_positions.append(eruption_position)

            radial_strains.append(','.join([str(x) for x in strain['radial_strain'].values]))
            tangential_strains.append(','.join([str(x) for x in strain['tangential_strain'].values]))
            energies.append(','.join([str(x) for x in energy['energy'].values]))
            maximum_amplitudes.append(','.join([str(x) for x in energy['maximum_amplitude'].values]))

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
         "energy": energies, "maximum_amplitude": maximum_amplitudes, "label": labels,
         "time_positions": time_positions})
    df.to_csv("../processed_data/all/exact/raw/%s.csv" % year, index=None)
    print "total", len(time_intervals)
    print "eruption_counter_per_sequence", eruption_counter_per_sequence

# eruption_counter_per_sequence Counter({0: 301934, 1: 86232, 2: 21404, 3: 2902, 4: 373, 5: 49, 6: 5, 7: 1})
# eruption_counter_per_sequence Counter({0: 366627, 1: 37952, 2: 6839, 3: 1228, 4: 212, 5: 37, 6: 4, 7: 1})
