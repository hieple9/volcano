import sys

sys.path.append("../")
from utils.utils import *

explosion = pd.read_csv("../data/eruption/explosion.csv")
dates = [date_parse(str(x)) for x in explosion.explosion.values]

strain_data = pd.read_csv('../data/strain/data_2009_2016.csv')
energy_data = pd.read_csv('../data/energy/data_2009_2016.csv')
strain_data.time = pd.Series(data=np.asarray([date_parse(str(x)) for x in strain_data.time]), index=strain_data.index)
energy_data.time = pd.Series(data=np.asarray([date_parse(str(x)) for x in energy_data.time]), index=energy_data.index)

ts_length = 101
radial_strains = []
tangential_strains = []
energies = []
maximum_amplitudes = []
labels = []
years = []


def extract_label(end_distance):
    if end_distance <= 5:
        return 1
    elif end_distance <= 10:
        return 2
    elif end_distance <= 15:
        return 3
    else:
        return 4

total_dates = len(dates)
for date_index in range(total_dates):
    date = dates[date_index]

    strain_index = strain_data[strain_data['time'] == date].index.tolist()
    energy_index = energy_data[energy_data['time'] == date].index.tolist()
    # check if there are no record or more than one record with the head time
    if len(strain_index) != 1 or len(energy_index) != 1:
        print "There are no record or more than one record with the head time!!!!!!!!!!!!!!!!!!!!"
        print "strain_index", strain_index
        print "energy_index", energy_index
        print date
        continue
    strain_index = strain_index[0]
    energy_index = energy_index[0]

    # end_distance is distance from the end point of time series to the nearest explosive eruption
    for end_distance in range(1, 21):
        strain = strain_data.iloc[strain_index - ts_length - end_distance: strain_index - end_distance]
        energy = energy_data.iloc[energy_index - ts_length - end_distance: energy_index - end_distance]

        if not is_missing_point(strain.time.values) and not is_missing_point(
                energy.time.values) and not is_invalid_data(strain['tangential_strain'].values) and not is_invalid_data(
            strain['radial_strain'].values) and not is_invalid_data(
            energy['maximum_amplitude'].values) and not is_invalid_data(
            energy['energy'].values) and not is_strain_noise(strain['tangential_strain'].values) and not is_strain_noise(
            strain['radial_strain'].values):

            years.append(date.year)
            radial_strains.append(','.join([str(x) for x in strain['radial_strain'].values]))
            tangential_strains.append(','.join([str(x) for x in strain['tangential_strain'].values]))
            energies.append(','.join([str(x) for x in energy['energy'].values]))
            maximum_amplitudes.append(','.join([str(x) for x in energy['maximum_amplitude'].values]))
            labels.append(extract_label(end_distance))

df = pd.DataFrame({"tangential_strain": tangential_strains, "radial_strain": radial_strains, "energy": energies,
                   "maximum_amplitude": maximum_amplitudes, "year": years, "label": labels})
df.to_csv("../processed_data/all/exact/diff/explosion_prior.csv", index=None)
print "total", total_dates
