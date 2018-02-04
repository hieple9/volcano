import pandas as pd
import numpy as np

file_reader = pd.read_csv("processed_data/pre/diff/2009_101_10_explosion.csv")
ts_len = 100
ahead = 10

current_labels = file_reader.current_labels.values
current_time_positions = file_reader.current_time_positions.values
print "total current explosion", sum(current_labels)
for i, j in zip(current_labels, current_time_positions):
    if i == 1 and j < 1:
        print "something wrong 1"

pre_labels = file_reader.pre_labels
pre_time_positions = file_reader.pre_time_positions
deflation = file_reader.deflation.values
print "total pre explosion", sum(pre_labels)
print "total location explosion", sum(pre_time_positions)
for i, j, k in zip(pre_labels, pre_time_positions, deflation):
    if i == 1 and j < 1:
        print "something wrong 2"
    if i == 1 and k == 0:
        print "something wrong 3"

energy = file_reader.energy
pre_energy = file_reader.pre_energy
for current, pre in zip(energy, pre_energy):
    current = np.fromstring(current, dtype=float, sep=',')
    pre = np.fromstring(pre, dtype=float, sep=',')
    if len(current) != len(pre) or len(current) != ts_len:
        print "something wrong 4"
    for i, j in zip(current[ahead:], pre[:ahead]):
        if i != j:
            print "sth wrong"

maximum_amplitude = file_reader.maximum_amplitude
pre_maximum_amplitude = file_reader.pre_maximum_amplitude
for current, pre in zip(maximum_amplitude, pre_maximum_amplitude):
    current = np.fromstring(current, dtype=float, sep=',')
    pre = np.fromstring(pre, dtype=float, sep=',')
    if len(current) != len(pre) or len(current) != ts_len:
        print "something wrong 4"
    for i, j in zip(current[ahead:], pre[:ahead]):
        if i != j:
            print "sth wrong"

radial_strain = file_reader.radial_strain
pre_radial_strain = file_reader.pre_radial_strain
for current, pre in zip(radial_strain, pre_radial_strain):
    current = np.fromstring(current, dtype=float, sep=',')
    pre = np.fromstring(pre, dtype=float, sep=',')
    if len(current) != len(pre) or len(current) != ts_len:
        print "something wrong 4"
    for i, j in zip(current[ahead:], pre[:ahead]):
        if i != j:
            print "sth wrong"

tangential_strain = file_reader.tangential_strain
pre_tangential_strain = file_reader.pre_tangential_strain
for current, pre in zip(tangential_strain, pre_tangential_strain):
    current = np.fromstring(current, dtype=float, sep=',')
    pre = np.fromstring(pre, dtype=float, sep=',')
    if len(current) != len(pre) or len(current) != ts_len:
        print "something wrong 4"
    for i, j in zip(current[ahead:], pre[:ahead]):
        if i != j:
            print "sth wrong"

time_interval = file_reader.time_interval
