test_time_positions = test_reader.pre_time_positions.values
test_deflation = test_reader.deflation.values
true_time_distribution = Counter()
false_time_distribution = Counter()
true_deflation = []
false_deflation = []
for i in range(len(test_labels)):
    if test_labels[i] == 1:
        time_pos = test_time_positions[i]
        if classified[i] == 1:
            true_time_distribution[time_pos] += 1
            true_deflation.append(test_deflation[i])
        else:
            false_time_distribution[time_pos] += 1
            false_deflation.append(test_deflation[i])

print "true"
print time_show(true_time_distribution)
get_statistics(true_deflation)
print "false"
print time_show(false_time_distribution)
get_statistics(false_deflation)