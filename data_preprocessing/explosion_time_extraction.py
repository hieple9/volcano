from utils.utils import *

explosion = pd.read_excel("../data/eruption/List summary.xlsx")
all_types = explosion.Type.values
print set(all_types)
non_explosives = ['Non-explosive', 'Non-explosive (M)', 'Non-Explosive']
explosives = ['Explosive', 'Explosive (M)']

explosion = explosion[explosion.Type.isin(explosives)]
deflation = explosion['EX-T'].values
print len(deflation)

# abnormal = [x for x in deflation if x > 0]
# print len(abnormal)
# print get_statistics(abnormal)

usable_deflation = [x for x in deflation if x != -999 and not np.isnan(x)]
mean = sp.mean(usable_deflation)
deflation = [x if (x != -999 and not np.isnan(x)) else mean for x in deflation]

print deflation
print len(deflation)
get_statistics(deflation)
# pd.DataFrame({"time": explosion['time'].values, 'deflation': deflation}).to_csv("../data/eruption/eruption.csv",
#                                                                                 index=None)
