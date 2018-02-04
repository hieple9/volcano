from utils.utils import *
import matplotlib.pyplot as plt

explosion = pd.read_excel("../data/eruption/List summary.xlsx")
explosion = explosion[(explosion.Type == 'Explosive') | (explosion.Type == 'Explosive (M)')]
explosion['time'] = pd.to_datetime(explosion['time'])

years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

all = []
for year in years:
    this_year = []
    for month in months:
        a = explosion[(explosion.time.dt.year == year) & (explosion.time.dt.month == month)]
        this_year.append(len(a))
    print len(explosion[(explosion.time.dt.year == year)]), this_year
    all.append(this_year)

print all

# Four axes, returned as a 2-d array
f, axarr = plt.subplots(4, 2)

axarr[0, 0].plot(range(1, len(all[0])+1), all[0], alpha=1, color='red')
axarr[0, 0].set_title('2009', fontsize=15)
axarr[0, 0].set_ylim(0, 170)
axarr[0, 0].set_ylabel('Explosions', fontsize=14)
axarr[0, 0].tick_params(labelsize=10)

axarr[0, 1].plot(range(1, len(all[1])+1), all[1], alpha=1, color='red')
axarr[0, 1].set_title('2010', fontsize=15)
axarr[0, 1].set_ylim(0, 170)
axarr[0, 1].tick_params(labelsize=10)

axarr[1, 0].plot(range(1, len(all[2])+1), all[2], alpha=1, color='red')
axarr[1, 0].set_title('2011', fontsize=15)
axarr[1, 0].set_ylim(0, 170)
axarr[1, 0].set_ylabel('Explosions', fontsize=14)
axarr[1, 0].tick_params(labelsize=12)

axarr[1, 1].plot(range(1, len(all[3])+1), all[3], alpha=1, color='red')
axarr[1, 1].set_title('2012', fontsize=15)
axarr[1, 1].set_ylim(0, 170)
axarr[1, 1].tick_params(labelsize=12)

axarr[2, 0].plot(range(1, len(all[4])+1), all[4], alpha=1, color='red')
axarr[2, 0].set_title('2013', fontsize=15)
axarr[2, 0].set_ylim(0, 170)
axarr[2, 0].set_ylabel('Explosions', fontsize=14)
axarr[2, 0].tick_params(labelsize=12)

axarr[2, 1].plot(range(1, len(all[5])+1), all[5], alpha=1, color='red')
axarr[2, 1].set_title('2014', fontsize=15)
axarr[2, 1].set_ylim(0, 170)
axarr[2, 1].tick_params(labelsize=12)

axarr[3, 0].plot(range(1, len(all[6])+1), all[6], alpha=1, color='red')
axarr[3, 0].set_title('2015', fontsize=15)
axarr[3, 0].set_ylim(0, 170)
axarr[3, 0].set_xlabel('Month', fontsize=14)
axarr[3, 0].set_ylabel('Explosions', fontsize=14)
axarr[3, 0].tick_params(labelsize=12)

axarr[3, 1].plot(range(1, len(all[7])+1), all[7], alpha=1, color='red')
axarr[3, 1].set_title('2016', fontsize=15)
axarr[3, 1].set_ylim(0, 170)
axarr[3, 1].set_xlabel('Month', fontsize=14)
axarr[3, 1].tick_params(labelsize=12)

plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
plt.setp([a.get_xticklabels() for a in axarr[2, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

# plt.show()
