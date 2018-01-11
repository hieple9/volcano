import sys
sys.path.append("../")
from utils.utils import *

eruption = pd.read_csv("../data/eruption/eruption.csv")
eruption['time'] = pd.to_datetime(eruption['time'])

years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]

for year in years:
    this_year = eruption[(eruption.time.dt.year == year)]
    this_year.to_csv("../data/eruption/eruption_%s.csv" % year, index=False)
