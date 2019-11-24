import numpy as np

import pycbc.filter
from pycbc.types import TimeSeries

eobt, eobh = np.loadtxt('./data/low-mass-SEOBNRv4_t_realh.dat', unpack=True)
dt, dh = np.loadtxt('./data/low-mass-IMRPhenomD_t_realh.dat', unpack=True)

ts1 = TimeSeries(eobh, delta_t=eobt[1]-eobt[0])
ts2 = TimeSeries(dh, delta_t=dt[1]-dt[0])

print(pycbc.filter.match(ts1, ts2))
