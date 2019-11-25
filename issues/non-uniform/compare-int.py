import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.interpolate import interp1d, splrep, splev

import chirpy_mk1 as cmk1
import chirpy_mk1.utils

# low mass times
tstart = -10000
tend = 100
npts = 100000
npts_factor = 80


# high mass times
#tstart = -4000
#tend = 100
#npts = 1000
#npts_factor = 2

print('npts/npts_factor = {}'.format(npts/npts_factor))
t = np.linspace(tstart, tend, npts)
t1 = time.time()
mk1 = cmk1.Mk1(times=t, q=1)
t2 = time.time()
orig_time_diff = t2 - t1
print("[original] time_diff = {}".format(orig_time_diff))
freq = mk1.freq_func(mk1.times)

t1 = time.time()
new_npts = npts / npts_factor
new_t = np.linspace(tstart, tend, new_npts)
new_amp = mk1.amp_func(new_t)
new_phase = mk1.phase_func(new_t)
new_amp = interp1d(new_t, new_amp)(t)
new_phase = interp1d(new_t, new_phase)(t)
new_h = new_amp * np.exp(-1.j * new_phase)
t2 = time.time()
time_diff = t2 - t1
print("[interp linear] time_diff = {}".format(time_diff))
print("[og / linear] = {}".format(orig_time_diff / time_diff))

match = chirpy_mk1.utils.match(np.real(new_h), np.real(mk1.h), t)
print(match)

from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import Rbf
t1 = time.time()
new_npts = npts / npts_factor
new_t = np.linspace(tstart, tend, new_npts)
new_amp = mk1.amp_func(new_t)
new_phase = mk1.phase_func(new_t)
new_amp = interp1d(new_t, new_amp)(t)
new_phase = interp1d(new_t, new_phase, kind='cubic')(t)
#new_amp = np.interp(t, new_t, new_amp)
#new_phase = np.interp(t, new_t, new_phase)
#new_amp =  IUS(new_t, new_amp, k=1)(t)
#new_phase =  IUS(new_t, new_phase, k=1)(t)
new_h = new_amp * np.exp(-1.j * new_phase)
t2 = time.time()
time_diff = t2 - t1
print("[cubic phase int] time_diff = {}".format(time_diff))
print("[og / cubic] = {}".format(orig_time_diff / time_diff))


match = chirpy_mk1.utils.match(np.real(new_h), np.real(mk1.h), t)
print(match)


