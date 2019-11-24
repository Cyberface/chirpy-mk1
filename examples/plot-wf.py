import matplotlib.pyplot as plt
import numpy as np

import chirpy_mk1 as cmk1

import phenom

# using the times array interface
t = np.linspace(-1000, 100, 1000)
mk1 = cmk1.Mk1(times=t, q=1)

# using the f_start interface
# srate_sec = 2**(-17)
# srate_M = phenom.StoM(srate_sec, 1)
# t_end = 100
# mk1 = cmk1.Mk1(q=1, f_start=0.06, srate=srate_M, t_end=t_end)

plt.figure()
plt.plot(mk1.times, np.real(mk1.h))
plt.xlabel('t/M')
plt.ylabel('Re(h)')
plt.savefig('wf-plot.png')
