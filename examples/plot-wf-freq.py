import matplotlib.pyplot as plt
import numpy as np

import chirpy_mk1 as cmk1

# using the times array interface
t = np.linspace(-1000, 100, 1000)
mk1 = cmk1.Mk1(times=t, q=1)


freq = mk1.freq_func(mk1.times)


plt.figure()
plt.plot(mk1.times, freq)
plt.xlabel('t/M')
plt.ylabel(r'$M \Omega_{22}$')
plt.savefig('wf-freq-plot.png')
