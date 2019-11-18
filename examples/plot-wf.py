import matplotlib.pyplot as plt
import numpy as np

import chirpy_mk1 as cmk1

t = np.linspace(-1000, 100, 1000)

mk1 = cmk1.Mk1(t, 1)

plt.figure()
plt.plot(mk1.times, np.real(mk1.h))
plt.xlabel('t/M')
plt.ylabel('Re(h)')
plt.savefig('wf-plot.png')
