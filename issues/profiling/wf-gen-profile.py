import numpy as np

import chirpy_mk1 as cmk1

# using the times array interface
t = np.linspace(-10000, 200, 1000000)
mk1 = cmk1.Mk1(times=t, q=1)
