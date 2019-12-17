import chirpy_mk1

from chirpy_mk1.ansatz import PSF_freq_ins
from chirpy_mk1.pn import TaylorT3_Omega_new

import numpy as np
import time
import cy_t3

def setup_ins_coeffs(eta):
    freq_inc_tc = PSF_freq_ins('tc',eta)
    #freq_inc_b = PSF_freq_ins('b',eta)
    #freq_inc_c = PSF_freq_ins('c',eta)

    params_freq_ins = {}
    params_freq_ins.update({
        'tc':np.float64(freq_inc_tc)
    })
    #params_freq_ins.update({
    #    'tc':np.float64(freq_inc_tc),
    #    'b':np.float64(freq_inc_b),
    #    'c':np.float64(freq_inc_c)
    #})
    return params_freq_ins


times = np.linspace(-1000, -500, 1e6)
eta = 0.25
M = 1
params_freq_ins = setup_ins_coeffs(eta)
tc = params_freq_ins['tc']

t1 = time.time()
og_t3 = TaylorT3_Omega_new(times, tc, eta, M)
t2 = time.time()
print('og dt = {}'.format(t2-t1))





t1 = time.time()
cy_y = cy_t3.cython_TaylorT3_Omega_new(times, tc, eta, M)
t2 = time.time()
print('cython dt = {}'.format(t2-t1))


np.testing.assert_array_almost_equal(og_t3, cy_y)

