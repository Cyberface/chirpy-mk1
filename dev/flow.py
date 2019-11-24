"""
trying to find start time for a given start
frequency.

this script tried to use
the TaylorF2 expression to estimate
the time for a given frequency
but in another script 'flow-root.py'
I use Newton's method and found that easier
to work with.

This script might not even run anymore...
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline as IUS


import chirpy_mk1
from chirpy_mk1.ansatz import PSF_freq_ins
from chirpy_mk1.ansatz import freq_ins_ansatz

import phenom


def pn_t_of_f(F0, t0, eta, M=1):

    v0 = (np.pi * M * F0 / 2 / np.pi)**(1/3.)
    # needed to do F0/2/np.pi to get scaling

    t2 = (5*M)  /  (256 * eta * v0**8)
    #t2 = (5*M)/eta * (8*np.pi*F0)**(-8./3.)
    #print("t2 = {}".format(t2))
    #print("t0 = {}".format(t0))

    # T2 t(v) expression
    t2 = t2 * (1 + v0**2*((743./252) +(11./3*eta))  - 32/5.*np.pi*v0**3 + \
            v0**4*(3058673/508032. + 5429./504*eta + 617./72. * eta**2) - \
            v0**5*(7729./252 - 13./3*eta)*np.pi
            )

    return t0 - t2

q = 1.

eta = phenom.eta_from_q(q)

freq_inc_tc = PSF_freq_ins('tc', eta)
freq_inc_b = PSF_freq_ins('b', eta)
freq_inc_c = PSF_freq_ins('c', eta)

params_freq_ins={}
params_freq_ins.update({
    'tc':freq_inc_tc,
    'b':freq_inc_b,
    'c':freq_inc_c
})


times = np.linspace(-2000, -4, 500)
#times = np.linspace(-1000, -500, 500)

freq_ins = freq_ins_ansatz(times, eta, params_freq_ins)

F0 = 0.067
#print("F0 = {}".format(F0))
#print("tc = ", params_freq_ins['tc'])
t0 = pn_t_of_f(F0, params_freq_ins['tc'], eta)

#print("t0 = {}".format(t0))

f0_from_t0 = freq_ins_ansatz(t0, eta, params_freq_ins)
#print("f0_from_t0 = {}".format(f0_from_t0))

# compuate f(t)

toff = pn_t_of_f(freq_ins, params_freq_ins['tc'], eta)

#sc = (toff-toff[0])[1] / (times-times[0])[1]
#print(1./sc)


#plt.figure()
#plt.plot(freq_ins, (toff-toff[0])/sc, label='calculated')
#plt.plot(freq_ins, times-times[0], label='real', ls='--')
#plt.legend()
#plt.show()


# full imr
mk1 = chirpy_mk1.Mk1(times, q)
mk1_freq = mk1.freq_func(mk1.times)

plt.figure()
plt.plot(mk1_freq, mk1.times-mk1.times[0], label='imr')
plt.plot(freq_ins, (toff-toff[0]), label='estimated')
plt.plot(freq_ins, times-times[0], label='real', ls='--')
plt.legend()
plt.show()

#plt.figure()
#plt.plot(times, freq_ins, label='Mk1 ins model')
#plt.axhline(F0, c='k', ls='--')
#plt.axvline(t0 + times[0], c='k', ls='--')
#plt.xlabel('t/M')
#plt.ylabel(r'$M \omega_{22}(t)$')
#plt.show()
