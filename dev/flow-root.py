import matplotlib
import matplotlib.pyplot as plt
import numpy as np


from scipy.interpolate import InterpolatedUnivariateSpline as IUS


import chirpy_mk1
from chirpy_mk1.ansatz import PSF_freq_ins
from chirpy_mk1.ansatz import freq_ins_ansatz

import phenom

q = 10.

eta = phenom.eta_from_q(q)

# we first try using the inspiral part of the mk1
# to find the time of frequency.

# get inspiral coefficients
freq_inc_tc = PSF_freq_ins('tc', eta)
freq_inc_b = PSF_freq_ins('b', eta)
freq_inc_c = PSF_freq_ins('c', eta)

params_freq_ins={}
params_freq_ins.update({
    'tc':freq_inc_tc,
    'b':freq_inc_b,
    'c':freq_inc_c
})


def f_of_t_ins(t, eta=eta, params_freq_ins=params_freq_ins):
    return freq_ins_ansatz(t, eta, params_freq_ins)

def func_ins(t, eta, params_freq_ins, f0):
    return f_of_t_ins(t, eta, params_freq_ins) - f0


# we also try using the complete IMR of the mk1
# because there might be times where you
# want the f_of_t at a time close to the merger where the
# inspiral model isn't good enough

mk1=chirpy_mk1.Mk1(np.array([-100]), q)

def f_of_t_imr(t, eta):
    return mk1.freq_func(np.array([t]), eta)

def func_imr(t, eta, f0):
    return f_of_t_imr(np.array([t]), eta) - f0

from scipy import optimize

# can't use a frequency too close to the merger with
# inspiral only.
# So maybe we first try the inspiral ansatz
# and if that fails use the full imr ansatz
# we should also do a check the input f_min
# and enfore that it should be smaller that the
# ringdown frequency
#f_min = 0.2 #this is typically too large for q=10 for ins method
#f_min = 0.09
f_min = 0.05
#f_min = 0.01

import time

t1=time.time()
root_ins = optimize.newton(func_ins, -100, args=(eta, params_freq_ins,  f_min))
t2=time.time()
dt_ins = t2-t1

t1=time.time()
root_imr = optimize.newton(func_imr, -100, args=(eta,  f_min))[0]
t2=time.time()
dt_imr = t2-t1

print('[ins] time at desired f_min ({}) = {}'.format(f_min, root_ins))
print('[imr] time at desired f_min ({}) = {}'.format(f_min, root_imr))

print("[ins] time taken = {}s".format(dt_ins))
print("[imr] time taken = {}s".format(dt_imr))

#times = np.linspace(-2000, -4, 500)
#times = np.linspace(-2000, 60, 500)
#times = np.linspace(-1000, -500, 500)

t0 = 1.01 * root_ins
print(root_ins)
print(t0)

times = np.linspace(t0, -10, 1000)

# compute inspiral part
freq_ins = freq_ins_ansatz(times, eta, params_freq_ins)

# compute imr
mk1=chirpy_mk1.Mk1(times, q)

plt.figure()
plt.plot(times, freq_ins, label='ins only', c='C0')
plt.plot(mk1.times, mk1.freq, label='imr', c='C1')
plt.axvline(root_ins, c='C0', ls='--')
plt.axvline(root_imr, c='C1', ls='--')
plt.axhline(f_min, c='k', ls='--')
plt.legend()
plt.show()



