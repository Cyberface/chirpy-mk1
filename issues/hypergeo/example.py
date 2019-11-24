import matplotlib.pyplot as plt
import numpy as np

import chirpy_mk1 as cmk1
from chirpy_mk1 import ansatz
import phenom

from scipy.special import hyp2f1
from mpmath import mp
from mpmath import hyp2f1 as mphyp2f1
from mpmath import tanh as mptanh
# set precision for sympy.
# 64 seems to be enough
mp.dps=64


##### BEGIN  standard chirpy

q = 1.
eta = phenom.eta_from_q(q)
times = np.linspace(-5, 500, 500)

fin_spin = phenom.remnant.FinalSpin0815(eta, 0, 0)
fring = phenom.remnant.fring(eta, 0, 0, fin_spin)
# convert to angular geometric
fring *= 2*np.pi
fdamp = phenom.remnant.fdamp(eta, 0, 0, fin_spin)
# convert to angular geometric
fdamp *= 2*np.pi

params_freq_mrd={}
params_freq_mrd.update({
    't0': ansatz.PSF_freq_mrd('t0', eta),
    'om_f' : fring,
    'b' : 1./fdamp,
    'offset': 0.175,
    'kappa': 0.44
})

phase_mrd = ansatz.analytic_phase_mrd_ansatz(times, **params_freq_mrd)

#print('t\tmodel phase')
#for t,  mphase in list(zip(times,  phase_mrd)):
#    print('{}\t{:.16f}'.format(t, mphase))

plt.figure()
plt.plot(times, phase_mrd)
plt.xlabel('t/M')
plt.ylabel('phase')
plt.show()


#### END STANDARD CHIPRY


def hypergeom_components(t, t0, b, om_f, offset=0.175, kappa=0.44):

    if type(t) in [int, float]:
        ts = np.array([t], dtype=np.float64)
    else:
        ts = t


    hyper = np.array([mphyp2f1(1, kappa, 1+kappa, 0.5*(1+mptanh((tt-t0)/b))) for tt in ts], dtype=np.float64)
    term1 = np.array([mptanh((tt-t0)/b) for tt in ts], dtype=np.float64)
    control_phase = offset * ts + (1/kappa) * 2**(-1-kappa) * b * (om_f - offset) * hyper * (1 + term1)**kappa

    aa = 1.
    bb = kappa
    cc = 1. + kappa
    zz = np.array([ 0.5*(1+mptanh((tt-t0)/b)) for tt in ts], dtype=np.float64)
    term1 = np.array([mptanh((tt-t0)/b) for tt in ts], dtype=np.float64)
    
    hyper = np.array([mphyp2f1(aa, bb, cc, zzz) for zzz in zz], dtype=np.float64)

    phase = offset * ts + (1/kappa) * 2**(-1-kappa) * b * (om_f - offset) * hyper * (1 + term1)**kappa

    sympy_res = {}
    sympy_res.update({
        'a':aa,
        'b':bb,
        'c':cc,
        'z':zz,
        'term1':term1,
        'phase':phase
        })

    aa = 1.
    bb = kappa
    cc = 1. + kappa
    zz = 0.5 * (1. + np.tanh( (ts-t0)/b ))
    term1 = np.tanh( (ts-t0)/b )

    hyper = hyp2f1(aa, bb, cc, zz)

    phase = offset * ts + (1/kappa) * 2**(-1-kappa) * b * (om_f - offset) * hyper * (1 + term1)**kappa

    numpy_res = {}
    numpy_res.update({
        'a':aa,
        'b':bb,
        'c':cc,
        'z':zz,
        'term1':term1,
        'phase':phase
        })

    return sympy_res, numpy_res, control_phase



sympy_res, numpy_res, control_phase = hypergeom_components(times, **params_freq_mrd)

print("value of constants a,b and c")

print("sympy version")
print("a = {}".format(sympy_res['a']))
print("b = {}".format(sympy_res['b']))
print("c = {}".format(sympy_res['c']))

print("numpy version")
print("a = {}".format(numpy_res['a']))
print("b = {}".format(numpy_res['b']))
print("c = {}".format(numpy_res['c']))

print("compare z")
print('t\tsz\tnz\tsphase\tnphase\tmodel phase\tcphase')
for t, sz, nz, sphase, nphase, mphase, cphase in list(zip(times, sympy_res['z'], numpy_res['z'], sympy_res['phase'], numpy_res['phase'], phase_mrd, control_phase)):
    print('{}\t{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}'.format(t, sz, nz, sphase, nphase, mphase, cphase))


plt.figure()
plt.plot(times, sympy_res['z'], label='sympy')
plt.plot(times, numpy_res['z'], label='numpy', ls='--')
plt.legend()
plt.show()

plt.figure()
plt.plot(times, np.abs(sympy_res['z'] - numpy_res['z']), label='|sympy-numpy|')
plt.legend()
plt.yscale('log')
plt.show()

