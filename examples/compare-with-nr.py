import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.integrate import quad
from scipy.optimize import minimize

import chirpy_mk1
from chirpy_mk1.nrutils import Psi4
from chirpy_mk1.utils import match

# get data
nrfiles = dict(
    q1='/Users/spx8sk/work/data/SXS_BBH_0180_Res4.h5',
    q2='/Users/spx8sk/work/data/SXS_BBH_0169_Res5.h5',
    q4='/Users/spx8sk/work/data/SXS_BBH_0167_Res5.h5',
    q5='/Users/spx8sk/work/data/SXS_BBH_0107_Res5.h5',
    q10='/Users/spx8sk/work/data/SXS_BBH_0303_Res5.h5',
    q18="/Users/spx8sk/work/git/stk/ml/waveforms/bob/q18a0a0c025_144-22-hybrid.h5"
)

ell = 2
mm = 2

npts_time = 1000*6
npts_mass_ratio = len(nrfiles)

t1=-3100
t2=60

psi4s = {}
for k,v in list(nrfiles.items()):
    psi4s.update({k:Psi4(v, ell, mm, npts_time,t1=t1,t2=t2)})


# plt.figure()
# plt.plot(psi4s['q1'].times_hlm, np.real(psi4s['q1'].hlm))
# plt.show()

# plt.figure()
# plt.plot(psi4s['q1'].times_hlm, np.real(psi4s['q1'].hlm_ang_freq))
# plt.show()


for k in psi4s.keys():
    print("working: {}".format(k))
    # generate mk1 waveform
    mk1 = chirpy_mk1.Mk1(psi4s[k].times_hlm, psi4s[k].q)

    # align over time and phase shift
    # win1 = t1 + 100
    # win2 = win1 + 500
    win1 = -500
    win2 = 60
    iNR = IUS(mk1.times, mk1.phase)
    iMk1 = IUS(mk1.times, psi4s[k].hlm_phase)

    def dephasing(z):
        """
        from T.D
        """
        dt, dphi = z
        return quad(lambda t: np.abs((iNR(t)-iMk1(t+dt)+dphi)),win1,win2, limit=200)[0]

    dt, dphi = minimize(dephasing, [0.77683568, 2.89636107],  tol=1e-12).x
    # print(mini)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(mk1.times, np.real(psi4s[k].hlm), label='NR')
    axes[0].plot(mk1.times + dt, np.real(mk1.h * np.exp(-1.j*dphi)), ls='--', label='Mk1')
    axes[0].legend
    axes[0].set_title(k)

    axes[1].plot(mk1.times, np.real(psi4s[k].hlm), label='NR')
    axes[1].plot(mk1.times + dt, np.real(mk1.h * np.exp(-1.j*dphi)), ls='--', label='Mk1')
    axes[1].set_xlim(-100,60)

    for ax in axes:
        ax.set_xlabel('t/M')
        ax.set_ylabel('Re(h)')

    plt.savefig('wf-compare-nr-{}.png'.format(k))
    plt.close()
    the_match = np.max(np.abs(match(mk1.h, psi4s[k].hlm, mk1.times)))

    print("match = {}".format(the_match))
