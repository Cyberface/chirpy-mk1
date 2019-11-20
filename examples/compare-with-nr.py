import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import chirpy_mk1
from chirpy_mk1.nrutils import Psi4
from chirpy_mk1.utils import match, coalign

import pycbc.types
import pycbc.waveform.utils

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

npts_time = 1000*10
npts_mass_ratio = len(nrfiles)

t1=-3100
#t1=-500
t2=100

psi4s = {}
for k,v in list(nrfiles.items()):
    psi4s.update({k:Psi4(v, ell, mm, npts_time,t1=t1,t2=t2)})

for k in psi4s.keys():
    print("working: {}".format(k))
    # generate mk1 waveform
    mk1 = chirpy_mk1.Mk1(psi4s[k].times_hlm, psi4s[k].q)

    mk1_ts = pycbc.types.TimeSeries(np.real(mk1.h), delta_t=mk1.times[1] - mk1.times[0])
    nr_ts = pycbc.types.TimeSeries(np.real(psi4s[k].hlm), delta_t=psi4s[k].times_hlm[1] - psi4s[k].times_hlm[0])

    mk1_ts, nr_ts = pycbc.waveform.utils.coalign_waveforms(mk1_ts, nr_ts)

    _, mk1_idx = mk1_ts.abs_max_loc()
    mk1_shift = mk1_ts.sample_times[mk1_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(nr_ts.sample_times - mk1_shift, nr_ts, label='NR')
    axes[0].plot(mk1_ts.sample_times - mk1_shift, mk1_ts, ls='--', label='Mk1')
    axes[0].legend
    axes[0].set_title(k)
    axes[0].set_xlim(t1, t2)

    axes[1].plot(nr_ts.sample_times - mk1_shift, nr_ts, label='NR')
    axes[1].plot(mk1_ts.sample_times - mk1_shift, mk1_ts, ls='--', label='Mk1')
    axes[1].set_xlim(-200,100)

    for ax in axes:
        ax.set_xlabel('t/M')
        ax.set_ylabel('Re(h)')

    plt.savefig('wf-compare-nr-{}.png'.format(k))
    plt.close()
    the_match =match(mk1.h, psi4s[k].hlm, mk1.times)

    print("match = {}".format(the_match))
