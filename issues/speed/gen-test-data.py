import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import lal
from pycbc import waveform
from pycbc.types import TimeSeries
import pycbc.filter
import pycbc.waveform.utils

matplotlib.rcParams.update({'font.size': 16})

import chirpy_mk1
from chirpy_mk1 import utils

import phenom

import time

def gen_pycbc_waveform(mass1, mass2, approximant, delta_t=1./8192, f_lower=30, distance=1, t1=None, t2=None, *args, **kwargs):
    q = mass1/mass2
    mtot = mass1 + mass2
    hp, hc = waveform.get_td_waveform(approximant=approximant,
                             mass1=mass1,
                             mass2=mass2,
                             delta_t=delta_t,
                             f_lower=f_lower,
                             distance=distance)
    times = phenom.StoM(hp.sample_times.numpy(), mtot)
    ylm = np.abs(lal.SpinWeightedSphericalHarmonic(0,0,-2,2,2))
    amp_scale = utils.td_amp_scale(mtot, distance) * ylm
    hp_array = hp.numpy() / amp_scale
    hc_array = hc.numpy() / amp_scale

    h_array = hp_array - 1.j*hc_array

    if t1 is None:
        t1 = times[0]
    if t2 is None:
        t2 = times[-1]

    mask = (times >= t1) & (times <= t2)

    times = times[mask]
    h_array = h_array[mask]

    return times, h_array




if __name__ == "__main__":


    # define parameters

    outdir = 'data'
    try:
        os.mkdir(outdir)
    except FileExistsError:
        YN = input("Directory '{}' exists. Overwrite existing data? [Y/N] ".format(outdir))
        yns = ["Y", "N"]
        if YN not in yns:
            print("please enter one of the following: {}".format(yns))
        else:
            if YN == 'Y':
                print("continuing...")
                pass
            else:
                sys.exit("Exiting!")

    sys1=dict(
            name_template='high-mass-{}',
            mass1=100,
            mass2=10,
            delta_t = 1./8192,
            f_lower=20,
            t1=None,
            t2=100
            )

    # derived parameters
    sys1.update({'q':sys1['mass1']/sys1['mass2'], 'mtot':sys1['mass1']+sys1['mass2']})
    
    sys2=dict(
            name_template='low-mass-{}',
            mass1=3,
            mass2=1,
            delta_t = 1./8192/2,
            f_lower=20,
            t1=None,
            t2=100
            )

    # derived parameters
    sys2.update({'q':sys2['mass1']/sys2['mass2'], 'mtot':sys2['mass1']+sys2['mass2']})

    systems = [sys1, sys2]

    # Note "SEOBNRv4" *HAS* to be first in this list
    approximants = ["SEOBNRv4", "IMRPhenomD"]
    
    results={}
    
    timings = {}
    
    for sys in systems:
        print("system parameters:")
        print(sys)
    
        for approximant in approximants:
            if approximants[0] != "SEOBNRv4":
                raise Exception("SEOBNRv4 must be first in 'approximants' list, got {}".format(approximants[0]))
    
            sys.update({'name':sys['name_template'].format(approximant)})
            sys.update({'approximant':approximant})
            print("working case:")
            print("{}".format(sys['name']))
    
            if approximant != "SEOBNRv4":
                sys.update({'t1':start_time})
    
    
            t1 = time.time()
            wf_t, wf_h = gen_pycbc_waveform(**sys)
            t2 = time.time()
            dt = t2-t1
            print("{}: generation time = {}".format(approximant, dt))
    
            timings.update({sys['name']:dt})
    
            # when we ifft phenomD we get a much longer time series than we need so
            # we get the start time from EOB and use that to trim the phenomD data
            if approximant == "SEOBNRv4":
                start_time = wf_t[0]
    
            this=dict(name=sys['name'], time=dt)
    
            results.update({sys['name']:this})
    
            outname = outdir + '/' + sys['name'] + "_t_realh" + ".dat"
    
            if approximant != "SEOBNRv4":
                mask = wf_t >= start_time
                wf_h = wf_h[mask]
                wf_t = wf_t[mask]
    
            np.savetxt(outname, list(zip(wf_t, np.real(wf_h))))
    
            if approximant == "SEOBNRv4":
                print("generate chirpy-mk1 waveform on EOB grid")
    
                t1 = time.time()
                mk1 = chirpy_mk1.mk1.Mk1(wf_t, sys['q'])
                t2 = time.time()
                dt = t2-t1
                print("chirpy_mk1: generation time = {}".format(dt))
                cname = sys['name_template'].format('chirpy_mk1')
                timings.update({cname:dt})
    
                outname = outdir + '/' + cname + "_t_realh" + ".dat"
                np.savetxt(outname, list(zip(mk1.times, np.real(mk1.h))))

            print("match between {} and the mk1:".format(approximant))
            wf_dt = wf_t[1] - wf_t[0]
            
            mk1_t = mk1.times
            mk1_dt = mk1_t[1] - mk1_t[0]
            
            #f_lower_int = phenom.HztoMf(sys['f_lower'], sys['mtot'])
            
            ts1 = TimeSeries(np.real(wf_h), delta_t=wf_dt)
            ts2 = TimeSeries(np.real(mk1.h), delta_t=mk1_dt)
            tlen = max(len(ts1), len(ts2))
            ts1.resize(tlen)
            ts2.resize(tlen)
            #ma, _  = pycbc.filter.match(ts1, ts2, low_frequency_cutoff=f_lower_int)
            ma, _  = pycbc.filter.match(ts1, ts2)
            print("\t {}".format(ma))


            print('plotting aligned waveform')
            #ts1, ts2 = pycbc.waveform.utils.coalign_waveforms(ts1, ts2, low_frequency_cutoff=f_lower_int)
            ts1, ts2 = pycbc.waveform.utils.coalign_waveforms(ts1, ts2)
            
            _, mk1_idx = ts2.abs_max_loc()
            mk1_shift = ts2.sample_times[mk1_idx]

            figname = outdir + '/' + "{}-vs-mk1".format(sys['name']) + '.png'

            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            axes[0].plot(ts1.sample_times - mk1_shift, ts1, label='{}'.format(approximant))
            axes[0].plot(ts2.sample_times - mk1_shift, ts2, ls='--', label='Mk1')
            axes[0].legend(loc='upper left')
            axes[0].set_title('match = {:.4f}'.format(ma))
            axes[0].set_xlim(start_time, 100)

            axes[1].plot(ts1.sample_times - mk1_shift, ts1, label='{}'.format(approximant))
            axes[1].plot(ts2.sample_times - mk1_shift, ts2, ls='--', label='Mk1')
            axes[1].set_xlim(-500,100)

            for ax in axes:
                ax.set_xlabel('t/M')
                ax.set_ylabel('Re(h)')

            plt.savefig(figname)
            plt.close()


    print("saving timing information")

    outname = outdir + '/' + 'timings.txt'
    f=open(outname,"w+")
    for k, v in timings.items():
        f.write("{}\t{}\n".format(k,v))
    f.close()

    print("done!")
