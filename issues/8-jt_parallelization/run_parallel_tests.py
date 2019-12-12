import chirpy_mk1
import numpy as np
import funcs
from chirpy_mk1.ansatz import PSF_amp_ins, PSF_freq_ins
import timeit

q = 1.
eta = 1./(1. + q)**2
times = np.arange(-1000.,-500.,.001)

freq_inc_tc = PSF_freq_ins('tc',eta)
req_inc_b = PSF_freq_ins('b',eta)
freq_inc_c = PSF_freq_ins('c',eta)
amp_a0 = PSF_amp_ins('a0', eta)
amp_a1 = PSF_amp_ins('a1', eta)

params = {
     'tc':freq_inc_tc,
     'b':freq_inc_b,
     'c':freq_inc_c,
     'a0':amp_a0,
     'a1':amp_a1
}

chrprun = "chirpy_mk1.amp_ins_ansatz(times,eta,params)"
funcsrun = "funcs.amp_ins_ansatz(times, eta, freq_inc_tc, amp_a0, amp_a1, freq_inc_b, freq_inc_c)"

iterations = 100
reruns = 100

chrpdata = timeit.repeat(chrprun,globals=globals(),number=iterations,repeat=reruns)/100
funcsdata = timeit.repeat(funcsrun,globals=globals(),number=iterations,repeat=reruns)/100

print('chirpy_mk1 mean time (in s): {}'.format(np.mean(chrpdata)))
print('parallel mean time (in s): {}'.format(np.mean(funcsdata)))