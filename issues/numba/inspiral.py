from numba import vectorize,float64,jit
import numpy as np
import math
from chirpy_mk1.ansatz import PSF_freq_ins, PSF_freq_int, PSF_freq_mrd
from chirpy_mk1.ansatz import PSF_amp_ins, PSF_amp_int, PSF_amp_mrd
from chirpy_mk1.ansatz import amp_ins_ansatz, amp_int_ansatz, amp_mrd_ansatz
import matplotlib.pyplot as plt
from chirpy_mk1.utils import G_Newt, c_ls, MSUN_SI, MTSUN_SI, MRSUN_SI, PC_SI, GAMMA

def setup_ins_coefficients(eta):    
    params_amp_ins={}
    freq_inc_tc = PSF_freq_ins('tc',eta)
    freq_inc_b = PSF_freq_ins('b',eta)
    freq_inc_c = PSF_freq_ins('c',eta)
    params_amp_ins.update({
    'a0': PSF_amp_ins('a0', eta),
    'a1': PSF_amp_ins('a1', eta)
    })

# amplitude ins model depends on freq ins model
    params_amp_ins.update({
    'tc':freq_inc_tc,
    'b':freq_inc_b,
    'c':freq_inc_c
    })
    params_freq_ins={}
    params_freq_ins.update({
    'tc':freq_inc_tc,
    'b':freq_inc_b,
    'c':freq_inc_c
    })
    return params_amp_ins


amp_ins = amp_ins_ansatz(times, eta, setup_ins_coefficients(eta))
plt.plot(times,amp_ins)