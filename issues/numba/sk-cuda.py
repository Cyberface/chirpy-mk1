import chirpy_mk1

from chirpy_mk1.utils import G_Newt, c_ls, MSUN_SI, MTSUN_SI, MRSUN_SI, PC_SI, GAMMA, Msun_to_sec
from chirpy_mk1.ansatz import PSF_freq_ins, PSF_freq_int, PSF_freq_mrd
from chirpy_mk1.ansatz import PSF_amp_ins, PSF_amp_int, PSF_amp_mrd
from chirpy_mk1.ansatz import amp_ins_ansatz, amp_int_ansatz, amp_mrd_ansatz
from chirpy_mk1.ansatz import freq_ins_ansatz, freq_int_ansatz, freq_mrd_ansatz
from chirpy_mk1.ansatz import analytic_phase_ins_ansatz, analytic_phase_int_ansatz
from chirpy_mk1.ansatz import analytic_phase_mrd_ansatz

from chirpy_mk1.ansatz import analytic_phase_mrd_ansatz_approx

import numpy as np
from scipy import optimize
import phenom

import math

from numba import vectorize

def setup_ins_coeffs(eta):
    freq_inc_tc = PSF_freq_ins('tc',eta)
    freq_inc_b = PSF_freq_ins('b',eta)
    freq_inc_c = PSF_freq_ins('c',eta)

    # amp
    params_amp_ins={}
    params_amp_ins.update({
        'a0': np.float64(PSF_amp_ins('a0', eta)),
        'a1': np.float64(PSF_amp_ins('a1', eta))
    })

    # amplitude ins model depends on freq ins model
    params_amp_ins.update({
        'tc':np.float64(freq_inc_tc),
        'b':np.float64(freq_inc_b),
        'c':np.float64(freq_inc_c)
    })
    
    return params_amp_ins

def mk1_amp_ins(times, eta, params_amp_ins):
    return amp_ins_ansatz(times, eta, params_amp_ins)

times = np.linspace(-1000, -500, 10000)
# times = np.arange(-1000, -500, 1./2048/4)
eta = np.float64(0.25)
params_amp_ins = setup_ins_coeffs(eta)

tc = params_amp_ins['tc']
a0 = params_amp_ins['a0']
a1 = params_amp_ins['a1']
b = params_amp_ins['b']
c = params_amp_ins['c']

target = 'cuda'

####### new numba functions

# @njit
# @njit(['float64[:](float64[:],float64,float64,float64)'])
@vectorize(['float64(float64,float64,float64,float64)'], target=target)
def numba_vec_TaylorT3_Omega_new(t, tc, eta, M):

#     Msec = Msun_to_sec(M)
    Msec = M

    pi2 = math.pi*math.pi

    c1 = eta/(5.*Msec)

    td = c1 * (tc - t)

#     td = np.sqrt(td**2 + 1)

    theta = td**(-1./8.) # -1./8. = -0.125

    theta2 = theta*theta
    theta3 = theta2*theta
    theta4 = theta3*theta
    theta5 = theta4*theta
    theta6 = theta5*theta
    theta7 = theta6*theta

    # pre factor
    ftaN = 1. / ( 8. * math.pi * Msec  )
    # 0PN
    fts1 = 1.
    # 0.5PN = 0 in GR
    # 1PN
    fta2 = 7.43/26.88 + 1.1/3.2 * eta
    # 1.5PN
    fta3 = -3./10. * math.pi
    # 2PN
    fta4 = 1.855099/14.450688 + 5.6975/25.8048 * eta + 3.71/20.48 * eta*eta
    # 2.5PN
    fta5 = (-7.729/21.504 + 1.3/25.6 * eta) * math.pi
    # 3PN
    fta6 = -7.20817631400877/2.88412611379200 + 5.3/20.0 * pi2 + 1.07/2.80 * GAMMA  \
           + (25.302017977/4.161798144 - 4.51/20.48 * pi2) * eta \
           - 3.0913/183.5008 * eta*eta + 2.35925/17.69472 * eta*eta*eta

    # 3.5PN
    fta7 = (-1.88516689/4.33520640 - 9.7765/25.8048 * eta + 1.41769/12.90240 * eta*eta) * math.pi

    # 3PN log term
    ftal6 = 1.07/2.80


    full = theta3*ftaN * (fts1 \
             + fta2*theta2 \
             + fta3*theta3 \
             + fta4*theta4 \
             + fta5*theta5 \
             + (fta6 + ftal6*math.log(2.*theta))*theta6 \
             + fta7*theta7)

    return full * 2 * math.pi # 2pi to go from freq to angular freq

@vectorize(['float64(float64,float64,float64,float64,float64)'], target=target)
def numba_vec_freq_ins_ansatz(t, eta, tc, b, c):
#     tc = params['tc']
#     b = params['b']
#     c = params['c']
    M = 1

    tau = eta * (tc - t) / (5*M)
    
    t3 = numba_vec_TaylorT3_Omega_new(t, tc, eta, np.float64(1))
#     t3 = np.array([explicit_TaylorT3_Omega_new(tt, tc, eta, M) for tt in t])
    
    model = (t3 + b*tau**(-9./8.) + c*tau**(-10./8.))

    return model

@vectorize(['float64(float64,float64)'], target=target)
def numba_vec_Hhat22_x(x, eta):

#     xarr = np.zeros(6, dtype=np.complex128)
    xarr = [0+1.j, 0+1.j, 0+1.j, 0+1.j, 0+1.j, 0+1.j]

    C = 0.577216 # is the Euler constant

    xarr[0] = 1.
    xarr[1] = -107./42 + 55*eta/42
    xarr[2] = 2.*math.pi
    xarr[3] = -2173./1512 - 1069.*eta/216 + 2047.*eta**2/1512
    xarr[4] = (-107*math.pi/21 - 24.*1.j*eta + 34.*math.pi*eta/21) # there is an i... not sure what to do...

    x5a = 27027409./646800 - 856.*C/105 + 428*1.j*math.pi/105 + 2.*math.pi**2/3
    x5b = (-278185./33264 + 41*math.pi**2/96)*eta - 20261.*eta**2/2772 + 114635.*eta**3/99792

    x5log =  - 428.*math.log(16*x)/105

    xarr[5] = (x5a) + x5b # there is an i...  not sure what to do...

    pre = math.sqrt(16.*math.pi/5) * 2 * eta

    pn = xarr[0] + x*xarr[1] + x**(3/2.)*xarr[2] + x**2*xarr[3] + x**(5/2.)*xarr[4] + x**3*(xarr[5] + x5log)

    return pre * abs(pn) * x

@vectorize(['float64(float64,float64,float64,float64,float64,float64,float64)'], target=target)
def numba_vec_amp_ins_ansatz(t, eta, tc, a0, a1, b, c):
    tau = (tc-t)
    
    GW22AngFreq = numba_vec_freq_ins_ansatz(t, eta, tc, b, c)
    OrgAngFreq = GW22AngFreq / 2.

    M = 1
    x = (M*OrgAngFreq)**(2./3)

#     T3amp = abs( numba_vec_Hhat22_x(x, eta) )
    T3amp = numba_vec_Hhat22_x(x, eta)

    model = T3amp + a0*tau**(-9./8.) + a1*tau**(-10./8.)

    return model

import time

t1 = time.time()
amp_numba_vec = numba_vec_amp_ins_ansatz(times, eta, tc, a0, a1, b, c)
t2 = time.time()


print('took = {}'.format(t2-t1))

