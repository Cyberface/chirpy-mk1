# from scipy.special import hyp2f1
from mpmath import mp
from mpmath import hyp2f1 as mphyp2f1
from mpmath import tanh as mptanh
# set precision for sympy.
# 64 seems to be enough
mp.dps=64

import numpy as np

from .pn import TaylorT3_phase, TaylorT3_Omega_new, Hhat22_x

def analytic_phase_ins_ansatz(t, eta, tc, b, c, M=1, phi0=0, m=2):
    tau = eta * (tc - t) / (5*M)

    t1=40*b*M / eta / tau**(1/8)
    t2=20*c*M / eta / tau**(1/4)

    phase = TaylorT3_phase(t, tc, eta, M=M, phi0=phi0, m=m) + t1 + t2

#     return phase - phase[0]
    return phase

def analytic_phase_int_ansatz(t, tp, tdamp, lor_amp, a0, a1):

    """
    because my times are negative and I have a log I have to evaluate the
    term inside the log with -(t-tp) instead of (t-tp)
    """

#     phase = a0 * np.log(-(t-tp)) + a1*t + lor_amp*np.arctan((t-tp)/tdamp)
    phase = a0 * np.log(np.abs(t-tp)) + a1*t + lor_amp*np.arctan((t-tp)/tdamp)
#     return phase - phase[0]
    return phase

def analytic_phase_mrd_ansatz(t, t0, b, om_f, offset=0.175, kappa=0.44):
    """
    the hyp2f1 function will produce an inf if the 'z' argument (4th arg) is >= 1.
    This can happen due to finite precision errors in there term (0.5*(1+mptanh((tt-t0)/b))
    as tt >> t0
    To fix this I used a stupid number of dps in sympy mpmath.
    But should do a clever truncate and attact a constant slope instead.
    """
    try:
        kappa = kappa.value
    except:
        pass

    if type(t) in [int, float]:
        ts = np.array([t], dtype=np.float64)
    else:
        ts = t
    hyper = np.array([mphyp2f1(1, kappa, 1+kappa, 0.5*(1+mptanh((tt-t0)/b))) for tt in ts], dtype=np.float64)
    term1 = np.array([mptanh((tt-t0)/b) for tt in ts], dtype=np.float64)
    phase = offset * ts + (1/kappa) * 2**(-1-kappa) * b * (om_f - offset) * hyper * (1 + term1)**kappa

#     return phase - phase[0]
    return phase

def freq_ins_ansatz(t, eta, params):
    """
    this is the frequency inspiral ansatz.
    I needed an separate function so that I could use it in the amplitude inspiral model
    """

    tc = params['tc']
    b = params['b']
    c = params['c']
    M = 1

    tau = eta * (tc - t) / (5*M)
    model = (TaylorT3_Omega_new(t, tc, eta, M) + b*tau**(-9./8.) + c*tau**(-10./8.))

    return model


def freq_int_ansatz(t, params):

    tdamp = params['tdamp']
    tp = params['tp']
    lor_amp = params['lor_amp']
    a0 = params['a0']
    a1 = params['a1']

    model = (lor_amp * tdamp) / ( (t-tp)**2 + tdamp**2 ) + a0 / (t-tp) + a1

    return model

def freq_mrd_ansatz(t, params):

    t0 = params['t0']
    kappa = params['kappa']
    b = params['b']
    om_f = params['om_f']
    offset = params['offset']

    dt = t - t0
    num = 1. + np.tanh((dt/b))
    den = 2.

    model = offset + (om_f - offset) * ((num/den)**(kappa))

    return model

def amp_ins_ansatz(t, eta, params):

    tc = params['tc']

    a0 = params['a0']
    a1 = params['a1']

    tau = (tc-t)

    params_freq_ins = {
        'tc':params['tc'],
        'b':params['b'],
        'c':params['c']
    }

    GW22AngFreq = freq_ins_ansatz(t, eta, params_freq_ins)
    OrgAngFreq = GW22AngFreq / 2.

    M = 1
    x = (M*OrgAngFreq)**(2./3)

    T3amp = np.abs( Hhat22_x(x, eta) )

    model = T3amp + a0*tau**(-9./8.) + a1*tau**(-10./8.)

    return model

def amp_int_ansatz(t, params):

    tdamp = params['tdamp']
    tp = params['tp']
    lor_amp = params['lor_amp']
    a0 = params['a0']
    a1 = params['a1']

    model = (lor_amp * tdamp) / ( (t-tp)**2 + tdamp**2 ) + a0 / (t - tp) + a1

    return model

def amp_mrd_ansatz(t, params):

    A0 = params['A0']
    a1 = params['a1']
    a2 = params['a2']
    a3 = params['a3']
    b = params['b']

    kappa = params['kappa']
    t0 = params['t0']

    fhat = ((1 + np.tanh( (t-t0)/b ))/2)**kappa

    den = 1. + (a1 * ( fhat**2 - fhat**4 ))
    den += a2 * (fhat**4 - fhat**6)
    den += a3 * (fhat**6 - fhat**8)


    tanh_ab = np.tanh((t-t0)/b)
    tanh_ab_over_2_p_05 = tanh_ab/2 + 0.5

    fhatdot =  kappa*(1 - tanh_ab**2)*(tanh_ab_over_2_p_05)**kappa/(2*b*(tanh_ab_over_2_p_05))

    # normalise so that A0 is peak amp
    # den /= np.max(den)
    # fhatdot /= np.max(fhatdot)

    model = A0*fhatdot * den

    return np.sqrt(model)

def PSF_ansatz(params, eta):

    eta2 = eta*eta
    eta3 = eta2*eta
    eta4 = eta3*eta

    a = params['a']
    b = params['b']
    c = params['c']
    d = params['d']
    e = params['e']

    model = a + b*eta + c*eta2 + d*eta3 + e*eta4
    return model

def PSF_freq_ins(name, eta):
    if name == 'tc':
        params = dict(a=470.597717, b=-5211.88496, c=27529.8452, d=-69999.9986, e=65736.7732)
        value = PSF_ansatz(params, eta)
    elif name == 'b':
        params = dict(a=-0.03290704, b=-3.63518207, c=27.5751235, d=-82.2823976, e=89.9564049)
        value = PSF_ansatz(params, eta)
    elif name == 'c':
        params = dict(a=0.50445336, b=8.38150584, c=-64.9118203, d=205.626413, e=-244.753155)
        value = PSF_ansatz(params, eta)

    return value

def PSF_freq_int(name, eta):
    if name == 'tdamp':
        params = dict(a=74.9257490, b=121.163911, c=-2304.73380, d=11707.1868, e=-19425.1658)
        value = PSF_ansatz(params, eta)
    elif name == 'tp':
        params = dict(a=41.4577189, b=50.9110867, c=-1072.31292, d=4619.15831, e=-6824.93772)
        value = PSF_ansatz(params, eta)
    elif name == 'lor_amp':
        params = dict(a=-8.68954175, b=-131.704263, c=1397.26505, d=-5633.73808, e=8096.02270)
        value = PSF_ansatz(params, eta)
    elif name == 'a0':
        params = dict(a=-10.9733699, b=-108.883606, c=1035.63718, d=-4081.64431, e=5831.75177)
        value = PSF_ansatz(params, eta)
    elif name == 'a1':
        params = dict(a=0.09206339, b=-0.67906359, c=4.49944842, d=-15.2687804, e=20.0192924)
        value = PSF_ansatz(params, eta)

    return value

def PSF_freq_mrd(name, eta):
    if name == 't0':
        params = dict(a=7.67140690, b=-2.76322419, c=22.3188512, d=-230.782467, e=921.244695)
        value = PSF_ansatz(params, eta)
    return value


def PSF_amp_ins(name, eta):
    if name == 'a0':
        params = dict(a=-34.6815964, b=846.370858, c=-8325.94229, d=34543.5326, e=-51438.2610)
        value = PSF_ansatz(params, eta)
    elif name == 'a1':
        params = dict(a=62.9956269, b=-1431.56072, c=12988.8856, d=-50271.4802, e=69999.9999)
        value = PSF_ansatz(params, eta)

    return value

def PSF_amp_int(name, eta):
    if name == 'tp':
        params = dict(a=54.7975453, b=-441.295715, c=3029.19697, d=-8677.58586, e=8569.67996)
        value = PSF_ansatz(params, eta)
    elif name == 'lor_amp':
        params = dict(a=0.33612583, b=-19.3781791, c=539.583918, d=-2765.56602, e=4623.06556)
        value = PSF_ansatz(params, eta)
    elif name == 'a0':
        params = dict(a=0.06625443, b=-41.1738150, c=300.284121, d=-1897.24393, e=3389.82409)
        value = PSF_ansatz(params, eta)
    elif name == 'a1':
        params = dict(a=0.00194817, b=0.71936078, c=-0.59881835, d=-0.06359352, e=1.86346496)
        value = PSF_ansatz(params, eta)

    return value

def PSF_amp_mrd(name, eta):
    if name == 'A0':
        params = dict(a=-0.13230411, b=7.38840303, c=42.0703464, d=452.332609, e=-336.085451)
        value = PSF_ansatz(params, eta)
    elif name == 'a1':
        params = dict(a=0.49126184, b=103.259720, c=-985.164119, d=3454.43062, e=-4334.34133)
        value = PSF_ansatz(params, eta)
    elif name == 'a2':
        params = dict(a=-7.22382530, b=-160.452199, c=1496.19345, d=-4561.12155, e=4607.34592)
        value = PSF_ansatz(params, eta)
    elif name == 'a3':
        params = dict(a=8.34501230, b=169.664308, c=-1361.57695, d=3391.01510, e=-2118.69289)
        value = PSF_ansatz(params, eta)
    elif name == 't0':
        params = dict(a=12.9240884, b=-11.2209381, c=134.758871, d=-1152.33440, e=2624.38931)
        value = PSF_ansatz(params, eta)
    elif name == 'kappa':
        params = dict(a=0.05366861, b=0.45794648, c=-5.06266379, d=27.2068135, e=-48.2870291)
        value = PSF_ansatz(params, eta)

    return value