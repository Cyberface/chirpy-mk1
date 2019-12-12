from math import pi as PI
from math import log, fabs, sqrt
from numba import vectorize, cuda

from math import pi as PI
from math import log, fabs, sqrt
GAMMA =  0.5772156649015329

@cuda.jit(device=True)
def TaylorT3_Omega_new(t, tc, eta, M):
    Msec = M

    pi2 = PI*PI

    c1 = eta/(5.*Msec)

    td = c1 * (tc - t)


    theta = td**(-1./8.)

    theta2 = theta*theta
    theta3 = theta2*theta
    theta4 = theta3*theta
    theta5 = theta4*theta
    theta6 = theta5*theta
    theta7 = theta6*theta

    # pre factor
    ftaN = 1. / ( 8. * PI * Msec  )
    # 0PN
    fts1 = 1.
    # 0.5PN = 0 in GR
    # 1PN
    fta2 = 7.43/26.88 + 1.1/3.2 * eta
    # 1.5PN
    fta3 = -3./10. * PI
    # 2PN
    fta4 = 1.855099/14.450688 + 5.6975/25.8048 * eta + 3.71/20.48 * eta*eta
    # 2.5PN
    fta5 = (-7.729/21.504 + 1.3/25.6 * eta) * PI
    # 3PN
    fta6 = -7.20817631400877/2.88412611379200 + 5.3/20.0 * pi2 + 1.07/2.80 * GAMMA  \
           + (25.302017977/4.161798144 - 4.51/20.48 * pi2) * eta \
           - 3.0913/183.5008 * eta*eta + 2.35925/17.69472 * eta*eta*eta

    # 3.5PN
    fta7 = (-1.88516689/4.33520640 - 9.7765/25.8048 * eta + 1.41769/12.90240 * eta*eta) * PI

    # 3PN log term
    ftal6 = 1.07/2.80


    full = theta3*ftaN * (fts1 \
             + fta2*theta2 \
             + fta3*theta3 \
             + fta4*theta4 \
             + fta5*theta5 \
             + (fta6 + ftal6*log(2.*theta))*theta6 \
             + fta7*theta7)

    return full * 2 * PI # 2pi to go from freq to angular freq

@cuda.jit(device=True)
def freq_ins_ansatz(t, eta, tc,b,c,M):
    """
    this is the frequency inspiral ansatz.
    I needed an separate function so that I could use it in the amplitude inspiral model
    """

    tau = eta * (tc - t) / (5*M)
    model = TaylorT3_Omega_new(t, tc, eta, M) + b*tau**(-9./8.) + c*tau**(-10./8.)

    return model

@cuda.jit(device=True)
def Hhat22_x(x, eta):
    C = 0.577216 # is the Euler constant

    xarr0 = 1.
    xarr1 = -107./42 + 55*eta/42
    xarr2 = 2.*PI
    xarr3 = -2173./1512 - 1069.*eta/216 + 2047.*eta**2/1512
    xarr4 = (-107*PI/21 - 24.*1.j*eta + 34.*PI*eta/21)

    x5a = 27027409./646800 - 856.*C/105 + 428*1.j*PI/105 + 2.*PI**2/3
    x5b = (-278185./33264 + 41*PI**2/96)*eta - 20261.*eta**2/2772 + 114635.*eta**3/99792

    x5log =  - 428.*log(16*x)/105

    xarr5 = (x5a) + x5b

    pre = sqrt(16.*PI/5) * 2 * eta

    pn = xarr0 + x*xarr1 + x**(3/2.)*xarr2 + x**2*xarr3 + x**(5/2.)*xarr4 + x**3*(xarr5 + x5log)

    return abs(pre * pn * x)

@vectorize(['float64(float64,float64,float64,float64,float64,float64,float64)'],target='cuda')
def amp_ins_ansatz(t, eta, tc, a0, a1, b, c):

    tau = (tc-t)

    GW22AngFreq = freq_ins_ansatz(t, eta, tc, b, c, 1.)
    OrgAngFreq = GW22AngFreq / 2.

    M = 1.
    x = (M*OrgAngFreq)**(2./3.)

    T3amp = Hhat22_x(x, eta)

    model = T3amp + a0*tau**(-9./8.) + a1*tau**(-10./8.)

    return model