import numpy as np

from .utils import Msun_to_sec
from .utils import G_Newt, c_ls, MSUN_SI, MTSUN_SI, MRSUN_SI, PC_SI, GAMMA

def TaylorT3_Omega_new(t, tc, eta, M):
    """
    22 mode angular GW frequency
    equation 7 in 0901.2437

    3.5PN term from https://arxiv.org/pdf/gr-qc/0610122.pdf and https://arxiv.org/pdf/0907.0700.pdf
    and this too apparently https://arxiv.org/pdf/gr-qc/0406012.pdf?

    https://git.ligo.org/lscsoft/lalsuite/blob/master/lalsimulation/src/LALSimInspiralTaylorT3.c

    https://git.ligo.org/lscsoft/lalsuite/blob/master/lalsimulation/src/LALSimInspiralPNCoefficients.c

    t: time
    tc: coalescence time
    eta: symmetric mass ratio
    M: total mass (Msun)
    """

    #Msec = Msun_to_sec(M)
    Msec = M

    pi2 = np.pi*np.pi

    c1 = eta/(5.*Msec)

    td = c1 * (tc - t)

#     td = np.sqrt(td**2 + 1)

    #theta = td**(-1./8.) # -1./8. = -0.125
    theta = td**(-0.125) # -1./8. = -0.125

    theta2 = theta*theta
    theta3 = theta2*theta
    theta4 = theta3*theta
    theta5 = theta4*theta
    theta6 = theta5*theta
    theta7 = theta6*theta

    # pre factor
    ftaN = 1. / ( 8. * np.pi * Msec  )
    # 0PN
    fts1 = 1.
    # 0.5PN = 0 in GR
    # 1PN
    fta2 = 7.43/26.88 + 1.1/3.2 * eta
    # 1.5PN
    fta3 = -3./10. * np.pi
    # 2PN
    fta4 = 1.855099/14.450688 + 5.6975/25.8048 * eta + 3.71/20.48 * eta*eta
    # 2.5PN
    fta5 = (-7.729/21.504 + 1.3/25.6 * eta) * np.pi
    # 3PN
    fta6 = -7.20817631400877/2.88412611379200 + 5.3/20.0 * pi2 + 1.07/2.80 * GAMMA  \
           + (25.302017977/4.161798144 - 4.51/20.48 * pi2) * eta \
           - 3.0913/183.5008 * eta*eta + 2.35925/17.69472 * eta*eta*eta

    # 3.5PN
    fta7 = (-1.88516689/4.33520640 - 9.7765/25.8048 * eta + 1.41769/12.90240 * eta*eta) * np.pi

    # 3PN log term
    ftal6 = 1.07/2.80


    full = theta3*ftaN * (fts1 \
             + fta2*theta2 \
             + fta3*theta3 \
             + fta4*theta4 \
             + fta5*theta5 \
             + (fta6 + ftal6*np.log(2.*theta))*theta6 \
             + fta7*theta7)

    return full * 2 * np.pi # 2pi to go from freq to angular freq

def TaylorT3_phase(t, tc, eta, M, phi0=0, m=2):
    """
    orbital phase, but returns by default the m=2 mode
    equation 3.10a in https://arxiv.org/pdf/0907.0700.pdf

    https://git.ligo.org/lscsoft/lalsuite/blob/master/lalsimulation/src/LALSimInspiralTaylorT3.c

    https://git.ligo.org/lscsoft/lalsuite/blob/master/lalsimulation/src/LALSimInspiralPNCoefficients.c

    t: time
    tc: coalescence time
    eta: symmetric mass ratio
    M: total mass (Msun)
    phi0: reference orbital phase, default 0
    m: m-mode default = 2
    """

    Msec = Msun_to_sec(M)
    Msec = M

    pi2 = np.pi*np.pi

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
    ftaN = -1. / eta
    # 0PN
    fts1 = 1.
    # 0.5PN = 0 in GR
    # 1PN
    fta2 = 3.715/8.064 + 5.5/9.6 * eta
    # 1.5PN
    fta3 = -3./4. * np.pi
    # 2PN
    fta4 = 9.275495/14.450688 + 2.84875/2.58048 * eta + 1.855/2.048 * eta*eta
    # 2.5PN
    fta5 = (3.8645/2.1504 - 6.5/25.6 * eta) * np.pi
    # 3PN
    fta6 = 83.1032450749357/5.7682522275840 - 5.3/4.0 * pi2 - 10.7/5.6 * GAMMA \
        + (-126.510089885/4.161798144 + 2.255/2.048 * pi2) * eta \
        + 1.54565/18.35008 * eta*eta - 1.179625/1.769472 * eta*eta*eta


    # 3.5PN
    fta7 = (1.88516689/1.73408256 + 4.88825/5.16096 * eta - 1.41769/5.16096 * eta*eta) * np.pi

    # 3PN log term
    ftal6 = -10.7/5.6

    # 2.5 PN log term
    ftal5 = np.log(theta)


    full = ftaN/theta5 * (fts1 \
             + fta2*theta2 \
             + fta3*theta3 \
             + fta4*theta4 \
             + (fta5*ftal5)*theta5 \
             + (fta6 + ftal6*np.log(2.*theta))*theta6 \
             + fta7*theta7)

    return m*(phi0 + full)

def Hhat22_x(x, eta):
    """
    https://arxiv.org/pdf/0802.1249.pdf - eq. 9.4a

    here we leave the expression to depend on the post-newtonian
    parameter 'x' so that you can choose how to calculate it.
    e.g., from PN like TaylorT3 or from the model which
    is TaylorT3 + corrections
    """

    xarr = np.zeros(6, dtype=np.complex128)

    C = 0.577216 # is the Euler constant

    xarr[0] = 1.
    xarr[1] = -107./42 + 55*eta/42
    xarr[2] = 2.*np.pi
    xarr[3] = -2173./1512 - 1069.*eta/216 + 2047.*eta**2/1512
    xarr[4] = (-107*np.pi/21 - 24.*1.j*eta + 34.*np.pi*eta/21) # there is an i... not sure what to do...

    x5a = 27027409./646800 - 856.*C/105 + 428*1.j*np.pi/105 + 2.*np.pi**2/3
    x5b = (-278185./33264 + 41*np.pi**2/96)*eta - 20261.*eta**2/2772 + 114635.*eta**3/99792

    x5log =  - 428.*np.log(16*x)/105

    xarr[5] = (x5a) + x5b # there is an i...  not sure what to do...

    pre = np.sqrt(16.*np.pi/5) * 2 * eta

    x1 = x
    x2 = x1*x1
    x3 = x2*x1
    x12 = np.sqrt(x)
    x32 = x1 * x12
    x52 = x2 * x12

    #pn = xarr[0] + x*xarr[1] + x**(3/2.)*xarr[2] + x**2*xarr[3] + x**(5/2.)*xarr[4] + x**3*(xarr[5] + x5log)
    pn = xarr[0] + x1*xarr[1] + x32*xarr[2] + x2*xarr[3] + x52*xarr[4] + x3*(xarr[5] + x5log)

    return pre * pn * x

def Hhat22_T3(t, t0, eta, M):
    """
    https://arxiv.org/pdf/0802.1249.pdf - eq. 9.4a
    Post-Newtonian expression for (l,m)=(2,2) time domain
    amplitude assuming TaylorT3 frequency evolution
    """

    GW22AngFreq = TaylorT3_Omega_new(t, t0, eta, M)
    OrgAngFreq = GW22AngFreq/2

    x = (M*OrgAngFreq)**(2./3)

    return Hhat22_x(x, eta)
