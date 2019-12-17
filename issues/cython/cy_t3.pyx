#import chirpy_mk1

from chirpy_mk1.utils import G_Newt, c_ls, MSUN_SI, MTSUN_SI, MRSUN_SI, PC_SI, GAMMA, Msun_to_sec

import numpy as np

def cython_TaylorT3_Omega_new(t, tc, eta, M):

#     Msec = Msun_to_sec(M)
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
