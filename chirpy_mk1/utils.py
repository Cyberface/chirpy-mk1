
# Newton gravity constant
# lal.G_SI: 6.67384e-11
G_Newt = 6.67384e-11

# light speed
# lal.C_SI: 299792458.0
c_ls = 299792458.0

# lal.MSUN_SI: 1.9885469549614615e+30
MSUN_SI = 1.9885469549614615e+30

MTSUN_SI = 4.925491025543576e-06

# lal.MRSUN_SI: 1476.6250614046494
MRSUN_SI = 1476.6250614046494

# lal.PC_SI: 3.085677581491367e+16
PC_SI = 3.085677581491367e+16

# lal.GAMMA: 0.5772156649015329
GAMMA = 0.5772156649015329


def Msun_to_sec(M):
    """
    convert mass (in units of solar masses)
    into seconds
    """
#     return M *lal.MSUN_SI* G_Newt / c_ls**3.
    return M * MTSUN_SI
