import numpy as np


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

def td_amp_scale(mtot, distance):
    """
    mtot in solar masses
    distance in Mpc
    M*G/c^2 * M_sun / dist
    """
    return mtot * MRSUN_SI / (distance * 1e6*PC_SI)

def match(h1, h2, times):

    dt = times[1] - times[0]
    n = len(times)
    df = 1.0/(n*dt)
    norm = 4. * df

    h1_fft = np.fft.fft(h1)
    h2_fft = np.fft.fft(h2)

    h1h1_sq = np.vdot(h1_fft, h1_fft) * norm
    h2h2_sq = np.vdot(h2_fft, h2_fft) * norm

    h1h1 = dt * np.sqrt(h1h1_sq)
    h2h2 = dt * np.sqrt(h2h2_sq)

    ifft = np.fft.ifft(np.conj(h1_fft) * h2_fft)

    ts = ifft / h1h1 / h2h2 * 4 * dt
    m = np.max(np.abs(ts))

    return m 

def matched_filter(h1, h2, times):
    """
    not tested
    """

    dt = times[1] - times[0]
    n = len(times)
    df = 1.0/(n*dt)
    norm = 4. * df

    h1_fft = np.fft.fft(h1)
    h2_fft = np.fft.fft(h2)

    h1h1_sq = np.vdot(h1_fft, h1_fft) * norm
    h2h2_sq = np.vdot(h2_fft, h2_fft) * norm

    h1h1 = dt * np.sqrt(h1h1_sq)

    ifft = np.fft.ifft(np.conj(h1_fft) * h2_fft)

    snr = ifft / h1h1 * 4 * dt

    return snr 

def coalign(h1, h2, times, t1=None, t2=None):
    """
    h1, h2 = complex
    uses an ifft to find time and phase shift
    and returns times+tshift, h1 * rotation
    """
    if t1 is None:
        t1 = times[0]
    if t2 is None:
        t2 = times[-1]
    mask = (times > t1) & (times < t2)

    ma_ts = matched_filter(h1[mask], h2[mask], times[mask])
    max_loc = np.argmax(np.abs(ma_ts))
    rotation = ma_ts[max_loc] / np.abs(ma_ts[max_loc])
    dt = times[1] - times[0]
    time_shift =  max_loc * dt

    duration = times[-1] - times[0]
    # not sure about this
    # it doesn't seem to work properly
    # sometimes and I think it's because
    # when I do the ifft I don't take into
    # account the cyclic nature
    if np.abs(time_shift) >= duration/2:
        time_shift = -duration + time_shift + dt

    h1_shifted = np.fft.ifft(np.fft.fft(h1)*rotation)

    new_times = times + time_shift
    return new_times, h1_shifted

def planck_taper(times, t1, t2):
    """times: array of times
    t1. for t<=t1 then return 0
    t2. for t>=t2 then return 1
    else return 1./(np.exp((t2-t1)/(t-t1)+(t2-t1)/(t-t2))+1)"""
    tout = []
    for t in times:
        if t<=t1:
            tout.append(0.)
        elif t>=t2:
            tout.append(1.)
        else:
            tout.append(1./(np.exp((t2-t1)/(t-t1)+(t2-t1)/(t-t2))+1))
    return np.array(tout)

def mass1_from_mtotal_eta(mtotal, eta):
    """Returns the primary mass from the total mass and symmetric mass
    ratio.
    """
    return 0.5 * mtotal * (1.0 + (1.0 - 4.0 * eta)**0.5)


def mass2_from_mtotal_eta(mtotal, eta):
    """Returns the secondary mass from the total mass and symmetric mass
    ratio.
    """
    return 0.5 * mtotal * (1.0 - (1.0 - 4.0 * eta)**0.5)
