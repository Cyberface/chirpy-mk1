from .utils import G_Newt, c_ls, MSUN_SI, MTSUN_SI, MRSUN_SI, PC_SI, GAMMA
from .ansatz import PSF_freq_ins, PSF_freq_int, PSF_freq_mrd
from .ansatz import PSF_amp_ins, PSF_amp_int, PSF_amp_mrd
from .ansatz import amp_ins_ansatz, amp_int_ansatz, amp_mrd_ansatz
from .ansatz import freq_ins_ansatz, freq_int_ansatz, freq_mrd_ansatz
from .ansatz import analytic_phase_ins_ansatz, analytic_phase_int_ansatz
from .ansatz import analytic_phase_mrd_ansatz

import numpy as np
import phenom

class Mk1(object):
    def __init__(self, times, q):
        """
        times to evaluate the model at
        q : mass-ratio

        attributes
        self.times
        self.amp
        self.freq (gw 22 mode ang freq)
        self.phase (gw 22 mode phase)
        self.h (complex)

        self.q
        self.eta
        self.fin_spin
        self.fring
        self.fdamp
        """

        self.times = times
        self.q = q
        self.eta = phenom.eta_from_q(self.q)

        self.fin_spin = phenom.remnant.FinalSpin0815(self.eta, 0, 0)
        self.fring = phenom.remnant.fring(self.eta, 0, 0, self.fin_spin)
        self.fdamp = phenom.remnant.fdamp(self.eta, 0, 0, self.fin_spin)


        # freq model
        # build dict for params. parameter space fit values and fixed values

        freq_inc_tc = PSF_freq_ins('tc',self.eta)
        freq_inc_b = PSF_freq_ins('b',self.eta)
        freq_inc_c = PSF_freq_ins('c',self.eta)

        self.params_freq_ins={}
        self.params_freq_ins.update({
            'tc':freq_inc_tc,
            'b':freq_inc_b,
            'c':freq_inc_c
        })

        self.params_freq_int={}
        self.params_freq_int.update({
            'tdamp':PSF_freq_int('tdamp',self.eta),
            'tp':PSF_freq_int('tp',self.eta),
            'lor_amp':PSF_freq_int('lor_amp',self.eta),
            'a0':PSF_freq_int('a0',self.eta),
            'a1':PSF_freq_int('a1',self.eta)
        })

        self.params_freq_mrd={}
        self.params_freq_mrd.update({
            't0':PSF_freq_mrd('t0', self.eta),
            'om_f' : self.fring*2.*np.pi,
            'b' : 1./self.fdamp/2./np.pi,
            'offset': 0.175,
            'kappa':0.44
        })

        # amp
        self.params_amp_ins={}
        self.params_amp_ins.update({
            'a0': PSF_amp_ins('a0', self.eta),
            'a1': PSF_amp_ins('a1', self.eta)
        })

        self.params_amp_ins.update({
            'tc':freq_inc_tc,
            'b':freq_inc_b,
            'c':freq_inc_c
        })

        self.params_amp_int={}
        self.params_amp_int.update({
            'tdamp' : 132.,
            'tp':PSF_amp_int('tp', self.eta),
            'lor_amp':PSF_amp_int('lor_amp', self.eta),
            'a0':PSF_amp_int('a0', self.eta),
            'a1':PSF_amp_int('a1', self.eta)
        })

        self.params_amp_mrd={}
        self.params_amp_mrd.update({
            'b' : 1./self.fdamp/2./np.pi,
            'A0':PSF_amp_mrd('A0', self.eta),
            'a1':PSF_amp_mrd('a1', self.eta),
            'a2':PSF_amp_mrd('a2', self.eta),
            'a3':PSF_amp_mrd('a3', self.eta),
            't0':PSF_amp_mrd('t0', self.eta),
            'kappa':PSF_amp_mrd('kappa', self.eta)
        })

        # full amp
        self.amp = self.amp_func(self.times, self.eta)

        # full freq
        self.freq = self.freq_func(self.times, self.eta)

        # full phase analytic
        self.phase = self.phase_func(self.times, self.eta)

        self.h = self.amp * np.exp(-1.j * self.phase)

    def amp_func(self, times, eta, t0_ins_int=-200, t0_int_mr=-40):

        mask_ins = times <= t0_ins_int
        mask_int = (times > t0_ins_int) & (times <= t0_int_mr)
        mask_mr = times > t0_int_mr

        amp_ins = amp_ins_ansatz(times[mask_ins], eta, self.params_amp_ins)
        amp_int = amp_int_ansatz(times[mask_int], self.params_amp_int)
        amp_mr = amp_mrd_ansatz(times[mask_mr], self.params_amp_mrd)

#         enforce some level on continuity
        a0 = amp_ins[-1] - amp_int[0]
        amp_int += a0

        amp_ins_int = np.concatenate((amp_ins, amp_int))

        a1 = amp_mr[0] - amp_ins_int[-1]
        amp_ins_int += a1

        imr_amp = np.concatenate((amp_ins_int, amp_mr))
#         imr_amp = np.concatenate((amp_ins, amp_int, amp_mr))

        return imr_amp

    def freq_func(self, times, eta, t0_ins_int = -500, t0_int_mr = -5):

        mask_ins = times <= t0_ins_int
        mask_int = (times > t0_ins_int) & (times <= t0_int_mr)
        mask_mr = times > t0_int_mr

        freq_ins = freq_ins_ansatz(times[mask_ins], eta, self.params_freq_ins)
        freq_int = freq_int_ansatz(times[mask_int], self.params_freq_int)
        freq_mr = freq_mrd_ansatz(times[mask_mr], self.params_freq_mrd)

        imr_freq = np.concatenate((freq_ins, freq_int, freq_mr))

        return imr_freq

    def phase_func(self, times, eta, t0_ins_int = -500, t0_int_mr = -5):

        # to get connection correct need to do this +dt
        # and then later not use all the int and the mrd phases
        dt = times[1] - times[0]

        mask_ins = times <= t0_ins_int+dt
        mask_int = (times > t0_ins_int) & (times <= t0_int_mr + dt)
        mask_mr = times > t0_int_mr

        phase_ins = analytic_phase_ins_ansatz(times[mask_ins], eta, **self.params_freq_ins)
        phase_int = analytic_phase_int_ansatz(times[mask_int], **self.params_freq_int)
        phase_mrd = analytic_phase_mrd_ansatz(times[mask_mr], **self.params_freq_mrd)

        # leave ins where it is and connect int and mrd by C(0)

        ins_int = phase_ins[-1] - phase_int[0]
        phase_int += ins_int

        ins_int_mrd = phase_int[-1] - phase_mrd[0]
        phase_mrd += ins_int_mrd

        analytic_imr_phase = np.concatenate((phase_ins[:-1], phase_int[:-1], phase_mrd))
        analytic_imr_phase -= analytic_imr_phase[0]

        return np.array(analytic_imr_phase, dtype=np.float64)
