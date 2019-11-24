from .utils import G_Newt, c_ls, MSUN_SI, MTSUN_SI, MRSUN_SI, PC_SI, GAMMA
from .ansatz import PSF_freq_ins, PSF_freq_int, PSF_freq_mrd
from .ansatz import PSF_amp_ins, PSF_amp_int, PSF_amp_mrd
from .ansatz import amp_ins_ansatz, amp_int_ansatz, amp_mrd_ansatz
from .ansatz import freq_ins_ansatz, freq_int_ansatz, freq_mrd_ansatz
from .ansatz import analytic_phase_ins_ansatz, analytic_phase_int_ansatz
from .ansatz import analytic_phase_mrd_ansatz

from .ansatz import analytic_phase_mrd_ansatz_approx

import numpy as np
from scipy import optimize
import phenom

class Mk1(object):
    def __init__(self, times=None, q=1, f_start=None, srate=None, t_end=60):
        """
        times: times to evaluate the model at (in units of total mass)
            if None then will use f_start, srate and t_end
        q : mass-ratio

        f_start: angular gw 22 geometric frequency, desired start frequency
        srate = sample rate in geometric units
        t_end = end time for times array, only if times is not explicitly given
             - geometric units

        attributes
        self.times
        self.amp
        self.phase (gw 22 mode phase)
        self.h (complex)

        self.q
        self.eta
        self.fin_spin
        self.fring (angular geometric)
        self.fdamp (angluar geometric)

        the frequency is not computed automatically but can be computed using
        self.freq_func(times)
        """

        # 1. set up coefficients of the model
        # 2. find t_start corresponding to f_start (current 22 mode)
        # 3. build 'times' array
        # 4. compute model on 'times' array

        self.q = q
        self.times = times
        self.f_start = f_start
        self.srate = srate
        self.t_end = t_end

        self.eta = phenom.eta_from_q(self.q)

        self.fin_spin = phenom.remnant.FinalSpin0815(self.eta, 0, 0)
        self.fring = phenom.remnant.fring(self.eta, 0, 0, self.fin_spin)
        # convert to angular geometric
        self.fring *= 2*np.pi
        self.fdamp = phenom.remnant.fdamp(self.eta, 0, 0, self.fin_spin)
        # convert to angular geometric
        self.fdamp *= 2*np.pi

        # sanity check inputs: times not explicitly given
        if self.times is None:
            if self.srate is None:
                raise ValueError("times is None but no srate given")
            if self.f_start is None:
                raise ValueError("times is None but no srate given")
            if self.f_start > self.fring:
                raise Exception(
                        "f_start ({}) > ringdown frequency ({})".format(
                            self.f_start, self.fring))

        # sanity check inputs: times are explicitly given
        if self.times is not None:
            if self.f_start is not None:
                raise ValueError("f_start given but 'times' is not None.")
            if self.srate is not None:
                raise ValueError("srate given but 'times' is not None.")

        # freq model
        # build dict for params. parameter space fit values and fixed values
        self._setup_model_coefficients()

        # determine the 'times' array to generate the model on.
        # if a 'times' array is given then use that
        # if not then generate the waveform from the given 'f_start'
        # note: this requires a root finding operation.
        self.times = self._setup_times_array(times=self.times, f_start=self.f_start, srate=self.srate, t_end=self.t_end)

        # compute amplitude and phase

        # amplitude
        self.amp = self.amp_func(self.times)

        # phase
        self.phase = self.phase_func(self.times)

        # compute complex strain
        self.h = self.amp * np.exp(-1.j * self.phase)

    def _setup_model_coefficients(self):
        """
        sets up a bunch of class attributes that
        contain dictions for all the model coefficients.
        """
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
            'om_f' : self.fring,
            'b' : 1./self.fdamp,
            'offset': 0.175,
            'kappa':0.44
        })

        # amp
        self.params_amp_ins={}
        self.params_amp_ins.update({
            'a0': PSF_amp_ins('a0', self.eta),
            'a1': PSF_amp_ins('a1', self.eta)
        })

        # amplitude ins model depends on freq ins model
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
            'b' : 1./self.fdamp,
            'A0':PSF_amp_mrd('A0', self.eta),
            'a1':PSF_amp_mrd('a1', self.eta),
            'a2':PSF_amp_mrd('a2', self.eta),
            'a3':PSF_amp_mrd('a3', self.eta),
            't0':PSF_amp_mrd('t0', self.eta),
            'kappa':PSF_amp_mrd('kappa', self.eta)
        })

        # return void
        return

    def _setup_times_array(self, times, f_start, srate, t_end=60):
        """
        function to determine the times array to generate the model on.
        If an explicit times array is not given then use
        a root finding algorithm to find the time (t_start) corresponding
        to f_start and then generate a times array with a sample rate of
        1./srate from t_start until t_end.
        """
        if times is None:
            # compute times from f_start

            # to compute t_start, the time at f_start
            # we by default use the inspiral freq model of mk1
            # because it is much faster.
            # However, if the desired start frequency is too high,
            # typically set by f_start = 0.1 (in angular geometric units)
            # then we use the full imr freq model of mk1.
            if self.f_start > 0.1:
                f_of_t_func = self._f_of_t_to_min_imr
            else:
                f_of_t_func = self._f_of_t_to_min_ins

            try:
                self.t_start_from_f_start = self._t_of_f(f_start, f_of_t_func)
            except RuntimeError:
                print("time of frequency root finding failed.")
                print("you requested f_start = {}".format(f_start))
                print("the ringdown frequency is = {}".format(self.fring))
                print("possible problem:")
                print("if you ask for an f_start too close to this then")
                print("the root finding can fail. Try a smaller f_start")
                raise

            out_times = np.arange(self.t_start_from_f_start, t_end, srate)
        else:
            out_times = times
        return out_times

    def _f_of_t_to_min_ins(self, t, f_start):
        """
        returns the frequency at time 't'  minus 'f_start'
        using only the inspiral model of mk1.
        Used by scipy.optimize.
        """
        f = freq_ins_ansatz(t, self.eta, self.params_freq_ins)
        return f - f_start

    def _f_of_t_to_min_imr(self, t, f_start):
        """
        returns the frequency at time 't' minus 'f_start'
        using only the complete imr freq
        model of mk1.
        Used by scipy.optimize.
        """
        f = self.freq_func(np.array([t]), self.eta)[0]
        return f - f_start

    def _t_of_f(self, f_start, func, t_guess=-100):
        """
        uses the Newton method to root find.
        Finds the time at which the frequency 'f_start' occurs.
        Used to setup initial conditions.
        Note:
            Will only work work quasi-circular, non-precessing
            and monotonic frequency functions.
        """
        out = optimize.newton(func, t_guess, args=(f_start,))
        return out

    def amp_func(self, times, t0_ins_int=-200, t0_int_mrd=-40):
        """
        returns the mk1 amplitude model on the time grid 'times'
        """

        mask_ins = times <= t0_ins_int
        mask_int = (times > t0_ins_int) & (times <= t0_int_mrd)
        mask_mrd = times > t0_int_mrd

        try:
            amp_ins = amp_ins_ansatz(times[mask_ins], self.eta, self.params_amp_ins)
        except:
            amp_ins = np.array([])
        try:
            amp_int = amp_int_ansatz(times[mask_int], self.params_amp_int)
        except:
            amp_int = np.array([])
        try:
            amp_mr = amp_mrd_ansatz(times[mask_mrd], self.params_amp_mrd)
        except:
            amp_mr = np.array([])

        # enforce some level on continuity
        # shift int to match ins at t0_ins_int
        # then shift ins+int to match with mrd at t0_int_mrd
        amp_ins_t0 = amp_ins_ansatz(t0_ins_int, self.eta, self.params_amp_ins)
        amp_int_t0 = amp_int_ansatz(t0_ins_int, self.params_amp_int)

        a0 = amp_ins_t0 - amp_int_t0
        amp_int += a0
        amp_ins_int = np.concatenate((amp_ins, amp_int))

        amp_int_t1 = amp_int_ansatz(t0_int_mrd, self.params_amp_int)
        amp_mrd_t1 = amp_mrd_ansatz(t0_int_mrd, self.params_amp_mrd)

        a1 = amp_mrd_t1 - (amp_int_t1 + a0)
        amp_ins_int += a1

        imr_amp = np.concatenate((amp_ins_int, amp_mr))

        return imr_amp

    def freq_func(self, times, t0_ins_int = -500, t0_int_mrd = -5):
        """
        returns the mk1 frequency model on the time grid 'times'
        """

        mask_ins = times <= t0_ins_int
        mask_int = (times > t0_ins_int) & (times <= t0_int_mrd)
        mask_mrd = times > t0_int_mrd

        try:
            freq_ins = freq_ins_ansatz(times[mask_ins], self.eta, self.params_freq_ins)
        except:
            freq_ins = np.array([])
        try:
            freq_int = freq_int_ansatz(times[mask_int], self.params_freq_int)
        except:
            freq_int = np.array([])
        try:
            freq_mrd = freq_mrd_ansatz(times[mask_mrd], self.params_freq_mrd)
        except:
            freq_mrd = np.array([])

        imr_freq = np.concatenate((freq_ins, freq_int, freq_mrd))

        return imr_freq

    def phase_func(self, times, t0_ins_int = -500, t0_int_mrd = -5):
        """
        returns the mk1 phase model on the time grid 'times'
        """

        mask_ins = times <= t0_ins_int
        mask_int = (times > t0_ins_int) & (times <= t0_int_mrd)
        mask_mr = times > t0_int_mrd

        try:
            phase_ins = analytic_phase_ins_ansatz(times[mask_ins], self.eta, **self.params_freq_ins)
        except:
            phase_ins = np.array([])
        try:
            phase_int = analytic_phase_int_ansatz(times[mask_int], **self.params_freq_int)
        except:
            phase_int = np.array([])
        try:
            # phase_mrd = analytic_phase_mrd_ansatz(times[mask_mr], **self.params_freq_mrd)
            phase_mrd = analytic_phase_mrd_ansatz_approx(times[mask_mr], **self.params_freq_mrd)
        except:
            phase_mrd = np.array([])

        # leave ins where it is and connect int and mrd by C(0)

        phase_ins_t0 = analytic_phase_ins_ansatz(t0_ins_int, self.eta, **self.params_freq_ins)
        phase_int_t0 = analytic_phase_int_ansatz(t0_ins_int, **self.params_freq_int)

        phase_int_t1 = analytic_phase_int_ansatz(t0_int_mrd, **self.params_freq_int)
        phase_mrd_t1 = analytic_phase_mrd_ansatz(t0_int_mrd, **self.params_freq_mrd)

        ins_int = phase_ins_t0 - phase_int_t0
        phase_int += ins_int

        ins_int_mrd = (phase_int_t1+ins_int) - phase_mrd_t1
        phase_mrd += ins_int_mrd

        analytic_imr_phase = np.concatenate((phase_ins, phase_int, phase_mrd))

        # phase aligned at t=0
        # should change this to be phi_ref option
        analytic_imr_phase -= analytic_phase_mrd_ansatz(0, **self.params_freq_mrd)

        return np.array(analytic_imr_phase, dtype=np.float64)
