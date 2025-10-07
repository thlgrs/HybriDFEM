# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:24:41 2024

@author: ibouckaert
"""

import numpy as np
from warnings import warn
import warnings 
import os
import matplotlib.pyplot as plt
from copy import deepcopy


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    file_short_name = filename.replace(os.path.dirname(filename), "")
    file_short_name = file_short_name.replace("\\", "")
    return f"Warning! In file {file_short_name}, line {lineno}: {message}\n"

warnings.formatwarning = custom_warning_format


class Surface: 
    
    def __init__(self, k_n, k_s):
        self.stiff = {}
        self.stiff0 = {}
        self.stress = {}
        self.disps = {}

        self.tag = 'LINEL'
        self.stiff['kn'] = k_n
        self.stiff0['kn'] = k_n

        self.stiff['ks'] = k_s
        self.stiff0['ks'] = k_s

        self.stiff['ksn'] = 0
        self.stiff0['ksn'] = 0
        self.stiff['kns'] = 0
        self.stiff0['kns'] = 0
    
        self.stress['s'] = 0
        self.stress['t'] = 0
        self.disps['n'] = 0 
        self.disps['s'] = 0

        self.tol_disp = 1e-30

    def copy(self): 
        
        return deepcopy(self)

    def commit(self):
        self.stress_conv = self.stress.copy()
        self.disps_conv = self.disps.copy()
        self.stiff_conv = self.stiff.copy()

    def revert_commit(self): 
        
        self.stress = self.stress_conv.copy()
        self.disps = self.disps_conv.copy()
        self.stiff = self.stiff_conv.copy()

    def get_forces(self): 
        
        return np.array([self.stress['s'], self.stress['t']])

    def set_elongs(self, d_n, d_s):
        self.disps['n'] = d_n
        self.disps['s'] = d_s

    def update(self, dL):
        self.disps['n'] += dL[0]
        self.disps['s'] += dL[1]

        self.stress['s'] = self.stiff['kn'] * self.disps['n']
        self.stress['t'] = self.stiff['ks'] * self.disps['s']

    def get_k_tan(self):
        return (self.stiff['kn'], self.stiff['ks'], self.stiff['kns'], self.stiff['ksn'])

    def get_k_init(self):
        return (self.stiff0['kn'], self.stiff0['ks'], 0, 0)

    def to_ommit(self):
        return False


class NoTension_EP(Surface):

    def __init__(self, kn, ks):

        super().__init__(kn, ks)
        self.tag = 'NTEP'
        # Initialize accumulated plastic shear strain and state variables
        self.commit()

    def to_ommit(self):

        return False

    def update(self, dL):

        self.disps['n'] += dL[0]
        self.disps['s'] += dL[1]

        if self.disps['n'] > self.tol_disp:
            #     # No tension allowed, set stress and stiffness to zero
            # print('Ommitting')
            self.stress['s'] = 0.
            self.stiff['kn'] = 0.
            # self.stress['t'] = 0.
            # self.stiff['ks'] = 0.
        else:
            # Elastic behavior for normal stress
            self.stress['s'] = self.disps['n'] * self.stiff0['kn']
            self.stiff['kn'] = self.stiff0['kn']

        self.stress['t'] = (self.disps['s']) * self.stiff0['ks']
        self.stiff['ks'] = self.stiff0['ks']


class NoTension_CD(Surface):

    def __init__(self, kn, ks):
        super().__init__(kn, ks)
        self.disps['s_p'] = 0
        self.disps['s_p_temp'] = 0
        # Initialize accumulated plastic shear strain and state variables
        self.tag = 'NTCD'
        self.commit()

    def to_ommit(self):
        if self.disps['n'] > self.tol_disp:
            # print('Omitted')

            self.disps['s_p_temp'] = self.disps['s']
            # print('s_p', self.disps['s_p_temp'])
            return True

        return False

    def commit(self):
        # if self.to_ommit(): 
        self.disps['s_p'] = self.disps['s_p_temp']
        super().commit()

    def update(self, dL):
        self.disps['n'] += dL[0]
        self.disps['s'] += dL[1]

        self.stress['s'] = self.disps['n'] * self.stiff0['kn']
        self.stiff['kn'] = self.stiff0['kn']

        self.stress['t'] = (self.disps['s'] - self.disps['s_p']) * self.stiff0['ks']
        self.stiff['ks'] = self.stiff0['ks']


class Coulomb(Surface):

    def __init__(self, kn, ks, mu, c=0, psi=0, ft=0):

        super().__init__(kn, ks)
        self.mu = mu
        self.tag = 'COUL'

        self.stress['s_temp'] = 0
        self.stress['t_temp'] = 0
        self.disps['s_temp'] = 0
        self.disps['n_temp'] = 0
        self.disps['d_s_p'] = 0
        self.disps['d_n_p'] = 0
        self.c = c
        self.psi = psi
        self.ft = ft
        self.disps['n_t'] = self.ft / self.stiff0['kn']

        self.activated = None

        self.commit()

    def to_ommit(self):

        if (self.disps['n_temp'] - self.disps['n_t'] + self.disps['d_n_p']) > self.tol_disp:
            self.disps['s_p_temp'] = self.disps['s']
            # print('Ommiting')
            # print(self.disps['n_temp'])
            return True

        return False

    def get_forces(self):

        return np.array([self.stress['s_temp'], self.stress['t_temp']])

    def commit(self):

        self.stress['s'] = self.stress['s_temp']
        self.stress['t'] = self.stress['t_temp']
        self.disps['s'] = self.disps['s_temp']
        self.disps['n'] = self.disps['n_temp']

        self.activated = None

        super().commit()

    def update(self, dL):

        self.disps['n_temp'] += dL[0]
        self.disps['s_temp'] += dL[1]

        D = np.array([[self.stiff0['kn'], 0],
                      [0, self.stiff0['ks']]])

        d_sigma_tr = D @ dL
        sigma_trial = np.array([self.stress['s_temp'], self.stress['t_temp']]) + d_sigma_tr

        if self.activated is None:
            F_tr1 = sigma_trial[1] + self.mu * sigma_trial[0] - self.c
            F_tr2 = - sigma_trial[1] + self.mu * sigma_trial[0] - self.c

            if F_tr1 > 0 and F_tr2 > 0:
                if F_tr2 > F_tr1:
                    self.activated = 'F2'
                else:
                    self.activated = 'F1'

        elif self.activated == 'F1':
            F_tr1 = 1
            F_tr2 = 0
        elif self.activated == 'F2':
            F_tr1 = 0
            F_tr2 = 1

        if F_tr1 <= 0 and F_tr2 <= 0:  # Elastic step
            # print('Elastic')
            self.stress['s_temp'] = sigma_trial[0]
            self.stress['t_temp'] = sigma_trial[1]
            self.stiff['kn'] = self.stiff0['kn']
            self.stiff['ks'] = self.stiff0['ks']
            self.stiff['kns'] = 0.
            self.stiff['ksn'] = 0.

        elif F_tr2 > 0:  # Projection on F2
            # print('F2')
            self.activated = 'F2'

            c = self.c
            p = self.psi
            m = self.mu
            kn = self.stiff0['kn']
            ks = self.stiff0['ks']
            s = self.stress['s_temp']
            t = self.stress['t_temp']

            denom = kn * m * p + ks
            d_n = (c * p * kn + dL[0] * kn * ks + dL[1] * kn * ks * p - kn * m * p * s + kn * p * t) / denom
            d_s = (-c * ks + dL[0] * kn * ks * m + dL[1] * kn * ks * m * p + ks * m * s - ks * t) / denom
            d_l = (c - dL[0] * kn * m + dL[1] * ks - m * s + t) / denom
            # d_l_n = 

            Kn = kn - (m * p * kn ** 2) / denom
            Ks = ks - (ks ** 2) / denom
            Ksn = kn * ks * m / denom
            Kns = kn * ks * p / denom
            self.stiff['kn'] = Kn
            self.stiff['ksn'] = Ksn
            self.stiff['ks'] = Ks
            self.stiff['kns'] = Kns

            self.stress['s_temp'] += d_n
            self.stress['t_temp'] += d_s
            self.disps['d_s_p'] += d_l
            self.disps['d_n_p'] -= d_l * p

        elif F_tr1 > 0:
            # print('F1')
            self.activated = 'F1'

            c = self.c
            p = self.psi
            m = self.mu
            kn = self.stiff0['kn']
            ks = self.stiff0['ks']
            s = self.stress['s_temp']
            t = self.stress['t_temp']

            denom = kn * m * p + ks
            d_n = (c * p * kn + dL[0] * kn * ks - dL[1] * kn * ks * p - kn * m * p * s - kn * p * t) / denom
            d_s = (c * ks - dL[0] * kn * ks * m + dL[1] * kn * ks * m * p - ks * m * s - ks * t) / denom
            d_l = (-c + dL[0] * kn * m + dL[1] * ks + m * s + t) / denom

            Kn = kn - (m * p * kn ** 2) / denom
            Ks = ks - (ks ** 2) / denom
            Ksn = - kn * ks * m / denom
            Kns = - kn * ks * p / denom
            self.stiff['kn'] = Kn
            self.stiff['ksn'] = Ksn
            self.stiff['ks'] = Ks
            self.stiff['kns'] = Kns

            self.stress['s_temp'] += d_n
            self.stress['t_temp'] += d_s

            self.disps['d_s_p'] += d_l
            self.disps['d_n_p'] += d_l * p


class bond_slip_tc(Surface):

    def __init__(self, tb0, tb1):

        kn = 1e17
        # ks = 33e9 / (2 * .1)
        ks = 1e12
        # ks = 5e9

        super().__init__(kn, ks)

        self.stress['tb0'] = tb0
        self.stress['tb1'] = tb1

        self.tag = 'BSTC'
        self.disps['s_p'] = 0

        self.reduced = False
        # Initialize accumulated plastic shear strain and state variables
        self.commit()

    def to_ommit(self):

        return False

    def update(self, dL):

        self.disps['n'] += dL[0]
        self.disps['s'] += dL[1]

        self.stress['s'] = self.disps['n'] * self.stiff0['kn']

        s_tr = self.stiff0['ks'] * (self.disps['s'] - self.disps['s_p'])
        f_tr = abs(s_tr) - (self.stress['tb0'])

        # Elastic step
        if f_tr <= 0:
            self.stress['t'] = s_tr
            self.stiff['ks'] = deepcopy(self.stiff0['ks'])

        # Plastic step
        else:
            d_g = f_tr / (self.stiff0['ks'])

            self.stress['t'] = (self.stress['tb0']) * np.sign(self.disps['s'])
            self.disps['s_p'] += d_g * np.sign(s_tr)

            self.stiff['ks'] = 0.
