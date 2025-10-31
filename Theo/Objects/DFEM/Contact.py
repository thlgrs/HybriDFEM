# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:24:41 2024

@author: ibouckaert
"""

import os
import warnings
from copy import deepcopy

import numpy as np


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    file_short_name = filename.replace(os.path.dirname(filename), "")
    file_short_name = file_short_name.replace("\\", "")
    return f"Warning! In file {file_short_name}, line {lineno}: {message}\n"


warnings.formatwarning = custom_warning_format


class Contact:

    def __init__(self, k_n, k_s):
        self.stiff = {}
        self.stiff0 = {}
        self.force = {}
        self.disps = {}

        self.stiff['kn'] = k_n
        self.stiff0['kn'] = k_n

        self.stiff['ks'] = k_s
        self.stiff0['ks'] = k_s

        self.stiff['kns'] = 0
        self.stiff0['kns'] = 0

        self.stiff['ksn'] = 0
        self.stiff0['ksn'] = 0
        self.tag = 'LINEL'
        self.force['n'] = 0
        self.force['s'] = 0
        self.disps['n'] = 0
        self.disps['s'] = 0

        self.tol_disp = 1e-15

    def copy(self):
        return deepcopy(self)

    def commit(self):
        self.force_conv = self.force.copy()
        self.disps_conv = self.disps.copy()
        self.stiff_conv = self.stiff.copy()

    def revert_commit(self):
        self.force = self.force_conv.copy()
        self.disps = self.disps_conv.copy()
        self.stiff = self.stiff_conv.copy()

    def get_forces(self):
        return np.array([self.force['n'], self.force['s']])

    def set_elongs(self, d_n, d_s):
        self.disps['n'] = d_n
        self.disps['s'] = d_s

    def get_elongs(self):
        return self.disps['n'], self.disps['s']

    def update(self, dL):
        self.disps['n'] += dL[0]
        self.disps['s'] += dL[1]

        self.force['n'] = self.stiff['kn'] * self.disps['n']
        self.force['s'] = self.stiff['ks'] * self.disps['s']

    def get_k_tan(self):
        return (self.stiff['kn'], self.stiff['ks'], self.stiff['kns'], self.stiff['ksn'])

    def get_k_init(self):
        return (self.stiff0['kn'], self.stiff0['ks'], 0, 0)

    def to_ommit(self):
        return False


class Bilinear(Contact):

    def __init__(self, k_n, k_s, fy, a):

        super().__init__(k_n, k_s)

        self.force['fy'] = fy
        self.disps['y'] = fy / self.stiff0['kn']
        self.a = a
        self.tag = 'BILIN'
        self.commit()

    def update(self, dL):

        self.disps['n'] += dL[0]
        self.disps['s'] += dL[1]

        if abs(self.disps['n']) < self.disps['y']:
            self.force['n'] = self.stiff0['kn'] * self.disps['n']
            self.stiff['kn'] = deepcopy(self.stiff0['kn'])

        else:
            self.force['n'] = self.force['fy'] * np.sign(self.disps['n']) + self.a * self.stiff0['ks'] * (
                    self.disps['n'] - self.disps['y'] * np.sign(self.disps['n']))
            self.stiff['kn'] = self.a * self.stiff0['kn']

        self.force['s'] = self.stiff0['ks'] * self.disps['s']
        self.stiff['ks'] = deepcopy(self.stiff0['ks'])


class NoTension_EP(Contact):

    # No-tension contact with equivalent plastic damage

    def __init__(self, kn, ks):

        super().__init__(kn, ks)
        self.name = 'Elastic no-tension contact law'
        self.disps['s_p'] = 0
        self.tag = 'NTEP'
        # self.c = 10
        # Commit the initial state
        self.commit()

    def update(self, dL):
        """
        Update material properties based on current strains.
        """
        self.disps['n'] += dL[0]
        self.disps['s'] += dL[1]

        if self.disps['n'] > self.tol_disp:

            # print('Tension')
            self.force['n'] = 0.
            self.stiff['kn'] = 0.


        else:
            self.force['n'] = self.disps['n'] * self.stiff0['kn']
            self.stiff['kn'] = deepcopy(self.stiff0['kn'])

        self.force['s'] = self.disps['s'] * self.stiff0['ks']
        self.stiff['ks'] = deepcopy(self.stiff0['ks'])


class NoTension_CD(Contact):

    # No-tension contact with contact deletion in tension

    def __init__(self, kn, ks):
        super().__init__(kn, ks)
        self.name = 'Elastic no-tension contact law'
        self.disps['s_p'] = 0
        self.disps['s_p_temp'] = 0
        # self.c = 10
        self.tag = 'NTCD'
        # Commit the initial state
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
        """
        Update material properties based on current strains.
        """

        self.disps['n'] += dL[0]
        self.disps['s'] += dL[1]
        # print('Disp', self.disps['n'], self.disps['s'])

        self.force['n'] = self.disps['n'] * self.stiff0['kn']
        self.stiff['kn'] = self.stiff0['kn']

        # print('s_p', self.disps['s_p'])
        self.force['s'] = (self.disps['s'] - self.disps['s_p']) * self.stiff0['ks']
        self.stiff['ks'] = self.stiff0['ks']

        # print('Force', self.force['n'], self.force['s'])


class Coulomb(Contact):

    def __init__(self, kn, ks, mu, c=0, psi=0, ft=0):

        super().__init__(kn, ks)
        self.mu = mu
        self.tag = 'COUL'

        self.force['n_temp'] = 0
        self.force['s_temp'] = 0
        self.disps['s_temp'] = 0
        self.disps['n_temp'] = 0
        self.disps['d_s_p'] = 0
        self.disps['d_n_p'] = 0
        self.c = c
        self.psi = psi
        self.ft = ft
        self.disps['n_t'] = self.ft / self.stiff0['kn']
        self.tag = 'COUL'
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

        return np.array([self.force['n_temp'], self.force['s_temp']])

    def commit(self):

        # if self.activated == 'F1' or self.activated == 'F2': 
        #     d_n = self.force['n_temp'] - self.force['n']
        #     d_s = self.force['s_temp'] - self.force['s']
        #     dL_n = self.disps['n_temp'] - self.disps['n']
        #     dL_s = self.disps['s_temp'] - self.disps['s']

        #     self.stiff['kn'] = d_n / dL_n
        #     self.stiff['ks'] = d_s / dL_s
        #     self.stiff['kns'] = d_n / dL_s
        #     self.stiff['ksn'] = d_s / dL_n

        self.force['n'] = self.force['n_temp']
        self.force['s'] = self.force['s_temp']
        self.disps['s'] = self.disps['s_temp']
        self.disps['n'] = self.disps['n_temp']

        self.activated = None

        super().commit

    def update(self, dL):

        self.disps['n_temp'] += dL[0]
        self.disps['s_temp'] += dL[1]

        D = np.array([[self.stiff0['kn'], 0],
                      [0, self.stiff0['ks']]])

        # print('delta_s', D@dL)
        # print('forces', self.force['n'], self.force['s'])
        d_sigma_tr = D @ dL
        sigma_trial = np.array([self.force['n_temp'], self.force['s_temp']]) + d_sigma_tr

        # print(sigma_trial)
        # print('sig_trial', sigma_trial)
        if self.activated is None:
            F_tr1 = sigma_trial[1] + self.mu * sigma_trial[0] - self.c
            F_tr2 = - sigma_trial[1] + self.mu * sigma_trial[0] - self.c
            # F_tr3 = sigma_trial[0]

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
            self.force['n_temp'] = sigma_trial[0]
            self.force['s_temp'] = sigma_trial[1]
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
            s = self.force['n_temp']
            t = self.force['s_temp']

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

            # Kn = kn - (m*p*kn**2) / denom
            # Ks = ks - (ks**2) / denom
            # Ksn = - kn*ks*m / denom
            # Kns = - kn*ks*p / denom
            # if abs(dL[0]) > self.tol_disp: 
            #     Kn = d_n/dL[0]
            #     Ksn = d_s/dL[0]
            #     self.stiff['kn'] = Kn
            #     self.stiff['ksn'] = Ksn
            # if abs(dL[1]) > self.tol_disp: 
            #     Ks = d_s/dL[1]
            #     Kns = d_n/dL[1]
            #     self.stiff['ks'] = Ks
            #     self.stiff['kns'] = Kns

            self.force['n_temp'] += d_n
            self.force['s_temp'] += d_s
            self.disps['d_s_p'] += d_l
            self.disps['d_n_p'] += d_l * p

        elif F_tr1 > 0:
            # print('F1')
            self.activated = 'F1'

            c = self.c
            p = self.psi
            m = self.mu
            kn = self.stiff0['kn']
            ks = self.stiff0['ks']
            s = self.force['n_temp']
            t = self.force['s_temp']

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
            # Secant stiffness ? 
            # if abs(dL[0]) > self.tol_disp: 
            #     Kn = d_n/dL[0]
            #     Ksn = d_s/dL[0]
            #     self.stiff['kn'] = Kn
            #     self.stiff['ksn'] = Ksn
            # if abs(dL[1]) > self.tol_disp: 
            #     Ks = d_s/dL[1]
            #     Kns = d_n/dL[1]
            #     self.stiff['ks'] = Ks
            #     self.stiff['kns'] = Kns

            self.force['n_temp'] += d_n
            self.force['s_temp'] += d_s

            self.disps['d_s_p'] += d_l
            self.disps['d_n_p'] -= d_l * p
