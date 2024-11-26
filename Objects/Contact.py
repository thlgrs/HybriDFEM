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

        self.force['n'] = 0
        self.force['s'] = 0
        self.disps['n'] = 0
        self.disps['s'] = 0

        self.commit()

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

    def update(self):
        self.force['n'] = self.stiff['kn'] * self.disps['n']
        self.force['s'] = self.stiff['ks'] * self.disps['s']

    def get_k_tan(self):
        return (self.stiff['kn'], self.stiff['ks'], 0)

    def get_k_init(self):
        return (self.stiff0['kn'], self.stiff0['ks'], 0)

    def to_ommit(self):
        return False


class NoTension(Contact):

    def __init__(self, kn, ks, cheating=False):

        super().__init__(kn, ks)
        self.cheating = cheating

    def to_ommit(self):

        if self.disps['n'] >= 1e-3:
            return True
        return False

    def update(self):

        if self.disps['n'] >= 1e-25:
            # No tension allowed, set stress and stiffness to zero
            self.force['n'] = 0
            self.stiff['kn'] = 0
            if not self.cheating:
                self.force['s'] = 0
                self.stiff['ks'] = 0

        else:

            # Elastic behavior for normal stress
            self.force['n'] = self.disps['n'] * self.stiff0['kn']
            self.stiff['kn'] = self.stiff0['kn']
            if not self.cheating:
                self.force['s'] = self.stiff0['ks'] * (self.disps['s'])
                self.stiff['ks'] = self.stiff0['ks']

        if self.cheating:
            self.force['s'] = self.stiff0['ks'] * (self.disps['s'])
            self.stiff['ks'] = self.stiff0['ks']
