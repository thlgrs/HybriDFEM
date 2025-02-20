# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:24:41 2024

@author: ibouckaert
"""

import os
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    file_short_name = filename.replace(os.path.dirname(filename), "")
    file_short_name = file_short_name.replace("\\", "")
    return f"Warning! In file {file_short_name}, line {lineno}: {message}\n"


warnings.formatwarning = custom_warning_format


class Material:

    def __init__(self, E, nu, corr_fact=1, shear_def=True):

        self.stiff = {}
        self.stiff0 = {}
        self.stress = {}
        self.strain = {}
        self.state = {}

        self.stiff['E'] = E
        self.stiff0['E'] = E

        self.nu = nu
        self.chi = corr_fact
        self.shear_def = shear_def

        if self.shear_def:
            self.stiff['G'] = (1 / self.chi) * E / (2 * (1 + nu))
            self.stiff0['G'] = (1 / self.chi) * E / (2 * (1 + nu))
        else:
            self.stiff['G'] = 1e6 * (1 / self.chi) * E / (2 * (1 + nu))
            self.stiff0['G'] = 1e6 * (1 / self.chi) * E / (2 * (1 + nu))

        self.stress['s'] = 0
        self.stress['t'] = 0
        self.strain['e'] = 0
        self.strain['g'] = 0

        self.commit()

    def copy(self):

        return deepcopy(self)

    def commit(self):

        self.stress_conv = deepcopy(self.stress)
        self.strain_conv = deepcopy(self.strain)
        self.stiff_conv = deepcopy(self.stiff)
        self.state_conv = deepcopy(self.state)

    def revert_commit(self):

        self.stress = deepcopy(self.stress_conv)
        self.strain = deepcopy(self.strain_conv)
        self.stiff = deepcopy(self.stiff_conv)
        self.state = deepcopy(self.state_conv)

    def get_forces(self):

        return np.array([self.stress['s'], self.stress['t']])

    def set_elongs(self, eps, gamma):

        self.strain['e'] = eps
        self.strain['g'] = gamma

    def update(self):

        self.stress['s'] = self.stiff['E'] * self.strain['e']
        self.stress['t'] = self.stiff['G'] * self.strain['g']

    def get_k_tan(self):

        return (self.stiff['E'], self.stiff['G'], 0)

    def get_k_init(self):

        return (self.stiff0['E'], self.stiff0['G'], 0)

    def to_ommit(self):

        return False


class Bilinear_Mat(Material):

    def __init__(self, E, nu, fy, corr_fact=1, shear_def=True):

        super().__init__(E, nu, corr_fact=corr_fact, shear_def=shear_def)

        self.stress['f_y'] = fy
        self.strain['e_y'] = fy / E

        self.commit()

    def update(self):

        # Elastic step: 
        if abs(self.strain['e']) <= self.strain['e_y']:
            self.stress['s'] = self.stiff0['E'] * self.strain['e']
            self.stiff['E'] = deepcopy(self.stiff0['E'])

        else:
            self.stress['s'] = np.sign(self.strain['e']) * self.stress['f_y']
            self.stiff['E'] = 0.

        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = deepcopy(self.stiff0['G'])


class Plastic_Mat(Material):

    def __init__(self, E, nu, fy, corr_fact=1, shear_def=True):

        super().__init__(E, nu, corr_fact=corr_fact, shear_def=shear_def)

        self.stress['f_y'] = fy
        self.strain['e_p'] = 0.

        self.commit()

    def plot_stress_strain(self):

        eps = np.linspace(0, 2 * self.strain['e_y'], 100)
        eps = np.append(eps, np.linspace(2 * self.strain['e_y'], -2 * self.strain['e_y'], 200))
        eps = np.append(eps, np.linspace(-2 * self.strain['e_y'], 3 * self.strain['e_y'], 300))
        eps = np.append(eps, np.linspace(3 * self.strain['e_y'], -3 * self.strain['e_y'], 400))
        gamma = np.zeros(len(eps))

        sig = np.zeros(len(eps))
        e_p = np.zeros(len(eps))

        for i in range(len(eps)):
            self.set_elongs(eps[i], gamma[i])
            self.update()
            sig[i] = self.stress['s']
            e_p[i] = self.strain['e_p']

        plt.figure(None, figsize=(6, 6))
        plt.plot(eps, sig, label='eps')
        plt.plot(e_p, sig, label='e_p')
        plt.grid(True)
        plt.legend()

    def update(self):

        s_tr = self.stiff0['E'] * (self.strain['e'] - self.strain['e_p'])
        f_tr = abs(s_tr) - (self.stress['f_y'])

        # Elastic step
        if f_tr <= 0:
            self.stress['s'] = s_tr
            self.stiff['E'] = deepcopy(self.stiff0['E'])

        # Plastic step
        else:
            d_g = f_tr / (self.stiff0['E'])

            self.stress['s'] = (1 - d_g * self.stiff0['E'] / abs(s_tr)) * s_tr
            self.strain['e_p'] += d_g * np.sign(s_tr)

            self.stiff['E'] = 0.

        # Shear behaviour is linear elastic
        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = self.stiff0['G']
