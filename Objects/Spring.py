# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 08:56:51 2024

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


class Spring_2D:

    def __init__(self, l_n, l_s, h, b, block=None, contact=None, surface=None):

        if contact is not None:
            if (block is not None) and (surface is not None): warnings.warn(
                'Contact/Surface or Material Law defined simultaneously')
            self.law = deepcopy(contact)
            self.h = 1
            self.b = 1
            self.l_n = 1
            self.l_s = 1
        elif surface is not None:
            if (block is not None): warnings.warn('Contact/Surface/Material Law defined simultaneously')
            self.law = deepcopy(surface)
            self.h = h
            self.b = b
            self.l_n = 1
            self.l_s = 1
        else:
            if block is None: warnings.warn('Should define at least one constitutive law')
            self.law = deepcopy(block.material)
            self.h = h
            self.b = b
            self.l_n = l_n
            self.l_s = l_s

        self.A = self.h * self.b

    def commit(self):

        self.law.commit()

    def revert_commit(self):

        self.law.revert_commit()

    def get_forces(self):

        return self.A * self.law.get_forces()

    def set_elongs(self, dL):

        self.dL = dL.copy()
        self.law.set_elongs(self.dL[0] / self.l_n, self.dL[1] / self.l_n)

    def update(self):

        self.law.update()

    def get_k_spring(self):

        E, G, EG = self.law.get_k_tan()
        E0, G0, EG0 = self.law.get_k_init()

        k_nn = E * self.A / self.l_n
        k_ns = EG * self.A / self.l_n
        k_ss = G * self.A / self.l_n

        k_spring = np.array([[k_nn, k_ns], [k_ns, k_ss]])

        if np.around(k_nn, 10) == 0:  k_spring[0, 0] = 1e-5 * E0 * self.A / self.l_n
        if np.around(k_ss, 10) == 0:  k_spring[1, 1] = 1e-5 * G0 * self.A / self.l_n

        return k_spring

    def to_ommit(self):

        return self.law.to_ommit()

    def get_k_spring0(self):

        E, G, EG = self.law.get_k_init()

        k_nn = E * self.A / self.l_n
        k_ns = EG * self.A / self.l_n
        k_ss = G * self.A / self.l_n

        k_spring0 = np.array([[k_nn, k_ns], [k_ns, k_ss]])

        return k_spring0
