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
    file_short_name = filename.replace(os.path.dirname(filename),"")
    file_short_name = file_short_name.replace("\\","")
    return f"Warning! In file {file_short_name}, line {lineno}: {message}\n"

warnings.formatwarning = custom_warning_format


class Surface: 
    
    def __init__(self, k_n, k_s):
        
        self.stiff = {}
        self.stiff0 = {}
        self.stress = {}
        self.disps = {}
        
        self.stiff['kn'] = k_n
        self.stiff0['kn'] = k_n
        
        self.stiff['ks'] = k_s
        self.stiff0['ks'] = k_s
    
        self.stress['s'] = 0
        self.stress['t'] = 0 
        self.disps['n'] = 0 
        self.disps['s'] = 0
        
        self.commit()
        
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

    def update(self): 

        self.stress['s'] = self.stiff['kn'] * self.disps['n']
        self.stress['t'] = self.stiff['ks'] * self.disps['s']

    def get_k_tan(self): 
        
        return (self.stiff['kn'], self.stiff['ks'], 0)
    
    def get_k_init(self): 

        return (self.stiff0['kn'], self.stiff0['ks'], 0)
    
    def to_ommit(self): 
        
        return False
        
    