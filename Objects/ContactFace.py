# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:57:55 2024

@author: ibouckaert
"""

import numpy as np
from warnings import warn
import warnings 
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from Objects import ContactPair as CP

def custom_warning_format(message, category, filename, lineno, file=None, line=None): 
    file_short_name = filename.replace(os.path.dirname(filename),"")
    file_short_name = file_short_name.replace("\\","")
    return f"Warning! In file {file_short_name}, line {lineno}: {message}\n"

warnings.formatwarning = custom_warning_format

class CF_2D: 
    
    def __init__(self,cf,nb_cp,lin_geom,offset=-1,contact=None,surface=None):
        
        self.xe1 = cf['x_e1'].copy()
        self.xe2 = cf['x_e2'].copy()
        
        self.bl_A = cf['Block A']
        self.bl_B = cf['Block B']
        
        if self.bl_A.b != self.bl_B.b: warnings.warn('Cannot handle blocks with different depths')
        self.b = self.bl_A.b
        
        self.t = (self.xe2 - self.xe1) / np.linalg.norm(self.xe2 - self.xe1)
        self.n = np.array([[0, -1],[1, 0]]) @ self.t
        self.angle = np.arctan2(self.t[1], self.t[0])
        
        self.cps = []
        
        self.lin_geom = lin_geom
        
        if offset == -1: # Stress - strain
            
            if ((self.bl_A.material is None) or (self.bl_B.material is None)) and surface is None: 
                warn('No material was defined for Blocks with stress-strain approach')
            
            h_cp = np.linalg.norm(self.xe2 - self.xe1) / nb_cp
            x_cp = self.xe1 + .5 * h_cp * self.t
            
            for i in np.arange(nb_cp): 
                
                x_cp = self.xe1 + (i+.5) * h_cp * self.t
            
                l_Ax = np.dot((x_cp - self.bl_A.ref_point),-self.n)
                l_Ay = np.dot((x_cp - self.bl_A.ref_point), self.t)
                l_Bx = np.dot((x_cp - self.bl_B.ref_point), self.n)
                l_By = np.dot((x_cp - self.bl_B.ref_point), self.t)
                
                self.cps.append(CP.CP_2D(x_cp, l_Ax, l_Ay, l_Bx, l_By, self.angle - np.pi/2, h_cp, self.b, surface=surface, block_A=self.bl_A, block_B=self.bl_B, lin_geom=self.lin_geom))
                
        elif offset >= 0: 
            
            if nb_cp != 2: 
                warn('Trying to use contact or surface law with more than 2 CPs')
                
            if offset >= np.linalg.norm(self.xe2 - self.xe1)/2: 
                warn('Offset exceeds dimensions of CF')
            
            if not contact and not surface: 
                warn('Don\'t forget to specify a force-displacement law')
                
            h_cp = np.linalg.norm(self.xe2 - self.xe1) / nb_cp
            x_cp = self.xe1 + offset * self.t
            
            for i in np.arange(2): 
                  
                l_Ax = np.dot((x_cp - self.bl_A.ref_point),-self.n)
                l_Ay = np.dot((x_cp - self.bl_A.ref_point), self.t)
                l_Bx = np.dot((x_cp - self.bl_B.ref_point), self.n)
                l_By = np.dot((x_cp - self.bl_B.ref_point), self.t)
                
                self.cps.append(CP.CP_2D(x_cp, l_Ax, l_Ay, l_Bx, l_By, self.angle-np.pi/2, h_cp, self.b, contact=contact, lin_geom=self.lin_geom))
                
                x_cp += (np.linalg.norm(self.xe2 - self.xe1) - 2*offset) * self.t
                
        else: 
            warn('The definition of the CPs is not valid')
            
    def commit(self):
        
        for cp in self.cps: 
            cp.commit()
            
    # Method to revert committed changes to the contact pairs
    def revert_commit(self): 
        
        for cp in self.cps: 
            cp.revert_commit()

    def get_pf_glob(self, qf_glob): 
        
        self.qf_glob = qf_glob.copy()
        
        self.get_pf_loc()
        
        self.pf_glob = np.zeros(6)
        
        T = T3x3(self.angle-np.pi/2)
        
        self.pf_glob[:3] = np.transpose(T) @ self.pf_loc[:3]
        self.pf_glob[3:] = np.transpose(T) @ self.pf_loc[3:]
        
        return self.pf_glob
    
    def get_pf_loc(self): 
        
        T = T3x3(self.angle-np.pi/2)
        
        self.qf_loc = np.zeros(6)
        self.qf_loc[:3] = T @ self.qf_glob[:3]
        self.qf_loc[3:] = T @ self.qf_glob[3:]
        
        self.pf_loc = np.zeros(6)
        
        for cp in self.cps: 
            
            pc_loc = cp.get_pc_loc(self.qf_loc)
            
            if not cp.to_ommit(): 
                self.pf_loc += pc_loc
                
        
    def get_kf_glob(self): 
        
        self.get_kf_loc()
        
        self.kf_glob = np.zeros((6,6))
        
        T = T3x3(self.angle - np.pi/2)
        
        self.kf_glob[:3,:3] = np.transpose(T) @ self.kf_loc[:3,:3] @ T
        self.kf_glob[:3,3:] = np.transpose(T) @ self.kf_loc[:3,3:] @ T
        self.kf_glob[3:,:3] = np.transpose(T) @ self.kf_loc[3:,:3] @ T
        self.kf_glob[3:,3:] = np.transpose(T) @ self.kf_loc[3:,3:] @ T
        
        return self.kf_glob
    
    def get_kf_glob0(self): 
        
        self.get_kf_loc0()
        
        self.kf_glob0 = np.zeros((6,6))
        
        T = T3x3(self.angle - np.pi/2)
        
        self.kf_glob0[:3,:3] = np.transpose(T) @ self.kf_loc0[:3,:3] @ T
        self.kf_glob0[:3,3:] = np.transpose(T) @ self.kf_loc0[:3,3:] @ T
        self.kf_glob0[3:,:3] = np.transpose(T) @ self.kf_loc0[3:,:3] @ T
        self.kf_glob0[3:,3:] = np.transpose(T) @ self.kf_loc0[3:,3:] @ T
        
        return self.kf_glob0
        
    def get_kf_loc(self):
        
        self.kf_loc = np.zeros((6,6))
        
        for cp in self.cps: 
            
            kc_loc = cp.get_kc_loc()
            
            if not cp.to_ommit(): 
                self.kf_loc += kc_loc
                
    def get_kf_loc0(self):
        
        self.kf_loc0 = np.zeros((6,6))
        
        for cp in self.cps: 
            
            kc_loc0 = cp.get_kc_loc0()
            
            if not cp.to_ommit(): 
                self.kf_loc0 += kc_loc0
    
    def plot_cf(self,scale): 
        
        x = np.array([self.xe1[0], self.xe2[0]])
        y = np.array([self.xe1[1], self.xe2[1]])
        
        c = (self.xe1 + self.xe2) / 2
        
        import matplotlib as mpl
        tA = mpl.markers.MarkerStyle(marker=">")
        tA._transform = tA.get_transform().rotate(self.angle)
        
        plt.plot(x,y,marker='.',markersize=5,color='red',linewidth=.75)
        plt.plot(c[0],c[1],marker=tA,markersize=5,color='red',linewidth=.75)
        
        for cp in self.cps: 
            cp.plot(scale)
        
            
def T3x3(a): 

     return np.array([[np.cos(a), np.sin(a), 0],
                      [-np.sin(a), np.cos(a), 0],
                      [0, 0, 1]])     
            