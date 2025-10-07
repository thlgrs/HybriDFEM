# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:24:41 2024

@author: ibouckaert
"""

import numpy as np
import sys
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


class Material:

    def __init__(self, E, nu, corr_fact=1, shear_def=True):
        
        self.stiff = {}
        self.stiff0 = {}
        self.stress = {}
        self.strain = {}
        self.state = {}

        self.stiff['E'] = E
        self.stiff0['E'] = E

        self.tag = 'LINEL'

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

        self.tol_disp = 1e-20
        
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

    def update(self, dL):

        self.strain['e'] += dL[0]
        self.strain['g'] += dL[1]

        self.stress['s'] = self.stiff['E'] * self.strain['e']
        self.stress['t'] = self.stiff['G'] * self.strain['g']

    def get_k_tan(self):

        return (self.stiff['E'], self.stiff['G'], 0, 0)

    def get_k_init(self):

        return (self.stiff0['E'], self.stiff0['G'], 0, 0)

    def to_ommit(self): 
        
        return False


class NoTension_Mat(Material):

    def __init__(self, E, nu, corr_fact=1, shear_def=True):
        super().__init__(E, nu, corr_fact=corr_fact, shear_def=shear_def)
        self.tag = 'NTMAT'

    def to_ommit(self):
        if self.strain['e'] > 0:
            return True
        return False


class Bilinear_Mat(Material):

    def __init__(self, E, nu, fy, alpha, corr_fact=1, shear_def=True): 
        
        super().__init__(E, nu, corr_fact=corr_fact, shear_def=shear_def)

        self.stress['f_y'] = fy
        self.strain['e_y'] = fy / E
        self.a = alpha
        self.tag = 'BILIN'
        self.commit()

    def update(self, dL):

        self.strain['e'] += dL[0]
        self.strain['g'] += dL[1]
        
        # Elastic step: 
        d_e = abs(self.strain['e']) - self.strain['e_y']

        if d_e <= 0:
            self.stress['s'] = self.stiff0['E'] * self.strain['e']
            self.stiff['E'] = deepcopy(self.stiff0['E'])

        else:
            d_f = d_e * self.a * self.stiff0['E']

            self.stress['s'] = (self.stress['f_y'] + d_f) * np.sign(self.strain['e'])
            self.stiff['E'] = self.a * self.stiff0['E']
            
        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = deepcopy(self.stiff0['G'])


class Plastic_Mat(Material):

    def __init__(self, E, nu, fy, corr_fact=1, shear_def=True): 
        
        super().__init__(E, nu, corr_fact=corr_fact, shear_def=shear_def)

        self.stress['f_y'] = fy
        self.strain['e_p'] = 0.
        self.tag = 'PLAST'
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

    def update(self, dL):

        self.strain['e'] += dL[0]
        self.strain['g'] += dL[1]
        
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
            # self.strain['e_p'] += d_g * np.sign(s_tr)
            
            self.stiff['E'] = 0.

        #Shear behaviour is linear elastic
        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = self.stiff0['G']


class Plastic_Stiffness_Deg(Material):

    def __init__(self, E, nu, fy, corr_fact=1, shear_def=True):

        super().__init__(E, nu, corr_fact=corr_fact, shear_def=shear_def)

        self.stress['f_y'] = fy
        self.stiff['E_el'] = E
        self.strain['e_y'] = fy / E
        self.tag = 'STIFFDEG'
        self.commit()

    def plot_stress_strain(self):

        eps = np.linspace(0, 2 * self.strain['e_y'], 100)
        eps = np.append(eps, np.linspace(2 * self.strain['e_y'], 0, 200))
        eps = np.append(eps, np.linspace(0, 4 * self.strain['e_y'], 300))
        eps = np.append(eps, np.linspace(4 * self.strain['e_y'], 0, 400))
        gamma = np.zeros(len(eps))

        sig = np.zeros(len(eps))

        for i in range(len(eps)):
            self.set_elongs(eps[i], gamma[i])
            self.update()
            sig[i] = self.stress['s']

        plt.figure(None, figsize=(6, 6))
        plt.plot(eps * 100, sig / 1e6, label='eps')
        plt.grid(True)
        plt.legend()
        plt.xlabel('Strain [%]')
        plt.ylabel('Stress [MPa]')

        self.revert_commit()

    def update(self, dL):

        self.strain['e'] += dL[0]
        self.strain['g'] += dL[1]
        # Elastic step
        if abs(self.strain['e']) <= self.strain['e_y']:
            self.stress['s'] = self.stiff['E_el'] * self.strain['e']
            self.stiff['E'] = self.stiff['E_el']

        # Plastic step
        else:
            # print('Plastic')
            self.stress['s'] = np.sign(self.strain['e']) * self.stress['f_y']
            self.stiff['E_el'] = self.stress['f_y'] / abs(self.strain['e'])
            self.strain['e_y'] = abs(self.strain['e'])
            self.stiff['E'] = 0.
        # Shear behaviour is linear elastic
        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = self.stiff0['G']


class concrete_EC(Material):

    def __init__(self, Ec, fcd, fct=0., f_res=0., eps_c=0.002, eps_cu=0.0035, fail_crit=False):

        nu = 0.0

        super().__init__(Ec, nu)

        self.stress['f_cd'] = fcd
        self.stress['f_ct'] = fct
        self.stress['f_res'] = f_res
        self.strain['e_c'] = eps_c
        self.strain['e_t'] = self.stress['f_ct'] / self.stiff0['E']
        self.strain['e_cu'] = eps_cu
        self.fail_crit = fail_crit

        self.nameTag = "Eurocode ULS concrete model"

        self.commit()

    def to_ommit(self):
        if self.strain['e'] > self.strain['e_t']:
            return True
        return False

    def plot_stress_strain(self):

        """
        Displays the σ − ε and τ − γ curve of the material
        
        """
        import matplotlib.pyplot as plt

        eps = np.linspace(-self.strain['e_cu'], 2 * self.strain['e_t'], 500)
        gamma = np.linspace(-self.strain['e_cu'], 2 * self.strain['e_t'], 500)
        sigma = np.zeros(500)
        tau = np.zeros(500)

        for i in range(500):

            if i == 0:
                d_eps = eps[0]
                d_gam = gamma[0]
            else:
                d_eps = eps[i] - eps[i - 1]
                d_gam = gamma[i] - gamma[i - 1]

            self.update([d_eps, d_gam])
            sigma[i] = self.stress['s']
            tau[i] = self.stress['t']

        plt.figure(figsize=(5, 5), dpi=200)

        plt.plot(eps * 100, sigma / 1e6, color='black', linewidth=0.9, label=r'$\sigma(\varepsilon)$')
        if self.fail_crit:
            plt.scatter(-self.strain['e_cu'] * 100, -self.stress['f_cd'] / 1e6, color='red', marker='*', s=100,
                        label='Failure')
        # plt.plot(gamma*100, tau/1e6, color='black', linewidth=0.7, label=r'$\tau(\gamma)$', linestyle='dashed')
        plt.xlabel(r'Strain $\varepsilon$ or $\gamma$ [\%]')
        plt.ylabel(r'Stress $\sigma$ or $\tau$ [MPa]')
        plt.title('Stress-strain relation \n' + self.nameTag)
        plt.axhline(y=0, color='k', linewidth=.7)
        plt.axvline(x=0, color='k', linewidth=.7)
        plt.legend()
        plt.grid(True)

        self.revert_commit()

    def update(self, dL):

        """
        Updates the values of current strains to eps and gamma in the 
        self.strains dictionary
        Updates the values of current stresses corresponding to eps and gamma 
        in the self.stresses dictionary, according to a bilinear asymmetric 
        σ − ε and linear τ − γ relationship
        
        """
        super().update(dL)

        if self.strain['e'] > 1e-10:

            if self.strain['e'] > 11 * self.strain['e_t'] / 10:

                self.stress['s'] = self.stress['f_res']
                self.stress['t'] = 0
                self.stiff['E'] = 0
                self.stiff['G'] = 0

            elif self.strain['e'] > self.strain['e_t']:

                self.stress['t'] = self.stiff0['G'] * self.strain['g']
                self.stiff['E'] = 0 * (self.stress['f_res'] - self.stress['f_ct']) / self.strain['e_t']
                self.stress['s'] = self.stress['f_ct'] + self.stiff['E'] * (self.strain['e'] - self.strain['e_t'])
                self.stiff['G'] = self.stiff0['G']

            else:
                self.stress['s'] = self.stiff0['E'] * self.strain['e']
                self.stress['t'] = self.stiff0['G'] * self.strain['g']
                self.stiff['E'] = self.stiff0['E']
                self.stiff['G'] = self.stiff0['G']

        else:
            if self.strain['e'] >= - self.strain['e_c']:
                self.stress['s'] = (self.stress['f_cd'] / self.strain['e_c'] ** 2) * self.strain['e'] ** 2 \
                                   + 2 * (self.stress['f_cd'] / self.strain['e_c']) * self.strain['e']
                self.stiff['E'] = 2 * (self.stress['f_cd'] / self.strain['e_c'] ** 2) * self.strain['e'] \
                                  + 2 * (self.stress['f_cd'] / self.strain['e_c'])
            else:
                self.stress['s'] = - self.stress['f_cd']
                self.stiff['E'] = 0

        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = self.stiff0['G']

        if self.fail_crit:
            if self.strain['e'] < -self.strain['e_cu']:
                eps = self.strain['e']
                sys.exit(f'Concrete has failed in compression: eps_c = {eps}')


class steel_EC(Material):

    def __init__(self, E, fyd, eps_ud=0.04, fail_crit=False, alpha=0.00357):

        nu = 0.0

        super().__init__(E, nu)

        self.stress['f_yd'] = fyd
        self.strain['e_ud'] = eps_ud
        self.strain['e_y'] = self.stress['f_yd'] / self.stiff0['E']
        self.fail_crit = fail_crit
        self.alpha = alpha
        # self.k = k

        self.nameTag = "Eurocode ULS steel model"

        self.commit()

    def plot_stress_strain(self):

        """
        Displays the σ − ε and τ − γ curve of the material
        
        """
        import matplotlib.pyplot as plt

        eps = np.linspace(-self.strain['e_ud'], self.strain['e_ud'], 1000)
        gamma = np.linspace(-2.5 * self.strain['e_y'], 2.5 * self.strain['e_y'], 1000)
        sigma = np.zeros(1000)
        tau = np.zeros(1000)

        for i in range(1000):

            if i == 0:
                d_eps = eps[0]
                d_gam = gamma[0]
            else:
                d_eps = eps[i] - eps[i - 1]
                d_gam = gamma[i] - gamma[i - 1]

            self.update([d_eps, d_gam])
            sigma[i] = self.stress['s']
            tau[i] = self.stress['t']

        plt.figure(figsize=(5, 5), dpi=200)
        plt.plot(eps * 100, sigma / 1e6, color='black', linewidth=0.9, label=r'$\sigma(\varepsilon)$')
        plt.plot(gamma * 100, tau / 1e6, color='black', linewidth=0.7, label=r'$\tau(\gamma)$', linestyle='dashed')
        if self.fail_crit:
            plt.scatter(-4, -self.stress['f_yd'] / 1e6, color='red', marker='*', s=100, label='Failure')
            plt.scatter(4, self.stress['f_yd'] / 1e6, color='red', marker='*', s=100)

        plt.xlabel(r'Strain $\varepsilon$ or $\gamma$ [\%]')
        plt.ylabel(r'Stress $\sigma$ or $\tau$ [MPa]')
        # plt.title('Stress-strain relation \n' + self.nameTag)
        plt.axhline(y=0, color='k', linewidth=.7)
        plt.axvline(x=0, color='k', linewidth=.7)
        plt.legend()
        plt.grid(True)

        self.revert_commit()

    def update(self, dL):

        """
        Updates the values of current strains to eps and gamma in the 
        self.strains dictionary
        Updates the values of current stresses corresponding to eps and gamma 
        in the self.stresses dictionary, according to a bilinear asymmetric 
        σ − ε and linear τ − γ relationship
        
        """
        super().update(dL)

        if abs(self.strain['e']) < self.strain['e_y']:
            self.stress['s'] = self.stiff0['E'] * self.strain['e']
            self.stiff['E'] = self.stiff0['E']

        else:

            eps_p = abs(self.strain['e']) - self.strain['e_y']
            f_p = self.alpha * self.stiff0['E'] * eps_p
            self.stress['s'] = (self.stress['f_yd'] + f_p) * np.sign(self.strain['e'])
            self.stiff['E'] = self.stiff0['E'] * self.alpha

        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = self.stiff0['G']

        if self.fail_crit:
            if abs(self.strain['e']) > self.strain['e_ud']:
                eps = self.strain['e']
                sys.exit(f'Steel has failed in tension: eps_s = {eps}')


class steel_tensionchord(Material):

    def __init__(self, fy, fu, ey, eu, fail_crit=False):

        nu = 0.0
        super().__init__(fy / ey, nu)

        self.stiff0['E'] = fy / ey
        self.stiff['E'] = fy / ey
        self.stiff['E_sh'] = (fu - fy) / (eu - ey)

        self.stress['f_y'] = fy
        self.stress['f_u'] = fu
        self.strain['e_u'] = eu
        self.strain['e_y'] = ey

        self.fail_crit = fail_crit
        self.tag = 'STC'

        self.nameTag = "Tension Chord steel model"

        self.commit()

    def plot_stress_strain(self):

        """
        Displays the σ − ε and τ − γ curve of the material
        
        """
        import matplotlib.pyplot as plt

        eps = np.linspace(-self.strain['e_u'], self.strain['e_u'], 1000)
        gamma = np.linspace(-2.5 * self.strain['e_y'], 2.5 * self.strain['e_y'], 1000)
        sigma = np.zeros(1000)
        tau = np.zeros(1000)

        for i in range(1000):

            if i == 0:
                d_eps = eps[0]
                d_gam = gamma[0]
            else:
                d_eps = eps[i] - eps[i - 1]
                d_gam = gamma[i] - gamma[i - 1]

            self.update([d_eps, d_gam])
            sigma[i] = self.stress['s']
            tau[i] = self.stress['t']

        plt.figure(figsize=(5, 5), dpi=200)
        plt.plot(eps * 100, sigma / 1e6, color='black', linewidth=0.9, label=r'$\sigma(\varepsilon)$')
        plt.plot(gamma * 100, tau / 1e6, color='black', linewidth=0.7, label=r'$\tau(\gamma)$', linestyle='dashed')
        if self.fail_crit:
            # print(self.stress['f_u'])
            plt.scatter(-self.strain['e_u'] * 100, -self.stress['f_u'] / 1e6, color='red', marker='*', s=100,
                        label='Failure')
            plt.scatter(self.strain['e_u'] * 100, self.stress['f_u'] / 1e6, color='red', marker='*', s=100)

        plt.xlabel(r'Strain $\varepsilon$ or $\gamma$ [\%]')
        plt.ylabel(r'Stress $\sigma$ or $\tau$ [MPa]')
        # plt.title('Stress-strain relation \n' + self.nameTag)
        plt.axhline(y=0, color='k', linewidth=.7)
        plt.axvline(x=0, color='k', linewidth=.7)
        plt.legend()
        plt.grid(True)

        self.revert_commit()

    def update(self, dL):

        """
        Updates the values of current strains to eps and gamma in the 
        self.strains dictionary
        Updates the values of current stresses corresponding to eps and gamma 
        in the self.stresses dictionary, according to a bilinear asymmetric 
        σ − ε and linear τ − γ relationship
        
        """
        super().update(dL)

        if abs(self.strain['e']) < self.strain['e_y']:
            self.stress['s'] = self.stiff0['E'] * self.strain['e']
            self.stiff['E'] = self.stiff0['E']
            self.yielded = False

        else:
            eps_p = abs(self.strain['e']) - self.strain['e_y']
            f_p = self.stiff['E_sh'] * eps_p
            self.stress['s'] = (self.stress['f_y'] + f_p) * np.sign(self.strain['e'])
            self.stiff['E'] = self.stiff['E_sh']
            self.yielded = True

        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = self.stiff0['G']


class concrete_tensionchord(Material):

    def __init__(self, fc, Ec):

        super().__init__(Ec, 0.0)

        self.cracked = False
        self.randomized = False
        # import random
        # self.stress['fct'] =  random.gauss(1, 0.1) * fc
        # print(self.stress['fct'])
        self.stress['fct'] = fc
        self.strain['e_ct'] = self.stress['fct'] / Ec
        self.tag = 'CTC'

        self.commit()

    def plot_stress_strain(self):

        """
        Displays the σ − ε and τ − γ curve of the material
        
        """
        import matplotlib.pyplot as plt

        eps = np.linspace(-4 * self.strain['e_ct'], 2 * self.strain['e_ct'], 1000)
        sigma = np.zeros(1000)

        for i in range(1000):

            if i == 0:
                d_eps = eps[0]
            else:
                d_eps = eps[i] - eps[i - 1]

            self.update([d_eps, 0])
            sigma[i] = self.stress['s']

        plt.figure(figsize=(5, 5), dpi=200)
        plt.plot(eps * 100, sigma / 1e6, color='black', linewidth=0.9, label=r'$\sigma(\varepsilon)$')

        plt.xlabel(r'Strain $\varepsilon$ [\%]')
        plt.ylabel(r'Stress $\sigma$ [MPa]')
        # plt.title('Stress-strain relation \n' + self.nameTag)
        plt.axhline(y=0, color='k', linewidth=.7)
        plt.axvline(x=0, color='k', linewidth=.7)
        plt.legend()
        plt.grid(True)

        self.revert_commit()

    def update(self, dL):

        super().update(dL)

        if not self.randomized:
            import random
            self.stress['fct'] = random.gauss(1, 0.02) * self.stress['fct']
            self.strain['e_ct'] = self.stress['fct'] / self.stiff0['E']
            self.randomized = True

        if self.strain['e'] <= self.strain['e_ct']:
            self.stress['s'] = self.strain['e'] * self.stiff0['E']
            self.stiff['E'] = self.stiff0['E']
        else:
            self.stress['s'] = 0.
            self.stiff['E'] = 0.
            self.cracked = True

        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = self.stiff0['G']


class Mixed_Hardening_Mat(Material):
    def __init__(self, E, nu, fy, H, r=0.0, corr_fact=1, shear_def=True):
        super().__init__(E, nu, corr_fact=corr_fact, shear_def=shear_def)

        self.stress['f_y'] = fy
        self.strain['e_p'] = 0.0
        self.state['alpha'] = 0.0  # kinematic
        self.state['kappa'] = 0.0  # isotropic
        self.H = H  # total hardening modulus
        self.r = r  # isotropic/kinematic ratio (0 = kin., 1 = iso.)
        self.tag = 'MIXED_HARD'

        self.commit()

    # def 

    def update(self, dL):
        self.strain['e'] += dL[0]
        self.strain['g'] += dL[1]

        E = self.stiff0['E']
        H = self.H
        r = self.r

        # conv

        # Trial stress
        sig_tr = self.stress['s'] + E * dL[0]
        alpha = self.state['alpha']
        kappa = self.state['kappa']

        sig_red_tr = sig_tr - alpha
        phi = abs(sig_red_tr) - (self.stress['f_y'] + kappa)

        phi_tol = 1e-8
        if phi <= phi_tol:
            # Elastic step
            self.stress['s'] = sig_tr
            self.stiff['E'] = E
            # print("phi:", phi, "stiffness:", self.stiff['E'])  # <-- here
            self.state['alpha'] = alpha
            self.state['kappa'] = kappa

        else:
            # Plastic step
            h_eff = E + H
            miu = phi / h_eff
            sign_red = np.sign(sig_red_tr)

            # === Debug prints ===
            # print(f"phi: {phi:.4e}, miu: {miu:.4e}, sig_tr: {sig_tr:.4e}, sig_red_tr: {sig_red_tr:.4e}, alpha: {alpha:.4e}, kappa: {kappa:.4e}, stiff[E]: {self.stiff['E']:.4e}")

            self.stress['s'] = sig_tr - miu * E * sign_red
            self.state['alpha'] = alpha + (1 - r) * H * miu * sign_red
            self.state['kappa'] = kappa + r * H * miu
            self.strain['e_p'] += miu * sign_red
            self.stiff['E'] = (E * H) / h_eff

        # Always elastic in shear
        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = self.stiff0['G']


class popovics_concrete(Material):

    def __init__(self, fc, ec, ecu, n, ft, et, etu, b, gc=None):

        nu = 0.0
        E_0 = fc * n / (ec * (n - 1))

        super().__init__(E_0, nu)

        self.stress['fc'] = - fc
        self.stress['ft'] = ft

        self.n = n
        self.b = b

        # if gc is not None: 
        #     if fc*ec*np.pi / gc < 0 or fc*ec*np.pi / gc > 1: 
        #         print('Dissipation energy is not large enough')

        #     self.n = np.pi / np.arcsin(fc*ec*np.pi / gc)

        self.strain['ec'] = - ec
        self.strain['et'] = et
        self.strain['ecu'] = - ecu
        self.strain['etu'] = etu

        self.tag = 'PPC'
        self.nameTag = 'Popovics concrete model'

        self.commit()

    def plot_stress_strain(self):

        import matplotlib.pyplot as plt

        eps = np.linspace(1.2 * self.strain['ecu'], 1.2 * self.strain['etu'], 500)
        gamma = np.linspace(1.2 * self.strain['ecu'], 1.2 * self.strain['etu'], 500)
        E = np.zeros(500)
        sigma = np.zeros(500)
        tau = np.zeros(500)

        for i in range(500):

            if i == 0:
                d_eps = eps[0]
                d_gam = gamma[0]
            else:
                d_eps = eps[i] - eps[i - 1]
                d_gam = gamma[i] - gamma[i - 1]

            self.update([d_eps, d_gam])
            sigma[i] = self.stress['s']
            tau[i] = self.stress['t']
            E[i] = self.stiff['E']

        plt.figure(figsize=(5, 5), dpi=200)

        plt.plot(eps * 100, sigma / 1e6, color='black', linewidth=0.9, label=r'$\sigma(\varepsilon)$')

        # plt.plot(gamma*100, tau/1e6, color='black', linewidth=0.7, label=r'$\tau(\gamma)$', linestyle='dashed')
        plt.xlabel(r'Strain $\varepsilon$ or $\gamma$ [\%]')
        plt.ylabel(r'Stress $\sigma$ or $\tau$ [MPa]')
        plt.title('Stress-strain relation \n' + self.nameTag)
        plt.axhline(y=0, color='k', linewidth=.7)
        plt.axvline(x=0, color='k', linewidth=.7)
        plt.legend()
        plt.grid(True)

        # plt.figure(figsize=(5,5), dpi=200)

        # plt.plot(eps*100, E/1e9, color='black', linewidth=0.9, label=r'$\sigma(\varepsilon)$')

        # # plt.plot(gamma*100, tau/1e6, color='black', linewidth=0.7, label=r'$\tau(\gamma)$', linestyle='dashed')
        # plt.xlabel(r'Strain $\varepsilon$ or $\gamma$ [\%]')
        # plt.ylabel(r'Stress $\sigma$ or $\tau$ [MPa]')
        # plt.title('Stress-strain relation \n' + self.nameTag)
        # plt.axhline(y=0, color='k', linewidth=.7)
        # plt.axvline(x=0, color='k', linewidth=.7)
        # plt.legend()
        # plt.grid(True)

        self.revert_commit()

    def update(self, dL):

        super().update(dL)

        # Compression
        if self.strain['e'] <= 0:

            if self.strain['e'] < self.strain['ecu']:
                self.stress['s'] = 0.
                self.stiff['E'] = 0.
            else:
                r_e = self.strain['e'] / self.strain['ec']
                self.stress['s'] = (self.stress['fc'] * r_e) * (self.n / (self.n - 1 + (r_e) ** self.n))
                self.stiff['E'] = - self.stress['fc'] * self.n ** 2 * r_e ** self.n / (
                            self.strain['ec'] * (self.n + r_e ** self.n - 1) ** 2) \
                                  + self.stress['fc'] * self.n / (self.strain['ec'] * (self.n + r_e ** self.n - 1))
        else:

            if self.strain['e'] > self.strain['etu']:
                self.stress['s'] = 0.
                self.stiff['E'] = 0.
            elif self.strain['e'] > self.strain['et']:
                self.stress['s'] = self.stress['ft'] * self.b ** (
                            (self.strain['e'] - self.strain['et']) / (self.strain['etu'] - self.strain['et']))
                self.stiff['E'] = (self.stress['ft'] * np.log(self.b) * (self.b) ** (
                            (self.strain['e'] - self.strain['et']) / (self.strain['etu'] - self.strain['et']))) \
                                  / (self.strain['etu'] - self.strain['et'])
            else:
                self.stiff['E'] = self.stress['ft'] / self.strain['et']
                self.stress['s'] = self.stiff['E'] * self.strain['e']

        self.stress['t'] = self.stiff['G'] * self.strain['g']


class KSP_concrete(Material):

    def __init__(self, fc, ec, fu, ecu, ft, et, etu, b, gc=None, gt=None):

        nu = 0.0
        E_0 = 2 * fc / ec

        super().__init__(E_0, nu)

        self.stress['fc'] = - fc
        self.stress['fu'] = - fu
        self.stress['ft'] = ft

        self.b = b

        if gc is not None:
            R = fu / fc
            ecu = (1 / (1 + R)) * (2 * gc / fc - fc / E_0 + (1 + R) * ec + R ** 2 * fc / E_0)

        if gt is not None:
            etu = (np.log(b) / (ft * (b - 1))) * (gt) + et

        #     self.n = np.pi / np.arcsin(fc*ec*np.pi / gc)

        self.strain['ec'] = - ec
        self.strain['et'] = et
        self.strain['ecu'] = - ecu
        self.strain['etu'] = etu

        self.tag = 'KSP'
        self.nameTag = 'Kent-Scott-Park concrete model'

        self.commit()

    def plot_stress_strain(self):

        import matplotlib.pyplot as plt

        eps = np.linspace(1.2 * self.strain['ecu'], 1.2 * self.strain['etu'], 500)
        gamma = np.linspace(1.2 * self.strain['ecu'], 1.2 * self.strain['etu'], 500)
        E = np.zeros(500)
        sigma = np.zeros(500)
        tau = np.zeros(500)

        for i in range(500):

            if i == 0:
                d_eps = eps[0]
                d_gam = gamma[0]
            else:
                d_eps = eps[i] - eps[i - 1]
                d_gam = gamma[i] - gamma[i - 1]

            self.update([d_eps, d_gam])
            sigma[i] = self.stress['s']
            tau[i] = self.stress['t']
            E[i] = self.stiff['E']

        plt.figure(figsize=(5, 5), dpi=200)

        plt.plot(eps * 100, sigma / 1e6, color='black', linewidth=0.9, label=r'$\sigma(\varepsilon)$')

        # plt.plot(gamma*100, tau/1e6, color='black', linewidth=0.7, label=r'$\tau(\gamma)$', linestyle='dashed')
        plt.xlabel(r'Strain $\varepsilon$ or $\gamma$ [\%]')
        plt.ylabel(r'Stress $\sigma$ or $\tau$ [MPa]')
        plt.title('Stress-strain relation \n' + self.nameTag)
        plt.axhline(y=0, color='k', linewidth=.7)
        plt.axvline(x=0, color='k', linewidth=.7)
        plt.legend()
        plt.grid(True)

        plt.figure(figsize=(5, 5), dpi=200)

        plt.plot(eps * 100, E / 1e9, color='black', linewidth=0.9, label=r'$\sigma(\varepsilon)$')

        # plt.plot(gamma*100, tau/1e6, color='black', linewidth=0.7, label=r'$\tau(\gamma)$', linestyle='dashed')
        plt.xlabel(r'Strain $\varepsilon$ or $\gamma$ [\%]')
        plt.ylabel(r'Stress $\sigma$ or $\tau$ [MPa]')
        plt.title('Stress-strain relation \n' + self.nameTag)
        plt.axhline(y=0, color='k', linewidth=.7)
        plt.axvline(x=0, color='k', linewidth=.7)
        plt.legend()
        plt.grid(True)

        self.revert_commit()

    def update(self, dL):

        super().update(dL)

        # Compression
        if self.strain['e'] <= 0:

            if self.strain['e'] < self.strain['ecu']:
                self.stress['s'] = self.stress['fu']
                self.stiff['E'] = 0.
            elif self.strain['e'] < self.strain['ec']:
                self.stiff['E'] = (self.stress['fu'] - self.stress['fc']) / (self.strain['ecu'] - self.strain['ec'])
                self.stress['s'] = self.stiff['E'] * (self.strain['e'] - self.strain['ec']) + self.stress['fc']
            else:
                a = - self.stress['fc'] / self.strain['ec'] ** 2
                b = - 2 * a * self.strain['ec']
                self.stress['s'] = a * self.strain['e'] ** 2 + b * self.strain['e']
                self.stiff['E'] = 2 * a * self.strain['e'] + b

        else:

            if self.strain['e'] > self.strain['etu']:
                self.stress['s'] = 0.
                self.stiff['E'] = 0.
            elif self.strain['e'] > self.strain['et']:
                self.stress['s'] = self.stress['ft'] * self.b ** (
                            (self.strain['e'] - self.strain['et']) / (self.strain['etu'] - self.strain['et']))
                self.stiff['E'] = (self.stress['ft'] * np.log(self.b) * (self.b) ** (
                            (self.strain['e'] - self.strain['et']) / (self.strain['etu'] - self.strain['et']))) \
                                  / (self.strain['etu'] - self.strain['et'])
            else:
                self.stiff['E'] = self.stress['ft'] / self.strain['et']
                self.stress['s'] = self.stiff['E'] * self.strain['e']

        self.stress['t'] = self.stiff['G'] * self.strain['g']


class concrete_EC_softening(Material):

    def __init__(self, Ec, fcd, alpha=-0.1, eps_c=0.002, fail_crit=False, gt=None):

        nu = 0.0

        super().__init__(Ec, nu)

        import random

        self.stress['f_cd'] = fcd
        self.strain['e_c'] = eps_c
        self.alpha = alpha
        self.fail_crit = fail_crit

        if gt is not None:  # Use fracture energy to regularize ultimate strain
            e_t = fcd / Ec
            e_f = e_t + 2 * gt / fcd
            self.alpha = - fcd / (Ec * (e_f - e_t))

        self.nameTag = "Eurocode ULS concrete model with softening"

        self.commit()

    def to_ommit(self):

        return False

    def plot_stress_strain(self):

        """
        Displays the σ − ε and τ − γ curve of the material
        
        """
        import matplotlib.pyplot as plt

        eps = np.linspace(-10 * self.strain['e_c'], 0, 500)
        gamma = np.linspace(-10 * self.strain['e_c'], 0, 500)
        sigma = np.zeros(500)
        tau = np.zeros(500)

        for i in range(500):

            if i == 0:
                d_eps = eps[0]
                d_gam = gamma[0]
            else:
                d_eps = eps[i] - eps[i - 1]
                d_gam = gamma[i] - gamma[i - 1]

            self.update([d_eps, d_gam])
            sigma[i] = self.stress['s']
            tau[i] = self.stress['t']

        plt.figure(figsize=(5, 5), dpi=200)

        plt.plot(eps * 100, sigma / 1e6, color='black', linewidth=0.9, label=r'$\sigma(\varepsilon)$')

        # plt.plot(gamma*100, tau/1e6, color='black', linewidth=0.7, label=r'$\tau(\gamma)$', linestyle='dashed')
        plt.xlabel(r'Strain $\varepsilon$ or $\gamma$ [\%]')
        plt.ylabel(r'Stress $\sigma$ or $\tau$ [MPa]')
        plt.title('Stress-strain relation \n' + self.nameTag)
        plt.axhline(y=0, color='k', linewidth=.7)
        plt.axvline(x=0, color='k', linewidth=.7)
        plt.legend()
        plt.grid(True)

        self.revert_commit()

    def update(self, dL):

        """
        Updates the values of current strains to eps and gamma in the 
        self.strains dictionary
        Updates the values of current stresses corresponding to eps and gamma 
        in the self.stresses dictionary, according to a bilinear asymmetric 
        σ − ε and linear τ − γ relationship
        
        """
        super().update(dL)

        if self.strain['e'] > self.tol_disp:

            self.stress['s'] = 0.
            self.stress['t'] = 0.
            self.stiff['E'] = 0.
            self.stiff['G'] = 0.

        else:
            if self.strain['e'] >= - self.strain['e_c']:
                self.stress['s'] = (self.stress['f_cd'] / self.strain['e_c'] ** 2) * self.strain['e'] ** 2 \
                                   + 2 * (self.stress['f_cd'] / self.strain['e_c']) * self.strain['e']
                self.stiff['E'] = 2 * (self.stress['f_cd'] / self.strain['e_c'] ** 2) * self.strain['e'] \
                                  + 2 * (self.stress['f_cd'] / self.strain['e_c'])
            else:
                self.stress['s'] = - self.stress['f_cd'] - (abs(self.strain['e']) - self.strain['e_c']) * (
                            self.alpha * self.stiff0['E'])
                self.stiff['E'] = self.alpha * self.stiff0['E']

        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = self.stiff0['G']
