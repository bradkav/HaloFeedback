# HaloFeedback
import time
import warnings
from abc import ABC, abstractmethod
from time import time as timeit

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate import quad, simps
from scipy.interpolate import interp1d
from scipy.special import ellipeinc, ellipkinc, ellipe, ellipk, betainc
from scipy.special import gamma as Gamma
from scipy.special import gamma as Gamma_func
from scipy.special import beta as Beta

try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x

from scipy.special import hyp2f1

# ------------------
G_N = 4.3021937e-3  # (km/s)^2 pc/M_sun
c = 2.99792458e5  # km/s

# Conversion factors
pc_to_km = 3.08567758149137e13

# Numerical parameters
N_GRID = 10000  # Number of grid points in the specific energy
N_KICK = 50  # Number of points to use for integration over Delta-epsilon

float_2eps = 2.0 * np.finfo(float).eps
# ------------------

# Alternative elliptic function which is valid for m > 1
def ellipeinc_alt(phi, m):
    beta = np.arcsin(np.clip(np.sqrt(m) * np.sin(phi), 0, 1))
    return np.sqrt(m) * ellipeinc(beta, 1 / m) + ((1 - m) / np.sqrt(m)) * ellipkinc(
        beta, 1 / m
    )

r_6 = 1e-6

class DistributionFunction(ABC):
    """
    Base class for phase space distribution of a DM spike surrounding a black
    hole with an orbiting body. Child classes must implement the following:

    Methods
        - rho_init(): initial density function
        - f_init() initial phase-space distribution function

    Attributes
        - r_sp: DM halo extent [pc]. Used for making grids for the calculation.
        - IDstr_model: ID string used for file names.
    """

    def __init__(self, M_BH=1e3, M_NS=1.0, Lambda=-1, b_max_fixed = -1):
        self.M_BH = M_BH  # Solar mass
        self.M_NS = M_NS  # Solar mass


        if Lambda <= 0:
            self.Lambda = np.sqrt(M_BH / M_NS)
        else:
            self.Lambda = Lambda
            
        self.b_max_fixed = b_max_fixed
        if (b_max_fixed > 0):
            print("> Warning in DistributionFunction: b_max_fixed has been set - Lambda will be ignored...")

        self.r_isco = 6.0 * G_N * M_BH / c ** 2

        # Initialise grid of r, eps and f(eps)
        self.r_grid = np.geomspace(self.r_isco, 1e5 * self.r_isco, int(0.9*N_GRID))
        self.r_grid = np.append(
            self.r_grid, np.geomspace(1.01 * self.r_grid[-1], 1e3 * self.r_sp, int(0.1*N_GRID))
        )
        self.r_grid = np.sort(self.r_grid)
        self.eps_grid = self.psi(self.r_grid)
        self.f_eps = self.f_init(self.eps_grid)

        # Density of states
        self.DoS = (
            np.sqrt(2) * (np.pi * G_N * self.M_BH) ** 3 * self.eps_grid ** (-5 / 2.0)
        )

        # Define a string which specifies the model parameters
        # and numerical parameters (for use in file names etc.)
        self.IDstr_num = "lnLambda=%.1f" % (np.log(self.Lambda),)

    @abstractmethod
    def rho_init(self, r):
        """
        Initial DM density of the system [MSun / pc^3].

        Parameters:
            - r : distance from center of spike [pc]
        """
        pass

    @abstractmethod
    def f_init(self, eps):
        """
        Initial phase-space distribution function.

        Parameters
            - eps : float or np.array
            Energy per unit mass in (km/s)^2
        """
        pass

    def plotDF(self):
        plt.figure()
        plt.semilogy(self.eps_grid, self.f_init(), "k--")
        plt.semilogy(self.eps_grid, self.f_eps)
        plt.xlim(1.0e8, 5e8)
        plt.ylim(1e3, 1e9)
        plt.xlabel(r"$\mathcal{E} = \Psi(r) - \frac{1}{2}v^2$ [(km/s)$^2$]")
        plt.ylabel(r"$f(\mathcal{E})$ [$M_\odot$ pc$^{-3}$ (km/s)$^{-3}$]")
        plt.show()

    def psi(self, r):
        """Gravitational potential as a function of r"""
        return G_N * self.M_BH / r

    def v_circ(self, r):
        """Circular velocity as a function of r"""
        return np.sqrt(self.psi(r))
        
    def v_max(self, r):
        """Maximum velocity as a function of r"""
        return np.sqrt(2 * self.psi(r))

    def rho(self, r, v_cut=-1):
        """DM mass density computed from f(eps).

        Parameters:
            - r : radius in pc
            - v_cut : maximum speed to include in density calculation
                     (defaults to v_max if not specified)
        """
        if v_cut < 0:
            v_cut = self.v_max(r)

        #BJK: Reduced precision of this calculation (number of values in v_list)
        v_cut = np.clip(v_cut, 0, self.v_max(r))
        vlist = np.sqrt(np.linspace(0, v_cut ** 2, 10000))
        flist = np.interp(
            self.psi(r) - 0.5 * vlist ** 2,
            self.eps_grid[::-1],
            self.f_eps[::-1],
            left=0,
            right=0,
        )
        integ = vlist ** 2 * flist
        return 4 * np.pi * simps(integ, vlist)

    def sigma_v(self, r):
        v_cut = self.v_max(r)

        v_cut = np.clip(v_cut, 0, self.v_max(r))
        vlist = np.sqrt(np.linspace(0, v_cut ** 2, 250))
        flist = np.interp(
            self.psi(r) - 0.5 * vlist ** 2,
            self.eps_grid[::-1],
            self.f_eps[::-1],
            left=0,
            right=0,
        )
        integ = vlist ** 4 * flist
        return np.sqrt(np.trapz(integ, vlist) / np.trapz(vlist ** 2 * flist, vlist))

    def TotalMass(self):
        return np.trapz(-self.P_eps(), self.eps_grid)

    def TotalEnergy(self):
        return np.trapz(-self.P_eps() * self.eps_grid, self.eps_grid)

    def b_90(self, v_orb):
        return G_N * self.M_NS / (v_orb ** 2)

    def b_min(self, v_orb):
        #return 0.001 / pc_to_km
        #return 15.0 / pc_to_km
        _b_min = 6 * G_N *self.M_NS / c**2 
        return _b_min

    def b_max(self, v_orb):
        if (self.b_max_fixed > 0):
            return self.b_max_fixed
    
        _b_max = self.Lambda * np.sqrt(self.b_90(v_orb) ** 2 + self.b_min(v_orb) ** 2)
        #print(_b_max)
        return _b_max

    def eps_min(self, v_orb):
        return 2 * v_orb ** 2 / (1 + self.b_max(v_orb) ** 2 / self.b_90(v_orb) ** 2)

    def eps_max(self, v_orb):
        return 2 * v_orb ** 2 / (1 + self.b_min(v_orb) ** 2 / self.b_90(v_orb) ** 2)



    def dfdt(self, r0, v_orb, v_cut=-1):
        """Time derivative of the distribution function f(eps).

        Parameters:
            - r0 : radial position of the perturbing body [pc]
            - v_orb: orbital velocity of the perturbing body [km/s]
            - v_cut: optional, only scatter with particles slower than v_cut [km/s]
                        defaults to v_max(r) (i.e. all particles)
        """

        f_minus = self.dfdt_minus(r0, v_orb, v_cut, N_KICK)
        f_plus = self.dfdt_plus(r0, v_orb, v_cut, N_KICK)

        #N_plus = np.trapz(self.DoS*f_plus, self.eps_grid)
        #N_minus = np.trapz(-self.DoS*f_minus, self.eps_grid)
        
        N_plus = 1
        N_minus = 1
        #print(N_minus, N_plus)

        return f_minus + (N_minus/N_plus)*f_plus

    #BJK - 24/07/2023
    def p_scat_dist(self, dE, eps, r0, v_orb, method="A"):

        v_star = np.sqrt(2*(self.psi(r0) - eps))
        _b90 = G_N*self.M_NS/v_orb**2
        
        DoS = np.sqrt(2) * (np.pi * G_N * self.M_BH) ** 3 * eps ** (-5 / 2.0)
        
        integ = 0.0
        if (method == "A"):
            def calc_dE(b):
                return 2*v_orb**2*( 1 + b**2/_b90**2)**-1
            
      
            dE_max = calc_dE(self.b_min(v_orb))
            dE_min = calc_dE(self.b_max(v_orb))
            
            integ = _b90**2*v_orb**2*dE**-2
            
        elif (method == "B"):
            _b90_tilde = G_N*self.M_NS/(v_star**2 + v_orb**2)
            def calc_dE(b):
                return 2*v_orb**2*(1 + 2*v_star**2/v_orb**2)*(1 + b**2/_b90_tilde**2)**-1
            
            
            dE_max = calc_dE(self.b_min(v_orb))
            dE_min = calc_dE(self.b_max(v_orb))
            
            integ = _b90_tilde**2*v_orb**2*(1 + 2*v_star**2/v_orb**2)*dE**-2
        
        
        if (dE < dE_min) or (dE > dE_max):
            return 0
        else:
            return (16*np.pi**3*r0*v_star/DoS)*integ
        

    #BJK - 24/07/2023
    def p_scat(self, r0, v_orb, method="A"):

        v_star = np.sqrt(2*(self.psi(r0) - self.eps_grid))
        _b90 = G_N*self.M_NS/v_orb**2
        
        
        integ = 0.0
        if (method == "A"):
            def calc_dE(b):
                return 2*v_orb**2*( 1 + b**2/_b90**2)**-1
            
            dE_max = calc_dE(self.b_min(v_orb))
            dE_min = calc_dE(self.b_max(v_orb))
            
            integ = _b90**2*v_orb**2*(dE_min**-1 - dE_max**-1)
            
        elif (method == "B"):
            _b90_tilde = G_N*self.M_NS/(v_star**2 + v_orb**2)
            def calc_dE(b):
                return 2*v_orb**2*(1 + 2*v_star**2/v_orb**2)*(1 + b**2/_b90_tilde**2)**-1
            
            dE_max = calc_dE(self.b_min(v_orb))
            dE_min = calc_dE(self.b_max(v_orb))
            
            integ = _b90_tilde**2*v_orb**2*(1 + 2*v_star**2/v_orb**2)*(dE_min**-1 - dE_max**-1)
        
        return (16*np.pi**3*r0*v_star/self.DoS)*integ


    def delta_f(self, r0, v_orb, dt, v_cut=-1):
        """Change in f over a time-step dt.
        Automatically prevents f_eps going below zero.

        Parameters:
            - r0 : radial position of the perturbing body [pc]
            - v_orb: orbital velocity [km/s]
            - dt: time-step [s]
            - v_cut: optional, only scatter with particles slower than v_cut [km/s]
                        defaults to v_max(r) (i.e. all particles)
        """

        METHOD = "A"
        
        if (METHOD == "A"):
            f_minus = self.dfdt_minus(r0, v_orb, v_cut, N_KICK) * dt
                
            correction = np.clip(self.f_eps / (-f_minus + 1e-50), 0, 1)
            f_minus = np.clip(f_minus, -self.f_eps, 0)
            f_plus = self.dfdt_plus(r0, v_orb, v_cut, N_KICK, correction) * dt

        #f_minus = self.dfdt_minus(r0, v_orb, v_cut, N_KICK) * dt
        elif (METHOD == "B"):
            f_minus = self.dfdt_minus_B(r0, v_orb, N_KICK) * dt
            correction = np.clip(self.f_eps / (-f_minus + 1e-50), 0, 1)
            f_minus = np.clip(f_minus, -self.f_eps, 0)
            f_plus = self.dfdt_plus_B(r0, v_orb, N_KICK, correction) * dt
        #print(self.eps_min(v_orb))
        #print(self.eps_max(v_orb))


        # Don't remove more particles than there are particles...
        #correction = np.clip(self.f_eps / (-f_minus + 1e-50), 0, 1)
        
        #f_minus = np.clip(f_minus, -self.f_eps, 0)
        #f_plus = self.dfdt_plus(r0, v_orb, v_cut, N_KICK, correction) * dt
        #f_plus = self.dfdt_plus_B(r0, v_orb, N_KICK, correction) * dt
        #f_plus = 0.0
        
        MAKE_PLOT = False
        
        if (MAKE_PLOT):
            #eps2 = np.geomspace(1e6, 1e9, 100)
            #p2 = 0.0*eps2
            #for i, eps in tqdm(enumerate(eps2)):
            #    p2[i] =  self.p_scat(r0, v_orb, eps)
        
            T_orb = 2*np.pi*r0*pc_to_km/v_orb
            dN = (dt/T_orb)
            #print(dN)
    
        
            plt.figure()
            #plt.semilogy(self.eps_grid, self.f_eps)
            plt.semilogy(self.eps_grid, -f_minus/self.f_eps/dN)
            plt.semilogy(self.eps_grid, f_plus/self.f_eps/dN)
            #plt.axvline(G_N*self.M_BH/r0, linestyle='--', color='k')
            for N in [1, 10, 100, 1e3, 1e4]:
                plt.axvline(G_N*self.M_BH/(N*self.r_isco), linestyle='--', color='k')
            
            #plt.semilogy(eps2, p2)
            plt.axvline(0.25*G_N*self.M_BH/(r0), linestyle='--', color='g')
            plt.xlim(0, 4.5e8)
            plt.show()

        return f_minus + f_plus

    def P_delta_eps(self, v, delta_eps):
        """
        Calcuate PDF for delta_eps
        """
        norm = self.b_90(v) ** 2 / (self.b_max(v) ** 2 - self.b_min(v) ** 2)
        return 2 * norm * v ** 2 / (delta_eps ** 2)

    def P_eps(self):
        """Calculate the PDF d{P}/d{eps}"""
        return (
            np.sqrt(2)
            * np.pi ** 3
            * (G_N * self.M_BH) ** 3
            * self.f_eps
            / self.eps_grid ** 2.5
        )

    def calc_delta_eps(self, v, n_kick=1):
        """
        Calculate average delta_eps integrated over different
        bins (and the corresponding fraction of particles which
        scatter with that delta_eps).
        """
        eps_min = self.eps_min(v)
        eps_max = self.eps_max(v)

        norm = self.b_90(v) ** 2 / (self.b_max(v) ** 2 - self.b_min(v) ** 2)

        eps_edges = np.linspace(eps_min, eps_max, n_kick + 1)
        #print(eps_max)

        def F_norm(eps):
            return -norm * 2 * v ** 2 / (eps)

        def F_avg(eps):
            return -norm * 2 * v ** 2 * np.log(eps)

        frac = np.diff(F_norm(eps_edges))
        eps_avg = np.diff(F_avg(eps_edges)) / frac

        return eps_avg, frac
 
    def dEdt_DF(self, r, v_cut = -1, average = False):
        """Rate of change of energy due to DF (km/s)^2 s^-1 M_sun.
        
        Parameters:
            - r : radial position of the perturbing body [pc]
            - v_cut: optional, only scatter with particles slower than v_cut [km/s]
                        defaults to v_max(r) (i.e. all particles)
            - average: determine whether to average over different radii
                        (average = False is default and should be correct).
        
        """
        v_orb = np.sqrt(G_N * (self.M_BH + self.M_NS) / r)
        
        CoulombLog = np.log(self.Lambda)

        if average:
            warnings.warn(
                "Setting 'average = True' is not necessarily the right thing to do..."
            )
            r_list = r + np.linspace(-1, 1, 3) * self.b_max(v_orb)
            rho_list = np.array([self.rho(r1, v_cut) for r1 in r_list])
            rho_eff = np.trapz(rho_list * r_list, r_list) / np.trapz(r_list, r_list)
        else:
            rho_eff = self.rho(r, v_cut)

        return (
            (1 / pc_to_km)
            * 4
            * np.pi
            * G_N ** 2
            * self.M_NS ** 2
            * rho_eff
            * CoulombLog
            / v_orb
        )

    def E_orb(self, r):
        return -0.5 * G_N * (self.M_BH + self.M_NS) / r

    def T_orb(self, r):
        return (
            2
            * np.pi
            * np.sqrt(pc_to_km ** 2 * r ** 3 / (G_N * (self.M_BH + self.M_NS)))
        )

    def interpolate_DF(self, eps_old, correction=1):
        # Distribution of particles before they scatter
        if hasattr(correction, "__len__"):
            f_old = np.interp(
                eps_old[::-1],
                self.eps_grid[::-1],
                self.f_eps[::-1] * correction[::-1],
                left=0,
                right=0,
            )[::-1]
        else:
            f_old = np.interp(
                eps_old[::-1], self.eps_grid[::-1], self.f_eps[::-1], left=0, right=0
            )[::-1]
        return f_old

    def delta_eps_of_b(self, v_orb, b):
        v_rel = 1.0*v_orb
        b90 = self.b_90(v_rel)
        return -2 * v_rel ** 2 * (1 + b ** 2 / b90 ** 2) ** -1

    # ---------------------
    # ----- df/dt      ----
    # ---------------------

    def dfdt_minus(self, r0, v_orb, v_cut=-1, n_kick=N_KICK):
        """Particles to remove from the distribution function at energy E."""
        if v_cut < 0:
            v_cut = self.v_max(r0)

        df = np.zeros(N_GRID)

        # Calculate sizes of kicks and corresponding weights for integration
        if n_kick == 1:  # Replace everything by the average if n_kick = 1
            delta_eps_list = (
                -2 * v_orb ** 2 * np.log(1 + self.Lambda ** 2) / self.Lambda ** 2,
            )
            frac_list = (1,)
        else:
            b_list = np.geomspace(self.b_min(v_orb), self.b_max(v_orb), n_kick)
            delta_eps_list = self.delta_eps_of_b(v_orb, b_list)

            # Step size for trapezoidal integration
            step = delta_eps_list[1:] - delta_eps_list[:-1]
            step = np.append(step, 0)
            step = np.append(0, step)

            # Make sure that the integral is normalised correctly
            renorm = np.trapz(self.P_delta_eps(v_orb, delta_eps_list), delta_eps_list)
            frac_list = 0.5 * (step[:-1] + step[1:]) / renorm

        # Sum over the kicks
        for delta_eps, b, frac in zip(delta_eps_list, b_list, frac_list):
            # Define which energies are allowed to scatter
            mask = (self.eps_grid > self.psi(r0) * (1 - b / r0) - 0.5 * v_cut ** 2) & (
                self.eps_grid < self.psi(r0) * (1 + b / r0)
            )


            r_eps = G_N * self.M_BH / self.eps_grid[mask]
            r_cut = G_N * self.M_BH / (self.eps_grid[mask] + 0.5 * v_cut ** 2)

            L1 = np.minimum((r0 - r0 ** 2 / r_eps) / b, 0.999999)
            alpha1 = np.arccos(L1)
            L2 = np.maximum((r0 - r0 ** 2 / r_cut) / b, -0.999999)
            alpha2 = np.arccos(L2)

            m = (2 * b / r0) / (1 - (r0 / r_eps) + b / r0)
            mask1 = (m <= 1) & (alpha2 > alpha1)
            mask2 = (m > 1) & (alpha2 > alpha1)

            N1 = np.zeros(len(m))
            if np.sum(mask1) > 0:
                N1[mask1] = ellipe(m[mask1]) - ellipeinc(
                    (np.pi - alpha2[mask1]) / 2, m[mask1]
                )
            if np.sum(mask2) > 0:
                N1[mask2] = ellipeinc_alt((np.pi - alpha1[mask2]) / 2, m[mask2])
            df[mask] += (
                -frac
                * self.f_eps[mask]
                * (1 + b ** 2 / self.b_90(v_orb) ** 2) ** 2
                * np.sqrt(1 - r0 / r_eps + b / r0)
                * N1
            )

        T_orb = (2 * np.pi * r0 * pc_to_km) / v_orb
        norm = (
            2
            * np.sqrt(2 * (self.psi(r0)))
            * 4
            * np.pi ** 2
            * r0
            * (self.b_90(v_orb) ** 2 / (v_orb) ** 2)
        )
        result = norm * df / T_orb / self.DoS
        result[self.eps_grid >= 0.9999*self.psi(self.r_isco)] *= 0
        return result
    
    def dE_of_b_B(self, b, r0, v_orb):
        delta = np.clip(self.psi(r0) - self.eps_grid, 0, 1e30)
        v_star = np.sqrt(2*delta)
        
        b_90_tilde = G_N*self.M_NS/(v_star**2 + v_orb**2)
        chi = (1 + b**2/b_90_tilde**2)**-1
        return -2*v_orb**2*(1 + 2*v_star**2/v_orb**2)*chi
        
    def dfdt_minus_B(self, r0, v_orb, n_kick=N_KICK):
        mask = self.eps_grid < self.psi(r0)
        delta = np.clip(self.psi(r0) - self.eps_grid, 0, 1e30)
        v_star = np.sqrt(2*delta)
        
        b_90_tilde = G_N*self.M_NS/(v_star**2 + v_orb**2)
    
        dE_min = self.dE_of_b_B(self.b_max(v_orb), r0, v_orb)
        dE_max = self.dE_of_b_B(self.b_min(v_orb), r0, v_orb)
        p_eps = (16*np.pi**3*r0/self.DoS) * b_90_tilde**2*v_orb**2*v_star* \
                    (1 + 2*v_star**2/v_orb**2)*(dE_min**-1 - dE_max**-1)
        
        T_orb = (2 * np.pi * r0 * pc_to_km) / v_orb
        
        df = 0.0*self.f_eps
        df[mask] = self.f_eps[mask]*p_eps[mask]/T_orb
        return df
        
    def dfdt_plus_B(self, r0, v_orb, n_kick=N_KICK, correction=1):
        
        dE_mesh = np.zeros((n_kick, len(self.eps_grid)))
        dE_min_list = self.dE_of_b_B(self.b_max(v_orb), r0, v_orb)
        dE_max_list = self.dE_of_b_B(self.b_min(v_orb), r0, v_orb)
        #Check for appropriate spacing here...
        dE_mesh = np.geomspace(dE_min_list, dE_max_list, n_kick)
        
        eps_mesh = np.tile(self.eps_grid, (n_kick, 1))
        
        eps_old = eps_mesh - dE_mesh
        
        f_inter = interp1d(self.eps_grid, self.f_eps*correction, bounds_error=False, fill_value = 0.0)
        f_old = f_inter(eps_old)
        
        mask = eps_old < self.psi(r0)
        #np.where(mask, X, 0)
        #-------------------
        
        delta = np.clip(self.psi(r0) - eps_old, 0, 1e30)
        v_star = np.sqrt(2*delta)
        
        b_90_tilde = G_N*self.M_NS/(v_star**2 + v_orb**2)
    
        P_eps = (16*np.pi**3*r0/self.DoS) * b_90_tilde**2*v_orb**2*v_star* \
                    (1 + 2*v_star**2/v_orb**2)*dE_mesh**-2
        
        T_orb = (2 * np.pi * r0 * pc_to_km) / v_orb
        
        #DoS_corr = (self.eps_grid/eps_old)**(5/2)
        DoS_corr = 1.0
        
        #df = 0.0*self.f_eps
        #df[mask] = self.f_eps[mask]*p_eps[mask]/T_orb
        df = -np.trapz(mask*P_eps*f_old*DoS_corr, dE_mesh, axis=0)/T_orb
        #print(df)
        return df
        
        
        
        

    def dfdt_plus(self, r0, v_orb, v_cut=-1, n_kick=N_KICK, correction=1):
        """Particles to add back into distribution function from E - dE -> E."""
        if v_cut < 0:
            v_cut = self.v_max(r0)

        T_orb = (2 * np.pi * r0 * pc_to_km) / v_orb

        df = np.zeros(N_GRID)

        # Calculate sizes of kicks and corresponding weights for integration
        if n_kick == 1:  # Replace everything by the average if n_kick = 1
            delta_eps_list = (
                -2 * v_orb ** 2 * np.log(1 + self.Lambda ** 2) / self.Lambda ** 2,
            )
            frac_list = (1,)
        else:
            b_list = np.geomspace(self.b_min(v_orb), self.b_max(v_orb), n_kick)
            delta_eps_list = self.delta_eps_of_b(v_orb, b_list)

            # Step size for trapezoidal integration
            step = delta_eps_list[1:] - delta_eps_list[:-1]
            step = np.append(step, 0)
            step = np.append(0, step)

            # Make sure that the integral is normalised correctly
            renorm = np.trapz(self.P_delta_eps(v_orb, delta_eps_list), delta_eps_list)
            frac_list = 0.5 * (step[:-1] + step[1:]) / renorm

        # Sum over the kicks
        for delta_eps, b, frac in zip(delta_eps_list, b_list, frac_list):
            # Value of specific energy before the kick
            eps_old = self.eps_grid - delta_eps
  
            # Define which energies are allowed to scatter
            mask = (eps_old > self.psi(r0) * (1 - b / r0) - 0.5 * v_cut ** 2) & (
                eps_old < self.psi(r0) * (1 + b / r0)
            )

            # Sometimes, this mask has no non-zero entries
            if np.sum(mask) > 0:
                r_eps = G_N * self.M_BH / eps_old[mask]
                r_cut = G_N * self.M_BH / (eps_old[mask] + 0.5 * v_cut ** 2)

                eps0 = G_N*self.M_BH/r0

                # Distribution of particles before they scatter
                f_old = self.interpolate_DF(eps_old[mask], correction)

                L1 = np.minimum((r0 - r0 ** 2 / r_eps) / b, 0.999999)
                alpha1 = np.arccos(L1)
                
                L2 = np.maximum((r0 - r0 ** 2 / r_cut) / b, -0.999999)
         
                alpha2 = np.arccos(L2)

                m = (2 * b / r0) / (1 - (r0 / r_eps) + b / r0)
                mask1 = (m <= 1) & (alpha2 > alpha1)
                mask2 = (m > 1) & (alpha2 > alpha1)

                N1 = np.zeros(len(m))
                if np.sum(mask1) > 0:
                    N1[mask1] = ellipe(m[mask1]) - ellipeinc(
                        (np.pi - alpha2[mask1]) / 2, m[mask1]
                    )
                if np.sum(mask2) > 0:
                    N1[mask2] = ellipeinc_alt(
                        (np.pi - alpha1[mask2]) / 2, m[mask2]
                    )  # - ellipeinc_alt((np.pi - alpha2[mask2])/2, m[mask2])

                df[mask] += (
                    frac
                    * f_old
                    * (1 + b ** 2 / self.b_90(v_orb) ** 2) ** 2
                    * np.sqrt(1 - r0 / r_eps + b / r0)
                    * N1
                )

        T_orb = (2 * np.pi * r0 * pc_to_km) / v_orb
        norm = (
            2
            * np.sqrt(2 * (self.psi(r0)))
            * 4
            * np.pi ** 2
            * r0
            * (self.b_90(v_orb) ** 2 / (v_orb) ** 2)
        )
        result = norm * df / T_orb / self.DoS
        result[self.eps_grid >= 0.9999*self.psi(self.r_isco)] *= 0
        return result

    def dMdtdE_ej(self, r0, v_orb, v_cut=-1, n_kick=N_KICK, correction=np.ones(N_GRID)):
        """Particles to add back into distribution function from E - dE -> E."""
        if v_cut < 0:
            v_cut = self.v_max(r0)

        T_orb = (2 * np.pi * r0 * pc_to_km) / v_orb

        df = np.zeros(N_GRID)
        

        # Calculate sizes of kicks and corresponding weights for integration
        if n_kick == 1:  # Replace everything by the average if n_kick = 1
            delta_eps_list = (
                -2 * v_orb ** 2 * np.log(1 + self.Lambda ** 2) / self.Lambda ** 2,
            )
            frac_list = (1,)
        else:
            b_list = np.geomspace(self.b_min(v_orb), self.b_max(v_orb), n_kick)
            delta_eps_list = self.delta_eps_of_b(v_orb, b_list)
            E0 = np.min(self.eps_grid) + np.min(delta_eps_list)
            E_grid_new = -np.geomspace(np.abs(E0), 0.1, N_GRID)
           
            # Step size for trapezoidal integration
            step = delta_eps_list[1:] - delta_eps_list[:-1]
            step = np.append(step, 0)
            step = np.append(0, step)

            # Make sure that the integral is normalised correctly
            renorm = np.trapz(self.P_delta_eps(v_orb, delta_eps_list), delta_eps_list)
            frac_list = 0.5 * (step[:-1] + step[1:]) / renorm

        # Sum over the kicks
        for delta_eps, b, frac in zip(delta_eps_list, b_list, frac_list):
            # Value of specific energy before the kick
            eps_old = E_grid_new - delta_eps
    
            # Define which energies are allowed to scatter
            mask = (eps_old > self.psi(r0) * (1 - b / r0) - 0.5 * v_cut ** 2) & (
                eps_old < self.psi(r0) * (1 + b / r0)
            )

            # Sometimes, this mask has no non-zero entries
            if np.sum(mask) > 0:
                r_eps = G_N * self.M_BH / eps_old[mask]
                r_cut = G_N * self.M_BH / (eps_old[mask] + 0.5 * v_cut ** 2)

                # Distribution of particles before they scatter
                f_old = self.interpolate_DF(eps_old[mask], correction)

                L1 = np.minimum((r0 - r0 ** 2 / r_eps) / b, 0.999999)

                alpha1 = np.arccos(L1)
                L2 = np.maximum((r0 - r0 ** 2 / r_cut) / b, -0.999999)
                alpha2 = np.arccos(L2)

                m = (2 * b / r0) / (1 - (r0 / r_eps) + b / r0)
                mask1 = (m <= 1) & (alpha2 > alpha1)
                mask2 = (m > 1) & (alpha2 > alpha1)

                N1 = np.zeros(len(m))
                if np.sum(mask1) > 0:
                    N1[mask1] = ellipe(m[mask1]) - ellipeinc(
                        (np.pi - alpha2[mask1]) / 2, m[mask1]
                    )
                if np.sum(mask2) > 0:
                    N1[mask2] = ellipeinc_alt(
                        (np.pi - alpha1[mask2]) / 2, m[mask2]
                    )  # - ellipeinc_alt((np.pi - alpha2[mask2])/2, m[mask2])

                df[mask] += (
                    frac
                    * f_old
                    * (1 + b ** 2 / self.b_90(v_orb) ** 2) ** 2
                    * np.sqrt(1 - r0 / r_eps + b / r0)
                    * N1
                )

        T_orb = (2 * np.pi * r0 * pc_to_km) / v_orb
        norm = (
            2
            * np.sqrt(2 * (self.psi(r0)))
            * 4
            * np.pi ** 2
            * r0
            * (self.b_90(v_orb) ** 2 / (v_orb) ** 2)
        )
        result = norm * df / T_orb
        
        return E_grid_new, result

    def dEdt_ej(self, r0, v_orb, v_cut=-1, n_kick=N_KICK, correction=np.ones(N_GRID)):
        """Calculate energy carried away by particles which are completely unbound.

        Parameters:
            - r0 : radial position of the perturbing body [pc]
            - v_orb: orbital velocity [km/s]
            - v_cut: optional, only scatter with particles slower than v_cut [km/s]
                        defaults to v_max(r) (i.e. all particles)
            - n_kick: optional, number of grid points to use when integrating over
                        Delta-eps (defaults to N_KICK = 100).
        """
        if v_cut < 0:
            v_cut = self.v_max(r0)

        T_orb = (2 * np.pi * r0 * pc_to_km) / v_orb

        dE = np.zeros(N_GRID)

        # Calculate sizes of kicks and corresponding weights for integration
        if n_kick == 1:  # Replace everything by the average if n_kick = 1
            delta_eps_list = (
                -2 * v_orb ** 2 * np.log(1 + self.Lambda ** 2) / self.Lambda ** 2,
            )
            frac_list = (1,)

        else:
            b_list = np.geomspace(self.b_min(v_orb), self.b_max(v_orb), n_kick)
            delta_eps_list = self.delta_eps_of_b(v_orb, b_list)

            # Step size for trapezoidal integration
            step = delta_eps_list[1:] - delta_eps_list[:-1]
            step = np.append(step, 0)
            step = np.append(0, step)

            # Make sure that the integral is normalised correctly
            renorm = np.trapz(self.P_delta_eps(v_orb, delta_eps_list), delta_eps_list)
            frac_list = 0.5 * (step[:-1] + step[1:]) / renorm

        # Sum over the kicks
        for delta_eps, b, frac in zip(delta_eps_list, b_list, frac_list):

            # Maximum impact parameter which leads to the ejection of particles
            b_ej_sq = self.b_90(v_orb) ** 2 * ((2 * v_orb ** 2 / self.eps_grid) - 1)

            # Define which energies are allowed to scatter
            mask = (
                (self.eps_grid > self.psi(r0) * (1 - b / r0) - 0.5 * v_cut ** 2)
                & (self.eps_grid < self.psi(r0) * (1 + b / r0))
                & (b ** 2 < b_ej_sq)
            )

            r_eps = G_N * self.M_BH / self.eps_grid[mask]
            r_cut = G_N * self.M_BH / (self.eps_grid[mask] + 0.5 * v_cut ** 2)

            if np.sum(mask) > 0:

                L1 = np.minimum((r0 - r0 ** 2 / r_eps) / b, 0.999999)
                alpha1 = np.arccos(L1)
                L2 = np.maximum((r0 - r0 ** 2 / r_cut) / b, -0.999999)
                alpha2 = np.arccos(L2)

                m = (2 * b / r0) / (1 - (r0 / r_eps) + b / r0)
                mask1 = (m <= 1) & (alpha2 > alpha1)
                mask2 = (m > 1) & (alpha2 > alpha1)

                N1 = np.zeros(len(m))
                if np.sum(mask1) > 0:
                    N1[mask1] = ellipe(m[mask1]) - ellipeinc(
                        (np.pi - alpha2[mask1]) / 2, m[mask1]
                    )
                if np.sum(mask2) > 0:
                    N1[mask2] = ellipeinc_alt((np.pi - alpha1[mask2]) / 2, m[mask2])

                dE[mask] += (
                    -frac
                    * correction[mask]
                    * self.f_eps[mask]
                    * (1 + b ** 2 / self.b_90(v_orb) ** 2) ** 2
                    * np.sqrt(1 - r0 / r_eps + b / r0)
                    * N1
                    * (self.eps_grid[mask] + delta_eps)
                )

        norm = (
            2
            * np.sqrt(2 * (self.psi(r0)))
            * 4
            * np.pi ** 2
            * r0
            * (self.b_90(v_orb) ** 2 / (v_orb) ** 2)
        )
        return norm * np.trapz(dE, self.eps_grid) / T_orb


class PowerLawSpike(DistributionFunction):
    """
    A spike with a power law profile:

        rho(r) = rho_sp * (r_sp / r)^gamma.

    The parameter r_sp is defined as r_sp = 0.2 r_h, where r_h is the radius of
    the sphere within which the DM mass is twice the central BH mass.

    Notes
    -----
    The parameters are not properties, so r_sp will not have the correct value
    if rho_sp or gamma are changed after initialization.
    """

    def __init__(self, M_BH=1e3, M_NS=1.0, gamma=7 / 3, rho_sp=226, Lambda=-1, b_max_fixed=-1):
        if gamma <= 1:
            raise ValueError("gamma must be greater than 1")
        self.M_BH = M_BH  # Solar mass
        self.M_NS = M_NS  # Solar mass
        self.gamma = gamma  # Slope of DM density profile
        self.rho_sp = rho_sp  # Solar mass/pc^3
        self.r_sp = (
            (3 - self.gamma)
            * (0.2 ** (3.0 - self.gamma))
            * self.M_BH
            / (2 * np.pi * self.rho_sp)
        ) ** (
            1.0 / 3.0
        )  # pc

        self.IDstr_model = f"gamma={gamma:.2f}_rhosp={rho_sp:.1f}"

        self.xi_init = 1 - betainc(gamma - 1 / 2, 3 / 2, 1 / 2)

        super().__init__(M_BH, M_NS, Lambda, b_max_fixed)

    def f_init(self, eps):
        A1 = self.r_sp / (G_N * self.M_BH)
        return (
            self.rho_sp
            * (
                self.gamma
                * (self.gamma - 1)
                * A1 ** self.gamma
                * np.pi ** -1.5
                / np.sqrt(8)
            )
            * (Gamma(-1 + self.gamma) / Gamma(-1 / 2 + self.gamma))
            * self.eps_grid ** (-(3 / 2) + self.gamma)
        )

    def rho_init(self, r):
        return self.rho_sp * (self.r_sp / r) ** self.gamma


class PlateauSpike(DistributionFunction):
    """
    A spike with no DM particles whose orbits are completely contained within
    an annihilation plateau of radius r_p:

        rho(r)=
            rho_s (r_s / r)^gamma,                   r_s > r > r_p
            rho_s (r_s / r_p)^gamma (r_p / r)^{1/2}, r_p > r

    The parameter r_sp is defined as r_sp = 0.2 r_h, where r_h is the radius of
    the sphere within which the DM mass is twice the central BH mass.

    Notes
    -----
    The parameters are not properties, so r_sp will not have the correct value
    if rho_sp or gamma are changed after initialization.
    """

    def __init__(self, M_BH=1e3, M_NS=1.0, gamma=7 / 3, rho_sp=226, r_p=0.0, Lambda=-1, b_max_fixed=-1):
        self.M_BH = M_BH  # Solar mass
        self.M_NS = M_NS  # Solar mass
        self.gamma = gamma  # Slope of DM density profile
        self.rho_sp = rho_sp  # Solar mass/pc^3
        self.r_sp = (
            (3 - self.gamma)
            * (0.2 ** (3.0 - self.gamma))
            * self.M_BH
            / (2 * np.pi * self.rho_sp)
        ) ** (
            1.0 / 3.0
        )  # pc

        if gamma <= 1:
            raise ValueError("gamma must be greater than 1")

        if r_p > r_sp:
            raise ValueError("annihilation plateau radius larger than spike")

        self.r_p = r_p
        self.IDstr_model = f"gamma={gamma:.2f}_rhosp={rho_sp:.1f}_rp={r_p:.2E}"

        super().__init__(M_BH, M_NS, Lambda, b_max_fixed)

    def f_init(self, eps):
        def f_init(eps):
            if G_N * self.M_BH / self.r_sp < eps and eps <= G_N * self.M_BH / self.r_p:
                return (
                    self.rho_sp
                    * ((eps * self.r_sp) / (G_N * self.M_BH)) ** self.gamma
                    * (
                        (1 - self.gamma)
                        * self.gamma
                        * Beta(
                            -1 + self.gamma, 0.5, (G_N * self.M_BH) / (eps * self.r_sp)
                        )
                        + (np.sqrt(np.pi) * Gamma(1 + self.gamma))
                        / Gamma(-0.5 + self.gamma)
                    )
                ) / (2.0 * np.sqrt(2) * eps ** 1.5 * np.pi ** 2)
            else:
                return 0.0

        return np.vectorize(f_init)(eps)

    def rho_init(self, r):
        def rho_init(r):
            if r >= self.r_p:
                return self.rho_sp * (self.r_sp / r) ** self.gamma
            elif r < self.r_p:
                return (
                    self.rho_sp
                    * (self.r_sp / self.r_p) ** self.gamma
                    * (self.r_p / r) ** 0.5
                )

        return np.vectorize(rho_init)(r)
        
        
class GeneralizedNFWSpike(DistributionFunction):
    """
    
    """

    def __init__(self, M_BH=1e3, M_NS=1.0, rho_6=1e16, gamma_sp=7/3, r_t=1e-8, alpha=2, Lambda=-1, b_max_fixed=-1):
        
        self.M_BH   = M_BH
        self.M_NS   = M_NS
        self.rho_6  = rho_6
        self.r_t    = r_t
        self.gamma_sp = gamma_sp
        self.alpha  = alpha
        
        _k = (3 - gamma_sp)*0.2**(3-gamma_sp)/(2*np.pi)
        self.rho_sp = ((rho_6*r_6**gamma_sp)*((_k*M_BH)**(-gamma_sp/3)))**(1/(1-gamma_sp/3))
        self.r_sp = (
            (3 - self.gamma_sp)
            * (0.2 ** (3.0 - self.gamma_sp))
            * self.M_BH
            / (2 * np.pi * self.rho_sp)
        ) ** (
            1.0 / 3.0
        )  # pc
        
        self.r_isco = 6.0 * G_N * M_BH / c ** 2
        
        #Strictly speaking, r_min should be 2*r_isco (https://arxiv.org/abs/1305.2619)
        self.r_min  = self.r_isco
        
        super().__init__(M_BH, M_NS, Lambda, b_max_fixed)
    
    def f_init(self, E):
        A1 = r_6 / (G_N * self.M_BH)
        psi_t =  (G_N * self.M_BH)/self.r_t
        alpha = self.alpha
        gamma = self.gamma_sp
        
        AB_pre = 4*(gamma - 1)/(4*alpha**2 + 8*alpha*gamma + 4*gamma**2 - 1)
        A = psi_t*(2*alpha + 2*gamma + 1)*hyp2f1(alpha+2, alpha + gamma, alpha+gamma + 1/2, -E/psi_t)
        B = gamma*E*hyp2f1(alpha + 2, alpha + gamma + 1, alpha + gamma + 3/2, -E/psi_t)
        C = (psi_t**2/E)*hyp2f1(alpha + 2, alpha + gamma - 1, alpha + gamma - 1/2, -E/psi_t) 
        return (
            self.rho_6 * (2*np.pi) ** -1.5 * (psi_t**(-alpha-2)) *(A1 ** gamma)
            * (Gamma_func(alpha + gamma + 1) / Gamma_func(alpha + gamma - 1/2))
            * E ** (alpha + gamma - 1/2)
            * (AB_pre*(A + B) + C)
        )
        
    def rho_init(self, r):
        return self.rho_6*(r/r_6)**(-self.gamma_sp)*(1 + r/self.r_t)**-self.alpha
        
    def M_DM_ini(self, r):
        x = r/r_6
        H = hyp2f1(self.alpha, 3 - self.gamma_sp, 4 - self.gamma_sp, -r/self.r_t)
        return (4*np.pi*r_6**3*self.rho_6/(3 - self.gamma_sp))*x**(3 - self.gamma_sp)*H

        

        

