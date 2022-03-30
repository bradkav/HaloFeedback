# HaloFeedback
import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import simpson
from scipy.special import ellipeinc, ellipkinc, ellipe, betainc
from scipy.special import gamma as Gamma
from scipy.special import beta as Beta

# ------------------
G_N = 4.300905557082141e-3  # [(km/s)^2 pc/M_sun] [Legacy: 4.3021937e-3]
c = 299792.458 # [km/s] [Legacy: 2.9979e5]

# Conversion factors
pc_to_km = 3.085677581491367e13 # [km] [Legacy: 3.085677581e13]

# Numerical parameters
N_GRID = 10000  # Number of grid points in the specific energy.
N_KICK = 50  # Number of points to use for integration over Delta-epsilon. [Legacy: 50]

float_2eps = 2.0 * np.finfo(float).eps
# ------------------

def ellipeinc_alt(phi, m):
    """ An alternative elliptic function that is valid for m > 1."""
    beta = np.arcsin(np.clip(np.sqrt(m) * np.sin(phi), 0, 1))
    return np.sqrt(m) * ellipeinc(beta, 1 / m) + ((1 - m) / np.sqrt(m)) * ellipkinc(beta, 1 / m)


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

    def __init__(self, m1: float = 1e3, m2: float = 1.0, mDM: float = 0):
        self.m1 = m1  # [M_sun]
        self.m2 = m2  # [M_sun]
        self.mDM = mDM # [M_sun]

        self.r_isco = 6.0 * G_N * m1 / c ** 2

        # Initialise grid of r, eps and f(eps) and append an extra loose grid far away.
        self.r_grid = np.geomspace(self.r_isco, 1e5 * self.r_isco, int(0.9 *N_GRID))
        self.r_grid = np.append(
            self.r_grid, np.geomspace(1.01 * self.r_grid[-1], 1e3 * self.r_sp, int(0.1*N_GRID))
        )
        self.r_grid = np.sort(self.r_grid)
        self.eps_grid = self.psi(self.r_grid)
        self.f_eps = self.f_init(self.eps_grid)

        # Density of states
        self.DoS = (
            np.sqrt(2) * (np.pi * G_N * self.m1) ** 3 * self.eps_grid ** (-5/2)
        )

        # Define a string which specifies the model parameters
        # and numerical parameters (for use in file names etc.)
        self.IDstr_num = "lnLambda=%.1f" % (np.log(np.sqrt(m2/m1)),)

    @abstractmethod
    def rho_init(self, r):
        """ The initial dark matter density [M_sun/pc^3] of the system at distance r from the
        halo center.

        Parameters:
            - r : distance [pc] from center of spike.
        """
        pass

    @abstractmethod
    def f_init(self, eps):
        """ The initial phase-space distribution function at energy eps.

        Parameters
            - eps : float or np.array Energy per unit mass in (km/s)^2
        """
        pass

    def plotDF(self):
        """ Plots the initial and current distribution function of the spike. """
        plt.figure()
        
        plt.loglog(self.eps_grid, self.f_init(self.eps_grid), "k--", label = "Initial DF")
        plt.loglog(self.eps_grid, self.f_eps)
        
        plt.ylabel(r"$f(\mathcal{E})$ [$M_\odot$ pc$^{-3}$ (km/s)$^{-3}$]")
        plt.xlabel(r"$\mathcal{E} = \Psi(r) - \frac{1}{2}v^2$ [(km/s)$^2$]")
        plt.legend()
        plt.show()
        
        return plt.gca()

    def psi(self, r: float) -> float:
        """ The gravitational potential [km^2/s^2] at distance r [pc]."""
        return G_N *self.m1 /r # [km^2/s^2]

    def v_max(self, r: float) -> float:
        """ The maximum velocity [km/s] allowed for bound orbits in the system at position r [pc]."""
        return np.sqrt(2 * self.psi(r)) # [km/s]

    def rho(self, r: float, v_cut: float = -1) -> float:
        """ Returns the local density [M_sun/pc^3] of the dark matter particles at position
        r [pc] from the halo center, that move slower than v_cut [km/s].

        Parameters:
            - r: The distance from the dark matter halo center.
            - v_cut : maximum speed to include in density calculation
                     (defaults to v_max if not specified)
        """
        if v_cut < 0: v_cut = self.v_max(r)

        v_cut = np.clip(v_cut, 0, self.v_max(r))
        vlist = np.sqrt(np.linspace(0, v_cut ** 2, 20000))
        
        # Interpolate the integrand onto the new array vlist.
        flist = np.interp(self.psi(r) - 0.5 * vlist ** 2,
            self.eps_grid[::-1], self.f_eps[::-1],
            left = 0, right = 0,
        )
        integ = vlist ** 2 * flist
        return 4 * np.pi *simpson(integ, vlist) # [M_sun/pc^3]

    def averageVelocity(self, r: float) -> float:
        """ Returns the local average velocity [km/s] <u> from the velocity distribution of the
        dark matter particles at position r [pc] from the halo center.
        """
        v_cut = self.v_max(r)

        # Interpolate the integrand onto the new array vlist.
        v_cut = np.clip(v_cut, 0, self.v_max(r))
        vlist = np.sqrt(np.linspace(0, v_cut**2, 250))
        flist = np.interp(self.psi(r) -0.5 *vlist **2,
            self.eps_grid[::-1], self.f_eps[::-1],
            left = 0, right = 0,
        )
        
        integ = vlist ** 3 * flist
        return np.sqrt(np.trapz(integ, vlist) / np.trapz(vlist ** 2 * flist, vlist)) # [km/s]
    
    def averageSquaredVelocity(self, r: float) -> float:
        """ Returns the local average squared velocity [km/s] <u^2> (or root mean squared velocity) from the velocity distribution of the
        dark matter particles at position r [pc] from the halo center.
        """
        v_cut = self.v_max(r)

        # Interpolate the integrand onto the new array vlist.
        v_cut = np.clip(v_cut, 0, self.v_max(r))
        vlist = np.sqrt(np.linspace(0, v_cut**2, 250))
        flist = np.interp(self.psi(r) -0.5 *vlist **2,
            self.eps_grid[::-1], self.f_eps[::-1],
            left = 0, right = 0,
        )
        
        integ = vlist ** 4 * flist
        return np.sqrt(np.trapz(integ, vlist) / np.trapz(vlist ** 2 * flist, vlist)) # [km/s]

    def velocityDispersion(self, r: float) -> float:
        """ Returns the local velocity dispersion [km/s] from the velocity distribution of the dark matter
        particles at position r [pc] from the halo center.
        """
        u2 = self.averageSquaredVelocity(r)
        u = self.averageSquaredVelocity(r)
        
        return np.sqrt(u2 -u**2) # [km/s]
    
    def m(self) -> float:
        """ The total mass [M_sun] of the binary system. """
        return self.m1 +self.m2 # [M_sun]
    
    def mu(self) -> float:
        """ The reduced mass [M_sun] of the binary system. """
        return self.m1 *self.m2 /self.m() # [M_sun]

    def totalMass(self) -> float:
        """ The total mass of dark matter particles in the halo. """
        return simpson(-self.P_eps(), self.eps_grid)

    def totalEnergy(self) -> float:
        """ The total energy of the dark matter halo. """
        return simpson(-self.P_eps() * self.eps_grid, self.eps_grid)

    def b_90(self, r2: float, Delta_u: float) -> float:
        """ The impact parameter [pc] at which dark matter particles are deflected at a 90 degree angle.
            Delta_u relative velocity of the orbiting body and dark matter particles, usually set at u_orb
            of the companion object m2.
        """
        return G_N *(self.m2 +self.mDM) / (Delta_u ** 2) # [pc]

    def b_min(self, r2: float, v_orb: float) -> float:
        """ The minimum impact parameter [pc] is the radius of the companion m2. """
        return self.R/pc_to_km if self.R != -1 else 6.0 * G_N * self.m2/ c ** 2 # [pc]

    def b_max(self, r2: float, v_orb: float = -1) -> float:
        """ The maximum impact parameter [pc] as calculated from gravitational force equivalance O(sqrt(q)).
        
        Parameters:
            - r2 is the separation [pc] of the two components.
            - v_orb is the instant velocity [km/s] of the orbiting body. If not specified, defaults to circular orbital velocity.
        """
        
        if v_orb == -1: v_orb = np.sqrt(G_N * (self.m1 + self.m2) / r2) # [km/s]
        
        return np.sqrt(self.m2/self.m1) *r2 # [pc]

    def Lambda(self, r2: float, v_orb: float = -1) -> float:
        """ The coulomb logarithm of the dynamical friction force induced by the dark matter particles.
         
        Parameters:
            - r2 is the separation [pc] of the two components.
            - v_orb is the instant velocity [km/s] of the orbiting body. If not specified, defaults to circular orbital velocity.
        """
        
        if v_orb == -1: v_orb = np.sqrt(G_N * (self.m1 + self.m2) / r2) # [km/s]
        
        b90 = self.b_90(r2, v_orb) # [pc]
        
        return np.sqrt((self.b_max(r2, v_orb)**2 +b90**2)/(self.b_min(r2, v_orb)**2 +b90**2))

    def eps_min(self, r2: float, v_orb: float) -> float:
        """ The minimum energy for the average delta_eps calculation in calc_delta_eps()."""
        return 2 * v_orb ** 2 / (1 + self.b_max(r2, v_orb) ** 2 / self.b_90(r2, v_orb) ** 2)

    def eps_max(self, r2: float, v_orb: float) -> float:
        return 2 * v_orb ** 2 / (1 + self.b_min(r2, v_orb) ** 2 / self.b_90(r2, v_orb) ** 2)


    def df(self, r2: float, v_orb: float, v_cut: float = -1) -> np.array:
        """The change of the distribution function f(eps) during an orbit.

        Parameters:
            - r2 is the radial position [pc] of the perturbing body.
            - v_orb is the orbital velocity [km/s] of the perturbing body.
            - v_cut (optional), only scatter with particles slower than v_cut [km/s]
                    defaults to v_max(r) (i.e. all particles).
        """
        
        df_minus = self.df_minus(r2, v_orb, v_cut, N_KICK)
        df_plus = self.df_plus(r2, v_orb, v_cut, N_KICK)
        
        # TODO: What is this meant for?
        N_plus = 1 # np.trapz(self.DoS*f_plus, self.eps_grid)
        N_minus = 1 # np.trapz(-self.DoS*f_minus, self.eps_grid)
        
        return df_minus + df_plus *(N_minus/N_plus)
    
    def dfdt(self, r2: float, v_orb: float, v_cut: float = -1) -> np.array:
        """Time derivative of the distribution function f(eps).

        Parameters:
            - r2 is the radial position [pc] of the perturbing body.
            - v_orb is the orbital velocity [km/s] of the perturbing body.
            - v_cut (optional), only scatter with particles slower than v_cut [km/s]
                    defaults to v_max(r) (i.e. all particles).
        """
        T_orb = self.T_orb(r2) # [s]
        
        return self.df(r2, v_orb, v_cut) /T_orb

    def delta_f(self, r0: float, v_orb: float, dt: float, v_cut: float = -1) -> np.array:
        """[Deprecated] This shouldn't be used in new applications. TODO: Remove?
        
        Change in f over a time-step dt where it is automatically
        adjusted to prevent f_eps from becoming negative.

        Parameters:
            - r2 is the radial position [pc] of the perturbing body.
            - v_orb is the orbital velocity [km/s] of the perturbing body.
            - dt: time-step [s]
            - v_cut (optional), only scatter with particles slower than v_cut [km/s]
                    defaults to v_max(r) (i.e. all particles).
        """

        f_minus = self.dfdt_minus(r0, v_orb, v_cut, N_KICK) * dt

        # Don't remove more particles than there are particles...
        correction = np.clip(self.f_eps / (-f_minus + 1e-50), 0, 1)
        
        f_minus = np.clip(f_minus, -self.f_eps, 0)
        f_plus = self.dfdt_plus(r0, v_orb, v_cut, N_KICK, correction) * dt


        return f_minus + f_plus

    def P_delta_eps(self, r: float, v: float, delta_eps: float) -> float:
        """ Calcuate PDF for delta_eps. """
        norm = self.b_90(r, v) ** 2 / (self.b_max(r, v) ** 2 - self.b_min(r, v) ** 2)
        
        return 2 * norm * v ** 2 / (delta_eps ** 2)

    def P_eps(self):
        """Calculate the PDF d{P}/d{eps}"""
        return (
            np.sqrt(2)
            * np.pi ** 3
            * (G_N * self.m1) ** 3
            * self.f_eps
            / self.eps_grid ** 2.5
        )

    def calc_delta_eps(self, r: float, v: float, n_kick: int = 1) -> list:
        """ Calculate average delta_eps integrated over different bins (and the corresponding
        fraction of particles which scatter with that delta_eps).
        """
        eps_min = self.eps_min(r, v)
        eps_max = self.eps_max(r, v)

        norm = self.b_90(r, v) ** 2 / (self.b_max(r, v) ** 2 - self.b_min(r, v) ** 2)

        eps_edges = np.linspace(eps_min, eps_max, n_kick + 1)

        def F_norm(eps):
            return -norm * 2 * v ** 2 / (eps)

        def F_avg(eps):
            return -norm * 2 * v ** 2 * np.log(eps)

        frac = np.diff(F_norm(eps_edges))
        eps_avg = np.diff(F_avg(eps_edges)) / frac

        return eps_avg, frac
 
    def dEdt_DF(self, r: float, v_orb: float = -1, v_cut: float = -1, average: bool = False) -> float:
        """Rate of change of energy due to DF (km/s)^2 s^-1 M_sun.
        
        Parameters:
            - r is the radial position of the perturbing body [pc]
            - v_orb the velocity [km/s] of the body, when not given assume circular Keplerian orbits.
            - v_cut (optional), only scatter with particles slower than v_cut [km/s]
                        defaults to v_max(r) (i.e. all particles)
            - average determines whether to average over different radii
                        (average = False is default and should be correct).
        
        """
        if v_orb < 0: v_orb = np.sqrt(G_N * (self.m1 + self.m2) / r) # [km/s]

        if average:
            warnings.warn(
                "Setting 'average = True' is not necessarily the right thing to do..."
            )
            r_list = r + np.linspace(-1, 1, 3) * self.b_max(r, v_orb)
            rho_list = np.array([self.rho(r1, v_cut) for r1 in r_list])
            rho_eff = np.trapz(rho_list * r_list, r_list) / np.trapz(r_list, r_list)
        else:
            rho_eff = self.rho(r, v_cut)

        return 4 *np.pi * G_N **2 * self.m2 *(self.m2 +self.mDM) * rho_eff * np.log(self.Lambda(r, v_orb)) / v_orb /pc_to_km # [km]

    def E_orb(self, a: float) -> float:
        """ The orbital energy of the binary system at semi-major axis [pc]. """
        return -0.5 * G_N * (self.m1 + self.m2) / a

    def T_orb(self, a: float) -> float:
        """ The orbital period of the binary system at semi-major axis [pc]. """
        return (2 * np.pi * np.sqrt(pc_to_km ** 2 * a ** 3 / (G_N * (self.m1 + self.m2))) ) # [s]

    def interpolate_DF(self, eps_old, correction = 1):
        """ Internal function for interpolating the DF on df_plus calculations. """
        
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

    def delta_eps_of_b(self, r2: float, v_orb: float, b: float) -> float:
        """ The change of energy based on the impact parameter of the scattering. """
        b90 = self.b_90(r2, v_orb) # [pc]
        
        return -2 * v_orb ** 2 * (1 + b**2 / b90**2) ** -1

    # ---------------------
    # ----- df/dt      ----
    # ---------------------

    def df_minus(self, r0: float, v_orb: float, v_cut: float = -1, n_kick: int = 1) -> np.array:
        """Particles to remove from the distribution function at energy E. """
        
        if v_cut < 0: v_cut = self.v_max(r0)

        df = np.zeros(N_GRID)

        # Calculate sizes of kicks and corresponding weights for integration
        if n_kick == 1:  # Replace everything by the average if n_kick = 1
            delta_eps_list = (
                -2 * v_orb ** 2 * np.log(1 + self.Lambda(r0, v_orb) ** 2) / self.Lambda(r0, v_orb) ** 2,
            )
            frac_list = (1,)
        else:
            b_list = np.geomspace(self.b_min(r0, v_orb), self.b_max(r0, v_orb), n_kick)
            delta_eps_list = self.delta_eps_of_b(r0, v_orb, b_list)

            # Step size for trapezoidal integration
            step = delta_eps_list[1:] - delta_eps_list[:-1]
            step = np.append(step, 0)
            step = np.append(0, step)

            # Make sure that the integral is normalised correctly
            renorm = np.trapz(self.P_delta_eps(r0, v_orb, delta_eps_list), delta_eps_list)
            frac_list = 0.5 * (step[:-1] + step[1:]) / renorm

        # Sum over the kicks
        for delta_eps, b, frac in zip(delta_eps_list, b_list, frac_list):
            # Define which energies are allowed to scatter
            mask = (self.eps_grid > self.psi(r0) * (1 - b / r0) - 0.5 * v_cut ** 2) & (
                self.eps_grid < self.psi(r0) * (1 + b / r0)
            )


            r_eps = G_N * self.m1 / self.eps_grid[mask]
            r_cut = G_N * self.m1 / (self.eps_grid[mask] + 0.5 * v_cut ** 2)

            L1 = np.minimum((r0 - r0 ** 2 / r_eps) / b, 0.999999)
            alpha1 = np.arccos(L1)
            L2 = np.maximum((r0 - r0 ** 2 / r_cut) / b, -0.999999)
            alpha2 = np.arccos(L2)

            m = (2 * b / r0) / (1 - (r0 / r_eps) + b / r0)
            mask1 = (m <= 1) & (alpha2 > alpha1)
            mask2 = (m > 1) & (alpha2 > alpha1)


            N1 = np.zeros(len(m))
            if np.any(mask1):
                N1[mask1] = ellipe(m[mask1]) - ellipeinc(
                    (np.pi - alpha2[mask1]) / 2, m[mask1]
                )
            if np.any(mask2):
                N1[mask2] = ellipeinc_alt((np.pi - alpha1[mask2]) / 2, m[mask2])
            df[mask] += (
                -frac
                * self.f_eps[mask]
                * (1 + b ** 2 / self.b_90(r0, v_orb) ** 2) ** 2
                * np.sqrt(1 - r0 / r_eps + b / r0)
                * N1
            )

        norm = (
            2
            * np.sqrt(2 * (self.psi(r0)))
            * 4
            * np.pi ** 2
            * r0
            * (self.b_90(r0, v_orb) ** 2 / (v_orb) ** 2)
        )
        result = norm * df / self.DoS
        result[self.eps_grid >= 0.9999 *self.psi(self.r_isco)] *= 0
        
        return result

    def df_plus(self, r0: float, v_orb: float, v_cut: float = -1, n_kick: int = 1, correction = 1) -> np.array:
        """Particles to add back into distribution function from E - dE -> E. """
        
        if v_cut < 0: v_cut = self.v_max(r0)

        df = np.zeros(N_GRID)

        # Calculate sizes of kicks and corresponding weights for integration
        if n_kick == 1:  # Replace everything by the average if n_kick = 1
            delta_eps_list = (
                -2 * v_orb ** 2 * np.log(1 + self.Lambda(r0, v_orb) ** 2) / self.Lambda(r0, v_orb) ** 2,
            )
            frac_list = (1,)
        else:
            b_list = np.geomspace(self.b_min(r0, v_orb), self.b_max(r0, v_orb), n_kick)
            delta_eps_list = self.delta_eps_of_b(r0, v_orb, b_list)

            # Step size for trapezoidal integration
            step = delta_eps_list[1:] - delta_eps_list[:-1]
            step = np.append(step, 0)
            step = np.append(0, step)

            # Make sure that the integral is normalised correctly
            renorm = np.trapz(self.P_delta_eps(r0, v_orb, delta_eps_list), delta_eps_list)
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
            if np.any(mask):
                r_eps = G_N * self.m1 / eps_old[mask]
                r_cut = G_N * self.m1 / (eps_old[mask] + 0.5 * v_cut ** 2)

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
                if np.any(mask1):
                    N1[mask1] = ellipe(m[mask1]) - ellipeinc(
                        (np.pi - alpha2[mask1]) / 2, m[mask1]
                    )
                if np.any(mask2):
                    N1[mask2] = ellipeinc_alt(
                        (np.pi - alpha1[mask2]) / 2, m[mask2]
                    )  # - ellipeinc_alt((np.pi - alpha2[mask2])/2, m[mask2])

                df[mask] += (
                    frac
                    * f_old
                    * (1 + b ** 2 / self.b_90(r0, v_orb) ** 2) ** 2
                    * np.sqrt(1 - r0 / r_eps + b / r0)
                    * N1
                )
        norm = (
            2
            * np.sqrt(2 * (self.psi(r0)))
            * 4
            * np.pi ** 2
            * r0
            * (self.b_90(r0, v_orb) ** 2 / (v_orb) ** 2)
        )
        result = norm * df / self.DoS
        result[self.eps_grid >= 0.9999 *self.psi(self.r_isco)] *= 0
        
        return result

    def dEdt_ej(self, r0: float, v_orb: float, v_cut: float = -1, n_kick: int = N_KICK, correction = np.ones(N_GRID)):
        """Calculate carried away by particles which are completely unbound.

        Parameters:
            - r0 : radial position of the perturbing body [pc]
            - v_orb: orbital velocity [km/s]
            - v_cut: optional, only scatter with particles slower than v_cut [km/s]
                        defaults to v_max(r) (i.e. all particles)
            - n_kick: optional, number of grid points to use when integrating over
                        Delta-eps (defaults to N_KICK = 100).
        """
        if v_cut < 0: v_cut = self.v_max(r0)

        T_orb = (2 * np.pi * r0 * pc_to_km) / v_orb

        dE = np.zeros(N_GRID)

        # Calculate sizes of kicks and corresponding weights for integration
        if n_kick == 1:  # Replace everything by the average if n_kick = 1
            delta_eps_list = (
                -2 * v_orb ** 2 * np.log(1 + self.Lambda(r0, v_orb) ** 2) / self.Lambda(r0, v_orb) ** 2,
            )
            frac_list = (1,)

        else:
            b_list = np.geomspace(self.b_min(r0, v_orb), self.b_max(r0, v_orb), n_kick)
            delta_eps_list = self.delta_eps_of_b(r0, v_orb, b_list)

            # Step size for trapezoidal integration
            step = delta_eps_list[1:] - delta_eps_list[:-1]
            step = np.append(step, 0)
            step = np.append(0, step)

            # Make sure that the integral is normalised correctly
            renorm = np.trapz(self.P_delta_eps(r0, v_orb, delta_eps_list), delta_eps_list)
            frac_list = 0.5 * (step[:-1] + step[1:]) / renorm

        # Sum over the kicks
        for delta_eps, b, frac in zip(delta_eps_list, b_list, frac_list):

            # Maximum impact parameter which leads to the ejection of particles
            b_ej_sq = self.b_90(r0, v_orb) ** 2 * ((2 * v_orb ** 2 / self.eps_grid) - 1)

            # Define which energies are allowed to scatter
            mask = (
                (self.eps_grid > self.psi(r0) * (1 - b / r0) - 0.5 * v_cut ** 2)
                & (self.eps_grid < self.psi(r0) * (1 + b / r0))
                & (b ** 2 < b_ej_sq)
            )

            r_eps = G_N * self.m1 / self.eps_grid[mask]
            r_cut = G_N * self.m1 / (self.eps_grid[mask] + 0.5 * v_cut ** 2)

            if np.any(mask):

                L1 = np.minimum((r0 - r0 ** 2 / r_eps) / b, 0.999999)
                alpha1 = np.arccos(L1)
                L2 = np.maximum((r0 - r0 ** 2 / r_cut) / b, -0.999999)
                alpha2 = np.arccos(L2)

                m = (2 * b / r0) / (1 - (r0 / r_eps) + b / r0)
                mask1 = (m <= 1) & (alpha2 > alpha1)
                mask2 = (m > 1) & (alpha2 > alpha1)

                N1 = np.zeros(len(m))
                if np.any(mask1):
                    N1[mask1] = ellipe(m[mask1]) - ellipeinc(
                        (np.pi - alpha2[mask1]) / 2, m[mask1]
                    )
                if np.any(mask2):
                    N1[mask2] = ellipeinc_alt((np.pi - alpha1[mask2]) / 2, m[mask2])

                dE[mask] += (
                    -frac
                    * correction[mask]
                    * self.f_eps[mask]
                    * (1 + b ** 2 / self.b_90(r0, v_orb) ** 2) ** 2
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
            * (self.b_90(r0, v_orb) ** 2 / (v_orb) ** 2)
        )
        return norm * np.trapz(dE, self.eps_grid) / T_orb


class PowerLawSpike(DistributionFunction):
    """ A spike with a power law profile:

        rho(r) = rho_sp * (r_sp / r)^gamma.

    The parameter r_sp is defined as r_sp = 0.2 r_h, where r_h is the radius of
    the sphere within which the DM mass is twice the central BH mass.

    Notes
    -----
    The parameters are not properties, so r_sp will not have the correct value
    if rho_sp or gamma are changed after initialization.
    """

    def __init__(self, m1: float = 1e3, m2: float = 1.0, gamma: float = 7/3, rho_sp: float = 226, R: float = -1, mDM: float = 0):
        if gamma <= 1: raise ValueError("Slope must be greater than 1")
        
        self.m1 = m1  # [M_sun]
        self.m2 = m2  # [M_sun]
        self.mDM = mDM # [M_sun]
        
        self.gamma = gamma  # Slope of DM density profile
        self.rho_sp = rho_sp  # [M_sun/pc^3]
        
        self.R = R # [km] The radius of m2.
        self.r_sp = (
            (3 - self.gamma)
            * (0.2 ** (3 - self.gamma))
            * self.m1
            / (2 * np.pi * self.rho_sp)
        ) ** (1/3)  # [pc]

        self.IDstr_model = f"gamma={gamma:.2f}_rhosp={rho_sp:.1f}"

        self.xi_init = 1 - betainc(gamma - 1 / 2, 3 / 2, 1 / 2)

        super().__init__(m1, m2, mDM)

    def f_init(self, eps):
        A1 = self.r_sp / (G_N * self.m1)
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
    """ A spike with no DM particles whose orbits are completely contained within
    an annihilation plateau of radius r_p:

        rho(r) =
            rho_s (r_s / r)^gamma,                   r_s > r > r_p
            rho_s (r_s / r_p)^gamma (r_p / r)^{1/2}, r_p > r

    The parameter r_sp is defined as r_sp = 0.2 r_h, where r_h is the radius of
    the sphere within which the DM mass is twice the central BH mass.

    Notes
    -----
    The parameters are not properties, so r_sp will not have the correct value
    if rho_sp or gamma are changed after initialization.
    """

    def __init__(self, m1: float = 1e3, m2: float = 1.0, gamma: float = 7/3, rho_sp: float = 226, r_p: float = 0.0, R: float = -1, mDM: float = 0):
        self.m1 = m1  # [M_sun]
        self.m2 = m2  # [M_sun]
        self.mDM = mDM # [M_sun]
        
        self.gamma = gamma  # Slope of DM density profile
        self.rho_sp = rho_sp  # M_sun/pc^3
        self.R = R # [km] The radius of m2.
        self.r_sp = (
            (3 - self.gamma)
            * (0.2 ** (3.0 - self.gamma))
            * self.m1
            / (2 * np.pi * self.rho_sp)
        ) ** (
            1.0 / 3.0
        )  # pc

        if gamma <= 1: raise ValueError("The slope must be greater than 1")

        if r_p > self.r_sp: raise ValueError("The annihilation plateau radius shouldn't be larger than spike.")

        self.r_p = r_p
        self.IDstr_model = f"gamma={gamma:.2f}_rhosp={rho_sp:.1f}_rp={r_p:.2E}"

        super().__init__(m1, m2, mDM)

    def f_init(self, eps):
        def f_init(eps):
            if G_N * self.m1 / self.r_sp < eps and eps <= G_N * self.m1 / self.r_p:
                return (
                    self.rho_sp
                    * ((eps * self.r_sp) / (G_N * self.m1)) ** self.gamma
                    * (
                        (1 - self.gamma)
                        * self.gamma
                        * Beta(
                            -1 + self.gamma, 0.5, (G_N * self.m1) / (eps * self.r_sp)
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