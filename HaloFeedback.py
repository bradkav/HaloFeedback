#HaloFeedback
import numpy as np
from scipy.special import gamma as Gamma_func
from scipy.special import ellipeinc, ellipkinc
from scipy.interpolate import interp1d


from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate import quad

import matplotlib.pyplot as plt

from time import time as timeit
import time

import warnings

#------------------
G_N = 4.302e-3 #(km/s)^2 pc/M_sun
c = 2.99792458e5 #km/s

#Coulomb factor
#Lambda = np.exp(3.0)

#Conversion factors
pc_to_km = 3.0857e13

#Numerical parameters
N_GRID = 20000  #Number of grid points in the specific energy
N_KICK = 100    #Number of points to use for integration over Delta-epsilon

float_2eps = 2. * np.finfo(float).eps

#------------------

#Alternative elliptic function which is valid for m > 1
def ellipeinc_alt(phi, m):
    beta = np.arcsin(np.clip(np.sqrt(m)*np.sin(phi), 0, 1))
    #print(np.min(np.sqrt(m)*np.sin(phi)), np.max(np.sqrt(m)*np.sin(phi)))
    return np.sqrt(m)*ellipeinc(beta, 1/m) + ((1-m)/np.sqrt(m))*ellipkinc(beta,1/m)


class DistributionFunction():
    def __init__(self, M_BH=1e3, M_NS = 1.0, gamma=7./3., rho_sp=226, Lambda=-1):
        self.M_BH = M_BH    #Solar mass
        self.M_NS = M_NS    #Solar mass
        self.gamma = gamma  #Slope of DM density profile
        self.rho_sp = rho_sp    #Solar mass/pc^3
        
        if (Lambda <= 0):
            self.Lambda = np.sqrt(M_BH/M_NS)
        else:
            self.Lambda = Lambda
        
        #Spike radius and ISCO
        self.r_sp = ((3-gamma)*(0.2**(3.0-gamma))*M_BH/(2*np.pi*rho_sp))**(1.0/3.0) #pc
        self.r_isco = 6.0*G_N*M_BH/c**2
        
        #Initialise grid of r, eps and f(eps)
        self.r_grid = np.geomspace(self.r_isco, 1e-3*self.r_sp, N_GRID-100)
        self.r_grid = np.append(self.r_grid, np.geomspace(1.01*self.r_grid[-1], 1e3*self.r_sp, 100))
        self.eps_grid = self.psi(self.r_grid)
    
        self.f_eps = self.f_init()

        #Define a string which specifies the model parameters
        #and numerical parameters (for use in file names etc.)
        self.IDstr_num = "lnLambda=%.1f"%(np.log(self.Lambda),)
        self.IDstr_model = "gamma=%.2f_rhosp=.%1f"%(gamma, rho_sp)
        
        
    def f_init(self):
        A1 = (self.r_sp/(G_N*self.M_BH))
        return self.rho_sp*(self.gamma*(self.gamma - 1)*A1**self.gamma*np.pi**-1.5/np.sqrt(8))*(Gamma_func(-1 + self.gamma)/Gamma_func(-1/2 + self.gamma))*self.eps_grid**(-(3/2) + self.gamma) 
    
    def psi(self, r):
        """Gravitational potential as a function of r"""
        return G_N*self.M_BH/r
        
    def v_max(self, r):
        """Maximum velocity as a function of r"""
        return np.sqrt(2*self.psi(r))
    
    def rho(self, r, v_cut=-1):
        """DM mass density computed from f(eps).
        
        Parameters: 
            - r : radius in pc
            - v_cut : maximum speed to include in density calculation
                     (defaults to v_max if not specified)
        """
        if (v_cut < 0):
            v_cut = self.v_max(r)
            
        v_cut = np.clip(v_cut, 0, self.v_max(r))
        vlist = np.sqrt(np.linspace(0, v_cut**2, 500))
        flist = np.interp(self.psi(r) - 0.5*vlist**2, self.eps_grid[::-1], self.f_eps[::-1], left=0, right=0)
        integ = vlist**2*flist
        return 4*np.pi*np.trapz(integ, vlist)
        
        
    def sigma_v(self, r):
        #if (v_cut < 0):
        v_cut = self.v_max(r)
            
        v_cut = np.clip(v_cut, 0, self.v_max(r))
        vlist = np.sqrt(np.linspace(0, v_cut**2, 500))
        flist = np.interp(self.psi(r) - 0.5*vlist**2, self.eps_grid[::-1], self.f_eps[::-1], left=0, right=0)
        integ = vlist**4*flist
        return np.sqrt(np.trapz(integ, vlist)/np.trapz(vlist**2*flist, vlist))

    def rho_init(self, r):
        """Initial DM density of the system"""
        return self.rho_sp*(r/self.r_sp)**-self.gamma

    def TotalMass(self):
        return np.trapz(-self.P_eps(), self.eps_grid)

    def TotalEnergy(self):
        return np.trapz(-self.P_eps()*self.eps_grid, self.eps_grid)
        
    #Density of states
    def DoS(self):
        return np.sqrt(2)*(np.pi*G_N*self.M_BH)**3*self.eps_grid**(-5/2.)




    def b_90(self, v_orb):
        return G_N*self.M_NS/(v_orb**2)
        
    def b_min(self, v_orb):
        return 15./pc_to_km
        
    def b_max(self, v_orb):
        return self.Lambda*np.sqrt(self.b_90(v_orb)**2 + self.b_min(v_orb)**2)
        
    def eps_min(self, v_orb):
        return 2*v_orb**2/(1 + self.b_max(v_orb)**2/self.b_90(v_orb)**2)
    
    def eps_max(self, v_orb):
        return 2*v_orb**2/(1 + self.b_min(v_orb)**2/self.b_90(v_orb)**2)
    
    
    
    
    def dfdt(self, r0, v_orb, v_cut=-1):
        """Time derivative of the distribution function f(eps).
        
        Parameters:
            - r0 : radial position of the perturbing body [pc]
            - v_orb: orbital velocity [km/s]
            - v_cut: optional, only scatter with particles slower than v_cut [km/s]
                        defaults to v_max(r) (i.e. all particles)
        """
    
        return self.dfdt_minus(r0, v_orb, v_cut, N_KICK)  + self.dfdt_plus(r0, v_orb, v_cut, N_KICK)
    
    

    def P_delta_eps(self, v, delta_eps):
        """
        Calcuate PDF for delta_eps
        """  
        norm = self.b_90(v)**2/(self.b_max(v)**2 - self.b_min(v)**2)
        return 2*norm*v**2/(delta_eps**2)
        
        
    def P_eps(self):
        """Calculate the PDF d{P}/d{eps}"""
        return np.sqrt(2)*np.pi**3*(G_N*self.M_BH)**3*self.f_eps/self.eps_grid**2.5
        
    def calc_delta_eps(self, v, n_kick=1):
        """
        Calculate average delta_eps integrated over different
        bins (and the corresponding fraction of particles which
        scatter with that delta_eps).
        """
        eps_min = self.eps_min(v)
        eps_max = self.eps_max(v)
        
        norm = self.b_90(v)**2/(self.b_max(v)**2 - self.b_min(v)**2)
        
        eps_edges = np.linspace(eps_min, eps_max, n_kick+1)
        
        def F_norm(eps):
            return -norm*2*v**2/(eps)
            
        def F_avg(eps):
            return -norm*2*v**2*np.log(eps)
            
        frac = np.diff(F_norm(eps_edges))
        eps_avg = np.diff(F_avg(eps_edges))/frac
        
        #*1.000450641
        return eps_avg, frac
        
        
    def dEdt_DF(self, r, SPEED_CUT = False, average = False):
        """Rate of change of energy due to DF (km/s)^2 s^-1 M_sun"""
        v_orb = np.sqrt(self.psi(r))
        
        if (SPEED_CUT):
            v_cut = v_orb
        else:
            v_cut = -1
            
        #CoulombLog = 0.5*np.log((1 + self.b_max(v_orb)**2/self.b_90(v_orb)**2)/(1 + self.b_min(v_orb)**2/self.b_90(v_orb)**2))
        CoulombLog = np.log(self.Lambda)
            
        if (average):
            warnings.warn("Setting 'average = True' is not necessarily the right thing to do...")
            r_list = r + np.linspace(-1, 1, 3)*self.b_max(v_orb)
            rho_list = np.array([self.rho(r1, v_cut) for r1 in r_list])
            rho_eff = np.trapz(rho_list*r_list, r_list)/np.trapz(r_list, r_list)
        else:
            rho_eff = self.rho(r, v_cut)
        
            
        return (1/pc_to_km)*4*np.pi*G_N**2*self.M_NS**2*rho_eff*CoulombLog/v_orb

    def E_orb(self,r):
        return -0.5*G_N*(self.M_BH + self.M_NS)/r
        
    def T_orb(self,r):
        return 2*np.pi*np.sqrt(pc_to_km**2*r**3/(G_N*(self.M_BH + self.M_NS)))
        

    def interpolate_DF(self, eps_old):
        # Distribution of particles before they scatter
        f_old = np.interp(eps_old[::-1], self.eps_grid[::-1], self.f_eps[::-1], left=0, right=0)[::-1]
        return f_old


    def delta_eps_of_b(self, v_orb, b):
        b90 = self.b_90(v_orb)
        return -2*v_orb**2*(1 + b**2/b90**2)**-1

            
#---------------------
#----- df/dt      ----
#---------------------
    
             
    def dfdt_minus(self, r0, v_orb, v_cut=-1, n_kick = 1):
        """Particles to remove from the distribution function at energy E."""
        if (v_cut < 0):
            v_cut = self.v_max(r0)
        
        df = np.zeros(N_GRID)
        
        
        # Calculate sizes of kicks and corresponding weights for integration
        if (n_kick == 1):   #Replace everything by the average if n_kick = 1
            delta_eps_list = (-2*v_orb**2*np.log(1+self.Lambda**2)/self.Lambda**2,)
            frac_list = (1,)
      
        else:
           b_list = np.geomspace(self.b_min(v_orb), self.b_max(v_orb), n_kick)
           delta_eps_list = self.delta_eps_of_b(v_orb, b_list)
           
           #Step size for trapezoidal integration
           step = delta_eps_list[1:] - delta_eps_list[:-1]
           step = np.append(step, 0)
           step = np.append(0, step)
       
           #Make sure that the integral is normalised correctly
           renorm = np.trapz(self.P_delta_eps(v_orb, delta_eps_list), delta_eps_list)
           frac_list = 0.5*(step[:-1] + step[1:])/renorm
       
        #Sum over the kicks
        for delta_eps, b, frac in zip(delta_eps_list, b_list, frac_list):

            r_eps = G_N*self.M_BH/self.eps_grid
            r_cut = G_N*self.M_BH/(self.eps_grid + 0.5*v_cut**2)
        
            #Define which energies are allowed to scatter
            mask = (self.eps_grid > self.psi(r0)*(1-b/r0) - 0.5*v_cut**2) & (self.eps_grid < self.psi(r0)*(1+b/r0))
        
            L1 = np.minimum((r0 - r0**2/r_eps[mask])/b, 1)
            alpha1 = np.arccos(L1)
            L2 = np.maximum((r0 - r0**2/r_cut[mask])/b, -1)
            alpha2 = np.arccos(L2)

            m = (2*b/r0)/(1 - (r0/r_eps[mask]) + b/r0)
            mask1 = (m <= 1) & (alpha2 > alpha1)
            mask2 = (m > 1) & (alpha2 > alpha1)
            N1 = np.zeros(len(m))
            N1[mask1] = (ellipeinc((np.pi-alpha1[mask1])/2, m[mask1]) - ellipeinc((np.pi - alpha2[mask1])/2, m[mask1]))
            N1[mask2] = (ellipeinc_alt((np.pi-alpha1[mask2])/2, m[mask2]) - ellipeinc_alt((np.pi - alpha2[mask2])/2, m[mask2]))
            df[mask] += -frac*self.f_eps[mask]*(1+b**2/self.b_90(v_orb)**2)**2*np.sqrt(1 - r0/r_eps[mask] + b/r0)*N1
            
        T_orb = (2*np.pi*r0*pc_to_km)/v_orb
        norm = 2*np.sqrt(2*(self.psi(r0)))*4*np.pi**2*r0*(self.b_90(v_orb)**2/(v_orb)**2)
        return norm*df/T_orb/self.DoS()
        
    def dfdt_plus(self, r0, v_orb, v_cut=-1, n_kick = 1):
        """Particles to add back into distribution function from E - dE -> E."""
        if (v_cut < 0):
            v_cut = self.v_max(r0)
        
        T_orb = (2*np.pi*r0*pc_to_km)/v_orb
        
        df = np.zeros(N_GRID)
        
        
        # Calculate sizes of kicks and corresponding weights for integration
        if (n_kick == 1):   #Replace everything by the average if n_kick = 1
            delta_eps_list = (-2*v_orb**2*np.log(1+self.Lambda**2)/self.Lambda**2,)
            frac_list = (1,)
      
        else:
           b_list = np.geomspace(self.b_min(v_orb), self.b_max(v_orb), n_kick)
           delta_eps_list = self.delta_eps_of_b(v_orb, b_list)
           
           #Step size for trapezoidal integration
           step = delta_eps_list[1:] - delta_eps_list[:-1]
           step = np.append(step, 0)
           step = np.append(0, step)
           
           #Make sure that the integral is normalised correctly
           renorm = np.trapz(self.P_delta_eps(v_orb, delta_eps_list), delta_eps_list)
           frac_list = 0.5*(step[:-1] + step[1:])/renorm
       

        #Sum over the kicks
        for delta_eps, b, frac in zip(delta_eps_list, b_list, frac_list):
            
            #Value of specific energy before the kick
            eps_old = self.eps_grid - delta_eps
        
            r_eps = G_N*self.M_BH/eps_old
            r_cut = G_N*self.M_BH/(eps_old + 0.5*v_cut**2)
        
            #Define which energies are allowed to scatter
            mask = (eps_old > self.psi(r0)*(1-b/r0) - 0.5*v_cut**2) & (eps_old < self.psi(r0)*(1+b/r0))
        
            # Distribution of particles before they scatter
            f_old = self.interpolate_DF(eps_old[mask])

            L1 = np.minimum((r0 - r0**2/r_eps[mask])/b, 1)
            alpha1 = np.arccos(L1)
            L2 = np.maximum((r0 - r0**2/r_cut[mask])/b, -1)
            alpha2 = np.arccos(L2)

            m = (2*b/r0)/(1 - (r0/r_eps[mask]) + b/r0)
            mask1 = (m <= 1) & (alpha2 > alpha1)
            mask2 = (m > 1) & (alpha2 > alpha1)
            N1 = np.zeros(len(m))
            N1[mask1] = (ellipeinc((np.pi-alpha1[mask1])/2, m[mask1]) - ellipeinc((np.pi - alpha2[mask1])/2, m[mask1]))
            N1[mask2] = (ellipeinc_alt((np.pi-alpha1[mask2])/2, m[mask2]) - ellipeinc_alt((np.pi - alpha2[mask2])/2, m[mask2]))
            
            df[mask] += +frac*f_old*(1+b**2/self.b_90(v_orb)**2)**2*np.sqrt(1 - r0/r_eps[mask] + b/r0)*N1
            
        T_orb = (2*np.pi*r0*pc_to_km)/v_orb
        norm = 2*np.sqrt(2*(self.psi(r0)))*4*np.pi**2*r0*(self.b_90(v_orb)**2/(v_orb)**2)
        return norm*df/T_orb/self.DoS()
        
        
        
    def dEdt_ej(self, r0, v_orb, v_cut=-1, n_kick = N_KICK):
        """Calculate carried away by particles which are completely unbound.
        
        Parameters:
            - r0 : radial position of the perturbing body [pc]
            - v_orb: orbital velocity [km/s]
            - v_cut: optional, only scatter with particles slower than v_cut [km/s]
                        defaults to v_max(r) (i.e. all particles)
            - n_kick: optional, number of grid points to use when integrating over
                        Delta-eps (defaults to N_KICK = 100).
        """
        if (v_cut < 0):
            v_cut = self.v_max(r0)
        
        T_orb = (2*np.pi*r0*pc_to_km)/v_orb
        
        dE = np.zeros(N_GRID)
        
        
        # Calculate sizes of kicks and corresponding weights for integration
        if (n_kick == 1):   #Replace everything by the average if n_kick = 1
            delta_eps_list = (-2*v_orb**2*np.log(1+self.Lambda**2)/self.Lambda**2,)
            frac_list = (1,)
      
        else:
            b_list = np.geomspace(self.b_min(v_orb), self.b_max(v_orb), n_kick)
            delta_eps_list = self.delta_eps_of_b(v_orb, b_list)
           
            #Step size for trapezoidal integration
            step = delta_eps_list[1:] - delta_eps_list[:-1]
            step = np.append(step, 0)
            step = np.append(0, step)
       
            #Make sure that the integral is normalised correctly
            renorm = np.trapz(self.P_delta_eps(v_orb, delta_eps_list), delta_eps_list)
            frac_list = 0.5*(step[:-1] + step[1:])/renorm

        #Sum over the kicks
        for delta_eps, b, frac in zip(delta_eps_list, b_list, frac_list):
        
            r_eps = G_N*self.M_BH/self.eps_grid
            r_cut = G_N*self.M_BH/(self.eps_grid + 0.5*v_cut**2)
        
            #Maximum impact parameter which leads to the ejection of particles 
            b_ej_sq = self.b_90(v_orb)**2*((2*v_orb**2/self.eps_grid) - 1)    
                
            #Define which energies are allowed to scatter
            mask = (self.eps_grid > self.psi(r0)*(1-b/r0) - 0.5*v_cut**2) & (self.eps_grid < self.psi(r0)*(1+b/r0)) & (b**2 < b_ej_sq)
        
            L1 = np.minimum((r0 - r0**2/r_eps[mask])/b, 1)
            alpha1 = np.arccos(L1)
            L2 = np.maximum((r0 - r0**2/r_cut[mask])/b, -1)
            alpha2 = np.arccos(L2)

            m = (2*b/r0)/(1 - (r0/r_eps[mask]) + b/r0)
            mask1 = (m <= 1) & (alpha2 > alpha1)
            mask2 = (m > 1) & (alpha2 > alpha1)
            N1 = np.zeros(len(m))
            N1[mask1] = (ellipeinc((np.pi-alpha1[mask1])/2, m[mask1]) - ellipeinc((np.pi - alpha2[mask1])/2, m[mask1]))
            N1[mask2] = (ellipeinc_alt((np.pi-alpha1[mask2])/2, m[mask2]) - ellipeinc_alt((np.pi - alpha2[mask2])/2, m[mask2]))

            dE[mask] += -frac*self.f_eps[mask]*(1+b**2/self.b_90(v_orb)**2)**2*np.sqrt(1 - r0/r_eps[mask] + b/r0)*N1*(self.eps_grid[mask] + delta_eps)
            
        norm = 2*np.sqrt(2*(self.psi(r0)))*4*np.pi**2*r0*(self.b_90(v_orb)**2/(v_orb)**2)
        return norm*np.trapz(dE, self.eps_grid)/T_orb