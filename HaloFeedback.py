#HaloFeedback
import numpy as np
from scipy.special import gamma as Gamma_func
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
N_grid = 10000  #Number of grid points in the specific energy

#------------------


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
        self.r_grid = np.geomspace(self.r_isco, 1e3*self.r_sp, N_grid)
        #self.r_grid = np.append(self.r_grid, np.space(2*self.r_sp, 1e20*self.r_sp, 100))
        
        self.eps_grid = self.psi(self.r_grid)

        
        
        A1 = (self.r_sp/(G_N*M_BH))
        self.f_eps = self.rho_sp*(gamma*(gamma - 1)*A1**gamma*np.pi**-1.5/np.sqrt(8))*(Gamma_func(-1 + gamma)/Gamma_func(-1/2 + gamma))*self.eps_grid**(-(3/2) + gamma) 

        #Define a string which specifies the model parameters
        #and numerical parameters (for use in file names etc.)
        self.IDstr_num = "lnLambda=%.1f"%(np.log(self.Lambda),)
        self.IDstr_model = "gamma=%.2f_rhosp=.%1f"%(gamma, rho_sp)
        
    
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

    def rho_init(self, r):
        """Initial DM density of the system"""
        return self.rho_sp*(r/self.r_sp)**-self.gamma

    def TotalMass(self):
        return np.trapz(-self.P_eps(), self.eps_grid)

    def TotalEnergy(self):
        return np.trapz(-self.P_eps()*self.eps_grid, self.eps_grid)




    def b_90(self, v_orb):
        return G_N*self.M_NS/(v_orb**2)
        
    def b_min(self, v_orb):
        return 15./pc_to_km
        
    def b_max(self, v_orb):
        return self.Lambda*np.sqrt(self.b_90(v_orb)**2 + self.b_min(v_orb)**2)
        
    def eps_min(self, v_orb):
        return 2*v_orb**2/(1+ self.b_max(v_orb)**2/self.b_90(v_orb)**2)
    
    def eps_max(self, v_orb):
        return 2*v_orb**2/(1 + self.b_min(v_orb)**2/self.b_90(v_orb)**2)
    
    
    
    
    def dfdt(self, r0, v_orb, v_cut=-1, method = 2, n_kick = 1):
        """Time derivative of the distribution function f(eps).
        
        Parameters:
            - r0 : radial position of the perturbing body [pc]
            - v_orb: orbital velocity [km/s]
            - v_cut: optional, only scatter with particles slower than v_cut [km/s]
                        defaults to v_max(r) (i.e. all particles)
            - method: optional, method = 1 or method = 2 chooses two different
                        approximate calculations of the scattering probability
                        (method 2 should be more accurate)
            - n_kick: optional, number of 'points' to include in the numerical
                    integration over delta-eps. Default: n_kick = 1, which
                    simply uses the average value of delta-eps.
        """
        
        if (n_kick > 1):
            warnings.warn('n_kick > 1 is not guaranteed to conserve energy!')
        
        assert (method in [1, 2])
    
        if (method == 1):
            return self.dfdt_minus_1(r0, v_orb, v_cut) + self.dfdt_plus_1(r0, v_orb, v_cut, n_kick)
        elif (method == 2):
            return self.dfdt_minus_2(r0, v_orb, v_cut) + self.dfdt_plus_2(r0, v_orb, v_cut, n_kick)
    

       

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
        
        
    def dEdt_DF(self, r, SPEED_CUT = False):
        """Rate of change of energy due to DF (km/s)^2 s^-1 M_sun"""
        v_orb = np.sqrt(self.psi(r))
        
        if (SPEED_CUT):
            v_cut = v_orb
        else:
            v_cut = -1
            
        #CoulombLog = 0.5*np.log((1 + self.b_max(v_orb)**2/self.b_90(v_orb)**2)/(1 + self.b_min(v_orb)**2/self.b_90(v_orb)**2))
        CoulombLog = np.log(self.Lambda)
            
        return (1/pc_to_km)*4*np.pi*G_N**2*self.M_NS**2*self.rho(r, v_cut)*CoulombLog/v_orb

    def E_orb(self,r):
        return -0.5*G_N*(self.M_BH + self.M_NS)/r
        
    def T_orb(self,r):
        return 2*np.pi*np.sqrt(pc_to_km**2*r**3/(G_N*(self.M_BH + self.M_NS)))
        

    def interpolate_DF(self, eps_old):
        # Distribution of particles before they scatter
        f_old = np.interp(eps_old[::-1], self.eps_grid[::-1], self.f_eps[::-1], left=0, right=0)[::-1]
        
        #DF_interp = interp1d(self.eps_grid[::-1], self.f_eps[::-1], bounds_error=False, fill_value = 0.0, kind='cubic')
        #f_old = DF_interp(eps_old[::-1])[::-1]
        return f_old

            
#---------------------
#----- METHOD 1   ----
#---------------------
    
    
    def dfdt_minus_1(self, r0, v_orb, v_cut=-1):
        """Particles to subtract from distribution function"""
        
        if (v_cut < 0):
            v_cut = self.v_max(r0)
        
        T_orb = (2*np.pi*r0*pc_to_km)/v_orb
        
        r_eps = G_N*self.M_BH/self.eps_grid
        
        #Define which energies are allowed to scatter
        mask = (self.eps_grid > self.psi(r0) - 0.5*v_cut**2) & (self.eps_grid < self.psi(r0))

        df = np.zeros(N_grid)
        df[mask] = -self.f_eps[mask]*8*self.b_max(v_orb)**2*r0*np.sqrt(1/r0 - 1/r_eps[mask])/r_eps[mask]**2.5
        return df/T_orb
        

    def dfdt_plus_1(self, r0, v_orb, v_cut=-1, n_kick = 1):
        """Particles to add back into distribution function."""
        
        if (v_cut < 0):
            v_cut = self.v_max(r0)
                
        df = np.zeros(N_grid)
        
        # Calculate sizes of kicks and corresponding weights for integration
        if (n_kick == 1):   #Replace everything by the average if n_kick = 1
            #delta_eps_list = (self.calc_delta_eps(v_orb)[0],)
            delta_eps_list = (-2*v_orb**2*np.log(1+self.Lambda**2)/self.Lambda**2,)
            frac_list = (1,)
        else:
            eps_min = self.eps_min(v_orb)
            eps_max = self.eps_max(v_orb)
       
            delta_eps_list = np.geomspace(-eps_max, -eps_min, n_kick)
                
            #Step size for trapezoidal integration
            step = delta_eps_list[1:] - delta_eps_list[:-1]
            step = np.append(step, 0)
            step = np.append(0, step)
        
            #Make sure that the integral is normalised correctly
            renorm = np.trapz(self.P_delta_eps(v_orb, delta_eps_list), delta_eps_list)
        
            #Calculate weights for each term
            frac_list = self.P_delta_eps(v_orb, delta_eps_list)*0.5*(step[:-1] + step[1:])/renorm
        

        #Sum over the kicks
        for delta_eps, frac in zip(delta_eps_list, frac_list):
        
            #Value of specific energy before the kick
            eps_old = self.eps_grid - delta_eps
    
            #Which particles can scatter?
            mask = (eps_old  > self.psi(r0) - 0.5*v_cut**2) & (eps_old < self.psi(r0))
    
            # Distribution of particles before they scatter
            f_old = self.interpolate_DF(eps_old[mask])
        
            
            r_eps = G_N*self.M_BH/eps_old[mask]
            df[mask] += frac*(f_old*8*self.b_max(v_orb)**2*r0*np.sqrt(1/r0 - 1/r_eps)/r_eps**2.5)*(self.eps_grid[mask]/eps_old[mask])**2.5

        T_orb = (2*np.pi*r0*pc_to_km)/v_orb

        return df/T_orb
        
#---------------------
#----- METHOD 2   ----
#---------------------

    #Angular integral
    def B_fun(self, r, eps):
        
        fudge = 1e-7 #Need a small offset to stop some values going negative
        
        A = r**2
        B = np.sqrt(eps)*(2*eps - self.psi(r))*np.sqrt(self.psi(r) - eps + fudge)
        C = self.psi(r)**2*np.arctan(np.sqrt(eps/(self.psi(r) - eps + fudge)))
        return r**2*(B + C)
        
        
    def dfdt_minus_2(self, r0, v_orb, v_cut=-1):
        """Particles to subtract from distribution function"""
        
        if (v_cut < 0):
            v_cut = self.v_max(r0)
        
        r_eps = G_N*self.M_BH/self.eps_grid
        r_cut = G_N*self.M_BH/(self.eps_grid + 0.5*v_cut**2)
        
        #Define which energies are allowed to scatter
        mask = (self.eps_grid > self.psi(r0 + self.b_max(v_orb)) - 0.5*v_cut**2) & (self.eps_grid < self.psi(r0 - self.b_max(v_orb)))

        #Angular integrals
        B1 = self.B_fun(np.minimum(r0 + self.b_max(v_orb), r_eps[mask]), self.eps_grid[mask])
        B2 = self.B_fun(np.maximum(r0 - self.b_max(v_orb), r_cut[mask]), self.eps_grid[mask])

        #Plot comparing the different P_scat methods...
        #Just ignore it...
        """
        plt.figure(figsize=(8,5.5))
        
        plt.plot(self.eps_grid, 8*self.b_max(v_orb)**2*r0*np.sqrt(1/r0 - 1/r_eps)/r_eps**2.5, label='Method 1', zorder=2)
        plt.plot(self.eps_grid[mask], (self.b_max(v_orb)*self.eps_grid[mask]*(G_N*self.M_BH)**-3)*(B1 - B2), label='Method 2', zorder=3)
        
        plt.axvline(self.psi(r0), linestyle='--', color='k')
        plt.axvline(self.psi(r0 - self.b_max(v_orb)), linestyle=':', color='k')
        plt.axvline(self.psi(r0 + self.b_max(v_orb)), linestyle=':', color='k')
        
        #plt.ylim()
        plt.legend(loc=8)
        
        plt.xlabel(r"$\mathcal{E}$ [(km/s)$^2$]")
        plt.ylabel(r"$P_\mathrm{scat}(\mathcal{E}, r_0)$")
        
        axins = inset_axes(plt.gca(), width=2.2, height=2.0, loc=2)
        axins.tick_params(labelleft=False, labelright=True)
        
        axins.plot(self.eps_grid, 8*self.b_max(v_orb)**2*r0*np.sqrt(1/r0 - 1/r_eps)/r_eps**2.5, zorder=2)
        axins.plot(self.eps_grid[mask], (self.b_max(v_orb)*self.eps_grid[mask]*(G_N*self.M_BH)**-3)*(B1 - B2), zorder=3)
        
        axins.axvline(self.psi(r0), linestyle='--', color='k')
        axins.axvline(self.psi(r0 - self.b_max(v_orb)), linestyle=':', color='k')
        axins.axvline(self.psi(r0 + self.b_max(v_orb)), linestyle=':', color='k')
        
        axins.set_xlim(self.psi(r0 + self.b_max(v_orb))*0.99, self.psi(r0 - self.b_max(v_orb))*1.01)
        
        axins.set_ylim(0, 0.0005)
        
        plt.savefig("../../plots/Pscat.pdf", bbox_inches="tight")
        
        plt.show()
        """

        df = np.zeros(N_grid)
        df[mask] = -self.f_eps[mask]*(self.b_max(v_orb)*self.eps_grid[mask]*(G_N*self.M_BH)**-3)*(B1 - B2)
        
        T_orb = (2*np.pi*r0*pc_to_km)/v_orb
        
        return df/T_orb
                
                
    def dfdt_plus_2(self, r0, v_orb, v_cut=-1, n_kick = 1):
        """Particles to add back into distribution function."""
        if (v_cut < 0):
            v_cut = self.v_max(r0)
        
        df = np.zeros(N_grid)
        
        # Calculate sizes of kicks and corresponding weights for integration
        if (n_kick == 1):   #Replace everything by the average if n_kick = 1
            #delta_eps_list = (self.calc_delta_eps(v_orb)[0],)
            delta_eps_list = (-2*v_orb**2*np.log(1+self.Lambda**2)/self.Lambda**2,)
            frac_list = (1,)
        else:
            eps_min = self.eps_min(v_orb)
            eps_max = self.eps_max(v_orb)
       
            delta_eps_list = np.geomspace(-eps_max, -eps_min, n_kick)
                
            #Step size for trapezoidal integration
            step = delta_eps_list[1:] - delta_eps_list[:-1]
            step = np.append(step, 0)
            step = np.append(0, step)
        
            #Make sure that the integral is normalised correctly
            renorm = np.trapz(self.P_delta_eps(v_orb, delta_eps_list), delta_eps_list)
        
            #Calculate weights for each term
            frac_list = self.P_delta_eps(v_orb, delta_eps_list)*0.5*(step[:-1] + step[1:])/renorm
        
        #print(delta_eps_list, 2*v_orb**2*np.log(1+self.Lambda**2)/self.Lambda**2)
        
        # Sum over the kicks
        for delta_eps, frac in zip(delta_eps_list, frac_list):
            
            #Value of specific energy before the kick
            eps_old = self.eps_grid - delta_eps
        
            r_eps = G_N*self.M_BH/eps_old
            r_cut = G_N*self.M_BH/(eps_old + 0.5*v_cut**2)
        
            #Define which energies are allowed to scatter
            mask = (eps_old > self.psi(r0 + self.b_max(v_orb)) - 0.5*v_cut**2) & (eps_old < self.psi(r0 - self.b_max(v_orb)))
            
            #Angular integrals
            B1 = self.B_fun(np.minimum(r0 + self.b_max(v_orb), r_eps[mask]), eps_old[mask])
            B2 = self.B_fun(np.maximum(r0 - self.b_max(v_orb), r_cut[mask]), eps_old[mask])

            # Distribution of particles before they scatter
            f_old = self.interpolate_DF(eps_old[mask])

            # Multiply by weights and P_scat
            df[mask] += frac*f_old*(self.b_max(v_orb)*eps_old[mask]*(G_N*self.M_BH)**-3)*(B1 - B2)*(self.eps_grid[mask]/eps_old[mask])**2.5
        
        T_orb = (2*np.pi*r0*pc_to_km)/v_orb
        return df/T_orb