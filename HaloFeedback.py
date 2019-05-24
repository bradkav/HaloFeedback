#HaloFeedback
import numpy as np
from scipy.special import gamma as Gamma_func
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from time import time as timeit

#------------------
G_N = 4.302e-3 #(km/s)^2 pc/M_sun
c = 2.99792458e5 #km/s
Lambda = np.exp(3.0)

n_kick = 3

#Conversion factors
pc_to_km = 3.0857e13


#Numerical parameters

N_grid = 10000

#------------------


class DistributionFunction():
    def __init__(self, M_BH=1e3, M_NS = 1.0, gamma=7./3., rho_sp=226):
        self.M_BH = M_BH    #Solar mass
        self.M_NS = M_NS    #Solar mass
        self.gamma = gamma
        self.rho_sp = rho_sp    #Solar mass/pc^3
        
        self.r_sp = ((3-gamma)*(0.2**(3.0-gamma))*M_BH/(2*np.pi*rho_sp))**(1.0/3.0) #pc
        self.r_isco = 6.0*G_N*M_BH/c**2
        
        #Initialise f(eps)
        self.r_grid = np.geomspace(self.r_isco,1e-2*self.r_sp, N_grid)
        #self.r_grid = np.append(self.r_grid, np.max(self.r_grid)*(1 - np.geomspace(1e-5, 1e-1, 1000)))
        #self.r_grid = np.sort(self.r_grid)
        
        self.eps_grid = self.psi(self.r_grid)
        A1 = (self.r_sp/(G_N*M_BH))
        self.f_eps = self.rho_sp*(gamma*(gamma - 1)*A1**gamma*np.pi**-1.5/np.sqrt(8))*(Gamma_func(-1 +gamma)/Gamma_func(-1/2 + gamma))*self.eps_grid**(-(3/2) + gamma) 

        
        self.strID = "lnLambda=%.1f_n=%d"%(np.log(Lambda), n_kick)
        #self.eps_new, self.eps_old = np.meshgrid(self.eps_grid, self.eps_grid)
        #self.delta_eps = self.eps_new - self.eps_old
    
        
        
    def psi(self, r):
        return G_N*self.M_BH/r
        
    def v_max(self, r):
        return np.sqrt(2*self.psi(r))
    
    def rho(self, r, v_cut=-1):
        if (v_cut < 0):
            v_cut = self.v_max(r)
            
        v_cut = np.clip(v_cut, 0, self.v_max(r))
        
        vlist = np.sqrt(np.linspace(0, v_cut**2, 500))
        flist = np.interp(self.psi(r) - 0.5*vlist**2, self.eps_grid[::-1], self.f_eps[::-1], left=0, right=0)
        
        integ = vlist**2*flist
        
        return 4*np.pi*np.trapz(integ, vlist)

    def rho_init(self, r):
        return self.rho_sp*(r/self.r_sp)**-self.gamma

    #def P_delta_eps(self):
    

    def dfdt_minus(self, r0, v_orb, v_cut=-1):
        #print("Do I have to change b_max for the three cases?")
        
        
        if (v_cut < 0):
            v_cut = self.v_max(r0)
        
        
        T_orb = 2*np.pi*r0*pc_to_km/v_orb
        
        b_min = G_N*self.M_NS/(v_orb**2)
        b_max = b_min*Lambda
        
        r_eps = G_N*self.M_BH/self.eps_grid
        
        mask = (self.eps_grid > self.psi(r0) - 0.5*v_cut**2) & (self.eps_grid < self.psi(r0))


        df = np.zeros(N_grid)
        df[mask] = -self.f_eps[mask]*8*b_max**2*r0*np.sqrt(1/r0 - 1/r_eps[mask])/r_eps[mask]**2.5

        return df/T_orb
        
    def dfdt_plus(self, r0, v_orb, v_cut=-1):
        if (v_cut < 0):
            v_cut = self.v_max(r0)
        
        T_orb = 2*np.pi*r0*pc_to_km/v_orb
        
        b_min = G_N*self.M_NS/(v_orb**2)
        b_max = b_min*Lambda
        #print(b_max)
        
        df = np.zeros(N_grid)
        #delta_eps = 4*G_N**2*self.M_NS**2*np.log(Lambda)/(v_orb**2*b_max**2)
        
        delta_eps_list, frac_list = self.calc_delta_eps(v_orb)
        #print(delta_eps_list)
        """
        if (n_kick == 1):
            delta_eps_list = [-2*v_orb**2*np.log(1+Lambda**2)/Lambda**2 ,]
            
        if (n_kick == 2):
            delta_eps_list = [-4*v_orb**2*np.log((1+Lambda**2)/(1+0.5*Lambda**2))/Lambda**2, -4*v_orb**2*np.log(1+0.5*Lambda**2)/Lambda**2]
            
        if (n_kick == 3):
            delta_eps_list = [  -6*v_orb**2*np.log((3+3*Lambda**2)/(3+2*Lambda**2))/Lambda**2,
                                -6*v_orb**2*np.log((3+2*Lambda**2)/(3+Lambda**2))/Lambda**2,
                                -6*v_orb**2*np.log((3+Lambda**2)/3)/Lambda**2]
        """
        #print(np.log10(-np.array(delta_eps_list)))
        
        #delta_eps_list = [-2*v_orb**2*np.log(1+Lambda**2)/Lambda**2 ,]
        
        
        
        for delta_eps, frac in zip(delta_eps_list, frac_list):
            #delta_eps = -2*v_orb**2*np.log(1+Lambda**2)/Lambda**2
            #print(delta_eps)
        
            eps_old = self.eps_grid - delta_eps
        
            mask = (eps_old  > self.psi(r0) - 0.5*v_cut**2) & (eps_old < self.psi(r0))
        
            #t0 = timeit()
            f_old = np.interp(eps_old[mask][::-1], self.eps_grid[::-1],
                                    self.f_eps[::-1], left=0, right=0)[::-1]
            #t1 = timeit()
            #print("interpolating:", t1 - t0)
        
            r_eps = G_N*self.M_BH/eps_old[mask]
            #t2 = timeit()
            #print("calc r_eps:", t2 - t1)
        
        
            #t3 = timeit()
            #print("masking:", t3 - t2)
        
            #t4 = timeit()
            #print("init array:", t4 - t3)
        
        
            df[mask] += frac*(f_old*8*b_max**2*r0*np.sqrt(1/r0 - 1/r_eps)/r_eps**2.5)*(self.eps_grid[mask]/eps_old[mask])**2.5
            #t5 = timeit()
            #print("calc df:", t5 - t4)
        
        #print("done")
        return (df/T_orb)#/len(delta_eps_list)
        
        
        
    def dfdt_plus_full(self, r0, v_orb, v_cut=-1):
        if (v_cut < 0):
            v_cut = self.v_max(r0)
        
        T_orb = 2*np.pi*r0*pc_to_km/v_orb
        
        b_min = G_N*self.M_NS/(v_orb**2)
        b_max = b_min*Lambda
        #delta_eps = 4*G_N**2*self.M_NS**2*np.log(Lambda)/(v_orb**2*b_max**2)
        
        df = np.zeros(N_grid)
        
        #istart = np.argmin(self.eps_grid[self.eps_grid > self.psi(r0)])
        
        #print(istart)
        #print(self.eps_grid[istart-1:istart+1])
        
        #Redefine grids and lists...
        #Refine these exact routines...
                
        #Need to define a mask makes sense mathematically
        mask1 = (delta_eps > -2*v**2) & (delta_eps < -2*v**2/(1+Lambda**2))
                
        t1 = timeit()
        P_delta_eps = self.delta_eps*self.calcP_delta_eps(self.delta_eps, v_orb)
        t2 = timeit()
        print(t2 - t1)
        #delta_eps *= self.calcP_delta_eps(delta_eps, v_orb)
        t3 = timeit()
        print(t3 - t2)
        r_eps = G_N*self.M_BH/self.eps_old
        t4 = timeit()
        print(t4 - t3)
        mask = (self.eps_grid  > self.psi(r0) - 0.5*v_cut**2) & (self.eps_grid < self.psi(r0))
        t5 = timeit()
        print(t5 - t4)
        
        #plt.figure()
        #plt.contourf(delta_eps)
        #plt.colorbar()
        #plt.show()
        y = (self.eps_new[:,mask]/self.eps_old[:,mask])**2.5*P_delta_eps[:,mask]*self.f_eps[mask]*8*b_max**2*r0*np.sqrt(1/r0 - 1/r_eps[:,mask])/r_eps[:,mask]**2.5
        print(y.shape)
        df[mask] = np.trapz(y, self.delta_eps[:,mask], axis=0)
        #df[mask] = np.sum(y*self.delta_eps[:,mask], axis=0)
        t6 = timeit()
        print(t6 - t5)
        #df = 0
        #for i in range(N_grid-1, istart,-1):
            #eps_old = self.eps_grid - delta_eps...
        #    eps_old = self.eps_grid
        #    delta_eps = self.eps_grid[i] - self.eps_grid
            #print(np.min(delta_eps), np.max(delta_eps))

            
        #    mask = (eps_old  > self.psi(r0) - 0.5*v_cut**2) & (eps_old < self.psi(r0))
        #    P_delta_eps = self.calcP_delta_eps(delta_eps, v_orb)
            #print(delta_eps)
            #eps_old = np.roll(self.eps_grid)
        #delta_eps = -2*v_orb**2*np.log(1+Lambda**2)/Lambda**2
        
        #eps_old = self.eps_grid - delta_eps
        
        #f_old = np.interp(eps_old[::-1], self.eps_grid[::-1],
        #                        self.f_eps[::-1], left=0, right=0)[::-1]
        
        
            #mask = (eps_old  > self.psi(r0) - 0.5*v_cut**2) & (eps_old < self.psi(r0))

        
            #df[i] = np.trapz((self.eps_grid[i]/eps_old[mask])**2.5*P_delta_eps[mask]*self.f_eps[mask]*8*b_max**2*r0*np.sqrt(1/r0 - 1/r_eps[mask])/r_eps[mask]**2.5, delta_eps[mask])
        
        return (df/T_orb)#*(self.eps_grid/eps_old)**2.5
        
    
    def dfdt_plus_full2(self, r0, v_orb, v_cut=-1):
        if (v_cut < 0):
            v_cut = self.v_max(r0)
        
        T_orb = 2*np.pi*r0*pc_to_km/v_orb
        
        b_min = G_N*self.M_NS/(v_orb**2)
        b_max = b_min*Lambda
        
        df = np.zeros(N_grid)
        
        eps_min = self.psi(r0) - 0.5*v_cut**2
        eps_max = self.psi(r0)
        mask1 = (self.eps_grid  > eps_min) & (self.eps_grid < eps_max)
        #print(self.eps_grid[0], self.eps_grid[1])
    
        print(np.log10(2*v_orb**2),np.log10(2*v_orb**2/(1+Lambda**2)))
        mask2 = (self.eps_grid > eps_min-2*v_orb**2) & (self.eps_grid < eps_max-2*v_orb**2/(1+Lambda**2))
        mask2 = np.atleast_2d(mask2).T
        
        print(mask2)
        
        plt.figure()
        
        mask_de = (self.delta_eps > -2*v_orb**2) & (self.delta_eps < -2*v_orb**2/(1+Lambda**2))
        plt.contourf(np.log10(self.eps_grid), np.log10(self.eps_grid), np.log10((self.delta_eps*mask_de + 1e-30).T))
        plt.plot([np.log10(self.eps_grid[0]), np.log10(self.eps_grid[-1])],[np.log10(self.eps_grid[0]), np.log10(self.eps_grid[-1])], linestyle='--', color='k',zorder=5)
        plt.xlabel("Initial energy")
        plt.ylabel("Final energy")
        plt.colorbar()
        
        plt.figure()
        
        plt.contourf(np.ones((N_grid, N_grid))*mask1)
        
        
        plt.figure()
        plt.contourf(np.ones((N_grid, N_grid))*mask2)
        
        plt.colorbar()
        plt.show()
        
        
        
    def dfdt(self, r0, v_orb, v_cut=-1, use_average=True):
        if (use_average):
            return self.dfdt_minus(r0, v_orb, v_cut) + self.dfdt_plus(r0, v_orb, v_cut)
        else:
            return self.dfdt_minus(r0, v_orb, v_cut) + self.dfdt_plus_full2(r0, v_orb, v_cut)
        
        
      
    def P_eps(self):
        return np.sqrt(2)*np.pi**3*(G_N*self.M_BH)**3*self.f_eps/self.eps_grid**2.5
        
        
    def calcP_delta_eps(self, delta_eps, v):
        mask = (delta_eps > -2*v**2) & (delta_eps < -2*v**2/(1+Lambda**2))
        return mask*2*(v**2/Lambda**2)/(np.abs(delta_eps)+1e-30)**2
        
        
    def calc_delta_eps(self, v):
        eps_min = 2*v**2/(1+Lambda**2)
        eps_max = 2*v**2
        
        eps_edges = np.linspace(eps_min, eps_max, n_kick+1)
        
        def F_norm(eps):
            return -2*v**2/(eps*Lambda**2)
        def F_avg(eps):
            return -2*v**2*np.log(eps)/Lambda**2
            
        frac = np.diff(F_norm(eps_edges))
        eps_avg = np.diff(F_avg(eps_edges))/frac
        #print(eps_avg/v**2)
        
        return eps_avg, frac
        
"""
def v_max()


def f_initial(E):
    return 


@np.vectorize
def Frac_func(x):
    if (x == 0):
        return -0.5
    if (x == 1):
        return 0.5
    A = (2/(3*np.pi))*np.sqrt(x - x**2)*(8*x**2 - 2*x - 3)
    B = (1/np.pi)*np.arctan((0.5 - x)/np.sqrt(x-x**2))
    return A - B

def calcFrac(E, r1, r2, v_cut, M_BH=1e3):
    r1_mod = np.max(r1, G_N*M_BH/(v_cut**2/2 + E))
    r2_mod = np.min(r2, G_N*M_BH/E)
    
    return Frac_func(r2_mod*E/(G_N*M_BH)) - Frac_func(r1_mod*E/(G_N*M_BH))
        
"""