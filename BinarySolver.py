import numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy.special import erf
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d,interp2d,RectBivariateSpline, griddata

import time

import HaloFeedback

import argparse
import os

from matplotlib import pyplot as plt

#############################
#### fundamental constants ##
#############################

G_N = 6.67408e-8
c_light = 2.99792458e10

s = 1. #s 
year = 365.25*24.*3600.*s
km = 1e5
Msun = 1.98855e33  # kg
pc = 3.08567758149137e18

#Parse the arguments!                                                       
parser = argparse.ArgumentParser(description='...')
parser.add_argument('-M1', '--M1', help='Larger BH mass M1 in M_sun', type=float, default=1000.)
parser.add_argument('-M2', '--M2', help="Smalller BH mass M2 in M_sun", type=float, default = 1.0)
parser.add_argument('-rho6', '--rho6', help='Spike density normalisation [1e13 M_sun/pc^3]', type=float, default=1.0)
parser.add_argument('-gamma', '--gamma', help='slope of DM spike', type=float, default=2.3333)

parser.add_argument('-system', '--system', help='Type of system to evolve: "vacuum", "static", "dynamic" [default], "PBH"', type=str, default='dynamic')

parser.add_argument('-r_i', '--r_i', help='Initial radius in pc', type=float, default = -1)
parser.add_argument('-short', '--short', help='Set to 1 to finish before r_isco', type=int, default = 0)
parser.add_argument('-dN_ini', '--dN_ini', help='Initial time-step size in number of orbits', type=float, default = 10.0)
parser.add_argument('-dN_max', '--dN_max', help='Maximum time-step size in number of orbits', type=float, default = 250.0)

parser.add_argument('-verbose', '--verbose', type=int, default=1)

parser.add_argument('-outdir', '--outdir', help='Directory where results will be stored.', type=str, default="runs/")
parser.add_argument('-IDtag', '--IDtag', help='Optional IDtag to add on the end of the file names', type=str, default="NONE")

#Add an *OPTIONAL* ID tag or ID string here...
args = parser.parse_args()
verbose = args.verbose

IDstr = "M1_%.4f_M2_%.4f_rho6_%.4f_gamma_%.4f"%(args.M1, args.M2, args.rho6, args.gamma) 
if (args.IDtag != "NONE"):
    IDstr += "_" + args.IDtag

print("> Run ID: ", IDstr)

output_folder = args.outdir
output_folder = os.path.join(output_folder, '')

#Generate folder structure if needed
OUTPUT = True
if (OUTPUT):
    if not (os.path.isdir(output_folder)):
        os.mkdir(output_folder)
print(" ")

system = (args.system).lower()

############################
#  Simulation parameters   #
############################

NPeriods_ini = args.dN_ini
dN_max = args.dN_max

if (args.short > 0):
    SHORT = True
else:
    SHORT = False

######################################
# BH, spike and binary parameters   ##
######################################

M1 = args.M1*Msun 
M2 = args.M2*Msun

M = M1 + M2

gamma_sp = args.gamma
rho_6 = args.rho6*1e13*Msun/pc**3

if (system == 'pbh'):
    print("> Setting spike parameters for a PBH...")
    gamma_sp = 9.0/4.0
    rho6 = 1.396e13*(args.M1)**(3/4)*Msun/pc**3

m_tilde = ((3.-gamma_sp)*(0.2**(3.0-gamma_sp))*M1/(2*np.pi))
r_6 = 1e-6*pc
rho_sp = (rho_6*r_6**gamma_sp/(m_tilde**(gamma_sp/3.0)))**(1/(1-gamma_sp/3)) # cgs 


#Calculate the initial radius based on projected merger time
r_grav  = G_N/(c_light**2.) * 2.* M1
r_isco  = 3 * r_grav
r_in    = 2 * r_grav


if (SHORT):
    r_end = 80*r_isco
else:
    r_end = r_isco

    
def calc_Torb(r):
    return 2 * np.pi * np.sqrt(r ** 3 / (G_N * M))

def calc_f(r):
    return 2/calc_Torb(r)

def calc_vorb(r):
    return 2*np.pi*r/calc_Torb(r)


#Initial values of a few different parameters
r0_initial = args.r_i*pc
if (system != "vacuum"):
    dist = HaloFeedback.PowerLawSpike(gamma = gamma_sp, M_BH = M1/Msun, M_NS = M2/Msun, rho_sp = rho_sp/(Msun/pc**3.))
f_initial  = calc_f(r0_initial)
    
print("> System properties:")
print(">    M_1, M_2 [M_sun]: ", M1/Msun, M2/Msun)
print(">    gamma_sp, rho_6 [M_sun/pc^3]: ", gamma_sp, rho_6/(Msun/pc**3))
print(">    r_i [pc]:", r0_initial/pc)
print(" ")
        

################################
##### Equations of motion ######
################################
    
    
def get_density(r):
    if (system == "vacuum"):
        rho_xi = 0.0
    elif (system == "static"):
        rho_xi = dist.rho_init(r/pc)*dist.xi_init
    elif (system in ["dynamic", "pbh"]):
        v_orb = calc_vorb(r)
        rho_xi = dist.rho(r/pc, v_cut=v_orb/(km/s))
    return rho_xi*Msun/pc**3

# Radiation reaction
# See, for example, eq. 226 in https://arxiv.org/pdf/1310.1528.pdf
GW_prefactor = -64.*(G_N**3.)*M1*M2*(M)/(5.*(c_light**5.))
def GW_term(t,r):
    return GW_prefactor/r**3

# Gravitational "friction"
# See, for example, https://arxiv.org/pdf/1604.02034.pdf
lnLambda = np.log(np.sqrt(M1/M2))
DF_prefactor = -(8*np.pi*G_N**0.5*M2*lnLambda/(M**0.5*M1))
def DF_term(t, r):
    return DF_prefactor*r**2.5*get_density(r)
    
#Derivatives (for feeding into the ODE solver)
def drdt_ode(t, r):
    GW = GW_term(t, r)
    if (system == "vacuum"):
        DF = 0
    else:
        DF = DF_term(t, r)
    #print(DF/GW)
    return GW + DF
    
def save_trajectory():
    htxt = 'Columns: t [s], r [pc], f_GW [Hz], rho_eff (< v_orb) [Msun/pc^3]'
    output = list(zip(t_list, r_list/pc, f_list, rho_list/(Msun/pc**3)))
    fname = output_folder + "trajectory_" + IDstr + ".txt.gz"
    np.savetxt(fname, output, header=htxt,  fmt='%.10e')
    return None
    

#################################################
############ DYNAMIC DRESS ######################
#################################################
t_list = np.array([0.])
r_list = np.array([r0_initial])
f_list = np.array([f_initial])
rho_list = np.array([get_density(r0_initial)])

start_time = time.time()
#print("> Evolving system with EFFECTIVE DENSITY PROFILE...")


NPeriods = 1*NPeriods_ini
r0 = r0_initial
t0 = 0.
#currentPeriod = DF_current.T_orb(current_r/pc)
i = 0

#integrator = ode(drdt_ode).set_integrator(method)
#integrator.set_f_params(DF_current)
#integrator.set_initial_value(r0_initial, current_t) #.set_f_params(2.0).set_jac_params(2.0)

OUTPUT_ALL = False


dN = 1.0*NPeriods_ini


#print(r0/pc, r_end/pc)
while (r0 > r_end):
    
    dt = calc_Torb(r0)*dN
    #print(calc_Torb(r0))
    
    dN = np.clip(dN, 0, dN_max)
    
    v_orb = calc_vorb(r0)
    #print(v_orb/(km/s))
    if (system in ["dynamic", "pbh"]):
        dfdt1 = dist.dfdt(r0/pc, v_orb/(km/s), v_cut=v_orb/(km/s))
    
    
    
    #Add 'excess' checks here...
    if (system in ["dynamic", "pbh"]):
        excess_list = -(2/3)*dt*dfdt1/(dist.f_eps + 1e-30)
        excess = np.max(excess_list[1:]) #Omit the DF at isco                       
                                     
        if (excess > 1):
            dN /= excess*1.1
            if (verbose > 2):
                print("Too large! New value of dN = ", dN)
                                                     
        elif (excess > 1e-1):
            dN /= 1.1
            if (verbose > 2):
                print("Getting large! New value of dN = ", dN)
                                                                 
        elif ((excess < 1e-2) and (i%100 == 0) and (i > 0) and (dN < dN_max)):
            dN *= 1.1
            if (verbose > 2):
                print("Increasing! New value of dN = ", dN)

        dt = calc_Torb(r0)*dN
    

    #Use Ralston's Method (RK2) to evolve the system 
    drdt1 = drdt_ode(t0, r0)

    r0          += (2/3)*dt*drdt1
    if (system in ["dynamic", "pbh"]):
        dist.f_eps  += (2/3)*dt*dfdt1

    drdt2 = drdt_ode(t0, r0)
    
    if (system in ["dynamic", "pbh"]):
        v_orb = calc_vorb(r0)
        dfdt2 = dist.dfdt(r0/pc, v_orb/(km/s), v_cut=v_orb/(km/s))

    r0          += (dt/12)*(9*drdt2 - 5*drdt1)
    if (system in ["dynamic", "pbh"]):
        dist.f_eps  += (dt/12)*(9*dfdt2 - 5*dfdt1)
    
    t0 += dt
    
    f0 = calc_f(r0)
    
    t_list = np.append(t_list, t0)
    r_list = np.append(r_list, r0)
    f_list = np.append(f_list, f0)
    rho_list = np.append(rho_list, get_density(r0))
    
    if (i%1000==0):
        if (verbose > 1):
            print(f">    r/r_end = {r0/r_end:.5f}; f_GW [Hz] = {f0:.5f}; t [s] = {t0:.5f}; rho_eff [Msun/pc^3] = {rho_list[-1]/(Msun/pc**3):.4e}")
 
        #Update the output file
        if (verbose > 0):
            if (OUTPUT): save_trajectory()
    
    i = i+1
   
  
#Correct final point to be exactly r_end (rather than < r_end)
inds = np.argsort(r_list)
t_last = np.interp(r_end, r_list[inds], t_list[inds])
f_last = calc_f(r_end)

t_list[-1] = t_last
r_list[-1] = r_end
f_list[-1] = f_last
rho_list[-1] = get_density(r_end*1.000001)
   
if (OUTPUT): save_trajectory()

#Make some plots
fig, ax = plt.subplots(ncols=2, nrows=1,figsize=(10, 5))
ax[0].semilogy(t_list, r_list/pc)
ax[0].set_xlabel(r"$t$ [s]")
ax[0].set_ylabel(r"$R$ [pc]")
    
ax[1].loglog(r_list/pc, rho_list/(Msun/pc**3))
ax[1].set_xlabel(r"$R$ [pc]")
ax[1].set_ylabel(r"$\rho_{\mathrm{eff}, v < v_\mathrm{orb}}(R)$ [$M_\odot\,\mathrm{pc}^{-3}$]")

plt.suptitle(IDstr.replace("_", "\_"))
plt.tight_layout()

plt.savefig(output_folder +  "Evolution_" + IDstr + ".pdf", bbox_inches='tight')

plt.show()
print("> Done")               
print("> Time needed: %s seconds" % (time.time() - start_time))                        
      

