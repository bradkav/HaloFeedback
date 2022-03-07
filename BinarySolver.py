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

#Parse the arguments!                                                       
parser = argparse.ArgumentParser(description='...')
parser.add_argument('-M1', '--M1', help='Larger BH mass M1 in M_sun', type=float, default=1000.)
parser.add_argument('-M2', '--M2', help="Smalller BH mass M2 in M_sun", type=float, default = 1.0)
parser.add_argument('-rho6', '--rho6', help='Spike density normalisation [1e16 M_sun/pc^3]', type=float, default=0.5)
parser.add_argument('-gamma', '--gamma', help='slope of DM spike', type=float, default=2.333)

parser.add_argument('-r_i', '--r_i', help='Initial radius in pc', type=float, default = -1)
parser.add_argument('-short', '--short', help='Set to 1 to finish before r_isco', type=int, default = 0)
parser.add_argument('-dN_ini', '--dN_ini', help='Initial time-step size in orbits', type=int, default = 10.0)
parser.add_argument('-dN_max', '--dN_max', help='Maximum time-step size in orbits', type=float, default = 250.0)

parser.add_argument('-IDtag', '--IDtag', help='Optional IDtag to add on the end of the file names', type=str, default="NONE")
parser.add_argument('-verbose', '--verbose', type=int, default=1)

parser.add_argument('-outdir', '--outdir', help='Directory where results will be stored.', type=str, default="runs/")

#Add an *OPTIONAL* ID tag or ID string here...
args = parser.parse_args()
verbose = args.verbose

IDstr = "M1_%.4f_M2_%.4f_rho6_%.4f_gamma_%.4f"%(args.M1, args.M2, args.rho6, args.gamma) 
if (args.IDtag != "NONE"):
    IDstr += "_" + args.IDtag

print("> Run ID: ", IDstr)


output_folder = args.outdir
output_folder = os.path.join(output_folder, '')

#Set to True in order to refine the solution at small radii (when 
#the number of orbits per step should be O(1) or less).
#Still in development...
REFINE = False

#Generate folder structure if needed
OUTPUT = True
if (OUTPUT):
    if not (os.path.isdir(output_folder)):
        os.mkdir(output_folder)
print(" ")

#############################
#### fundamental constants ##
#############################
G_N = 6.67430e-8 # cgs units    4.302e-3 #(pc/solar mass) (km/s)^2
c_light = 2.9979e10 #cgs units  

tUnit = 1. #s 
year = 365.*24.*3600.

Msun = 1.989e33 #g
km = 1.e5 #cm
pc = 3.085677581e18 #cm
AU = 14959787070000 #cm 
rhoUnits = Msun/(pc**3.)

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
# IMBH, spike and binary parameters ##
######################################

M1 = args.M1*Msun #g 
M2 = args.M2*Msun

M = M1 + M2
nu = (M1*M2)/(M**2.) # symmetric mass ratio

gamma_sp = args.gamma
A = ((3.-gamma_sp)*(0.2**(3.0-gamma_sp))*M1/(2*np.pi))**(1.0/3.0)
r_ref = 1e-6*pc
rho_sp = ((args.rho6*1e16*Msun/(pc**3.))*(r_ref/A)**gamma_sp)**(1/(1-gamma_sp/3)) # cgs 

#Calculate the initial radius based on projected merger time
r_grav = G_N/(c_light**2.) * 2.* M1 #cm
r_isco = 3. * r_grav
r_in = 2.*r_grav

#t_target = 5*year
Nyr_target = 20
r_5yr = (5*year*4*64*G_N**3*M*M1*M2/(5*c_light**5) + r_isco**4)**(1/4)
r_Nyr = (Nyr_target*year*4*64*G_N**3*M*M1*M2/(5*c_light**5) + r_isco**4)**(1/4)

if (args.r_i < 0):
    #print(f"> Starting {Nyr_target} years from merger in vacuum...")
    r_initial = 2.5*r_5yr
else:
    r_initial = args.r_i*pc

if (SHORT):
    r_end = 50*r_isco
else:
    r_end = r_isco
    
print("> System properties:")
print(">    M_1, M_2 [M_sun]: ", M1/Msun, M2/Msun)
print(">    gamma_sp, rho_6 [M_sun/pc^3]: ", gamma_sp, args.rho6*1e16)
print(">    r_i [pc]:", r_initial/pc)
print(" ")
    
################################
##### Equations of motion ######
################################
    

# Radiation reaction
# See, for example, eq. 226 in https://arxiv.org/pdf/1310.1528.pdf
def GW_term(t,r):
    return -64.*(G_N**3.)*M1*M2*(M1+M2)/(5.*(c_light**5.)*(r**3.))


# Gravitational "friction"
# See, for example, https://arxiv.org/pdf/1604.02034.pdf
def DF_term(t_,r_,DF):
    preFactor = 2.*r_**2./(G_N*M1*M2)
    currentPeriod = DF.T_orb(current_r/pc)
    currentV = 2.*np.pi*current_r/currentPeriod
    result = - preFactor * DF.dEdt_DF(r_/pc, v_cut = currentV/km) * km**2. * Msun
    return result
    

#Calculate orbital velocity of the binary
def orbitalV(r_pc, DF):
    currentPeriod = DF.T_orb(r_pc)
    return 2.*np.pi*(r_pc*pc)/currentPeriod # in cgs
    
    
#Derivatives (for feeding into the ODE solver)
def drdt_ode(t,r,arg):
    DF = arg
    return GW_term(t,r) + DF_term(t,r,DF)

def drdt_noDM_ode(t,r):
    return GW_term(t,r)
    


#Initial values of a few different parameters
gamma_initial = gamma_sp
r0_initial = r_initial
DF_current = HaloFeedback.PowerLawSpike(gamma = gamma_initial, M_BH = M1/Msun, M_NS = M2/Msun, rho_sp = rho_sp/(Msun/pc**3.))
f_initial = 2./DF_current.T_orb(r0_initial/pc)


#################################################
############ DYNAMIC DRESS ######################
#################################################
t_dynamic = np.array([0.])
r_dynamic = np.array([r0_initial])
f_dynamic = np.array([f_initial])

start_time = time.time()
print("> Evolving system with DYNAMIC DM DRESS...")

#Grid over which to calculate the density profile
r_grid_pc = np.logspace(-10.,-5.5,num=600, endpoint=True)

DF_current = HaloFeedback.PowerLawSpike(gamma = gamma_initial, M_BH = M1/Msun, M_NS = M2/Msun, rho_sp = rho_sp/(Msun/pc**3.))

rhoeff_dynamic = np.array([DF_current.rho(r0_initial/pc,v_cut=orbitalV(r0_initial/pc, DF_current)/km)])

NPeriods = 1*NPeriods_ini
current_r = r_initial
current_t = 0.
currentPeriod = DF_current.T_orb(current_r/pc)
i = 0

#integrator = ode(drdt_ode).set_integrator(method)
#integrator.set_f_params(DF_current)
#integrator.set_initial_value(r0_initial, current_t) #.set_f_params(2.0).set_jac_params(2.0)

htxt = 'Columns: t [s], r [pc], f_GW [Hz], rho_eff (< v_orb) [Msun/pc^3]'
OUTPUT_ALL = False

#rho = 2.0/3.0
#print("Check out Ralston's/Heun's method... RESCALE THINGS???")
#Check dt size in df1!?!

dN = 1.0*NPeriods_ini

delta_rho = 1e30
SWITCHED = False

#print(r_isco/pc)
#print(180.0*r_isco/pc)
#print(DF_current.r_sp)

while (current_r > r_end):

    currentPeriod = DF_current.T_orb(current_r/pc)
    currentV = 2.*np.pi*current_r/currentPeriod
    current_f = 2./currentPeriod
    
    #if ((i > 0) and (i%1000 == 0)):
    #    dN = 2*dN
    #    print("Increasing!! New value of dN = ", dN)
    
    
    r_old = 1.0*current_r
    
    dt = currentPeriod*dN
       
    #Diagnostic plots
    if (i < -1e6):
        #print(np.trapz(DF_current.DoS*DF_current.dfdt_plus(current_r/pc, currentV/km, v_cut=currentV/km, n_kick=HaloFeedback.N_KICK), DF_current.eps_grid))
        #print(np.trapz(-DF_current.DoS*DF_current.dfdt_minus(current_r/pc, currentV/km, v_cut=currentV/km, n_kick=HaloFeedback.N_KICK), DF_current.eps_grid))
        plt.figure()
    
        plt.loglog(DF_current.eps_grid, dt*DF_current.dfdt_plus(current_r/pc, currentV/km, v_cut=currentV/km, n_kick=HaloFeedback.N_KICK))
        plt.loglog(DF_current.eps_grid, -dt*DF_current.dfdt_minus(current_r/pc, currentV/km, v_cut=currentV/km, n_kick=HaloFeedback.N_KICK))
        #plt.loglog(DF_current.eps_grid,  (2/3)*dt*DF_current.dfdt(current_r/pc, currentV/(km), v_cut=currentV/km)/DF_current.f_eps)
        #plt.loglog(DF_current.eps_grid, -(2/3)*dt*DF_current.dfdt(current_r/pc, currentV/(km), v_cut=currentV/km)/DF_current.f_eps)
    
        E0 = DF_current.psi(current_r/pc)
        plt.axvline(E0, linestyle='--', color='grey')
        plt.axvline(DF_current.psi(DF_current.r_isco), linestyle='--', color='grey')
        plt.axvline(DF_current.eps_grid[0], linestyle='--', color='m')
        
        plt.figure()
    
        #plt.loglog(DF_current.eps_grid, dt*DF_current.dfdt_plus(current_r/pc, currentV/km, v_cut=currentV/km, n_kick=HaloFeedback.N_KICK))
        #plt.loglog(DF_current.eps_grid, -dt*DF_current.dfdt_minus(current_r/pc, currentV/km, v_cut=currentV/km, n_kick=HaloFeedback.N_KICK))
        plt.loglog(DF_current.eps_grid,  dt*DF_current.dfdt(current_r/pc, currentV/(km), v_cut=currentV/km)/DF_current.f_eps)
        plt.loglog(DF_current.eps_grid, -dt*DF_current.dfdt(current_r/pc, currentV/(km), v_cut=currentV/km)/DF_current.f_eps)
    
        E0 = DF_current.psi(current_r/pc)
        plt.axvline(E0, linestyle='--', color='grey')
        plt.axvline(DF_current.psi(DF_current.r_isco), linestyle='--', color='grey')
        plt.axvline(DF_current.eps_grid[0], linestyle='--', color='m')
        #plt.xlim(E0*1e-2, E0*1e2)
        #plt.ylim(1e14, 1e20)
        plt.show()
    
    
    if (i > 0):
        delta_rho = 0.5*(rhoeff_dynamic[-1] - rhoeff_dynamic[-2])/(rhoeff_dynamic[-1] + rhoeff_dynamic[-2])/dN
    
    dN = np.clip(dN, 0, dN_max)
    
    #print(delta_rho)
    #if (((np.abs(delta_rho) > 1e-5) or (current_r > 1e-20*pc)) and (SWITCHED == False)):
    #dN = 10
    dfdt1 = DF_current.dfdt(current_r/pc, currentV/(km), v_cut=currentV/km)
    
    excess_list = -(2/3)*dt*dfdt1/(DF_current.f_eps + 1e-30)
    excess = np.max(excess_list[1:]) #Omit the DF at isco
    #excess_num = np.sum(excess_list > 1)
    if (excess > 1):
        dN /= excess*1.1
        if (verbose > 2):
            print("Too large! New value of dN = ", dN)
        dt = currentPeriod*dN
    
        #dfdt1 = DF_current.dfdt(current_r/pc, currentV/(km), v_cut=currentV/km)

    elif (excess > 1e-1):
        dN /= 1.1
        if (verbose > 2):
            print("Getting large! New value of dN = ", dN)
        dt = currentPeriod*dN
        
        #dfdt1 = DF_current.dfdt(current_r/pc, currentV/(km), v_cut=currentV/km)
        
    elif ((excess < 1e-2) and (i%100 == 0) and (i > 0) and (dN < dN_max)):
        dN *= 1.1
        if (verbose > 2):
            print("Increasing! New value of dN = ", dN)
        dt = currentPeriod*dN
        #df1 = dt*DF_current.dfdt(current_r/pc, currentV/(km), v_cut=currentV/km)


    drdt1 = drdt_ode(current_t, current_r, DF_current)

    current_r += (2/3)*dt*drdt1
    DF_current.f_eps += (2/3)*dt*dfdt1

    currentPeriod = DF_current.T_orb(current_r/pc)
    currentV = 2.*np.pi*current_r/currentPeriod

    drdt2 = drdt_ode(current_t, current_r, DF_current)
    dfdt2 = DF_current.dfdt(current_r/pc, currentV/(km), v_cut=currentV/km)

    current_r += (dt/12)*(9*drdt2 - 5*drdt1)
    DF_current.f_eps += (dt/12)*(9*dfdt2 - 5*dfdt1)
    
    current_t += dt

    if (REFINE):
        if (( np.abs(current_r - r_old) / current_r > 3.e-5) and (dN > 2)):
            dN = np.floor(dN*0.95)
        if (( np.abs(current_r - r_old) / current_r > 3.e-5) and (dN <= 2)):        
            dN = dN * 0.9
    
    #In the dynamic case, we might want to print out the density profile often in
    #the early part of the evolution, for illustration purposes
    if ((i%1000==0) or ((i < 501) and (i%10 == 0)) or ((i < 5001) and (i%100 == 0))):
        if (verbose > 1 and i > 0):
            print(f">    r/r_end = {current_r/r_end:.10f}; f_GW [Hz] = {current_f:.10f}; rho_eff [Msun/pc^3] = {rhoeff_dynamic[-1]:.4e}; delta-rho/rho = {delta_rho:.4e}; dN = {dN:.2f}")
            #print(">    r/r_end = ", current_r/r_end, "; f_GW [Hz] = ", current_f, "; rho_eff [Msun/pc^3] = ", rhoeff_dynamic[-1], "; delta-rho/rho = ", delta_rho)
        #print(">    Time needed so far: %s seconds" % (time.time() - start_time))

        #Update the output file
        if (verbose > 0):
            output2 = list(zip(t_dynamic, r_dynamic/pc, f_dynamic, rhoeff_dynamic))
            nameFileDynamic = output_folder + "trajectory_" + IDstr + ".txt.gz"
            if (OUTPUT): np.savetxt(nameFileDynamic, output2, header=htxt,  fmt='%.10e')

        if (OUTPUT_ALL):
            #Density profiles
            rho_grid     = np.asarray([DF_current.rho(r_) for r_ in r_grid_pc])
            rho_grid_cut = np.asarray([DF_current.rho(r_,v_cut=orbitalV(r_, DF_current)/km) for r_ in r_grid_pc])
        
            timeString = "%3.1f" % (current_t/year)        
    
            #Output a snapshot of the density profiles
            currentSnapshot = np.column_stack((r_grid_pc, rho_grid, rho_grid_cut)) 
            nameFileTxt = output_folder + "DMspike_" + str(i) + "_t_" + timeString + ".dat"
            if (OUTPUT):
                np.savetxt(nameFileTxt, currentSnapshot, header="Columns: r [pc], rho [Msun/pc^3], rho (< v_circ) [Msun/pc^3]")

            #Save current status of the simulation (which might be necessary for restarting or debugging)
            if (OUTPUT): np.savetxt(output_folder + "current_DF.dat", list(zip(DF_current.eps_grid, DF_current.f_eps)), header = "Distribution at step i = " + str(i) + ". Columns: E, f(E)")
            if (OUTPUT): np.savetxt(output_folder + "checkpoint.dat", [NPeriods])
            
    """    
    if (( np.abs(current_r - r_old) / current_r > 3.e-5) and (dN > 2)):
        dN = np.floor(dN*0.95)
    if (( np.abs(current_r - r_old) / current_r > 3.e-5) and (dN <= 2)):        
        dN = dN * 0.9
    """

    #Update the distribution function
    #Note that this is an incredibly simple Euler step. There is definitely a more refined way to do this!
    #integrator.set_f_params(DF_current)
    
    t_dynamic = np.append(t_dynamic, current_t)
    r_dynamic = np.append(r_dynamic, current_r)
    f_dynamic = np.append(f_dynamic, current_f)
    rhoeff_dynamic = np.append(rhoeff_dynamic, DF_current.rho(current_r/pc,v_cut=orbitalV(current_r/pc, DF_current)/km))
    
    i = i+1
   
#Correct final point to be exactly r_end (rather than < r_end)
t_last = np.interp(r_end, r_dynamic, t_dynamic)
f_last = 2/DF_current.T_orb(r_end/pc)

t_dynamic[-1] = t_last
r_dynamic[-1] = r_end
f_dynamic[-1] = f_last
rhoeff_dynamic[-1] = DF_current.rho(r_end*1.000001/pc,v_cut=orbitalV(r_end*1.000001/pc, DF_current)/km)
   
output2 = np.column_stack((t_dynamic, r_dynamic/pc, f_dynamic, rhoeff_dynamic))
nameFileDynamic = output_folder + "trajectory_" + IDstr + ".txt.gz"
if (OUTPUT): np.savetxt(nameFileDynamic, output2, header=htxt,  fmt='%.10e')

#Make some plots

fig, ax = plt.subplots(ncols=2, nrows=1,figsize=(10, 5))
ax[0].semilogy(t_dynamic, r_dynamic/pc)
ax[0].set_xlabel(r"$t$ [s]")
ax[0].set_ylabel(r"$R$ [pc]")
    
ax[1].loglog(r_dynamic/pc, rhoeff_dynamic)
ax[1].set_xlabel(r"$R$ [pc]")
ax[1].set_ylabel(r"$\rho_{\mathrm{eff}, v < v_\mathrm{orb}}(R)$ [$M_\odot\,\mathrm{pc}^{-3}$]")

plt.suptitle(IDstr)
plt.tight_layout()

plt.savefig(output_folder +  "Evolution_" + IDstr + ".pdf", bbox_inches='tight')

print("> Done")               
print("> Time needed: %s seconds" % (time.time() - start_time))                        
      

