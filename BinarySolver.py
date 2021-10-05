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

#Parse the arguments!                                                       
parser = argparse.ArgumentParser(description='...')
parser.add_argument('-M1', '--M1', help='Larger BH mass M1 in M_sun', type=float, default=1000.)
parser.add_argument('-M2', '--M2', help="Smalller BH mass M2 in M_sun", type=float, default = 1.0)
parser.add_argument('-rho6', '--rho6', help='Spike density normalisation [1e16 M_sun/pc^3]', type=float, default=0.5)
parser.add_argument('-gamma', '--gamma', help='slope of DM spike', type=float, default=2.333)

parser.add_argument('-r_i', '--r_i', help='Initial radius in pc', type=float, default = -1)
parser.add_argument('-short', '--short', help='Set to 1 to finish before r_isco', type=int, default = 0)
parser.add_argument('-dN_ini', '--dN_ini', help='Initial time-step size in orbits', type=int, default = 500)

parser.add_argument('-IDtag', '--IDtag', help='Optional IDtag to add on the end of the file names', type=str, default="NONE")

#Add an *OPTIONAL* ID tag or ID string here...
args = parser.parse_args()
IDstr = "M1_%.1f_M2_%.1f_rho6_%.4f_gamma_%.4f"%(args.M1, args.M2, args.rho6, args.gamma) 
if (args.IDtag != "NONE"):
    IDstr += "_" + args.IDtag

print("> Run ID: ", IDstr)

#Generate folder structure if needed
OUTPUT = True
if (OUTPUT):
    if not (os.path.isdir("runs/")):
        os.mkdir("runs/")
    output_folder = "runs/" + IDstr + "/"
    if not (os.path.isdir(output_folder)):
        print("> Creating folder: ", output_folder)
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

t_target = 5*year
r_5yr = (t_target*4*64*G_N**3*M*M1*M2/(5*c_light**5) + r_isco**4)**(1/4)

if (args.r_i < 0):
    r_initial = 3*r_5yr
else:
    r_initial = args.r_i*pc

if (SHORT):
    r_end = 10*r_isco
else:
    r_end = r_isco
    
print("> System properties:")
print(">    M_1, M_2 [M_sun]: ", M1/Msun, M2/Msun)
print(">    gamma_sp, rho_6 [M_sun/pc^3]: ", gamma_sp, args.rho6/(Msun*pc**-3.))
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
    preFactor = 2.*r_[0]**2./(G_N*M1*M2)
    currentPeriod = DF.T_orb(current_r/pc)
    currentV = 2.*np.pi*current_r/currentPeriod
    result = - preFactor * DF.dEdt_DF(r_[0]/pc, v_cut = currentV/km) * km**2. * Msun
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
    


#Integration method
method = "dopri5" 

#Initial values of a few different parameters
gamma_initial = gamma_sp
r0_initial = r_initial
DF_current = HaloFeedback.PowerLawSpike(gamma = gamma_initial, M_BH = M1/Msun, M_NS = M2/Msun, rho_sp = rho_sp/(Msun/pc**3.))
f_initial = 1./DF_current.T_orb(r0_initial/pc)

#############################################################
########### INTEGRATING THE VACUUM SYSTEM ###################
#############################################################

t_vacuum = np.array([0.])
r_vacuum = np.array([r0_initial])
f_vacuum = np.array([f_initial])

start_time = time.time()
print("> Evolving system in VACUUM...")               

DF_vacuum = HaloFeedback.PowerLawSpike(gamma = gamma_initial, M_BH = M1/Msun, M_NS = M2/Msun, rho_sp = rho_sp/(Msun/pc**3.))

integrator = ode(drdt_noDM_ode).set_integrator(method) #('dopri5')
integrator.set_initial_value(r0_initial, 0.) #.set_f_params(2.0).set_jac_params(2.0)

NPeriods = 1*NPeriods_ini
current_r = r_initial
current_t = 0.
currentPeriod = DF_vacuum.T_orb(current_r/pc)
i = 0

while integrator.successful() and (current_r > r_end):

    #Very crude Euler integrator
    dt = currentPeriod*NPeriods
    current_t = integrator.t + dt
    r_old = 1.0*current_r
    current_r = integrator.integrate(integrator.t + dt)[0]
    
    currentPeriod = DF_vacuum.T_orb(current_r/pc)
    current_f = 1./currentPeriod 
       
    t_vacuum = np.append(t_vacuum, current_t)
    r_vacuum = np.append(r_vacuum, current_r)
    f_vacuum = np.append(f_vacuum, current_f)
    
    if (i%5000==0):
        print(">    r/r_end = ", current_r/r_end, "; f_orb [Hz] = ", current_f)    
    
    #Manually adjust step sizes
    if (( np.abs(current_r - r_old) / current_r > 3.e-5) and (NPeriods > 2)):
        NPeriods = np.floor(NPeriods*0.95)
    if (( np.abs(current_r - r_old) / current_r > 3.e-5) and (NPeriods <= 2)):        
        NPeriods = NPeriods * 0.9

    i = i+1

output_vac = np.column_stack((t_vacuum, r_vacuum/pc, f_vacuum))
nameFileVacuum = output_folder + "output_vacuum.dat"
if (OUTPUT): np.savetxt(nameFileVacuum, output_vac, header="Columns: t [s], r [pc], f_orb [Hz]")

print("> Done")               
print("> Time needed: %s seconds" % (time.time() - start_time))
print(" ")


#################################################
############ STATIC DRESS #######################
#################################################
t_static = np.array([0.])
r_static = np.array([r0_initial])
f_static = np.array([f_initial])

start_time = time.time()
print("> Evolving system with STATIC DM DRESS...")               

DF_static = HaloFeedback.PowerLawSpike(gamma = gamma_initial, M_BH = M1/Msun, M_NS = M2/Msun, rho_sp = rho_sp/(Msun/pc**3.))

integrator = ode(drdt_ode).set_integrator(method)
integrator.set_f_params(DF_static)
integrator.set_initial_value(r0_initial, 0.)

NPeriods = 1*NPeriods_ini
current_r = r_initial
current_t = 0.
currentPeriod = DF_vacuum.T_orb(current_r/pc)
i = 0

while integrator.successful() and (current_r > r_end):
    
    dt = currentPeriod*NPeriods
    current_t = integrator.t + dt
    r_old = 1.0*current_r
    current_r = integrator.integrate(integrator.t + dt)[0]
    
    currentPeriod = DF_vacuum.T_orb(current_r/pc)
    current_f = 1./currentPeriod 
       
    t_static = np.append(t_static, current_t)
    r_static = np.append(r_static, current_r)
    f_static = np.append(f_static, current_f)

    if (i%5000==0):
        print(">    r/r_end = ", current_r/r_end, "; f_orb [Hz] = ", current_f)    
    
    if (( np.abs(current_r - r_old) / current_r > 3.e-5) and (NPeriods > 2)):
        NPeriods = np.floor(NPeriods*0.95)
    if (( np.abs(current_r - r_old) / current_r > 3.e-5) and (NPeriods <= 2)):        
        NPeriods = NPeriods * 0.9
        
    i = i+1

output1 = np.column_stack((t_static, r_static/pc, f_static))
nameFileStatic = output_folder + "output_static_dress_" + IDstr + ".dat"
if (OUTPUT): np.savetxt(nameFileStatic, output1, header="Columns: t [s], r [pc], f_orb [Hz]")

print("> Done")               
print("> Time needed: %s seconds" % (time.time() - start_time))
print(" ")


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

NPeriods = 1*NPeriods_ini
current_r = r_initial
current_t = 0.
currentPeriod = DF_vacuum.T_orb(current_r/pc)
i = 0

integrator = ode(drdt_ode).set_integrator(method)
integrator.set_f_params(DF_current)
integrator.set_initial_value(r0_initial, current_t) #.set_f_params(2.0).set_jac_params(2.0)

while integrator.successful() and (current_r > r_end):

    dt = currentPeriod*NPeriods
    current_t = integrator.t + dt
    r_old = 1.0*current_r
    current_r = integrator.integrate(integrator.t + dt)[0]


    currentPeriod = DF_current.T_orb(current_r/pc)
    currentV = 2.*np.pi*current_r/currentPeriod
    current_f = 1./currentPeriod
       
    t_dynamic = np.append(t_dynamic, current_t)
    r_dynamic = np.append(r_dynamic, current_r)
    f_dynamic = np.append(f_dynamic, current_f)
    
    #In the dynamic case, we might want to print out the density profile often in
    #the early part of the evolution, for illustration purposes
    if ((i%1000==0) or ((i < 501) and (i%10 == 0)) or ((i < 5001) and (i%100 == 0))):
        print(">    r/r_end = ", current_r/r_end, "; f_orb [Hz] = ", current_f)
        #print(">    Time needed so far: %s seconds" % (time.time() - start_time))

        #Density profiles
        rho_grid     = np.asarray([DF_current.rho(r_) for r_ in r_grid_pc])
        rho_grid_cut = np.asarray([DF_current.rho(r_,v_cut=orbitalV(r_, DF_current)/km) for r_ in r_grid_pc])
        
        timeString = "%3.1f" % (current_t/year)        
    
        #Output a snapshot of the density profiles
        currentSnapshot = np.column_stack((r_grid_pc, rho_grid, rho_grid_cut)) 
        nameFileTxt = output_folder + "DMspike_" + str(i) + "_t_" + timeString + ".dat"
        if (OUTPUT):
            np.savetxt(nameFileTxt, currentSnapshot, header="Columns: r [pc], rho [Msun/pc^3], rho (< v_circ) [Msun/pc^3]")

        #Update the output file
        output2 = list(zip(t_dynamic, r_dynamic/pc, f_dynamic))
        nameFileDynamic = output_folder + "output_dynamic_dress_" + IDstr + ".dat"
        if (OUTPUT): np.savetxt(nameFileDynamic, output2, header="Columns: t [s], r [pc], f_orb [Hz]")

        #Save current status of the simulation (which might be necessary for restarting or debugging)
        if (OUTPUT): np.savetxt(output_folder + "current_DF.dat", list(zip(DF_current.eps_grid, DF_current.f_eps)), header = "Distribution at step i = " + str(i) + ". Columns: E, f(E)")
        if (OUTPUT): np.savetxt(output_folder + "checkpoint.dat", [NPeriods])
        
    if (( np.abs(current_r - r_old) / current_r > 3.e-5) and (NPeriods > 2)):
        NPeriods = np.floor(NPeriods*0.95)
    if (( np.abs(current_r - r_old) / current_r > 3.e-5) and (NPeriods <= 2)):        
        NPeriods = NPeriods * 0.9

    #Update the distribution function
    #Note that this is an incredibly simple Euler step. There is definitely a more refined way to do this!
    DF_current.f_eps += DF_current.delta_f(current_r/pc, currentV/(km), dt, v_cut=currentV/km)
    integrator.set_f_params(DF_current)
    
    i = i+1
   
output2 = np.column_stack((t_dynamic, r_dynamic/pc, f_dynamic))
nameFileDynamic = output_folder + "output_dynamic_dress_" + IDstr + ".dat"
if (OUTPUT): np.savetxt(nameFileDynamic, output2) 

print("> Done")               
print("> Time needed: %s seconds" % (time.time() - start_time))                        
      

