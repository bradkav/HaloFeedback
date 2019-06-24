import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import cumtrapz

import HaloFeedback
from HaloFeedback import G_N

import matplotlib

#Save the plots to file?
SAVE_PLOTS = False
plot_dir = "../../plots/HaloFeedback/"

#Only affect particles below the orbital speed?
SPEED_CUT = True

#Specify which method to use for the calculations
# METHOD = 1 is simpler, but METHOD = 2 should be more accurate
METHOD = 2

# Initialise distribution function
DF = HaloFeedback.DistributionFunction(Lambda = np.exp(3.0), M_BH = 1000)

#Radius position and velocity of the orbiting body
r0 = 1.084e-8
v0 = np.sqrt(G_N*DF.M_BH/(r0))

v_cut = -1
file_label = ""
if (SPEED_CUT):
    v_cut = v0
    file_label="speedcut_Method" + str(METHOD) + "_"

#Orbital time in seconds
T_orb = 2*np.pi*r0*3.0857e13/v0

#Number of orbits to evolve
N_orb = 24000
orbits_per_step = 100
N_step = int(N_orb/orbits_per_step)

dt = T_orb*orbits_per_step

t_list = dt*N_step*np.linspace(0, 1, N_step + 1)

#Number of radial points to calculate the density at
N_r = 200

print("    Number of orbits:", N_orb)
print("    Time [days]:", N_step*dt/(3600*24))

#Initial energy of the halo
E0 = DF.TotalEnergy()
#print(DF.TotalMass())

#Radial grid for calculating the density
#r_list = np.geomspace(DF.r_isco, 1e7*r0, N_r-1)
r_list = np.geomspace(DF.r_isco, 1e3*r0, N_r-1)
r_list =  np.sort(np.append(r_list, r0))
rho_list = np.zeros((N_step+1, N_r))
rho_avg_list = np.zeros(N_step+1)

#Which index refers to r0?
r0_ind = np.where(r_list == r0)[0][0]

M_list = np.zeros(N_step)

DF_list = np.zeros(N_step+1)

#Calculate initial DF energy loss rate
DF_list[0] = DF.dEdt_DF(r0, SPEED_CUT)

#Initial density
if (SPEED_CUT):
    #If we're only interested in particles below the local orbital speed
    rho0 = np.array([DF.rho(r, v_cut = np.sqrt(G_N*DF.M_BH/r)) for r in r_list])
else:
    rho0 = np.array([DF.rho(r) for r in r_list])

rho_list[0,:] = rho0
rho_avg_list[0] = ((r0 - DF.b_max(v0))*DF.rho(r0 - DF.b_max(v0)) + (r0 + DF.b_max(v0))*DF.rho(r0 + DF.b_max(v0)) )/(2*r0)


rho0_full = np.array([DF.rho(r) for r in r_list])
E0_alt = 0.5*4*np.pi*np.trapz(rho0_full*r_list**2*DF.psi(r_list), r_list)


print("    ")
print("    Initial energy of the halo [(km/s)^2]:", E0)
print("    Initial energy of the halo, alternative [(km/s)^2]:", E0_alt)
#print(E0/E0_alt)
print("    ")

M0 = DF.TotalMass()
M0_alt = 4*np.pi*np.trapz(rho0_full*r_list**2, r_list)
print("    Initial mass of the halo [M_sun]:", M0_alt)

delta_eps = np.zeros(N_step + 1)
delta_eps[0] = 0

#----------- Evolving the system and plotting f(eps) ----------

#M_list[0] = M0

plt.figure()
cmap = matplotlib.cm.get_cmap('Spectral')


for i in range(N_step):
    #Plot the distribution function f(eps)
    plt.loglog(DF.eps_grid, DF.f_eps,alpha = 0.5,color=cmap(i/N_step))
    
    #Calculate the density profile
    if (SPEED_CUT):
        rho_list[i+1,:] = np.array([DF.rho(r, v_cut = np.sqrt(G_N*DF.M_BH/r)) for r in r_list])
    else:
        rho_list[i+1,:]= np.array([DF.rho(r) for r in r_list])
    
    
    M_list[i] = DF.TotalMass()
    #Time-step using the improved Euler method
    df_dt_1 = DF.dfdt(r0=r0, v_orb=v0, v_cut=v_cut, method=METHOD)
    DF.f_eps += df_dt_1*dt
    df_dt_2 = DF.dfdt(r0=r0, v_orb=v0, v_cut=v_cut, method=METHOD)
    DF.f_eps += 0.5*dt*(df_dt_2 - df_dt_1)
    
    #Change in energy of the halo
    delta_eps[i+1] = DF.TotalEnergy() - E0

    #Dynamical friction energy loss
    DF_list[i+1] = DF.dEdt_DF(r0, SPEED_CUT)


#plt.xlim(1e6, np.max(DF.eps_grid))
plt.xlim(1.5e8, 4.5e8)
#plt.ylim(1e0, 1e9)
plt.ylim(1e2, 1e8)

plt.axvline(G_N*DF.M_BH/r0, linestyle='--', color='k')

plt.xlabel(r'$\mathcal{E} = \Psi(r) - \frac{1}{2}v^2$ [(km/s)$^2$]')
plt.ylabel(r'$f(\mathcal{E})$ [$M_\odot$ pc$^{-3}$ (km/s)$^{-3}$]')

for n in [0, N_orb/4, N_orb/2, 3*N_orb/4, N_orb]:
    plt.plot([0,0], [-1, -1], '-', color=cmap(n/N_orb), label = str(int(n)) + ' orbits')
plt.legend(loc='best')

if (SAVE_PLOTS):
    plt.savefig(plot_dir + "f_eps_" + file_label + DF.IDstr_num + ".pdf", bbox_inches='tight')

#------------------------- Density -------------------------


plt.figure()

for i in range(N_step):
    plt.loglog(r_list,rho_list[i,:], alpha=0.5, color=cmap(i/N_step))
plt.axvline(r0, linestyle='--', color='black')
plt.axvline(r0 + DF.b_max(v0), linestyle=':', color='black')
plt.axvline(r0 - DF.b_max(v0), linestyle=':', color='black')

for n in [0, N_orb/4, N_orb/2, 3*N_orb/4, N_orb]:
    plt.plot([0,0], [-1, -1], '-', color=cmap(n/N_orb), label = str(int(n)) + ' orbits')
plt.legend(loc='best')


plt.xlabel(r'$r$ [pc]')
if (SPEED_CUT):
    plt.ylabel(r'$\rho_{v < v_\mathrm{orb}}(r)$ [$M_\odot$ pc$^{-3}$]')
else:   
    plt.ylabel(r'$\rho(r)$ [$M_\odot$ pc$^{-3}$]')

if (SAVE_PLOTS):
    plt.savefig(plot_dir + "Density_" + file_label + DF.IDstr_num + ".pdf", bbox_inches='tight')


#------------------------ Density ratio -----------------

plt.figure()


for i in range(N_step):
    plt.semilogx(r_list,rho_list[i,:]/rho0, alpha=0.5, color=cmap(i/N_step))
plt.axvline(r0, linestyle='--', color='black')
plt.axvline(r0 + DF.b_max(v0), linestyle=':', color='black')
plt.axvline(r0 - DF.b_max(v0), linestyle=':', color='black')


for n in [0, N_orb/4, N_orb/2, 3*N_orb/4, N_orb]:
    plt.plot([0,0], [-1, -1], '-', color=cmap(n/N_orb), label = str(int(n)) + ' orbits')
plt.legend(loc='lower right')

#plt.axvline(G_N*DF.M_BH/(G_N*DF.M_BH/r0 + v0**2), linestyle='--', color='k')
#plt.axvline(G_N*DF.M_BH/(G_N*DF.M_BH/r0 - 0.5*v0**2), linestyle='--', color='k')

plt.xlabel(r'$r$ [pc]')
if (SPEED_CUT):
    plt.ylabel(r'$\rho_{v < v_\mathrm{orb}}(r)/\rho_{0,v < v_\mathrm{orb}}(r)$')
else:
    plt.ylabel(r'$\rho(r)/\rho_0(r)$')
    
plt.ylim(0, 2.0)

if (SAVE_PLOTS):
    plt.savefig(plot_dir + "Density_ratio_" + file_label + DF.IDstr_num + ".pdf", bbox_inches='tight')


#---------------- Energy Conservation -----------------

DeltaE = cumtrapz(DF_list, t_list, initial=0)

plt.figure()

plt.plot(t_list/T_orb, DeltaE, label="Halo")
plt.plot(t_list/T_orb, -delta_eps, linestyle='--',label="NS")
plt.plot(t_list/T_orb, t_list*DF_list[0], linestyle=':',label="NS (linearised)")

plt.xlabel("Number of orbits")
plt.ylabel(r"$|\Delta E|$ [$M_\odot$ (km/s)$^2$]")

plt.legend(loc='best')

if (SAVE_PLOTS):
    plt.savefig(plot_dir + "DeltaE_" + file_label + DF.IDstr_num + ".pdf", bbox_inches='tight')


#---------------- Diagnostics -------------------------

# Changing density over time
#Delta_rho_dt = cumtrapz(rho_avg_list, t_list, initial=0)/rho_avg_list[0]



rho_full = np.array([DF.rho(r) for r in r_list])
Ef_alt = 0.5*4*np.pi*np.trapz(r_list**2*rho_full*DF.psi(r_list), r_list)

#print(DF.rho(r0), rho_avg_list[-1])

print("  ")
print("   Fractional Change in halo mass:", (DF.TotalMass() - M0)/M0)
print("   Change in halo energy [(km/s)^2]:", DF.TotalEnergy() - E0)
#print("   Change in halo energy (2):", Ef_alt - E0_alt)

print("  ")
print("   Dynamical friction energy change [(km/s)^2]:", DeltaE[-1])
print("   Fractional error in DF:", ((DF.TotalEnergy() - E0) + DeltaE[-1])/(DeltaE[-1]))

plt.show()

