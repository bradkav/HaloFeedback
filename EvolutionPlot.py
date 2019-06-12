import numpy as np
import matplotlib.pyplot as plt

import HaloFeedback
from HaloFeedback import G_N

import matplotlib

#Save the plots to file?
SAVE_PLOTS = False
plot_dir = "../../plots/HaloFeedback/"

#Only affect particles below the orbital speed?
SPEED_CUT = False



# Initialise distribution function
DF = HaloFeedback.DistributionFunction(Lambda = np.exp(3.0))

#Radius position and velocity of the orbiting body
r0 = 1.084e-8
v0 = np.sqrt(G_N*DF.M_BH/(r0))

v_cut = -1
file_label = ""
if (SPEED_CUT):
    v_cut = v0
    file_label="speedcut_"

#Orbital time in seconds
T_orb = 2*np.pi*r0*3.0857e13/v0

#Number of orbits to evolve
N_orb = 24000
orbits_per_step = 100
N_step = int(N_orb/orbits_per_step)
dt = T_orb*orbits_per_step

#Number of radial points to calculate the density at
N_r = 200

print("    Number of orbits:", N_orb)
print("    Time [days]:", N_step*dt/(3600*24))

#Initial energy of the halo
E0 = np.trapz(-DF.P_eps()*DF.eps_grid, DF.eps_grid)

#Radial grid for calculating the density
r_list = np.geomspace(DF.r_isco, 100*r0, N_r-1)
r_list =  np.sort(np.append(r_list, r0))
rho_list = np.zeros((N_step, N_r))

#Initial density
if (SPEED_CUT):
    #If we're only interested in particles below the local orbital speed
    rho0 = np.array([DF.rho(r, v_cut = np.sqrt(G_N*DF.M_BH/r)) for r in r_list])
else:
    rho0 = np.array([DF.rho(r) for r in r_list])



#----------- Evolving the system and plotting f(eps) ----------

plt.figure()
cmap = matplotlib.cm.get_cmap('Spectral')


for i in range(N_step):
    #Plot the distribution function f(eps)
    plt.loglog(DF.eps_grid, DF.f_eps,alpha = 0.5,color=cmap(i/N_step))
    
    #Calculate the density profile
    if (SPEED_CUT):
        rho_list[i,:] = np.array([DF.rho(r, v_cut = np.sqrt(G_N*DF.M_BH/r)) for r in r_list])
    else:
        rho_list[i,:]= np.array([DF.rho(r) for r in r_list])
    
    #Time-step using the improved Euler method
    df_dt_1 = DF.dfdt(r0=r0, v_orb=v0, v_cut=v_cut)
    DF.f_eps += df_dt_1*dt
    df_dt_2 = DF.dfdt(r0=r0, v_orb=v0, v_cut=v_cut)
    DF.f_eps += 0.5*dt*(df_dt_2 - df_dt_1)

plt.xlim(1e6, np.max(DF.eps_grid))
plt.ylim(1e0, 1e9)

plt.xlabel(r'$\mathcal{E} = \Psi(r) - \frac{1}{2}v^2$ [(km/s)$^2$]')
plt.ylabel(r'$f(\mathcal{E})$ [$M_\odot$ pc$^{-3}$ (km/s)$^{-3}$]')

for n in [0, N_orb/4, N_orb/2, 3*N_orb/4, N_orb]:
    plt.plot([0,0], [-1, -1], '-', color=cmap(n/N_orb), label = str(int(n)) + ' orbits')
plt.legend(loc='best')

if (SAVE_PLOTS):
    plt.savefig(plot_dir + "f_eps_" + file_label + DF.IDstr_num + ".pdf", bbox_inches='tight')

#---------------- Diagnostics -------------------------

#Final energy of the halo
E1 = np.trapz(-DF.P_eps()*DF.eps_grid, DF.eps_grid)

dE_DF = (1/3.0857e+13)*(N_step*dt)*4*np.pi*G_N**2*1**2*DF.rho_init(r0)*np.log(DF.Lambda)/v0
#print("    Analytic energy gain due to DF:", dE_DF)
#print("    Measured energy gain due to DF:", E1 - E0)
#print("    Fractional error:", -(E1 - E0)/dE_DF)
#print("    Total Halo 'Mass' [final]:", np.trapz(-DF.P_eps(), DF.eps_grid))


#------------------------- Density -------------------------


plt.figure()

for i in range(N_step):
    plt.loglog(r_list,rho_list[i,:], alpha=0.5, color=cmap(i/N_step))
plt.axvline(r0, linestyle='--', color='black')

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

plt.show()

