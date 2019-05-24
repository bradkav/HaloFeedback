import numpy as np
import matplotlib.pyplot as plt

import HaloFeedback
from HaloFeedback import G_N

import matplotlib

from tqdm import tqdm

DF = HaloFeedback.DistributionFunction()

r0 = 1.084e-8
v0 = np.sqrt(G_N*DF.M_BH/(r0))
#print(v0)

T_orb = 105.6
#N_orb = 2
#N_orb = 24000
N_orb = 100000
orbits_per_step = 200
N_step = int(N_orb/orbits_per_step)
dt = T_orb*orbits_per_step
N_r = 200

print("    The initial value of g(v) is 0.582123 for all r!")

print("    Total Halo 'Mass' [initial]:", np.trapz(-DF.P_eps(), DF.eps_grid))

print("    Number of orbits:", N_orb)
print("    Time [days]:", N_step*dt/(3600*24))

E0 = np.trapz(-DF.P_eps()*DF.eps_grid, DF.eps_grid)

r_list = np.geomspace(DF.r_isco, 100*r0, N_r-1)
r_list = np.sort(np.append(r_list, r0))
rho_list = np.zeros((N_step, N_r))

rho0 = np.array([DF.rho(r, v_cut = np.sqrt(G_N*DF.M_BH/(r))) for r in r_list])
#rho0 = np.array([DF.rho(r) for r in r_list])

#plt.figure()

#plt.loglog(DF.eps_grid, DF.f_eps + DF.dfdt_plus(r0=r0, v_orb=v0, v_cut=v0)*dt)
#plt.loglog(DF.eps_grid, DF.f_eps + DF.dfdt_minus(r0=r0, v_orb=v0, v_cut=v0)*dt)

delta_eps_list, frac_list = DF.calc_delta_eps(v0)





plt.figure()

#print("Check Mass conservation!")
#v_cut= np.sqrt(G_N*DF.M_BH/(r))

cmap = matplotlib.cm.get_cmap('Spectral')

#cmap = matplotlib.cm.get_cmap('Blues')

for i in tqdm(range(N_step)):
    plt.loglog(DF.eps_grid, DF.f_eps, alpha = 0.5,color=cmap(i/N_step))
    rho_list[i,:] = np.array([DF.rho(r, v_cut = np.sqrt(G_N*DF.M_BH/(r))) for r in r_list])
    df_dt_1 = DF.dfdt(r0=r0, v_orb=v0, v_cut = v0)
    DF.f_eps += df_dt_1*dt
    df_dt_2 = DF.dfdt(r0=r0, v_orb=v0, v_cut = v0)
    DF.f_eps += 0.5*dt*(df_dt_2 - df_dt_1)
    
for i in range(20):
    plt.axvline(G_N*DF.M_BH/(r0) + delta_eps_list[0]*i, linestyle='--', color='k')
#plt.axvline(G_N*DF.M_BH/(r0) + delta_eps_list[0], linestyle='--', color='k')
    

plt.ylim(1e-6, 1e9)

plt.xlabel(r'$\mathcal{E} = \Psi(r) - \frac{1}{2}v^2$ [km/s]')
plt.ylabel(r'$f(\mathcal{E})$ [$M_\odot$ pc$^{-3}$ (km/s)$^{-3}$]')

for n in [0, N_orb/4, N_orb/2, 3*N_orb/4, N_orb]:
    plt.plot([0,0], [-1, -1], '-', color=cmap(n/N_orb), label = str(int(n)) + ' orbits')
plt.legend(loc='lower right')



#---------------- Diagnostics -------------------------

E1 = np.trapz(-DF.P_eps()*DF.eps_grid, DF.eps_grid)

dE_DF = (1/3.0857e+13)*0.583*(N_step*dt)*4*np.pi*G_N**2*1**2*DF.rho_init(r0)*np.log(HaloFeedback.Lambda)/v0
print("    Analytic DF:", dE_DF)
print("    Measured DF:", E1 - E0)
print("    Fractional error:", -(E1 - E0)/dE_DF)


print("    Total Halo 'Mass' [final]:", np.trapz(-DF.P_eps(), DF.eps_grid))


#------------------------ Density ratio -----------------

plt.figure()

#for i in range(1000:)
for i in range(N_step):
    plt.semilogx(r_list,rho_list[i,:]/rho0, alpha=0.5, color=cmap(i/N_step))
plt.axvline(r0, linestyle='--', color='black')
#plt.loglog(DF.eps_grid, DF.f_eps+DF.dfdt(r0=r0, v_orb=v0, v_cut=np.sqrt(2)*v0)*106*10)

for n in [0, N_orb/4, N_orb/2, 3*N_orb/4, N_orb]:
    plt.plot([0,0], [-1, -1], '-', color=cmap(n/N_orb), label = str(int(n)) + ' orbits')
plt.legend(loc='lower right')

plt.xlabel(r'$r$ [pc]')
plt.ylabel(r'$\rho_{v < v_\mathrm{orb}}(r)/\rho_{0, v < v_\mathrm{orb}}(r)$')
plt.ylim(0, 2.0)

b_min = G_N*DF.M_NS/(v0**2)
b_max = b_min*HaloFeedback.Lambda

#plt.axvline(r0 - b_max, color='k', linestyle='--')
#plt.axvline(r0 + b_max, color='k', linestyle='--')

plt.savefig("../plots/Density_ratio_speedcut_" + DF.strID + ".pdf", bbox_inches='tight')


#------------------------- Density -------------------------


plt.figure()

#for i in range(1000:)
for i in range(N_step):
    plt.loglog(r_list,rho_list[i,:], alpha=0.5, color=cmap(i/N_step))
plt.axvline(r0, linestyle='--', color='black')
#plt.loglog(DF.eps_grid, DF.f_eps+DF.dfdt(r0=r0, v_orb=v0, v_cut=np.sqrt(2)*v0)*106*10)

for n in [0, N_orb/4, N_orb/2, 3*N_orb/4, N_orb]:
    plt.plot([0,0], [-1, -1], '-', color=cmap(n/N_orb), label = str(int(n)) + ' orbits')
plt.legend(loc='best')

#plt.ylim(1e18, 1e23)

plt.xlabel(r'$r$ [pc]')
plt.ylabel(r'$\rho_{v < v_\mathrm{orb}}(r)$ [$M_\odot$ pc$^{-3}$]')

plt.savefig("../plots/Density_speedcut_" + DF.strID + ".pdf", bbox_inches='tight')

plt.show()