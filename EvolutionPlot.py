import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import cumtrapz

import HaloFeedback
from HaloFeedback import G_N

from matplotlib import gridspec
import matplotlib

# Save the plots to file?
SAVE_PLOTS = True
plot_dir = "plots/"

# Only affect particles below the orbital speed?
SPEED_CUT = True

# Initialise distribution function
DF = HaloFeedback.DistributionFunction(M_BH=1000)

# Radius position and velocity of the orbiting body
r0 = 1e-8  # pc
v0 = np.sqrt(G_N * (DF.M_BH + DF.M_NS) / (r0))

v_cut = -1
file_label = ""
if SPEED_CUT:
    v_cut = v0
    file_label = "speedcut_"

# Orbital time in seconds
T_orb = 2 * np.pi * r0 * 3.0857e13 / v0

# Number of orbits to evolve
N_orb = 40000
orbits_per_step = 250
N_step = int(N_orb / orbits_per_step)

dt = T_orb * orbits_per_step

t_list = dt * N_step * np.linspace(0, 1, N_step + 1)


print("    Number of orbits:", N_orb)
print("    Total Time [days]:", N_step * dt / (3600 * 24))


# Radial grid for calculating the density
N_r = 200
r_list = np.geomspace(0.9e-9, 1.1e-7, N_r - 1)
r_list = np.sort(np.append(r_list, r0))

# List of densities
rho_list = np.zeros((N_step + 1, N_r))

# Which index refers to r0?
r0_ind = np.where(r_list == r0)[0][0]

# Keep track of halo mass
M_list = np.zeros(N_step)
M_list[0] = DF.TotalMass()

# Rate of dynamical friction energy loss
DF_list = np.zeros(N_step + 1)
DF_list[0] = DF.dEdt_DF(r0, v_cut)

# Total energy of the halo
E_list = np.zeros(N_step + 1)
E_list[0] = DF.TotalEnergy()

# Keep track of how much energy is carried
# away by ejected particles
E_ej_tot = 0.0 * t_list


# Initial density
if SPEED_CUT:
    # If we're only interested in particles below the local orbital speed
    rho0 = np.array(
        [DF.rho(r, v_cut=np.sqrt(G_N * (DF.M_BH + DF.M_NS) / r)) for r in r_list]
    )
else:
    rho0 = np.array([DF.rho(r) for r in r_list])

rho_list[0, :] = rho0

# ----------- Evolving the system and plotting f(eps) ----------

cmap = matplotlib.cm.get_cmap("Spectral")

plt.figure()

for i in range(N_step):
    # Plot the distribution function f(eps)
    plt.semilogy(DF.eps_grid, DF.f_eps, alpha=0.5, color=cmap(i / N_step))

    # Calculate the density profile at this timestep
    if SPEED_CUT:
        rho_list[i + 1, :] = np.array(
            [DF.rho(r, v_cut=np.sqrt(G_N * (DF.M_BH + DF.M_NS) / r)) for r in r_list]
        )
    else:
        rho_list[i + 1, :] = np.array([DF.rho(r) for r in r_list])

    # Total halo mass
    M_list[i] = DF.TotalMass()

    # Total energy carried away so far by unbound particles
    E_ej_tot[i + 1] = E_ej_tot[i] + DF.dEdt_ej(r0=r0, v_orb=v0, v_cut=v_cut) * dt

    # Time-step using the improved Euler method
    df1 = DF.delta_f(r0=r0, v_orb=v0, dt=dt, v_cut=v_cut)
    DF.f_eps += df1
    df2 = DF.delta_f(r0=r0, v_orb=v0, dt=dt, v_cut=v_cut)
    DF.f_eps += 0.5 * (df2 - df1)

    # Change in energy of the halo
    E_list[i + 1] = DF.TotalEnergy()

    # Dynamical friction energy loss rate
    DF_list[i + 1] = DF.dEdt_DF(r0, v_cut)


plt.xlim(1.0e8, 4.5e8)
plt.ylim(1e3, 1e9)

plt.axvline(G_N * DF.M_BH / r0, linestyle="--", color="k")

plt.xlabel(r"$\mathcal{E} = \Psi(r) - \frac{1}{2}v^2$ [(km/s)$^2$]")
plt.ylabel(r"$f(\mathcal{E})$ [$M_\odot$ pc$^{-3}$ (km/s)$^{-3}$]")

for n in [0, N_orb / 4, N_orb / 2, 3 * N_orb / 4, N_orb]:
    plt.plot(
        [0, 0], [-1, -1], "-", color=cmap(n / N_orb), label=str(int(n)) + " orbits"
    )
plt.legend(loc="best")

if SAVE_PLOTS:
    plt.savefig(
        plot_dir + "f_eps_" + file_label + DF.IDstr_num + ".pdf", bbox_inches="tight"
    )


# ------------------------- Density -------------------------


fig = plt.figure(figsize=(6, 6))

gs = fig.add_gridspec(4, 4)
ax0 = fig.add_subplot(gs[1:, :])
ax1 = fig.add_subplot(gs[0, :])


for i in range(N_step):
    ax0.loglog(r_list, rho_list[i, :], alpha=0.5, color=cmap(i / N_step))
    ax1.semilogx(r_list, rho_list[i, :] / rho0, alpha=0.5, color=cmap(i / N_step))
ax0.axvline(r0, linestyle="--", color="black")

for n in [0, N_orb / 4, N_orb / 2, 3 * N_orb / 4, N_orb]:
    ax0.plot(
        [0, 0], [-1, -1], "-", color=cmap(n / N_orb), label=str(int(n)) + " orbits"
    )

ax0.plot([0, 0], [-1, -1], "w-", label="($\sim$43 days)")
ax0.legend(loc="best", fontsize=12)

ax0.set_xlim(1e-9, 1e-7)
ax0.set_ylim(1e18, 1e22)

ax0.text(
    1.1e-9,
    1.2e18,
    "$m_1 = 1000\\,M_\\odot$\n$m_2 = 1\\,M_\\odot$",
    ha="left",
    va="bottom",
    fontsize=14,
)
ax0.text(
    0.92e-8, 2e21, "Orbital radius", ha="center", va="center", fontsize=12, rotation=90
)

ax0.set_xlabel(r"$r$ [pc]")
if SPEED_CUT:
    ax0.set_ylabel(r"$\rho_{v < v_\mathrm{orb}}(r)$ [$M_\odot$ pc$^{-3}$]")
else:
    ax0.set_ylabel(r"$\rho(r)$ [$M_\odot$ pc$^{-3}$]")

ax1.axvline(r0, linestyle="--", color="black")
if SPEED_CUT:
    ax1.set_ylabel(
        r"$\frac{\rho_{v < v_\mathrm{orb}}(r)}{\rho_{0,v < v_\mathrm{orb}}(r)}$"
    )
else:
    ax1.set_ylabel(r"$\rho(r)/\rho_0(r)$")

ax1.set_xlim(1e-9, 1e-7)
ax1.set_ylim(0, 2.0)
ax1.set_yticks(np.linspace(0, 2, 21), minor=True)
ax1.set_xticklabels([])

if SAVE_PLOTS:
    plt.savefig(
        plot_dir + "Density_" + file_label + DF.IDstr_num + ".pdf", bbox_inches="tight"
    )


# ------------------------ Density ratio -----------------

plt.figure()


for i in range(N_step):
    plt.semilogx(r_list, rho_list[i, :] / rho0, alpha=0.5, color=cmap(i / N_step))
plt.axvline(r0, linestyle="--", color="black")


for n in [0, N_orb / 4, N_orb / 2, 3 * N_orb / 4, N_orb]:
    plt.plot(
        [0, 0], [-1, -1], "-", color=cmap(n / N_orb), label=str(int(n)) + " orbits"
    )
plt.legend(loc="lower right")

plt.xlabel(r"$r$ [pc]")
if SPEED_CUT:
    plt.ylabel(r"$\rho_{v < v_\mathrm{orb}}(r)/\rho_{0,v < v_\mathrm{orb}}(r)$")
else:
    plt.ylabel(r"$\rho(r)/\rho_0(r)$")

plt.ylim(0, 2.0)

if SAVE_PLOTS:
    plt.savefig(
        plot_dir + "Density_ratio_" + file_label + DF.IDstr_num + ".pdf",
        bbox_inches="tight",
    )


# ---------------- Energy Conservation -----------------

DeltaE = cumtrapz(DF_list, t_list, initial=0)

plt.figure()

plt.plot(
    t_list / T_orb,
    -((E_list - E_list[0]) + E_ej_tot),
    linestyle="-",
    label="DM Halo + Ejected particles",
)
plt.plot(t_list / T_orb, DeltaE, linestyle="--", label="Dynamical Friction")
plt.plot(
    t_list / T_orb,
    t_list * DF_list[0],
    linestyle=":",
    label="Dynamical Friction (linearised)",
)

plt.xlabel("Number of orbits")
plt.ylabel(r"$|\Delta E|$ [$M_\odot$ (km/s)$^2$]")

plt.legend(loc="best", fontsize=14)

if SAVE_PLOTS:
    plt.savefig(
        plot_dir + "DeltaE_" + file_label + DF.IDstr_num + ".pdf", bbox_inches="tight"
    )


# ---------------- Diagnostics -------------------------


rho_full = np.array([DF.rho(r) for r in r_list])
Ef_alt = 0.5 * 4 * np.pi * np.trapz(r_list ** 2 * rho_full * DF.psi(r_list), r_list)

print("  ")
print("    Change in halo energy [(km/s)^2]:", DF.TotalEnergy() - E_list[0])
print("    Energy in ejected particles:", E_ej_tot[-1])
print("    Dynamical friction energy change [(km/s)^2]:", DeltaE[-1])

print(
    "    Fractional error in energy conservation:",
    ((DF.TotalEnergy() - E_list[0] + E_ej_tot[-1]) + DeltaE[-1]) / (DeltaE[-1]),
)

print("  ")
print(
    "    Delta rho/rho(r0):",
    DF.rho(r0, v_cut=np.sqrt(G_N * (DF.M_BH + DF.M_NS) / r0)) / rho0[r0_ind],
)

plt.show()

